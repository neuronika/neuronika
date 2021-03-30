use super::{Ancestor, Broadcasted, GraphBuilder, Parameters, Tensor};
use ndarray::{
    concatenate, linalg::general_mat_mul, linalg::general_mat_vec_mul, stack, Array2, ArrayView1,
    Axis, DimMax, Dimension, Ix1, Ix2, RemoveAxis, Zip,
};
use std::cell::{Cell, Ref, RefCell};
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

// ===================================== Computational Graph Aux. Components =====================================

/// Forward action counter. Ensures that the actual computation only happens when the node is fully accumulated.
#[derive(Debug, PartialEq)]
pub enum ForwardAction {
    Evaluate,
    Cached,
}

/// Backward action counter. Keeps track of the gradient accumulation operation.
#[derive(Debug, PartialEq)]
pub enum BackwardAction {
    // Set the gradient.
    Set,
    // Accumulates the gradient.
    Increment,
}

/// Keeps track of the number of times that a node in the computational graph
/// has been evaluated during either the forward or the backward pass.
#[derive(Debug, Default)]
pub struct PassCounter {
    forward_count: Cell<usize>,
    backward_count: Cell<usize>,
}

impl PassCounter {
    pub fn clear(&self) {
        self.forward_count.set(0);
        self.backward_count.set(0);
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.forward_count.get() == 0
    }

    pub fn recurse_backward(&self) -> bool {
        let backward_count = self.backward_count.get();
        let forward_count = self.forward_count.get();

        if backward_count == forward_count {
            self.clear();
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn forward_action(&self) -> ForwardAction {
        let count = self.forward_count.get();
        self.forward_count.set(count + 1);

        match count {
            0 => ForwardAction::Evaluate,
            _ => ForwardAction::Cached,
        }
    }

    #[inline(always)]
    pub fn backward_action(&self) -> BackwardAction {
        let backward_count = self.backward_count.get();

        let action = match backward_count {
            0 => BackwardAction::Set,
            _ => BackwardAction::Increment,
        };

        self.backward_count.set(backward_count + 1);
        action
    }
}

// ===================================== Accumulation + Reduction Function =====================================

fn accumulate<D, E>(lhs: &mut Tensor<D>, rhs: &Tensor<E>, scale: f32, action: &BackwardAction)
where
    D: Dimension,
    E: Dimension,
{
    if lhs.len() == 1 {
        let (rhs_sum, zip) = (rhs.sum(), Zip::from(lhs));
        match action {
            BackwardAction::Set => zip.for_each(|lhs_el| *lhs_el = rhs_sum * scale),
            BackwardAction::Increment => zip.for_each(|lhs_el| *lhs_el += rhs_sum * scale),
        }
        return;
    }

    if lhs.ndim() > rhs.ndim() {
        let zip = Zip::from(lhs).and_broadcast(rhs);
        match action {
            BackwardAction::Set => zip.for_each(|lhs_el, rhs_el| *lhs_el = *rhs_el * scale),
            BackwardAction::Increment => zip.for_each(|lhs_el, rhs_el| *lhs_el += *rhs_el * scale),
        }
        return;
    }

    let mut dyn_rhs = rhs.clone().into_dyn();
    for i in (lhs.ndim()..rhs.ndim()).rev() {
        let axis = Axis(i);
        let (first, rest) = dyn_rhs.view_mut().split_at(axis, 1);
        Zip::from(first.remove_axis(axis))
            .and(rest.lanes(axis))
            .for_each(|dst, src| *dst += src.sum());
        dyn_rhs.index_axis_inplace(axis, 0);
    }

    let (mut done, static_rhs) = {
        (
            false,
            dyn_rhs
                .as_standard_layout()
                .into_dimensionality::<D>()
                .unwrap(),
        )
    };
    for i in 0..static_rhs.ndim() {
        let axis = Axis(i);
        if lhs.len_of(axis) == 1 {
            done = true;
            Zip::from(lhs.lanes_mut(axis))
                .and(static_rhs.lanes(axis))
                .for_each(|dest_lane, src_lane| {
                    let zip = Zip::from(dest_lane);
                    match action {
                        BackwardAction::Set => {
                            zip.for_each(|dest_view_el| *dest_view_el = src_lane.sum() * scale)
                        }
                        BackwardAction::Increment => {
                            zip.for_each(|dest_view_el| *dest_view_el += src_lane.sum() * scale)
                        }
                    }
                });
        }
    }
    if !done {
        let zip = Zip::from(lhs).and_broadcast(&static_rhs);
        match action {
            BackwardAction::Set => zip.for_each(|lhs_el, rhs_el| *lhs_el = rhs_el * scale),
            BackwardAction::Increment => zip.for_each(|lhs_el, rhs_el| *lhs_el += rhs_el * scale),
        }
    }
}

// ===================================== Computational Graph Components Trait =====================================

/// Node of a computational graph.
pub trait Node: Debug + 'static {
    type Data;
    type Gradient;

    /// Computes the forward signal of the node, possibly using the one propagated by its parents.
    fn forward(&self);

    /// Computes the backward signal of the node, using the one provided.
    fn backward(&self, grad: &Ref<Self::Gradient>);

    /// Returns the data of the node.
    fn data(&self) -> Ref<Self::Data>;

    /// Checks wether the node requires the computation of the gradient.
    fn requires_grad(&self) -> bool;

    /// Resets the forward and backward pass counts of the node.
    fn clear(&self);
}

// ====================================== Computational Graph Component: Learnable Parameter ======================================

#[derive(Debug)]
pub struct Parameter<D>
where
    D: Dimension,
{
    pub(crate) data: RefCell<Tensor<D>>,
    pub(crate) grad: RefCell<Tensor<D>>,
}

impl<D> Parameter<D>
where
    D: Dimension + 'static,
{
    pub fn new(data: Tensor<D>) -> GraphBuilder<Self, D> {
        let grad = Tensor::zeros(data.raw_dim());
        let node = Rc::new(Parameter {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
        });

        let ancestor = GraphBuilder::new(Rc::clone(&node), Parameters::new());
        let mut upstream = Parameters::new();
        ancestor.insert(upstream);
        GraphBuilder::new(node, upstream)
    }

    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.grad.borrow()
    }

    pub fn zero_grad(&self) {
        self.grad.borrow_mut().map_inplace(|el| *el = 0.0);
    }
}

impl<D> Node for Parameter<D>
where
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        // Nothing
    }

    fn backward(&self, gradient: &Ref<Self::Gradient>) {
        accumulate(
            &mut self.grad.borrow_mut(),
            gradient,
            1.0,
            &BackwardAction::Increment,
        );
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        true
    }

    fn clear(&self) {
        // Nothing
    }
}

// ============================================= Computational Graph Component: Input =============================================

#[derive(Debug)]
pub struct Input<D>
where
    D: Dimension,
{
    pub(crate) data: RefCell<Tensor<D>>,
}

impl<D> Input<D>
where
    D: Dimension + 'static,
{
    pub fn new(data: Tensor<D>) -> GraphBuilder<Self, D> {
        GraphBuilder::new(
            Rc::new(Input {
                data: RefCell::new(data),
            }),
            Parameters::new(),
        )
    }
}

impl<D> Node for Input<D>
where
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        // Nothing
    }

    fn backward(&self, _: &Ref<Self::Gradient>) {
        // Nothing
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        false
    }

    fn clear(&self) {
        // Nothing
    }
}

// ============================================ Computational Graph Internal Component: Negation  ============================================

#[derive(Debug)]
pub struct Negation<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Negation<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = -operand.data().deref();
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Negation<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.deref_mut())
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| *self_data_el = -operand_data_el);
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        accumulate(
            &mut self.grad.borrow_mut(),
            grad,
            -1.0,
            &self.counter.backward_action(),
        );

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================================== Computational Graph Internal Component: Addition ==============================================

/// The `add
///
///
#[derive(Debug)]
pub struct Addition<Lhs, Rhs, D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    data: RefCell<Tensor<Broadcasted<D, E>>>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<E>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<Lhs, Rhs, D, E> Addition<Lhs, Rhs, D, E>
where
    Lhs: Node<Data = Tensor<D>>,
    Rhs: Node<Data = Tensor<E>>,
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() + rhs.data().deref();
        let lhs_grad = Tensor::zeros(lhs.data().raw_dim());
        let rhs_grad = Tensor::zeros(rhs.data().raw_dim());

        Self {
            data: RefCell::new(data),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs,
            rhs,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<Lhs, Rhs, D, E> Node for Addition<Lhs, Rhs, D, E>
where
    Lhs: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    Rhs: Node<Data = Tensor<E>, Gradient = Tensor<E>>,
    D: Dimension + DimMax<E> + 'static,
    E: Dimension + 'static,
{
    type Data = Tensor<Broadcasted<D, E>>;
    type Gradient = Tensor<Broadcasted<D, E>>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        Zip::from(self_data.deref_mut())
            .and_broadcast(lhs_data.deref())
            .and_broadcast(rhs_data.deref())
            .par_for_each(|self_data_el, lhs_data_el, rhs_data_el| {
                *self_data_el = *lhs_data_el + *rhs_data_el
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let action = self.counter.backward_action();

        accumulate(&mut self.lhs_grad.borrow_mut(), grad.deref(), 1.0, &action);
        accumulate(&mut self.rhs_grad.borrow_mut(), grad.deref(), 1.0, &action);

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// ========================================= Computational Graph Internal Component: Subtraction  =========================================

#[derive(Debug)]
pub struct Subtraction<LHS, RHS, D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    data: RefCell<Tensor<Broadcasted<D, E>>>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<E>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS, D, E> Subtraction<LHS, RHS, D, E>
where
    LHS: Node<Data = Tensor<D>>,
    RHS: Node<Data = Tensor<E>>,
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() - rhs.data().deref();
        let lhs_grad = Tensor::zeros(lhs.data().raw_dim());
        let rhs_grad = Tensor::zeros(rhs.data().raw_dim());

        Self {
            data: RefCell::new(data),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs,
            rhs,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS, D, E> Node for Subtraction<LHS, RHS, D, E>
where
    LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    RHS: Node<Data = Tensor<E>, Gradient = Tensor<E>>,
    D: Dimension + DimMax<E> + 'static,
    E: Dimension + 'static,
{
    type Data = Tensor<Broadcasted<D, E>>;
    type Gradient = Tensor<Broadcasted<D, E>>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        Zip::from(self_data.deref_mut())
            .and_broadcast(lhs_data.deref())
            .and_broadcast(rhs_data.deref())
            .par_for_each(|self_data_el, lhs_data_el, rhs_data_el| {
                *self_data_el = *lhs_data_el - *rhs_data_el
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let action = self.counter.backward_action();

        accumulate(&mut self.lhs_grad.borrow_mut(), grad.deref(), 1.0, &action);
        accumulate(&mut self.rhs_grad.borrow_mut(), grad.deref(), -1.0, &action);

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// ========================================= Computational Graph Internal Component: Multiplication  =========================================

#[derive(Debug)]
pub struct Multiplication<LHS, RHS, D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    data: RefCell<Tensor<Broadcasted<D, E>>>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<E>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS, D, E> Multiplication<LHS, RHS, D, E>
where
    LHS: Node<Data = Tensor<D>>,
    RHS: Node<Data = Tensor<E>>,
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() * rhs.data().deref();
        let lhs_grad = Tensor::zeros(lhs.data().raw_dim());
        let rhs_grad = Tensor::zeros(rhs.data().raw_dim());

        Self {
            data: RefCell::new(data),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs,
            rhs,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS, D, E> Node for Multiplication<LHS, RHS, D, E>
where
    LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    RHS: Node<Data = Tensor<E>, Gradient = Tensor<E>>,
    D: Dimension + DimMax<E> + 'static,
    E: Dimension + 'static,
{
    type Data = Tensor<Broadcasted<D, E>>;
    type Gradient = Tensor<Broadcasted<D, E>>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        Zip::from(self_data.deref_mut())
            .and_broadcast(lhs_data.deref())
            .and_broadcast(rhs_data.deref())
            .par_for_each(|self_data_el, lhs_data_el, rhs_data_el| {
                *self_data_el = *lhs_data_el * *rhs_data_el
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let action = self.counter.backward_action();
        let rhs_data = self.rhs.data();
        let lhs_data = self.lhs.data();
        let down_grad = grad.deref();
        let mut tmp = Tensor::zeros(down_grad.raw_dim());

        Zip::from(&mut tmp)
            .and(down_grad)
            .and_broadcast(rhs_data.deref())
            .par_for_each(|res, grad, rhs| *res = *grad * *rhs);
        accumulate(&mut self.lhs_grad.borrow_mut(), &tmp, 1.0, &action);

        Zip::from(&mut tmp)
            .and(down_grad)
            .and_broadcast(lhs_data.deref())
            .par_for_each(|res, grad, lhs| *res = *grad * *lhs);
        accumulate(&mut self.rhs_grad.borrow_mut(), &tmp, 1.0, &action);

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// =========================================== Computational Graph Internal Component: Division  ===========================================

#[derive(Debug)]
pub struct Division<LHS, RHS, D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    data: RefCell<Tensor<Broadcasted<D, E>>>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<E>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS, D, E> Division<LHS, RHS, D, E>
where
    LHS: Node<Data = Tensor<D>>,
    RHS: Node<Data = Tensor<E>>,
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() / rhs.data().deref();
        let lhs_grad = Tensor::zeros(lhs.data().raw_dim());
        let rhs_grad = Tensor::zeros(rhs.data().raw_dim());

        Self {
            data: RefCell::new(data),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs,
            rhs,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS, D, E> Node for Division<LHS, RHS, D, E>
where
    LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    RHS: Node<Data = Tensor<E>, Gradient = Tensor<E>>,
    D: Dimension + DimMax<E> + 'static,
    E: Dimension + 'static,
{
    type Data = Tensor<Broadcasted<D, E>>;
    type Gradient = Tensor<Broadcasted<D, E>>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        Zip::from(self_data.deref_mut())
            .and_broadcast(lhs_data.deref())
            .and_broadcast(rhs_data.deref())
            .par_for_each(|self_data_el, lhs_data_el, rhs_data_el| {
                *self_data_el = *lhs_data_el / *rhs_data_el
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let action = self.counter.backward_action();
        let rhs_data = self.rhs.data();
        let lhs_data = self.lhs.data();
        let down_grad = grad.deref();
        let mut tmp = Tensor::zeros(down_grad.raw_dim());

        Zip::from(&mut tmp)
            .and(down_grad)
            .and_broadcast(rhs_data.deref())
            .par_for_each(|res, grad, rhs| *res = *grad / *rhs);
        accumulate(&mut self.lhs_grad.borrow_mut(), &tmp, 1.0, &action);

        Zip::from(&mut tmp)
            .and(down_grad)
            .and_broadcast(lhs_data.deref())
            .and_broadcast(rhs_data.deref())
            .par_for_each(|res, grad, lhs, rhs| *res = *grad * *lhs / rhs.powi(2));
        accumulate(&mut self.rhs_grad.borrow_mut(), &tmp, -1.0, &action);

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// ============================ Computational Graph Internal Component: Matrix Mult.  ============================

#[derive(Debug)]
pub struct Dot<LHS, RHS> {
    data: RefCell<Tensor<Ix2>>,
    grad: RefCell<Tensor<Ix2>>,
    lhs_grad: RefCell<Tensor<Ix2>>,
    rhs_grad: RefCell<Tensor<Ix2>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> Dot<LHS, RHS>
where
    LHS: Node<Data = Tensor<Ix2>>,
    RHS: Node<Data = Tensor<Ix2>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        let data = lhs.data().dot(rhs.data().deref());

        let grad = Tensor::zeros(data.raw_dim());
        let lhs_grad = Tensor::zeros(lhs.data().raw_dim());
        let rhs_grad = Tensor::zeros(rhs.data().raw_dim());

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs,
            rhs,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS> Node for Dot<LHS, RHS>
where
    LHS: Node<Data = Tensor<Ix2>, Gradient = Tensor<Ix2>>,
    RHS: Node<Data = Tensor<Ix2>, Gradient = Tensor<Ix2>>,
{
    type Data = Tensor<Ix2>;
    type Gradient = Tensor<Ix2>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();
        let mut res_data = self.data.borrow_mut();

        general_mat_mul(
            1.0,
            lhs_data.deref(),
            rhs_data.deref(),
            0.0,
            res_data.deref_mut(),
        );
    }

    fn backward(&self, input_grad: &Ref<Self::Gradient>) {
        let action = self.counter.backward_action();

        let mut self_grad = self.grad.borrow_mut();
        let down_grad = input_grad.deref();

        accumulate(self_grad.deref_mut(), down_grad, 1.0, &action);

        if self.counter.recurse_backward() {
            let lhs_data = self.lhs.data();
            let mut lhs_grad = self.lhs_grad.borrow_mut();
            let rhs_data = self.rhs.data();
            let mut rhs_grad = self.rhs_grad.borrow_mut();
            let grad = self.grad.borrow();

            general_mat_mul(
                1.0,
                grad.deref(),
                &rhs_data.deref().t(),
                0.0,
                lhs_grad.deref_mut(),
            );
            general_mat_mul(
                1.0,
                &lhs_data.deref().t(),
                grad.deref(),
                0.0,
                rhs_grad.deref_mut(),
            );

            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    fn clear(&self) {
        if !self.counter.is_zero() {
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Mat. Vec. Prod.  ============================

#[derive(Debug)]
pub struct VectorDot<LHS, RHS> {
    data: RefCell<Tensor<Ix1>>,
    grad: RefCell<Tensor<Ix1>>,
    lhs_grad: RefCell<Tensor<Ix2>>,
    rhs_grad: RefCell<Tensor<Ix1>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> VectorDot<LHS, RHS>
where
    LHS: Node<Data = Tensor<Ix2>>,
    RHS: Node<Data = Tensor<Ix1>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        let (data, grad, lhs_grad, rhs_grad) = {
            let lhs_data = lhs.data();
            let rhs_data = rhs.data();
            let data = lhs_data.dot(rhs_data.deref());
            let grad = Tensor::zeros(data.raw_dim());
            (
                data,
                grad,
                Tensor::zeros(lhs_data.raw_dim()),
                Tensor::zeros(rhs_data.raw_dim()),
            )
        };

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs,
            rhs,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS> Node for VectorDot<LHS, RHS>
where
    LHS: Node<Data = Tensor<Ix2>, Gradient = Tensor<Ix2>>,
    RHS: Node<Data = Tensor<Ix1>, Gradient = Tensor<Ix1>>,
{
    type Data = Tensor<Ix1>;
    type Gradient = Tensor<Ix1>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();
        let mut self_data = self.data.borrow_mut();

        general_mat_vec_mul(
            1.0,
            lhs_data.deref(),
            rhs_data.deref(),
            0.0,
            self_data.deref_mut(),
        );
    }

    fn backward(&self, input_grad: &Ref<Self::Gradient>) {
        let action = self.counter.backward_action();

        let mut self_grad = self.grad.borrow_mut();
        let down_grad = input_grad.deref();

        accumulate(self_grad.deref_mut(), down_grad, 1.0, &action);

        if self.counter.recurse_backward() {
            let lhs_data = self.lhs.data();
            let mut lhs_grad = self.lhs_grad.borrow_mut();
            let rhs_data = self.rhs.data();
            let mut rhs_grad = self.rhs_grad.borrow_mut();
            let grad = self.grad.borrow();

            Zip::from(lhs_grad.rows_mut())
                .and(grad.deref())
                .for_each(|row, grad_el| {
                    Zip::from(row)
                        .and(rhs_data.deref())
                        .for_each(|row_el, rhs_data_el| *row_el = *rhs_data_el * *grad_el);
                });

            general_mat_vec_mul(
                1.0,
                &lhs_data.deref().t(),
                grad.deref(),
                0.0,
                rhs_grad.deref_mut(),
            );

            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Inner Prod.  ============================

#[derive(Debug)]
pub struct ScalarProduct<LHS, RHS> {
    data: RefCell<Tensor<Ix1>>,
    lhs_grad: RefCell<Tensor<Ix1>>,
    rhs_grad: RefCell<Tensor<Ix1>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> ScalarProduct<LHS, RHS>
where
    LHS: Node<Data = Tensor<Ix1>>,
    RHS: Node<Data = Tensor<Ix1>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let (data, lhs_grad, rhs_grad) = {
            let lhs_data = lhs.data();
            let rhs_data = rhs.data();
            (
                Tensor::<Ix1>::from(vec![lhs_data.dot(rhs_data.deref())]),
                Tensor::zeros(lhs_data.raw_dim()),
                Tensor::zeros(rhs_data.raw_dim()),
            )
        };

        Self {
            data: RefCell::new(data),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs,
            rhs,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS> Node for ScalarProduct<LHS, RHS>
where
    LHS: Node<Data = Tensor<Ix1>, Gradient = Tensor<Ix1>>,
    RHS: Node<Data = Tensor<Ix1>, Gradient = Tensor<Ix1>>,
{
    type Data = Tensor<Ix1>;
    type Gradient = Tensor<Ix1>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();
        let mut self_data = self.data.borrow_mut();

        self_data[0] = lhs_data.dot(rhs_data.deref());
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let action = self.counter.backward_action();

        let lhs_data = self.lhs.data();
        let mut lhs_grad = self.lhs_grad.borrow_mut();
        let rhs_data = self.rhs.data();
        let mut rhs_grad = self.rhs_grad.borrow_mut();
        let down_grad = grad.deref();

        accumulate(
            lhs_grad.deref_mut(),
            rhs_data.deref(),
            down_grad[0],
            &action,
        );
        accumulate(
            rhs_grad.deref_mut(),
            lhs_data.deref(),
            down_grad[0],
            &action,
        );

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Power  ============================

#[derive(Debug)]
pub struct Power<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    exp: i32,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Power<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>, exp: i32) -> Self {
        let data = operand.data().map(|el| el.powi(exp));
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            exp,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Power<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let (mut self_data, operand_data, exp) =
            { (self.data.borrow_mut(), self.operand.data(), self.exp) };

        Zip::from(self_data.deref_mut())
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| {
                *self_data_el = operand_data_el.powi(exp)
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let down_grad = grad.deref();
        let exp = self.exp;

        let zip = Zip::from(self_grad.deref_mut())
            .and(down_grad)
            .and(operand_data.deref());

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip.for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el = *down_grad_el * operand_data_el.powi(exp - 1) * exp as f32
                });
            }
            BackwardAction::Increment => {
                zip.for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el += *down_grad_el * operand_data_el.powi(exp - 1) * exp as f32
                });
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Sum Reduction  ============================

#[derive(Debug)]
pub struct Sum<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<Ix1>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Sum<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let requires_grad = operand.requires_grad();
        let (data, grad) = {
            let operand_data = operand.data();
            (
                Tensor::<Ix1>::from(vec![operand_data.sum()]),
                Tensor::zeros(operand_data.raw_dim()),
            )
        };

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Sum<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<Ix1>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        self_data[0] = operand_data.sum();
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let action = self.counter.backward_action();

        let mut self_grad = self.grad.borrow_mut();
        let down_grad = grad.deref();

        accumulate(self_grad.deref_mut(), down_grad, 1.0, &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Natural Log.  ============================

#[derive(Debug)]
pub struct Logn<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Logn<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().map(|el| el.ln());
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Logn<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.deref_mut())
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| *self_data_el = operand_data_el.ln());
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let down_grad = grad.deref();

        let zip = Zip::from(self_grad.deref_mut())
            .and(down_grad.deref())
            .and(operand_data.deref());

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el = *down_grad_el / *operand_data_el
                });
            }
            BackwardAction::Increment => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el += *down_grad_el / *operand_data_el
                });
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: ReLU  ============================

#[derive(Debug)]
pub struct Relu<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Relu<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().map(|el| if *el < 0.0 { 0.0 } else { *el });
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Relu<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.deref_mut())
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| {
                *self_data_el = if *operand_data_el > 0.0 {
                    *operand_data_el
                } else {
                    0.0
                }
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let down_grad = grad;

        let zip = Zip::from(self_grad.deref_mut())
            .and(down_grad.deref())
            .and(operand_data.deref());

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el = if *operand_data_el > 0.0 {
                        *down_grad_el
                    } else {
                        0.0
                    }
                });
            }
            BackwardAction::Increment => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el += if *operand_data_el > 0.0 {
                        *down_grad_el
                    } else {
                        0.0
                    }
                });
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: LeakyReLU  ============================

#[derive(Debug)]
pub struct LeakyRelu<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> LeakyRelu<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand
            .data()
            .map(|el| if *el < 0.0 { 0.01 * el } else { *el });
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for LeakyRelu<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.deref_mut())
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| {
                *self_data_el = if *operand_data_el > 0.0 {
                    *operand_data_el
                } else {
                    0.01 * operand_data_el
                }
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let down_grad = grad.deref();

        let zip = Zip::from(self_grad.deref_mut())
            .and(down_grad)
            .and(operand_data.deref());

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el = if *operand_data_el > 0.0 {
                        *down_grad_el
                    } else {
                        0.01 * down_grad_el
                    }
                });
            }
            BackwardAction::Increment => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el += if *operand_data_el > 0.0 {
                        *down_grad_el
                    } else {
                        0.01 * down_grad_el
                    }
                });
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Softplus  ============================

#[derive(Debug)]
pub struct Softplus<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Softplus<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().map(|el| {
            if *el < -15.0 {
                0.0
            } else if *el > 15.0 {
                *el
            } else {
                (1.0 + el.exp()).ln()
            }
        });
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Softplus<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.deref_mut())
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| {
                *self_data_el = if *operand_data_el < -15.0 {
                    0.0
                } else if *operand_data_el > 15.0 {
                    *operand_data_el
                } else {
                    (1.0 + operand_data_el.exp()).ln()
                }
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let down_grad = grad.deref();

        let zip = Zip::from(self_grad.deref_mut())
            .and(down_grad)
            .and(operand_data.deref());

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el = if *operand_data_el >= 15.0 {
                        *down_grad_el
                    } else if *operand_data_el <= -15.0 {
                        0.0
                    } else {
                        down_grad_el / (1.0 + (-*operand_data_el).exp())
                    }
                });
            }
            BackwardAction::Increment => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el += if *operand_data_el >= 15.0 {
                        *down_grad_el
                    } else if *operand_data_el <= -15.0 {
                        0.0
                    } else {
                        down_grad_el / (1.0 + (-*operand_data_el).exp())
                    }
                });
            }
        }
        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Sigmoid  ============================

#[derive(Debug)]
pub struct Sigmoid<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Sigmoid<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().map(|el| {
            if *el >= 15.0 {
                1.0
            } else if *el <= -15.0 {
                0.0
            } else {
                1.0 / (1.0 + (-el).exp())
            }
        });

        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Sigmoid<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.deref_mut())
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| {
                *self_data_el = if *operand_data_el >= 15.0 {
                    1.0
                } else if *operand_data_el <= -15.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-*operand_data_el).exp())
                }
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let down_grad = grad.deref();

        let zip = Zip::from(self_grad.deref_mut())
            .and(down_grad)
            .and(operand_data.deref());

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el = *down_grad_el * *operand_data_el * (1.0 - *operand_data_el)
                });
            }
            BackwardAction::Increment => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el += *down_grad_el * *operand_data_el * (1.0 - *operand_data_el)
                });
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Hyper. Tangent  ============================

#[derive(Debug)]
pub struct Tanh<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Tanh<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().map(|el| el.tanh());
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Tanh<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.deref_mut())
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| *self_data_el = operand_data_el.tanh());
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let down_grad = grad.deref();

        let zip = Zip::from(self_grad.deref_mut())
            .and(down_grad)
            .and(operand_data.deref());

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el = *down_grad_el * (1.0 - operand_data_el.powi(2))
                });
            }
            BackwardAction::Increment => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el += *down_grad_el * (1.0 - operand_data_el.powi(2))
                });
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Exponential  ============================

#[derive(Debug)]
pub struct Exp<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Exp<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().map(|el| el.exp());
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Exp<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.deref_mut())
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| *self_data_el = operand_data_el.exp());
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let down_grad = grad.deref();

        let zip = Zip::from(self_grad.deref_mut())
            .and(down_grad)
            .and(operand_data.deref());

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el = *down_grad_el * *operand_data_el
                });
            }
            BackwardAction::Increment => {
                zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
                    *self_grad_el += *down_grad_el * *operand_data_el
                });
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Softmax  ============================

#[derive(Debug)]
pub struct Softmax<OP, D>
where
    D: Dimension,
{
    axis: usize,
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    jacobian: RefCell<Array2<f32>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Softmax<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>, axis: usize) -> Self {
        let (data, j_dim) = {
            let op_data = operand.data();
            let mut data = Tensor::zeros(op_data.raw_dim());

            Zip::from(op_data.lanes(Axis(axis)))
                .and(data.lanes_mut(Axis(axis)))
                .for_each(|lane_self, lane_new| {
                    let max = lane_self.fold(std::f32::MIN, |x, y| x.max(*y));
                    let num = &lane_self.map(|el| (el - max).exp());
                    let den = num.sum();
                    Zip::from(lane_new)
                        .and(num)
                        .for_each(|lane_new_el, num_el| *lane_new_el = *num_el / den);
                });

            (data, op_data.shape()[axis])
        };
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            axis,
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            jacobian: RefCell::new(Array2::zeros((j_dim, j_dim))),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Softmax<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();
        let axis = self.axis;

        Zip::from(operand_data.lanes(Axis(axis)))
            .and(self_data.lanes_mut(Axis(axis)))
            .for_each(|lane_self, lane_new| {
                let max = lane_self.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = &lane_self.map(|el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_new)
                    .and(num)
                    .for_each(|lane_new_el, num_el| *lane_new_el = *num_el / den);
            });
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let mut jacobian = self.jacobian.borrow_mut();
        let axis = self.axis;

        fn fill_jacobian(jacobian: &mut Array2<f32>, array: &ArrayView1<f32>) {
            for (row_idx, (mut row, row_val)) in jacobian
                .rows_mut()
                .into_iter()
                .zip(array.iter())
                .enumerate()
            {
                for (col_idx, (grad, col_val)) in row
                    .as_slice_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(array.iter())
                    .enumerate()
                {
                    if row_idx == col_idx {
                        *grad = row_val * (1.0 - col_val);
                    } else {
                        *grad = -row_val * col_val;
                    }
                }
            }
        }

        let beta = match self.counter.backward_action() {
            BackwardAction::Set => 0.0,
            BackwardAction::Increment => 1.0,
        };

        Zip::from(self_grad.lanes_mut(Axis(axis)))
            .and(operand_data.lanes(Axis(axis)))
            .and(grad.lanes(Axis(axis)))
            .for_each(|mut d_g_col, data_col, grad_col| {
                fill_jacobian(&mut jacobian, &data_col);
                general_mat_vec_mul(1.0, &jacobian, &grad_col, beta, &mut d_g_col);
            });

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Transposition  ============================

#[derive(Debug)]
pub struct Transpose<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Transpose<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().t().to_owned();
        let grad = Tensor::zeros(data.raw_dim());
        let requires_grad = operand.requires_grad();

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Transpose<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        self_data.assign(&operand_data.t());
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut self_grad = self.grad.borrow_mut();
        let down_grad = grad.deref();

        let zip = Zip::from(self_grad.deref_mut()).and(down_grad);

        match self.counter.backward_action() {
            BackwardAction::Set => zip.par_for_each(|dest, src| *dest = *src),
            BackwardAction::Increment => zip.par_for_each(|dest, src| *dest = *src),
        };

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

// ============================ Computational Graph Internal Component: Concatenate  ============================

#[derive(Debug)]
pub struct Concatenate<LHS, RHS, D>
where
    LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    RHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    axis: usize,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<D>>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS, D> Concatenate<LHS, RHS, D>
where
    LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    RHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>, axis: usize) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let (data, lhs_grad, rhs_grad) = {
            let lhs_data = lhs.data();
            let rhs_data = rhs.data();
            (
                concatenate(Axis(axis), &[lhs_data.view(), rhs_data.view()]).unwrap(),
                Tensor::zeros(lhs_data.raw_dim()),
                Tensor::zeros(rhs_data.raw_dim()),
            )
        };

        Self {
            data: RefCell::new(data),
            axis,
            lhs,
            rhs,
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS, D> Node for Concatenate<LHS, RHS, D>
where
    LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    RHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Gradient = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();
        let axis = self.axis;

        let (mut lhs_, mut rhs_) = self_data
            .view_mut()
            .split_at(Axis(axis), lhs_data.len_of(Axis(axis)));
        Zip::from(lhs_data.deref())
            .and(&mut lhs_)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
        Zip::from(rhs_data.deref())
            .and(&mut rhs_)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut lhs_grad = self.lhs_grad.borrow_mut();
        let mut rhs_grad = self.rhs_grad.borrow_mut();
        let axis = self.axis;

        let (lhs_, rhs_) = grad
            .view()
            .split_at(Axis(axis), lhs_grad.len_of(Axis(axis)));

        let zip_lhs = Zip::from(lhs_grad.deref_mut()).and(&lhs_);
        let zip_rhs = Zip::from(rhs_grad.deref_mut()).and(&rhs_);

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip_lhs.for_each(|single_el, fused_el| *single_el = *fused_el);
                zip_rhs.for_each(|single_el, fused_el| *single_el = *fused_el);
            }
            BackwardAction::Increment => {
                zip_lhs.for_each(|single_el, fused_el| *single_el += *fused_el);
                zip_rhs.for_each(|single_el, fused_el| *single_el += *fused_el);
            }
        }

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// ============================ Computational Graph Internal Component: Stack  ============================

#[derive(Debug)]
pub struct Stack<LHS, RHS, D>
where
    LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    RHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: RemoveAxis,
{
    data: RefCell<Tensor<D::Larger>>,
    axis: usize,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<D>>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS, D> Stack<LHS, RHS, D>
where
    LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    RHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>, axis: usize) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let (data, lhs_grad, rhs_grad) = {
            let lhs_data = lhs.data();
            let rhs_data = rhs.data();
            (
                stack(Axis(axis), &[lhs_data.view(), rhs_data.view()]).unwrap(),
                Tensor::zeros(lhs_data.raw_dim()),
                Tensor::zeros(rhs_data.raw_dim()),
            )
        };

        Self {
            data: RefCell::new(data),
            axis,
            lhs,
            rhs,
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS, D> Node for Stack<LHS, RHS, D>
where
    LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    RHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: RemoveAxis + 'static,
{
    type Data = Tensor<D::Larger>;
    type Gradient = Tensor<D::Larger>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();
        let axis = self.axis;

        let mut subview_iter = self_data.axis_iter_mut(Axis(axis));

        let subview_left = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();
        let subview_right = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        Zip::from(lhs_data.deref())
            .and(subview_left)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
        Zip::from(rhs_data.deref())
            .and(subview_right)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        let mut lhs_grad = self.lhs_grad.borrow_mut();
        let mut rhs_grad = self.rhs_grad.borrow_mut();
        let axis = self.axis;

        let mut subview_iter = grad.axis_iter(Axis(axis));

        let subview_left = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();
        let subview_right = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        let zip_lhs = Zip::from(lhs_grad.deref_mut()).and(subview_left);
        let zip_rhs = Zip::from(rhs_grad.deref_mut()).and(subview_right);

        match self.counter.backward_action() {
            BackwardAction::Set => {
                zip_lhs.for_each(|single_el, fused_el| *single_el = *fused_el);
                zip_rhs.for_each(|single_el, fused_el| *single_el = *fused_el);
            }
            BackwardAction::Increment => {
                zip_lhs.for_each(|single_el, fused_el| *single_el += *fused_el);
                zip_rhs.for_each(|single_el, fused_el| *single_el += *fused_el);
            }
        }

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// ============================ Computational Graph Internal Component: Unsqueeze  ============================

#[derive(Debug)]
pub struct Unsqueeze<OP, D>
where
    D: RemoveAxis,
{
    data: RefCell<Tensor<D::Larger>>,
    axis: usize,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> Unsqueeze<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: RemoveAxis,
{
    pub fn new(operand: Rc<OP>, axis: usize) -> Self {
        let requires_grad = operand.requires_grad();
        let (data, grad) = {
            let operand_data = operand.data();
            (
                operand_data.clone().insert_axis(Axis(axis)),
                Tensor::zeros(operand_data.raw_dim()),
            )
        };

        Self {
            data: RefCell::new(data),
            axis,
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Node for Unsqueeze<OP, D>
where
    OP: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: RemoveAxis + 'static,
{
    type Data = Tensor<D::Larger>;
    type Gradient = Tensor<D::Larger>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let mut mut_array = self_data
            .axis_iter_mut(Axis(self.axis))
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();
        let operand_data = self.operand.data();

        Zip::from(&mut mut_array)
            .and(operand_data.deref())
            .par_for_each(|self_data_el, operand_data_el| *self_data_el = *operand_data_el);
    }

    fn backward(&self, grad: &Ref<Self::Gradient>) {
        {
            let mut self_grad = self.grad.borrow_mut();
            let axis = self.axis;
            let down_grad = grad
                .axis_iter(Axis(axis))
                .next()
                .unwrap()
                .into_dimensionality::<D>()
                .unwrap();

            let zip = Zip::from(self_grad.deref_mut()).and(&down_grad);

            match self.counter.backward_action() {
                BackwardAction::Set => {
                    zip.par_for_each(|self_grad_el, down_grad_el| *self_grad_el = *down_grad_el)
                }
                BackwardAction::Increment => {
                    zip.par_for_each(|self_grad_el, down_grad_el| *self_grad_el += *down_grad_el)
                }
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{accumulate, BackwardAction};
    use ndarray::array;

    #[test]
    fn assign_test() {
        let mut scalar_trgt = array![0.0];
        let mut vector_trgt = array![0.0, 0.0, 0.0];
        let mut matrix_trgt = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        let scalar = array![1.0];
        let vector = array![1.0, 1.0, 1.0];
        let matrix = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];

        // Scalar scalar assignment.
        accumulate(&mut scalar_trgt, &scalar, 1.0, &BackwardAction::Set);
        assert!(scalar_trgt[0] - scalar[0] <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 0.0);

        // Scalar scalar vector.
        accumulate(&mut scalar_trgt, &vector, 1.0, &BackwardAction::Set);
        assert!(scalar_trgt[0] - 3.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 0.0);

        // Scalar scalar matrix.
        accumulate(&mut scalar_trgt, &matrix, 1.0, &BackwardAction::Set);
        assert!(scalar_trgt[0] - 9.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 0.0);

        // Vector scalar assignment.
        accumulate(&mut vector_trgt, &scalar, 1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt, array![1.0, 1.0, 1.0]);
        vector_trgt.map_inplace(|el| *el = 0.0);

        // Vector vector assignment.
        accumulate(&mut vector_trgt, &vector, 1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt, array![1.0, 1.0, 1.0]);
        vector_trgt.map_inplace(|el| *el = 0.0);

        // Vector matrix assignment.
        accumulate(&mut vector_trgt, &matrix, 1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt, array![3.0, 3.0, 3.0]);
        vector_trgt.map_inplace(|el| *el = 0.0);

        // Matrix scalar assignment.
        accumulate(&mut matrix_trgt, &scalar, 1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt,
            array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 0.0);

        // Matrix vector assignment.
        accumulate(&mut matrix_trgt, &vector, 1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt,
            array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 0.0);

        // Matrix matrix assignment.
        accumulate(&mut matrix_trgt, &matrix, 1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt,
            array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 0.0);
    }

    #[test]
    fn scaled_assign_test() {
        let mut scalar_trgt = array![0.0];
        let mut vector_trgt = array![0.0, 0.0, 0.0];
        let mut matrix_trgt = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

        let scalar = array![1.0];
        let vector = array![1.0, 1.0, 1.0];
        let matrix = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];

        // Scalar scalar assignment.
        accumulate(&mut scalar_trgt, &scalar, -1.0, &BackwardAction::Set);
        assert!(scalar_trgt[0] - scalar[0] <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 0.0);

        // Scalar scalar vector.
        accumulate(&mut scalar_trgt, &vector, -1.0, &BackwardAction::Set);
        assert!(scalar_trgt[0] - 3.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 0.0);

        // Scalar scalar matrix.
        accumulate(&mut scalar_trgt, &matrix, -1.0, &BackwardAction::Set);
        assert!(scalar_trgt[0] - 9.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 0.0);

        // Vector scalar assignment.
        accumulate(&mut vector_trgt, &scalar, -1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt, -array![1.0, 1.0, 1.0]);
        vector_trgt.map_inplace(|el| *el = 0.0);

        // Vector vector assignment.
        accumulate(&mut vector_trgt, &vector, -1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt, -array![1.0, 1.0, 1.0]);
        vector_trgt.map_inplace(|el| *el = 0.0);

        // Vector matrix assignment.
        accumulate(&mut vector_trgt, &matrix, -1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt, -array![3.0, 3.0, 3.0]);
        vector_trgt.map_inplace(|el| *el = 0.0);

        // Matrix scalar assignment.
        accumulate(&mut matrix_trgt, &scalar, -1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt,
            -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 0.0);

        // Matrix vector assignment.
        accumulate(&mut matrix_trgt, &vector, -1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt,
            -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 0.0);

        // Matrix matrix assignment.
        accumulate(&mut matrix_trgt, &matrix, -1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt,
            -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 0.0);
    }

    #[test]
    fn add_assign_test() {
        let mut scalar_trgt = array![5.0];
        let mut vector_trgt = array![5.0, 5.0, 5.0];
        let mut matrix_trgt = array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];

        let scalar = array![5.0];
        let vector = array![5.0, 5.0, 5.0];
        let matrix = array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];

        // Scalar scalar assignment.
        accumulate(&mut scalar_trgt, &scalar, 1.0, &BackwardAction::Increment);
        assert!(scalar_trgt[0] - 10.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 5.0);

        // Scalar scalar vector.
        accumulate(&mut scalar_trgt, &vector, 1.0, &BackwardAction::Increment);
        assert!(scalar_trgt[0] - 20.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 5.0);

        // Scalar scalar matrix.
        accumulate(&mut scalar_trgt, &matrix, 1.0, &BackwardAction::Increment);
        assert!(scalar_trgt[0] - 50.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 5.0);

        // Vector scalar assignment.
        accumulate(&mut vector_trgt, &scalar, 1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt, array![10.0, 10.0, 10.0]);
        vector_trgt.map_inplace(|el| *el = 5.0);

        // Vector vector assignment.
        accumulate(&mut vector_trgt, &vector, 1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt, array![10.0, 10.0, 10.0]);
        vector_trgt.map_inplace(|el| *el = 5.0);

        // Vector matrix assignment.
        accumulate(&mut vector_trgt, &matrix, 1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt, array![20.0, 20.0, 20.0]);
        vector_trgt.map_inplace(|el| *el = 5.0);

        // Matrix scalar assignment.
        accumulate(&mut matrix_trgt, &scalar, 1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt,
            array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 5.0);

        // Matrix vector assignment.
        accumulate(&mut matrix_trgt, &vector, 1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt,
            array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 5.0);

        // Matrix matrix assignment.
        accumulate(&mut matrix_trgt, &matrix, 1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt,
            array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 5.0);
    }

    #[test]
    fn scaled_add_assign_test() {
        let mut scalar_trgt = array![5.0];
        let mut vector_trgt = array![5.0, 5.0, 5.0];
        let mut matrix_trgt = array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];

        let scalar = array![5.0];
        let vector = array![5.0, 5.0, 5.0];
        let matrix = array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];

        // Scalar scalar assignment.
        accumulate(&mut scalar_trgt, &scalar, -1.0, &BackwardAction::Increment);
        assert!(scalar_trgt[0] - 0.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 5.0);

        // Scalar scalar vector.
        accumulate(&mut scalar_trgt, &vector, -1.0, &BackwardAction::Increment);
        assert!(scalar_trgt[0] - 10.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 5.0);

        // Scalar scalar matrix.
        accumulate(&mut scalar_trgt, &matrix, -1.0, &BackwardAction::Increment);
        assert!(scalar_trgt[0] - 40.0 <= f32::EPSILON);
        scalar_trgt.map_inplace(|el| *el = 5.0);

        // Vector scalar assignment.
        accumulate(&mut vector_trgt, &scalar, -1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt, array![0.0, 0.0, 0.0]);
        vector_trgt.map_inplace(|el| *el = 5.0);

        // Vector vector assignment.
        accumulate(&mut vector_trgt, &vector, -1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt, array![-0.0, -0.0, -0.0]);
        vector_trgt.map_inplace(|el| *el = 5.0);

        // Vector matrix assignment.
        accumulate(&mut vector_trgt, &matrix, -1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt, array![-10.0, -10.0, -10.0]);
        vector_trgt.map_inplace(|el| *el = 5.0);

        // Matrix scalar assignment.
        accumulate(&mut matrix_trgt, &scalar, -1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt,
            array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 5.0);

        // Matrix vector assignment.
        accumulate(&mut matrix_trgt, &vector, -1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt,
            array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 5.0);

        // Matrix matrix assignment.
        accumulate(&mut matrix_trgt, &matrix, -1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt,
            array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        );
        matrix_trgt.map_inplace(|el| *el = 5.0);
    }
}
