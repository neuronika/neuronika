use super::numeric::{Broadcast, Broadcasted, Tensor};
use super::{GraphBuilder, Trackable};
use ndarray::{
    arr1, linalg::general_mat_vec_mul, Array2, ArrayView1, Axis, Dimension, Ix1, Ix2, RemoveAxis,
    Zip,
};
use std::cell::{Cell, Ref, RefCell};
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static PARAM_ID: AtomicUsize = AtomicUsize::new(0); // TODO: remove me

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

fn accumulate<D, E>(target: &mut Tensor<D>, source: &Tensor<E>, scale: f32, action: &BackwardAction)
where
    D: Dimension + RemoveAxis,
    E: Dimension + RemoveAxis,
{
    let (trgt_data, source_data) = { (target.array_mut(), source.array()) };

    if trgt_data.len() == 1 {
        match action {
            BackwardAction::Set => Zip::from(trgt_data).apply(|el| *el = source_data.sum() * scale),
            BackwardAction::Increment => {
                Zip::from(trgt_data).apply(|el| *el += source_data.sum() * scale)
            }
        }
    } else {
        match trgt_data.ndim().cmp(&source_data.ndim()) {
            std::cmp::Ordering::Less => {
                let mut dyn_source = source_data.sum_axis(Axis(0)).into_dyn();
                while trgt_data.ndim() < dyn_source.ndim() {
                    dyn_source = dyn_source.sum_axis(Axis(0));
                }
                let static_source = dyn_source.into_dimensionality::<D>().unwrap();
                let mut axis_of_len_one = false;
                for i in 0..trgt_data.ndim() {
                    let size = trgt_data.len_of(Axis(i));
                    if size == 1_usize {
                        axis_of_len_one = true;
                        match action {
                            BackwardAction::Set => {
                                Zip::from(trgt_data.lanes_mut(Axis(i)))
                                    .and(static_source.lanes(Axis(i)))
                                    .apply(|dest_lane, src_lane| {
                                        Zip::from(dest_lane).apply(|dest_view_el| {
                                            *dest_view_el = src_lane.sum() * scale
                                        });
                                    });
                            }
                            BackwardAction::Increment => {
                                Zip::from(trgt_data.lanes_mut(Axis(i)))
                                    .and(static_source.lanes(Axis(i)))
                                    .apply(|dest_lane, src_lane| {
                                        Zip::from(dest_lane).apply(|dest_view_el| {
                                            *dest_view_el += src_lane.sum() * scale
                                        });
                                    });
                            }
                        }
                    }
                }
                if !axis_of_len_one {
                    match action {
                        BackwardAction::Set => {
                            Zip::from(trgt_data)
                                .and(&static_source)
                                .apply(|el_trgt, el_source| *el_trgt = *el_source * scale);
                        }
                        BackwardAction::Increment => {
                            Zip::from(trgt_data)
                                .and(&static_source)
                                .apply(|el_trgt, el_source| *el_trgt += *el_source * scale);
                        }
                    }
                }
            }
            std::cmp::Ordering::Equal => {
                let source_same_dim = source_data.view().into_dimensionality::<D>().unwrap();
                let mut axis_of_len_one = false;
                for i in 0..trgt_data.ndim() {
                    let size = trgt_data.len_of(Axis(i));
                    if size == 1_usize {
                        axis_of_len_one = true;
                        match action {
                            BackwardAction::Set => {
                                Zip::from(trgt_data.lanes_mut(Axis(i)))
                                    .and(source_same_dim.lanes(Axis(i)))
                                    .apply(|dest_lane, src_lane| {
                                        Zip::from(dest_lane).apply(|dest_view_el| {
                                            *dest_view_el = src_lane.sum() * scale
                                        });
                                    });
                            }
                            BackwardAction::Increment => {
                                Zip::from(trgt_data.lanes_mut(Axis(i)))
                                    .and(source_same_dim.lanes(Axis(i)))
                                    .apply(|dest_lane, src_lane| {
                                        Zip::from(dest_lane).apply(|dest_view_el| {
                                            *dest_view_el += src_lane.sum() * scale
                                        });
                                    });
                            }
                        }
                    }
                }
                if !axis_of_len_one {
                    match action {
                        BackwardAction::Set => {
                            Zip::from(trgt_data)
                                .and_broadcast(&source_same_dim)
                                .apply(|el_trgt, el_source| *el_trgt = *el_source * scale);
                        }
                        BackwardAction::Increment => {
                            Zip::from(trgt_data)
                                .and_broadcast(&source_same_dim)
                                .apply(|el_trgt, el_source| *el_trgt += *el_source * scale);
                        }
                    }
                }
            }
            std::cmp::Ordering::Greater => match action {
                BackwardAction::Set => {
                    Zip::from(trgt_data)
                        .and_broadcast(source_data)
                        .apply(|el_trgt, el_source| *el_trgt = *el_source * scale);
                }
                BackwardAction::Increment => {
                    Zip::from(trgt_data)
                        .and_broadcast(source_data)
                        .apply(|el_trgt, el_source| *el_trgt += *el_source * scale);
                }
            },
        }
    }
}

// ===================================== Computational Graph Components Trait =====================================

pub trait Op: Debug + 'static {
    type Data;
    type Grad;
    fn forward(&self);
    fn backward(&self, grad: &Ref<Self::Grad>);
    fn data(&self) -> Ref<Self::Data>;
    fn requires_grad(&self) -> bool;
    fn clear(&self);
}

// =============================== Computational Graph Leaf: Learnable Parameter ===============================

#[derive(Debug)]
pub struct Param<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) id: usize,
    pub(crate) data: RefCell<Tensor<D>>,
    pub(crate) grad: RefCell<Tensor<D>>,
}

impl<D> Param<D>
where
    D: Dimension + RemoveAxis + 'static,
{
    pub fn new(data: Tensor<D>) -> GraphBuilder<Self, D> {
        let zeroed_data = data.zeros_from();
        let node = Rc::new(Param {
            id: PARAM_ID.fetch_add(1, Ordering::SeqCst),
            data: RefCell::new(data),
            grad: RefCell::new(zeroed_data),
        });
        let upstream = vec![GraphBuilder::new(Rc::clone(&node), Vec::new()).as_trackable()];

        GraphBuilder::new(node, upstream)
    }

    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.grad.borrow()
    }

    pub fn zero_grad(&self) {
        self.grad.borrow_mut().set_zero();
    }
}

impl<D> Op for Param<D>
where
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;
    fn forward(&self) {}
    fn backward(&self, gradient: &Ref<Self::Grad>) {
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
    fn clear(&self) {}
}

// ============================= Computational Graph Leaf: Non Learnable Input =============================

#[derive(Debug)]
pub struct Input<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) data: RefCell<Tensor<D>>,
}

impl<D> Input<D>
where
    D: Dimension + RemoveAxis + 'static,
{
    pub fn new(data: Tensor<D>) -> GraphBuilder<Self, D> {
        GraphBuilder::new(
            Rc::new(Input {
                data: RefCell::new(data),
            }),
            Vec::new(),
        )
    }
}

impl<D> Op for Input<D>
where
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;
    fn forward(&self) {}
    fn backward(&self, _: &Ref<Self::Grad>) {}
    fn data(&self) -> Ref<Self::Data> {
        self.data.borrow()
    }
    fn requires_grad(&self) -> bool {
        false
    }
    fn clear(&self) {}
}

// ============================ Computational Graph Internal Component: Negation  ============================

#[derive(Debug)]
pub struct NegOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> NegOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = -operand.data().deref();
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        NegOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for NegOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.array_mut())
            .and(operand_data.array())
            .par_apply(|self_data_el, operand_data_el| *self_data_el = -operand_data_el);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
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

// ============================ Computational Graph Internal Component: Addition  ============================

#[derive(Debug)]
pub struct AddOp<LHS, RHS, D, E>
where
    D: Dimension + RemoveAxis + Broadcast<E>,
    E: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<Broadcasted<D, E>>>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<E>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS, D, E> AddOp<LHS, RHS, D, E>
where
    LHS: Op<Data = Tensor<D>>,
    RHS: Op<Data = Tensor<E>>,
    D: Dimension + RemoveAxis + Broadcast<E>,
    E: Dimension + RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() + rhs.data().deref();
        let lhs_grad = lhs.data().zeros_from();
        let rhs_grad = rhs.data().zeros_from();

        AddOp {
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

impl<LHS, RHS, D, E> Op for AddOp<LHS, RHS, D, E>
where
    LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    RHS: Op<Data = Tensor<E>, Grad = Tensor<E>>,
    D: Dimension + RemoveAxis + Broadcast<E> + 'static,
    E: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<Broadcasted<D, E>>;
    type Grad = Tensor<Broadcasted<D, E>>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        Zip::from(self_data.array_mut())
            .and_broadcast(lhs_data.array())
            .and_broadcast(rhs_data.array())
            .par_apply(|self_data_el, lhs_data_el, rhs_data_el| {
                *self_data_el = *lhs_data_el + *rhs_data_el
            });
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
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

// ============================ Computational Graph Internal Component: Subtraction  ============================

#[derive(Debug)]
pub struct SubOp<LHS, RHS, D, E>
where
    D: Dimension + RemoveAxis + Broadcast<E>,
    E: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<Broadcasted<D, E>>>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<E>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS, D, E> SubOp<LHS, RHS, D, E>
where
    LHS: Op<Data = Tensor<D>>,
    RHS: Op<Data = Tensor<E>>,
    D: Dimension + RemoveAxis + Broadcast<E>,
    E: Dimension + RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() - rhs.data().deref();
        let lhs_grad = lhs.data().zeros_from();
        let rhs_grad = rhs.data().zeros_from();

        SubOp {
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

impl<LHS, RHS, D, E> Op for SubOp<LHS, RHS, D, E>
where
    LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    RHS: Op<Data = Tensor<E>, Grad = Tensor<E>>,
    D: Dimension + RemoveAxis + Broadcast<E> + 'static,
    E: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<Broadcasted<D, E>>;
    type Grad = Tensor<Broadcasted<D, E>>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        Zip::from(self_data.array_mut())
            .and_broadcast(lhs_data.array())
            .and_broadcast(rhs_data.array())
            .par_apply(|self_data_el, lhs_data_el, rhs_data_el| {
                *self_data_el = *lhs_data_el - *rhs_data_el
            });
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
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

// ============================ Computational Graph Internal Component: Multiplication  ============================

#[derive(Debug)]
pub struct MulOp<LHS, RHS, D, E>
where
    D: Dimension + RemoveAxis + Broadcast<E>,
    E: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<Broadcasted<D, E>>>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<E>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS, D, E> MulOp<LHS, RHS, D, E>
where
    LHS: Op<Data = Tensor<D>>,
    RHS: Op<Data = Tensor<E>>,
    D: Dimension + RemoveAxis + Broadcast<E>,
    E: Dimension + RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() * rhs.data().deref();
        let lhs_grad = lhs.data().zeros_from();
        let rhs_grad = rhs.data().zeros_from();

        MulOp {
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

impl<LHS, RHS, D, E> Op for MulOp<LHS, RHS, D, E>
where
    LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    RHS: Op<Data = Tensor<E>, Grad = Tensor<E>>,
    D: Dimension + RemoveAxis + Broadcast<E> + 'static,
    E: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<Broadcasted<D, E>>;
    type Grad = Tensor<Broadcasted<D, E>>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        Zip::from(self_data.array_mut())
            .and_broadcast(lhs_data.array())
            .and_broadcast(rhs_data.array())
            .par_apply(|self_data_el, lhs_data_el, rhs_data_el| {
                *self_data_el = *lhs_data_el * *rhs_data_el
            });
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();
        accumulate(
            &mut self.lhs_grad.borrow_mut(),
            &(grad.deref() * &self.rhs.data()),
            1.0,
            &action,
        );
        accumulate(
            &mut self.rhs_grad.borrow_mut(),
            &(grad.deref() * &self.lhs.data()),
            1.0,
            &action,
        );

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

// ============================ Computational Graph Internal Component: Division  ============================

#[derive(Debug)]
pub struct DivOp<LHS, RHS, D, E>
where
    D: Dimension + RemoveAxis + Broadcast<E>,
    E: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<Broadcasted<D, E>>>,
    lhs_grad: RefCell<Tensor<D>>,
    rhs_grad: RefCell<Tensor<E>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS, D, E> DivOp<LHS, RHS, D, E>
where
    LHS: Op<Data = Tensor<D>>,
    RHS: Op<Data = Tensor<E>>,
    D: Dimension + RemoveAxis + Broadcast<E>,
    E: Dimension + RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() / rhs.data().deref();
        let lhs_grad = lhs.data().zeros_from();
        let rhs_grad = rhs.data().zeros_from();

        DivOp {
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

impl<LHS, RHS, D, E> Op for DivOp<LHS, RHS, D, E>
where
    LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    RHS: Op<Data = Tensor<E>, Grad = Tensor<E>>,
    D: Dimension + RemoveAxis + Broadcast<E> + 'static,
    E: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<Broadcasted<D, E>>;
    type Grad = Tensor<Broadcasted<D, E>>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut self_data = self.data.borrow_mut();
        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        Zip::from(self_data.array_mut())
            .and_broadcast(lhs_data.array())
            .and_broadcast(rhs_data.array())
            .par_apply(|self_data_el, lhs_data_el, rhs_data_el| {
                *self_data_el = *lhs_data_el / *rhs_data_el
            });
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        accumulate(
            &mut self.lhs_grad.borrow_mut(),
            &(grad.deref() / &self.rhs.data()),
            1.0,
            &action,
        );

        let mut tmp = grad.deref() * &self.lhs.data();
        Zip::from(tmp.array_mut())
            .and_broadcast(self.rhs.data().array())
            .apply(|tmp_el, rhs_el| *tmp_el /= rhs_el.powi(2));

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
pub struct DotOp<LHS, RHS> {
    data: RefCell<Tensor<Ix2>>,
    grad: RefCell<Tensor<Ix2>>,
    lhs_grad: RefCell<Tensor<Ix2>>,
    rhs_grad: RefCell<Tensor<Ix2>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> DotOp<LHS, RHS>
where
    LHS: Op<Data = Tensor<Ix2>>,
    RHS: Op<Data = Tensor<Ix2>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        let data = Tensor::new(lhs.data().array().dot(rhs.data().array()));

        let grad = data.zeros_from();
        let lhs_grad = lhs.data().zeros_from();
        let rhs_grad = rhs.data().zeros_from();

        DotOp {
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

impl<LHS, RHS> Op for DotOp<LHS, RHS>
where
    LHS: Op<Data = Tensor<Ix2>, Grad = Tensor<Ix2>>,
    RHS: Op<Data = Tensor<Ix2>, Grad = Tensor<Ix2>>,
{
    type Data = Tensor<Ix2>;
    type Grad = Tensor<Ix2>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        self.lhs.data().mat_mul(
            &self.rhs.data(),
            &mut self.data.borrow_mut(),
            1.0,
            0.0,
            false,
            false,
        );
    }

    fn backward(&self, input_grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        accumulate(
            &mut self.grad.borrow_mut(),
            input_grad.deref(),
            1.0,
            &action,
        );

        if self.counter.recurse_backward() {
            let rhs_data = self.rhs.data();
            let lhs_data = self.lhs.data();
            let grad = self.grad.borrow();

            grad.deref().mat_mul(
                &rhs_data,
                &mut self.lhs_grad.borrow_mut(),
                1.0,
                0.0,
                false,
                true,
            );
            lhs_data.mat_mul(
                grad.deref(),
                &mut self.rhs_grad.borrow_mut(),
                1.0,
                0.0,
                true,
                false,
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
pub struct DotVecOp<LHS, RHS> {
    data: RefCell<Tensor<Ix1>>,
    grad: RefCell<Tensor<Ix1>>,
    lhs_grad: RefCell<Tensor<Ix2>>,
    rhs_grad: RefCell<Tensor<Ix1>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> DotVecOp<LHS, RHS>
where
    LHS: Op<Data = Tensor<Ix2>>,
    RHS: Op<Data = Tensor<Ix1>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        let data = Tensor::new(lhs.data().array().dot(rhs.data().array()));

        let grad = data.zeros_from();
        let lhs_grad = lhs.data().zeros_from();
        let rhs_grad = rhs.data().zeros_from();

        DotVecOp {
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

impl<LHS, RHS> Op for DotVecOp<LHS, RHS>
where
    LHS: Op<Data = Tensor<Ix2>, Grad = Tensor<Ix2>>,
    RHS: Op<Data = Tensor<Ix1>, Grad = Tensor<Ix1>>,
{
    type Data = Tensor<Ix1>;
    type Grad = Tensor<Ix1>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        self.lhs.data().mat_vec_mul(
            &self.rhs.data(),
            &mut self.data.borrow_mut(),
            1.0,
            0.0,
            false,
        );
    }

    fn backward(&self, input_grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        accumulate(
            &mut self.grad.borrow_mut(),
            input_grad.deref(),
            1.0,
            &action,
        );

        if self.counter.recurse_backward() {
            let rhs_data = self.rhs.data();
            let lhs_data = self.lhs.data();
            let grad = self.grad.borrow();

            Zip::from(self.lhs_grad.borrow_mut().array_mut().genrows_mut())
                .and(grad.array())
                .apply(|row, grad_el| {
                    Zip::from(row)
                        .and(rhs_data.array())
                        .apply(|row_el, rhs_data_el| *row_el = *rhs_data_el * *grad_el);
                });

            lhs_data.mat_vec_mul(
                grad.deref(),
                &mut self.rhs_grad.borrow_mut(),
                1.0,
                0.0,
                true,
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
pub struct ScalProdOp<LHS, RHS> {
    data: RefCell<Tensor<Ix1>>,
    lhs_grad: RefCell<Tensor<Ix1>>,
    rhs_grad: RefCell<Tensor<Ix1>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> ScalProdOp<LHS, RHS>
where
    LHS: Op<Data = Tensor<Ix1>>,
    RHS: Op<Data = Tensor<Ix1>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let lhs_grad = lhs.data().zeros_from();
        let rhs_grad = rhs.data().zeros_from();

        let data = Tensor::new(arr1(&[lhs.data().array().dot(rhs.data().array())]));

        ScalProdOp {
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

impl<LHS, RHS> Op for ScalProdOp<LHS, RHS>
where
    LHS: Op<Data = Tensor<Ix1>, Grad = Tensor<Ix1>>,
    RHS: Op<Data = Tensor<Ix1>, Grad = Tensor<Ix1>>,
{
    type Data = Tensor<Ix1>;
    type Grad = Tensor<Ix1>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();
        let mut data = self.data.borrow_mut();
        data.array_mut()[0] = self.lhs.data().array().dot(self.rhs.data().array());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        accumulate(
            &mut self.lhs_grad.borrow_mut(),
            &self.rhs.data().deref(),
            grad.deref().array()[0],
            &action,
        );
        accumulate(
            &mut self.rhs_grad.borrow_mut(),
            &self.lhs.data().deref(),
            grad.deref().array()[0],
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
pub struct PowOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    exp: i32,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> PowOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>, exp: i32) -> Self {
        let data = Tensor::new(operand.data().array().map(|el| el.powi(exp)));
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        PowOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            exp,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for PowOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let (mut self_data, operand_data, exp) =
            { (self.data.borrow_mut(), self.operand.data(), self.exp) };
        Zip::from(self_data.array_mut())
            .and(operand_data.array())
            .par_apply(|self_data_el, operand_data_el| *self_data_el = operand_data_el.powi(exp));
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let exp = self.exp;

        match self.counter.backward_action() {
            BackwardAction::Set => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .apply(|self_grad_el, down_grad_el, operand_data_el| {
                        *self_grad_el = *down_grad_el * operand_data_el.powi(exp - 1) * exp as f32
                    });
            }
            BackwardAction::Increment => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .apply(|self_grad_el, down_grad_el, operand_data_el| {
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
pub struct SumOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<Ix1>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> SumOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().sum();
        let grad = operand.data().zeros_from();
        let requires_grad = operand.requires_grad();

        SumOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for SumOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<Ix1>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        self.data.borrow_mut().array_mut()[0] = self.operand.data().array().sum();
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        accumulate(&mut self.grad.borrow_mut(), grad, 1.0, &action);

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
pub struct LnOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> LnOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = Tensor::new(operand.data().array().map(|el| el.ln()));
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        LnOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for LnOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static + Broadcast<D>,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.array_mut())
            .and(operand_data.array())
            .par_apply(|self_data_el, operand_data_el| *self_data_el = operand_data_el.ln());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();

        match self.counter.backward_action() {
            BackwardAction::Set => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
                        *self_grad_el = *down_grad_el / *operand_data_el
                    });
            }
            BackwardAction::Increment => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
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
pub struct ReLUOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> ReLUOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = Tensor::new(
            operand
                .data()
                .array()
                .map(|el| if *el < 0.0 { 0.0 } else { *el }),
        );
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        ReLUOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for ReLUOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.array_mut())
            .and(operand_data.array())
            .par_apply(|self_data_el, operand_data_el| {
                *self_data_el = if *operand_data_el > 0.0 {
                    *operand_data_el
                } else {
                    0.0
                }
            });
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();

        match self.counter.backward_action() {
            BackwardAction::Set => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
                        *self_grad_el = if *operand_data_el > 0.0 {
                            *down_grad_el
                        } else {
                            0.0
                        }
                    });
            }
            BackwardAction::Increment => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
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
pub struct LeakyReLUOp<OP, D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> LeakyReLUOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = Tensor::new(
            operand
                .data()
                .array()
                .map(|el| if *el < 0.0 { 0.01 * el } else { *el }),
        );
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        LeakyReLUOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for LeakyReLUOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.array_mut())
            .and(operand_data.array())
            .par_apply(|self_data_el, operand_data_el| {
                *self_data_el = if *operand_data_el > 0.0 {
                    *operand_data_el
                } else {
                    0.01 * operand_data_el
                }
            });
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();

        match self.counter.backward_action() {
            BackwardAction::Set => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
                        *self_grad_el = if *operand_data_el > 0.0 {
                            *down_grad_el
                        } else {
                            0.01 * down_grad_el
                        }
                    });
            }
            BackwardAction::Increment => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
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
pub struct SoftplusOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> SoftplusOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = Tensor::new(operand.data().array().map(|el| {
            if *el < -15.0 {
                0.0
            } else if *el > 15.0 {
                *el
            } else {
                (1.0 + el.exp()).ln()
            }
        }));
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        SoftplusOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for SoftplusOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.array_mut())
            .and(operand_data.array())
            .par_apply(|self_data_el, operand_data_el| {
                *self_data_el = if *operand_data_el < -15.0 {
                    0.0
                } else if *operand_data_el > 15.0 {
                    *operand_data_el
                } else {
                    (1.0 + operand_data_el.exp()).ln()
                }
            });
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();

        match self.counter.backward_action() {
            BackwardAction::Set => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
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
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
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
pub struct SigmoidOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> SigmoidOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = Tensor::new(operand.data().array().map(|el| {
            if *el >= 15.0 {
                1.0
            } else if *el <= -15.0 {
                0.0
            } else {
                1.0 / (1.0 + (-el).exp())
            }
        }));

        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        SigmoidOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for SigmoidOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.array_mut())
            .and(operand_data.array())
            .par_apply(|self_data_el, operand_data_el| {
                *self_data_el = if *operand_data_el >= 15.0 {
                    1.0
                } else if *operand_data_el <= -15.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-*operand_data_el).exp())
                }
            });
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();

        match self.counter.backward_action() {
            BackwardAction::Set => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
                        *self_grad_el = *down_grad_el * *operand_data_el * (1.0 - *operand_data_el)
                    });
            }
            BackwardAction::Increment => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
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
pub struct TanhOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> TanhOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = Tensor::new(operand.data().array().map(|el| el.tanh()));
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        TanhOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for TanhOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.array_mut())
            .and(operand_data.array())
            .par_apply(|self_data_el, operand_data_el| *self_data_el = operand_data_el.tanh());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();

        match self.counter.backward_action() {
            BackwardAction::Set => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
                        *self_grad_el = *down_grad_el * (1.0 - operand_data_el.powi(2))
                    });
            }
            BackwardAction::Increment => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
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
pub struct ExpOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> ExpOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = Tensor::new(operand.data().array().map(|el| el.exp()));
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        ExpOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for ExpOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();

        Zip::from(self_data.array_mut())
            .and(operand_data.array())
            .par_apply(|self_data_el, operand_data_el| *self_data_el = operand_data_el.exp());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();

        match self.counter.backward_action() {
            BackwardAction::Set => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
                        *self_grad_el = *down_grad_el * *operand_data_el
                    });
            }
            BackwardAction::Increment => {
                Zip::from(self_grad.array_mut())
                    .and(grad.array())
                    .and(operand_data.array())
                    .par_apply(|self_grad_el, down_grad_el, operand_data_el| {
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
pub struct SoftmaxOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    axis: usize,
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    jacobian: RefCell<Array2<f32>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> SoftmaxOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>, axis: usize) -> Self {
        let (data, j_dim) = {
            let op_data = operand.data();
            (op_data.softmax(axis), op_data.shape()[axis])
        };
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        SoftmaxOp {
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

impl<OP, D> Op for SoftmaxOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut self_data = self.data.borrow_mut();
        let operand_data = self.operand.data();
        let axis = self.axis;

        Zip::from(operand_data.array().lanes(Axis(axis)))
            .and(self_data.array_mut().lanes_mut(Axis(axis)))
            .apply(|lane_self, lane_new| {
                let max = lane_self.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = &lane_self.map(|el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_new)
                    .and(num)
                    .apply(|lane_new_el, num_el| *lane_new_el = *num_el / den);
            });
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let mut self_grad = self.grad.borrow_mut();
        let operand_data = self.operand.data();
        let mut jacobian = self.jacobian.borrow_mut();
        let axis = self.axis;

        fn fill_jacobian(jacobian: &mut Array2<f32>, array: &ArrayView1<f32>) {
            for (row_idx, (mut row, row_val)) in jacobian
                .genrows_mut()
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

        Zip::from(self_grad.array_mut().lanes_mut(Axis(axis)))
            .and(operand_data.array().lanes(Axis(axis)))
            .and(grad.array().lanes(Axis(axis)))
            .apply(|mut d_g_col, data_col, grad_col| {
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
pub struct TOp<OP, D>
where
    D: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<D>>,
    grad: RefCell<Tensor<D>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP, D> TOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().t();
        let grad = operand.data().zeros_from();
        let requires_grad = operand.requires_grad();

        TOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand,
            requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP, D> Op for TOp<OP, D>
where
    OP: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();
        self.data
            .borrow_mut()
            .array_mut()
            .assign(&self.operand.data().array().t());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        accumulate(&mut self.grad.borrow_mut(), &grad.t(), 1.0, &action);

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
    use super::{accumulate, BackwardAction, Tensor};
    use ndarray::array;

    #[test]
    fn assign_test() {
        let mut scalar_trgt = Tensor { array: array![0.0] };
        let mut vector_trgt = Tensor {
            array: array![0.0, 0.0, 0.0],
        };
        let mut matrix_trgt = Tensor {
            array: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        };

        let scalar = Tensor { array: array![1.0] };
        let vector = Tensor {
            array: array![1.0, 1.0, 1.0],
        };
        let matrix = Tensor {
            array: array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        };

        // Scalar scalar assignment.
        accumulate(&mut scalar_trgt, &scalar, 1.0, &BackwardAction::Set);
        assert_eq!(scalar_trgt.array[0], scalar.array[0]);

        // Scalar scalar vector.
        accumulate(&mut scalar_trgt, &vector, 1.0, &BackwardAction::Set);
        assert_eq!(scalar_trgt.array[0], 3.0);

        // Scalar scalar matrix.
        accumulate(&mut scalar_trgt, &matrix, 1.0, &BackwardAction::Set);
        assert_eq!(scalar_trgt.array[0], 9.0);

        // Vector scalar assignment.
        accumulate(&mut vector_trgt, &scalar, 1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt.array, array![1.0, 1.0, 1.0]);

        // Vector vector assignment.
        accumulate(&mut vector_trgt, &vector, 1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt.array, array![1.0, 1.0, 1.0]);

        // Vector matrix assignment.
        accumulate(&mut vector_trgt, &matrix, 1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt.array, array![3.0, 3.0, 3.0]);

        // Matrix scalar assignment.
        accumulate(&mut matrix_trgt, &scalar, 1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt.array,
            array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );

        // Matrix vector assignment.
        accumulate(&mut matrix_trgt, &vector, 1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt.array,
            array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );

        // Matrix matrix assignment.
        accumulate(&mut matrix_trgt, &matrix, 1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt.array,
            array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );
    }

    #[test]
    fn scaled_assign_test() {
        let mut scalar_trgt = Tensor { array: array![0.0] };
        let mut vector_trgt = Tensor {
            array: array![0.0, 0.0, 0.0],
        };
        let mut matrix_trgt = Tensor {
            array: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        };

        let scalar = Tensor { array: array![1.0] };
        let vector = Tensor {
            array: array![1.0, 1.0, 1.0],
        };
        let matrix = Tensor {
            array: array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        };

        // Scalar scalar assignment.
        accumulate(&mut scalar_trgt, &scalar, -1.0, &BackwardAction::Set);
        assert_eq!(scalar_trgt.array[0], -scalar.array[0]);

        // Scalar scalar vector.
        accumulate(&mut scalar_trgt, &vector, -1.0, &BackwardAction::Set);
        assert_eq!(scalar_trgt.array[0], -3.0);

        // Scalar scalar matrix.
        accumulate(&mut scalar_trgt, &matrix, -1.0, &BackwardAction::Set);
        assert_eq!(scalar_trgt.array[0], -9.0);

        // Vector scalar assignment.
        accumulate(&mut vector_trgt, &scalar, -1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt.array, -array![1.0, 1.0, 1.0]);

        // Vector vector assignment.
        accumulate(&mut vector_trgt, &vector, -1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt.array, -array![1.0, 1.0, 1.0]);

        // Vector matrix assignment.
        accumulate(&mut vector_trgt, &matrix, -1.0, &BackwardAction::Set);
        assert_eq!(vector_trgt.array, -array![3.0, 3.0, 3.0]);

        // Matrix scalar assignment.
        accumulate(&mut matrix_trgt, &scalar, -1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt.array,
            -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );

        // Matrix vector assignment.
        accumulate(&mut matrix_trgt, &vector, -1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt.array,
            -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );

        // Matrix matrix assignment.
        accumulate(&mut matrix_trgt, &matrix, -1.0, &BackwardAction::Set);
        assert_eq!(
            matrix_trgt.array,
            -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        );
    }

    #[test]
    fn add_assign_test() {
        let mut scalar_trgt = Tensor { array: array![5.0] };
        let mut vector_trgt = Tensor {
            array: array![5.0, 5.0, 5.0],
        };
        let mut matrix_trgt = Tensor {
            array: array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
        };

        let scalar = Tensor { array: array![5.0] };
        let vector = Tensor {
            array: array![5.0, 5.0, 5.0],
        };
        let matrix = Tensor {
            array: array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
        };

        // Scalar scalar assignment.
        accumulate(&mut scalar_trgt, &scalar, 1.0, &BackwardAction::Increment);
        assert_eq!(scalar_trgt.array[0], 10.0);

        // Scalar scalar vector.
        accumulate(&mut scalar_trgt, &vector, 1.0, &BackwardAction::Increment);
        assert_eq!(scalar_trgt.array[0], 25.0);

        // Scalar scalar matrix.
        accumulate(&mut scalar_trgt, &matrix, 1.0, &BackwardAction::Increment);
        assert_eq!(scalar_trgt.array[0], 70.0);

        // Vector scalar assignment.
        accumulate(&mut vector_trgt, &scalar, 1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt.array, array![10.0, 10.0, 10.0]);

        // Vector vector assignment.
        accumulate(&mut vector_trgt, &vector, 1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt.array, array![15.0, 15.0, 15.0]);

        // Vector matrix assignment.
        accumulate(&mut vector_trgt, &matrix, 1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt.array, array![30.0, 30.0, 30.0]);

        // Matrix scalar assignment.
        accumulate(&mut matrix_trgt, &scalar, 1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt.array,
            array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
        );

        // Matrix vector assignment.
        accumulate(&mut matrix_trgt, &vector, 1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt.array,
            array![[15.0, 15.0, 15.0], [15.0, 15.0, 15.0], [15.0, 15.0, 15.0]]
        );

        // Matrix matrix assignment.
        accumulate(&mut matrix_trgt, &matrix, 1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt.array,
            array![[20.0, 20.0, 20.0], [20.0, 20.0, 20.0], [20.0, 20.0, 20.0]]
        );
    }

    #[test]
    fn scaled_add_assign_test() {
        let mut scalar_trgt = Tensor { array: array![5.0] };
        let mut vector_trgt = Tensor {
            array: array![5.0, 5.0, 5.0],
        };
        let mut matrix_trgt = Tensor {
            array: array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
        };

        let scalar = Tensor { array: array![5.0] };
        let vector = Tensor {
            array: array![5.0, 5.0, 5.0],
        };
        let matrix = Tensor {
            array: array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
        };

        // Scalar scalar assignment.
        &mut accumulate(&mut scalar_trgt, &scalar, -1.0, &BackwardAction::Increment);
        assert_eq!(scalar_trgt.array[0], 0.0);
        scalar_trgt.set_zero();

        // Scalar scalar vector.
        &mut accumulate(&mut scalar_trgt, &vector, -1.0, &BackwardAction::Increment);
        assert_eq!(scalar_trgt.array[0], -15.0);
        scalar_trgt.set_zero();

        // Scalar scalar matrix.
        &mut accumulate(&mut scalar_trgt, &matrix, -1.0, &BackwardAction::Increment);
        assert_eq!(scalar_trgt.array[0], -45.0);
        scalar_trgt.set_zero();

        // Vector scalar assignment.
        &mut accumulate(&mut vector_trgt, &scalar, -1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt.array, array![0.0, 0.0, 0.0]);
        vector_trgt.set_zero();

        // Vector vector assignment.
        &mut accumulate(&mut vector_trgt, &vector, -1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt.array, array![-5.0, -5.0, -5.0]);
        vector_trgt.set_zero();

        // Vector matrix assignment.
        &mut accumulate(&mut vector_trgt, &matrix, -1.0, &BackwardAction::Increment);
        assert_eq!(vector_trgt.array, array![-15.0, -15.0, -15.0]);
        vector_trgt.set_zero();

        // Matrix scalar assignment.
        &mut accumulate(&mut matrix_trgt, &scalar, -1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt.array,
            array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        );
        matrix_trgt.set_zero();

        // Matrix vector assignment.
        &mut accumulate(&mut matrix_trgt, &vector, -1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt.array,
            array![[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]
        );
        matrix_trgt.set_zero();

        // Matrix matrix assignment.
        &mut accumulate(&mut matrix_trgt, &matrix, -1.0, &BackwardAction::Increment);
        assert_eq!(
            matrix_trgt.array,
            array![[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]
        );
        matrix_trgt.set_zero();
    }
}
