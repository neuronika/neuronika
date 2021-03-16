use super::{Trackable, Var};
use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

static PARAM_ID: AtomicUsize = AtomicUsize::new(0);

use super::numeric::{BackwardAction, Broadcast, Broadcasted, ForwardAction, PassCounter, Tensor};
use ndarray::{arr1, Array2, Dimension, Ix1, Ix2, RemoveAxis, Zip};

#[derive(Debug)]
pub enum Borrow<'data, A: 'data> {
    FromRefCell(Ref<'data, A>),
    Plain(&'data A),
}

impl<'data, T: 'data> Deref for Borrow<'data, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match *self {
            Borrow::FromRefCell(ref val) => val.deref(),
            Borrow::Plain(ref val) => val.deref(),
        }
    }
}

pub trait Op: Debug + 'static {
    type Data;
    type Grad;
    fn forward(&self);
    fn backward(&self, grad: &Ref<Self::Grad>);
    fn data(&self) -> Borrow<Self::Data>;
    fn requires_grad(&self) -> bool;
    fn clear(&self);
}

impl<D> Op for Rc<dyn Op<Data = Tensor<D>, Grad = Tensor<D>>>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;
    fn forward(&self) {
        self.deref().forward()
    }
    fn backward(&self, gradient: &Ref<Self::Grad>) {
        self.deref().backward(gradient)
    }
    fn data(&self) -> Borrow<Self::Data> {
        self.deref().data()
    }
    fn requires_grad(&self) -> bool {
        self.deref().requires_grad()
    }
    fn clear(&self) {
        self.deref().clear()
    }
}

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
    pub fn new(data: Tensor<D>) -> Var<Self, D> {
        let zeroed_data = data.zeros_from();
        let node = Rc::new(Param {
            id: PARAM_ID.fetch_add(1, Ordering::SeqCst),
            data: RefCell::new(data),
            grad: RefCell::new(zeroed_data),
        });
        let upstream = vec![Var::new(Rc::clone(&node), Vec::new()).as_trackable()];

        Var::new(node, upstream)
    }

    pub fn grad(&self) -> Borrow<Tensor<D>> {
        Borrow::FromRefCell(self.grad.borrow())
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
        self.grad
            .borrow_mut()
            .accumulate(gradient, 1.0, &BackwardAction::Increment);
    }
    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
    }
    fn requires_grad(&self) -> bool {
        true
    }
    fn clear(&self) {}
}

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
    pub fn new(data: Tensor<D>) -> Var<Self, D> {
        Var::new(
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
    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
    }
    fn requires_grad(&self) -> bool {
        false
    }
    fn clear(&self) {}
}

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
            operand: operand,
            requires_grad: requires_grad,
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

        self.data
            .borrow_mut()
            .data
            .assign(&(-&self.operand.data().deref().data));
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        self.grad
            .borrow_mut()
            .accumulate(grad, -1.0, &self.counter.backward_action());

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
            lhs: lhs,
            rhs: rhs,
            requires_grad: requires_grad,
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

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        self.data.borrow_mut().add_fwd(&lhs_data, &rhs_data);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.lhs_grad
            .borrow_mut()
            .accumulate(grad.deref(), 1.0, &action);
        self.rhs_grad
            .borrow_mut()
            .accumulate(grad.deref(), 1.0, &action);

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

    fn data(&self) -> Borrow<'_, Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

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
            lhs: lhs,
            rhs: rhs,
            requires_grad: requires_grad,
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

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        self.data.borrow_mut().sub_fwd(&lhs_data, &rhs_data);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.lhs_grad
            .borrow_mut()
            .accumulate(grad.deref(), 1.0, &action);
        self.rhs_grad
            .borrow_mut()
            .accumulate(grad.deref(), -1.0, &action);

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

    fn data(&self) -> Borrow<'_, Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

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
            lhs: lhs,
            rhs: rhs,
            requires_grad: requires_grad,
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

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        self.data.borrow_mut().mul_fwd(&lhs_data, &rhs_data);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();
        self.lhs_grad
            .borrow_mut()
            .accumulate(&(grad.deref() * &self.rhs.data()), 1.0, &action);
        self.rhs_grad
            .borrow_mut()
            .accumulate(&(grad.deref() * &self.lhs.data()), 1.0, &action);

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

    fn data(&self) -> Borrow<'_, Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

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
            lhs: lhs,
            rhs: rhs,
            requires_grad: requires_grad,
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

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        self.data.borrow_mut().div_fwd(&lhs_data, &rhs_data);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.lhs_grad
            .borrow_mut()
            .accumulate(&(grad.deref() / &self.rhs.data()), 1.0, &action);

        let mut tmp = grad.deref() * &self.lhs.data();
        Zip::from(&mut tmp.data)
            .and_broadcast(&self.rhs.data().data)
            .apply(|tmp_el, rhs_el| *tmp_el /= rhs_el.powi(2));

        self.rhs_grad.borrow_mut().accumulate(&tmp, -1.0, &action);

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

    fn data(&self) -> Borrow<'_, Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

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

        let data = Tensor {
            data: lhs.data().data.dot(&rhs.data().data),
        };

        let grad = data.zeros_from();
        let lhs_grad = lhs.data().zeros_from();
        let rhs_grad = rhs.data().zeros_from();

        DotOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs: lhs,
            rhs: rhs,
            requires_grad: requires_grad,
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

        self.grad
            .borrow_mut()
            .accumulate(input_grad.deref(), 1.0, &action);

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

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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

        let data = Tensor {
            data: lhs.data().data.dot(&rhs.data().data),
        };

        let grad = data.zeros_from();
        let lhs_grad = lhs.data().zeros_from();
        let rhs_grad = rhs.data().zeros_from();

        DotVecOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs: lhs,
            rhs: rhs,
            requires_grad: requires_grad,
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

        self.grad
            .borrow_mut()
            .accumulate(input_grad.deref(), 1.0, &action);

        if self.counter.recurse_backward() {
            let rhs_data = self.rhs.data();
            let lhs_data = self.lhs.data();
            let grad = self.grad.borrow();

            Zip::from(self.lhs_grad.borrow_mut().data.genrows_mut())
                .and(&grad.data)
                .apply(|mut row, grad_el| {
                    row.assign(&rhs_data.data.map(|el| el * grad_el));
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

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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

        let data = Tensor {
            data: arr1(&[lhs.data().data.dot(&rhs.data().data)]),
        };

        ScalProdOp {
            data: RefCell::new(data),
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            lhs: lhs,
            rhs: rhs,
            requires_grad: requires_grad,
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
        data.data[0] = self.lhs.data().data.dot(&self.rhs.data().data);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.lhs_grad.borrow_mut().accumulate(
            &self.rhs.data().deref(),
            grad.deref().data[0],
            &action,
        );
        self.rhs_grad.borrow_mut().accumulate(
            &self.lhs.data().deref(),
            grad.deref().data[0],
            &action,
        );

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_grad.borrow());
            self.rhs.backward(&self.rhs_grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
        let data = Tensor {
            data: operand.data().deref().data.map(|el| el.powi(exp)),
        };
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        PowOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            exp: exp,
            requires_grad: requires_grad,
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

        let (mut data, exp) = { (self.data.borrow_mut(), self.exp) };
        data.pow_fwd(self.operand.data().deref(), exp);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.grad
            .borrow_mut()
            .pow_bkwrd(grad, &self.operand.data(), &action, self.exp);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
            operand: operand,
            requires_grad: requires_grad,
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

        self.data.borrow_mut().data[0] = self.operand.data().data.sum();
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.grad.borrow_mut().accumulate(grad, 1.0, &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
        let data = Tensor {
            data: operand.data().deref().data.map(|el| el.ln()),
        };
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        LnOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
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

        let mut data = self.data.borrow_mut();
        data.data.assign(&self.operand.data().deref().data);
        data.data.map_inplace(|el| *el = el.ln());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();
        self.grad
            .borrow_mut()
            .accumulate(&(grad.deref() / &self.operand.data()), 1.0, &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
        let data = Tensor {
            data: operand
                .data()
                .deref()
                .data
                .map(|el| if *el < 0.0 { 0.0 } else { *el }),
        };
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        ReLUOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
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
        self.data.borrow_mut().relu_fwd(&self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.grad
            .borrow_mut()
            .relu_bkwrd(grad, &self.data(), &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
        let data = Tensor {
            data: operand
                .data()
                .deref()
                .data
                .map(|el| if *el < 0.0 { 0.01 } else { *el }),
        };
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        LeakyReLUOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
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
        self.data.borrow_mut().leaky_relu_fwd(&self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.grad
            .borrow_mut()
            .leaky_relu_bkwrd(grad, &self.data(), &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
        let data = Tensor {
            data: operand.data().deref().data.map(|el| {
                if *el < -15.0 {
                    0.0
                } else if *el > 15.0 {
                    *el
                } else {
                    (1.0 + el.exp()).ln()
                }
            }),
        };
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        SoftplusOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
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
        self.data.borrow_mut().softplus_fwd(&self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();
        self.grad
            .borrow_mut()
            .softplus_bkwrd(grad, &self.operand.data(), &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
        let data = Tensor {
            data: operand.data().deref().data.map(|el| {
                if *el >= 15.0 {
                    1.0
                } else if *el <= -15.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-el).exp())
                }
            }),
        };

        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        SigmoidOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
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
        self.data.borrow_mut().sigmoid_fwd(&self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();
        self.grad
            .borrow_mut()
            .sigmoid_bkwrd(grad, &self.data(), &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
        let data = Tensor {
            data: operand.data().deref().data.map(|el| el.tanh()),
        };
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        TanhOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
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
        self.data.borrow_mut().tanh_fwd(&self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();
        self.grad
            .borrow_mut()
            .tanh_bkwrd(grad, &self.data(), &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
        let data = Tensor {
            data: operand.data().deref().data.map(|el| el.exp()),
        };
        let grad = data.zeros_from();
        let requires_grad = operand.requires_grad();

        ExpOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
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
        self.data.borrow_mut().exp_fwd(&self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();
        self.grad
            .borrow_mut()
            .exp_bkwrd(grad, &self.data(), &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
            axis: axis,
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            jacobian: RefCell::new(Array2::zeros((j_dim, j_dim))),
            operand: operand,
            requires_grad: requires_grad,
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
        self.data
            .borrow_mut()
            .softmax_fwd(&self.operand.data(), self.axis);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.grad.borrow_mut().softmax_bkwrd(
            grad,
            &self.data.borrow(),
            &mut self.jacobian.borrow_mut(),
            &action,
            self.axis,
        );

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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
            operand: operand,
            requires_grad: requires_grad,
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
            .data
            .assign(&self.operand.data().data.t());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = self.counter.backward_action();

        self.grad.borrow_mut().accumulate(&grad.t(), 1.0, &action);

        if self.counter.recurse_backward() {
            self.operand.backward(&self.grad.borrow());
        }
    }

    fn data(&self) -> Borrow<Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
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

// #[derive(Debug)]
// pub struct VStackOp<LHS, RHS, D, E>
// where
//     D: Dimension + RemoveAxis + VStack<E>,
//     E: Dimension + RemoveAxis,
// {
//     data: RefCell<Tensor<VStacked<D, E>>>,
//     lhs: Rc<LHS>,
//     rhs: Rc<RHS>,
//     lhs_grad: RefCell<Tensor<D>>,
//     rhs_grad: RefCell<Tensor<E>>,
//     requires_grad: bool,
//     counter: PassCounter,
// }

// impl<LHS, RHS, D, E> VStackOp<LHS, RHS, D, E>
// where
//     LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
//     RHS: Op<Data = Tensor<E>, Grad = Tensor<E>>,
//     D: Dimension + RemoveAxis + VStack<E>,
//     E: Dimension + RemoveAxis,
// {
//     pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
//         let data = lhs.data().vstack(rhs.data().deref());

//         let requires_grad = lhs.requires_grad() || rhs.requires_grad();

//         let lhs_grad = lhs.data().zeros_from();
//         let rhs_grad = rhs.data().zeros_from();

//         VStackOp {
//             data: RefCell::new(data),
//             lhs: lhs,
//             rhs: rhs,
//             lhs_grad: RefCell::new(lhs_grad),
//             rhs_grad: RefCell::new(rhs_grad),
//             requires_grad: requires_grad,
//             counter: PassCounter::default(),
//         }
//     }
// }

// impl<LHS, RHS, D, E> Op for VStackOp<LHS, RHS, D, E>
// where
//     LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
//     RHS: Op<Data = Tensor<E>, Grad = Tensor<E>>,
//     D: Dimension + RemoveAxis + VStack<E> + 'static,
//     E: Dimension + RemoveAxis + 'static,
// {
//     type Data = Tensor<VStacked<D, E>>;
//     type Grad = Tensor<VStacked<D, E>>;

//     fn forward(&self) {
//         if self.counter.forward_action() == ForwardAction::Cached {
//             return;
//         }

//         self.lhs.forward();
//         self.rhs.forward();

//         self.data
//             .borrow_mut()
//             .vstack_fwd(self.lhs.data().deref(), self.rhs.data().deref())
//     }

//     fn backward(&self, grad: &Ref<Self::Grad>) {
//         let action = self.counter.backward_action();

//         grad.deref().vstack_bkwrd(
//             &mut self.lhs_grad.borrow_mut(),
//             &mut self.rhs_grad.borrow_mut(),
//             &action,
//         );

//         if self.counter.recurse_backward() {
//             self.lhs.backward(&self.lhs_grad.borrow());
//             self.rhs.backward(&self.rhs_grad.borrow());
//         }
//     }

//     fn clear(&self) {
//         if !self.counter.is_zero() {
//             self.lhs.clear();
//             self.rhs.clear();
//             self.counter.clear();
//         }
//     }

//     fn data(&self) -> Borrow<'_, Self::Data> {
//         Borrow::FromRefCell(self.data.borrow())
//     }

//     fn requires_grad(&self) -> bool {
//         self.requires_grad
//     }
// }

// #[derive(Debug)]
// pub struct HStackOp<LHS, RHS, D, E>
// where
//     D: Dimension + RemoveAxis + HStack<E>,
//     E: Dimension + RemoveAxis,
// {
//     data: RefCell<Tensor<HStacked<D, E>>>,
//     lhs: Rc<LHS>,
//     rhs: Rc<RHS>,
//     lhs_grad: RefCell<Tensor<D>>,
//     rhs_grad: RefCell<Tensor<E>>,
//     requires_grad: bool,
//     counter: PassCounter,
// }

// impl<LHS, RHS, D, E> HStackOp<LHS, RHS, D, E>
// where
//     LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
//     RHS: Op<Data = Tensor<E>, Grad = Tensor<E>>,
//     D: Dimension + RemoveAxis + HStack<E>,
//     E: Dimension + RemoveAxis,
// {
//     pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
//         let data = lhs.data().hstack(rhs.data().deref());

//         let requires_grad = lhs.requires_grad() || rhs.requires_grad();

//         let lhs_grad = lhs.data().zeros_from();
//         let rhs_grad = rhs.data().zeros_from();

//         HStackOp {
//             data: RefCell::new(data),
//             lhs: lhs,
//             rhs: rhs,
//             lhs_grad: RefCell::new(lhs_grad),
//             rhs_grad: RefCell::new(rhs_grad),
//             requires_grad: requires_grad,
//             counter: PassCounter::default(),
//         }
//     }
// }

// impl<LHS, RHS, D, E> Op for HStackOp<LHS, RHS, D, E>
// where
//     LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
//     RHS: Op<Data = Tensor<E>, Grad = Tensor<E>>,
//     D: Dimension + RemoveAxis + HStack<E> + 'static,
//     E: Dimension + RemoveAxis + 'static,
// {
//     type Data = Tensor<HStacked<D, E>>;
//     type Grad = Tensor<HStacked<D, E>>;

//     fn forward(&self) {
//         if self.counter.forward_action() == ForwardAction::Cached {
//             return;
//         }

//         self.lhs.forward();
//         self.rhs.forward();

//         self.data
//             .borrow_mut()
//             .hstack_fwd(self.lhs.data().deref(), self.rhs.data().deref())
//     }

//     fn backward(&self, grad: &Ref<Self::Grad>) {
//         let action = self.counter.backward_action();

//         grad.hstack_bkwrd(
//             &mut self.lhs_grad.borrow_mut(),
//             &mut self.rhs_grad.borrow_mut(),
//             &action,
//         );

//         if self.counter.recurse_backward() {
//             self.lhs.backward(&self.lhs_grad.borrow());
//             self.rhs.backward(&self.rhs_grad.borrow());
//         }
//     }

//     fn clear(&self) {
//         if !self.counter.is_zero() {
//             self.lhs.clear();
//             self.rhs.clear();
//             self.counter.clear();
//         }
//     }

//     fn data(&self) -> Borrow<'_, Self::Data> {
//         Borrow::FromRefCell(self.data.borrow())
//     }

//     fn requires_grad(&self) -> bool {
//         self.requires_grad
//     }
// }

// #[derive(Debug)]
// pub struct DStackOp<LHS, RHS, D>
// where
//     D: Dimension + RemoveAxis + DStack<D>,
// {
//     data: RefCell<Tensor<DStacked<D, D>>>,
//     lhs: Rc<LHS>,
//     rhs: Rc<RHS>,
//     lhs_grad: RefCell<Tensor<D>>,
//     rhs_grad: RefCell<Tensor<D>>,
//     requires_grad: bool,
//     counter: PassCounter,
// }

// impl<LHS, RHS, D> DStackOp<LHS, RHS, D>
// where
//     LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
//     RHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
//     D: Dimension + RemoveAxis + DStack<D>,
// {
//     pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
//         let data = lhs.data().dstack(rhs.data().deref());

//         let requires_grad = lhs.requires_grad() || rhs.requires_grad();

//         let lhs_grad = lhs.data().zeros_from();
//         let rhs_grad = rhs.data().zeros_from();

//         DStackOp {
//             data: RefCell::new(data),
//             lhs: lhs,
//             rhs: rhs,
//             lhs_grad: RefCell::new(lhs_grad),
//             rhs_grad: RefCell::new(rhs_grad),
//             requires_grad: requires_grad,
//             counter: PassCounter::default(),
//         }
//     }
// }

// impl<LHS, RHS, D> Op for DStackOp<LHS, RHS, D>
// where
//     LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
//     RHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
//     D: Dimension + RemoveAxis + DStack<D> + 'static,
// {
//     type Data = Tensor<DStacked<D, D>>;
//     type Grad = Tensor<DStacked<D, D>>;

//     fn forward(&self) {
//         if self.counter.forward_action() == ForwardAction::Cached {
//             return;
//         }

//         self.lhs.forward();
//         self.rhs.forward();

//         self.data
//             .borrow_mut()
//             .dstack_fwd(self.lhs.data().deref(), self.rhs.data().deref())
//     }

//     fn backward(&self, grad: &Ref<Self::Grad>) {
//         let action = self.counter.backward_action();

//         grad.dstack_bkwrd(
//             &mut self.lhs_grad.borrow_mut(),
//             &mut self.rhs_grad.borrow_mut(),
//             &action,
//         );

//         if self.counter.recurse_backward() {
//             self.lhs.backward(&self.lhs_grad.borrow());
//             self.rhs.backward(&self.rhs_grad.borrow());
//         }
//     }

//     fn clear(&self) {
//         if !self.counter.is_zero() {
//             self.lhs.clear();
//             self.rhs.clear();
//             self.counter.clear();
//         }
//     }

//     fn data(&self) -> Borrow<'_, Self::Data> {
//         Borrow::FromRefCell(self.data.borrow())
//     }

//     fn requires_grad(&self) -> bool {
//         self.requires_grad
//     }
//}
