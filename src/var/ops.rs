use super::Var;
use num_traits::pow;
use std::cell::{Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;

use super::numeric::{BackwardAction, ForwardAction, Max, Maximum, PassCounter, Tensor};
use ndarray::{Dimension, RemoveAxis};

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
    pub(crate) data: RefCell<Tensor<D>>,
    pub(crate) grad: RefCell<Tensor<D>>,
}

impl<D> Param<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(data: Tensor<D>) -> Var<Self> {
        let zeroed_data = data.zeros();
        let node = Rc::new(Param {
            data: RefCell::new(data),
            grad: RefCell::new(zeroed_data),
        });
        let upstream = vec![Var::new(Rc::clone(&node), Vec::new())];

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
    D: Dimension + RemoveAxis,
{
    type Data = Tensor<D>;
    type Grad = Tensor<D>;
    fn forward(&self) {}
    fn backward(&self, gradient: &Ref<Self::Grad>) {
        self.grad
            .borrow_mut()
            .accumulate(gradient, 1.0, BackwardAction::Increment);
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
    D: Dimension + RemoveAxis,
{
    pub fn new(data: Tensor<D>) -> Var<Self> {
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
    D: Dimension + RemoveAxis,
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
        let grad = data.zeros();
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
    D: Dimension + RemoveAxis,
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
            .accumulate(self.operand.data().deref(), -1.0, BackwardAction::Set);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        self.grad
            .borrow_mut()
            .accumulate(grad, -1.0, self.counter.backward_action());

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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<Maximum<D, E>>>,
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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() + rhs.data().deref();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    type Data = Tensor<Maximum<D, E>>;
    type Grad = Tensor<Maximum<D, E>>;

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
            .accumulate(grad.deref(), 1.0, action);
        self.rhs_grad
            .borrow_mut()
            .accumulate(grad.deref(), 1.0, action);

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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<Maximum<D, E>>>,
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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() - rhs.data().deref();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    type Data = Tensor<Maximum<D, E>>;
    type Grad = Tensor<Maximum<D, E>>;

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
            .accumulate(grad.deref(), 1.0, action);
        self.rhs_grad
            .borrow_mut()
            .accumulate(grad.deref(), -1.0, action);

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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<Maximum<D, E>>>,
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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() * rhs.data().deref();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    type Data = Tensor<Maximum<D, E>>;
    type Grad = Tensor<Maximum<D, E>>;

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
            .accumulate(&(grad.deref() * &self.rhs.data()), 1.0, action);
        self.rhs_grad
            .borrow_mut()
            .accumulate(&(grad.deref() * &self.lhs.data()), 1.0, action);

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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    data: RefCell<Tensor<Maximum<D, E>>>,
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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() / rhs.data().deref();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

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
    D: Dimension + RemoveAxis + Max<E>,
    E: Dimension + RemoveAxis,
{
    type Data = Tensor<Maximum<D, E>>;
    type Grad = Tensor<Maximum<D, E>>;

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
        match self.counter.backward_action() {
            BackwardAction::Set => {
                assign_v(
                    &mut self.lhs_grad.borrow_mut(),
                    grad.deref() / &self.rhs.data(),
                );
                let mut tmp = grad.deref() * &self.lhs.data();
                div_assign_pow(&mut tmp, &self.rhs.data(), 2);
                scaled_assign(&mut self.rhs_grad.borrow_mut(), &tmp, -1.0);
            }

            BackwardAction::Increment => {
                add_assign_v(
                    &mut self.lhs_grad.borrow_mut(),
                    grad.deref() / &self.rhs.data(),
                );
                let mut tmp = grad.deref() * &self.lhs.data();
                div_assign_pow(&mut tmp, &self.rhs.data(), 2);
                scaled_add_assign(&mut self.lhs_grad.borrow_mut(), &tmp, -1.0);
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

    fn data(&self) -> Borrow<'_, Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

#[derive(Debug)]
pub struct DotOp<LHS, RHS> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> DotOp<LHS, RHS>
where
    LHS: Op<Data = DataRepr>,
    RHS: Op<Data = DataRepr>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        let data = match (lhs.data().deref(), rhs.data().deref()) {
            (DataRepr::Matrix(lhs_val), DataRepr::Vector(rhs_val)) => {
                DataRepr::Vector(lhs_val.dot(rhs_val))
            }
            (DataRepr::Matrix(lhs_val), DataRepr::Matrix(rhs_val)) => {
                DataRepr::Matrix(lhs_val.dot(rhs_val))
            }
            _ => panic!("error: matrix dot product is defined only for matrices and vectors."),
        };

        let grad = data.zeros();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

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
    LHS: Op<Data = DataRepr, Grad = DataRepr>,
    RHS: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        match self.rhs.data().deref() {
            DataRepr::Matrix(_) => mat_mat_mul(
                &mut self.data.borrow_mut(),
                1.0,
                self.lhs.data().deref(),
                self.rhs.data().deref(),
                0.0,
                false,
                false,
            ),

            DataRepr::Vector(_) => mat_vec_mul(
                &mut self.data.borrow_mut(),
                1.0,
                self.lhs.data().deref(),
                self.rhs.data().deref(),
                0.0,
                false,
            ),
            _ => panic!("error: attempted matrix product on invalid inputs."),
        }
    }

    fn backward(&self, input_grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                assign(&mut self.grad.borrow_mut(), input_grad.deref());
            }
            BackwardAction::Increment => {
                add_assign(&mut self.grad.borrow_mut(), input_grad.deref());
            }
        }

        if self.counter.recurse_backward() {
            let rhs_data = self.rhs.data();
            let lhs_data = self.lhs.data();
            let grad = self.grad.borrow();

            match rhs_data.deref() {
                DataRepr::Matrix(_) => {
                    mat_mat_mul(
                        &mut self.lhs_grad.borrow_mut(),
                        1.0,
                        grad.deref(),
                        &rhs_data,
                        0.0,
                        false,
                        true,
                    );
                    mat_mat_mul(
                        &mut self.rhs_grad.borrow_mut(),
                        1.0,
                        &lhs_data,
                        grad.deref(),
                        0.0,
                        true,
                        false,
                    );
                }
                DataRepr::Vector(rhs_val) => {
                    if let DataRepr::Vector(grad_val) = grad.deref() {
                        mat_vec_mul_backward_lhs(
                            &mut self.lhs_grad.borrow_mut(),
                            grad_val,
                            rhs_val,
                        );
                        mat_vec_mul(
                            &mut self.rhs_grad.borrow_mut(),
                            1.0,
                            &lhs_data,
                            grad.deref(),
                            0.0,
                            true,
                        );
                    } else {
                        panic!("error: the incoming gradient should be a vector");
                    }
                }
                _ => panic!("error: rhs cannot be a scalar."),
            }

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
    data: RefCell<DataRepr>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> ScalProdOp<LHS, RHS>
where
    LHS: Op<Data = DataRepr>,
    RHS: Op<Data = DataRepr>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

        let data = match (lhs.data().deref(), rhs.data().deref()) {
            (DataRepr::Vector(lhs_val), DataRepr::Vector(rhs_val)) => {
                DataRepr::Scalar(lhs_val.dot(rhs_val))
            }
            _ => panic!("error: vector dot product is defined only between vectors."),
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
    LHS: Op<Data = DataRepr, Grad = DataRepr>,
    RHS: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        match self.data.borrow_mut().deref() {
            DataRepr::Scalar(mut _val) => {
                match (self.lhs.data().deref(), self.rhs.data().deref()) {
                    (DataRepr::Vector(lhs_val), DataRepr::Vector(rhs_val)) => {
                        _val = lhs_val.dot(rhs_val);
                    }
                    _ => panic!("error: vector dot product is defined only between vectors."),
                }
            }
            _ => panic!("error: vector dot product result must be a scalar."),
        }
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                if let DataRepr::Scalar(grad_val) = grad.deref() {
                    scaled_assign(
                        &mut self.lhs_grad.borrow_mut(),
                        &self.rhs.data().deref(),
                        *grad_val,
                    );
                    scaled_assign(
                        &mut self.rhs_grad.borrow_mut(),
                        &self.lhs.data().deref(),
                        *grad_val,
                    );
                } else {
                    panic!("error: gradient of dot product should be a scalar.")
                };
            }
            BackwardAction::Increment => {
                if let DataRepr::Scalar(grad_val) = grad.deref() {
                    scaled_add_assign(
                        &mut self.lhs_grad.borrow_mut(),
                        &self.rhs.data().deref(),
                        *grad_val,
                    );
                    scaled_add_assign(
                        &mut self.rhs_grad.borrow_mut(),
                        &self.lhs.data().deref(),
                        *grad_val,
                    );
                } else {
                    panic!("error: gradient of dot product should be a scalar.")
                };
            }
        }
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
pub struct PowOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    exp: u16,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> PowOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>, exp: u16) -> Self {
        let data = operand.data().deref().map(|el| pow(el, exp as usize));
        let grad = data.zeros();
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

impl<OP> Op for PowOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let (mut data, exp) = { (self.data.borrow_mut(), self.exp) };
        pow_forward(&mut data, self.operand.data().deref(), exp);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => pow_diff_assign(
                &mut self.grad.borrow_mut(),
                grad,
                &self.operand.data(),
                self.exp,
            ),
            BackwardAction::Increment => pow_diff_add_assign(
                &mut self.grad.borrow_mut(),
                grad,
                &self.operand.data(),
                self.exp,
            ),
        }

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
pub struct SumOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> SumOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().sum();
        let grad = operand.data().zeros();
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

impl<OP> Op for SumOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        assign(&mut self.data.borrow_mut(), &self.operand.data().sum());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => assign(&mut self.grad.borrow_mut(), grad),
            BackwardAction::Increment => add_assign(&mut self.grad.borrow_mut(), grad),
        }

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
pub struct LnOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> LnOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().map(|el| el.ln());
        let grad = data.zeros();
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

impl<OP> Op for LnOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut data = self.data.borrow_mut();
        assign(&mut data, self.operand.data().deref());
        data.map_inplace(|el| el.ln());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => assign_v(
                &mut self.grad.borrow_mut(),
                grad.deref() / &self.operand.data(),
            ),
            BackwardAction::Increment => add_assign_v(
                &mut self.grad.borrow_mut(),
                grad.deref() / &self.operand.data(),
            ),
        }

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
pub struct ReLUOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> ReLUOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand
            .data()
            .deref()
            .map(|el| if el < 0.0 { 0.0 } else { el });
        let grad = data.zeros();
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

impl<OP> Op for ReLUOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();
        relu_forward(&mut self.data.borrow_mut(), &self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                relu_diff_assign(&mut self.grad.borrow_mut(), grad, &self.data())
            }
            BackwardAction::Increment => {
                relu_diff_add_assign(&mut self.grad.borrow_mut(), grad, &self.data())
            }
        }

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
pub struct LeakyReLUOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
    slope: f32,
}

impl<OP> LeakyReLUOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>, slope: f32) -> Self {
        let data = operand
            .data()
            .deref()
            .map(|el| if el < 0.0 { el * slope } else { el });
        let grad = data.zeros();
        let requires_grad = operand.requires_grad();

        LeakyReLUOp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
            slope: slope,
        }
    }
}

impl<OP> Op for LeakyReLUOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();
        leakyrelu_forward(
            &mut self.data.borrow_mut(),
            &self.operand.data(),
            self.slope,
        );
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                leakyrelu_diff_assign(&mut self.grad.borrow_mut(), grad, &self.data(), self.slope)
            }
            BackwardAction::Increment => leakyrelu_diff_add_assign(
                &mut self.grad.borrow_mut(),
                grad,
                &self.data(),
                self.slope,
            ),
        }

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
pub struct SoftplusOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> SoftplusOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().map(|el| {
            if el < -15.0 {
                0.0
            } else if el > 15.0 {
                el
            } else {
                (1.0 + el.exp()).ln()
            }
        });
        let grad = data.zeros();
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

impl<OP> Op for SoftplusOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();
        softplus_forward(&mut self.data.borrow_mut(), &self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                softplus_diff_assign(&mut self.grad.borrow_mut(), grad, &self.operand.data())
            }
            BackwardAction::Increment => {
                softplus_diff_add_assign(&mut self.grad.borrow_mut(), grad, &self.operand.data())
            }
        }

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
pub struct SigmoidOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> SigmoidOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().map(|el| {
            if el >= 15.0 {
                1.0
            } else if el <= -15.0 {
                0.0
            } else {
                1.0 / (1.0 + (-el).exp())
            }
        });

        let grad = data.zeros();
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

impl<OP> Op for SigmoidOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();
        sigmoid_forward(&mut self.data.borrow_mut(), &self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                sigmoid_diff_assign(&mut self.grad.borrow_mut(), grad, &self.data())
            }
            BackwardAction::Increment => {
                sigmoid_diff_add_assign(&mut self.grad.borrow_mut(), grad, &self.data())
            }
        }

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
pub struct TanhOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> TanhOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().map(|el| el.tanh());
        let grad = data.zeros();
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

impl<OP> Op for TanhOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();
        tanh_forward(&mut self.data.borrow_mut(), &self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                tanh_diff_assign(&mut self.grad.borrow_mut(), grad, &self.data())
            }
            BackwardAction::Increment => {
                tanh_diff_add_assign(&mut self.grad.borrow_mut(), grad, &self.data.borrow())
            }
        }

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
pub struct ExpOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> ExpOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().map(|el| el.exp());
        let grad = data.zeros();
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

impl<OP> Op for ExpOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();
        exp_forward(&mut self.data.borrow_mut(), &self.operand.data());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => exp_diff_assign(&mut self.grad.borrow_mut(), grad, &self.data()),
            BackwardAction::Increment => {
                exp_diff_add_assign(&mut self.grad.borrow_mut(), grad, &self.data.borrow())
            }
        }

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
pub struct SoftmaxOp<OP> {
    axis: usize,
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    jacobian: RefCell<Matrix>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> SoftmaxOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>, axis: usize) -> Self {
        let (data, j_dim) = {
            let op_data = operand.data();
            let len = match axis {
                0 => op_data.shape()[0],
                1 => op_data.shape()[1],
                _ => panic!("error: maximum number of axis is 2."),
            };
            (op_data.softmax(axis), len)
        };
        let grad = data.zeros();
        let requires_grad = operand.requires_grad();

        SoftmaxOp {
            axis: axis,
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            jacobian: RefCell::new(Matrix::zeros((j_dim, j_dim))),
            operand: operand,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> Op for SoftmaxOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();
        softmax_forward(&mut self.data.borrow_mut(), &self.operand.data(), self.axis);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        let action = match self.counter.backward_action() {
            BackwardAction::Set => 0.0,
            BackwardAction::Increment => 1.0,
        };

        softmax_backward(
            &mut self.grad.borrow_mut(),
            grad,
            &self.data.borrow(),
            &mut self.jacobian.borrow_mut(),
            action,
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
pub struct TOp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> TOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().t();
        let grad = operand.data().zeros();
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

impl<OP> Op for TOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();
        assign(&mut self.data.borrow_mut(), &self.operand.data().t());
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => assign(&mut self.grad.borrow_mut(), &grad.t()),
            BackwardAction::Increment => add_assign(&mut self.grad.borrow_mut(), &grad.t()),
        }

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
pub struct MultiCatOp<OP> {
    axis: usize,
    data: RefCell<DataRepr>,
    operands: Vec<Rc<OP>>,
    op_grads: Vec<RefCell<DataRepr>>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> MultiCatOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operands: &[Rc<OP>], axis: usize) -> Self {
        let reprs: Vec<Borrow<DataRepr>> = operands.iter().map(|op| op.data()).collect();
        let data_refs: Vec<&DataRepr> = reprs.iter().map(|bor| bor.deref()).collect();

        let data = DataRepr::cat(&data_refs[..], axis);

        let requires_grad = operands
            .iter()
            .fold(true, |acc, op| acc || op.requires_grad());

        let op_grads: Vec<RefCell<DataRepr>> = data_refs
            .iter()
            .map(|data| RefCell::new(data.zeros()))
            .collect();

        MultiCatOp {
            axis: axis,
            data: RefCell::new(data),
            operands: operands.to_vec(),
            op_grads: op_grads,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> Op for MultiCatOp<OP>
where
    OP: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }
        for op in &self.operands[..] {
            op.forward();
        }

        let mut data = self.data.borrow_mut();
        let operands_data = self.operands.iter().map(|op| op.data()).collect();

        multicat_forward(&mut data, operands_data, self.axis);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        {
            let ops_data = self.operands.iter().map(|op| op.data()).collect();
            let mut ops_grads: Vec<RefMut<DataRepr>> =
                self.op_grads.iter().map(|grad| grad.borrow_mut()).collect();

            match self.counter.backward_action() {
                BackwardAction::Set => multicat_backward_assign(
                    &self.data.borrow(),
                    ops_data,
                    grad.deref(),
                    ops_grads.as_mut_slice(),
                    self.axis,
                ),
                BackwardAction::Increment => multicat_backward_add_assign(
                    &self.data.borrow(),
                    ops_data,
                    grad.deref(),
                    ops_grads.as_mut_slice(),
                    self.axis,
                ),
            }
        }

        if self.counter.recurse_backward() {
            for (op, op_grad) in self.operands.iter().zip(self.op_grads.iter()) {
                op.backward(&op_grad.borrow());
            }
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
            for op in &self.operands[..] {
                op.clear();
            }
        }
    }
}

#[derive(Debug)]
pub struct BinCatOp<LHS, RHS> {
    axis: usize,
    data: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> BinCatOp<LHS, RHS>
where
    LHS: Op<Data = DataRepr, Grad = DataRepr>,
    RHS: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>, axis: usize) -> Self {
        let data = DataRepr::cat(&[lhs.data().deref(), rhs.data().deref()], axis);

        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

        BinCatOp {
            axis: axis,
            data: RefCell::new(data),
            lhs: lhs,
            rhs: rhs,
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS> Op for BinCatOp<LHS, RHS>
where
    LHS: Op<Data = DataRepr, Grad = DataRepr>,
    RHS: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        cat_forward(
            &mut self.data.borrow_mut(),
            self.lhs.data().deref(),
            self.lhs.data().deref(),
            self.axis,
        );
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => cat_backward_assign(
                &mut self.lhs_grad.borrow_mut(),
                &mut self.rhs_grad.borrow_mut(),
                grad,
                self.axis,
            ),

            BackwardAction::Increment => cat_backward_add_assign(
                &mut self.lhs_grad.borrow_mut(),
                &mut self.rhs_grad.borrow_mut(),
                grad,
                self.axis,
            ),
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

    fn data(&self) -> Borrow<'_, Self::Data> {
        Borrow::FromRefCell(self.data.borrow())
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}
