use super::Var;
use num_traits::{pow, One, Zero};
use std::cell::{Ref, RefCell};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::rc::Rc;

use super::numeric::{
    add, add_assign, add_assign_v, assign, assign_v, div, div_assign_pow, mat_mat_mul, mat_vec_mul,
    mul, pow_diff_add_assign, pow_diff_assign, relu_diff_add_assign, relu_diff_assign,
    relu_forward, scaled_add_assign, scaled_assign, sub, sub_assign, BackwardAction, DataRepr,
    ForwardAction, PassCounter,
};

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

pub trait InternalRepr: Debug + 'static {
    type Data;
    type Grad;
    fn forward(&self);
    fn backward(&self, grad: &Ref<Self::Grad>);
    fn data(&self) -> Borrow<Self::Data>;
    fn requires_grad(&self) -> bool;
    fn clear(&self);
}

impl<A: 'static + Copy> InternalRepr
    for Rc<dyn InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>>
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;
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
pub struct Parameter<A: 'static + Copy> {
    pub(crate) data: RefCell<DataRepr<A>>,
    pub(crate) grad: RefCell<DataRepr<A>>,
}

impl<A> Parameter<A>
where
    A: Copy
        + Debug
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Neg<Output = A>
        + Zero
        + One
        + Send
        + Sync,
{
    pub fn new(data: DataRepr<A>) -> Var<Self, A> {
        let zeroed_data = data.zeros();
        let node = Rc::new(Parameter {
            data: RefCell::new(data),
            grad: RefCell::new(zeroed_data),
        });
        let upstream = vec![Var::new(Rc::clone(&node), Vec::new())];

        Var::new(node, upstream)
    }

    pub fn grad(&self) -> Borrow<DataRepr<A>> {
        Borrow::FromRefCell(self.grad.borrow())
    }

    pub fn zero_grad(&self) {
        self.grad.borrow_mut().set_zero();
    }
}

impl<A> InternalRepr for Parameter<A>
where
    A: Copy
        + Debug
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Zero
        + One
        + Send
        + Sync,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;
    fn forward(&self) {}
    fn backward(&self, gradient: &Ref<Self::Grad>) {
        add_assign(&mut self.grad.borrow_mut(), gradient);
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
pub struct Input<A: Copy> {
    pub(crate) data: RefCell<DataRepr<A>>,
}

impl<A> Input<A>
where
    A: Copy
        + Debug
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Neg<Output = A>
        + Zero
        + One
        + Send
        + Sync,
{
    pub fn new(data: DataRepr<A>) -> Var<Self, A> {
        Var::new(
            Rc::new(Input {
                data: RefCell::new(data),
            }),
            Vec::new(),
        )
    }
}

impl<A: 'static + Copy + Debug> InternalRepr for Input<A> {
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;
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
pub struct InternalNeg<A, OP>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    grad: RefCell<DataRepr<A>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, OP> InternalNeg<A, OP>
where
    A: Copy + Send + Sync + Zero + One + Neg<Output = A>,
    OP: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = -operand.data().deref();
        let grad = data.zeros();
        let requires_grad = operand.requires_grad();

        InternalNeg {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<A, OP> InternalRepr for InternalNeg<A, OP>
where
    A: 'static
        + Copy
        + Debug
        + Zero
        + One
        + Send
        + Sync
        + Neg<Output = A>
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign,
    OP: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        scaled_assign(
            &mut self.data.borrow_mut(),
            self.operand.data().deref(),
            A::one().neg(),
        );
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => scaled_assign(&mut self.grad.borrow_mut(), grad, A::one().neg()),
            BackwardAction::Increment => {
                scaled_add_assign(&mut self.grad.borrow_mut(), grad, A::one().neg())
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
pub struct InternalAdd<A, LHS, RHS>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    lhs_grad: RefCell<DataRepr<A>>,
    rhs_grad: RefCell<DataRepr<A>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, LHS, RHS> InternalAdd<A, LHS, RHS>
where
    A: Copy
        + Zero
        + One
        + Neg
        + Send
        + Sync
        + Add<Output = A>
        + Div<Output = A>
        + Mul<Output = A>
        + Sub<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() + rhs.data().deref();
        let lhs_grad = lhs.data().deref().zeros();
        let rhs_grad = rhs.data().deref().zeros();

        InternalAdd {
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

impl<A, LHS, RHS> InternalRepr for InternalAdd<A, LHS, RHS>
where
    A: 'static
        + Debug
        + Copy
        + Zero
        + One
        + Neg
        + Send
        + Sync
        + Add<Output = A>
        + AddAssign
        + Sub<Output = A>
        + SubAssign
        + Mul<Output = A>
        + MulAssign
        + Div<Output = A>
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        add(&mut self.data.borrow_mut(), &lhs_data, &rhs_data);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                assign(&mut self.lhs_grad.borrow_mut(), grad.deref());
                assign(&mut self.rhs_grad.borrow_mut(), grad.deref());
            }
            BackwardAction::Increment => {
                add_assign(&mut self.lhs_grad.borrow_mut(), grad.deref());
                add_assign(&mut self.rhs_grad.borrow_mut(), grad.deref());
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
pub struct InternalSub<A, LHS, RHS>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    lhs_grad: RefCell<DataRepr<A>>,
    rhs_grad: RefCell<DataRepr<A>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, LHS, RHS> InternalSub<A, LHS, RHS>
where
    A: Copy
        + Zero
        + One
        + Neg
        + Send
        + Sync
        + Add<Output = A>
        + Div<Output = A>
        + Mul<Output = A>
        + Sub<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() - rhs.data().deref();
        let lhs_grad = lhs.data().deref().zeros();
        let rhs_grad = rhs.data().deref().zeros();

        InternalSub {
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

impl<A, LHS, RHS> InternalRepr for InternalSub<A, LHS, RHS>
where
    A: 'static
        + Debug
        + Copy
        + Zero
        + One
        + Neg<Output = A>
        + Send
        + Sync
        + Add<Output = A>
        + AddAssign
        + Sub<Output = A>
        + SubAssign
        + Mul<Output = A>
        + MulAssign
        + Div<Output = A>
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        sub(&mut self.data.borrow_mut(), &lhs_data, &rhs_data);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                assign(&mut self.lhs_grad.borrow_mut(), grad.deref());
                scaled_assign(
                    &mut self.rhs_grad.borrow_mut(),
                    grad.deref(),
                    A::one().neg(),
                );
            }
            BackwardAction::Increment => {
                add_assign(&mut self.lhs_grad.borrow_mut(), grad.deref());
                sub_assign(&mut self.rhs_grad.borrow_mut(), grad.deref());
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
pub struct InternalMul<A, LHS, RHS>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    lhs_grad: RefCell<DataRepr<A>>,
    rhs_grad: RefCell<DataRepr<A>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, LHS, RHS> InternalMul<A, LHS, RHS>
where
    A: Copy
        + Zero
        + One
        + Neg
        + Send
        + Sync
        + Add<Output = A>
        + Div<Output = A>
        + Mul<Output = A>
        + Sub<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() * rhs.data().deref();
        let lhs_grad = lhs.data().deref().zeros();
        let rhs_grad = rhs.data().deref().zeros();

        InternalMul {
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

impl<A, LHS, RHS> InternalRepr for InternalMul<A, LHS, RHS>
where
    A: 'static
        + Debug
        + Copy
        + Zero
        + One
        + Neg<Output = A>
        + Send
        + Sync
        + Add<Output = A>
        + AddAssign
        + Sub<Output = A>
        + SubAssign
        + Mul<Output = A>
        + MulAssign
        + Div<Output = A>
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        mul(&mut self.data.borrow_mut(), &lhs_data, &rhs_data);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                assign_v(
                    &mut self.lhs_grad.borrow_mut(),
                    grad.deref() * &self.rhs.data(),
                );
                assign_v(
                    &mut self.rhs_grad.borrow_mut(),
                    grad.deref() * &self.lhs.data(),
                );
            }
            BackwardAction::Increment => {
                add_assign_v(
                    &mut self.lhs_grad.borrow_mut(),
                    grad.deref() * &self.rhs.data(),
                );
                add_assign_v(
                    &mut self.rhs_grad.borrow_mut(),
                    grad.deref() * &self.lhs.data(),
                );
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
pub struct InternalDiv<A, LHS, RHS>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    lhs_grad: RefCell<DataRepr<A>>,
    rhs_grad: RefCell<DataRepr<A>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, LHS, RHS> InternalDiv<A, LHS, RHS>
where
    A: Copy
        + Zero
        + One
        + Neg
        + Send
        + Sync
        + Add<Output = A>
        + Div<Output = A>
        + Mul<Output = A>
        + Sub<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() / rhs.data().deref();
        let lhs_grad = lhs.data().deref().zeros();
        let rhs_grad = rhs.data().deref().zeros();

        InternalDiv {
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

impl<A, LHS, RHS> InternalRepr for InternalDiv<A, LHS, RHS>
where
    A: 'static
        + Debug
        + Copy
        + Zero
        + One
        + Neg<Output = A>
        + Send
        + Sync
        + Add<Output = A>
        + AddAssign
        + Sub<Output = A>
        + SubAssign
        + Mul<Output = A>
        + MulAssign
        + Div<Output = A>
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let lhs_data = self.lhs.data();
        let rhs_data = self.rhs.data();

        div(&mut self.data.borrow_mut(), &lhs_data, &rhs_data);
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => {
                assign_v(
                    &mut self.lhs_grad.borrow_mut(),
                    grad.deref() / &self.rhs.data(),
                );
                let mut tmp =
                    &(grad.deref() * &DataRepr::Scalar(A::one().neg())) * &self.lhs.data();
                div_assign_pow(&mut tmp, &self.rhs.data(), 2);
                assign_v(&mut self.rhs_grad.borrow_mut(), tmp);
            }

            BackwardAction::Increment => {
                add_assign_v(
                    &mut self.lhs_grad.borrow_mut(),
                    grad.deref() / &self.rhs.data(),
                );
                let mut tmp =
                    &(grad.deref() * &DataRepr::Scalar(A::one().neg())) * &self.lhs.data();
                div_assign_pow(&mut tmp, &self.rhs.data(), 2);
                add_assign_v(&mut self.lhs_grad.borrow_mut(), tmp);
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
pub struct InternalDot<A, LHS, RHS>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    grad: RefCell<DataRepr<A>>,
    lhs_grad: RefCell<DataRepr<A>>,
    rhs_grad: RefCell<DataRepr<A>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, LHS, RHS> InternalDot<A, LHS, RHS>
where
    A: 'static
        + Debug
        + Copy
        + Zero
        + One
        + Neg
        + Send
        + Sync
        + Add<Output = A>
        + Div<Output = A>
        + Mul<Output = A>
        + Sub<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = match lhs.data().deref() {
            DataRepr::Scalar(_) => {
                panic!("error: dot product for the scalar variant is undefined.")
            }
            DataRepr::Vector(_) => panic!("error: matrix dot product is undefined for vectors."),
            DataRepr::Matrix(lhs_val) => match rhs.data().deref() {
                DataRepr::Scalar(_) => {
                    panic!("error: dot product for the scalar variant is undefined.")
                }
                DataRepr::Vector(rhs_val) => DataRepr::Vector(lhs_val.dot(rhs_val)),
                DataRepr::Matrix(rhs_val) => DataRepr::Matrix(lhs_val.dot(rhs_val)),
            },
        };
        let grad = data.zeros();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

        InternalDot {
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

impl<A, LHS, RHS> InternalRepr for InternalDot<A, LHS, RHS>
where
    A: 'static
        + Debug
        + Copy
        + Zero
        + One
        + Neg<Output = A>
        + Send
        + Sync
        + Add<Output = A>
        + AddAssign
        + Sub<Output = A>
        + SubAssign
        + Mul<Output = A>
        + MulAssign
        + Div<Output = A>
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        match self.rhs.data().deref() {
            DataRepr::Matrix(_) => mat_mat_mul(
                &mut self.data.borrow_mut(),
                A::one(),
                self.lhs.data().deref(),
                self.rhs.data().deref(),
                A::zero(),
                false,
                false,
            ),

            DataRepr::Vector(_) => mat_vec_mul(
                &mut self.data.borrow_mut(),
                A::one(),
                self.lhs.data().deref(),
                self.rhs.data().deref(),
                A::zero(),
            ),
            _ => panic!("error: attempted matrix product on two vectors."),
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
                        A::one(),
                        grad.deref(),
                        &rhs_data,
                        A::zero(),
                        false,
                        true,
                    );
                    mat_mat_mul(
                        &mut self.rhs_grad.borrow_mut(),
                        A::one(),
                        &lhs_data,
                        grad.deref(),
                        A::zero(),
                        true,
                        false,
                    );
                }
                DataRepr::Vector(rhs_val) => {
                    if let DataRepr::Vector(grad_val) = grad.deref() {
                        assign_v(
                            &mut self.lhs_grad.borrow_mut(),
                            DataRepr::Scalar(grad_val.dot(rhs_val)),
                        );
                        mat_vec_mul(
                            &mut self.rhs_grad.borrow_mut(),
                            A::one(),
                            &lhs_data,
                            grad.deref(),
                            A::zero(),
                        );
                    } else {
                        panic!("error: the gradient of rhs should be a vector");
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
pub struct InternalVecDot<A, LHS, RHS>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    lhs_grad: RefCell<DataRepr<A>>,
    rhs_grad: RefCell<DataRepr<A>>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, LHS, RHS> InternalVecDot<A, LHS, RHS>
where
    A: 'static
        + Debug
        + Copy
        + Zero
        + One
        + Neg<Output = A>
        + Send
        + Sync
        + Add<Output = A>
        + AddAssign
        + Sub<Output = A>
        + SubAssign
        + Mul<Output = A>
        + MulAssign
        + Div<Output = A>
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let lhs_grad = lhs.data().deref().zeros();
        let rhs_grad = lhs.data().deref().zeros();
        let data = match (lhs.data().deref(), rhs.data().deref()) {
            (DataRepr::Vector(lhs_val), DataRepr::Vector(rhs_val)) => {
                DataRepr::Scalar(lhs_val.dot(rhs_val))
            }
            _ => panic!("error: vector dot product is defined only between vectors."),
        };
        InternalVecDot {
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

impl<A, LHS, RHS> InternalRepr for InternalVecDot<A, LHS, RHS>
where
    A: 'static
        + Debug
        + Copy
        + Zero
        + One
        + Neg<Output = A>
        + Send
        + Sync
        + Add<Output = A>
        + AddAssign
        + Sub<Output = A>
        + SubAssign
        + Mul<Output = A>
        + MulAssign
        + Div<Output = A>
        + DivAssign,
    LHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
    RHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

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
pub struct InternalPow<A, OP>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    grad: RefCell<DataRepr<A>>,
    operand: Rc<OP>,
    exp: u16,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, OP> InternalPow<A, OP>
where
    A: Copy + Send + Sync + Zero + One + Neg<Output = A>,
    OP: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    pub fn new(operand: Rc<OP>, exp: u16) -> Self {
        let data = operand.data().deref().map(|el| pow(*el, exp as usize));
        let grad = data.zeros();
        let requires_grad = operand.requires_grad();

        InternalPow {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            exp: exp,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<A, OP> InternalRepr for InternalPow<A, OP>
where
    A: 'static
        + Copy
        + Debug
        + Zero
        + One
        + Send
        + Sync
        + Neg<Output = A>
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + TryFrom<u16>,
    OP: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let (mut data, exp) = { (self.data.borrow_mut(), self.exp) };
        assign(&mut data, self.operand.data().deref());
        data.map_inplace(|x| pow(*x, exp as usize));
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
pub struct InternalSum<A, OP>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    grad: RefCell<DataRepr<A>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, OP> InternalSum<A, OP>
where
    A: Copy + Send + Sync + Zero + One,
    OP: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().sum();
        let grad = data.zeros();
        let requires_grad = operand.requires_grad();

        InternalSum {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<A, OP> InternalRepr for InternalSum<A, OP>
where
    A: 'static
        + Copy
        + Debug
        + Zero
        + One
        + Send
        + Sync
        + Neg<Output = A>
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign,
    OP: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

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
pub struct InternalReLU<A, OP>
where
    A: Copy,
{
    data: RefCell<DataRepr<A>>,
    grad: RefCell<DataRepr<A>>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<A, OP> InternalReLU<A, OP>
where
    A: Copy + Send + Sync + Zero + One + PartialOrd,
    OP: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand
            .data()
            .deref()
            .map(|el| if *el > A::zero() { A::zero() } else { *el });
        let grad = data.zeros();
        let requires_grad = operand.requires_grad();

        InternalReLU {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<A, OP> InternalRepr for InternalReLU<A, OP>
where
    A: 'static
        + Copy
        + Debug
        + Zero
        + One
        + Send
        + Sync
        + Neg<Output = A>
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + PartialOrd,
    OP: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
{
    type Data = DataRepr<A>;
    type Grad = DataRepr<A>;

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
