use super::Var;
use num_traits::pow;
use std::cell::{Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;

use super::numeric::{
    add, add_assign, add_assign_v, assign, assign_v, cat_backward_add_assign, cat_backward_assign,
    cat_forward, div, div_assign_pow, exp_diff_add_assign, exp_diff_assign, exp_forward,
    mat_mat_mul, mat_vec_mul, mat_vec_mul_backward_lhs, mul, multicat_backward_add_assign,
    multicat_backward_assign, multicat_forward, pow_diff_add_assign, pow_diff_assign, pow_forward,
    relu_diff_add_assign, relu_diff_assign, relu_forward, scaled_add_assign, scaled_assign,
    sigmoid_diff_add_assign, sigmoid_diff_assign, sigmoid_forward, softmax_backward,
    softmax_forward, sub, sub_assign, BackwardAction, DataRepr, ForwardAction, Matrix, PassCounter,
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

impl InternalRepr for Rc<dyn InternalRepr<Data = DataRepr, Grad = DataRepr>> {
    type Data = DataRepr;
    type Grad = DataRepr;
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
pub struct Parameter {
    pub(crate) data: RefCell<DataRepr>,
    pub(crate) grad: RefCell<DataRepr>,
}

impl Parameter {
    pub fn new(data: DataRepr) -> Var<Self> {
        let zeroed_data = data.zeros();
        let node = Rc::new(Parameter {
            data: RefCell::new(data),
            grad: RefCell::new(zeroed_data),
        });
        let upstream = vec![Var::new(Rc::clone(&node), Vec::new())];

        Var::new(node, upstream)
    }

    pub fn grad(&self) -> Borrow<DataRepr> {
        Borrow::FromRefCell(self.grad.borrow())
    }

    pub fn zero_grad(&self) {
        self.grad.borrow_mut().set_zero();
    }
}

impl InternalRepr for Parameter {
    type Data = DataRepr;
    type Grad = DataRepr;
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
pub struct Input {
    pub(crate) data: RefCell<DataRepr>,
}

impl Input {
    pub fn new(data: DataRepr) -> Var<Self> {
        Var::new(
            Rc::new(Input {
                data: RefCell::new(data),
            }),
            Vec::new(),
        )
    }
}

impl InternalRepr for Input {
    type Data = DataRepr;
    type Grad = DataRepr;
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
pub struct InternalNeg<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalNeg<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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

impl<OP> InternalRepr for InternalNeg<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

    fn forward(&self) {
        if self.counter.forward_action() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        scaled_assign(
            &mut self.data.borrow_mut(),
            self.operand.data().deref(),
            -1.0,
        );
    }

    fn backward(&self, grad: &Ref<Self::Grad>) {
        match self.counter.backward_action() {
            BackwardAction::Set => scaled_assign(&mut self.grad.borrow_mut(), grad, -1.0),
            BackwardAction::Increment => scaled_add_assign(&mut self.grad.borrow_mut(), grad, -1.0),
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
pub struct InternalAdd<LHS, RHS> {
    data: RefCell<DataRepr>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> InternalAdd<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr>,
    RHS: InternalRepr<Data = DataRepr>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() + rhs.data().deref();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

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

impl<LHS, RHS> InternalRepr for InternalAdd<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    RHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

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
pub struct InternalSub<LHS, RHS> {
    data: RefCell<DataRepr>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> InternalSub<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr>,
    RHS: InternalRepr<Data = DataRepr>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() - rhs.data().deref();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

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

impl<LHS, RHS> InternalRepr for InternalSub<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    RHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

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
                scaled_assign(&mut self.rhs_grad.borrow_mut(), grad.deref(), -1.0);
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
pub struct InternalMul<LHS, RHS> {
    data: RefCell<DataRepr>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> InternalMul<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr>,
    RHS: InternalRepr<Data = DataRepr>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() * rhs.data().deref();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

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

impl<LHS, RHS> InternalRepr for InternalMul<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    RHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

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
pub struct InternalDiv<LHS, RHS> {
    data: RefCell<DataRepr>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> InternalDiv<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr>,
    RHS: InternalRepr<Data = DataRepr>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = lhs.data().deref() / rhs.data().deref();
        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

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

impl<LHS, RHS> InternalRepr for InternalDiv<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    RHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    type Data = DataRepr;
    type Grad = DataRepr;

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
pub struct InternalDot<LHS, RHS> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> InternalDot<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr>,
    RHS: InternalRepr<Data = DataRepr>,
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

impl<LHS, RHS> InternalRepr for InternalDot<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    RHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalVecDot<LHS, RHS> {
    data: RefCell<DataRepr>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> InternalVecDot<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr>,
    RHS: InternalRepr<Data = DataRepr>,
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

impl<LHS, RHS> InternalRepr for InternalVecDot<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    RHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalPow<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    exp: u16,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalPow<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>, exp: u16) -> Self {
        let data = operand.data().deref().map(|el| pow(el, exp as usize));
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

impl<OP> InternalRepr for InternalPow<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalSum<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalSum<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().sum();
        let grad = operand.data().zeros();
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

impl<OP> InternalRepr for InternalSum<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalLn<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalLn<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().map(|el| el.ln());
        let grad = data.zeros();
        let requires_grad = operand.requires_grad();

        InternalLn {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> InternalRepr for InternalLn<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
            BackwardAction::Set => {
                assign_v(&mut self.grad.borrow_mut(), grad.deref() / &self.data())
            }
            BackwardAction::Increment => {
                add_assign_v(&mut self.grad.borrow_mut(), grad.deref() / &self.data())
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
pub struct InternalReLU<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalReLU<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand
            .data()
            .deref()
            .map(|el| if el > 0.0 { 0.0 } else { el });
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

impl<OP> InternalRepr for InternalReLU<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalSigmoid<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalSigmoid<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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

        InternalSigmoid {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> InternalRepr for InternalSigmoid<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalExp<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalExp<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().map(|el| el.exp());
        let grad = data.zeros();
        let requires_grad = operand.requires_grad();

        InternalExp {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> InternalRepr for InternalExp<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalSoftmax<OP> {
    axis: usize,
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    jacobian: RefCell<Matrix>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalSoftmax<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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

        InternalSoftmax {
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

impl<OP> InternalRepr for InternalSoftmax<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalT<OP> {
    data: RefCell<DataRepr>,
    grad: RefCell<DataRepr>,
    operand: Rc<OP>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalT<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let data = operand.data().deref().t();
        let grad = operand.data().zeros();
        let requires_grad = operand.requires_grad();

        InternalT {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            operand: operand,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> InternalRepr for InternalT<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalMultiConcat<OP> {
    axis: usize,
    data: RefCell<DataRepr>,
    operands: Vec<Rc<OP>>,
    op_grads: Vec<RefCell<DataRepr>>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<OP> InternalMultiConcat<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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

        InternalMultiConcat {
            axis: axis,
            data: RefCell::new(data),
            operands: operands.to_vec(),
            op_grads: op_grads,
            requires_grad: requires_grad,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> InternalRepr for InternalMultiConcat<OP>
where
    OP: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
pub struct InternalBinConcat<LHS, RHS> {
    axis: usize,
    data: RefCell<DataRepr>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    lhs_grad: RefCell<DataRepr>,
    rhs_grad: RefCell<DataRepr>,
    requires_grad: bool,
    counter: PassCounter,
}

impl<LHS, RHS> InternalBinConcat<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    RHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>, axis: usize) -> Self {
        let data = DataRepr::cat(&[lhs.data().deref(), rhs.data().deref()], axis);

        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        let lhs_grad = lhs.data().zeros();
        let rhs_grad = rhs.data().zeros();

        InternalBinConcat {
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

impl<LHS, RHS> InternalRepr for InternalBinConcat<LHS, RHS>
where
    LHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    RHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
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
