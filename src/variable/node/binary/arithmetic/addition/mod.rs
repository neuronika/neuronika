use super::{
    broadcasted_zeros, expect_tensor, expect_tensor_mut, push_gradient, reduce, Backward,
    BroadTensor, Broadcasted, Data, Forward, Gradient, Overwrite, Tensor,
};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{DimMax, Dimension, Zip};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

pub struct Addition<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    computed: Cell<bool>,
}

impl<Lhs, Rhs> Addition<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(broadcasted_zeros(&left.data(), &right.data()));

        Self {
            left,
            right,
            data,
            computed: Cell::new(false),
        }
    }
}

impl<Lhs, Rhs> Data for Addition<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for Addition<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .for_each(|v, l, r| *v = l + r);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    gradient: RefCell<Option<BroadTensor<Lhs::Dim, Rhs::Dim>>>,
    shape: Broadcasted<Lhs::Dim, Rhs::Dim>,
    overwrite: Cell<bool>,
    left: Rc<Lhs>,
    right: Rc<Rhs>,
}

impl<Lhs, Rhs> AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let gradient = broadcasted_zeros(&left.gradient(), &right.gradient());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            left,
            right,
        }
    }
}

impl<Lhs, Rhs> Gradient for AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<Lhs, Rhs> Overwrite for AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<Lhs, Rhs> Backward for AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn backward(&self) {
        let reduced = reduce(self.left.gradient().raw_dim(), &self.gradient());
        push_gradient(&self.left, &reduced);

        let reduced = reduce(self.right.gradient().raw_dim(), &self.gradient());
        push_gradient(&self.right, &reduced);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    gradient: RefCell<Option<BroadTensor<T::Dim, U::Dim>>>,
    shape: Broadcasted<T::Dim, U::Dim>,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T, U> AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    pub fn new(diff: Rc<T>, no_diff: Rc<U>) -> Self {
        let gradient = broadcasted_zeros(&diff.gradient(), &no_diff.data());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            operand: diff,
            overwrite: Cell::new(true),
        }
    }
}

impl<T, U> Gradient for AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    type Dim = Broadcasted<T::Dim, U::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn backward(&self) {
        let reduced = reduce(self.operand.gradient().raw_dim(), &self.gradient());
        push_gradient(&self.operand, &reduced);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

#[cfg(test)]
mod test;
