#[cfg(test)]
use super::{assert_almost_equals, new_tensor};

use super::{
    expect_tensor, expect_tensor_mut, reduce, Backward, BroadTensor, Broadcasted, Forward, Tensor,
};
use ndarray::{DimMax, Dimension, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Rc<RefCell<Tensor<D>>>,
    right_data: Rc<RefCell<Tensor<E>>>,
    data: Rc<RefCell<BroadTensor<D, E>>>,
    computed: Cell<bool>,
}

impl<D, E> Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_data: Rc<RefCell<Tensor<D>>>,
        right_data: Rc<RefCell<Tensor<E>>>,
        data: Rc<RefCell<BroadTensor<D, E>>>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
            computed: Cell::default(),
        }
    }
}

impl<D, E> Forward for Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left_data.borrow())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|v, l, r| *v = l + r);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct AdditionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    right_gradient: Rc<RefCell<Option<Tensor<E>>>>,
    gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    shape: Broadcasted<D, E>,
}

impl<D, E> AdditionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        right_gradient: Rc<RefCell<Option<Tensor<E>>>>,
        gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
        shape: Broadcasted<D, E>,
    ) -> Self {
        Self {
            left_gradient,
            right_gradient,
            gradient,
            shape,
        }
    }
}

impl<D, E> Backward for AdditionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut left_gradient = expect_tensor_mut(&self.left_gradient);
        let mut right_gradient = expect_tensor_mut(&self.right_gradient);
        let gradient = expect_tensor(&self.gradient);

        {
            let reduced = reduce(left_gradient.raw_dim(), &gradient);
            *left_gradient += &reduced;
        }

        {
            let reduced = reduce(right_gradient.raw_dim(), &gradient);
            *right_gradient += &reduced;
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    shape: Broadcasted<D, E>,
}

impl<D, E> AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
        shape: Broadcasted<D, E>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            shape,
        }
    }
}

impl<D, E> Backward for AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        {
            let reduced = reduce(operand_gradient.raw_dim(), &gradient);
            *operand_gradient += &reduced;
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<E>>>>,
    gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    shape: Broadcasted<D, E>,
}

impl<D, E> AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<E>>>>,
        gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
        shape: Broadcasted<D, E>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            shape,
        }
    }
}

impl<D, E> Backward for AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        {
            let reduced = reduce(operand_gradient.raw_dim(), &gradient);
            *operand_gradient += &reduced;
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// #[cfg(test)]
// mod test;
