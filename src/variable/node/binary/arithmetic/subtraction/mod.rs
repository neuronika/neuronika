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

pub struct Subtraction<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Rc<RefCell<Tensor<D>>>,
    right_data: Rc<RefCell<Tensor<E>>>,
    data: Rc<RefCell<BroadTensor<D, E>>>,
    computed: Cell<bool>,
}

impl<D, E> Subtraction<D, E>
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

impl<D, E> Forward for Subtraction<D, E>
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
            .for_each(|v, l, r| *v = l - r);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct SubtractionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    right_gradient: Rc<RefCell<Option<Tensor<E>>>>,
    gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    shape: Broadcasted<D, E>,
}

impl<D, E> SubtractionBackward<D, E>
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

impl<D, E> Backward for SubtractionBackward<D, E>
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
            *right_gradient -= &reduced;
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct SubtractionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    shape: Broadcasted<D, E>,
}

impl<D, E> SubtractionBackwardLeft<D, E>
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

impl<D, E> Backward for SubtractionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        {
            let reduced = reduce(operand_gradient.raw_dim(), &gradient);
            *operand_gradient += &reduced
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct SubtractionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<E>>>>,
    gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    shape: Broadcasted<D, E>,
}

impl<D, E> SubtractionBackwardRight<D, E>
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

impl<D, E> Backward for SubtractionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        {
            let reduced = reduce(gradient.raw_dim(), &gradient);
            *operand_gradient -= &reduced;
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
