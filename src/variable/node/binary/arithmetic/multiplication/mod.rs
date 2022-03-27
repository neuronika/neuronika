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

pub struct Multiplication<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Rc<RefCell<Tensor<D>>>,
    right_data: Rc<RefCell<Tensor<E>>>,
    data: Rc<RefCell<BroadTensor<D, E>>>,
    computed: Cell<bool>,
}

impl<D, E> Multiplication<D, E>
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

impl<D, E> Forward for Multiplication<D, E>
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
            .for_each(|v, l, r| *v = l * r);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct MultiplicationBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Rc<RefCell<Tensor<D>>>,
    left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    right_data: Rc<RefCell<Tensor<E>>>,
    right_gradient: Rc<RefCell<Option<Tensor<E>>>>,
    gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    shape: Broadcasted<D, E>,
    buffer: Rc<RefCell<Option<BroadTensor<D, E>>>>,
}

impl<D, E> MultiplicationBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_data: Rc<RefCell<Tensor<D>>>,
        left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        right_data: Rc<RefCell<Tensor<E>>>,
        right_gradient: Rc<RefCell<Option<Tensor<E>>>>,
        gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
        shape: Broadcasted<D, E>,
        buffer: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    ) -> Self {
        Self {
            left_data,
            left_gradient,
            right_data,
            right_gradient,
            gradient,
            shape,
            buffer,
        }
    }
}

impl<D, E> Backward for MultiplicationBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut left_gradient = expect_tensor_mut(&self.left_gradient);
        let mut right_gradient = expect_tensor_mut(&self.right_gradient);
        let mut buffer = expect_tensor_mut(&self.buffer);
        let gradient = expect_tensor(&self.gradient);

        {
            {
                let right_data = self.right_data.borrow();

                Zip::from(&mut *buffer)
                    .and(&*gradient)
                    .and_broadcast(&*right_data)
                    .for_each(|d, g, r| *d = g * r);

                let reduced = reduce(left_gradient.raw_dim(), &buffer);
                *left_gradient += &reduced;
            }

            {
                let left_data = self.left_data.borrow();

                Zip::from(&mut *buffer)
                    .and(&*gradient)
                    .and_broadcast(&*left_data)
                    .for_each(|d, g, r| *d = g * r);

                let reduced = reduce(right_gradient.raw_dim(), &buffer);
                *right_gradient += &reduced;
            }
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    right_data: Rc<RefCell<Tensor<E>>>,
    gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    shape: Broadcasted<D, E>,
    buffer: Rc<RefCell<Option<BroadTensor<D, E>>>>,
}

impl<D, E> MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        right_data: Rc<RefCell<Tensor<E>>>,
        gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
        shape: Broadcasted<D, E>,
        buffer: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    ) -> Self {
        Self {
            left_gradient,
            right_data,
            gradient,
            shape,
            buffer,
        }
    }
}

impl<D, E> Backward for MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut left_gradient = expect_tensor_mut(&self.left_gradient);
        let gradient = expect_tensor(&self.gradient);
        let mut buffer = expect_tensor_mut(&self.buffer);

        {
            let right_data = self.right_data.borrow();

            Zip::from(&mut *buffer)
                .and(&*gradient)
                .and_broadcast(&*right_data)
                .for_each(|d, g, v| *d = g * v);

            let reduced = reduce(left_gradient.raw_dim(), &buffer);
            *left_gradient += &reduced;
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Rc<RefCell<Tensor<D>>>,
    right_gradient: Rc<RefCell<Option<Tensor<E>>>>,
    gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    shape: Broadcasted<D, E>,
    buffer: Rc<RefCell<Option<BroadTensor<D, E>>>>,
}

impl<D, E> MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        left_data: Rc<RefCell<Tensor<D>>>,
        right_gradient: Rc<RefCell<Option<Tensor<E>>>>,
        gradient: Rc<RefCell<Option<BroadTensor<D, E>>>>,
        shape: Broadcasted<D, E>,
        buffer: Rc<RefCell<Option<BroadTensor<D, E>>>>,
    ) -> Self {
        Self {
            left_data,
            right_gradient,
            gradient,
            shape,
            buffer,
        }
    }
}

impl<D, E> Backward for MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut right_gradient = expect_tensor_mut(&self.right_gradient);
        let gradient = expect_tensor(&self.gradient);
        let mut buffer = expect_tensor_mut(&self.buffer);

        {
            let left_data = self.left_data.borrow();

            Zip::from(&mut *buffer)
                .and(&*gradient)
                .and_broadcast(&*left_data)
                .for_each(|d, g, v| *d = g * v);

            let reduced = reduce(right_gradient.raw_dim(), &buffer);
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
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
