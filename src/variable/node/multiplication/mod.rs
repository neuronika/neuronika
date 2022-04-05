use std::rc::Rc;

use ndarray::{Array, DimMax, Dimension, Zip};

use crate::variable::{
    gradient::{BufferedGradient, Gradient},
    utils::{reduce, Broadcast, Shared},
};

use super::{Backward, Forward};

pub(crate) struct Multiplication<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Shared<Array<f32, D>>,
    right_data: Shared<Array<f32, E>>,
    data: Shared<Array<f32, Broadcast<D, E>>>,
}

impl<D, E> Multiplication<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        left_data: Shared<Array<f32, D>>,
        right_data: Shared<Array<f32, E>>,
        data: Shared<Array<f32, Broadcast<D, E>>>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
        }
    }
}

impl<D, E> Forward for Multiplication<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left_data.borrow())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|v, &l, &r| *v = l * r);
    }
}

pub(crate) struct MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    right_data: Shared<Array<f32, E>>,
    left_gradient: Rc<Gradient<D>>,
    gradient: Rc<BufferedGradient<Broadcast<D, E>>>,
}

impl<D, E> MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        right_data: Shared<Array<f32, E>>,
        left_gradient: Rc<Gradient<D>>,
        gradient: Rc<BufferedGradient<Broadcast<D, E>>>,
    ) -> Self {
        Self {
            right_data,
            left_gradient,
            gradient,
        }
    }
}

impl<D, E> Backward for MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut buffer = self.gradient.buffer_mut();
        let right_data = self.right_data.borrow();
        Zip::from(&mut *buffer)
            .and(&*self.gradient.borrow())
            .and_broadcast(&*right_data)
            .for_each(|d, &g, &v| *d = g * v);

        let reduced = reduce(self.left_gradient.shape(), &buffer);
        *self.left_gradient.borrow_mut() += &reduced;
    }
}

pub(crate) struct MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Shared<Array<f32, D>>,
    right_gradient: Rc<Gradient<E>>,
    gradient: Rc<BufferedGradient<Broadcast<D, E>>>,
}

impl<D, E> MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        left_data: Shared<Array<f32, D>>,
        right_gradient: Rc<Gradient<E>>,
        gradient: Rc<BufferedGradient<Broadcast<D, E>>>,
    ) -> Self {
        Self {
            left_data,
            right_gradient,
            gradient,
        }
    }
}

impl<D, E> Backward for MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut buffer = self.gradient.buffer_mut();
        Zip::from(&mut *buffer)
            .and(&*self.gradient.borrow())
            .and_broadcast(&*self.left_data.borrow())
            .for_each(|d, &g, &v| *d = g * v);

        let reduced = reduce(self.right_gradient.shape(), &buffer);
        *self.right_gradient.borrow_mut() += &reduced;
    }
}

pub(crate) struct MultiplicationBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left: MultiplicationBackwardLeft<D, E>,
    right: MultiplicationBackwardRight<D, E>,
}

impl<D, E> MultiplicationBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        left: MultiplicationBackwardLeft<D, E>,
        right: MultiplicationBackwardRight<D, E>,
    ) -> Self {
        Self { left, right }
    }
}

impl<D, E> Backward for MultiplicationBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        self.left.backward();
        self.right.backward();
    }
}

#[cfg(test)]
mod test;
