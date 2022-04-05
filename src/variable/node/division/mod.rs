use std::rc::Rc;

use ndarray::{Array, DimMax, Dimension, Zip};

use crate::variable::{
    gradient::{BufferedGradient, Gradient},
    utils::{reduce, Broadcast, Shared},
};

use super::{Backward, Forward};

pub(crate) struct Division<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Shared<Array<f32, D>>,
    right_data: Shared<Array<f32, E>>,
    data: Shared<Array<f32, Broadcast<D, E>>>,
}

impl<D, E> Division<D, E>
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

impl<D, E> Forward for Division<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left_data.borrow())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|v, &l, &r| *v = l / r);
    }
}

pub(crate) struct DivisionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    right_data: Shared<Array<f32, E>>,
    left_gradient: Rc<Gradient<D>>,
    gradient: Rc<BufferedGradient<Broadcast<D, E>>>,
}

impl<D, E> DivisionBackwardLeft<D, E>
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

impl<D, E> Backward for DivisionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut buffer = self.gradient.buffer_mut();
        Zip::from(&mut *buffer)
            .and(&*self.gradient.borrow())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|d, &g, &r| *d = g / r);

        let reduced = reduce(self.left_gradient.shape(), &buffer);
        *self.left_gradient.borrow_mut() += &reduced;
    }
}

pub(crate) struct DivisionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Shared<Array<f32, D>>,
    right_data: Shared<Array<f32, E>>,
    right_gradient: Rc<Gradient<E>>,
    gradient: Rc<BufferedGradient<Broadcast<D, E>>>,
}

impl<D, E> DivisionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        left_data: Shared<Array<f32, D>>,
        right_data: Shared<Array<f32, E>>,
        right_gradient: Rc<Gradient<E>>,
        gradient: Rc<BufferedGradient<Broadcast<D, E>>>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            right_gradient,
            gradient,
        }
    }
}

impl<D, E> Backward for DivisionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut buffer = self.gradient.buffer_mut();
        Zip::from(&mut *buffer)
            .and(&*self.gradient.borrow())
            .and_broadcast(&*self.left_data.borrow())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|d, &g, &l, &r| *d = -g * l / r.powi(2));

        let reduced = reduce(self.right_gradient.shape(), &buffer);
        *self.right_gradient.borrow_mut() += &reduced;
    }
}

pub(crate) struct DivisionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left: DivisionBackwardLeft<D, E>,
    right: DivisionBackwardRight<D, E>,
}

impl<D, E> DivisionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        left: DivisionBackwardLeft<D, E>,
        right: DivisionBackwardRight<D, E>,
    ) -> Self {
        Self { left, right }
    }
}

impl<D, E> Backward for DivisionBackward<D, E>
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
