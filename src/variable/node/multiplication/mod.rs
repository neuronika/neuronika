use super::{
    reduce, Backward, Broadcasted, Forward, SharedTensor, SwitchableBufferedTensor,
    SwitchableTensor,
};
use ndarray::{DimMax, Dimension, Zip};
use std::rc::Rc;

pub struct Multiplication<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: SharedTensor<D>,
    right_data: SharedTensor<E>,
    data: SharedTensor<Broadcasted<D, E>>,
}

impl<D, E> Multiplication<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_data: SharedTensor<D>,
        right_data: SharedTensor<E>,
        data: SharedTensor<Broadcasted<D, E>>,
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

pub struct MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_gradient: Rc<SwitchableTensor<D>>,
    right_data: SharedTensor<E>,
    gradient: Rc<SwitchableBufferedTensor<Broadcasted<D, E>>>,
}

impl<D, E> MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_gradient: Rc<SwitchableTensor<D>>,
        right_data: SharedTensor<E>,
        gradient: Rc<SwitchableBufferedTensor<Broadcasted<D, E>>>,
    ) -> Self {
        Self {
            left_gradient,
            right_data,
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
            .and(&*self.gradient.array())
            .and_broadcast(&*right_data)
            .for_each(|d, &g, &v| *d = g * v);

        let reduced = reduce(self.left_gradient.shape(), &buffer);
        *self.left_gradient.array_mut() += &reduced;
    }
}

pub struct MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    right_gradient: Rc<SwitchableTensor<E>>,
    left_data: SharedTensor<D>,
    gradient: Rc<SwitchableBufferedTensor<Broadcasted<D, E>>>,
}

impl<D, E> MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        right_gradient: Rc<SwitchableTensor<E>>,
        left_data: SharedTensor<D>,
        gradient: Rc<SwitchableBufferedTensor<Broadcasted<D, E>>>,
    ) -> Self {
        Self {
            right_gradient,
            left_data,
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
            .and(&*self.gradient.array())
            .and_broadcast(&*self.left_data.borrow())
            .for_each(|d, &g, &v| *d = g * v);

        let reduced = reduce(self.right_gradient.shape(), &buffer);
        *self.right_gradient.array_mut() += &reduced;
    }
}

pub struct MultiplicationBackward<D, E>
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
    pub fn new(
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
