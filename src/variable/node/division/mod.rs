use super::{
    reduce, Backward, Broadcasted, Forward, SharedTensor, SwitchableBufferedTensor,
    SwitchableTensor,
};
use ndarray::{DimMax, Dimension, Zip};
use std::rc::Rc;

pub struct Division<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: SharedTensor<D>,
    right_data: SharedTensor<E>,
    data: SharedTensor<Broadcasted<D, E>>,
}

impl<D, E> Division<D, E>
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

pub struct DivisionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_gradient: Rc<SwitchableTensor<D>>,
    right_data: SharedTensor<E>,
    gradient: Rc<SwitchableBufferedTensor<Broadcasted<D, E>>>,
}

impl<D, E> DivisionBackwardLeft<D, E>
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

impl<D, E> Backward for DivisionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut buffer = self.gradient.buffer_mut();
        Zip::from(&mut *buffer)
            .and(&*self.gradient.array())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|d, &g, &r| *d = g / r);

        let reduced = reduce(self.left_gradient.shape(), &buffer);
        *self.left_gradient.array_mut() += &reduced;
    }
}

pub struct DivisionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: SharedTensor<D>,
    right_data: SharedTensor<E>,
    right_gradient: Rc<SwitchableTensor<E>>,
    gradient: Rc<SwitchableBufferedTensor<Broadcasted<D, E>>>,
}

impl<D, E> DivisionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_data: SharedTensor<D>,
        right_data: SharedTensor<E>,
        right_gradient: Rc<SwitchableTensor<E>>,
        gradient: Rc<SwitchableBufferedTensor<Broadcasted<D, E>>>,
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
            .and(&*self.gradient.array())
            .and_broadcast(&*self.left_data.borrow())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|d, &g, &l, &r| *d = -g * l / r.powi(2));

        let reduced = reduce(self.right_gradient.shape(), &buffer);
        *self.right_gradient.array_mut() += &reduced;
    }
}

pub struct DivisionBackward<D, E>
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
    pub fn new(left: DivisionBackwardLeft<D, E>, right: DivisionBackwardRight<D, E>) -> Self {
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
