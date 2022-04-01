use super::{reduce, Backward, Broadcast, Forward, Gradient, Shared};
use ndarray::{Array, DimMax, Dimension, Zip};
use std::rc::Rc;

pub(crate) struct Subtraction<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Shared<Array<f32, D>>,
    right_data: Shared<Array<f32, E>>,
    data: Shared<Array<f32, Broadcast<D, E>>>,
}

impl<D, E> Subtraction<D, E>
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

impl<D, E> Forward for Subtraction<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left_data.borrow())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|v, &l, &r| *v = l - r);
    }
}

pub(crate) struct SubtractionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<Broadcast<D, E>>>,
}

impl<D, E> SubtractionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<Broadcast<D, E>>>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D, E> Backward for SubtractionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let reduced = reduce(self.operand_gradient.shape(), &self.gradient.borrow());
        *self.operand_gradient.borrow_mut() += &reduced
    }
}

pub(crate) struct SubtractionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<Gradient<E>>,
    gradient: Rc<Gradient<Broadcast<D, E>>>,
}

impl<D, E> SubtractionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<E>>,
        gradient: Rc<Gradient<Broadcast<D, E>>>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D, E> Backward for SubtractionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let reduced = reduce(self.gradient.shape(), &self.gradient.borrow());
        *self.operand_gradient.borrow_mut() -= &reduced;
    }
}

pub(crate) struct SubtractionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left: SubtractionBackwardLeft<D, E>,
    right: SubtractionBackwardRight<D, E>,
}

impl<D, E> SubtractionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        left: SubtractionBackwardLeft<D, E>,
        right: SubtractionBackwardRight<D, E>,
    ) -> Self {
        Self { left, right }
    }
}

impl<D, E> Backward for SubtractionBackward<D, E>
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
