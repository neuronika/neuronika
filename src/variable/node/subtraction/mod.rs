use super::{reduce, Backward, Broadcasted, Forward, SharedTensor, SwitchableTensor};
use ndarray::{DimMax, Dimension, Zip};
use std::rc::Rc;

pub struct Subtraction<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: SharedTensor<D>,
    right_data: SharedTensor<E>,
    data: SharedTensor<Broadcasted<D, E>>,
}

impl<D, E> Subtraction<D, E>
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

pub struct SubtractionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<Broadcasted<D, E>>>,
}

impl<D, E> SubtractionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<Broadcasted<D, E>>>,
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
        let reduced = reduce(self.operand_gradient.shape(), &self.gradient.array());
        *self.operand_gradient.array_mut() += &reduced
    }
}

pub struct SubtractionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<E>>,
    gradient: Rc<SwitchableTensor<Broadcasted<D, E>>>,
}

impl<D, E> SubtractionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<E>>,
        gradient: Rc<SwitchableTensor<Broadcasted<D, E>>>,
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
        let reduced = reduce(self.gradient.shape(), &self.gradient.array());
        *self.operand_gradient.array_mut() -= &reduced;
    }
}

pub struct SubtractionBackward<D, E>
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
    pub fn new(left: SubtractionBackwardLeft<D, E>, right: SubtractionBackwardRight<D, E>) -> Self {
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
