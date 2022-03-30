use super::{reduce, Backward, BroadTensor, Broadcasted, Forward, OptionalTensor, Tensor};
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

pub struct SubtractionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
}

impl<D, E> SubtractionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<D>>,
        gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
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
        let reduced = reduce(self.operand_gradient.shape(), &self.gradient.content());
        *self.operand_gradient.content_mut() += &reduced
    }
}

pub struct SubtractionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<OptionalTensor<E>>,
    gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
}

impl<D, E> SubtractionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<E>>,
        gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
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
        let reduced = reduce(self.gradient.shape(), &self.gradient.content());
        *self.operand_gradient.content_mut() -= &reduced;
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
