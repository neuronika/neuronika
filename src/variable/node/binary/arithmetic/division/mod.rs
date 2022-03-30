use super::{reduce, Backward, BroadTensor, Broadcasted, Forward, OptionalTensor, Shared, Tensor};
use ndarray::{DimMax, Dimension, Zip};
use std::{cell::Cell, rc::Rc};

pub struct Division<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Shared<Tensor<D>>,
    right_data: Shared<Tensor<E>>,
    data: Shared<BroadTensor<D, E>>,
    computed: Cell<bool>,
}

impl<D, E> Division<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_data: Shared<Tensor<D>>,
        right_data: Shared<Tensor<E>>,
        data: Shared<BroadTensor<D, E>>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
            computed: Cell::default(),
        }
    }
}

impl<D, E> Forward for Division<D, E>
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
            .for_each(|v, l, r| *v = l / r);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct DivisionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_gradient: Rc<OptionalTensor<D>>,
    right_data: Shared<Tensor<E>>,
    gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
    buffer: Rc<OptionalTensor<Broadcasted<D, E>>>,
}

impl<D, E> DivisionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_gradient: Rc<OptionalTensor<D>>,
        right_data: Shared<Tensor<E>>,
        gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
        buffer: Rc<OptionalTensor<Broadcasted<D, E>>>,
    ) -> Self {
        Self {
            left_gradient,
            right_data,
            gradient,
            buffer,
        }
    }
}

impl<D, E> Backward for DivisionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut buffer = self.buffer.content_mut();
        Zip::from(&mut *buffer)
            .and(&*self.gradient.content())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|d, g, r| *d = g / r);

        let reduced = reduce(self.left_gradient.shape(), &buffer);
        *self.left_gradient.content_mut() += &reduced;
    }
}

pub struct DivisionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Shared<Tensor<D>>,
    right_data: Shared<Tensor<E>>,
    right_gradient: Rc<OptionalTensor<E>>,
    gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
    buffer: Rc<OptionalTensor<Broadcasted<D, E>>>,
}

impl<D, E> DivisionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_data: Shared<Tensor<D>>,
        right_data: Shared<Tensor<E>>,
        right_gradient: Rc<OptionalTensor<E>>,
        gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
        buffer: Rc<OptionalTensor<Broadcasted<D, E>>>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            right_gradient,
            gradient,
            buffer,
        }
    }
}

impl<D, E> Backward for DivisionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut buffer = self.buffer.content_mut();
        Zip::from(&mut *buffer)
            .and(&*self.gradient.content())
            .and_broadcast(&*self.left_data.borrow())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|d, g, l, r| *d = -g * l / r.powi(2));

        let reduced = reduce(self.right_gradient.shape(), &buffer);
        *self.right_gradient.content_mut() += &reduced;
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
