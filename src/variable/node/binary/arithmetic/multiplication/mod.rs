use super::{reduce, Backward, BroadTensor, Broadcasted, Forward, OptionalTensor, Tensor};
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

pub struct MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_gradient: Rc<OptionalTensor<D>>,
    right_data: Rc<RefCell<Tensor<E>>>,
    gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
    buffer: Rc<OptionalTensor<Broadcasted<D, E>>>,
}

impl<D, E> MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left_gradient: Rc<OptionalTensor<D>>,
        right_data: Rc<RefCell<Tensor<E>>>,
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

impl<D, E> Backward for MultiplicationBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let mut buffer = self.buffer.content_mut();
        let right_data = self.right_data.borrow();
        Zip::from(&mut *buffer)
            .and(&*self.gradient.content())
            .and_broadcast(&*right_data)
            .for_each(|d, g, v| *d = g * v);

        let reduced = reduce(self.left_gradient.shape(), &buffer);
        *self.left_gradient.content_mut() += &reduced;
    }
}

pub struct MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    right_gradient: Rc<OptionalTensor<E>>,
    left_data: Rc<RefCell<Tensor<D>>>,
    gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
    buffer: Rc<OptionalTensor<Broadcasted<D, E>>>,
}

impl<D, E> MultiplicationBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        right_gradient: Rc<OptionalTensor<E>>,
        left_data: Rc<RefCell<Tensor<D>>>,
        gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
        buffer: Rc<OptionalTensor<Broadcasted<D, E>>>,
    ) -> Self {
        Self {
            right_gradient,
            left_data,
            gradient,
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
        let mut buffer = self.buffer.content_mut();
        Zip::from(&mut *buffer)
            .and(&*self.gradient.content())
            .and_broadcast(&*self.left_data.borrow())
            .for_each(|d, g, v| *d = g * v);

        let reduced = reduce(self.right_gradient.shape(), &buffer);
        *self.right_gradient.content_mut() += &reduced;
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
