use std::rc::Rc;

use ndarray::{Array, DimMax, Dimension, Zip};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::{accumulate, Broadcast, Shared},
};

pub(crate) struct Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Shared<Array<f32, D>>,
    right_data: Shared<Array<f32, E>>,
    data: Shared<Array<f32, Broadcast<D, E>>>,
}

impl<D, E> Addition<D, E>
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

impl<D, E> Forward for Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left_data.borrow())
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|v, &l, &r| *v = l + r);
    }
}
pub(crate) struct AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
    gradient: Rc<Gradient<Array<f32, Broadcast<D, E>>, Broadcast<D, E>>>,
}

impl<D, E> AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
        gradient: Rc<Gradient<Array<f32, Broadcast<D, E>>, Broadcast<D, E>>>,
    ) -> Self {
        debug_assert!(operand_gradient
            .borrow()
            .broadcast(gradient.shape())
            .is_some());

        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D, E> Backward for AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        accumulate(
            &mut self.operand_gradient.borrow_mut(),
            &self.gradient.borrow(),
        );
    }
}

pub(crate) struct AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand_gradient: Rc<Gradient<Array<f32, E>, E>>,
    gradient: Rc<Gradient<Array<f32, Broadcast<D, E>>, Broadcast<D, E>>>,
}

impl<D, E> AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<Array<f32, E>, E>>,
        gradient: Rc<Gradient<Array<f32, Broadcast<D, E>>, Broadcast<D, E>>>,
    ) -> Self {
        debug_assert!(operand_gradient
            .borrow()
            .broadcast(gradient.shape())
            .is_some());

        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D, E> Backward for AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        accumulate(
            &mut self.operand_gradient.borrow_mut(),
            &self.gradient.borrow(),
        );
    }
}

pub(crate) struct AdditionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left: AdditionBackwardLeft<D, E>,
    right: AdditionBackwardRight<D, E>,
}

impl<D, E> AdditionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        left: AdditionBackwardLeft<D, E>,
        right: AdditionBackwardRight<D, E>,
    ) -> Self {
        Self { left, right }
    }
}

impl<D, E> Backward for AdditionBackward<D, E>
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
