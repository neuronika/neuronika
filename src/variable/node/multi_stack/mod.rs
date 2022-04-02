use std::rc::Rc;

use ndarray::{Array, Axis, Dimension, RemoveAxis};

use crate::variable::{gradient::Gradient, utils::Shared};

use super::{Backward, Forward};

pub(crate) struct MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    operands_data: Vec<Shared<Array<f32, D>>>,
    data: Shared<Array<f32, D::Larger>>,
    axis: Axis,
}

impl<D> MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operands_data: Vec<Shared<Array<f32, D>>>,
        data: Shared<Array<f32, D::Larger>>,
        axis: usize,
    ) -> Self {
        Self {
            operands_data,
            data,
            axis: Axis(axis),
        }
    }
}

impl<D> Forward for MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    fn forward(&self) {
        let mut data = self.data.borrow_mut();
        self.operands_data
            .iter()
            .zip(data.axis_iter_mut(self.axis))
            .for_each(|(operand, mut axis_data)| {
                let operand_data = operand.borrow();
                axis_data.assign(&operand_data)
            });
    }
}

pub(crate) struct MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    operands_gradients: Vec<Rc<Gradient<D>>>,
    gradient: Rc<Gradient<D::Larger>>,
    axis: Axis,
}

impl<D> MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operands_gradients: Vec<Rc<Gradient<D>>>,
        gradient: Rc<Gradient<D::Larger>>,
        axis: usize,
    ) -> Self {
        Self {
            operands_gradients,
            gradient,
            axis: Axis(axis),
        }
    }
}

impl<D> Backward for MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        self.operands_gradients
            .iter()
            .map(|operand| operand.borrow_mut())
            .zip(self.gradient.borrow().axis_iter(self.axis))
            .for_each(|(mut operand_gradient, grad_view)| {
                *operand_gradient += &grad_view;
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
