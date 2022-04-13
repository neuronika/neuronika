use std::rc::Rc;

use ndarray::{arr0, Array, Dimension, Ix0, Zip};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
};

pub(crate) struct Mean<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, Ix0>>,
}

impl<D> Mean<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_data: Shared<Array<f32, D>>, data: Shared<Array<f32, Ix0>>) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for Mean<D>
where
    D: Dimension,
{
    fn forward(&self) {
        *self.data.borrow_mut() = arr0(self.operand_data.borrow().mean().unwrap());
    }
}

pub(crate) struct MeanBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<Ix0>>,
}

impl<D> MeanBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_gradient: Rc<Gradient<D>>, gradient: Rc<Gradient<Ix0>>) -> Self {
        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D> Backward for MeanBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = self.operand_gradient.borrow_mut();
        let den = operand_gradient.len() as f32;

        Zip::from(&mut *operand_gradient)
            .and_broadcast(&*self.gradient.borrow())
            .for_each(|op_grad_el, &grad_el| *op_grad_el += grad_el / den);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
