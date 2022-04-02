use std::rc::Rc;

use crate::variable::{gradient::Gradient, utils::Shared};

use ndarray::{Array, Dimension, Zip};

use super::{Backward, Forward};

pub(crate) struct Transpose<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
}

impl<D> Transpose<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_data: Shared<Array<f32, D>>, data: Shared<Array<f32, D>>) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for Transpose<D>
where
    D: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and(self.operand_data.borrow().t())
            .for_each(|v, &o| *v = o);
    }
}

pub(crate) struct TransposeBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<D>>,
}

impl<D> TransposeBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_gradient: Rc<Gradient<D>>, gradient: Rc<Gradient<D>>) -> Self {
        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D> Backward for TransposeBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        *self.operand_gradient.borrow_mut() += &self.gradient.borrow().t();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
