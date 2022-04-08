use std::rc::Rc;

use ndarray::{Array, Dimension, Zip};

use crate::{gradient::Gradient, utils::Shared};

use super::{Backward, Forward};

pub(crate) struct Sqrt<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
}

impl<D> Sqrt<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_data: Shared<Array<f32, D>>, data: Shared<Array<f32, D>>) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for Sqrt<D>
where
    D: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .for_each(|v, &o| *v = o.sqrt());
    }
}

pub(crate) struct SqrtBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    data: Shared<Array<f32, D>>,
    gradient: Rc<Gradient<D>>,
}

impl<D> SqrtBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<D>>,
        data: Shared<Array<f32, D>>,
        gradient: Rc<Gradient<D>>,
    ) -> Self {
        Self {
            operand_gradient,
            data,
            gradient,
        }
    }
}

impl<D> Backward for SqrtBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(&mut *self.operand_gradient.borrow_mut())
            .and(&*self.gradient.borrow())
            .and(&*self.data.borrow())
            .for_each(|op_grad_el, &grad_el, &data| *op_grad_el += grad_el / (data * 2.));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
