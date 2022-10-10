use std::rc::Rc;

use ndarray::{Array, Dimension, Zip};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
};

pub(crate) struct Sigmoid<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
}

impl<D> Sigmoid<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_data: Shared<Array<f32, D>>, data: Shared<Array<f32, D>>) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for Sigmoid<D>
where
    D: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .for_each(|v, &o| *v = 1. / (1. + (-o).exp()));
    }
}

pub(crate) struct SigmoidBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
    data: Shared<Array<f32, D>>,
    gradient: Rc<Gradient<Array<f32, D>, D>>,
}

impl<D> SigmoidBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
        data: Shared<Array<f32, D>>,
        gradient: Rc<Gradient<Array<f32, D>, D>>,
    ) -> Self {
        Self {
            operand_gradient,
            data,
            gradient,
        }
    }
}

impl<D> Backward for SigmoidBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(&mut *self.operand_gradient.borrow_mut())
            .and(&*self.gradient.borrow())
            .and(&*self.data.borrow())
            .for_each(|op_grad_el, &grad_el, &data_el| {
                *op_grad_el += grad_el * data_el * (1. - data_el)
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
