use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{Dimension, Zip};
use std::rc::Rc;

pub struct SoftPlus<D>
where
    D: Dimension,
{
    operand_data: SharedTensor<D>,
    data: SharedTensor<D>,
}

impl<D> SoftPlus<D>
where
    D: Dimension,
{
    pub fn new(operand_data: SharedTensor<D>, data: SharedTensor<D>) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for SoftPlus<D>
where
    D: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .for_each(|v, &o| *v = (1. + o.exp()).ln());
    }
}

pub struct SoftPlusBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    operand_data: SharedTensor<D>,
    gradient: Rc<SwitchableTensor<D>>,
}

impl<D> SoftPlusBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        operand_data: SharedTensor<D>,
        gradient: Rc<SwitchableTensor<D>>,
    ) -> Self {
        Self {
            operand_gradient,
            operand_data,
            gradient,
        }
    }
}

impl<D> Backward for SoftPlusBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(&mut *self.operand_gradient.array_mut())
            .and(&*self.gradient.array())
            .and(&*self.operand_data.borrow())
            .for_each(|op_grad_el, &grad_el, &op_data_el| {
                *op_grad_el += grad_el / (1. + (-op_data_el).exp())
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
