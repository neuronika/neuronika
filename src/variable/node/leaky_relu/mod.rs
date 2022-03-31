use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{Dimension, Zip};
use std::rc::Rc;

#[allow(clippy::upper_case_acronyms)]
pub struct LeakyReLU<D>
where
    D: Dimension,
{
    operand_data: SharedTensor<D>,
    data: SharedTensor<D>,
}

impl<D> LeakyReLU<D>
where
    D: Dimension,
{
    pub fn new(operand_data: SharedTensor<D>, data: SharedTensor<D>) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for LeakyReLU<D>
where
    D: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .for_each(|v, &o| {
                *v = ((o > 0.) as u8 as f32) * o + ((o <= 0.) as u8 as f32) * (0.01 * o)
            });
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct LeakyReLUBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    operand_data: SharedTensor<D>,
    gradient: Rc<SwitchableTensor<D>>,
}

impl<D> LeakyReLUBackward<D>
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

impl<D> Backward for LeakyReLUBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(&mut *self.operand_gradient.array_mut())
            .and(&*self.gradient.array())
            .and(&*self.operand_data.borrow())
            .for_each(|op_grad_el, &grad_el, &op_data_el| {
                *op_grad_el += ((op_data_el > 0.) as u8 as f32) * grad_el
                    + ((op_data_el <= 0.) as u8 as f32) * 0.01;
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
