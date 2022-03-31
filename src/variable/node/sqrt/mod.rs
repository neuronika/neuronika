use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{Dimension, Zip};
use std::rc::Rc;

pub struct Sqrt<D>
where
    D: Dimension,
{
    operand_data: SharedTensor<D>,
    data: SharedTensor<D>,
}

impl<D> Sqrt<D>
where
    D: Dimension,
{
    pub fn new(operand_data: SharedTensor<D>, data: SharedTensor<D>) -> Self {
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

pub struct SqrtBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    data: SharedTensor<D>,
    gradient: Rc<SwitchableTensor<D>>,
}

impl<D> SqrtBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        data: SharedTensor<D>,
        gradient: Rc<SwitchableTensor<D>>,
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
        Zip::from(&mut *self.operand_gradient.array_mut())
            .and(&*self.gradient.array())
            .and(&*self.data.borrow())
            .for_each(|op_grad_el, &grad_el, &data| *op_grad_el += grad_el / (data * 2.));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
