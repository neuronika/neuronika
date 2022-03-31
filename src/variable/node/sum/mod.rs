use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{arr0, Dimension, Ix0, Zip};
use std::rc::Rc;

pub struct Sum<D>
where
    D: Dimension,
{
    operand_data: SharedTensor<D>,
    data: SharedTensor<Ix0>,
}

impl<D> Sum<D>
where
    D: Dimension,
{
    pub fn new(operand_data: SharedTensor<D>, data: SharedTensor<Ix0>) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for Sum<D>
where
    D: Dimension,
{
    fn forward(&self) {
        *self.data.borrow_mut() = arr0(self.operand_data.borrow().sum());
    }
}

pub struct SumBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<Ix0>>,
}

impl<D> SumBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<Ix0>>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D> Backward for SumBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(&mut *self.operand_gradient.array_mut())
            .and_broadcast(&*self.gradient.array())
            .for_each(|op_grad_el, &grad_el| *op_grad_el += grad_el);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
