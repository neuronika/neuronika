use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{arr0, Dimension, Ix0, Zip};
use std::rc::Rc;

pub struct Mean<D>
where
    D: Dimension,
{
    operand_data: SharedTensor<D>,
    data: SharedTensor<Ix0>,
}

impl<D> Mean<D>
where
    D: Dimension,
{
    pub fn new(operand_data: SharedTensor<D>, data: SharedTensor<Ix0>) -> Self {
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

pub struct MeanBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<Ix0>>,
}

impl<D> MeanBackward<D>
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

impl<D> Backward for MeanBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = self.operand_gradient.array_mut();
        let den = operand_gradient.len() as f32;

        Zip::from(&mut *operand_gradient)
            .and_broadcast(&*self.gradient.array())
            .for_each(|op_grad_el, &grad_el| *op_grad_el += grad_el / den);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
