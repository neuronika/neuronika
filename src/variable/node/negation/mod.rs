use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{Dimension, Zip};
use std::rc::Rc;

pub struct Negation<D>
where
    D: Dimension,
{
    operand_data: SharedTensor<D>,
    data: SharedTensor<D>,
}

impl<D> Negation<D>
where
    D: Dimension,
{
    pub fn new(operand_data: SharedTensor<D>, data: SharedTensor<D>) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for Negation<D>
where
    D: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .for_each(|v, &o| *v = -o);
    }
}

pub struct NegationBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<D>>,
}

impl<D> NegationBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<D>>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D> Backward for NegationBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        *self.operand_gradient.array_mut() -= &*self.gradient.array();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
