use super::{Backward, Forward, Gradient, Shared};
use ndarray::{Array, Dimension, Zip};
use std::rc::Rc;

pub(crate) struct Negation<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
}

impl<D> Negation<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_data: Shared<Array<f32, D>>, data: Shared<Array<f32, D>>) -> Self {
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

pub(crate) struct NegationBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<D>>,
}

impl<D> NegationBackward<D>
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

impl<D> Backward for NegationBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        *self.operand_gradient.borrow_mut() -= &*self.gradient.borrow();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
