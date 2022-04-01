use super::{Backward, Forward, Gradient, Shared};
use ndarray::{arr0, Array, Array0, Dimension, Ix0, Zip};
use std::rc::Rc;

pub(crate) struct Sum<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array0<f32>>,
}

impl<D> Sum<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_data: Shared<Array<f32, D>>, data: Shared<Array0<f32>>) -> Self {
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

pub(crate) struct SumBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<Ix0>>,
}

impl<D> SumBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_gradient: Rc<Gradient<D>>, gradient: Rc<Gradient<Ix0>>) -> Self {
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
        Zip::from(&mut *self.operand_gradient.borrow_mut())
            .and_broadcast(&*self.gradient.borrow())
            .for_each(|op_grad_el, &grad_el| *op_grad_el += grad_el);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
