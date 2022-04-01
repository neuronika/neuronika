use super::{Backward, Forward, Gradient, Shared};
use ndarray::{Array, Dimension, Zip};
use std::rc::Rc;

pub(crate) struct Power<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
    exp: i32,
}

impl<D> Power<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_data: Shared<Array<f32, D>>,
        data: Shared<Array<f32, D>>,
        exp: i32,
    ) -> Self {
        Self {
            operand_data,
            data,
            exp,
        }
    }
}

impl<D> Forward for Power<D>
where
    D: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .for_each(|v, &o| *v = o.powi(self.exp));
    }
}

pub(crate) struct PowerBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    operand_data: Shared<Array<f32, D>>,
    gradient: Rc<Gradient<D>>,
    exp: i32,
}

impl<D> PowerBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<D>>,
        operand_data: Shared<Array<f32, D>>,
        gradient: Rc<Gradient<D>>,
        exp: i32,
    ) -> Self {
        Self {
            operand_gradient,
            operand_data,
            gradient,
            exp,
        }
    }
}

impl<D> Backward for PowerBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(&mut *self.operand_gradient.borrow_mut())
            .and(&*self.gradient.borrow())
            .and(&*self.operand_data.borrow())
            .for_each(|op_grad_el, &grad_el, &op_data_el| {
                *op_grad_el += grad_el * op_data_el.powi(self.exp - 1) * self.exp as f32;
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
