use super::{Backward, Forward, Gradient, Shared};
use ndarray::{Array, Dimension, Zip};
use std::rc::Rc;

#[allow(clippy::upper_case_acronyms)]
pub(crate) struct ReLU<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
}

impl<D> ReLU<D>
where
    D: Dimension,
{
    pub(crate) fn new(operand_data: Shared<Array<f32, D>>, data: Shared<Array<f32, D>>) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for ReLU<D>
where
    D: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .for_each(|v, &o| *v = o.max(0.));
    }
}

#[allow(clippy::upper_case_acronyms)]
pub(crate) struct ReLUBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    operand_data: Shared<Array<f32, D>>,
    gradient: Rc<Gradient<D>>,
}

impl<D> ReLUBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<D>>,
        operand_data: Shared<Array<f32, D>>,
        gradient: Rc<Gradient<D>>,
    ) -> Self {
        Self {
            operand_gradient,
            operand_data,
            gradient,
        }
    }
}

impl<D> Backward for ReLUBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(&mut *self.operand_gradient.borrow_mut())
            .and(&*self.gradient.borrow())
            .and(&*self.operand_data.borrow())
            .for_each(|op_grad_el, &grad_el, &op_data_el| {
                *op_grad_el += ((op_data_el > 0.) as usize as f32) * grad_el;
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
