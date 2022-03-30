use super::{Backward, Forward, OptionalTensor, Tensor};
use ndarray::{Dimension, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};
pub struct Transpose<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    computed: Cell<bool>,
}

impl<D> Transpose<D>
where
    D: Dimension,
{
    pub fn new(operand_data: Rc<RefCell<Tensor<D>>>, data: Rc<RefCell<Tensor<D>>>) -> Self {
        Self {
            operand_data,
            data,
            computed: Cell::default(),
        }
    }
}

impl<D> Forward for Transpose<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(self.operand_data.borrow().t())
            .for_each(|v, o| *v = *o);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct TransposeBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    gradient: Rc<OptionalTensor<D>>,
}

impl<D> TransposeBackward<D>
where
    D: Dimension,
{
    pub fn new(operand_gradient: Rc<OptionalTensor<D>>, gradient: Rc<OptionalTensor<D>>) -> Self {
        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D> Backward for TransposeBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        *self.operand_gradient.content_mut() += &self.gradient.content().t();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
