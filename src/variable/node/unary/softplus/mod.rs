use super::{Backward, Forward, OptionalTensor, Tensor};
use ndarray::{Dimension, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct SoftPlus<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    computed: Cell<bool>,
}

impl<D> SoftPlus<D>
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

impl<D> Forward for SoftPlus<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .for_each(|v, o| *v = (1.0 + o.exp()).ln());
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct SoftPlusBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    operand_data: Rc<RefCell<Tensor<D>>>,
    gradient: Rc<OptionalTensor<D>>,
}

impl<D> SoftPlusBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<D>>,
        operand_data: Rc<RefCell<Tensor<D>>>,
        gradient: Rc<OptionalTensor<D>>,
    ) -> Self {
        Self {
            operand_gradient,
            operand_data,
            gradient,
        }
    }
}

impl<D> Backward for SoftPlusBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(&mut *self.operand_gradient.content_mut())
            .and(&*self.gradient.content())
            .and(&*self.operand_data.borrow())
            .for_each(|op_grad_el, &grad_el, &op_data_el| {
                *op_grad_el += grad_el / (1. + (-op_data_el).exp())
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
