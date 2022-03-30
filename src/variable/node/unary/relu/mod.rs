use super::{Backward, Forward, OptionalTensor, Tensor};
use ndarray::{Dimension, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[allow(clippy::upper_case_acronyms)]
pub struct ReLU<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    computed: Cell<bool>,
}

impl<D> ReLU<D>
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

impl<D> Forward for ReLU<D>
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
            .for_each(|v, o| *v = o.max(0.));
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct ReLUBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    operand_data: Rc<RefCell<Tensor<D>>>,
    gradient: Rc<OptionalTensor<D>>,
}

impl<D> ReLUBackward<D>
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

impl<D> Backward for ReLUBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(&mut *self.operand_gradient.content_mut())
            .and(&*self.gradient.content())
            .and(&*self.operand_data.borrow())
            .for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += ((*op_data_el > 0.) as usize as f32) * grad_el;
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
