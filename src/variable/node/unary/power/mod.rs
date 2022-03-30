use super::{Backward, Forward, OptionalTensor, Tensor};
use ndarray::{Dimension, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Power<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    exp: i32,
    computed: Cell<bool>,
}

impl<D> Power<D>
where
    D: Dimension,
{
    pub fn new(
        operand_data: Rc<RefCell<Tensor<D>>>,
        data: Rc<RefCell<Tensor<D>>>,
        exp: i32,
    ) -> Self {
        Self {
            operand_data,
            data,
            exp,
            computed: Cell::default(),
        }
    }
}

impl<D> Forward for Power<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let exp = self.exp;
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .for_each(|v, o| *v = o.powi(exp));
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct PowerBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    operand_data: Rc<RefCell<Tensor<D>>>,
    gradient: Rc<OptionalTensor<D>>,
    exp: i32,
}

impl<D> PowerBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<D>>,
        operand_data: Rc<RefCell<Tensor<D>>>,
        gradient: Rc<OptionalTensor<D>>,
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
        Zip::from(&mut *self.operand_gradient.content_mut())
            .and(&*self.gradient.content())
            .and(&*self.operand_data.borrow())
            .for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += grad_el * op_data_el.powi(self.exp - 1) * self.exp as f32;
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
