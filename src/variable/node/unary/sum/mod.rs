use super::{Backward, Forward, OptionalTensor, Tensor};
use ndarray::{arr0, Dimension, Ix0, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Sum<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<Ix0>>>,
    computed: Cell<bool>,
}

impl<D> Sum<D>
where
    D: Dimension,
{
    pub fn new(operand_data: Rc<RefCell<Tensor<D>>>, data: Rc<RefCell<Tensor<Ix0>>>) -> Self {
        Self {
            operand_data,
            data,
            computed: Cell::default(),
        }
    }
}

impl<D> Forward for Sum<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        *self.data.borrow_mut() = arr0(self.operand_data.borrow().sum());
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct SumBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    gradient: Rc<OptionalTensor<Ix0>>,
}

impl<D> SumBackward<D>
where
    D: Dimension,
{
    pub fn new(operand_gradient: Rc<OptionalTensor<D>>, gradient: Rc<OptionalTensor<Ix0>>) -> Self {
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
        Zip::from(&mut *self.operand_gradient.content_mut())
            .and_broadcast(&*self.gradient.content())
            .for_each(|op_grad_el, &grad_el| *op_grad_el += grad_el);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
