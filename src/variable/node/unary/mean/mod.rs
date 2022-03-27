#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{arr0, Dimension, Ix0, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Mean<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<Ix0>>>,
    computed: Cell<bool>,
}

impl<D> Mean<D>
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

impl<D> Forward for Mean<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        *self.data.borrow_mut() = arr0(self.operand_data.borrow().mean().unwrap());
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct MeanBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix0>>>>,
}

impl<D> MeanBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix0>>>>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D> Backward for MeanBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);
        let den = operand_gradient.len() as f32;

        Zip::from(&mut *operand_gradient)
            .and_broadcast(&*gradient)
            .for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el / den);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(arr0(0.));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
