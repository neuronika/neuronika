#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{Dimension, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct TanH<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    computed: Cell<bool>,
}

impl<D> TanH<D>
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

impl<D> Forward for TanH<D>
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
            .for_each(|v, o| *v = o.tanh());
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct TanHBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    data: Rc<RefCell<Tensor<D>>>,
    gradient: Rc<RefCell<Option<Tensor<D>>>>,
    shape: D,
}

impl<D> TanHBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        data: Rc<RefCell<Tensor<D>>>,
        gradient: Rc<RefCell<Option<Tensor<D>>>>,
        shape: D,
    ) -> Self {
        Self {
            operand_gradient,
            data,
            gradient,
            shape,
        }
    }
}

impl<D> Backward for TanHBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);
        let data = self.data.borrow();

        Zip::from(&mut *operand_gradient)
            .and(&*gradient)
            .and(&*data)
            .for_each(|op_grad_el, grad_el, data_el| {
                *op_grad_el += *grad_el * (1.0 - data_el.powi(2))
            })
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
