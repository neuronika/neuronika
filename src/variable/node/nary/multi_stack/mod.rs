#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{Axis, Dimension, RemoveAxis};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    operands_data: Vec<Rc<RefCell<Tensor<D>>>>,
    data: Rc<RefCell<Tensor<D::Larger>>>,
    axis: usize,
    computed: Cell<bool>,
}

impl<D> MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operands_data: Vec<Rc<RefCell<Tensor<D>>>>,
        data: Rc<RefCell<Tensor<D::Larger>>>,
        axis: usize,
    ) -> Self {
        Self {
            operands_data,
            data,
            axis,
            computed: Cell::default(),
        }
    }
}

impl<D> Forward for MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (mut data, axis) = (self.data.borrow_mut(), self.axis);

        self.operands_data
            .iter()
            .zip(data.axis_iter_mut(Axis(axis)))
            .for_each(|(operand, mut axis_data)| {
                let operand_data = operand.borrow();
                axis_data.assign(&operand_data)
            });
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    operands_gradients: Vec<Rc<RefCell<Option<Tensor<D>>>>>,
    gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
    shape: D::Larger,
    axis: usize,
}

impl<D> MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operands_gradients: Vec<Rc<RefCell<Option<Tensor<D>>>>>,
        gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
        shape: D::Larger,
        axis: usize,
    ) -> Self {
        Self {
            operands_gradients,
            gradient,
            shape,
            axis,
        }
    }
}

impl<D> Backward for MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let (axis, grad) = (self.axis, expect_tensor(&self.gradient));

        self.operands_gradients
            .iter()
            .map(expect_tensor_mut)
            .zip(grad.axis_iter(Axis(axis)))
            .for_each(|(mut operand_gradient, grad_view)| {
                *operand_gradient += &grad_view;
            });
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
