use super::{Backward, Forward, OptionalTensor, Tensor};
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
    operands_gradients: Vec<Rc<OptionalTensor<D>>>,
    gradient: Rc<OptionalTensor<D::Larger>>,
    axis: Axis,
}

impl<D> MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operands_gradients: Vec<Rc<OptionalTensor<D>>>,
        gradient: Rc<OptionalTensor<D::Larger>>,
        axis: usize,
    ) -> Self {
        Self {
            operands_gradients,
            gradient,
            axis: Axis(axis),
        }
    }
}

impl<D> Backward for MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        self.operands_gradients
            .iter()
            .map(|operand| operand.content_mut())
            .zip(self.gradient.content().axis_iter(self.axis))
            .for_each(|(mut operand_gradient, grad_view)| {
                *operand_gradient += &grad_view;
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
