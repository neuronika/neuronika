#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{Axis, Dimension, Slice};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct MultiConcatenate<D>
where
    D: Dimension,
{
    operands_data: Vec<Rc<RefCell<Tensor<D>>>>,
    data: Rc<RefCell<Tensor<D>>>,
    axis: usize,
    computed: Cell<bool>,
}

impl<D> MultiConcatenate<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operands_data: Vec<Rc<RefCell<Tensor<D>>>>,
        data: Rc<RefCell<Tensor<D>>>,
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

impl<D> Forward for MultiConcatenate<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (axis, mut offset, mut data) = (self.axis, 0, self.data.borrow_mut());

        self.operands_data.iter().for_each(|operand| {
            let operand_data = operand.borrow();
            let axis_len = operand_data.len_of(Axis(axis));
            let slice = Slice::from(offset..axis_len + offset);

            let mut view_mut = data.slice_axis_mut(Axis(axis), slice);
            view_mut.assign(&operand_data);

            offset += axis_len;
        });
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}
pub struct MultiConcatenateBackward<D>
where
    D: Dimension,
{
    operands_gradients: Vec<Rc<RefCell<Option<Tensor<D>>>>>,
    gradient: Rc<RefCell<Option<Tensor<D>>>>,
    shape: D,
    axis: usize,
}

impl<D> MultiConcatenateBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operands_gradients: Vec<Rc<RefCell<Option<Tensor<D>>>>>,
        gradient: Rc<RefCell<Option<Tensor<D>>>>,
        shape: D,
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

impl<D> Backward for MultiConcatenateBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let (axis, gradient, mut offset) = (self.axis, expect_tensor(&self.gradient), 0);

        self.operands_gradients
            .iter()
            .map(expect_tensor_mut)
            .for_each(|mut operand_gradient| {
                let axis_len = operand_gradient.len_of(Axis(axis));
                let grad_view =
                    gradient.slice_axis(Axis(axis), Slice::from(offset..axis_len + offset));

                *operand_gradient += &grad_view;
                offset += axis_len;
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
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
