use super::{Backward, Forward, OptionalTensor, Tensor};
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
    axis: Axis,
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
            axis: Axis(axis),
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
        let (mut offset, mut data) = (0, self.data.borrow_mut());

        self.operands_data.iter().for_each(|operand| {
            let operand_data = operand.borrow();
            let axis_len = operand_data.len_of(self.axis);
            let slice = Slice::from(offset..axis_len + offset);

            data.slice_axis_mut(self.axis, slice).assign(&operand_data);
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
    operands_gradients: Vec<Rc<OptionalTensor<D>>>,
    gradient: Rc<OptionalTensor<D>>,
    axis: Axis,
}

impl<D> MultiConcatenateBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operands_gradients: Vec<Rc<OptionalTensor<D>>>,
        gradient: Rc<OptionalTensor<D>>,
        axis: usize,
    ) -> Self {
        Self {
            operands_gradients,
            gradient,
            axis: Axis(axis),
        }
    }
}

impl<D> Backward for MultiConcatenateBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let gradient = self.gradient.content();
        let mut offset = 0;
        self.operands_gradients
            .iter()
            .map(|operand| operand.content_mut())
            .for_each(|mut operand_gradient| {
                let axis_len = operand_gradient.len_of(self.axis);

                *operand_gradient +=
                    &gradient.slice_axis(self.axis, Slice::from(offset..axis_len + offset));
                offset += axis_len;
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
