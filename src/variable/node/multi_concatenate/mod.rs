use std::rc::Rc;

use ndarray::{Array, Axis, Dimension, Slice};

use crate::variable::{gradient::Gradient, utils::Shared};

use super::{Backward, Forward};

pub(crate) struct MultiConcatenate<D>
where
    D: Dimension,
{
    operands_data: Vec<Shared<Array<f32, D>>>,
    data: Shared<Array<f32, D>>,
    axis: Axis,
}

impl<D> MultiConcatenate<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operands_data: Vec<Shared<Array<f32, D>>>,
        data: Shared<Array<f32, D>>,
        axis: usize,
    ) -> Self {
        Self {
            operands_data,
            data,
            axis: Axis(axis),
        }
    }
}

impl<D> Forward for MultiConcatenate<D>
where
    D: Dimension,
{
    fn forward(&self) {
        let (mut offset, mut data) = (0, self.data.borrow_mut());

        self.operands_data.iter().for_each(|operand| {
            let operand_data = operand.borrow();
            let axis_len = operand_data.len_of(self.axis);
            let slice = Slice::from(offset..axis_len + offset);

            data.slice_axis_mut(self.axis, slice).assign(&operand_data);
            offset += axis_len;
        });
    }
}

pub(crate) struct MultiConcatenateBackward<D>
where
    D: Dimension,
{
    operands_gradients: Vec<Rc<Gradient<D>>>,
    gradient: Rc<Gradient<D>>,
    axis: Axis,
}

impl<D> MultiConcatenateBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operands_gradients: Vec<Rc<Gradient<D>>>,
        gradient: Rc<Gradient<D>>,
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
        let gradient = self.gradient.borrow();
        let mut offset = 0;
        self.operands_gradients
            .iter()
            .map(|operand| operand.borrow_mut())
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
