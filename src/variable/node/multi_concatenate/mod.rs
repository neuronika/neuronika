use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{Axis, Dimension, Slice};
use std::rc::Rc;

pub struct MultiConcatenate<D>
where
    D: Dimension,
{
    operands_data: Vec<SharedTensor<D>>,
    data: SharedTensor<D>,
    axis: Axis,
}

impl<D> MultiConcatenate<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operands_data: Vec<SharedTensor<D>>,
        data: SharedTensor<D>,
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

pub struct MultiConcatenateBackward<D>
where
    D: Dimension,
{
    operands_gradients: Vec<Rc<SwitchableTensor<D>>>,
    gradient: Rc<SwitchableTensor<D>>,
    axis: Axis,
}

impl<D> MultiConcatenateBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operands_gradients: Vec<Rc<SwitchableTensor<D>>>,
        gradient: Rc<SwitchableTensor<D>>,
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
        let gradient = self.gradient.array();
        let mut offset = 0;
        self.operands_gradients
            .iter()
            .map(|operand| operand.array_mut())
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
