use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{Axis, Dimension, RemoveAxis, Zip};
use std::rc::Rc;

pub struct Concatenate<D>
where
    D: Dimension + RemoveAxis,
{
    left: SharedTensor<D>,
    right: SharedTensor<D>,
    data: SharedTensor<D>,
    axis: Axis,
}

impl<D> Concatenate<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        left: SharedTensor<D>,
        right: SharedTensor<D>,
        data: SharedTensor<D>,
        axis: usize,
    ) -> Self {
        Self {
            left,
            right,
            data,
            axis: Axis(axis),
        }
    }
}

impl<D> Forward for Concatenate<D>
where
    D: Dimension + RemoveAxis,
{
    fn forward(&self) {
        let lhs_data = self.left.borrow();
        let mut data = self.data.borrow_mut();
        let (mut lhs_portion, mut rhs_portion) = data
            .view_mut()
            .split_at(self.axis, lhs_data.len_of(self.axis));

        Zip::from(&mut lhs_portion)
            .and(&*lhs_data)
            .for_each(|fused_el, &single_el| *fused_el = single_el);

        Zip::from(&mut rhs_portion)
            .and(&*self.right.borrow())
            .for_each(|fused_el, &single_el| *fused_el = single_el);
    }
}

pub struct ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<D>>,
    axis: Axis,
}

impl<D> ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<D>>,
        axis: usize,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            axis: Axis(axis),
        }
    }
}

impl<D> Backward for ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let mut operand_gradient = self.operand_gradient.array_mut();
        let gradient = self.gradient.array();
        let (operand_gradient_slice, _) = gradient
            .view()
            .split_at(self.axis, operand_gradient.len_of(self.axis));

        *operand_gradient += &operand_gradient_slice;
    }
}

pub struct ConcatenateBackwardRight<D>
where
    D: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<D>>,
    axis: Axis,
    offset: usize,
}

impl<D> ConcatenateBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<D>>,
        axis: usize,
        offset: usize,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            axis: Axis(axis),
            offset,
        }
    }
}

impl<D> Backward for ConcatenateBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient.array();
        let (_, operand_gradient_slice) = gradient.view().split_at(self.axis, self.offset);

        *self.operand_gradient.array_mut() += &operand_gradient_slice;
    }
}

pub struct ConcatenateBackward<D>
where
    D: Dimension + RemoveAxis,
{
    left: ConcatenateBackwardLeft<D>,
    right: ConcatenateBackwardRight<D>,
}

impl<D> ConcatenateBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(left: ConcatenateBackwardLeft<D>, right: ConcatenateBackwardRight<D>) -> Self {
        Self { left, right }
    }
}

impl<D> Backward for ConcatenateBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        self.left.backward();
        self.right.backward();
    }
}

// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // #[cfg(test)]
// // mod test;
