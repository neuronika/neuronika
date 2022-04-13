use std::rc::Rc;

use ndarray::{Array, Axis, Dimension, RemoveAxis, Zip};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
};

pub(crate) struct Concatenate<D>
where
    D: Dimension + RemoveAxis,
{
    left: Shared<Array<f32, D>>,
    right: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
    axis: Axis,
}

impl<D> Concatenate<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        left: Shared<Array<f32, D>>,
        right: Shared<Array<f32, D>>,
        data: Shared<Array<f32, D>>,
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

pub(crate) struct ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<D>>,
    axis: Axis,
}

impl<D> ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<D>>,
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
        let mut operand_gradient = self.operand_gradient.borrow_mut();
        let gradient = self.gradient.borrow();
        let (operand_gradient_slice, _) = gradient
            .view()
            .split_at(self.axis, operand_gradient.len_of(self.axis));

        *operand_gradient += &operand_gradient_slice;
    }
}

pub(crate) struct ConcatenateBackwardRight<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<D>>,
    axis: Axis,
    offset: usize,
}

impl<D> ConcatenateBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<D>>,
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
        let gradient = self.gradient.borrow();
        let (_, operand_gradient_slice) = gradient.view().split_at(self.axis, self.offset);

        *self.operand_gradient.borrow_mut() += &operand_gradient_slice;
    }
}

pub(crate) struct ConcatenateBackward<D>
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
    pub(crate) fn new(
        left: ConcatenateBackwardLeft<D>,
        right: ConcatenateBackwardRight<D>,
    ) -> Self {
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

#[cfg(test)]
mod test;
