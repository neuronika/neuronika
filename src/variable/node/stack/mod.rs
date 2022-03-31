use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{Axis, Dimension, RemoveAxis, Zip};
use std::rc::Rc;

pub struct Stack<D>
where
    D: Dimension + RemoveAxis,
{
    left: SharedTensor<D>,
    right: SharedTensor<D>,
    data: SharedTensor<D::Larger>,
    axis: Axis,
}

impl<D> Stack<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        left: SharedTensor<D>,
        right: SharedTensor<D>,
        data: SharedTensor<D::Larger>,
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

impl<D> Forward for Stack<D>
where
    D: Dimension + RemoveAxis,
{
    fn forward(&self) {
        let lhs_data = self.left.borrow();
        let rhs_data = self.right.borrow();
        let mut data = self.data.borrow_mut();
        let mut subview_iter = data.axis_iter_mut(self.axis);
        let mut subview_left = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        Zip::from(&mut subview_left)
            .and(&*lhs_data)
            .for_each(|fused_el, &single_el| *fused_el = single_el);

        let mut subview_right = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        Zip::from(&mut subview_right)
            .and(&*rhs_data)
            .for_each(|fused_el, &single_el| *fused_el = single_el);
    }
}

pub struct StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<D::Larger>>,
    axis: Axis,
}

impl<D> StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<D::Larger>>,
        axis: usize,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            axis: Axis(axis),
        }
    }
}

impl<D> Backward for StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient.array();
        let operand_gradient_slice = gradient
            .axis_iter(self.axis)
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        *self.operand_gradient.array_mut() += &operand_gradient_slice;
    }
}

pub struct StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<D::Larger>>,
    axis: Axis,
}

impl<D> StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<D::Larger>>,
        axis: usize,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            axis: Axis(axis),
        }
    }
}

impl<D> Backward for StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient.array();
        let operand_gradient_slice = gradient
            .axis_iter(self.axis)
            .nth(1)
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        *self.operand_gradient.array_mut() += &operand_gradient_slice;
    }
}

pub struct StackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    left: StackBackwardLeft<D>,
    right: StackBackwardRight<D>,
}

impl<D> StackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(left: StackBackwardLeft<D>, right: StackBackwardRight<D>) -> Self {
        Self { left, right }
    }
}

impl<D> Backward for StackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        self.left.backward();
        self.right.backward();
    }
}

// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // #[cfg(test)]
// // mod test;
