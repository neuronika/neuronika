use std::rc::Rc;

use ndarray::{Array, Axis, Dimension, RemoveAxis, Zip};

use crate::variable::{gradient::Gradient, utils::Shared};

use super::{Backward, Forward};

pub(crate) struct Stack<D>
where
    D: Dimension + RemoveAxis,
{
    left: Shared<Array<f32, D>>,
    right: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D::Larger>>,
    axis: Axis,
}

impl<D> Stack<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        left: Shared<Array<f32, D>>,
        right: Shared<Array<f32, D>>,
        data: Shared<Array<f32, D::Larger>>,
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

pub(crate) struct StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<D::Larger>>,
    axis: Axis,
}

impl<D> StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<D::Larger>>,
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
        let gradient = self.gradient.borrow();
        let operand_gradient_slice = gradient
            .axis_iter(self.axis)
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        *self.operand_gradient.borrow_mut() += &operand_gradient_slice;
    }
}

pub(crate) struct StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<D::Larger>>,
    axis: Axis,
}

impl<D> StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<D::Larger>>,
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
        let gradient = self.gradient.borrow();
        let operand_gradient_slice = gradient
            .axis_iter(self.axis)
            .nth(1)
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        *self.operand_gradient.borrow_mut() += &operand_gradient_slice;
    }
}

pub(crate) struct StackBackward<D>
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
    pub(crate) fn new(left: StackBackwardLeft<D>, right: StackBackwardRight<D>) -> Self {
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
