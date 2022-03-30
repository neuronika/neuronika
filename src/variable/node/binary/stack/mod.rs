use super::{Backward, Forward, OptionalTensor, Tensor};
use ndarray::{Axis, Dimension, RemoveAxis, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Stack<D>
where
    D: Dimension + RemoveAxis,
{
    left: Rc<RefCell<Tensor<D>>>,
    right: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D::Larger>>>,
    axis: usize,
    computed: Cell<bool>,
}

impl<D> Stack<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        left: Rc<RefCell<Tensor<D>>>,
        right: Rc<RefCell<Tensor<D>>>,
        data: Rc<RefCell<Tensor<D::Larger>>>,
        axis: usize,
    ) -> Self {
        Self {
            left,
            right,
            data,
            axis,
            computed: Cell::default(),
        }
    }
}

impl<D> Forward for Stack<D>
where
    D: Dimension + RemoveAxis,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let lhs_data = self.left.borrow();
        let rhs_data = self.right.borrow();
        let mut data = self.data.borrow_mut();
        let axis = self.axis;
        let mut subview_iter = data.axis_iter_mut(Axis(axis));

        let mut subview_left = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        Zip::from(&*lhs_data)
            .and(&mut subview_left)
            .for_each(|single_el, fused_el| *fused_el = *single_el);

        let mut subview_right = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        Zip::from(&*rhs_data)
            .and(&mut subview_right)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    gradient: Rc<OptionalTensor<D::Larger>>,
    axis: Axis,
}

impl<D> StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<D>>,
        gradient: Rc<OptionalTensor<D::Larger>>,
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
        let gradient = self.gradient.content();
        let operand_gradient_slice = gradient
            .axis_iter(self.axis)
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        *self.operand_gradient.content_mut() += &operand_gradient_slice;
    }
}

pub struct StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    gradient: Rc<OptionalTensor<D::Larger>>,
    axis: Axis,
}

impl<D> StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<D>>,
        gradient: Rc<OptionalTensor<D::Larger>>,
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
        let gradient = self.gradient.content();
        let operand_gradient_slice = gradient
            .axis_iter(self.axis)
            .nth(1)
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        *self.operand_gradient.content_mut() += &operand_gradient_slice;
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
