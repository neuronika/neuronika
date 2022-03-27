#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
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

pub struct StackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    right_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
    shape: D::Larger,
    axis: usize,
}

impl<D> StackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        right_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
        shape: D::Larger,
        axis: usize,
    ) -> Self {
        Self {
            left_gradient,
            right_gradient,
            gradient,
            shape,
            axis,
        }
    }
}

impl<D> Backward for StackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let mut left_gradient = expect_tensor_mut(&self.left_gradient);
        let mut right_gradient = expect_tensor_mut(&self.right_gradient);
        let gradient = expect_tensor(&self.gradient);

        let mut subviews = gradient.axis_iter(Axis(self.axis));

        let left_gradient_slice = subviews.next().unwrap().into_dimensionality::<D>().unwrap();
        *left_gradient += &left_gradient_slice;

        let right_gradient_slice = subviews.next().unwrap().into_dimensionality::<D>().unwrap();
        *right_gradient += &right_gradient_slice;
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
    shape: D::Larger,
    axis: usize,
}

impl<D> StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
        shape: D::Larger,
        axis: usize,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            shape,
            axis,
        }
    }
}

impl<D> Backward for StackBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        let operand_gradient_slice = gradient
            .axis_iter(Axis(self.axis))
            .next()
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();
        *operand_gradient += &operand_gradient_slice;
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
    shape: D::Larger,
    axis: usize,
}

impl<D> StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
        shape: D::Larger,
        axis: usize,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            shape,
            axis,
        }
    }
}

impl<D> Backward for StackBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        let operand_gradient_slice = gradient
            .axis_iter(Axis(self.axis))
            .nth(1)
            .unwrap()
            .into_dimensionality::<D>()
            .unwrap();

        *operand_gradient += &operand_gradient_slice;
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // #[cfg(test)]
// // mod test;
