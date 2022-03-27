#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{Axis, Dimension, RemoveAxis, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Concatenate<D>
where
    D: Dimension + RemoveAxis,
{
    left: Rc<RefCell<Tensor<D>>>,
    right: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    axis: usize,
    computed: Cell<bool>,
}

impl<D> Concatenate<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        left: Rc<RefCell<Tensor<D>>>,
        right: Rc<RefCell<Tensor<D>>>,
        data: Rc<RefCell<Tensor<D>>>,
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

impl<D> Forward for Concatenate<D>
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

        let (mut lhs_portion, mut rhs_portion) = data
            .view_mut()
            .split_at(Axis(axis), lhs_data.len_of(Axis(axis)));

        Zip::from(&*lhs_data)
            .and(&mut lhs_portion)
            .for_each(|single_el, fused_el| *fused_el = *single_el);

        Zip::from(&*rhs_data)
            .and(&mut rhs_portion)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct ConcatenateBackward<D>
where
    D: Dimension + RemoveAxis,
{
    left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    right_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D>>>>,
    shape: D,
    axis: usize,
}

impl<D> ConcatenateBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        left_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        right_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D>>>>,
        shape: D,
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

impl<D> Backward for ConcatenateBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let mut left_gradient = expect_tensor_mut(&self.left_gradient);
        let mut right_gradient = expect_tensor_mut(&self.right_gradient);
        let gradient = expect_tensor(&self.gradient);

        let (left_gradient_slice, right_gradient_slice) = gradient
            .view()
            .split_at(Axis(self.axis), left_gradient.len_of(Axis(self.axis)));

        *left_gradient += &left_gradient_slice;
        *right_gradient += &right_gradient_slice;
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D>>>>,
    shape: D,
    axis: usize,
}

impl<D> ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D>>>>,
        shape: D,
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

impl<D> Backward for ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        let (operand_gradient_slice, _) = gradient
            .view()
            .split_at(Axis(self.axis), operand_gradient.len_of(Axis(self.axis)));

        *operand_gradient += &operand_gradient_slice;
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

pub struct ConcatenateBackwardRight<D>
where
    D: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D>>>>,
    shape: D,
    offset: usize,
    axis: usize,
}

impl<D> ConcatenateBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D>>>>,
        shape: D,
        offset: usize,
        axis: usize,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            shape,
            offset,
            axis,
        }
    }
}

impl<D> Backward for ConcatenateBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        let (_, operand_gradient_slice) = gradient.view().split_at(Axis(self.axis), self.offset);

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
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // #[cfg(test)]
// // mod test;
