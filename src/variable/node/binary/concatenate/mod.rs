use super::{Backward, Forward, OptionalTensor, Tensor};
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
    axis: Axis,
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
            axis: Axis(axis),
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

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    gradient: Rc<OptionalTensor<D>>,
    axis: Axis,
}

impl<D> ConcatenateBackwardLeft<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<D>>,
        gradient: Rc<OptionalTensor<D>>,
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
        let mut operand_gradient = self.operand_gradient.content_mut();
        let gradient = self.gradient.content();
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
    operand_gradient: Rc<OptionalTensor<D>>,
    gradient: Rc<OptionalTensor<D>>,
    axis: Axis,
    offset: usize,
}

impl<D> ConcatenateBackwardRight<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<D>>,
        gradient: Rc<OptionalTensor<D>>,
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
        let gradient = self.gradient.content();
        let (_, operand_gradient_slice) = gradient.view().split_at(self.axis, self.offset);

        *self.operand_gradient.content_mut() += &operand_gradient_slice;
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
