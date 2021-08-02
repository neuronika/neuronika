use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient,
    GradientOverwrite, Overwrite, Tensor,
};

use ndarray::{Axis, Dimension, Slice, Zip};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiConcatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiConcatenate<D: Dimension + 'static> {
    operands: Vec<Rc<dyn Data<Dim = D>>>,
    axis: usize,
    data: RefCell<Tensor<D>>,
    computed: Cell<bool>,
}

impl<D: Dimension + 'static> MultiConcatenate<D> {
    pub(crate) fn new(operands: Vec<Rc<dyn Data<Dim = D>>>, axis: usize, data: Tensor<D>) -> Self {
        let (data, computed) = (RefCell::new(data), Cell::new(false));

        Self {
            operands,
            axis,
            data,
            computed,
        }
    }
}

impl<D: Dimension> Data for MultiConcatenate<D> {
    type Dim = D;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<D: Dimension> Forward for MultiConcatenate<D> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (axis, mut offset, mut data) = (self.axis, 0, self.data.borrow_mut());

        self.operands.iter().for_each(|operand| {
            let operand_data = operand.data();
            let axis_len = operand_data.len_of(Axis(axis));
            let slice = Slice::from(offset..axis_len + offset);

            let view_mut = data.slice_axis_mut(Axis(axis), slice);
            Zip::from(view_mut)
                .and(&*operand_data)
                .for_each(|view_el, op_data_el| *view_el = *op_data_el);
            offset += axis_len;
        });
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiConcatenateBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiConcatenateBackward<D: Dimension> {
    gradient: RefCell<Option<Tensor<D>>>,
    shape: D,
    overwrite: Cell<bool>,
    operands: Vec<Rc<dyn GradientOverwrite<D>>>,
    axis: usize,
}

impl<D: Dimension> MultiConcatenateBackward<D> {
    pub(crate) fn new(operands: Vec<Rc<dyn GradientOverwrite<D>>>, axis: usize, shape: D) -> Self {
        let gradient = RefCell::new(Some(Tensor::zeros(shape.clone())));
        let overwrite = Cell::new(true);

        Self {
            gradient,
            shape,
            overwrite,
            operands,
            axis,
        }
    }
}

impl<D: Dimension> Gradient for MultiConcatenateBackward<D> {
    type Dim = D;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<D: Dimension> Overwrite for MultiConcatenateBackward<D> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<D: Dimension> Backward for MultiConcatenateBackward<D> {
    fn backward(&self) {
        let (axis, grad, mut offset) = (self.axis, &self.gradient.borrow(), 0);

        self.operands.iter().for_each(|operand| {
            let axis_len = operand.gradient().len_of(Axis(axis));

            let grad_view = grad
                .as_ref()
                .unwrap()
                .slice_axis(Axis(axis), Slice::from(offset..axis_len + offset));

            push_gradient(operand.as_ref(), &grad_view);
            offset += axis_len;
        });
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

#[cfg(test)]
mod test {}
