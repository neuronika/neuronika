use super::*;

use ndarray::{Axis, Slice, Zip};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiConcatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiConcatenate<D: Dimension + 'static> {
    operands: Vec<Rc<dyn Data<Dim = D>>>,
    axis: usize,
    data: RefCell<Tensor<D>>,
    computed: Cell<bool>,
}

impl<D: Dimension + 'static> MultiConcatenate<D> {
    pub(crate) fn new(operands: Vec<Rc<dyn Data<Dim = D>>>, axis: usize, shape: D) -> Self {
        let (data, computed) = (RefCell::new(Tensor::zeros(shape)), Cell::new(false));

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

        for operand in &self.operands {
            let operand_data = operand.data();
            let axis_len = operand_data.len_of(Axis(axis));
            let slice = Slice::from(offset..axis_len + offset);

            let view_mut = data.slice_axis_mut(Axis(axis), slice);
            Zip::from(view_mut)
                .and(&*operand_data)
                .for_each(|view_el, op_data_el| *view_el = *op_data_el);
            offset += axis_len;
        }
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
        for operand in &self.operands {
            let mut operand_grad = operand.gradient_mut();
            let axis_len = operand_grad.len_of(Axis(axis));
            let slice = Slice::from(offset..axis_len + offset);
            let grad_view = grad.as_ref().unwrap().slice_axis(Axis(axis), slice);
            let zip = Zip::from(&mut *operand_grad).and(grad_view);
            if operand.can_overwrite() {
                zip.for_each(|op_grad_el, grad_el| *op_grad_el = *grad_el);
                operand.set_overwrite(false);
            } else {
                zip.for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el);
            }
            offset += axis_len;
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}
