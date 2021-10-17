use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient,
    GradientOverwrite, Overwrite, Tensor,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

use ndarray::{Axis, Dimension, RemoveAxis, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

pub struct MultiStack<D: Dimension + RemoveAxis + 'static> {
    operands: Vec<Rc<dyn Data<Dim = D>>>,
    axis: usize,
    data: RefCell<Tensor<D::Larger>>,
    computed: Cell<bool>,
}

impl<D: Dimension + RemoveAxis + 'static> MultiStack<D> {
    pub(crate) fn new(
        operands: Vec<Rc<dyn Data<Dim = D>>>,
        axis: usize,
        tensor: Tensor<D::Larger>,
    ) -> Self {
        let (data, computed) = (RefCell::new(tensor), Cell::new(false));

        Self {
            operands,
            axis,
            data,
            computed,
        }
    }
}

impl<D: Dimension + RemoveAxis> Data for MultiStack<D> {
    type Dim = D::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<D: Dimension + RemoveAxis> Forward for MultiStack<D> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (mut data, axis) = (self.data.borrow_mut(), self.axis);

        self.operands
            .iter()
            .zip(data.axis_iter_mut(Axis(axis)))
            .for_each(|(operand, axis_data)| {
                let operand_data = operand.data();
                Zip::from(&mut axis_data.into_dimensionality::<D>().unwrap())
                    .and(&*operand_data)
                    .for_each(|axis_data_el, operand_data_el| *axis_data_el = *operand_data_el)
            });
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct MultiStackBackward<D: Dimension + RemoveAxis> {
    gradient: RefCell<Option<Tensor<D::Larger>>>,
    shape: D::Larger,
    overwrite: Cell<bool>,
    operands: Vec<Rc<dyn GradientOverwrite<D>>>,
    axis: usize,
}

impl<D: Dimension + RemoveAxis> MultiStackBackward<D> {
    pub(crate) fn new(
        operands: Vec<Rc<dyn GradientOverwrite<D>>>,
        axis: usize,
        shape: D::Larger,
    ) -> Self {
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

impl<D: Dimension + RemoveAxis> Gradient for MultiStackBackward<D> {
    type Dim = D::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<D: Dimension + RemoveAxis> Overwrite for MultiStackBackward<D> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<D: Dimension + RemoveAxis> Backward for MultiStackBackward<D> {
    fn backward(&self) {
        let (axis, grad) = (self.axis, &self.gradient.borrow());

        self.operands
            .iter()
            .zip(grad.as_ref().unwrap().axis_iter(Axis(axis)))
            .for_each(|(operand, grad_view)| {
                push_gradient(operand.as_ref(), &grad_view);
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
mod test;
