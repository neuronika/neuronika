use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Tensor,
};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::Zip;

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

pub struct Negation<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    computed: Cell<bool>,
}

impl<T: Data> Negation<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let data = Tensor::zeros(operand.data().raw_dim());

        Self {
            operand,
            data: RefCell::new(data),
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Negation<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand.data())
            .for_each(|v, o| *v = -o);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Negation<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

pub struct NegationBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T: Gradient + Overwrite> NegationBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            operand,
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for NegationBackward<T> {
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for NegationBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for NegationBackward<T> {
    fn backward(&self) {
        let mut op_grad = self.operand.gradient_mut();
        let grad = self.gradient();
        let zip = Zip::from(&mut *op_grad).and(&*grad);

        if self.operand.can_overwrite() {
            self.operand.set_overwrite(false);
            zip.for_each(|op_grad_el, grad_el| *op_grad_el = -grad_el);
        } else {
            zip.for_each(|op_grad_el, grad_el| *op_grad_el += -grad_el)
        }
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
