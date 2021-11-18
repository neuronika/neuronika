#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Tensor,
};
use ndarray::{Ix1, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sum ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Sum<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<Ix1>>,
    computed: Cell<bool>,
}

impl<T: Data> Sum<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(1));

        Self {
            operand,
            data,
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Sum<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        self.data.borrow_mut()[0] = self.operand.data().sum();
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Sum<T> {
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<T: Data> Debug for Sum<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sum")
            .field("data", &self.data.borrow())
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<T: Data> Display for Sum<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SumBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct SumBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<Ix1>>>,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T: Gradient + Overwrite> SumBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        Self {
            operand,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            overwrite: Cell::new(true),
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for SumBackward<T> {
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for SumBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for SumBackward<T> {
    fn backward(&self) {
        let mut op_grad = self.operand.gradient_mut();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and_broadcast(&*grad);
        if self.operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el| *op_grad_el = *grad_el);
            self.operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
    }
}

impl<T> Debug for SumBackward<T>
where
    T: Gradient + Overwrite,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SumBackward")
            .field("gradient", &self.gradient.borrow())
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T> Display for SumBackward<T>
where
    T: Gradient + Overwrite,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
