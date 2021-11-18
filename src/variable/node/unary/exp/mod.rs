#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Tensor,
};
use ndarray::Zip;
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exp ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Exp<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    computed: Cell<bool>,
}

impl<T: Data> Exp<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(operand.data().raw_dim()));

        Self {
            operand,
            data,
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Exp<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand.data())
            .for_each(|v, o| *v = o.exp());
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Exp<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<T: Data> Debug for Exp<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exp")
            .field("data", &self.data.borrow())
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<T: Data> Display for Exp<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ExpBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, data_el| *op_grad_el = *grad_el * data_el);
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, data_el| *op_grad_el += *grad_el * data_el);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

impl<T, U> Debug for ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpBackward")
            .field("gradient", &self.gradient.borrow())
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T, U> Display for ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
