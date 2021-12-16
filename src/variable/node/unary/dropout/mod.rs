#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Eval, Forward, Gradient, Overwrite, Tensor,
};
use ndarray::Zip;
use rand::thread_rng;
use rand_distr::{Bernoulli, Distribution};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Dropout<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    noise: RefCell<Tensor<T::Dim>>,
    distr: Bernoulli,
    p: f64,
    computed: Cell<bool>,
    train: Rc<Cell<bool>>,
}

impl<T: Data> Dropout<T> {
    pub fn new(operand: Rc<T>, p: f64, status: Rc<Cell<bool>>) -> Self {
        if !(0. ..=1.).contains(&p) {
            panic!(
                "error: dropout probability has to be between 0 and 1, but got {}.",
                p
            );
        }

        let (data, noise) = (
            RefCell::new(Tensor::zeros(operand.data().raw_dim())),
            RefCell::new(Tensor::zeros(operand.data().raw_dim())),
        );
        let distr = Bernoulli::new(1. - p).unwrap();

        Self {
            operand,
            data,
            noise,
            distr,
            p,
            computed: Cell::new(false),
            train: status,
        }
    }

    pub(crate) fn noise(&self) -> Ref<Tensor<T::Dim>> {
        self.noise.borrow()
    }

    pub(crate) fn status(&self) -> Rc<Cell<bool>> {
        self.train.clone()
    }
}

impl<T: Data> Forward for Dropout<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        if self.train.get() {
            let mut thread_rng = thread_rng();
            let (mut noise, distr, p) = (self.noise.borrow_mut(), &self.distr, &self.p);
            if (*p - 1.).abs() <= f64::EPSILON {
                Zip::from(&mut *self.data.borrow_mut()).for_each(|data_el| *data_el = 0.0);
            } else if *p <= f64::EPSILON {
                Zip::from(&mut *self.data.borrow_mut())
                    .and(&*self.operand.data())
                    .for_each(|data_el, operand_data_el| *data_el = *operand_data_el);
            } else {
                Zip::from(&mut *noise)
                    .for_each(|noise_el| *noise_el = distr.sample(&mut thread_rng) as i32 as f32);
                Zip::from(&mut *self.data.borrow_mut())
                    .and(&*self.operand.data())
                    .and(&*noise)
                    .for_each(|data_el, operand_data_el, noise_el| {
                        *data_el = (operand_data_el * noise_el) / (1. - *p as f32)
                    });
            }
        } else {
            self.data.borrow_mut().assign(&*self.operand.data());
        }
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Dropout<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<T: Data> Eval for Dropout<T> {
    fn train(&self) {
        self.train.set(true);
    }

    fn eval(&self) {
        self.train.set(false);
    }
}

impl<T: Data> Debug for Dropout<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dropout")
            .field("data", &self.data.borrow())
            .field("p", &self.p)
            .field("noise", &self.noise.borrow())
            .field("train", &self.train.get())
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<T: Data> Display for Dropout<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DropoutBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<Dropout<U>>,
    p: f64,
    train: Rc<Cell<bool>>,
}

impl<T, U> DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(
        diff_operand: Rc<T>,
        no_diff_operand: Rc<Dropout<U>>,
        p: f64,
        forward_status: Rc<Cell<bool>>,
    ) -> DropoutBackward<T, U> {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            p,
            train: forward_status,
        }
    }
}

impl<T, U> Gradient for DropoutBackward<T, U>
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

impl<T, U> Overwrite for DropoutBackward<T, U>
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

impl<T, U> Backward for DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        if self.train.get() {
            let mut op_grad = self.diff_operand.gradient_mut();
            let grad = self.gradient();
            let p = &self.p;
            if (*p - 1.).abs() <= f64::EPSILON {
                if self.diff_operand.can_overwrite() {
                    Zip::from(&mut *op_grad).for_each(|op_grad_el| *op_grad_el = 0.);
                    self.diff_operand.set_overwrite(false);
                }
            } else if *p <= f64::EPSILON {
                let zip = Zip::from(&mut *op_grad).and(&*grad);
                if self.diff_operand.can_overwrite() {
                    zip.for_each(|op_grad_el, grad_el| *op_grad_el = *grad_el);
                    self.diff_operand.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el);
                }
            } else {
                let noise = self.no_diff_operand.noise();
                let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*noise);
                if self.diff_operand.can_overwrite() {
                    zip.for_each(|op_grad_el, grad_el, noise_el| *op_grad_el = *grad_el * noise_el);
                    self.diff_operand.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad_el, grad_el, noise_el| {
                        *op_grad_el += *grad_el * noise_el
                    });
                }
            }
        } else {
            self.diff_operand.gradient_mut().assign(&*self.gradient());
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

impl<T, U> Debug for DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DropoutBackward")
            .field("gradient", &self.gradient.borrow())
            .field("p", &self.p)
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T, U> Display for DropoutBackward<T, U>
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
