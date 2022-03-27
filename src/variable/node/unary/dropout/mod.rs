#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{Dimension, Zip};
use rand::thread_rng;
use rand_distr::{Bernoulli, Distribution};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Dropout<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    noise: Rc<RefCell<Tensor<D>>>,
    distr: Bernoulli,
    p: f64,
    computed: Cell<bool>,
    status: Rc<Cell<bool>>,
}

impl<D> Dropout<D>
where
    D: Dimension,
{
    pub fn new(
        operand_data: Rc<RefCell<Tensor<D>>>,
        data: Rc<RefCell<Tensor<D>>>,
        p: f64,
        noise: Rc<RefCell<Tensor<D>>>,
        status: Rc<Cell<bool>>,
    ) -> Self {
        if !(0. ..=1.).contains(&p) {
            panic!(
                "Dropout probability has to be between 0 and 1, but got {}.",
                p
            );
        }

        let distr = Bernoulli::new(1. - p).unwrap();

        Self {
            operand_data,
            data,
            noise,
            distr,
            p,
            computed: Cell::default(),
            status,
        }
    }
}

impl<D> Forward for Dropout<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        if self.status.get() {
            let mut thread_rng = thread_rng();
            let (mut noise, distr, p) = (self.noise.borrow_mut(), &self.distr, &self.p);
            if 1.0 - self.p <= f64::EPSILON {
                Zip::from(&mut *self.data.borrow_mut()).for_each(|data_el| *data_el = 0.0);
            } else if *p <= f64::EPSILON {
                Zip::from(&mut *self.data.borrow_mut())
                    .and(&*self.operand_data.borrow())
                    .for_each(|data_el, operand_data_el| *data_el = *operand_data_el);
            } else {
                Zip::from(&mut *noise)
                    .for_each(|noise_el| *noise_el = distr.sample(&mut thread_rng) as i32 as f32);
                Zip::from(&mut *self.data.borrow_mut())
                    .and(&*self.operand_data.borrow())
                    .and(&*noise)
                    .for_each(|data_el, operand_data_el, noise_el| {
                        *data_el = (operand_data_el * noise_el) / (1. - *p as f32)
                    });
            }
        } else {
            self.data.borrow_mut().assign(&*self.operand_data.borrow());
        }
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct DropoutBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D>>>>,
    shape: D,
    noise: Rc<RefCell<Tensor<D>>>,
    p: f64,
    status: Rc<Cell<bool>>,
}

impl<D> DropoutBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D>>>>,
        shape: D,
        noise: Rc<RefCell<Tensor<D>>>,
        p: f64,
        status: Rc<Cell<bool>>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            shape,
            noise,
            p,
            status,
        }
    }
}

impl<D> Backward for DropoutBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        if self.status.get() {
            if self.p <= f64::EPSILON {
                Zip::from(&mut *operand_gradient)
                    .and(&*gradient)
                    .for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el);
            } else if 1.0 - self.p > f64::EPSILON {
                let noise = self.noise.borrow();
                Zip::from(&mut *operand_gradient)
                    .and(&*gradient)
                    .and(&*noise)
                    .for_each(|op_grad_el, grad_el, noise_el| *op_grad_el += *grad_el * noise_el);
            }
        } else {
            *operand_gradient += &*gradient;
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
