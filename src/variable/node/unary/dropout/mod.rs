use super::{Backward, Forward, OptionalTensor, Tensor};
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

        Self {
            operand_data,
            data,
            noise,
            distr: Bernoulli::new(1. - p).unwrap(),
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
        if !self.status.get() || self.p <= f64::EPSILON {
            self.data.borrow_mut().assign(&*self.operand_data.borrow());
            return;
        }

        if 1. - self.p <= f64::EPSILON {
            Zip::from(&mut *self.data.borrow_mut()).for_each(|data_el| *data_el = 0.);
            return;
        }

        let mut noise = self.noise.borrow_mut();
        // This zip must be kept separated from the following one, because
        // in that way we won't be able to parallelize the execution (`thread_rng`
        // does not implement `Sync` and `Send`).
        Zip::from(&mut *noise)
            .for_each(|noise_el| *noise_el = self.distr.sample(&mut thread_rng()) as i32 as f32);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .and(&*noise)
            .for_each(|data_el, operand_data_el, noise_el| {
                *data_el = (operand_data_el * noise_el) / (1. - self.p as f32)
            });
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
    operand_gradient: Rc<OptionalTensor<D>>,
    gradient: Rc<OptionalTensor<D>>,
    noise: Rc<RefCell<Tensor<D>>>,
    p: f64,
    status: Rc<Cell<bool>>,
}

impl<D> DropoutBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<D>>,
        gradient: Rc<OptionalTensor<D>>,
        noise: Rc<RefCell<Tensor<D>>>,
        p: f64,
        status: Rc<Cell<bool>>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
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
        if !self.status.get() || self.p <= f64::EPSILON {
            *self.operand_gradient.content_mut() += &*self.gradient.content();
            return;
        }

        Zip::from(&mut *self.operand_gradient.content_mut())
            .and(&*self.gradient.content())
            .and(&*self.noise.borrow())
            .for_each(|op_grad_el, grad_el, noise_el| *op_grad_el += *grad_el * noise_el);
    }
}

// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
