use std::{cell::Cell, rc::Rc};

use ndarray::{Array, Dimension, Zip};

use rand::thread_rng;

use rand_distr::{Bernoulli, Distribution};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
};

pub(crate) struct Dropout<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
    noise: Shared<Array<f32, D>>,
    distr: Bernoulli,
    p: f64,
    status: Rc<Cell<bool>>,
}

impl<D> Dropout<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_data: Shared<Array<f32, D>>,
        data: Shared<Array<f32, D>>,
        p: f64,
        noise: Shared<Array<f32, D>>,
        status: Rc<Cell<bool>>,
    ) -> Self {
        if !(0. ..=1.).contains(&p) {
            panic!("Wrong probability received: {}.", p);
        }

        Self {
            operand_data,
            data,
            noise,
            distr: Bernoulli::new(1. - p).unwrap(),
            p,
            status,
        }
    }
}

impl<D> Forward for Dropout<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if !self.status.get() || self.p == 0.0 {
            self.data.borrow_mut().assign(&*self.operand_data.borrow());
            return;
        }

        if (1. - self.p) == 0.0 {
            Zip::from(&mut *self.data.borrow_mut()).for_each(|data_el| *data_el = 0.);
            return;
        }

        let mut noise = self.noise.borrow_mut();
        Zip::from(&mut *noise)
            .for_each(|noise_el| *noise_el = self.distr.sample(&mut thread_rng()) as i32 as f32);
        // Remember: keep these zips separate
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand_data.borrow())
            .and(&*noise)
            .for_each(|data_el, &operand_data_el, &noise_el| {
                *data_el = (operand_data_el * noise_el) / (1. - self.p as f32)
            });
    }
}

pub(crate) struct DropoutBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
    gradient: Rc<Gradient<Array<f32, D>, D>>,
    noise: Shared<Array<f32, D>>,
    p: f64,
    status: Rc<Cell<bool>>,
}

impl<D> DropoutBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
        gradient: Rc<Gradient<Array<f32, D>, D>>,
        p: f64,
        noise: Shared<Array<f32, D>>,
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
        if !self.status.get() || self.p == 0.0 {
            *self.operand_gradient.borrow_mut() += &*self.gradient.borrow();
            return;
        }

        Zip::from(&mut *self.operand_gradient.borrow_mut())
            .and(&*self.gradient.borrow())
            .and(&*self.noise.borrow())
            .for_each(|op_grad_el, &grad_el, &noise_el| *op_grad_el += grad_el * noise_el);
    }
}

#[cfg(test)]
mod test;
