use std::{cell::Cell, rc::Rc};

use ndarray::{Array, Dimension, Zip};

use crate::variable::VarDiff;

use super::{IntoParam, Optimize, Optimizer, OptimizerStatus, Penalty};

/// Adagrad optimizer.
///
/// The algorithm has been proposed in [this paper](http://jmlr.org/papers/v12/duchi11a.html).
pub struct Adagrad<T>
where
    T: Penalty,
{
    lr: Cell<f32>,
    lr_decay: Cell<f32>,
    penalty: T,
    eps: Cell<f32>,
}

impl<T> OptimizerStatus for Adagrad<T>
where
    T: Penalty,
{
    fn get_lr(&self) -> f32 {
        self.lr.get()
    }

    fn set_lr(&self, lr: f32) {
        self.lr.set(lr)
    }
}

impl<T> Adagrad<T>
where
    T: Penalty,
{
    /// Creates a new Adagrad optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - learning rate.
    ///
    /// * `lr_decay` - the learning rate decay.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-10*.
    pub fn new(lr: f32, lr_decay: f32, penalty: T, eps: f32) -> Optimizer<Self> {
        let lr = Cell::new(lr);
        let lr_decay = Cell::new(lr_decay);
        let eps = Cell::new(eps);

        let status = Self {
            lr,
            lr_decay,
            penalty,
            eps,
        };

        Optimizer::new(status)
    }

    /// Returns the current learning rate.
    pub fn get_lr(&self) -> f32 {
        OptimizerStatus::get_lr(self)
    }

    /// Sets a new value for the learning rate.
    pub fn set_lr(&self, lr: f32) {
        OptimizerStatus::set_lr(self, lr);
    }

    /// Return the current learning rate decay parameter.
    pub fn get_lr_decay(&self) -> f32 {
        self.lr_decay.get()
    }

    /// Sets `lr_decay` as the  new value for the learning rate decay parameter.
    pub fn set_lr_decay(&self, lr_decay: f32) {
        self.lr_decay.set(lr_decay)
    }

    /// Return the current epsilon constant.
    pub fn get_eps(&self) -> f32 {
        self.eps.get()
    }

    /// Sets a  new value for the epsilon constant.
    pub fn set_eps(&self, eps: f32) {
        self.eps.set(eps)
    }
}

/// A parameter used by the Adagrad optimizer.
pub struct AdagradParam<D, T>
where
    D: Dimension,
    T: Penalty,
{
    variable: VarDiff<D>,
    step: usize,
    grad_sq: Array<f32, D>,
    status: Rc<Adagrad<T>>,
}

impl<D, T> Optimize for AdagradParam<D, T>
where
    D: Dimension,
    T: Penalty,
{
    fn optimize(&mut self) {
        self.step += 1;

        let lr = self.status.get_lr();
        let lr_decay = self.status.get_lr_decay();
        let eps = self.status.get_eps();
        let penalty = self.status.penalty;

        let clr = lr / (1.0 + (self.step - 1) as f32 * lr_decay);

        let mut data = self.variable.data_mut();
        let mut grad = self.variable.grad_mut();

        Zip::from(&mut *grad)
            .and(&*data)
            .for_each(|grad_el, data_el| *grad_el += penalty.penalize(data_el));

        Zip::from(&mut self.grad_sq)
            .and(&*grad)
            .for_each(|grad_sq_el, grad_el| *grad_sq_el += grad_el * grad_el);

        Zip::from(&mut *data)
            .and(&*grad)
            .and(&self.grad_sq)
            .for_each(|data_el, grad_el, grad_sq_el| {
                *data_el -= grad_el / (grad_sq_el.sqrt() + eps) * clr
            });
    }

    fn zero_grad(&mut self) {
        self.variable.zero_grad()
    }
}

impl<D, T> IntoParam<Adagrad<T>> for VarDiff<D>
where
    D: 'static + Dimension,
    T: 'static + Penalty,
{
    type Param = AdagradParam<D, T>;

    fn into_param(self, status: Rc<Adagrad<T>>) -> Self::Param {
        let variable = self;
        let step = 0;
        let grad_sq = Array::zeros(variable.grad().raw_dim());

        Self::Param {
            variable,
            step,
            grad_sq,
            status,
        }
    }
}

#[cfg(test)]
mod test;
