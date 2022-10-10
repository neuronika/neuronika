use std::{cell::Cell, rc::Rc};

use ndarray::{Array, Dimension, Zip};

use neuronika_variable::VarDiff;

use super::{IntoParam, Optimize, Optimizer, OptimizerStatus, Penalty};

/// Adam optimizer.
///
/// It has been proposed in
/// [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
pub struct Adam<T>
where
    T: Penalty,
{
    lr: Cell<f32>,
    penalty: T,
    beta1: Cell<f32>,
    beta2: Cell<f32>,
    eps: Cell<f32>,
}

impl<T> OptimizerStatus for Adam<T>
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

impl<T> Adam<T>
where
    T: Penalty,
{
    /// Creates a new Adam optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - learning rate.
    ///
    /// * `beta1` - coefficient for computing running average of the gradient. A good default is 0.9.
    ///
    /// * `beta2` - coefficient for computing running average of the squared gradient. A good
    /// default is 0.999.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-8*.
    pub fn new(lr: f32, beta1: f32, beta2: f32, penalty: T, eps: f32) -> Optimizer<Self> {
        let lr = Cell::new(lr);
        let beta1 = Cell::new(beta1);
        let beta2 = Cell::new(beta2);
        let eps = Cell::new(eps);

        let adam = Self {
            lr,
            penalty,
            beta1,
            beta2,
            eps,
        };

        Optimizer::new(adam)
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f32 {
        OptimizerStatus::get_lr(self)
    }

    /// Sets `lr` as the  new value for the learning rate.
    pub fn set_lr(&self, lr: f32) {
        OptimizerStatus::set_lr(self, lr);
    }

    /// Return the current values for the first exponential decay rate.
    pub fn get_beta1(&self) -> f32 {
        self.beta1.get()
    }

    /// Return the current values for the second exponential decay rate.
    pub fn get_beta2(&self) -> f32 {
        self.beta2.get()
    }

    /// Sets a new value for the first exponential decay rate.
    pub fn set_beta1(&self, beta1: f32) {
        self.beta1.set(beta1)
    }

    /// Sets a new value for the second exponential decay rate.
    pub fn set_beta2(&self, beta2: f32) {
        self.beta2.set(beta2)
    }

    /// Returns the current epsilon constant.
    pub fn get_eps(&self) -> f32 {
        self.eps.get()
    }

    /// Sets a new value for the epsilon constant.
    pub fn set_eps(&self, eps: f32) {
        self.eps.set(eps)
    }
}

/// A Parameter used by the Adam optimizer.
pub struct AdamParam<D, T>
where
    D: 'static + Dimension,
    T: Penalty,
{
    variable: VarDiff<D>,
    step: usize,
    exp_avg: Array<f32, D>,
    exp_avg_sq: Array<f32, D>,
    status: Rc<Adam<T>>,
}

impl<D, T> Optimize for AdamParam<D, T>
where
    T: Penalty,
    D: 'static + Dimension,
{
    fn optimize(&mut self) {
        self.step += 1;

        let beta1 = self.status.beta1.get();
        let beta2 = self.status.beta2.get();
        let lr = self.status.lr.get();
        let eps = self.status.eps.get();
        let penalty = self.status.penalty;

        let bias_correction1 = 1.0 - beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.step as i32);

        let mut data = self.variable.data_mut();
        let mut grad = self.variable.grad_mut();

        Zip::from(&mut *grad)
            .and(&*data)
            .for_each(|grad_el, data_el| *grad_el += penalty.penalize(data_el));

        Zip::from(&mut self.exp_avg)
            .and(&*grad)
            .for_each(|exp_avg_el, grad_el| {
                *exp_avg_el = *exp_avg_el * beta1 + grad_el * (1.0 - beta1)
            });

        Zip::from(&mut self.exp_avg_sq)
            .and(&*grad)
            .for_each(|exp_avg_sq_el, grad_el| {
                *exp_avg_sq_el = *exp_avg_sq_el * beta2 + grad_el * grad_el * (1.0 - beta2)
            });

        Zip::from(&mut *data)
            .and(&self.exp_avg)
            .and(&self.exp_avg_sq)
            .for_each(|data_el, exp_avg_el, exp_avg_sq_el| {
                *data_el -= exp_avg_el / ((exp_avg_sq_el.sqrt() / bias_correction2.sqrt()) + eps)
                    * (lr / bias_correction1)
            })
    }

    fn zero_grad(&mut self) {
        self.variable.zero_grad()
    }
}

impl<T, D> IntoParam<Adam<T>> for VarDiff<D>
where
    T: 'static + Penalty,
    D: 'static + Dimension,
{
    type Param = AdamParam<D, T>;

    fn into_param(self, status: Rc<Adam<T>>) -> Self::Param {
        let variable = self;
        let step = 0;
        let dim = variable.data().raw_dim();
        let exp_avg = Array::zeros(dim);
        let exp_avg_sq = exp_avg.clone();

        Self::Param {
            variable,
            step,
            exp_avg,
            exp_avg_sq,
            status,
        }
    }
}

#[cfg(test)]
mod test;
