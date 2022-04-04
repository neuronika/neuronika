use std::{cell::Cell, rc::Rc};

use ndarray::{Array, Dimension, Zip};

use crate::variable::VarDiff;

use super::{IntoParam, Optimize, Optimizer, OptimizerStatus, Penalty};

/// AMSGrad optimizer.
///
/// It is a variant of the *Adam* algorithm from the paper
/// [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ).
#[allow(clippy::upper_case_acronyms)]
pub struct AMSGrad<T>
where
    T: Penalty,
{
    lr: Cell<f32>,
    penalty: T,
    beta1: Cell<f32>,
    beta2: Cell<f32>,
    eps: Cell<f32>,
}

impl<T> OptimizerStatus for AMSGrad<T>
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

impl<T> AMSGrad<T>
where
    T: Penalty,
{
    /// Creates a new AMSGrad optimizer.
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

        let status = Self {
            lr,
            penalty,
            beta1,
            beta2,
            eps,
        };

        Optimizer::new(status)
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f32 {
        OptimizerStatus::get_lr(self)
    }

    /// Sets a new value for the learning rate.
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

    /// Return the current eps constant.
    pub fn get_eps(&self) -> f32 {
        self.eps.get()
    }

    /// Sets a new value for the eps constant.
    pub fn set_eps(&self, eps: f32) {
        self.eps.set(eps)
    }
}

/// A parameter used by the *AMSGrad* optimizer.
#[allow(clippy::upper_case_acronyms)]
pub struct AMSGradParam<D, T>
where
    D: Dimension,
    T: Penalty,
{
    variable: VarDiff<D>,
    step: usize,
    exp_avg: Array<f32, D>,
    exp_avg_sq: Array<f32, D>,
    max_exp_avg_sq: Array<f32, D>,
    status: Rc<AMSGrad<T>>,
}

impl<D, T> IntoParam<AMSGrad<T>> for VarDiff<D>
where
    D: 'static + Dimension,
    T: 'static + Penalty,
{
    type Param = AMSGradParam<D, T>;

    fn into_param(self, status: Rc<AMSGrad<T>>) -> Self::Param {
        let variable = self;
        let step = 0;
        let exp_avg = Array::zeros(variable.data().raw_dim());
        let exp_avg_sq = exp_avg.clone();
        let max_exp_avg_sq = exp_avg_sq.clone();

        Self::Param {
            variable,
            step,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
            status,
        }
    }
}

impl<D, T> Optimize for AMSGradParam<D, T>
where
    D: Dimension,
    T: Penalty,
{
    fn optimize(&mut self) {
        self.step += 1;

        let lr = self.status.get_lr();
        let beta1 = self.status.get_beta1();
        let beta2 = self.status.get_beta2();
        let penalty = self.status.penalty;
        let eps = self.status.get_eps();

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

        Zip::from(&mut self.max_exp_avg_sq)
            .and(&self.exp_avg_sq)
            .for_each(|max_exp_avg_sq_el, exp_avg_sq_el| {
                *max_exp_avg_sq_el = max_exp_avg_sq_el.max(*exp_avg_sq_el)
            });

        Zip::from(&mut *data)
            .and(&self.exp_avg)
            .and(&self.max_exp_avg_sq)
            .for_each(|data_el, exp_avg_el, max_exp_avg_sq_el| {
                *data_el -= exp_avg_el
                    / ((max_exp_avg_sq_el.sqrt() / bias_correction2.sqrt()) + eps)
                    * (lr / bias_correction1)
            });
    }

    fn zero_grad(&mut self) {
        self.variable.zero_grad()
    }
}

#[cfg(test)]
mod test;
