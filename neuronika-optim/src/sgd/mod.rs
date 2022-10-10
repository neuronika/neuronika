use std::{cell::Cell, rc::Rc};

use ndarray::{Array, Dimension, Zip};

use neuronika_variable::VarDiff;

use super::{IntoParam, Optimize, Optimizer, OptimizerStatus, Penalty};

/// Stochastic gradient descent optimizer.
#[allow(clippy::upper_case_acronyms)]
pub struct StochasticGD<T>
where
    T: Penalty,
{
    lr: Cell<f32>,
    penalty: T,
    momentum: Cell<Option<f32>>,
    dampening: Cell<Option<f32>>,
    nesterov: Cell<bool>,
}

impl<T> OptimizerStatus for StochasticGD<T>
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

impl<T> StochasticGD<T>
where
    T: Penalty,
{
    /// Creates a new stochastic gradient descent optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - learning rate.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `momentum` - momentum factor.
    ///
    /// * `dampening` - dampening factor for momentum.
    ///
    /// * `nesterov` - enables Nesterov momentum.
    ///
    /// Nesterov momentum is based on the formula from [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf).
    ///
    /// Please do note that the implementation with Momentum / Nesterov subtly differs from
    /// Sutskever et. al. and implementations in some other frameworks.
    ///
    /// Considering the specific case of momentum the update rule can be written as:
    ///
    /// ```text
    /// v(t+1) = μ * v(t) + g(t+1)
    /// p(t+1) = p(t) - lr * v(t+1)
    /// ```
    ///
    /// Where p, g, v, and μ denote the parameters, gradient, velocity and momentum respectively.
    ///
    /// This is in contrast to Sutskever et. al. and other frameworks which employ an update of the
    /// form:
    ///
    /// ```text
    /// v(t+1) = μ * v(t) + lr * g(t+1)
    /// p(t+1) = p(t) - v(t+1)
    /// ```
    pub fn new(
        lr: f32,
        penalty: T,
        momentum: impl Into<Option<f32>>,
        dampening: impl Into<Option<f32>>,
        nesterov: bool,
    ) -> Optimizer<Self> {
        let lr = Cell::new(lr);
        let momentum = Cell::new(momentum.into());
        let dampening = Cell::new(dampening.into());
        let nesterov = Cell::new(nesterov);

        if momentum.get().is_none() {
            assert!(
                dampening.get().is_none() && !nesterov.get(),
                "Dampening and Nesterov momentum flag should be enabled together with momentum."
            );
        }

        if let Some(dampening) = dampening.get() {
            assert!(
                (0.0..=1.0).contains(&dampening),
                "Dampening value should be between 0.0 and 1.0, got: {dampening}"
            );
        }

        let status = Self {
            lr,
            penalty,
            momentum,
            dampening,
            nesterov,
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

    /// Returns the current momentum option.
    pub fn get_momentum(&self) -> Option<f32> {
        self.momentum.get()
    }

    /// Sets a new value for the momentum option
    pub fn set_momentum(&self, momentum: f32) {
        self.momentum.set(Some(momentum));
    }

    /// Returns the current dampening option.
    pub fn get_dampening(&self) -> Option<f32> {
        self.dampening.get()
    }

    /// Sets a new value for the dampening option.
    pub fn set_dampening(&self, dampening: f32) {
        self.dampening.set(Some(dampening));
    }

    /// Returns `true` if the Nesterov momentum is enabled.
    pub fn get_nesterov(&self) -> bool {
        self.nesterov.get()
    }

    /// Sets a new value for the Nesterov momentum flag.
    pub fn set_nesterov(&self, nesterov: bool) {
        self.nesterov.set(nesterov);
    }
}

/// A parameter used by the SDG optimizer.
#[allow(clippy::upper_case_acronyms)]
pub struct SGDParam<D, T>
where
    D: Dimension,
    T: Penalty,
{
    variable: VarDiff<D>,
    buffer: Option<Array<f32, D>>,
    status: Rc<StochasticGD<T>>,
}

impl<D, T> IntoParam<StochasticGD<T>> for VarDiff<D>
where
    D: 'static + Dimension,
    T: 'static + Penalty,
{
    type Param = SGDParam<D, T>;

    fn into_param(self, status: Rc<StochasticGD<T>>) -> Self::Param {
        let variable = self;
        let buffer = status
            .get_momentum()
            .filter(|val| *val > f32::EPSILON)
            .map(|_| Array::zeros(variable.grad().raw_dim()));

        Self::Param {
            variable,
            buffer,
            status,
        }
    }
}

impl<D, T> Optimize for SGDParam<D, T>
where
    D: Dimension,
    T: Penalty,
{
    fn optimize(&mut self) {
        let lr = self.status.get_lr();
        let penalty = self.status.penalty;

        let mut data = self.variable.data_mut();
        let mut grad = self.variable.grad_mut();

        Zip::from(&mut *grad)
            .and(&*data)
            .for_each(|grad_el, data_el| *grad_el += penalty.penalize(data_el));

        match self.status.get_momentum().filter(|val| *val > f32::EPSILON) {
            None => {
                self.buffer = None;
                Zip::from(&mut *data)
                    .and(&*grad)
                    .for_each(|data_el, grad_el| *data_el -= grad_el * lr);
            }
            Some(momentum) => {
                let dampening = self.status.get_dampening().unwrap_or(0.0);
                if self.buffer.is_none() {
                    self.buffer = Some(Array::zeros(grad.raw_dim()));
                }

                Zip::from(self.buffer.as_mut().unwrap())
                    .and(&*grad)
                    .for_each(|buffer_el, grad_el| {
                        *buffer_el = *buffer_el * momentum + *grad_el * (1.0 - dampening)
                    });

                let zip = Zip::from(&mut *data).and(self.buffer.as_ref().unwrap());
                if self.status.get_nesterov() {
                    zip.and(&*grad).for_each(|data_el, buffer_el, grad_el| {
                        *data_el -= (grad_el + *buffer_el * momentum) * lr
                    });
                } else {
                    zip.for_each(|data_el, buffer_el| *data_el -= *buffer_el * lr);
                }
            }
        }
    }

    fn zero_grad(&mut self) {
        self.variable.zero_grad()
    }
}

#[cfg(test)]
mod test;
