use std::{cell::Cell, rc::Rc};

use crate::variable::VarDiff;

use ndarray::{Array, Dimension, Zip};

use super::{IntoParam, Optimize, Optimizer, OptimizerStatus, Penalty};

/// RMSProp optimizer.
///
/// It was proposed by *G. Hinton* in his
/// [course](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
///
/// The centered version first appears in
/// [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5.pdf).
///
/// The implementation here takes the square root of the gradient average before adding
/// epsilon. Do note that TensorFlow interchanges these two operations. The effective
/// learning rate is thus *lr' / (v.sqrt() + eps)* where *lr'* is the scheduled
/// learning rate and *v* is the weighted moving average of the square gradient.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSProp<T>
where
    T: Penalty,
{
    lr: Cell<f32>,
    alpha: Cell<Option<f32>>,
    momentum: Cell<Option<f32>>,
    centered: Cell<bool>,
    penalty: T,
    eps: Cell<f32>,
}

impl<T> OptimizerStatus for RMSProp<T>
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

impl<T> RMSProp<T>
where
    T: Penalty,
{
    /// Creates a *RMSProp* optimizer.
    ///
    /// # Arguments
    ///
    /// * `lr` - learning rate.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `alpha` - smoothing constant. A good default value is *0.99*.
    ///
    /// * `momentum` - momentum factor.
    ///
    /// * `centered` - sets the centered flag. When set the gradient is normalized by an estimation
    /// of its variance.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-8*.
    pub fn new(
        lr: f32,
        penalty: T,
        alpha: impl Into<Option<f32>>,
        momentum: impl Into<Option<f32>>,
        centered: bool,
        eps: f32,
    ) -> Optimizer<Self> {
        let lr = Cell::new(lr);
        let alpha = Cell::new(alpha.into());
        let momentum = Cell::new(momentum.into());
        let centered = Cell::new(centered);
        let eps = Cell::new(eps);

        if let Some(alpha) = alpha.get() {
            assert!(
                (0.0..=1.0).contains(&alpha),
                "Dampening value should be between 0.0 and 1.0, got: {alpha}"
            );
        }

        let status = Self {
            lr,
            penalty,
            alpha,
            momentum,
            centered,
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

    /// Return the current alpha value.
    pub fn get_alpha(&self) -> Option<f32> {
        self.alpha.get()
    }

    /// Sets a new value for the smoothing constant.
    pub fn set_alpha(&self, alpha: f32) {
        self.alpha.set(Some(alpha));
    }

    /// Returns the current momentum option.
    pub fn get_momentum(&self) -> Option<f32> {
        self.momentum.get()
    }

    /// Sets a new value for the momentum option
    pub fn set_momentum(&self, momentum: f32) {
        self.momentum.set(Some(momentum));
    }

    /// Returns `true` if the centered flag is enabled.
    pub fn get_centered(&self) -> bool {
        self.centered.get()
    }

    /// Sets a new value for the centered flag.
    pub fn set_centered(&self, centered: bool) {
        self.centered.set(centered);
    }

    /// Return the current epsilon constant.
    pub fn get_eps(&self) -> f32 {
        self.eps.get()
    }

    /// Sets a new value for the epsilon constant.
    pub fn set_eps(&self, eps: f32) {
        self.eps.set(eps)
    }
}

/// A parameter used by the *RMSProp* optimizer.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSPropParam<D, T>
where
    D: Dimension,
    T: Penalty,
{
    variable: VarDiff<D>,
    square_avg: Array<f32, D>,
    buffer: Option<Array<f32, D>>,
    grad_avg: Option<Array<f32, D>>,
    status: Rc<RMSProp<T>>,
}

impl<D, T> IntoParam<RMSProp<T>> for VarDiff<D>
where
    D: 'static + Dimension,
    T: 'static + Penalty,
{
    type Param = RMSPropParam<D, T>;

    fn into_param(self, status: Rc<RMSProp<T>>) -> Self::Param {
        let variable = self;
        let square_avg = Array::zeros(variable.data().raw_dim());
        let buffer = status.get_momentum().map(|_| square_avg.clone());
        let grad_avg = status.get_centered().then(|| square_avg.clone());

        Self::Param {
            variable,
            square_avg,
            buffer,
            grad_avg,
            status,
        }
    }
}

impl<D, T> Optimize for RMSPropParam<D, T>
where
    D: 'static + Dimension,
    T: 'static + Penalty,
{
    fn optimize(&mut self) {
        let lr = self.status.get_lr();
        let alpha = self.status.get_alpha().unwrap_or(0.0);
        let penalty = self.status.penalty;
        let eps = self.status.get_eps();

        let mut data = self.variable.data_mut();
        let mut grad = self.variable.grad_mut();

        Zip::from(&mut *grad)
            .and(&*data)
            .for_each(|grad_el, data_el| *grad_el += penalty.penalize(data_el));

        Zip::from(&mut self.square_avg)
            .and(&*grad)
            .for_each(|square_avg_el, grad_el| {
                *square_avg_el = *square_avg_el * alpha + grad_el * grad_el * (1.0 - alpha)
            });

        match (
            self.status.get_centered(),
            self.status
                .get_momentum()
                .filter(|momentum| *momentum > f32::EPSILON),
        ) {
            (true, Some(momentum)) => {
                if self.grad_avg.is_none() {
                    self.grad_avg = Some(Array::zeros(grad.raw_dim()));
                }
                if self.buffer.is_none() {
                    self.buffer = Some(Array::zeros(grad.raw_dim()));
                }

                Zip::from(self.grad_avg.as_mut().unwrap())
                    .and(&*grad)
                    .for_each(|grad_avg_el, grad_el| {
                        *grad_avg_el = *grad_avg_el * alpha + grad_el * (1.0 - alpha)
                    });

                Zip::from(self.buffer.as_mut().unwrap())
                    .and(&*grad)
                    .and(&self.square_avg)
                    .and(self.grad_avg.as_ref().unwrap())
                    .for_each(|buffer_el, grad_el, square_avg_el, grad_avg_el| {
                        *buffer_el = *buffer_el * momentum
                            + grad_el
                                / ((square_avg_el + (-grad_avg_el * grad_avg_el)).sqrt() + eps)
                    });

                Zip::from(&mut *data)
                    .and(self.buffer.as_ref().unwrap())
                    .for_each(|data_el, buffer_el| *data_el -= buffer_el * lr);
            }
            (false, Some(momentum)) => {
                if self.buffer.is_none() {
                    self.buffer = Some(Array::zeros(grad.raw_dim()));
                }
                self.grad_avg = None;

                Zip::from(self.buffer.as_mut().unwrap())
                    .and(&*grad)
                    .and(&mut self.square_avg)
                    .for_each(|buffer_el, grad_el, square_avg_el| {
                        *buffer_el = *buffer_el * momentum + grad_el / (square_avg_el.sqrt() + eps)
                    });

                Zip::from(&mut *data)
                    .and(self.buffer.as_ref().unwrap())
                    .for_each(|data_el, buffer_el| *data_el -= buffer_el * lr);
            }
            (true, None) => {
                if self.grad_avg.is_none() {
                    self.grad_avg = Some(Array::zeros(grad.raw_dim()));
                }
                self.buffer = None;

                Zip::from(self.grad_avg.as_mut().unwrap())
                    .and(&*grad)
                    .for_each(|grad_avg_el, grad_el| {
                        *grad_avg_el = *grad_avg_el * alpha + grad_el * (1.0 - alpha)
                    });

                Zip::from(&mut *data)
                    .and(&*grad)
                    .and(&self.square_avg)
                    .and(self.grad_avg.as_ref().unwrap())
                    .for_each(|data_el, grad_el, square_avg_el, grad_avg_el| {
                        *data_el -= grad_el
                            / ((square_avg_el + (-grad_avg_el * grad_avg_el)).sqrt() + eps)
                            * lr
                    });
            }
            (false, None) => {
                self.buffer = None;
                self.grad_avg = None;

                Zip::from(&mut *data)
                    .and(&*grad)
                    .and(&self.square_avg)
                    .for_each(|data_el, grad_el, square_avg_el| {
                        *data_el -= grad_el / (square_avg_el.sqrt() + eps) * lr
                    });
            }
        };
    }

    fn zero_grad(&mut self) {
        self.variable.zero_grad()
    }
}

#[cfg(test)]
mod test;
