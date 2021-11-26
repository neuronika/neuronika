use super::{Optimizer, Param, Penalty};
use ndarray::{ArrayD, ArrayViewMutD, Zip};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::cell::{Cell, RefCell};

#[allow(clippy::upper_case_acronyms)]
/// **Stochastic Gradient Descent** optimizer.
pub struct SGD<'a, T> {
    params: RefCell<Vec<SGDParam<'a>>>,
    lr: Cell<f32>,
    penalty: T,
}

#[allow(clippy::upper_case_acronyms)]
/// The parameter representation used by the *SDG* optimizer.
pub struct SGDParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
}

impl<'a> From<Param<'a>> for SGDParam<'a> {
    fn from(param: Param<'a>) -> Self {
        let Param { data, grad } = param;
        Self { data, grad }
    }
}

impl<'a, T: Penalty> Optimizer<'a> for SGD<'a, T> {
    type ParamRepr = SGDParam<'a>;

    fn step(&self) {
        let (lr, penalty, mut params) = (self.lr.get(), &self.penalty, self.params.borrow_mut());
        params.par_iter_mut().for_each(|param| {
            let (data, grad) = (&mut param.data, &param.grad);
            Zip::from(data).and(grad).for_each(|data_el, grad_el| {
                *data_el += -(grad_el + penalty.penalize(data_el)) * lr
            });
        });
    }

    fn zero_grad(&self) {
        self.params.borrow_mut().par_iter_mut().for_each(|param| {
            let grad = &mut param.grad;
            Zip::from(grad).for_each(|grad_el| *grad_el = 0.);
        });
    }

    fn get_lr(&self) -> f32 {
        self.lr.get()
    }

    fn set_lr(&self, lr: f32) {
        self.lr.set(lr)
    }
}

impl<'a, T: Penalty> SGD<'a, T> {
    /// Creates a new *SGD* optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - vector of [`Param`] to optimize.
    ///
    /// * `lr` - learning rate.
    ///
    /// * `penalty` - penalty regularization.
    pub fn new(parameters: Vec<Param<'a>>, lr: f32, penalty: T) -> Self {
        let params = RefCell::new(Self::build_params(parameters));
        let lr = Cell::new(lr);

        Self {
            params,
            lr,
            penalty,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f32 {
        Optimizer::get_lr(self)
    }

    /// Sets `lr` as the  new value for the learning rate.
    pub fn set_lr(&self, lr: f32) {
        Optimizer::set_lr(self, lr);
    }

    /// Performs a single stochastic gradient descent optimization step.
    pub fn step(&self) {
        Optimizer::step(self);
    }

    /// Zeroes the gradient of this optimizer's parameters.
    pub fn zero_grad(&self) {
        Optimizer::zero_grad(self);
    }

    /// Transforms this *SGD* optimizer in the *momentum* version of the algorithm.
    ///
    /// Nesterov momentum is based on the formula from
    /// [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf).
    ///
    /// Please **do note** that the implementation of SGD with Momentum/Nesterov subtly differs from
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
    ///
    /// # Arguments
    ///
    /// * `momentum` - the momentum factor.
    ///
    /// * `dampening` - the dampening factor for momentum.
    ///
    /// * `nesterov` - enables *Nesterov* momentum.
    pub fn with_momentum(
        self,
        momentum: f32,
        dampening: f32,
        nesterov: bool,
    ) -> SGDWithMomentum<'a, T> {
        let params: RefCell<Vec<SGDWithMomentumParam>> =
            RefCell::new(Self::build_params(self.params.into_inner()));

        SGDWithMomentum {
            params,
            lr: self.lr,
            penalty: self.penalty,
            momentum: Cell::new(momentum),
            dampening: Cell::new(dampening),
            nesterov: Cell::new(nesterov),
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
/// The momentum variant of the *Stochastic Gradient Descent* optimizer.
pub struct SGDWithMomentum<'a, T> {
    params: RefCell<Vec<SGDWithMomentumParam<'a>>>,
    lr: Cell<f32>,
    penalty: T,
    momentum: Cell<f32>,
    dampening: Cell<f32>,
    nesterov: Cell<bool>,
}

#[allow(clippy::upper_case_acronyms)]
/// The  parameter representation used by the *SDG with momentum* optimizer.
pub struct SGDWithMomentumParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    buffer: ArrayD<f32>,
}

impl<'a> From<Param<'a>> for SGDWithMomentumParam<'a> {
    fn from(param: Param<'a>) -> Self {
        let Param { data, grad } = param;
        let buffer = ArrayD::zeros(grad.raw_dim());
        Self { data, grad, buffer }
    }
}

impl<'a> From<SGDParam<'a>> for SGDWithMomentumParam<'a> {
    fn from(param: SGDParam<'a>) -> Self {
        let (data, grad) = (param.data, param.grad);
        let buffer = ArrayD::zeros(grad.raw_dim());
        Self { data, grad, buffer }
    }
}

impl<'a, T: Penalty> Optimizer<'a> for SGDWithMomentum<'a, T> {
    type ParamRepr = SGDWithMomentumParam<'a>;

    fn step(&self) {
        let (lr, penalty, momentum, dampening, nesterov, mut params) = (
            self.lr.get(),
            &self.penalty,
            &self.momentum.get(),
            &self.dampening.get(),
            &self.nesterov.get(),
            self.params.borrow_mut(),
        );

        params.par_iter_mut().for_each(|param| {
            let mut p_grad = param.grad.to_owned();
            Zip::from(&mut p_grad)
                .and(&param.data)
                .for_each(|p_grad_el, data_el| *p_grad_el += penalty.penalize(data_el));

            Zip::from(&mut param.buffer)
                .and(&p_grad)
                .for_each(|buffer_el, p_grad_el| {
                    *buffer_el = *buffer_el * *momentum + p_grad_el * (1. - dampening)
                });

            let zip = Zip::from(&mut param.data).and(&param.buffer);
            if *nesterov {
                zip.and(&p_grad).for_each(|data_el, buffer_el, p_grad_el| {
                    *data_el += -(p_grad_el + *buffer_el * *momentum) * lr
                });
            } else {
                zip.for_each(|data_el, buffer_el| *data_el += -*buffer_el * lr);
            }
        });
    }

    fn zero_grad(&self) {
        self.params.borrow_mut().par_iter_mut().for_each(|param| {
            let grad = &mut param.grad;
            Zip::from(grad).for_each(|grad_el| *grad_el = 0.);
        });
    }

    fn get_lr(&self) -> f32 {
        self.lr.get()
    }

    fn set_lr(&self, lr: f32) {
        self.lr.set(lr)
    }
}

impl<'a, T: Penalty> SGDWithMomentum<'a, T> {
    /// Returns the current learning rate.
    pub fn get_lr(&self) -> f32 {
        Optimizer::get_lr(self)
    }

    /// Sets `lr` as the new value for the learning rate.
    pub fn set_lr(&self, lr: f32) {
        Optimizer::set_lr(self, lr);
    }

    /// Returns the current momentum.
    pub fn get_momentum(&self) -> f32 {
        self.momentum.get()
    }

    /// Sets `momentum` as the new value for the momentum.
    pub fn set_momentum(&self, momentum: f32) {
        self.momentum.set(momentum);
    }

    /// Returns the current dampening value.
    pub fn get_dampening(&self) -> f32 {
        self.dampening.get()
    }

    /// Sets `dampening` as the current dampening value.
    pub fn set_dampening(&self, dampening: f32) {
        self.dampening.set(dampening);
    }

    /// Returns `true` if this optimizer has Nesterov momentum enabled, `false` otherwise.
    pub fn get_nesterov(&self) -> bool {
        self.nesterov.get()
    }

    /// Sets `nesterov` as the new value for the Nesterov momentum flag.
    pub fn set_nesterov(&self, nesterov: bool) {
        self.nesterov.set(nesterov);
    }

    /// Performs a single optimization step.
    pub fn step(&self) {
        Optimizer::step(self);
    }

    /// Zeroes the gradient of this optimizer's parameters.
    pub fn zero_grad(&self) {
        Optimizer::zero_grad(self);
    }
}

#[cfg(test)]
mod test;
