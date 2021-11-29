use super::{Optimizer, Param, Penalty};
use ndarray::{ArrayD, ArrayViewMutD, Zip};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::cell::{Cell, RefCell};
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// **Adagrad** optimizer.
///
/// The algorithm has been proposed in [this paper](http://jmlr.org/papers/v12/duchi11a.html).
pub struct Adagrad<'a, T: Penalty> {
    params: RefCell<Vec<AdagradParam<'a>>>,
    lr: Cell<f32>,
    lr_decay: Cell<f32>,
    penalty: T,
    eps: Cell<f32>,
}

impl<'a, T: Penalty> Adagrad<'a, T> {
    /// Creates a new *Adagrad* optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - vector of [`Param`] to optimize.
    ///
    /// * `lr` - learning rate.
    ///
    /// * `lr_decay` - the learning rate decay.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-10*.
    pub fn new(params: Vec<Param<'a>>, lr: f32, lr_decay: f32, penalty: T, eps: f32) -> Self {
        let params = RefCell::new(Self::build_params(params));
        let lr = Cell::new(lr);

        Self {
            params,
            lr,
            lr_decay: Cell::new(lr_decay),
            penalty,
            eps: Cell::new(eps),
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

    /// Return the current learning rate decay parameter.
    pub fn get_lr_decay(&self) -> f32 {
        self.lr_decay.get()
    }

    /// Sets `lr_decay` as the  new value for the learning rate decay parameter.
    pub fn set_lr_decay(&self, lr_decay: f32) {
        self.lr_decay.set(lr_decay)
    }

    /// Return the current *eps* constant.
    pub fn get_eps(&self) -> f32 {
        self.eps.get()
    }

    /// Sets `eps` as the  new value for the *eps* constant.
    pub fn set_eps(&self, eps: f32) {
        self.eps.set(eps)
    }

    /// Performs a single Adagrad optimization step.
    pub fn step(&self) {
        Optimizer::step(self);
    }

    /// Zeroes the gradient of this optimizer's parameters.
    pub fn zero_grad(&self) {
        Optimizer::zero_grad(self);
    }
}

/// A parameter used by the *Adagrad* optimizer.
pub struct AdagradParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    step: usize,
    grad_sq: ArrayD<f32>,
}

impl<'a> From<Param<'a>> for AdagradParam<'a> {
    fn from(param: Param<'a>) -> Self {
        let Param { data, grad } = param;
        let step = 0;
        let grad_sq = ArrayD::zeros(grad.raw_dim());

        Self {
            data,
            grad,
            step,
            grad_sq,
        }
    }
}

impl<'a, T: Penalty> Optimizer<'a> for Adagrad<'a, T> {
    type ParamRepr = AdagradParam<'a>;
    fn step(&self) {
        let (mut params, lr, lr_decay, penalty, eps) = (
            self.params.borrow_mut(),
            self.lr.get(),
            &self.lr_decay.get(),
            &self.penalty,
            &self.eps.get(),
        );

        params.par_iter_mut().for_each(|param| {
            let (step, grad_sq) = (&mut param.step, &mut param.grad_sq);

            *step += 1;
            let clr = lr / (1. + (*step - 1) as f32 * lr_decay);

            let mut p_grad = param.grad.to_owned();
            Zip::from(&mut p_grad)
                .and(&param.data)
                .for_each(|p_grad_el, data_el| *p_grad_el += penalty.penalize(data_el));

            Zip::from(grad_sq)
                .and(&p_grad)
                .for_each(|grad_sq_el, p_grad_el| *grad_sq_el += p_grad_el * p_grad_el);

            Zip::from(&mut param.data)
                .and(&p_grad)
                .and(&param.grad_sq)
                .for_each(|data_el, p_grad_el, grad_sq_el| {
                    *data_el += -p_grad_el / (grad_sq_el.sqrt() + eps) * clr
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

#[cfg(test)]
mod test;
