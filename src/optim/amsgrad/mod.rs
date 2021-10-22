use super::{Optimizer, Param, Penalty};
use ndarray::{ArrayD, ArrayViewMutD, Zip};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::cell::{Cell, RefCell};

/// **AMSGrad** optimizer.
///
/// It is a variant of the *Adam* algorithm from the paper
/// [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ).
#[allow(clippy::upper_case_acronyms)]
pub struct AMSGrad<'a, T: Penalty> {
    params: RefCell<Vec<AMSGradParam<'a>>>,
    lr: Cell<f32>,
    penalty: T,
    betas: Cell<(f32, f32)>,
    eps: Cell<f32>,
}

impl<'a, T: Penalty> AMSGrad<'a, T> {
    /// Creates a new *AMSGrad* optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - vector of [`Param`] to optimize.
    ///
    /// * `lr` - learning rate.
    ///
    /// * `betas` - a 2-tuple of coefficients used for computing running averages of the gradient
    /// and its square. Good default is: *(0.9, 0.999)*.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-8*.
    pub fn new(params: Vec<Param>, lr: f32, betas: (f32, f32), penalty: T, eps: f32) -> Self {
        let params = RefCell::new(Self::build_params(params));
        let lr = Cell::new(lr);

        Self {
            params,
            lr,
            penalty,
            betas: Cell::new(betas),
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

    /// Return the current values for the exponential decay rates.
    pub fn get_betas(&self) -> (f32, f32) {
        self.betas.get()
    }

    /// Sets `betas` as the  new value for the exponential decay rates.
    pub fn set_betas(&self, betas: (f32, f32)) {
        self.betas.set(betas)
    }

    /// Return the current *eps* constant.
    pub fn get_eps(&self) -> f32 {
        self.eps.get()
    }

    /// Sets `eps` as the  new value for the *eps* constant.
    pub fn set_eps(&self, eps: f32) {
        self.eps.set(eps)
    }

    /// Performs a single Adam optimization step.
    pub fn step(&self) {
        Optimizer::step(self);
    }

    /// Zeroes the gradient of this optimizer's parameters.
    pub fn zero_grad(&self) {
        Optimizer::zero_grad(self);
    }
}

/// A parameter used by the *AMSGrad* optimizer.
#[allow(clippy::upper_case_acronyms)]
pub struct AMSGradParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    step: usize,
    exp_avg: ArrayD<f32>,
    exp_avg_sq: ArrayD<f32>,
    max_exp_avg_sq: ArrayD<f32>,
}

impl<'a> From<Param> for AMSGradParam<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
        let step = 0;
        let (exp_avg, exp_avg_sq, max_exp_avg_sq) = {
            (
                ArrayD::zeros(grad.raw_dim()),
                ArrayD::zeros(grad.raw_dim()),
                ArrayD::zeros(grad.raw_dim()),
            )
        };

        Self {
            data,
            grad,
            step,
            exp_avg,
            exp_avg_sq,
            max_exp_avg_sq,
        }
    }
}

impl<'a, T: Penalty> Optimizer for AMSGrad<'a, T> {
    type ParamRepr = AMSGradParam<'a>;

    fn step(&self) {
        let (lr, penalty, mut params, (beta1, beta2), eps) = (
            self.lr.get(),
            &self.penalty,
            self.params.borrow_mut(),
            &self.betas.get(),
            &self.eps.get(),
        );

        params.par_iter_mut().for_each(|param| {
            let (step, exp_avg, exp_avg_sq, max_exp_avg_sq) = (
                &mut param.step,
                &mut param.exp_avg,
                &mut param.exp_avg_sq,
                &mut param.max_exp_avg_sq,
            );

            *step += 1;
            let bias_correction1 = 1. - beta1.powi(*step as i32);
            let bias_correction2 = 1. - beta2.powi(*step as i32);

            let mut p_grad = param.grad.to_owned();
            Zip::from(&mut p_grad)
                .and(&param.data)
                .for_each(|p_grad_el, data_el| *p_grad_el += penalty.penalize(data_el));

            Zip::from(exp_avg)
                .and(&p_grad)
                .for_each(|exp_avg_el, p_grad_el| {
                    *exp_avg_el = *exp_avg_el * beta1 + p_grad_el * (1. - beta1)
                });

            Zip::from(exp_avg_sq)
                .and(&p_grad)
                .for_each(|exp_avg_sq_el, p_grad_el| {
                    *exp_avg_sq_el = *exp_avg_sq_el * beta2 + p_grad_el * p_grad_el * (1. - beta2)
                });

            Zip::from(max_exp_avg_sq).and(&param.exp_avg_sq).for_each(
                |max_exp_avg_sq_el, exp_avg_sq_el| {
                    *max_exp_avg_sq_el = max_exp_avg_sq_el.max(*exp_avg_sq_el)
                },
            );

            Zip::from(&mut param.data)
                .and(&param.exp_avg)
                .and(&param.max_exp_avg_sq)
                .for_each(|data_el, exp_avg_el, max_exp_avg_sq_el| {
                    *data_el += exp_avg_el
                        / ((max_exp_avg_sq_el.sqrt() / bias_correction2.sqrt()) + *eps)
                        * (-lr / bias_correction1)
                })
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
