use super::{Optimizer, Param, Penalty};
use ndarray::{ArrayD, ArrayViewMutD, Zip};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AMSGrad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// The **AMSGrad** optimizer.
///
/// It is a variant of the *Adam* algorithm from the paper
/// [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ).
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct AMSGrad<'a, T: Penalty> {
    params: Vec<AMSGradParam<'a>>,
    lr: f32,
    penalty: T,
    betas: (f32, f32),
    eps: f32,
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
    /// * `betas` - a `tuple` of coefficients used for computing running averages of the gradient
    /// and its square. Good default is: *(0.9, 0.999)*.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-8*.
    pub fn new(params: Vec<Param>, lr: f32, penalty: T, betas: (f32, f32), eps: f32) -> Self {
        let params = Self::build_params(params);

        Self {
            params,
            lr,
            penalty,
            betas,
            eps,
        }
    }
}

/// A parameter used by the *AMSGrad* optmizier.
#[allow(clippy::clippy::upper_case_acronyms)]
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

    fn step(&mut self) {
        let (lr, penalty, params, (beta1, beta2), eps) = (
            &self.lr,
            &self.penalty,
            &mut self.params,
            &self.betas,
            &self.eps,
        );

        params.par_iter_mut().for_each(|param| {
            let (data, grad, step, exp_avg, exp_avg_sq, max_exp_avg_sq) = (
                &mut param.data,
                &param.grad,
                &mut param.step,
                &mut param.exp_avg,
                &mut param.exp_avg_sq,
                &mut param.max_exp_avg_sq,
            );

            *step += 1;
            let bias_correction1 = 1. - beta1.powi(*step as i32);
            let bias_correction2 = 1. - beta2.powi(*step as i32);

            Zip::from(exp_avg)
                .and(grad)
                .for_each(|exp_avg_el, grad_el| {
                    *exp_avg_el =
                        *exp_avg_el * beta1 + (grad_el + penalty.penalise(grad_el)) * (1. - beta1)
                });

            Zip::from(exp_avg_sq)
                .and(grad)
                .for_each(|exp_avg_sq_el, grad_el| {
                    *exp_avg_sq_el = *exp_avg_sq_el * beta2
                        + (grad_el + penalty.penalise(grad_el))
                            * (grad_el + penalty.penalise(grad_el))
                            * (1. - beta2)
                });

            Zip::from(max_exp_avg_sq).and(&param.exp_avg_sq).for_each(
                |max_exp_avg_sq_el, exp_avg_sq_el| {
                    *max_exp_avg_sq_el = max_exp_avg_sq_el.max(*exp_avg_sq_el)
                },
            );

            Zip::from(data)
                .and(&param.exp_avg)
                .and(&param.max_exp_avg_sq)
                .for_each(|data_el, exp_avg_el, max_exp_avg_sq_el| {
                    *data_el += exp_avg_el
                        / ((max_exp_avg_sq_el.sqrt() / bias_correction2.sqrt()) + *eps)
                        * (-lr / bias_correction1)
                })
        });
    }

    fn zero_grad(&mut self) {
        self.params.par_iter_mut().for_each(|param| {
            let grad = &mut param.grad;
            Zip::from(grad).for_each(|grad_el| *grad_el = 0.);
        });
    }
}
