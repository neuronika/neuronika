use super::{Optimizer, Penalty};
use crate::variable::Param;
use ndarray::{ArrayD, ArrayViewMutD, Zip};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Adam ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Adam<'a, T> {
    params: Vec<AdamParam<'a>>,
    lr: f32,
    penalty: T,
    betas: (f32, f32),
    eps: f32,
}

pub struct AdamParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    step: usize,
    exp_avg: ArrayD<f32>,
    exp_avg_sq: ArrayD<f32>,
}

impl<'a> From<Param> for AdamParam<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
        let step = 0;
        let (exp_avg, exp_avg_sq) =
            { (ArrayD::zeros(grad.raw_dim()), ArrayD::zeros(grad.raw_dim())) };
        Self {
            data,
            grad,
            step,
            exp_avg,
            exp_avg_sq,
        }
    }
}

impl<'a, T: Penalty> Optimizer<AdamParam<'a>> for Adam<'a, T> {
    fn step(&mut self) {
        let (lr, penalty, params, (beta1, beta2), eps) = (
            &self.lr,
            &self.penalty,
            &mut self.params,
            &self.betas,
            &self.eps,
        );

        params.par_iter_mut().for_each(|param| {
            let (data, grad, step, exp_avg, exp_avg_sq) = (
                &mut param.data,
                &param.grad,
                &mut param.step,
                &mut param.exp_avg,
                &mut param.exp_avg_sq,
            );

            let bias_correction1 = 1. - beta1.powi(*step as i32);
            let bias_correction2 = 1. - beta2.powi(*step as i32);

            *step += 1;

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

            Zip::from(data)
                .and(&param.exp_avg)
                .and(&param.exp_avg_sq)
                .for_each(|data_el, exp_avg_el, exp_avg_sq_el| {
                    *data_el = exp_avg_el
                        / ((exp_avg_sq_el.sqrt() / bias_correction2.sqrt()) + *eps)
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
