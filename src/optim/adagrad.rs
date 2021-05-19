use super::{Optimizer, Penalty};
use crate::variable::Param;
use ndarray::{ArrayD, ArrayViewMutD, Zip};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Adagrad ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// The **Adagrad** optimizer.
///
/// The algorithm has been proposed in [this paper](http://jmlr.org/papers/v12/duchi11a.html).
pub struct Adagrad<'a, T> {
    params: Vec<AdagradParam<'a>>,
    lr: f32,
    lr_decay: f32,
    penalty: T,
    eps: f32,
}

impl<'a, T> Adagrad<'a, T> {
    /// Creates a new *Adagrad* optimizer.
    ///
    /// # Arguments
    ///
    /// `params` - `Vec` of parameters to optimize.
    /// `lr` - learning rate.
    /// `lr_decay` - the learning rate decay.
    /// `penalty` - penalty regularization.
    /// `eps` - small constant for numerical stability. A good default value is *1e-10*.
    pub fn new(params: Vec<Param>, lr: f32, lr_decay: f32, penalty: T, eps: f32) -> Self {
        let params = {
            let mut vec = Vec::with_capacity(params.len());
            for param in params {
                vec.push(AdagradParam::from(param));
            }
            vec
        };

        Self {
            params,
            lr,
            lr_decay,
            penalty,
            eps,
        }
    }
}

/// A parameter used by the *Adagrad* optimizer.
struct AdagradParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    step: usize,
    grad_sq: ArrayD<f32>,
}

impl<'a> From<Param> for AdagradParam<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
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

impl<'a, T: Penalty> Optimizer<AdagradParam<'a>> for Adagrad<'a, T> {
    fn step(&mut self) {
        let (params, lr, lr_decay, penalty, eps) = (
            &mut self.params,
            &self.lr,
            &self.lr_decay,
            &self.penalty,
            &self.eps,
        );

        params.par_iter_mut().for_each(|param| {
            let (data, grad, step, grad_sq) = (
                &mut param.data,
                &param.grad,
                &mut param.step,
                &mut param.grad_sq,
            );

            *step += 1;
            let clr = *lr / (1. + (*step - 1) as f32 * lr_decay);

            Zip::from(grad_sq)
                .and(grad)
                .for_each(|grad_sq_el, grad_el| {
                    *grad_sq_el += (grad_el + penalty.penalise(grad_el))
                        * (grad_el + penalty.penalise(grad_el))
                });

            Zip::from(data).and(grad).and(&param.grad_sq).for_each(
                |data_el, grad_el, grad_sq_el| {
                    *data_el +=
                        -(grad_el + penalty.penalise(grad_el)) / (grad_sq_el.sqrt() + eps) * clr
                },
            );
        });
    }

    fn zero_grad(&mut self) {
        self.params.par_iter_mut().for_each(|param| {
            let grad = &mut param.grad;
            Zip::from(grad).for_each(|grad_el| *grad_el = 0.);
        });
    }
}
