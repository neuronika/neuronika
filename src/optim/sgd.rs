use super::{Optimizer, Penalty};
use crate::variable::Param;
use ndarray::{ArrayD, ArrayViewMutD, Zip};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stochastic Gradient Descent ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct SGD<'a, T> {
    params: Vec<SGDParam<'a>>,
    lr: f32,
    penalty: T,
}

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct SGDParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
}

impl<'a> From<Param> for SGDParam<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
        Self { data, grad }
    }
}

impl<'a, T: Penalty> Optimizer<SGDParam<'a>> for SGD<'a, T> {
    fn step(&mut self) {
        let (lr, penalty, params) = (&self.lr, &self.penalty, &mut self.params);
        params.par_iter_mut().for_each(|param| {
            let (data, grad) = (&mut param.data, &param.grad);
            Zip::from(data).and(grad).for_each(|data_el, grad_el| {
                *data_el = -(grad_el + penalty.penalise(grad_el)) * lr
            });
        });
    }

    fn zero_grad(&mut self) {
        self.params.par_iter_mut().for_each(|param| {
            let grad = &mut param.grad;
            Zip::from(grad).for_each(|grad_el| *grad_el = 0.);
        });
    }
}

impl<'a, T: Penalty> SGD<'a, T> {
    pub fn new(parameters: Vec<Param>, lr: f32, penalty: T) -> Self {
        let params = {
            let mut vec = Vec::with_capacity(parameters.len());
            for param in parameters {
                vec.push(SGDParam::from(param));
            }
            vec
        };
        Self {
            params,
            lr,
            penalty,
        }
    }

    pub fn with_momentum(
        self,
        momentum: f32,
        dampening: f32,
        nesterov: bool,
    ) -> SGDWithMomentum<'a, T> {
        let params = {
            let parameters = self.params;
            let mut vec = Vec::with_capacity(parameters.len());
            for param in parameters {
                vec.push(SGDParamWithMomentum::from(param));
            }
            vec
        };

        SGDWithMomentum {
            params,
            lr: self.lr,
            penalty: self.penalty,
            momentum,
            dampening,
            nesterov,
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stochastic Gradient Descent with Momentum ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct SGDWithMomentum<'a, T> {
    params: Vec<SGDParamWithMomentum<'a>>,
    lr: f32,
    penalty: T,
    momentum: f32,
    dampening: f32,
    nesterov: bool,
}

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct SGDParamWithMomentum<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    buffer: ArrayD<f32>,
}

impl<'a> From<Param> for SGDParamWithMomentum<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
        let buffer = ArrayD::zeros(grad.raw_dim());
        Self { data, grad, buffer }
    }
}

impl<'a> From<SGDParam<'a>> for SGDParamWithMomentum<'a> {
    fn from(param: SGDParam<'a>) -> Self {
        let (data, grad) = (param.data, param.grad);
        let buffer = ArrayD::zeros(grad.raw_dim());
        Self { data, grad, buffer }
    }
}

impl<'a, T: Penalty> Optimizer<SGDParamWithMomentum<'a>> for SGDWithMomentum<'a, T> {
    fn step(&mut self) {
        let (lr, penalty, momentum, dampening, nesterov, params) = (
            &self.lr,
            &self.penalty,
            &self.momentum,
            &self.dampening,
            &self.nesterov,
            &mut self.params,
        );
        params.par_iter_mut().for_each(|param| {
            Zip::from(&mut param.buffer)
                .and(&param.grad)
                .for_each(|buffer_el, grad_el| {
                    *buffer_el = *buffer_el * *momentum
                        + (grad_el + penalty.penalise(grad_el)) * (1. - dampening)
                });

            let zip = Zip::from(&mut param.data).and(&param.buffer);
            if *nesterov {
                zip.and(&param.grad)
                    .for_each(|data_el, buffer_el, grad_el| {
                        *data_el =
                            -(grad_el + penalty.penalise(grad_el)) * lr + *buffer_el * *momentum
                    });
            } else {
                zip.for_each(|data_el, buffer_el| *data_el = -*buffer_el * *lr);
            }
        });
    }

    fn zero_grad(&mut self) {
        self.params.par_iter_mut().for_each(|param| {
            let grad = &mut param.grad;
            Zip::from(grad).for_each(|grad_el| *grad_el = 0.);
        });
    }
}

impl<'a, T: Penalty> SGDWithMomentum<'a, T> {
    pub fn new(
        parameters: Vec<Param>,
        lr: f32,
        penalty: T,
        momentum: f32,
        dampening: f32,
        nesterov: bool,
    ) -> Self {
        let params = {
            let mut vec = Vec::new();
            for param in parameters {
                vec.push(SGDParamWithMomentum::from(param));
            }
            vec
        };
        Self {
            params,
            lr,
            penalty,
            momentum,
            dampening,
            nesterov,
        }
    }
}
