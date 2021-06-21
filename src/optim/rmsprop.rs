use super::{Optimizer, Param, Penalty};
use ndarray::{ArrayD, ArrayViewMutD, Zip};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::cell::{Cell, RefCell};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RMSProp ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// The **RMSProp** optimizer.
///
/// It was proposed by *G. Hinton* in his
/// [course](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
///
/// The centered version first appears in
/// [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850v5.pdf).
///
/// The implementation here takes the square root of the gradient average before adding
/// *epsilon*. Do note that TensorFlow interchanges these two operations. The effective
/// *learning rate* is thus *lr' / (v.sqrt() + eps)* where *lr'* is the scheduled
/// learning rate and *v* is the weighted moving average of the square gradient.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSProp<'a, T: Penalty> {
    params: RefCell<Vec<RMSPropParam<'a>>>,
    lr: Cell<f32>,
    alpha: f32,
    penalty: T,
    eps: f32,
}

impl<'a, T: Penalty> RMSProp<'a, T> {
    /// Creates a *RMSProp* optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - vector of [`Param`] to optimize.
    ///
    /// * `lr` - learning rate.
    ///
    /// * `alpha` - smoothing constant. A good default value is *0.99*.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-8*.
    pub fn new(params: Vec<Param>, lr: f32, alpha: f32, penalty: T, eps: f32) -> Self {
        let params = RefCell::new(Self::build_params(params));
        let lr = Cell::new(lr);

        Self {
            params,
            lr,
            alpha,
            penalty,
            eps,
        }
    }

    /// Transforms this *RMSProp* optimizer in the *centered RMSProp* version of the algorithm
    /// where the gradient is normalized by an estimation of its variance.
    pub fn centered(self) -> RMSPropCentered<'a, T> {
        let params: RefCell<Vec<RMSPropCenteredParam>> =
            RefCell::new(Self::build_params(self.params.into_inner()));
        let (lr, alpha, penalty, eps) = (self.lr, self.alpha, self.penalty, self.eps);

        RMSPropCentered {
            params,
            lr,
            alpha,
            penalty,
            eps,
        }
    }

    /// Transforms this *RMSProp* optimizer in the *momentum* version of the algorithm.
    ///
    /// # Arguments
    ///
    /// `momentum` - the momentum factor.
    pub fn with_momentum(self, momentum: f32) -> RMSPropWithMomentum<'a, T> {
        let params: RefCell<Vec<RMSPropWithMomentumParam>> =
            RefCell::new(Self::build_params(self.params.into_inner()));
        let (lr, alpha, penalty, eps) = (self.lr, self.alpha, self.penalty, self.eps);

        RMSPropWithMomentum {
            params,
            lr,
            alpha,
            penalty,
            eps,
            momentum,
        }
    }

    /// Transforms this *RMSProp* optimizer in the *centered RMSProp* version of the algorithm
    /// with momentum.
    ///
    /// # Arguments
    ///
    /// `momentum` - the momentum factor.
    pub fn centered_with_momentum(self, momentum: f32) -> RMSPropCenteredWithMomentum<'a, T> {
        let params: RefCell<Vec<RMSPropCenteredWithMomentumParam>> =
            RefCell::new(Self::build_params(self.params.into_inner()));
        let (lr, alpha, penalty, eps) = (self.lr, self.alpha, self.penalty, self.eps);

        RMSPropCenteredWithMomentum {
            params,
            lr,
            alpha,
            penalty,
            eps,
            momentum,
        }
    }
}

/// A parameter used by the *RMSProp* optimizer.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSPropParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    step: usize,
    square_avg: ArrayD<f32>,
}

impl<'a> From<Param> for RMSPropParam<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
        let step = 0;
        let square_avg = ArrayD::zeros(grad.raw_dim());

        Self {
            data,
            grad,
            step,
            square_avg,
        }
    }
}

impl<'a, T: Penalty> Optimizer for RMSProp<'a, T> {
    type ParamRepr = RMSPropParam<'a>;

    fn step(&self) {
        let (mut params, lr, alpha, penalty, eps) = (
            self.params.borrow_mut(),
            self.lr.get(),
            &self.alpha,
            &self.penalty,
            &self.eps,
        );

        params.par_iter_mut().for_each(|param| {
            let (step, square_avg) = (&mut param.step, &mut param.square_avg);

            *step += 1;

            let mut p_grad = param.grad.to_owned();
            Zip::from(&mut p_grad)
                .and(&param.data)
                .for_each(|p_grad_el, data_el| *p_grad_el += penalty.penalize(data_el));

            Zip::from(square_avg)
                .and(&p_grad)
                .for_each(|square_avg_el, p_grad_el| {
                    *square_avg_el += *square_avg_el * *alpha + p_grad_el * p_grad_el * (1. - alpha)
                });

            Zip::from(&mut param.data)
                .and(&p_grad)
                .and(&param.square_avg)
                .for_each(|data_el, p_grad_el, square_avg_el| {
                    *data_el += -p_grad_el / (square_avg_el.sqrt() + eps) * lr
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RMSPropWithMomentum ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// The *RMSProp* optimizer with *momentum*.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSPropWithMomentum<'a, T: Penalty> {
    params: RefCell<Vec<RMSPropWithMomentumParam<'a>>>,
    lr: Cell<f32>,
    alpha: f32,
    penalty: T,
    eps: f32,
    momentum: f32,
}

/// A parameter used by the *RMSProp* optimizer with momentum.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSPropWithMomentumParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    step: usize,
    square_avg: ArrayD<f32>,
    buffer: ArrayD<f32>,
}

impl<'a, T: Penalty> RMSPropWithMomentum<'a, T> {
    /// Creates a *RMSProp* optimizer with *momentum*.
    ///
    /// # Arguments
    ///
    /// * `params` - vector of [`Param`] to optimize.
    ///
    /// * `lr` - learning rate.
    ///
    /// * `alpha` - smoothing constant. A good default value is *0.99*.
    ///
    /// * `momentum` - momentum factor.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-8*.
    pub fn new(
        params: Vec<Param>,
        lr: f32,
        alpha: f32,
        momentum: f32,
        penalty: T,
        eps: f32,
    ) -> Self {
        let params = RefCell::new(Self::build_params(params));
        let lr = Cell::new(lr);

        Self {
            params,
            lr,
            alpha,
            penalty,
            eps,
            momentum,
        }
    }

    /// Transofrms this *RMSProp* optimizer in the *centered* variant with *momentum*.
    pub fn centered(self) -> RMSPropCenteredWithMomentum<'a, T> {
        let params: RefCell<Vec<RMSPropCenteredWithMomentumParam>> =
            RefCell::new(Self::build_params(self.params.into_inner()));
        let (lr, alpha, momentum, penalty, eps) =
            (self.lr, self.alpha, self.momentum, self.penalty, self.eps);

        RMSPropCenteredWithMomentum {
            params,
            lr,
            alpha,
            penalty,
            eps,
            momentum,
        }
    }
}

impl<'a> From<Param> for RMSPropWithMomentumParam<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
        let step = 0;
        let (square_avg, buffer) = (ArrayD::zeros(grad.raw_dim()), ArrayD::zeros(grad.raw_dim()));
        Self {
            data,
            grad,
            step,
            square_avg,
            buffer,
        }
    }
}

impl<'a> From<RMSPropParam<'a>> for RMSPropWithMomentumParam<'a> {
    fn from(param: RMSPropParam<'a>) -> Self {
        let (data, grad, step, square_avg) = (param.data, param.grad, param.step, param.square_avg);
        let buffer = ArrayD::zeros(grad.raw_dim());

        Self {
            data,
            grad,
            step,
            square_avg,
            buffer,
        }
    }
}

impl<'a, T: Penalty> Optimizer for RMSPropWithMomentum<'a, T> {
    type ParamRepr = RMSPropWithMomentumParam<'a>;

    fn step(&self) {
        let (mut params, lr, alpha, penalty, eps, momentum) = (
            self.params.borrow_mut(),
            self.lr.get(),
            &self.alpha,
            &self.penalty,
            &self.eps,
            &self.momentum,
        );

        params.par_iter_mut().for_each(|param| {
            let (step, square_avg, buffer) =
                (&mut param.step, &mut param.square_avg, &mut param.buffer);

            *step += 1;

            let mut p_grad = param.grad.to_owned();
            Zip::from(&mut p_grad)
                .and(&param.data)
                .for_each(|p_grad_el, data_el| *p_grad_el += penalty.penalize(data_el));

            Zip::from(square_avg)
                .and(&p_grad)
                .for_each(|square_avg_el, p_grad_el| {
                    *square_avg_el += *square_avg_el * *alpha + p_grad_el * p_grad_el * (1. - alpha)
                });

            Zip::from(buffer)
                .and(&p_grad)
                .and(&mut param.square_avg)
                .for_each(|buffer_el, p_grad_el, square_avg_el| {
                    *buffer_el = *buffer_el * *momentum + p_grad_el / (square_avg_el.sqrt() + eps)
                });

            Zip::from(&mut param.data)
                .and(&param.buffer)
                .for_each(|data_el, buffer_el| *data_el += -buffer_el * lr);
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RMSPropCentered ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// The *RMSProp* optimizer in its *centered* variant.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSPropCentered<'a, T: Penalty> {
    params: RefCell<Vec<RMSPropCenteredParam<'a>>>,
    lr: Cell<f32>,
    alpha: f32,
    penalty: T,
    eps: f32,
}

impl<'a, T: Penalty> RMSPropCentered<'a, T> {
    /// Creates a *centered RMSProp* optimizer.
    ///
    /// # Arguments
    ///
    /// * `params` - vector of [`Param`] to optimize.
    ///
    /// * `lr` - learning rate.
    ///
    /// * `alpha` - smoothing constant. A good default value is *0.99*.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-8*.
    pub fn new(params: Vec<Param>, lr: f32, alpha: f32, penalty: T, eps: f32) -> Self {
        let params = RefCell::new(Self::build_params(params));
        let lr = Cell::new(lr);

        Self {
            params,
            lr,
            alpha,
            penalty,
            eps,
        }
    }

    /// Transforms this *centered RMSProp* optimizer in the centered with momentum variant.
    ///
    /// # Arguments
    ///
    /// `momentum` - momentum factor.
    pub fn with_momentum(self, momentum: f32) -> RMSPropCenteredWithMomentum<'a, T> {
        let params: RefCell<Vec<RMSPropCenteredWithMomentumParam>> =
            RefCell::new(Self::build_params(self.params.into_inner()));
        let (lr, alpha, penalty, eps) = (self.lr, self.alpha, self.penalty, self.eps);

        RMSPropCenteredWithMomentum {
            params,
            lr,
            alpha,
            penalty,
            eps,
            momentum,
        }
    }
}

/// A parameter used by the *centered RMSProp* optimizer.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSPropCenteredParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    step: usize,
    square_avg: ArrayD<f32>,
    grad_avg: ArrayD<f32>,
}

impl<'a> From<Param> for RMSPropCenteredParam<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
        let step = 0;
        let (square_avg, grad_avg) = (ArrayD::zeros(grad.raw_dim()), ArrayD::zeros(grad.raw_dim()));

        Self {
            data,
            grad,
            step,
            square_avg,
            grad_avg,
        }
    }
}

impl<'a> From<RMSPropParam<'a>> for RMSPropCenteredParam<'a> {
    fn from(param: RMSPropParam<'a>) -> Self {
        let (data, grad, step, square_avg) = (param.data, param.grad, param.step, param.square_avg);
        let grad_avg = ArrayD::zeros(grad.raw_dim());

        Self {
            data,
            grad,
            step,
            square_avg,
            grad_avg,
        }
    }
}

impl<'a, T: Penalty> Optimizer for RMSPropCentered<'a, T> {
    type ParamRepr = RMSPropCenteredParam<'a>;

    fn step(&self) {
        let (mut params, lr, alpha, penalty, eps) = (
            self.params.borrow_mut(),
            self.lr.get(),
            &self.alpha,
            &self.penalty,
            &self.eps,
        );

        params.par_iter_mut().for_each(|param| {
            let (step, square_avg, grad_avg) =
                (&mut param.step, &mut param.square_avg, &mut param.grad_avg);

            *step += 1;

            let mut p_grad = param.grad.to_owned();
            Zip::from(&mut p_grad)
                .and(&param.data)
                .for_each(|p_grad_el, data_el| *p_grad_el += penalty.penalize(data_el));

            Zip::from(square_avg)
                .and(&p_grad)
                .for_each(|square_avg_el, p_grad_el| {
                    *square_avg_el += *square_avg_el * *alpha + p_grad_el * p_grad_el * (1. - alpha)
                });

            Zip::from(grad_avg)
                .and(&p_grad)
                .for_each(|grad_avg_el, p_grad_el| {
                    *grad_avg_el = *grad_avg_el * *alpha + p_grad_el * (1. - alpha)
                });

            Zip::from(&mut param.data)
                .and(&p_grad)
                .and(&param.square_avg)
                .and(&param.grad_avg)
                .for_each(|data_el, p_grad_el, square_avg_el, grad_avg_el| {
                    *data_el += -p_grad_el
                        / ((square_avg_el + (-grad_avg_el * grad_avg_el)).sqrt() + eps)
                        * lr
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ RMSPropCenteredWithMomentum ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// The *centered RMSProp* optimizer with *momentum*.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSPropCenteredWithMomentum<'a, T: Penalty> {
    params: RefCell<Vec<RMSPropCenteredWithMomentumParam<'a>>>,
    lr: Cell<f32>,
    alpha: f32,
    penalty: T,
    eps: f32,
    momentum: f32,
}

impl<'a, T: Penalty> RMSPropCenteredWithMomentum<'a, T> {
    /// Creates a *centered RMSProp* optimizer with *momentum*.
    ///
    /// # Arguments
    ///
    /// * `params` - vector of [`Param`] to optimize.
    ///
    /// * `lr` - learning rate.
    ///
    /// * `alpha` - smoothing constant. A good default value is *0.99*.
    ///
    /// * `momentum` - momentum factor.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `eps` - small constant for numerical stability. A good default value is *1e-8*.
    pub fn new(
        params: Vec<Param>,
        lr: f32,
        alpha: f32,
        momentum: f32,
        penalty: T,
        eps: f32,
    ) -> Self {
        let params = RefCell::new(Self::build_params(params));
        let lr = Cell::new(lr);

        Self {
            params,
            lr,
            alpha,
            penalty,
            eps,
            momentum,
        }
    }
}

/// A parameter used by the *centered RMSProp* optimizer with *momentum*.
#[allow(clippy::upper_case_acronyms)]
pub struct RMSPropCenteredWithMomentumParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    step: usize,
    square_avg: ArrayD<f32>,
    grad_avg: ArrayD<f32>,
    buffer: ArrayD<f32>,
}

impl<'a> From<Param> for RMSPropCenteredWithMomentumParam<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
        let step = 0;
        let (square_avg, grad_avg, buffer) = (
            ArrayD::zeros(grad.raw_dim()),
            ArrayD::zeros(grad.raw_dim()),
            ArrayD::zeros(grad.raw_dim()),
        );
        Self {
            data,
            grad,
            step,
            square_avg,
            grad_avg,
            buffer,
        }
    }
}

impl<'a> From<RMSPropParam<'a>> for RMSPropCenteredWithMomentumParam<'a> {
    fn from(param: RMSPropParam<'a>) -> Self {
        let (data, grad, step, square_avg) = (param.data, param.grad, param.step, param.square_avg);
        let (grad_avg, buffer) = (ArrayD::zeros(grad.raw_dim()), ArrayD::zeros(grad.raw_dim()));

        Self {
            data,
            grad,
            step,
            square_avg,
            grad_avg,
            buffer,
        }
    }
}

impl<'a> From<RMSPropCenteredParam<'a>> for RMSPropCenteredWithMomentumParam<'a> {
    fn from(param: RMSPropCenteredParam<'a>) -> Self {
        let (data, grad, step, square_avg, grad_avg) = (
            param.data,
            param.grad,
            param.step,
            param.square_avg,
            param.grad_avg,
        );
        let buffer = ArrayD::zeros(grad.raw_dim());

        Self {
            data,
            grad,
            step,
            square_avg,
            grad_avg,
            buffer,
        }
    }
}

impl<'a> From<RMSPropWithMomentumParam<'a>> for RMSPropCenteredWithMomentumParam<'a> {
    fn from(param: RMSPropWithMomentumParam<'a>) -> Self {
        let (data, grad, step, square_avg, buffer) = (
            param.data,
            param.grad,
            param.step,
            param.square_avg,
            param.buffer,
        );
        let grad_avg = ArrayD::zeros(grad.raw_dim());

        Self {
            data,
            grad,
            step,
            square_avg,
            grad_avg,
            buffer,
        }
    }
}

impl<'a, T: Penalty> Optimizer for RMSPropCenteredWithMomentum<'a, T> {
    type ParamRepr = RMSPropCenteredWithMomentumParam<'a>;
    fn step(&self) {
        let (mut params, lr, alpha, penalty, eps, momentum) = (
            self.params.borrow_mut(),
            self.lr.get(),
            &self.alpha,
            &self.penalty,
            &self.eps,
            &self.momentum,
        );

        params.par_iter_mut().for_each(|param| {
            let (step, square_avg, grad_avg, buffer) = (
                &mut param.step,
                &mut param.square_avg,
                &mut param.grad_avg,
                &mut param.buffer,
            );

            *step += 1;

            let mut p_grad = param.grad.to_owned();
            Zip::from(&mut p_grad)
                .and(&param.data)
                .for_each(|p_grad_el, data_el| *p_grad_el += penalty.penalize(data_el));

            Zip::from(square_avg)
                .and(&p_grad)
                .for_each(|square_avg_el, p_grad_el| {
                    *square_avg_el += *square_avg_el * *alpha + p_grad_el * p_grad_el * (1. - alpha)
                });

            Zip::from(grad_avg)
                .and(&p_grad)
                .for_each(|grad_avg_el, p_grad_el| {
                    *grad_avg_el = *grad_avg_el * *alpha + p_grad_el * (1. - alpha)
                });

            Zip::from(buffer)
                .and(&p_grad)
                .and(&param.square_avg)
                .and(&param.grad_avg)
                .for_each(|buffer_el, p_grad_el, square_avg_el, grad_avg_el| {
                    *buffer_el = *buffer_el * *momentum
                        + p_grad_el / ((square_avg_el + (-grad_avg_el * grad_avg_el)).sqrt() + eps)
                });

            Zip::from(&mut param.data)
                .and(&param.buffer)
                .for_each(|data_el, buffer_el| *data_el += -buffer_el * lr);
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
