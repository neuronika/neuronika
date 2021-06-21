use super::{Optimizer, Param, Penalty};
use ndarray::{ArrayD, ArrayViewMutD, Zip};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::cell::{Cell, RefCell};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stochastic Gradient Descent ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::upper_case_acronyms)]
/// The **Stochastic Gradient Descent** optimizer.
pub struct SGD<'a, T> {
    params: RefCell<Vec<SGDParam<'a>>>,
    lr: Cell<f32>,
    penalty: T,
}

#[allow(clippy::upper_case_acronyms)]
/// A parameter used by the *SDG* optimizer.
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

impl<'a, T: Penalty> Optimizer for SGD<'a, T> {
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
    /// Creates a new *SGD* optmizer.
    ///
    /// # Arguments
    ///
    /// * `params` - vector of [`Param`] to optimize.
    ///
    /// * `lr` - learning rate.
    ///
    /// * `penalty` - penalty regularization.
    pub fn new(parameters: Vec<Param>, lr: f32, penalty: T) -> Self {
        let params = RefCell::new(Self::build_params(parameters));
        let lr = Cell::new(lr);

        Self {
            params,
            lr,
            penalty,
        }
    }

    /// Transforms this *SGD* optimizer in the *momentum* version of the algorithm.
    ///
    /// Nesterov momentum is based on the formula from
    /// [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf).
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
            momentum,
            dampening,
            nesterov,
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stochastic Gradient Descent with Momentum ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[allow(clippy::upper_case_acronyms)]
/// The momentum variant of the *Stochastic Gradient Descent* optimizer.
pub struct SGDWithMomentum<'a, T> {
    params: RefCell<Vec<SGDWithMomentumParam<'a>>>,
    lr: Cell<f32>,
    penalty: T,
    momentum: f32,
    dampening: f32,
    nesterov: bool,
}

#[allow(clippy::upper_case_acronyms)]
/// A parameter used by the *SDG* with momentum optimizer.
pub struct SGDWithMomentumParam<'a> {
    data: ArrayViewMutD<'a, f32>,
    grad: ArrayViewMutD<'a, f32>,
    buffer: ArrayD<f32>,
}

impl<'a> From<Param> for SGDWithMomentumParam<'a> {
    fn from(param: Param) -> Self {
        let (data, grad) = param.get();
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

impl<'a, T: Penalty> Optimizer for SGDWithMomentum<'a, T> {
    type ParamRepr = SGDWithMomentumParam<'a>;

    fn step(&self) {
        let (lr, penalty, momentum, dampening, nesterov, mut params) = (
            self.lr.get(),
            &self.penalty,
            &self.momentum,
            &self.dampening,
            &self.nesterov,
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
    /// Creates a new *SGD* optmizer.
    ///
    /// # Arguments
    ///
    /// * `params` - vector of [`Param`] to optimize.
    ///
    /// * `lr` - learning rate.
    ///
    /// * `penalty` - penalty regularization.
    ///
    /// * `momentum` - the momentum factor.
    ///
    /// * `dampening` - the dampening factor for momentum.
    ///
    /// * `nesterov` - enables *Nesterov* momentum.
    ///
    /// Nesterov momentum is based on the formula from
    /// [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf).
    pub fn new(
        parameters: Vec<Param>,
        lr: f32,
        penalty: T,
        momentum: f32,
        dampening: f32,
        nesterov: bool,
    ) -> Self {
        let params = RefCell::new(Self::build_params(parameters));
        let lr = Cell::new(lr);

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
