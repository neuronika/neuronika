use super::variable::parameters::{Param, Parameters};
use crate::{variable::node::Gradient, Input, InputBackward};
use ndarray::{Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use std::rc::Rc;

pub trait FromParam<D: Dimension, T: Penalty> {
    fn from_param(parameter: Param<D>, learning_rate: f32, penalty: T) -> Box<dyn OptimParam>;
}
pub trait OptimParam {
    fn update(&self);
    fn zero_grad(&self);
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait Optimizer {
    fn step(&self);
    fn zero_grad(&self);
}

pub struct OptimParameters {
    params: Vec<Box<dyn OptimParam>>,
}

impl OptimParameters {
    fn new<
        T: Penalty,
        A: FromParam<Ix1, T>,
        B: FromParam<Ix2, T>,
        C: FromParam<Ix3, T>,
        D: FromParam<Ix4, T>,
        E: FromParam<Ix5, T>,
        F: FromParam<Ix6, T>,
        G: FromParam<IxDyn, T>,
    >(
        parameters: &Parameters,
        penalty: T,
        lr: f32,
    ) -> Self {
        let (p_oned, p_twod, p_threed, p_fourd, p_fived, p_sixd, p_dynd) = parameters.clone().get();
        let mut params: Vec<Box<dyn OptimParam>> = Vec::with_capacity(parameters.len());

        for param in p_oned {
            params.push(A::from_param(param, lr, penalty.clone()));
        }
        for param in p_twod {
            params.push(B::from_param(param, lr, penalty.clone()));
        }
        for param in p_threed {
            params.push(C::from_param(param, lr, penalty.clone()));
        }
        for param in p_fourd {
            params.push(D::from_param(param, lr, penalty.clone()));
        }
        for param in p_fived {
            params.push(E::from_param(param, lr, penalty.clone()));
        }
        for param in p_sixd {
            params.push(F::from_param(param, lr, penalty.clone()));
        }
        for param in p_dynd {
            params.push(G::from_param(param, lr, penalty.clone()));
        }
        Self { params }
    }
}

macro_rules! make_optimizer {
    ($name:ident, $param:ident) => {
        #[allow(clippy::clippy::upper_case_acronyms)]
        pub struct $name {
            params: OptimParameters,
        }

        impl $name {
            pub fn new<T: Penalty + 'static>(
                params: &Parameters,
                learning_rate: f32,
                penalty: T,
            ) -> Self {
                Self {
                    params: OptimParameters::new::<
                        T,
                        $param<Ix1, T>,
                        $param<Ix2, T>,
                        $param<Ix3, T>,
                        $param<Ix4, T>,
                        $param<Ix5, T>,
                        $param<Ix6, T>,
                        $param<IxDyn, T>,
                    >(params, penalty, learning_rate),
                }
            }
        }

        impl Optimizer for $name {
            fn step(&self) {
                for p in &self.params.params {
                    p.update()
                }
            }

            fn zero_grad(&self) {
                for p in &self.params.params {
                    p.zero_grad()
                }
            }
        }
    };
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Penalty Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait Penalty: Clone {
    fn penalise(&self, w: &f32) -> f32;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regularizations Struct ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Clone)]
pub struct L2 {
    lambda: f32,
}
#[derive(Clone)]
pub struct L1 {
    lambda: f32,
}

#[derive(Clone)]
pub struct ElasticNet {
    lambda_l1: f32,
    lambda_l2: f32,
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Penalty Trait Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl Penalty for L2 {
    fn penalise(&self, w: &f32) -> f32 {
        2. * self.lambda * w
    }
}

impl Penalty for L1 {
    fn penalise(&self, w: &f32) -> f32 {
        if *w != 0. {
            self.lambda * w.signum()
        } else {
            0.
        }
    }
}

impl Penalty for ElasticNet {
    fn penalise(&self, w: &f32) -> f32 {
        if *w != 0. {
            self.lambda_l1 * w.signum() + 2. * self.lambda_l2 * w
        } else {
            0.
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stochastic Gradient Descent ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct SGDParam<D: Dimension, T: Penalty> {
    input: Rc<Input<D>>,
    input_diff: Rc<InputBackward<D>>,
    penalty: T,
    learning_rate: f32,
}

impl<D: Dimension, T: Penalty> OptimParam for SGDParam<D, T> {
    fn update(&self) {
        let (mut data, grad) = (self.input.data_mut(), self.input_diff.gradient());
        let (lr, penalty) = (&self.learning_rate, &self.penalty);
        ndarray::Zip::from(&mut *data)
            .and(&*grad)
            .for_each(|data_el, grad_el| *data_el = -grad_el * *lr + penalty.penalise(grad_el));
    }

    fn zero_grad(&self) {
        self.input_diff.zero_grad()
    }
}

impl<D: Dimension + 'static, T: Penalty + 'static> FromParam<D, T> for SGDParam<D, T> {
    fn from_param(parameter: Param<D>, lr: f32, penalty: T) -> Box<dyn OptimParam> {
        let (input, input_diff) = parameter.get();
        Box::new(Self {
            input,
            input_diff,
            learning_rate: lr,
            penalty,
        })
    }
}

make_optimizer!(SDG, SGDParam);
