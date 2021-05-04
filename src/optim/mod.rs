use super::variable::parameters::{Param, Parameters};
use crate::{variable::node::Gradient, Input, InputBackward};
use ndarray::{Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use std::{marker::PhantomData, rc::Rc};

pub trait OptimParam<D: Dimension, T: Penalty> {
    fn input_diff(&self) -> Rc<InputBackward<D>>;
    fn update(&self);
    fn zero_grad(&self) {
        self.input_diff().zero_grad()
    }
    fn from_param(parameter: Param<D>, learning_rate: f32, penalty: T) -> Self;
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait Optimizer<T: Penalty> {
    type OneD: OptimParam<Ix1, T>;
    type TwoD: OptimParam<Ix2, T>;
    type ThreeD: OptimParam<Ix3, T>;
    type FourD: OptimParam<Ix4, T>;
    type FiveD: OptimParam<Ix5, T>;
    type SixD: OptimParam<Ix6, T>;
    type DynD: OptimParam<IxDyn, T>;

    fn step(&self) {
        {
            let params = self.params();
            for param in &params.oned {
                param.update();
            }
            for param in &params.twod {
                param.update()
            }
            for param in &params.threed {
                param.update()
            }
            for param in &params.fourd {
                param.update()
            }
            for param in &params.fived {
                param.update()
            }
            for param in &params.sixd {
                param.update()
            }
            for param in &params.dynd {
                param.update()
            }
        }
    }

    fn zero_grad(&self) {
        let params = self.params();
        for param in &params.oned {
            param.zero_grad()
        }
        for param in &params.twod {
            param.zero_grad()
        }
        for param in &params.threed {
            param.zero_grad()
        }
        for param in &params.fourd {
            param.zero_grad()
        }
        for param in &params.fived {
            param.zero_grad()
        }
        for param in &params.sixd {
            param.zero_grad()
        }
        for param in &params.dynd {
            param.zero_grad()
        }
    }

    fn params(
        &self,
    ) -> &OptimParameters<
        Self::OneD,
        Self::TwoD,
        Self::ThreeD,
        Self::FourD,
        Self::FiveD,
        Self::SixD,
        Self::DynD,
        T,
    >;
}

pub struct OptimParameters<OneD, TwoD, ThreeD, FourD, FiveD, SixD, DynD, T>
where
    OneD: OptimParam<Ix1, T>,
    TwoD: OptimParam<Ix2, T>,
    ThreeD: OptimParam<Ix3, T>,
    FourD: OptimParam<Ix4, T>,
    FiveD: OptimParam<Ix5, T>,
    SixD: OptimParam<Ix6, T>,
    DynD: OptimParam<IxDyn, T>,
    T: Penalty,
{
    oned: Vec<OneD>,
    twod: Vec<TwoD>,
    threed: Vec<ThreeD>,
    fourd: Vec<FourD>,
    fived: Vec<FiveD>,
    sixd: Vec<SixD>,
    dynd: Vec<DynD>,
    penalty: PhantomData<T>,
}

impl<OneD, TwoD, ThreeD, FourD, FiveD, SixD, DynD, T>
    OptimParameters<OneD, TwoD, ThreeD, FourD, FiveD, SixD, DynD, T>
where
    OneD: OptimParam<Ix1, T>,
    TwoD: OptimParam<Ix2, T>,
    ThreeD: OptimParam<Ix3, T>,
    FourD: OptimParam<Ix4, T>,
    FiveD: OptimParam<Ix5, T>,
    SixD: OptimParam<Ix6, T>,
    DynD: OptimParam<IxDyn, T>,
    T: Penalty,
{
    fn new(parameters: &Parameters, penalty: T, lr: f32) -> Self {
        let (p_oned, p_twod, p_threed, p_fourd, p_fived, p_sixd, p_dynd) = parameters.clone().get();
        let mut oned = Vec::with_capacity(p_oned.len());
        for param in p_oned {
            oned.push(OneD::from_param(param, lr, penalty.clone()));
        }
        let mut twod = Vec::with_capacity(p_twod.len());
        for param in p_twod {
            twod.push(TwoD::from_param(param, lr, penalty.clone()));
        }
        let mut threed = Vec::with_capacity(p_threed.len());
        for param in p_threed {
            threed.push(ThreeD::from_param(param, lr, penalty.clone()));
        }
        let mut fourd = Vec::with_capacity(p_fourd.len());
        for param in p_fourd {
            fourd.push(FourD::from_param(param, lr, penalty.clone()));
        }
        let mut fived = Vec::with_capacity(p_fived.len());
        for param in p_fived {
            fived.push(FiveD::from_param(param, lr, penalty.clone()));
        }
        let mut sixd = Vec::with_capacity(p_sixd.len());
        for param in p_sixd {
            sixd.push(SixD::from_param(param, lr, penalty.clone()));
        }
        let mut dynd = Vec::with_capacity(p_dynd.len());
        for param in p_dynd {
            dynd.push(DynD::from_param(param, lr, penalty.clone()));
        }
        Self {
            oned,
            twod,
            threed,
            fourd,
            fived,
            sixd,
            dynd,
            penalty: PhantomData,
        }
    }
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

impl<D: Dimension, T: Penalty> OptimParam<D, T> for SGDParam<D, T> {
    fn input_diff(&self) -> Rc<InputBackward<D>> {
        self.input_diff.clone()
    }

    fn update(&self) {
        let (mut data, grad) = (self.input.data_mut(), self.input_diff.gradient());
        let (lr, penalty) = (&self.learning_rate, &self.penalty);
        ndarray::Zip::from(&mut *data)
            .and(&*grad)
            .for_each(|data_el, grad_el| *data_el = -grad_el * *lr + penalty.penalise(grad_el));
    }
    fn from_param(parameter: Param<D>, lr: f32, penalty: T) -> Self {
        let (input, input_diff) = parameter.get();
        Self {
            input,
            input_diff,
            learning_rate: lr,
            penalty,
        }
    }
}
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct SGD<T: Penalty> {
    params: OptimParameters<
        SGDParam<Ix1, T>,
        SGDParam<Ix2, T>,
        SGDParam<Ix3, T>,
        SGDParam<Ix4, T>,
        SGDParam<Ix5, T>,
        SGDParam<Ix6, T>,
        SGDParam<IxDyn, T>,
        T,
    >,
}

impl<T: Penalty> SGD<T> {
    pub fn new(params: &Parameters, learning_rate: f32, penalty: T) -> Self {
        Self {
            params: OptimParameters::new(params, penalty, learning_rate),
        }
    }
}

impl<T: Penalty> Optimizer<T> for SGD<T> {
    type OneD = SGDParam<Ix1, T>;
    type TwoD = SGDParam<Ix2, T>;
    type ThreeD = SGDParam<Ix3, T>;
    type FourD = SGDParam<Ix4, T>;
    type FiveD = SGDParam<Ix5, T>;
    type SixD = SGDParam<Ix6, T>;
    type DynD = SGDParam<IxDyn, T>;

    fn params(
        &self,
    ) -> &OptimParameters<
        Self::OneD,
        Self::TwoD,
        Self::ThreeD,
        Self::FourD,
        Self::FiveD,
        Self::SixD,
        Self::DynD,
        T,
    > {
        &self.params
    }
}
