use super::graph::{Param, Parameters};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait Optimizer {
    fn step(&self, parameters: &Parameters);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Penalty Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait Penalty {
    fn penalise(&self, w: &f32) -> f32;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regularizations Struct ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct L2 {
    lambda: f32,
}
pub struct L1 {
    lambda: f32,
}

pub struct ElasticNet {
    lambda_l1: f32,
    lambda_l2: f32,
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Penalty Trait Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

// TODO: implement momentum
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stochastic Gradient Descent ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct SGD<T>
where
    T: Penalty,
{
    learning_rate: f32,
    penalty: T,
    // momentum: f32
}

impl<T> SGD<T>
where
    T: Penalty,
{
    pub fn new(learning_rate: f32, penalty: T) -> Self {
        Self {
            learning_rate,
            penalty,
        }
    }

    fn weight_update<D: ndarray::Dimension>(&self, param: &Param<D>) {
        let (mut data, grad) = (param.data_mut(), param.grad());
        let (lr, penalty) = (&self.learning_rate, &self.penalty);
        ndarray::Zip::from(&mut *data)
            .and(&*grad)
            .for_each(|data_el, grad_el| {
                *data_el = -grad_el * lr + Penalty::penalise(penalty, grad_el)
            });
    }
}

impl<T> Optimizer for SGD<T>
where
    T: Penalty,
{
    fn step(&self, parameters: &Parameters) {
        for parameter in parameters.get_oned() {
            self.weight_update(parameter)
        }
        for parameter in parameters.get_twod() {
            self.weight_update(parameter)
        }
        for parameter in parameters.get_threed() {
            self.weight_update(parameter)
        }
        for parameter in parameters.get_fourd() {
            self.weight_update(parameter)
        }
        for parameter in parameters.get_fived() {
            self.weight_update(parameter)
        }
        for parameter in parameters.get_sixd() {
            self.weight_update(parameter)
        }
        for parameter in parameters.get_dynd() {
            self.weight_update(parameter)
        }
    }
}
