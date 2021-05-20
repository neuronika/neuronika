//! Implementations of various optimization algorithms and penalty regularisations.
use crate::variable::Param;
pub use adagrad::Adagrad;
pub use adam::Adam;
pub use amsgrad::AMSGrad;
pub use rmsprop::{RMSProp, RMSPropCentered, RMSPropCenteredWithMomentum, RMSPropWithMomentum};
pub use sgd::{SGDWithMomentum, SGD};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Optimizer trait, defines the optimizer's logic.
pub trait Optimizer<T: From<Param>> {
    /// Performs a single optimization step.
    fn step(&mut self);

    /// Zeroes the gradients of all the optimizable parameters.
    fn zero_grad(&mut self);
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Penalty Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Penalty trait, defines the penalty regularisation's logic.
pub trait Penalty: Send + Sync {
    /// Applies the penatly to an element of the gradient.
    fn penalise(&self, w: &f32) -> f32;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regularizations Struct ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// L2 penalty, also known as *weight decay* or *Tichonov regularization*.
pub struct L2 {
    lambda: f32,
}

/// L1 penalty.
pub struct L1 {
    lambda: f32,
}

/// ElasticNet regularization, linearly combines the *L1* and *L2* penalties.
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

mod adagrad;
mod adam;
mod amsgrad;
mod rmsprop;
mod sgd;
