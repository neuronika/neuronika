//! Implementations of various optimization algorithms and penalty regularizations.
//!
//! Some of the most commonly used methods are already supported, and the interface is linear
//! enough, so that more sophisticated ones can be also easily integrated in the future.
//!
//! An optimizer holds a state, in the form of a *representation*, for each of the parameters to
//! optimize and it updates such parameters accordingly to the computed gradient and the state
//! itself.
//!
//! # Using an optimizer
//!
//! The first step to be performed in order to use any optimizer is to construct it.
//!
//! ## Constructing it
//!
//! To construct an optimizer you have to give it a vector of [`Param`](struct@Param) referring to
//! the parameters to optimize and to pass optimizer-specific setting such as the learning rate,
//! the regulatization, etc.
//!
//! ```
//! # use neuronika::Param;
//! # use neuronika::nn::{ModelStatus, Linear, Learnable};
//! # struct MLP {
//! #     lin1: Linear,
//! #     lin2: Linear,
//! #     lin3: Linear,
//! #     status: ModelStatus,     
//! # }
//! # impl MLP {
//! #     // Basic constructor.
//! #     fn new() -> Self {
//! #         let mut status = ModelStatus::default();
//! #
//! #         Self {
//! #            lin1: status.register(Linear::new(25, 30)),
//! #            lin2: status.register(Linear::new(30, 35)),
//! #            lin3: status.register(Linear::new(35, 5)),
//! #            status,
//! #         }
//! #     }
//! #
//! #     fn parameters(&self) -> Vec<Param> {
//! #        self.status.parameters()
//! #     }
//! # }
//! use neuronika;
//! use neuronika::optim::{SGD, Adam, L1, L2};
//!
//! let p = neuronika::rand(5).requires_grad();
//! let q = neuronika::rand(5).requires_grad();
//! let x = neuronika::rand(5);
//!
//! let y = p * x + q;
//! let optim = SGD::new(y.parameters(), 0.01, L1::new(0.05));
//!
//! let model = MLP::new();
//! let model_optim = Adam::new(model.parameters(), 0.01, (0.9, 0.999), L2::new(0.01), 1e-8);
//! ```
//!
//! ## Taking an optimization step
//!
//! All neuronika's optimizer implement a [`.step()`](Optimizer::step()) method that updates the
//! parameters.
//!
//! # Implementing an optimizer
//!
//! Implementing an optimizer in neuronika is quick and simple. The procedure consists in *3* steps:
//!
//! 1. Define its parameter's representation struct and specify how to build it from
//! [`Param`](crate::Param).
//!
//! 2. Define its struct.
//!
//! 3. Implement the [`Optimizer`] trait.
//!
//! Let's go through them by implementing the vanilla version of the stochastic gradient descent.
//!
//! Firstly, we define the SGD parameter's struct and the conversion from `Param`.
//!
//! ```
//! use neuronika::Param;
//! use ndarray::{ArrayD, ArrayViewMutD};
//!
//! struct SGDParam<'a> {
//!     data: ArrayViewMutD<'a, f32>,
//!     grad: ArrayViewMutD<'a, f32>,
//! }
//!
//! impl<'a> From<Param> for SGDParam<'a> {
//!     fn from(param: Param) -> Self {
//!         let (data, grad) = param.get();
//!         Self { data, grad }
//!     }
//! }
//! ```
//!
//! Being a basic optimizer, the `SGDParam` struct will only contain the gradient and the data for
//! each of the parameters to optimize.
//!
//! Nevertheless, do note that an optimizer's parameter representation acts as a container for the
//! additional informations, such as adaptive learning rates and moments of any kind, that may be
//! needed for the learning steps of more complex algorithms.
//!
//! Then, we define the SGD's struct.
//!
//! ```
//! use neuronika::Param;
//! use neuronika::optim::Penalty;
//!
//! # use ndarray::{ArrayD, ArrayViewMutD};
//! # struct SGDParam<'a> {
//! #     data: ArrayViewMutD<'a, f32>,
//! #     grad: ArrayViewMutD<'a, f32>,
//! # }
//! struct SGD<'a, T> {
//!     params: Vec<SGDParam<'a>>,
//!     lr: f32,
//!     penalty: T,
//! }
//! ```
//!
//! Lastly, we implement [`Optimizer`] for `SGD`.
//!
//! ```
//! use ndarray::Zip;
//! use neuronika::optim::Optimizer;
//! use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
//! # use neuronika::Param;
//! # use neuronika::optim::Penalty;
//! # use ndarray::{ArrayD, ArrayViewMutD};
//! # struct SGD<'a, T> {
//! #     params: Vec<SGDParam<'a>>,
//! #     lr: f32,
//! #     penalty: T,
//! # }
//! # struct SGDParam<'a> {
//! #     data: ArrayViewMutD<'a, f32>,
//! #     grad: ArrayViewMutD<'a, f32>,
//! # }
//! # impl<'a> From<Param> for SGDParam<'a> {
//! #     fn from(param: Param) -> Self {
//! #         let (data, grad) = param.get();
//! #         Self { data, grad }
//! #     }
//! # }
//!
//! impl<'a, T: Penalty> Optimizer for SGD<'a, T> {
//!     type ParamRepr = SGDParam<'a>;
//!
//!     fn step(&mut self) {
//!         let (lr, penalty, params) = (&self.lr, &self.penalty, &mut self.params);
//!
//!         params.par_iter_mut().for_each(|param| {
//!             let (data, grad) = (&mut param.data, &param.grad);
//!
//!             Zip::from(data).and(grad).for_each(|data_el, grad_el| {
//!                 *data_el += -(grad_el + penalty.penalise(data_el)) * lr
//!             });
//!         });
//!     }
//!
//!     fn zero_grad(&mut self) {
//!         self.params.par_iter_mut().for_each(|param| {
//!             let grad = &mut param.grad;
//!             Zip::from(grad).for_each(|grad_el| *grad_el = 0.);
//!         });
//!     }
//! }
//!
//! // Simple constructor.
//! impl<'a, T: Penalty> SGD<'a, T> {
//!   pub fn new(parameters: Vec<Param>, lr: f32, penalty: T) -> Self {
//!       Self {
//!           params: Self::build_params(parameters),
//!           lr,
//!           penalty,
//!       }
//!    }
//! }
//! ```
use crate::variable::Param;
pub use adagrad::{Adagrad, AdagradParam};
pub use adam::{Adam, AdamParam};
pub use amsgrad::{AMSGrad, AMSGradParam};
pub use rmsprop::{
    RMSProp, RMSPropCentered, RMSPropCenteredParam, RMSPropCenteredWithMomentum,
    RMSPropCenteredWithMomentumParam, RMSPropParam, RMSPropWithMomentum, RMSPropWithMomentumParam,
};
pub use sgd::{SGDParam, SGDWithMomentum, SGDWithMomentumParam, SGD};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizer Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Optimizer trait, defines the optimizer's logic.
pub trait Optimizer {
    /// The type for the internal representation of the optimizer's parameters.
    type ParamRepr: From<Param>;

    /// Performs a single optimization step.
    fn step(&mut self);

    /// Zeroes the gradients of all the optimizable parameters.
    fn zero_grad(&mut self);

    /// Transforms a vector of parameter representations into a vector of another kind of parameter
    /// representations.
    ///
    /// # Arguments
    ///
    /// `params` - parameter representations.
    fn build_params<U, T: From<U>>(params: Vec<U>) -> Vec<T> {
        let mut vec = Vec::with_capacity(params.len());
        for param in params {
            vec.push(T::from(param));
        }
        vec
    }
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

impl L2 {
    /// Creates a new L2 penalty regularization.
    ///
    /// # Arguments
    ///
    /// `lambda` - weight decay coefficient.
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
}

/// L1 penalty.
pub struct L1 {
    lambda: f32,
}

impl L1 {
    /// Creates a new L1 penalty regularization.
    ///
    /// # Arguments
    ///
    /// `lambda` - L1 regularization coefficient.
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
}
/// ElasticNet regularization, linearly combines the *L1* and *L2* penalties.
pub struct ElasticNet {
    lambda_l1: f32,
    lambda_l2: f32,
}

impl ElasticNet {
    /// Creates a new ElasticNet penalty regularization.
    ///
    /// # Arguments
    ///
    /// * `lambda_l2` - L2 regularization coefficient.
    ///
    /// * `lambda_l1` - L1 regularization coefficient.
    pub fn new(lambda_l1: f32, lambda_l2: f32) -> Self {
        Self {
            lambda_l1,
            lambda_l2,
        }
    }
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
        self.lambda * w.signum()
    }
}

impl Penalty for ElasticNet {
    fn penalise(&self, w: &f32) -> f32 {
        self.lambda_l1 * w.signum() + 2. * self.lambda_l2 * w
    }
}

mod adagrad;
mod adam;
mod amsgrad;
mod rmsprop;
mod sgd;
