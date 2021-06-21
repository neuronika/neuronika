//! Implementations of various optimization algorithms and penalty regularizations.
//!
//! Some of the most commonly used methods are already supported, and the interface is linear
//! enough, so that more sophisticated ones can be also easily integrated in the future. The
//! complete list can be found [here](#algorithms).
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
//! the regularization, etc.
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
//! use std::cell::{Cell, RefCell};
//!
//! # use ndarray::{ArrayD, ArrayViewMutD};
//! # struct SGDParam<'a> {
//! #     data: ArrayViewMutD<'a, f32>,
//! #     grad: ArrayViewMutD<'a, f32>,
//! # }
//! struct SGD<'a, T> {
//!     params: RefCell<Vec<SGDParam<'a>>>,
//!     lr: Cell<f32>,
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
//! # use std::cell::{Cell, RefCell};
//! # struct SGD<'a, T> {
//! #     params: RefCell<Vec<SGDParam<'a>>>,
//! #     lr: Cell<f32>,
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
//!     fn step(&self) {
//!         let (lr, penalty) = (self.lr.get(), &self.penalty);
//!
//!         self.params.borrow_mut().par_iter_mut().for_each(|param| {
//!             let (data, grad) = (&mut param.data, &param.grad);
//!
//!             Zip::from(data).and(grad).for_each(|data_el, grad_el| {
//!                 *data_el += -(grad_el + penalty.penalize(data_el)) * lr
//!             });
//!         });
//!     }
//!
//!     fn zero_grad(&self) {
//!         self.params.borrow_mut().par_iter_mut().for_each(|param| {
//!             let grad = &mut param.grad;
//!             Zip::from(grad).for_each(|grad_el| *grad_el = 0.);
//!         });
//!     }
//!
//!     fn get_lr(&self) -> f32 {
//!         self.lr.get()
//!     }
//!
//!     fn set_lr(&self, lr: f32) {
//!         self.lr.set(lr)    
//!     }
//! }
//!
//! // Simple constructor.
//! impl<'a, T: Penalty> SGD<'a, T> {
//!   pub fn new(parameters: Vec<Param>, lr: f32, penalty: T) -> Self {
//!       Self {
//!           params: RefCell::new(Self::build_params(parameters)),
//!           lr: Cell::new(lr),
//!           penalty,
//!       }
//!    }
//! }
//! ```
//!
//! # Algorithms
//!
//! List of all implemented optimizers.
//!
//! * [`Adagrad`] - Implements the Adagrad algorithm.
//!
//! * [`Adam`] - Implements the Adam algorithm.
//!
//! * [`AMSGrad`] - Implements the AMSGrad algorithm.
//!
//! * [`RMSProp`] - Implements the RMSProp algorithm.
//!
//! * [`SGD`] - Implements the stochastic gradient descent algorithm.
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
    fn step(&self);

    /// Zeroes the gradients of all the optimizable parameters.
    fn zero_grad(&self);

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

    /// Returns this optimizer's learning rate.
    fn get_lr(&self) -> f32;

    /// Sets this optimizer's learning rate.
    fn set_lr(&self, lr: f32);
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Penalty Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Penalty trait, defines the penalty regularization's logic.
pub trait Penalty: Send + Sync {
    /// Applies the penatly to an element of the gradient.
    fn penalize(&self, w: &f32) -> f32;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Regularization Structs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    fn penalize(&self, w: &f32) -> f32 {
        2. * self.lambda * w
    }
}

impl Penalty for L1 {
    fn penalize(&self, w: &f32) -> f32 {
        self.lambda * w.signum()
    }
}

impl Penalty for ElasticNet {
    fn penalize(&self, w: &f32) -> f32 {
        self.lambda_l1 * w.signum() + 2. * self.lambda_l2 * w
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimizers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mod adagrad;
mod adam;
mod amsgrad;
mod rmsprop;
mod sgd;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Learning Rate Scheduler Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Learning rate scheduler trait, defines the scheduler's logic.
pub trait LRScheduler {
    /// Updates the learning rate.
    fn step(&mut self);

    /// Returns an immutable reference to the last computed learning rate.
    fn get_last_lr(&self) -> &f32;

    /// Returns an immutable reference to the current learning rate.
    fn get_current_lr(&self) -> &f32;

    /// Returns an immutable reference to the current epoch.
    fn get_current_epoch(&self) -> &usize;

    /// Sets the current epoch.
    fn set_current_epoch(&mut self, epoch: usize);

    /// Prints the update of the learning rate. It should be called after `.step()`.
    fn print_lr(&self) {
        println!(
            "epoch {}: learning rate adjusted to {}",
            self.get_current_epoch() - 1,
            self.get_current_lr()
        );
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LambdaLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Sets the learning rate to the initial lr times a given function.
///
///```text
/// lrₜ = lr₀ * lr_fn(epoch)
///```
pub struct LambdaLR<'a, T: Optimizer, F: Fn(usize) -> f32> {
    optimizer: &'a T,
    lr_fn: F,
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
    initial_lr: f32,
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> LambdaLR<'a, T, F> {
    /// Creates a new LambdaLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `lr_fn` - function which computes a multiplicative factor given an `usize` parameter
    /// epoch.
    pub fn new(optimizer: &'a T, lr_fn: F) -> Self {
        let current_lr = optimizer.get_lr();
        Self {
            optimizer,
            lr_fn,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
            initial_lr: current_lr,
        }
    }
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> LRScheduler for LambdaLR<'a, T, F> {
    fn step(&mut self) {
        self.current_epoch += 1;

        self.last_lr = self.current_lr;
        self.current_lr = self.initial_lr * (self.lr_fn)(self.current_epoch);
        self.optimizer.set_lr(self.current_lr);
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiplicativeLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Multiplies the learning rate by the factor given in the specified function.
///
///```text
/// lrₜ = lrₜ₋₁ * lr_fn(epoch)
///```
pub struct MultiplicativeLR<'a, T: Optimizer, F: Fn(usize) -> f32> {
    optimizer: &'a T,
    lr_fn: F,
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> MultiplicativeLR<'a, T, F> {
    /// Creates a new MultiplicativeLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `lr_fn` - function which computes a multiplicative factor given an `usize` parameter
    /// epoch.
    pub fn new(optimizer: &'a T, lr_fn: F) -> Self {
        let current_lr = optimizer.get_lr();
        Self {
            optimizer,
            lr_fn,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
        }
    }
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> LRScheduler for MultiplicativeLR<'a, T, F> {
    fn step(&mut self) {
        self.current_epoch += 1;

        self.last_lr = self.current_lr;
        self.current_lr *= (self.lr_fn)(self.current_epoch);
        self.optimizer.set_lr(self.current_lr);
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StepLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Decays the learning rate by `gamma` every `step_size` epochs.
///
///```text
/// lrₜ = lrₜ₋₁ * gamma if t mod step_size == 0 else lrₜ₋₁
///```
pub struct StepLR<'a, T: Optimizer> {
    optimizer: &'a T,
    gamma: f32,
    step_size: usize,
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
}

impl<'a, T: Optimizer> StepLR<'a, T> {
    /// Creates a new StepLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `step_size` - period of learning rate decay.
    ///
    /// * `gamma` - multiplicative factor for the learning rate decay.
    pub fn new(optimizer: &'a T, step_size: usize, gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma,
            step_size,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
        }
    }
}

impl<'a, T: Optimizer> LRScheduler for StepLR<'a, T> {
    fn step(&mut self) {
        self.current_epoch += 1;

        if self.current_epoch.rem_euclid(self.step_size) == 0 {
            self.last_lr = self.current_lr;
            self.current_lr = self.last_lr * self.gamma;
            self.optimizer.set_lr(self.current_lr);
        }
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiStepLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Decays the learning rate by gamma once the number of epoch reaches one of the specified
/// milestones.
///
///```text
/// lrₜ = lrₜ₋₁ * gamma if t is a milestone else lrₜ₋₁
///```
pub struct MultiStepLR<'a, T: Optimizer, const N: usize> {
    optimizer: &'a mut T,
    gamma: f32,
    milestones: [usize; N],
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
}

impl<'a, T: Optimizer, const N: usize> MultiStepLR<'a, T, N> {
    /// Creates a new MultiStepLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `milestones` - list of epoch indices. Must be increasing.
    ///
    /// * `gamma` - multiplicative factor for the learning rate decay.
    pub fn new(optimizer: &'a mut T, milestones: [usize; N], gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma,
            milestones,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
        }
    }
}

impl<'a, T: Optimizer, const N: usize> LRScheduler for MultiStepLR<'a, T, N> {
    fn step(&mut self) {
        self.current_epoch += 1;

        if self
            .milestones
            .iter()
            .any(|milestone| *milestone == self.current_epoch)
        {
            self.last_lr = self.current_lr;
            self.current_lr = self.last_lr * self.gamma;
            self.optimizer.set_lr(self.current_lr);
        }
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ExponentialLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Decays the learning rate by `gamma` every epoch.
///
///```text
/// lrₜ = lrₜ₋₁ * gamma
///```
pub struct ExponentialLR<'a, T: Optimizer> {
    optimizer: &'a T,
    gamma: f32,
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
}

impl<'a, T: Optimizer> ExponentialLR<'a, T> {
    /// Creates a new ExponentialLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `gamma` - multiplicative factor for the learning rate decay.
    pub fn new(optimizer: &'a T, gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
        }
    }
}

impl<'a, T: Optimizer> LRScheduler for ExponentialLR<'a, T> {
    fn step(&mut self) {
        self.current_epoch += 1;

        self.last_lr = self.current_lr;
        self.current_lr = self.last_lr * self.gamma;
        self.optimizer.set_lr(self.current_lr);
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}
