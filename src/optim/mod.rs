// //! Implementations of various optimization algorithms and penalty regularizations.
// //!
// //! Some of the most commonly used methods are already supported, and the interface is linear
// //! enough, so that more sophisticated ones can be also easily integrated in the future. The
// //! complete list can be found [here](#algorithms).
// //!
// //! An optimizer holds a state, in the form of a *representation*, for each of the parameters to
// //! optimize and it updates them accordingly to both their gradient and the state itself.
// //!
// //! # Using an optimizer
// //!
// //! The first step to be performed in order to use any optimizer is to construct it.
// //!
// //! ## Constructing it
// //!
// //! To construct an optimizer you have to pass it a vector of [`Param`](struct@Param) referring to
// //! the parameters you whish to optimize. Depending on the kind of optimizer you may also need to
// //! pass several optimizer-specific setting such as the learning rate, the momentum, etc.
// //!
// //! The optimization algorithms provided by neuronika are designed to work both with variables and
// //! neural networks.
// //!
// //! ```
// //! # use neuronika::Param;
// //! # use neuronika::nn::{ModelStatus, Linear, Learnable};
// //! # struct NeuralNetwork {
// //! #     lin1: Linear,
// //! #     lin2: Linear,
// //! #     lin3: Linear,
// //! #     status: ModelStatus,
// //! # }
// //! # impl NeuralNetwork {
// //! #     // Basic constructor.
// //! #     fn new() -> Self {
// //! #         let mut status = ModelStatus::default();
// //! #
// //! #         Self {
// //! #            lin1: status.register(Linear::new(25, 30)),
// //! #            lin2: status.register(Linear::new(30, 35)),
// //! #            lin3: status.register(Linear::new(35, 5)),
// //! #            status,
// //! #         }
// //! #     }
// //! #
// //! #     fn parameters(&self) -> Vec<Param> {
// //! #        self.status.parameters()
// //! #     }
// //! # }
// //! use neuronika;
// //! use neuronika::optim::{SGD, Adam, L1, L2};
// //!
// //! let p = neuronika::rand(5).requires_grad();
// //! let q = neuronika::rand(5).requires_grad();
// //! let x = neuronika::rand(5);
// //!
// //! let y = p * x + q;
// //! let optim = SGD::new(y.parameters(), 0.01, L1::new(0.05));
// //!
// //! let model = NeuralNetwork::new();
// //! let model_optim = Adam::new(model.parameters(), 0.01, (0.9, 0.999), L2::new(0.01), 1e-8);
// //! ```
// //!
// //! ## Taking an optimization step
// //!
// //! All neuronika's optimizer implement a [`.step()`](Optimizer::step()) method that updates the
// //! parameters.
// //!
// //! # Implementing an optimizer
// //!
// //! Implementing an optimizer in neuronika is quick and simple. The procedure consists in *3* steps:
// //!
// //! 1. Define its parameter's representation struct and specify how to build it from
// //! [`Param`](crate::Param).
// //!
// //! 2. Define its struct.
// //!
// //! 3. Implement the [`Optimizer`] trait.
// //!
// //! Let's go through them by implementing the classic version of the stochastic gradient descent.
// //!
// //! Firstly, we define the SGD parameter's struct and the conversion from `Param`.
// //!
// //! ```
// //! use neuronika::Param;
// //! use ndarray::{ArrayD, ArrayViewMutD};
// //!
// //! struct SGDParam<'a> {
// //!     data: ArrayViewMutD<'a, f32>,
// //!     grad: ArrayViewMutD<'a, f32>,
// //! }
// //!
// //! impl<'a> From<Param<'a>> for SGDParam<'a> {
// //!     fn from(param: Param<'a>) -> Self {
// //!         let Param { data, grad } = param;
// //!         Self { data, grad }
// //!     }
// //! }
// //! ```
// //!
// //! Being a basic optimizer, the `SGDParam` struct will only contain the gradient and the data views
// //! for each of the parameters to optimize.
// //!
// //! Nevertheless, do note that an optimizer's parameter representation acts as a container for the
// //! additional information, such as adaptive learning rates and moments of any kind, that may be
// //! needed for the learning steps of more complex algorithms.
// //!
// //! Then, we define the SGD's struct.
// //!
// //! ```
// //! use neuronika::Param;
// //! use neuronika::optim::Penalty;
// //! use std::cell::{Cell, RefCell};
// //!
// //! # use ndarray::{ArrayD, ArrayViewMutD};
// //! # struct SGDParam<'a> {
// //! #     data: ArrayViewMutD<'a, f32>,
// //! #     grad: ArrayViewMutD<'a, f32>,
// //! # }
// //! struct SGD<'a, T> {
// //!     params: RefCell<Vec<SGDParam<'a>>>,
// //!     lr: Cell<f32>,
// //!     penalty: T,
// //! }
// //! ```
// //!
// //! Lastly, we implement [`Optimizer`] for `SGD`.
// //!
// //! ```
// //! use ndarray::Zip;
// //! use neuronika::optim::Optimizer;
// //! use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
// //! # use neuronika::Param;
// //! # use neuronika::optim::Penalty;
// //! # use ndarray::{ArrayD, ArrayViewMutD};
// //! # use std::cell::{Cell, RefCell};
// //! # struct SGD<'a, T> {
// //! #     params: RefCell<Vec<SGDParam<'a>>>,
// //! #     lr: Cell<f32>,
// //! #     penalty: T,
// //! # }
// //! # struct SGDParam<'a> {
// //! #     data: ArrayViewMutD<'a, f32>,
// //! #     grad: ArrayViewMutD<'a, f32>,
// //! # }
// //! # impl<'a> From<Param<'a>> for SGDParam<'a> {
// //! #     fn from(param: Param<'a>) -> Self {
// //! #         let Param { data, grad } = param;
// //! #         Self { data, grad }
// //! #     }
// //! # }
// //!
// //! impl<'a, T: Penalty> Optimizer<'a> for SGD<'a, T> {
// //!     type ParamRepr = SGDParam<'a>;
// //!
// //!     fn step(&self) {
// //!         let (lr, penalty) = (self.lr.get(), &self.penalty);
// //!
// //!         self.params.borrow_mut().par_iter_mut().for_each(|param| {
// //!             let (data, grad) = (&mut param.data, &param.grad);
// //!
// //!             Zip::from(data).and(grad).for_each(|data_el, grad_el| {
// //!                 *data_el += -(grad_el + penalty.penalize(data_el)) * lr
// //!             });
// //!         });
// //!     }
// //!
// //!     fn zero_grad(&self) {
// //!         self.params.borrow_mut().par_iter_mut().for_each(|param| {
// //!             let grad = &mut param.grad;
// //!             Zip::from(grad).for_each(|grad_el| *grad_el = 0.);
// //!         });
// //!     }
// //!
// //!     fn get_lr(&self) -> f32 {
// //!         self.lr.get()
// //!     }
// //!
// //!     fn set_lr(&self, lr: f32) {
// //!         self.lr.set(lr)
// //!     }
// //! }
// //!
// //! /// Simple constructor.
// //! impl<'a, T: Penalty> SGD<'a, T> {
// //!   pub fn new(parameters: Vec<Param<'a>>, lr: f32, penalty: T) -> Self {
// //!       Self {
// //!           params: RefCell::new(Self::build_params(parameters)),
// //!           lr: Cell::new(lr),
// //!           penalty,
// //!       }
// //!    }
// //! }
// //! ```
// //!
// //! # Adjusting the learning rate
// //!
// //! The [`lr_scheduler`] module provides several methods to adjust the learning rate based on the
// //! number of epochs.
// //!
// //! # Algorithms
// //!
// //! List of all implemented optimizers.
// //!
// //! * [`Adagrad`] - Implements the Adagrad algorithm.
// //!
// //! * [`Adam`] - Implements the Adam algorithm.
// //!
// //! * [`AMSGrad`] - Implements the AMSGrad algorithm.
// //!
// //! * [`RMSProp`] - Implements the RMSProp algorithm.
// //!
// //! * [`SGD`] - Implements the stochastic gradient descent algorithm.

mod adagrad;
mod adam;
mod amsgrad;
mod optimizer;
mod penalty;
mod rmsprop;
mod sgd;

pub mod lr_scheduler;

pub use adagrad::*;
pub use adam::*;
pub use optimizer::*;
pub use penalty::*;
pub use rmsprop::*;
pub use sgd::*;
