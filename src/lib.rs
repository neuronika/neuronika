// //! The `neuronika` crate provides auto-differentiation and dynamic neural networks.
// //!
// //! Neuronika is a machine learning framework written in pure Rust, built with a focus on ease of
// //! use, fast experimentation and performance.
// //!
// //! # Highlights
// //!
// //! * Define by run computational graphs.
// //! * Reverse-mode automatic differentiation.
// //! * Dynamic neural networks.
// //!
// //! # Variables
// //!
// //! The main building blocks of neuronika are *variables* and *differentiable variables*.
// //! This means that when using this crate you will be handling and manipulating instances of [`Var`]
// //! and [`VarDiff`].
// //!
// //! Variables are lean and powerful abstractions over the computational graph's nodes. Neuronika
// //! empowers you with the ability of imperatively building and differentiating such graphs with
// //! minimal amount of code and effort.
// //!
// //! Both differentiable and non-differentiable variables can be understood as *tensors*. You
// //! can perform all the basic arithmetic operations on them, such as: `+`, `-`, `*` and `/`.
// //! Refer to [`Var`] and [`VarDiff`] for a complete list of the available operations.
// //!
// //! It is important to note that cloning variables is extremely memory efficient as only a shallow
// //! copy is returned. Cloning a variable is thus the way to go if you need to use it several times.
// //!
// //! The provided API is linear in thought and minimal as it is carefully tailored around you, the
// //! user.
// //!
// //! ### Quickstart
// //!
// //! If you’re familiar with Pytorch or Numpy, you will easily follow these example. If not, brace
// //! yourself and follow along.
// //!
// //! First thing first, you should import neuronika.
// //!
// //! ```
// //! use neuronika;
// //! ```
// //!
// //! Neuronika's variables are designed to work with the [`f32`] data type, although this may change in
// //! the future, and can be initialized in many ways. In the following, we will show some of
// //! the possible alternatives:
// //!
// //! **With random or constant values**:
// //!
// //! Here `shape` determines the dimensionality of the output variable.
// //! ```
// //! let shape = [3, 4];
// //!
// //! let rand_variable = neuronika::rand(shape);
// //! let ones_variable = neuronika::ones(shape);
// //! let constant_variable = neuronika::full(shape, 7.);
// //!
// //! print!("Full variable:\n{}", constant_variable);
// //! ```
// //!
// //! Out:
// //!
// //! ```text
// //! [[7, 7, 7, 7],
// //! [7, 7, 7, 7],
// //! [7, 7, 7, 7]]
// //! ```
// //!
// //! **From a ndarray array**
// //!
// //! ```
// //! use ndarray::array;
// //!
// //! let array = array![1., 2.];
// //! let x_ndarray = neuronika::from_ndarray(array);
// //!
// //! print!("From ndarray:\n{}", x_ndarray);
// //! ```
// //! Out:
// //!
// //! ```text
// //! [1, 2]
// //! ```
// //!
// //! Accessing the underlying data is possible by using [`.data()`](crate::Data):
// //!
// //! ```
// //! let dim = (2, 2);
// //!
// //! let x = neuronika::rand(dim);
// //!
// //! assert_eq!(x.data().dim(), dim);
// //! ```
// //!
// //! ## Leaf Variables
// //!
// //! You can create leaf variables by using one of the many provided functions, such as [`zeros()`],
// //! [`ones()`], [`full()`] and [`rand()`]. Refer to the [complete list](#functions) for additional
// //! information.
// //!
// //! Leaf variables are so called because they form the *leaves* of the computational graph, as are
// //! not the result of any computation.
// //!
// //! Every leaf variable is by default created as non-differentiable, to promote it to a
// //! *differentiable* leaf, i. e. a variable for which you can compute the gradient, you can use
// //! [`.requires_grad()`](Var::requires_grad()).
// //!
// //! Differentiable leaf variables are leaves that have been promoted. You will encounter them
// //! very often in your journey through neuronika as they are the the main components of the
// //! neural networks' building blocks. To learn more in detail about those check the
// //! [`nn`](module@nn) module.
// //!
// //! Differentiable leaves hold a gradient, you can access it with [`.grad()`](VarDiff::grad()).
// //!
// //! ## Differentiability Arithmetic
// //!
// //! As stated before, you can manipulate variables by performing operations on them; the results of
// //! those computations will also be variables, although not leaf ones.
// //!
// //! The result of an operation between two differentiable variables will also be a differentiable
// //! variable and the converse holds for non-differentiable variables. However, things behave
// //! slightly differently when an operation is performed between a non-differentiable variable and a
// //! differentiable one, as the resulting variable will be differentiable.
// //!
// //! You can think of differentiability as a *sticky* property. The table that follows is a summary
// //! of how differentiability is broadcasted through variables.
// //!
// //!  **Operands** | Var     | VarDiff
// //! --------------|---------|---------
// //!  **Var**      | Var     | VarDiff
// //!  **VarDiff**  | VarDiff | VarDiff
// //!
// //!
// //! ## Differentiable Ancestors
// //!
// //! The differentiable ancestors of a variable are the differentiable leaves of the graph involved
// //! in its computation. Obviously, only [`VarDiff`] can have a set of ancestors.
// //!
// //! You can gain access, via mutable views, to all the ancestors of a variable by iterating through
// //! the vector of [`Param`] returned by [`.parameters()`](VarDiff::parameters()).
// //! To gain more insights about the role that such components fulfil in neuronika feel free to check
// //! the [`optim`] module.
// //!
// //! # Computational Graph
// //!
// //! A computational graph is implicitly created as you write your program. You can differentiate it
// //! with respect to some of the differentiable leaves, thus populating their gradients, by using
// //! [`.backward()`](VarDiff::backward()).
// //!
// //! It is important to note that the computational graph is *lazily* evaluated, this means that
// //! neuronika decouples the construction of the graph from the actual computation of the nodes'
// //! values. You must use `.forward()` in order to obtain the actual result of the computation.
// //!
// //!```
// //! # #[cfg(feature = "blas")]
// //! # extern crate blas_src;
// //!use neuronika;
// //!
// //!let x = neuronika::rand(5);      //----+
// //!let q = neuronika::rand((5, 5)); //    |- Those lines build the graph.
// //!                                 //    |
// //!let y = x.clone().vm(q).vv(x);   //----+
// //!                                 //
// //!y.forward();                     // After .forward() is called y contains the result.
// //!```
// //!
// //! ## Freeing and keeping the graph
// //!
// //! By default, computational graphs will persist in the program's memory. If you want or need to be
// //! more conservative about that you can wrap any arbitrary subset of the computations in an inner
// //! scope. This allows for the corresponding portion of the graph to be freed when the end of
// //! the scope is reached by the execution of your program.
// //!
// //!```
// //! # #[cfg(feature = "blas")]
// //! # extern crate blas_src;
// //!use neuronika;
// //!
// //!let w = neuronika::rand((3, 3)).requires_grad(); // -----------------+
// //!let b = neuronika::rand(3).requires_grad();      //                  |
// //!let x = neuronika::rand((10, 3));                // -----------------+- Leaves are created
// //!                                                 //
// //!{                                                // ---+
// //!     let h = x.mm(w.t()) + b;                    //    | w's and b's
// //!     h.forward();                                //    | grads are
// //!     h.backward(1.0);                            //    | accumulated
// //!}                                                // ---+             |- Graph is freed and
// //!                                                 // -----------------+  only leaves remain
// //!```
// #![doc(
//     html_logo_url = "https://raw.githubusercontent.com/neuronika/neuronika/main/misc/neuronika_brain.svg"
// )]
// #![doc(
//     html_favicon_url = "https://raw.githubusercontent.com/neuronika/neuronika/main/misc/neuronika_brain.ico"
// )]
pub use neuronika_variable::*;

pub mod optim {

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
    pub use neuronika_optim::*;
}

pub mod nn {
    // //! Basic building blocks for neural networks.
    // //!
    // //! Neuronika provides some pre-assembled components, you can either use them individually or
    // //! combine them into a bigger architecture. Take a look at the [complete list](#layers) to know
    // //! more.
    // //!
    // //! You can also customize the initialization of the parameters of such components, and that of any
    // //! other differentiable variable, by picking the function that best fits your needs from the
    // //! [`nn::init`](module@init) module.
    // //!
    // //! Refer to the [`nn::loss`](module@loss) module for loss functions.
    // //!
    // //! # Assembling a neural network
    // //!
    // //! The suggested way of building a model using neuronika's building blocks is to define a struct
    // //! encapsulating its components.
    // //!
    // //! The behavior of the model should be defined by including an appropriate method in its struct
    // //! implementation. Such method must specify how the components interact.
    // //!
    // //! Consider, for the sake of simplicity, a classical *multilayer perceptron* with three dense
    // //! layers for a multivariate regression task, let's see what it would look like in neuronika.
    // //!
    // //! We begin by defining its struct using the provided components.
    // //!
    // //! ```
    // //! use neuronika::nn;
    // //!
    // //! // Network definition.
    // //! struct NeuralNetwork {
    // //!     lin1: nn::Linear,
    // //!     lin2: nn::Linear,
    // //!     lin3: nn::Linear,
    // //! }
    // //! ```
    // //!
    // //! We'll also include a very simple constructor.
    // //!
    // //! ```
    // //! # use neuronika::nn;
    // //! # struct NeuralNetwork {
    // //! #    lin1: nn::Linear,
    // //! #    lin2: nn::Linear,
    // //! #    lin3: nn::Linear,
    // //! # }
    // //! impl NeuralNetwork {
    // //!     // Basic constructor.
    // //!     fn new() -> Self {
    // //!         Self {
    // //!             lin1: nn::Linear::new(25, 30),
    // //!             lin2: nn::Linear::new(30, 35),
    // //!             lin3: nn::Linear::new(35, 5),
    // //!         }
    // //!     }
    // //! }
    // //! ```
    // //!
    // //! As the last step, we have to specify how the multilayer perceptron behaves, then, we're done.
    // //!
    // //! ```
    // //! use ndarray::Ix2;
    // //! use neuronika::{Backward, Data, Forward, Gradient, MatMatMulT, Overwrite, VarDiff};
    // //! use neuronika::nn::Learnable;
    // //!
    // //! # use neuronika::nn;
    // //! # struct NeuralNetwork {
    // //! #     lin1: nn::Linear,
    // //! #     lin2: nn::Linear,
    // //! #     lin3: nn::Linear,
    // //! # }
    // //! impl NeuralNetwork {
    // //!     // NeuralNetwork behavior. Notice the presence of the ReLU non-linearity.
    // //!     fn forward<I, T, U>(
    // //!         &self,
    // //!         input: I,
    // //!     ) -> VarDiff<impl Data<Dim = Ix2>, impl Gradient<Dim = Ix2>>
    // //!     where
    // //!         I: MatMatMulT<Learnable<Ix2>>,
    // //!         I::Output: Into<VarDiff<T, U>>,
    // //!         T: Data<Dim = Ix2> + Forward,
    // //!         U: Gradient<Dim = Ix2>,
    // //!     {
    // //!         let out1 = self.lin1.forward(input).relu();
    // //!         let out2 = self.lin2.forward(out1).relu();
    // //!         let out3 = self.lin3.forward(out2);
    // //!         out3
    // //!     }
    // //! }
    // //! ```
    // //!
    // //! Here's a fictitious example of the newly created multilayer perceptron in use.
    // //!
    // //! ```
    // //! # use neuronika::nn;
    // //! # use ndarray::Ix2;
    // //! # use neuronika::{Backward, Data, Forward, Gradient, MatMatMulT, Overwrite, VarDiff};
    // //! # use neuronika::nn::Learnable;
    // //! # #[cfg(feature = "blas")]
    // //! # extern crate blas_src;
    // //! # struct NeuralNetwork {
    // //! #    lin1: nn::Linear,
    // //! #    lin2: nn::Linear,
    // //! #    lin3: nn::Linear,
    // //! # }
    // //! # impl NeuralNetwork {
    // //! #     // Basic constructor.
    // //! #     fn new() -> Self {
    // //! #         Self {
    // //! #             lin1: nn::Linear::new(25, 30),
    // //! #             lin2: nn::Linear::new(30, 35),
    // //! #             lin3: nn::Linear::new(35, 5),
    // //! #         }
    // //! #     }
    // //! # }
    // //! # impl NeuralNetwork {
    // //! #     // NeuralNetwork behavior. Notice the presence of the ReLU non-linearity.
    // //! #     fn forward<I, T, U>(
    // //! #         &self,
    // //! #         input: I,
    // //! #     ) -> VarDiff<impl Data<Dim = Ix2>, impl Gradient<Dim = Ix2>>
    // //! #     where
    // //! #         I: MatMatMulT<Learnable<Ix2>>,
    // //! #         I::Output: Into<VarDiff<T, U>>,
    // //! #         T: Data<Dim = Ix2> + Forward,
    // //! #         U: Gradient<Dim = Ix2>,
    // //! #     {
    // //! #         let out1 = self.lin1.forward(input).relu();
    // //! #         let out2 = self.lin2.forward(out1).relu();
    // //! #         let out3 = self.lin3.forward(out2);
    // //! #         out3
    // //! #     }
    // //! # }
    // //! let model = NeuralNetwork::new();
    // //!
    // //! // Random data to be given in input to the model.
    // //! let fictitious_data = neuronika::rand((200, 25));
    // //!
    // //! let out = model.forward(fictitious_data);
    // //! out.forward(); // Always remember to call forward() !
    // //! # assert_eq!(out.data().shape(), &[200, 5]);
    // //! ```
    // //! # Tracking parameters with ModelStatus
    // //!
    // //! In some circumstances you may find useful to group the parameters of a model. Consider for
    // //! instance the following scenario.
    // //!
    // //! ```
    // //! # use neuronika::nn;
    // //! # use ndarray::Ix2;
    // //! # use neuronika::{Backward, Data, Forward, Gradient, MatMatMulT, Overwrite, VarDiff};
    // //! # use neuronika::nn::Learnable;
    // //! # #[cfg(feature = "blas")]
    // //! # extern crate blas_src;
    // //! # struct NeuralNetwork {
    // //! #    lin1: nn::Linear,
    // //! #    lin2: nn::Linear,
    // //! #    lin3: nn::Linear,
    // //! # }
    // //! # impl NeuralNetwork {
    // //! #     // Basic constructor.
    // //! #     fn new() -> Self {
    // //! #         Self {
    // //! #             lin1: nn::Linear::new(25, 30),
    // //! #             lin2: nn::Linear::new(30, 35),
    // //! #             lin3: nn::Linear::new(35, 5),
    // //! #         }
    // //! #     }
    // //! # }
    // //! # impl NeuralNetwork {
    // //! #     // NeuralNetwork behavior. Notice the presence of the ReLU non-linearity.
    // //! #     fn forward<I, T, U>(
    // //! #         &self,
    // //! #         input: I,
    // //! #     ) -> VarDiff<impl Data<Dim = Ix2>, impl Gradient<Dim = Ix2>>
    // //! #     where
    // //! #         I: MatMatMulT<Learnable<Ix2>>,
    // //! #         I::Output: Into<VarDiff<T, U>>,
    // //! #         T: Data<Dim = Ix2>,
    // //! #         U: Gradient<Dim = Ix2>,
    // //! #     {
    // //! #         let out1 = self.lin1.forward(input).relu();
    // //! #         let out2 = self.lin2.forward(out1).relu();
    // //! #         let out3 = self.lin3.forward(out2);
    // //! #         out3
    // //! #     }
    // //! # }
    // //! let model = NeuralNetwork::new();
    // //!
    // //! let some_other_variable = neuronika::rand((1, 25)).requires_grad();
    // //!
    // //! // Random perturbed data.
    // //! let fictitious_data = neuronika::rand((200, 25)) + some_other_variable;
    // //!
    // //! let out = model.forward(fictitious_data);
    // //! assert_eq!(out.parameters().len(), 7); // 7 leaf ancestors !
    // //! ```
    // //!
    // //! You may notice how, if we feed in input to our neural network the result of an addition
    // //! operation, in which one of the operands is a differentiable variable, and then
    // //! request the network output's differentiable ancestors, we are given a vector containing 7
    // //! [`Param`](struct@Param).
    // //!
    // //! By doing some quick math: 7 = 2 * 3 + 1, and by noticing that each of the three linear layers
    // //! that the multilayer perceptron is made of has one learnable weight matrix and one learnable
    // //! bias vector, we can conclude that the presence of the seventh ancestors is due to the addition
    // //! between `fictitious_data` and `some_other_variable`.
    // //!
    // //! In fact, neuronika automatically tracks all the differentiable leaves that are involved in the
    // //! computation of the output variable when assembling the computational graph corresponding to
    // //! the issued operations.
    // //!
    // //! If you need to distinguish between the parameters of a model and another differentiable variable
    // //! or between the parameters of several different models, you can use [`ModelStatus`].
    // //!
    // //! With `ModelStatus` you can build the exact same neural network only varying the implementation
    // //! so slightly.
    // //!
    // //! ```
    // //!  use neuronika::Param;
    // //!  use neuronika::nn::{ModelStatus, Linear};
    // //!
    // //!  struct NeuralNetwork {
    // //!     lin1: Linear,
    // //!     lin2: Linear,
    // //!     lin3: Linear,
    // //!     status: ModelStatus,
    // //!  }
    // //!
    // //!  impl NeuralNetwork {
    // //!      fn new() -> Self {
    // //!          // Initialize an empty model status.
    // //!          let mut status = ModelStatus::default();
    // //!
    // //!          // We register each component whilst at the same time building the network.
    // //!          Self {
    // //!              lin1: status.register(Linear::new(25, 30)),
    // //!              lin2: status.register(Linear::new(30, 35)),
    // //!              lin3: status.register(Linear::new(35, 5)),
    // //!              status,
    // //!          }
    // //!      }
    // //!
    // //!      /// Returns the model's parameters.
    // //!      fn parameters(&self) -> Vec<Param> {
    // //!          // We are now able to access the parameter of the neural network.
    // //!          self.status.parameters()
    // //!      }
    // //!  }
    // //! ```
    // //!
    // //! At last, we verify that the number of registered parameters for the new version of our neural
    // //! network is indeed 6.
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
    // //! let model = NeuralNetwork::new();
    // //! assert_eq!(model.parameters().len(), 6);
    // //! ```
    // //! Do also note that in spite of the introduction of `ModelStatus`, the implementation of the
    // //! `.forward()` method has not changed at all.
    // //!
    // //! # Train and Eval
    // //!
    // //! The status of a model determines the behavior of its components. Certain building blocks, such
    // //! as the [`Dropout`], are turned on and off depending on whether the model is running in *training
    // //! mode* or in *inference mode*.
    // //!
    // //! You can set a network in training mode or in inference mode either by calling [`.train()`] and
    // //! [`.eval()`] directly on its output or by using `ModelStatus`.
    // //!
    // //! The former approach is preferable, as when multiple models are pipelined, calling `.train()`
    // //! and `.eval()` directly on the final outputs will switch the statuses of all the models.
    // //! Do also note that switching the status by using `ModelStatus` is the only way that allows for
    // //! selectively training and evaluating multiple models.
    // //!
    // //! Let's picture it with a simple example.
    // //!
    // //! [`.eval()`]: VarDiff::eval()
    // //! [`.train()`]: VarDiff::train()
    // //!
    // //! ```
    // //!  use neuronika::Param;
    // //!  use neuronika::nn::{ModelStatus, Linear, Dropout};
    // //!
    // //!  struct NeuralNetwork {
    // //!     lin1: Linear,
    // //!     drop: Dropout,
    // //!     lin2: Linear,
    // //!     status: ModelStatus,
    // //!  }
    // //!
    // //!  impl NeuralNetwork {
    // //!      fn new() -> Self {
    // //!          let mut status = ModelStatus::default();
    // //!
    // //!          // Similarly to what we did before, we register the components
    // //!          // to the network's status.
    // //!          // Now the dropout layer, and every other changeable
    // //!          // component, can be directly controlled by interacting
    // //!          // with the model itself, as it is synced with the one of
    // //!          // ModelStatus.
    // //!          Self {
    // //!              lin1: status.register(Linear::new(25, 35)),
    // //!              drop: status.register(Dropout::new(0.5)),
    // //!              lin2: status.register(Linear::new(35, 5)),
    // //!              status,
    // //!          }
    // //!      }
    // //!
    // //!      fn parameters(&self) -> Vec<Param> {
    // //!          self.status.parameters()
    // //!      }
    // //!
    // //!      /// Switches the network in training mode.
    // //!      fn train(&self) {
    // //!          self.status.train()
    // //!      }
    // //!
    // //!      /// Switches the network in inference mode.
    // //!      fn eval(&self) {
    // //!          self.status.eval()
    // //!      }
    // //!  }
    // //! ```
    // //!
    // //! # Layers
    // //!
    // //! Here are listed all neuronika's building blocks.
    // //!
    // //! ## Linear Layers
    // //!
    // //! * [`nn::Linear`](struct@Linear) - Applies a linear transformation to the incoming data.
    // //!
    // //! ## Recurrent Layers
    // //!
    // //! * [`nn::GRUCell`](struct@GRUCell) - A gated recurrent unit cell.
    // //!
    // //! * [`nn::LSTMCell`](struct@LSTMCell) - A long short term memory cell.
    // //!
    // //! ## Convolution Layers
    // //!
    // //! * [`nn::Conv1d`](struct@Conv1d) - Applies a temporal convolution over an input signal composed
    // //! of several input planes.
    // //!
    // //! * [`nn::GroupedConv1d`](struct@GroupedConv1d) - Applies a grouped temporal convolution over an
    // //! input signal composed of several input planes.
    // //!
    // //! * [`nn::Conv2d`](struct@Conv2d) - Applies a spatial convolution over an input signal composed
    // //! of several input planes.
    // //!
    // //! * [`nn::GroupedConv2d`](struct@GroupedConv2d) - Applies a grouped spatial convolution over an
    // //! input signal composed of several input planes.
    // //!
    // //! * [`nn::Conv3d`](struct@Conv3d) - Applies a volumetric convolution over an input signal composed
    // //! of several input planes.
    // //!
    // //! * [`nn::GroupedConv3d`](struct@GroupedConv3d) - Applies a grouped volumetric convolution over an
    // //! input signal composed of several input planes.
    // //!
    // //! ## Dropout Layers
    // //!
    // //! * [`nn::Dropout`](struct@Dropout) - During training, randomly zeroes some of the elements of
    // //! the input variable with probability *p* using samples from a Bernoulli distribution.
    //!
    //! # Layers' parameters initialization functions.
    //!
    //! These initializers define a way to set the initial random weights of neuronika's layers.
    //!
    //! # Using an initializer
    //!
    //! You can freely access any learnable component of any layer, as their visibility is public,
    //! and pass them, via a mutable reference, to the initialization function of your choice.
    //!
    //! ```
    //! use neuronika::nn;
    //! use neuronika::nn::init::{calculate_gain, xavier_normal};
    //!
    //! let mut lin = nn::Linear::new(10, 10);
    //!
    //! xavier_normal(&lin.weight, calculate_gain("relu"));
    //! ```
    pub use neuronika_nn::*;
}

pub mod data {
    //! Data loading and manipulation utilities.
    //!
    //! # Dataset Types
    //!
    //! Neuronika provides two kinds of datasets, an unlabeled one, that is [`Dataset`], and a labeled
    //! one, that is [`LabeledDataset`]. They both own their data uniquely.
    //!
    //! Datasets are basic containers for your data and are designed to easily interact with models
    //! built with neuronika. They are created with the help of the [`DataLoader`] struct which performs
    //! the actual I/O operations.
    //!
    //! Both datasets are generic on the [dimensionality] of their records and are organized as a tensors
    //! in which the length of the outermost axis is equal to the total number of records and the
    //! number of remaining axes represent the dimensionality of each data point.
    //!
    //! [`.from_csv()`]: crate::data::DataLoader::from_csv
    //! [dimensionality]: ndarray::Dimension
    //!
    //! # Loading Data
    //!
    //! At the core of neuronika data utilities is the [`DataLoader`] struct. It can be used to load
    //! data in *comma-separated values format* from a [*reader*](Read) or directly from a *.csv* file.
    //!
    //! Additional parsing settings are passed using `DataLoader`'s methods in the following way.
    //!
    //! ```should_panic
    //! use neuronika::data::DataLoader;
    //!
    //! let data = DataLoader::default()           // A typical use case would be
    //!     .with_labels(&[5, 6, 7])               // to load some data from
    //!     .with_delimiter(',')                   // a .csv file.
    //!     .from_csv("./folder/data.csv", 3, 1);
    //! ```
    //!
    //! The result of the loading operation is either a [`Dataset`] or a [`LabeledDataset`], depending
    //! on how the loader was configured.
    //!
    //! ## Handling Labels
    //!
    //! You may find useful, in many real world scenarios, to convert labels to floating point numbers.
    //! In neuronika this is quickly achievable with closures. Take a look at the following example.
    //!
    //! ```rust
    //! use neuronika::data::DataLoader;
    //!
    //! let csv_content = "\
    //!     Paw_size,Tail_length,Weight,Animal\n\
    //!     0.2,5.0,15.0,Dog\n\
    //!     0.08,12.0,4.0,Cat\n\
    //!     0.05,3.0,0.8,Mouse";
    //!
    //! let dataset = DataLoader::default().with_labels(&[3]).from_reader_fn(
    //!     csv_content.as_bytes(),
    //!     3,
    //!     1,
    //!     |(record, label): (Vec<f32>, String)| {
    //!         let float_label = match label.as_str() {
    //!             "Dog" => 1.,
    //!             "Cat" => 2.,
    //!              _ => 3.,
    //!         };
    //!         (record, vec![float_label])
    //!     },
    //! );
    //! ```
    pub use neuronika_data::*;
}
