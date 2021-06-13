//! Basic building blocks for neural networks.
//!
//! Neuronika provides some pre-assembled components, you can either use them individually or
//! combine them into a bigger architecture. Feel free to take a look at the
//! [complete list](#layers).
//!
//! You can also customise the initialisation of the parameters of such components, and that of any
//! other differentiable variable, by picking the function that best fits your needs from the
//! [`nn::init`](module@init) module.
//!
//! Refer to the [`nn::loss`](module@loss) module for loss functions.
//!
//! # Assembling a neural network
//!
//! The suggested way of bulding a model using neuronika's building blocks is to define a struct
//! encapsulating its components.
//!
//! The behaviour of the model should be defined by including an appropriate method in its struct
//! implementation. Such method must specify how the components interact.
//!
//! Consider, for the sake of simplicity, a classical *multilayer perceptron* with three dense
//! layers for a multivariate regression task, let's see what it would look like in neuronika.
//!
//! We begin by definining its struct using the provided components.
//!
//! ```
//! use neuronika::nn;
//!
//! // MLP definition.
//! struct MLP {
//!     lin1: nn::Linear,
//!     lin2: nn::Linear,
//!     lin3: nn::Linear,     
//! }
//! ```
//!
//! We'll also include a very simple constructor.
//!
//! ```
//! # use neuronika::nn;
//! # struct MLP {
//! #    lin1: nn::Linear,
//! #    lin2: nn::Linear,
//! #    lin3: nn::Linear,     
//! # }
//! impl MLP {
//!     // Basic constructor.
//!     fn new() -> Self {
//!         Self {
//!             lin1: nn::Linear::new(25, 30),
//!             lin2: nn::Linear::new(30, 35),
//!             lin3: nn::Linear::new(35, 5),
//!         }
//!     }
//! }
//! ```
//!
//! As the last step, we have to specify how the multilayer perceptron behaves, then, we're done.
//!
//! ```
//! use ndarray::Ix2;
//! use neuronika::{Backward, Data, Forward, Gradient, MatMatMulT, Overwrite, VarDiff};
//! use neuronika::nn::Learnable;
//!
//! # use neuronika::nn;
//! # struct MLP {
//! #     lin1: nn::Linear,
//! #     lin2: nn::Linear,
//! #     lin3: nn::Linear,     
//! # }
//! impl MLP {
//!     // MLP behaviour. Notice the presence of the ReLU non-linearity.
//!     fn forward<I, T, U>(
//!         &self,
//!         input: I,
//!     ) -> VarDiff<
//!             impl Data<Dim = Ix2> + Forward,
//!             impl Gradient<Dim = Ix2> + Overwrite + Backward
//!         >
//!     where
//!         I: MatMatMulT<Learnable<Ix2>>,
//!         I::Output: Into<VarDiff<T, U>>,
//!         T: Data<Dim = Ix2> + Forward,
//!         U: Gradient<Dim = Ix2> + Backward + Overwrite,
//!     {
//!         let out1 = self.lin1.forward(input).relu();
//!         let out2 = self.lin2.forward(out1).relu();
//!         let out3 = self.lin3.forward(out2);
//!         out3
//!     }
//! }
//! ```
//!
//! Here's a fictitious example of the newly created multilayer perceptron in use.
//!
//! ```
//! # use neuronika::nn;
//! # use ndarray::Ix2;
//! # use neuronika::{Backward, Data, Forward, Gradient, MatMatMulT, Overwrite, VarDiff};
//! # use neuronika::nn::Learnable;
//! # #[cfg(feature = "blas")]
//! # extern crate blas_src;
//! # struct MLP {
//! #    lin1: nn::Linear,
//! #    lin2: nn::Linear,
//! #    lin3: nn::Linear,     
//! # }
//! # impl MLP {
//! #     // Basic constructor.
//! #     fn new() -> Self {
//! #         Self {
//! #             lin1: nn::Linear::new(25, 30),
//! #             lin2: nn::Linear::new(30, 35),
//! #             lin3: nn::Linear::new(35, 5),
//! #         }
//! #     }
//! # }
//! # impl MLP {
//! #     // MLP behaviour. Notice the presence of the ReLU non-linearity.
//! #     fn forward<I, T, U>(
//! #         &self,
//! #         input: I,
//! #     ) -> VarDiff<
//! #             impl Data<Dim = Ix2> + Forward,
//! #             impl Gradient<Dim = Ix2> + Overwrite + Backward
//! #         >
//! #     where
//! #         I: MatMatMulT<Learnable<Ix2>>,
//! #         I::Output: Into<VarDiff<T, U>>,
//! #         T: Data<Dim = Ix2> + Forward,
//! #         U: Gradient<Dim = Ix2> + Backward + Overwrite,
//! #     {
//! #         let out1 = self.lin1.forward(input).relu();
//! #         let out2 = self.lin2.forward(out1).relu();
//! #         let out3 = self.lin3.forward(out2);
//! #         out3
//! #     }
//! # }
//! let model = MLP::new();
//!
//! // Random data to be given in input to the model.
//! let fictitious_data = neuronika::rand((200, 25));
//!
//! let mut out = model.forward(fictitious_data);
//! out.forward(); // Always remember to call forward() !
//! # assert_eq!(out.data().shape(), &[200, 5]);
//! ```
//! # Tracking parameters with ModelStatus
//!
//! In some circumstances you may find useful to group the parameters of a model. Consider for
//! instance the following scenario.
//!
//! ```
//! # use neuronika::nn;
//! # use ndarray::Ix2;
//! # use neuronika::{Backward, Data, Forward, Gradient, MatMatMulT, Overwrite, VarDiff};
//! # use neuronika::nn::Learnable;
//! # #[cfg(feature = "blas")]
//! # extern crate blas_src;
//! # struct MLP {
//! #    lin1: nn::Linear,
//! #    lin2: nn::Linear,
//! #    lin3: nn::Linear,     
//! # }
//! # impl MLP {
//! #     // Basic constructor.
//! #     fn new() -> Self {
//! #         Self {
//! #             lin1: nn::Linear::new(25, 30),
//! #             lin2: nn::Linear::new(30, 35),
//! #             lin3: nn::Linear::new(35, 5),
//! #         }
//! #     }
//! # }
//! # impl MLP {
//! #     // MLP behaviour. Notice the presence of the ReLU non-linearity.
//! #     fn forward<I, T, U>(
//! #         &self,
//! #         input: I,
//! #     ) -> VarDiff<
//! #             impl Data<Dim = Ix2> + Forward,
//! #             impl Gradient<Dim = Ix2> + Overwrite + Backward
//! #         >
//! #     where
//! #         I: MatMatMulT<Learnable<Ix2>>,
//! #         I::Output: Into<VarDiff<T, U>>,
//! #         T: Data<Dim = Ix2> + Forward,
//! #         U: Gradient<Dim = Ix2> + Backward + Overwrite,
//! #     {
//! #         let out1 = self.lin1.forward(input).relu();
//! #         let out2 = self.lin2.forward(out1).relu();
//! #         let out3 = self.lin3.forward(out2);
//! #         out3
//! #     }
//! # }
//! let model = MLP::new();
//!
//! let some_other_variable = neuronika::rand((1, 25)).requires_grad();
//!
//! // Random perturbated data.
//! let fictitious_data = neuronika::rand((200, 25)) + some_other_variable;
//!
//! let out = model.forward(fictitious_data);
//! assert_eq!(out.parameters().len(), 7); // 7 leaf ancestors !
//! ```
//!
//! You can notice how, if we give in input to the multilayer perceptron the result of an
//! addition operation, in which one of the operands is a differentiable variable, and then
//! request the *mlp* output's differentiable ancestors, we are given a vector containing 7
//! [`Param`](struct@Param).
//!
//! By doing some quick math: 7 = 2 * 3 + 1, and by noticing that each of the three linear layers
//! that the multilayer perceptron is made of has one learnable weight matrix and one learnable
//! bias vector, we can conclude that the presence of the seventh ancestors is due to the addition
//! between `fictitious_data` and `some_other_variable`.
//!
//! If you need to distinguish between the parameters of a model and another differentiable variable
//! or between the parameters of several different models, you can use [`ModelStatus`].
//!
//! With `ModelStatus` you can build the exact same multilayer perceptron only varying the
//! implementation so slightly.
//!
//! ```
//!  use neuronika::Param;
//!  use neuronika::nn::{ModelStatus, Linear};
//!
//!  struct MLP {
//!     lin1: Linear,
//!     lin2: Linear,
//!     lin3: Linear,
//!     status: ModelStatus,     
//!  }
//!
//!  impl MLP {
//!      fn new() -> Self {
//!          let mut status = ModelStatus::default();
//!
//!          Self {
//!              lin1: status.register(Linear::new(25, 30)),
//!              lin2: status.register(Linear::new(30, 35)),
//!              lin3: status.register(Linear::new(35, 5)),
//!              status,
//!          }
//!      }
//!
//!      fn parameters(&self) -> Vec<Param> {
//!          self.status.parameters()
//!      }
//!  }
//! ```
//!
//! Let's verify that the number of registered parameters for the new version of our multilayer
//! perceptron is indeed 6.
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
//! let model = MLP::new();
//! assert_eq!(model.parameters().len(), 6);
//! ```
//! Do also note that in spite of the introduction of `ModelStatus`, the implementation of the
//! `.forward()` method has not changed at all.
//!
//! # Train and Eval
//!
//! The status of a model determines the behaviour of its components. Certain building blocks, such
//! as the [`Dropout`], are turned on and off depending on wheter the model is running in *training
//! mode* or in *inference mode*.
//!
//! You can set a network in training mode or in inference mode either by calling [`.train()`] and
//! [`.eval()`] directly on its output or by using `ModelStatus`.
//!
//! The former approach is preferrable, as when multiple models are pipelined, calling `.train()`
//! and `.eval()` directly on the final outputs will switch the statuses of all the models.
//! Do also note that switching the status by using `ModelStatus` is the only way that allows for
//! selectively training and evaluating multiple models.
//!
//! Let's picture it with simple example.
//!
//! [`.eval()`]: VarDiff::eval()
//! [`.train()`]: VarDiff::train()
//!
//! ```
//!  use neuronika::Param;
//!  use neuronika::nn::{ModelStatus, Linear, Dropout};
//!
//!  struct MLP {
//!     lin1: Linear,
//!     drop: Dropout,
//!     lin2: Linear,
//!     status: ModelStatus,     
//!  }
//!
//!  impl MLP {
//!      fn new() -> Self {
//!          let mut status = ModelStatus::default();
//!
//!          Self {
//!              lin1: status.register(Linear::new(25, 35)),
//!              drop: status.register(Dropout::new(0.5)),
//!              lin2: status.register(Linear::new(35, 5)),
//!              status,
//!          }
//!      }
//!
//!      fn parameters(&self) -> Vec<Param> {
//!          self.status.parameters()
//!      }
//!
//!      fn train(&self) {
//!          self.status.train()
//!      }
//!
//!      fn eval(&self) {
//!          self.status.eval()
//!      }
//!  }
//! ```
//!
//! # Layers
//!
//! Here are listed all neuronika's building blocks.
//!
//! ## Linear Layers
//!
//! * [`nn::Linear`](struct@Linear) - Applies a linear transformation to the incoming data.
//!
//! ## Recurrent Layers
//!
//! * [`nn::GRUCell`](struct@GRUCell) - A gated recurrent unit cell.
//!
//! * [`nn::LSTMCell`](struct@LSTMCell) - A long short term memory cell.
//!
//! ## Convolution Layers
//!
//! * [`nn::Conv1d`](struct@Conv1d) - Applies a temporal convolution over an input signal composed
//! of several input planes.
//!
//! * [`nn::GroupedConv1d`](struct@GroupedConv1d) - Applies a grouped temporal convolution over an
//! input signal composed of several input planes.
//!
//! * [`nn::Conv2d`](struct@Conv2d) - Applies a spatial convolution over an input signal composed
//! of several input planes.
//!
//! * [`nn::GroupedConv2d`](struct@GroupedConv2d) - Applies a grouped spatial convolution over an
//! input signal composed of several input planes.
//!
//! * [`nn::Conv3d`](struct@Conv3d) - Applies a volumetric convolution over an input signal composed
//! of several input planes.
//!
//! * [`nn::GroupedConv3d`](struct@GroupedConv3d) - Applies a grouped volumetric convolution over an
//! input signal composed of several input planes.
//!
//! ## Dropout Layers
//!
//! * [`nn::Dropout`](struct@Dropout) - During training, randomly zeroes some of the elements of
//! the input variable with probability *p* using samples from a Bernoulli distribution.
use super::{Input, InputBackward, Param};
use crate::variable::{
    self,
    node::{
        Backward, Data, Dropout as DropoutNode, DropoutBackward as DropoutBackwardNode, Eval,
        Forward, Gradient, Overwrite,
    },
    MatMatMulT, Tensor, Var, VarDiff,
};
pub use convolution::{Constant, PaddingMode, Reflective, Replicative, Zero};
use convolution::{Convolve, ConvolveWithGroups};
use ndarray::{Ix1, Ix2, Ix3, Ix4, Ix5};
use std::{cell::Cell, rc::Rc};

pub(super) mod convolution;
pub mod init;
pub mod loss;

/// A generic parameter of a neural component.
pub type Learnable<D> = VarDiff<Input<D>, InputBackward<D>>;

/// A model's components status.
///
/// This struct should be used when you are interested in keeping track of the statuses and the
/// parameters of the components that are part of a neural network. There are many circumstances in
/// which this can be useful, such as when you have more than one model in a pipeline.
///
/// This struct stores all the [`Learnable`] associated to a given model and the model's status. It
/// is suggested to perform the registration of the layers at the model construction.
pub struct ModelStatus {
    params: Vec<Param>,
    train: Rc<Cell<bool>>,
}

impl ModelStatus {
    /// Returns a vector of [`Param`] linked to the learnable weights associated to a neural
    /// network.
    ///
    /// Conceptually, this method behaves similarly to [`.parameters()`](VarDiff::parameters()) when
    /// called on the differentiable variable outputted by the network. The key difference lies in
    /// the fact that, while the differentiable variable's `.parameters()` method would return *all*
    /// the differentiable leaves that took part in the computation of the output, possibly also
    /// the weights of another network, `ModelStatus`'s `.parameters()` method returns *only* the
    /// leaves that have been associated with it at the model's instantiation.
    ///
    /// Usually the result of this method is passed to an optimizer.
    pub fn parameters(&self) -> Vec<Param> {
        self.params.to_vec()
    }

    /// Registers a component.
    ///
    /// # Arguments
    ///
    /// `component` - layer to be registered.
    pub fn register<T: Register>(&mut self, mut component: T) -> T {
        component.register_params(&mut self.params);
        component.register_status(self.train.clone());
        component
    }

    /// Sets the status in training mode.
    pub fn train(&self) {
        <Self as Eval>::train(&self)
    }

    /// Sets the status in inference mode.
    pub fn eval(&self) {
        <Self as Eval>::eval(&self)
    }
}

impl Default for ModelStatus {
    /// Returns a new `ModelStatus` with empty parameters and status set to train.
    fn default() -> Self {
        Self {
            params: Vec::new(),
            train: Rc::new(Cell::new(true)),
        }
    }
}

impl Eval for ModelStatus {
    /// Sets the status to train.
    fn train(&self) {
        self.train.set(true)
    }

    /// Sets the status to eval.
    fn eval(&self) {
        self.train.set(false)
    }
}

/// Dropout input.
///
/// This trait is implemented by `Var` and `VarDiff`.
pub trait DropoutInput {
    type Output;

    fn dropout(self, p: f64, status: Rc<Cell<bool>>) -> Self::Output;
}

impl<T, U> DropoutInput for VarDiff<T, U>
where
    T: Data + Forward,
    U: Gradient<Dim = T::Dim> + Overwrite + Backward,
{
    type Output = VarDiff<DropoutNode<T>, DropoutBackwardNode<U, T>>;

    fn dropout(self, p: f64, status: Rc<Cell<bool>>) -> Self::Output {
        self.dropout_with_status(p, status)
    }
}

impl<T> DropoutInput for Var<T>
where
    T: Data + Forward,
{
    type Output = Var<DropoutNode<T>>;

    fn dropout(self, p: f64, status: Rc<Cell<bool>>) -> Self::Output {
        self.dropout_with_status(p, status)
    }
}

/// Registration for neuronika's components.
pub trait Register {
    /// Registers `self`'s parameters to the model's  status parameters `params`.
    fn register_params(&self, params: &mut Vec<Param>);

    /// Register `self`'s status to the model's status state `status`.
    fn register_status(&mut self, status: Rc<Cell<bool>>);
}

/// During training, randomly zeroes some of the elements of `self` with probability *p* using
/// samples from a Bernoulli distribution. Each channel will be zeroed out independently on
/// every forward call.
///
/// This has proven to be an effective technique for regularization and preventing the
/// co-adaptation of neurons as described in the paper
/// [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580).
///
/// Furthermore, the outputs are scaled by a factor of 1/(1 - p) during training. This means
/// that during evaluation the resulting variable simply computes an identity function.
pub struct Dropout {
    pub status: Rc<Cell<bool>>,
    pub p: f64,
}

impl Dropout {
    /// Creates a dropout layer.
    ///
    /// # Arguments
    ///
    /// `p` - probability of an element to be zeroed.
    pub fn new(p: f64) -> Self {
        let status = Rc::new(Cell::new(true));
        Self { status, p }
    }

    /// Applies the dropout to the variable in input.
    ///
    /// # Arguments
    ///
    /// `input`  - variable in input to the layer.
    pub fn forward<I: DropoutInput>(&self, input: I) -> I::Output {
        input.dropout(self.p, self.status.clone())
    }
}

impl Eval for Dropout {
    fn eval(&self) {
        self.status.set(false)
    }

    fn train(&self) {
        self.status.set(true)
    }
}

impl Register for Dropout {
    fn register_status(&mut self, status: Rc<Cell<bool>>) {
        self.status = status;
    }

    fn register_params(&self, _: &mut Vec<Param>) {}
}

/// Applies a **linear transformation** to the incoming data.
///
/// ```text
/// ʏ = xAᵀ + b
/// ```
pub struct Linear {
    pub weight: Learnable<Ix2>,
    pub bias: Learnable<Ix1>,
}

impl Linear {
    /// Creates a linear layer.
    ///
    /// # Arguments
    ///
    /// * `in_features` – size of each input sample.
    ///
    /// * `out_features` – size of each output sample.
    ///
    /// The learnable weight of the layer is of shape `(out_features, in_features)`. The learnable
    /// bias of the layer is of shape `out_features`.
    ///
    /// The values for both the weight and bias are initialised from *U(-k, k)* where
    /// `k = (1. / in_features as f32).sqrt()`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weight = Input::new(Tensor::zeros((out_features, in_features))).requires_grad();
        let mut bias = Input::new(Tensor::zeros(out_features)).requires_grad();
        let k = (1. / (in_features as f32)).sqrt();
        init::uniform(&mut weight, -k, k);
        init::uniform(&mut bias, -k, k);

        Self { weight, bias }
    }

    /// Applies the linear transformation *y = xA^T + b* to the incoming data.
    ///
    /// # Arguments
    ///
    /// `data` - a variable of shape *(N, in_features)*, the output's shape will be
    /// *(N, out_features)*.
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix2> + Forward, impl Gradient<Dim = Ix2> + Overwrite + Backward>
    where
        I: MatMatMulT<Learnable<Ix2>>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix2>,
        U: Gradient<Dim = Ix2> + Overwrite,
    {
        input.mm_t(self.weight.clone()).into() + self.bias.clone()
    }
}

impl Register for Linear {
    /// Registers the weight and the bias of this `Linear` instance.
    fn register_params(&self, params: &mut Vec<Param>) {
        params.extend(self.weight.parameters());
        params.extend(self.bias.parameters());
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}

/// A **long short-term memory (LSTM)** cell.
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct LSTMCell {
    pub weight_ih: Learnable<Ix2>,
    pub weight_hh: Learnable<Ix2>,
    pub bias_ih: Learnable<Ix1>,
    pub bias_hh: Learnable<Ix1>,
}

impl LSTMCell {
    /// Creates a new LSTMCell.
    ///
    /// # Arguments
    ///
    /// * `input_size` - number of expected features in the input.
    ///
    /// * `hidden_size` - number of features in the hidden state.
    ///
    /// All the weight and biases are initialised from *U(-k, k)* where
    /// `k = (1. / hidden_size as f32).sqrt()`.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let (weight_ih_shape, weight_hh_shape, bias_shape) = {
            let xhidden_size = 4 * hidden_size;
            (
                (xhidden_size, input_size),
                (xhidden_size, hidden_size),
                xhidden_size,
            )
        };
        let mut weight_ih = Input::new(Tensor::zeros(weight_ih_shape)).requires_grad();
        let mut weight_hh = Input::new(Tensor::zeros(weight_hh_shape)).requires_grad();
        let mut bias_ih = Input::new(Tensor::zeros(bias_shape)).requires_grad();
        let mut bias_hh = Input::new(Tensor::zeros(bias_shape)).requires_grad();

        let k = 1. / (hidden_size as f32).sqrt();
        init::uniform(&mut weight_ih, -k, k);
        init::uniform(&mut weight_hh, -k, k);
        init::uniform(&mut bias_ih, -k, k);
        init::uniform(&mut bias_hh, -k, k);

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    /// Computes a single **LSTM step**.
    ///
    /// # Arguments
    ///
    /// * `state` - a tuple of tensors, both of shape *(batch, hidden_size)*, containing the
    /// initial hidden state for each element in the batch and the initial cell's state for
    /// each element in the batch.
    ///
    /// * `input` - a variable containing the input features of shape *(batch, input_size)*.
    ///
    /// The **output** is a tuple of tensors made of the next hidden state for each element in
    /// the batch, of shape *(batch, hidden_size)* and the next cell's state for each element in
    /// the batch, of shape *(batch, hidden_size)*.
    pub fn forward<Cf, Cb, Hf, Hb, I, T, U>(
        &self,
        state: (VarDiff<Cf, Cb>, VarDiff<Hf, Hb>),
        input: I,
    ) -> (
        VarDiff<impl Data<Dim = Ix2> + Forward, impl Gradient<Dim = Ix2> + Overwrite + Backward>,
        VarDiff<impl Data<Dim = Ix2> + Forward, impl Gradient<Dim = Ix2> + Overwrite + Backward>,
    )
    where
        Cf: Data<Dim = Ix2>,
        Cb: Gradient<Dim = Ix2> + Overwrite,
        Hf: Data<Dim = Ix2>,
        Hb: Gradient<Dim = Ix2> + Overwrite,
        I: MatMatMulT<Learnable<Ix2>>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix2>,
        U: Gradient<Dim = Ix2> + Overwrite,
    {
        let (cell_state, hidden) = state;
        let gates = hidden.mm_t(self.weight_hh.clone())
            + self.bias_hh.clone()
            + input.mm_t(self.weight_ih.clone()).into()
            + self.bias_ih.clone();
        let gate_shape = {
            let (gates_shape_rows, gates_shape_cols) = gates.data().dim();
            (gates_shape_rows, gates_shape_cols / 4)
        };
        let chunked_gates = gates.chunks(gate_shape);
        let (input_gate, forget_gate, cell_state_gate, output_gate) = (
            chunked_gates[0].clone().sigmoid(),
            chunked_gates[1].clone().tanh(),
            chunked_gates[2].clone().sigmoid(),
            chunked_gates[3].clone().sigmoid(),
        );
        let new_cell_state = forget_gate * cell_state + (input_gate * cell_state_gate);
        let new_hidden = output_gate * new_cell_state.clone().tanh();

        (new_cell_state, new_hidden)
    }
}

impl Register for LSTMCell {
    /// Registers the weights and the biases of this LSTMCell instance.
    fn register_params(&self, params: &mut Vec<Param>) {
        params.extend(self.weight_hh.parameters());
        params.extend(self.weight_ih.parameters());
        params.extend(self.bias_hh.parameters());
        params.extend(self.bias_ih.parameters());
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}

/// A **gated recurrent unit (GRU)** cell.
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct GRUCell {
    pub weight_ih: Learnable<Ix2>,
    pub weight_hh: Learnable<Ix2>,
    pub bias_ih: Learnable<Ix1>,
    pub bias_hh: Learnable<Ix1>,
}

impl GRUCell {
    /// Creates a new GRUCell.
    ///
    /// # Arguments
    ///
    /// * `input_size` - number of expected features in the input.
    ///
    /// * `hidden_size` - number of features in the hidden state.
    ///
    /// All the weight and biases are initialised from *U(-k, k)* where
    /// `k = (1. / hidden_size as f32).sqrt()`.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        let (weight_ih_shape, weight_hh_shape, bias_shape) = {
            let xhidden_size = 3 * hidden_size;
            (
                (xhidden_size, input_size),
                (xhidden_size, hidden_size),
                xhidden_size,
            )
        };
        let mut weight_ih = Input::new(Tensor::zeros(weight_ih_shape)).requires_grad();
        let mut weight_hh = Input::new(Tensor::zeros(weight_hh_shape)).requires_grad();
        let mut bias_ih = Input::new(Tensor::zeros(bias_shape)).requires_grad();
        let mut bias_hh = Input::new(Tensor::zeros(bias_shape)).requires_grad();

        let k = 1. / (hidden_size as f32).sqrt();
        init::uniform(&mut weight_ih, -k, k);
        init::uniform(&mut weight_hh, -k, k);
        init::uniform(&mut bias_ih, -k, k);
        init::uniform(&mut bias_hh, -k, k);

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    /// Computes a single **GRU step**.
    ///
    /// * `hidden` - a variable of shape *(batch, hidden_size)*, containing the initial hidden state
    /// for each element in the batch.
    ///
    /// * `input` - a variable containing the input features of shape *(batch, input_size)*.
    ///
    /// The **output** is  a variable made of the next hidden state for each element in
    /// the batch, of shape *(batch, hidden_size)*.
    pub fn forward<Hf, Hb, I, T, U>(
        &self,
        hidden: VarDiff<Hf, Hb>,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix2> + Forward, impl Gradient<Dim = Ix2> + Overwrite + Backward>
    where
        Hf: Data<Dim = Ix2>,
        Hb: Gradient<Dim = Ix2> + Overwrite,
        I: MatMatMulT<Learnable<Ix2>>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix2>,
        U: Gradient<Dim = Ix2> + Overwrite,
    {
        let (igates, hgates) = {
            (
                input.mm_t(self.weight_ih.clone()).into() + self.bias_ih.clone(),
                hidden.clone().mm_t(self.weight_hh.clone()) + self.bias_hh.clone(),
            )
        };
        let gate_shape = {
            let (gates_shape_rows, gates_shape_cols) = hgates.data().dim();
            (gates_shape_rows, gates_shape_cols / 3)
        };
        let (chunked_igates, chunked_hgates) =
            (igates.chunks(gate_shape), hgates.chunks(gate_shape));

        let reset_gate = (chunked_hgates[0].clone() + chunked_igates[0].clone()).sigmoid();
        let input_gate = (chunked_hgates[1].clone() + chunked_igates[1].clone()).sigmoid();
        let new_gate =
            (chunked_igates[2].clone() + (chunked_hgates[2].clone() * reset_gate)).tanh();
        (hidden - new_gate.clone()) * input_gate + new_gate
    }
}

impl Register for GRUCell {
    /// Registers the weights and the biases of this `GRUCell` instance.
    fn register_params(&self, params: &mut Vec<Param>) {
        params.extend(self.weight_hh.parameters());
        params.extend(self.weight_ih.parameters());
        params.extend(self.bias_hh.parameters());
        params.extend(self.bias_ih.parameters());
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}

/// Applies a **temporal convolution** over an input signal composed of several input planes.
///
/// See also [`GroupedConv1d`].
pub struct Conv1d<Pad: PaddingMode> {
    pub padding: usize,
    pub padding_mode: Pad,
    pub stride: usize,
    pub dilation: usize,
    pub weight: Learnable<Ix3>,
    pub bias: Learnable<Ix1>,
}

impl<Pad: PaddingMode> Conv1d<Pad> {
    /// Creates a new Conv1d.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - number of planes in the input signal.
    ///
    /// * `out_channels` - number of planes in the output signal.
    ///
    /// * `kernel_size` - size of the kernel, a number for this one-dimensional case.
    ///
    /// * `padding` - padding to be applied to the input, a number for this one-dimensional case.
    ///
    /// * `padding_mode` - padding mode, it can be: [`Zero`], [`Constant`], [`Reflective`] or
    /// [`Replicative`].
    ///
    /// * `stride` - stride of the convolution, a number for this one-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a number for this
    /// one-dimensional case.
    ///
    /// The weight and the bias of the layer are initialised from *U(-k, k)* where
    /// `k = (1. /(in_channels * kernel_size) as f32).sqrt()`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        padding_mode: Pad,
        stride: usize,
        dilation: usize,
    ) -> Self {
        let mut weight =
            Input::new(Tensor::zeros((out_channels, in_channels, kernel_size))).requires_grad();
        let mut bias = Input::new(Tensor::zeros(out_channels)).requires_grad();

        let k = (1. / (in_channels * kernel_size) as f32).sqrt();
        init::uniform(&mut weight, -k, k);
        init::uniform(&mut bias, -k, k);

        Self {
            padding,
            padding_mode,
            stride,
            dilation,
            weight,
            bias,
        }
    }

    /// Computes a 1-dimensional convolution *(cross correlation)*.
    ///
    /// # Arguments
    ///
    /// `input` - signal to convolve.
    ///
    /// The **input** must be of shape *(N, Cin, L)*
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **L** is the **length** of the input
    ///
    /// The **kernel** must be of shape *(Cout, Cin, Lk)*
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Lk** is the **length** of the kernel
    ///
    /// The resulting output shape will be *(N, Cout, Lout)*
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix3> + Forward, impl Gradient<Dim = Ix3> + Overwrite + Backward>
    where
        I: Convolve<I, Learnable<Ix3>, Pad>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix3>,
        U: Gradient<Dim = Ix3> + Overwrite,
    {
        I::convolve(
            input,
            self.weight.clone(),
            &[self.stride],
            &[self.dilation],
            &[self.padding],
            self.padding_mode.clone(),
        )
        .into()
            + self.bias.clone()
    }
}

impl<Pad: PaddingMode> Register for Conv1d<Pad> {
    /// Registers the weight and the bias of this `Conv1d` instance.
    fn register_params(&self, params: &mut Vec<Param>) {
        params.extend(self.weight.parameters());
        params.extend(self.bias.parameters());
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}

/// Applies a **grouped temporal convolution** over an input signal composed of several input
/// planes.
pub struct GroupedConv1d<Pad: PaddingMode> {
    pub padding: usize,
    pub padding_mode: Pad,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub weight: Learnable<Ix3>,
    pub bias: Learnable<Ix1>,
}

impl<Pad: PaddingMode> GroupedConv1d<Pad> {
    /// Creates a new GroupedConv1d.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - number of planes in the input signal.
    ///
    /// * `out_channels` - number of planes in the output signal.
    ///
    /// * `kernel_size` - size of the kernel, a number for this one-dimensional case.
    ///
    /// * `padding` - padding to be applied to the input, a number for this one-dimensional case.
    ///
    /// * `padding_mode` - padding mode, it can be: [`Zero`], [`Constant`], [`Reflective`] or
    /// [`Replicative`].
    ///
    /// * `stride` - stride of the convolution, a number for this one-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a number for this
    /// one-dimensional case.
    ///
    /// * `groups` -  controls the connections between inputs and outputs. `in_channels` and
    /// `out_channels` must both be **divisible by groups**.
    ///
    /// For example:
    /// * at `groups = 1`, all inputs are convolved to all outputs.
    /// * at `groups = 2`, the operation becomes equivalent to having two convolutional layers side
    /// by side, each seeing half the input channels and producing half the output channels, and
    /// both subsequently concatenated.
    ///* at `groups = in_channels`, each input channel is convolved with its own set of filters.
    ///
    /// The weight and the bias of the layer are initialised from *U(-k, k)* where
    /// `k = (groups /(in_channels * kernel_size) as f32).sqrt()`.
    #[allow(clippy::clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        padding_mode: Pad,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Self {
        let mut weight = Input::new(Tensor::zeros((
            out_channels,
            in_channels / groups,
            kernel_size,
        )))
        .requires_grad();
        let mut bias = Input::new(Tensor::zeros(out_channels)).requires_grad();

        let k = (groups as f32 / (in_channels * kernel_size) as f32).sqrt();
        init::uniform(&mut weight, -k, k);
        init::uniform(&mut bias, -k, k);

        Self {
            padding,
            padding_mode,
            stride,
            dilation,
            groups,
            weight,
            bias,
        }
    }

    /// Computes a 1-dimensional grouped convolution *(cross correlation)*.
    ///
    /// # Arguments
    ///
    /// `input` - signal to convolve.
    ///
    /// The **input** must be of shape *(N, Cin, L)*
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **L** is the **length** of the input
    ///
    /// The **kernel** must be of shape *(Cout, Cin, Lk)*
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Lk** is the **length** of the kernel
    ///
    /// The resulting output shape will be *(N, Cout, Lout)*
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix3> + Forward, impl Gradient<Dim = Ix3> + Overwrite + Backward>
    where
        I: ConvolveWithGroups<I, Learnable<Ix3>, Pad>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix3>,
        U: Gradient<Dim = Ix3> + Overwrite,
    {
        I::convolve_with_groups(
            input,
            self.weight.clone(),
            &[self.stride],
            &[self.dilation],
            &[self.padding],
            self.padding_mode.clone(),
            self.groups,
        )
        .into()
            + self.bias.clone()
    }
}

impl<Pad: PaddingMode> Register for GroupedConv1d<Pad> {
    /// Registers the weight and the bias of this `GroupedConv1d` instance.
    fn register_params(&self, params: &mut Vec<Param>) {
        params.extend(self.weight.parameters());
        params.extend(self.bias.parameters());
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}

/// Applies a **spatial convolution** over an input signal composed of several input planes.
///
/// See also [`GroupedConv2d`].
pub struct Conv2d<Pad: PaddingMode> {
    pub padding: (usize, usize),
    pub padding_mode: Pad,
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
    pub weight: Learnable<Ix4>,
    pub bias: Learnable<Ix1>,
}

impl<Pad: PaddingMode> Conv2d<Pad> {
    /// Creates a new Conv2d.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - number of planes in the input signal.
    ///
    /// * `out_channels` - number of planes in the output signal.
    ///
    /// * `kernel_size` - size of the kernel, a 2-tuple for this two-dimensional case.
    ///
    /// * `padding` - padding to be applied to the input, a 2-tuple for this two-dimensional case.
    ///
    /// * `padding_mode` - padding mode, it can be: [`Zero`], [`Constant`], [`Reflective`] or
    /// [`Replicative`].
    ///
    /// * `stride` - stride of the convolution, a 2-tuple for this two-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 2-tuple for this
    /// two-dimensional case.
    ///
    /// The weight and the bias are initialised from *U(-k, k)* where
    /// `k = (1. /(in_channels * kernel_w * kernel_h) as f32).sqrt()`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        padding: (usize, usize),
        padding_mode: Pad,
        stride: (usize, usize),
        dilation: (usize, usize),
    ) -> Self {
        let (kernel_h, kernel_w) = kernel_size;
        let mut weight = Input::new(Tensor::zeros((
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
        )))
        .requires_grad();
        let mut bias = Input::new(Tensor::zeros(out_channels)).requires_grad();

        let k = (1. / (in_channels * kernel_h * kernel_w) as f32).sqrt();
        init::uniform(&mut weight, -k, k);
        init::uniform(&mut bias, -k, k);

        Self {
            padding,
            padding_mode,
            stride,
            dilation,
            weight,
            bias,
        }
    }

    /// Computes a 2-dimensional convolution *(cross correlation)*.
    ///
    /// # Arguments
    ///
    /// `input` - the signal to convolve.
    ///
    /// The **input** must be of shape *(N, Cin, H, W)*
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape *(Cout, Cin, Hk, Wk)*
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be *(N, Cout, Hout, Wout)*
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix4> + Forward, impl Gradient<Dim = Ix4> + Overwrite + Backward>
    where
        I: Convolve<I, Learnable<Ix4>, Pad>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix4>,
        U: Gradient<Dim = Ix4> + Overwrite,
    {
        let (stride_h, stride_w) = self.stride;
        let (padding_h, padding_w) = self.padding;
        let (dilation_h, dilation_w) = self.dilation;

        I::convolve(
            input,
            self.weight.clone(),
            &[stride_h, stride_w],
            &[dilation_h, dilation_w],
            &[padding_h, padding_w],
            self.padding_mode.clone(),
        )
        .into()
            + self.bias.clone()
    }
}

impl<Pad: PaddingMode> Register for Conv2d<Pad> {
    /// Registers the weight and the bias of this `Conv2d` instance.
    fn register_params(&self, params: &mut Vec<Param>) {
        params.extend(self.weight.parameters());
        params.extend(self.bias.parameters());
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}

/// Applies a **spatial grouped convolution** over an input signal composed of several input planes.
pub struct GroupedConv2d<Pad: PaddingMode> {
    pub padding: (usize, usize),
    pub padding_mode: Pad,
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
    pub weight: Learnable<Ix4>,
    pub bias: Learnable<Ix1>,
}

impl<Pad: PaddingMode> GroupedConv2d<Pad> {
    /// Creates a new GroupedConv2d.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - number of planes in the input signal.
    ///
    /// * `out_channels` - number of planes in the output signal.
    ///
    /// * `kernel_size` - size of the kernel, a 2-tuple  for this two-dimensional case.
    ///
    /// * `padding` - padding to be applied to the input, a 2-tuple  for this two-dimensional case.
    ///
    /// * `padding_mode` - padding mode, it can be: [`Zero`], [`Constant`], [`Reflective`] or
    /// [`Replicative`].
    ///
    /// * `stride` - stride of the convolution, a 2-tuple  for this two-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 2-tuple  for this
    /// two-dimensional case.
    ///
    /// * `groups` -  controls the connections between inputs and outputs. `in_channels` and
    /// `out_channels` must both be divisible by groups.
    ///
    /// For example:
    /// * at `groups = 1`, all inputs are convolved to all outputs.
    /// *  at `groups = 2`, the operation becomes equivalent to having two convolutional layers
    /// side by side, each seeing half the input channels and producing half the output channels,
    /// and both subsequently concatenated.
    /// * at `groups = in_channels`, each input channel is convolved with its own set of filters.
    ///
    /// The weight and the bias of the layer are initialised from *U(-k, k)* where
    /// `k = (groups /(in_channels * kernel_h * kernel_w) as f32).sqrt()`.
    #[allow(clippy::clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        padding: (usize, usize),
        padding_mode: Pad,
        stride: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Self {
        let (kernel_h, kernel_w) = kernel_size;
        let mut weight = Input::new(Tensor::zeros((
            out_channels,
            in_channels,
            kernel_h,
            kernel_w,
        )))
        .requires_grad();
        let mut bias = Input::new(Tensor::zeros(out_channels)).requires_grad();

        let k = (groups as f32 / (in_channels * kernel_h * kernel_w) as f32).sqrt();
        init::uniform(&mut weight, -k, k);
        init::uniform(&mut bias, -k, k);

        Self {
            padding,
            padding_mode,
            stride,
            dilation,
            groups,
            weight,
            bias,
        }
    }

    /// Computes a 2-dimensional grouped convolution *(cross correlation)*.
    ///
    /// # Arguments
    ///
    /// `input` - the signal to convolve.
    ///
    /// The **input** must be of shape *(N, Cin, H, W)*
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape *(Cout, Cin, Hk, Wk)*
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be *(N, Cout, Hout, Wout)*
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix4> + Forward, impl Gradient<Dim = Ix4> + Overwrite + Backward>
    where
        I: ConvolveWithGroups<I, Learnable<Ix4>, Pad>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix4>,
        U: Gradient<Dim = Ix4> + Overwrite,
    {
        let (stride_h, stride_w) = self.stride;
        let (padding_h, padding_w) = self.padding;
        let (dilation_h, dilation_w) = self.dilation;

        I::convolve_with_groups(
            input,
            self.weight.clone(),
            &[stride_h, stride_w],
            &[dilation_h, dilation_w],
            &[padding_h, padding_w],
            self.padding_mode.clone(),
            self.groups,
        )
        .into()
            + self.bias.clone()
    }
}

impl<Pad: PaddingMode> Register for GroupedConv2d<Pad> {
    /// Registers the weight and the bias of this `GroupedConv2d` instance.
    fn register_params(&self, params: &mut Vec<Param>) {
        params.extend(self.weight.parameters());
        params.extend(self.bias.parameters());
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}

/// Applies a **volumetric convolution** over an input signal composed of several input planes.
///
/// See also [`GroupedConv3d`].
pub struct Conv3d<Pad: PaddingMode> {
    pub padding: (usize, usize, usize),
    pub padding_mode: Pad,
    pub stride: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub weight: Learnable<Ix5>,
    pub bias: Learnable<Ix1>,
}

impl<Pad: PaddingMode> Conv3d<Pad> {
    /// Creates a new Conv3d.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - number of planes in the input signal.
    ///
    /// * `out_channels` - number of planes in the output signal.
    ///
    /// * `kernel_size` - size of the kernel, a 3-tuple for this three-dimensional case.
    ///
    /// * `padding` - padding to be applied to the input, a 3-tuple for this three-dimensional case.
    ///
    /// * `padding_mode` - padding mode, it can be: [`Zero`], [`Constant`], [`Reflective`] or
    /// [`Replicative`].
    ///
    /// * `stride` - stride of the convolution, a 3-tuple for this three-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 3-tuple for this
    /// three-dimensional case.
    ///
    /// The weight and the bias of the layer are initialised from *U(-k, k)* where
    /// `k = (1. /(in_channels * kernel_d * kernel_w * kernel_h) as f32).sqrt()`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        padding: (usize, usize, usize),
        padding_mode: Pad,
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Self {
        let (kernel_d, kernel_h, kernel_w) = kernel_size;
        let mut weight = Input::new(Tensor::zeros((
            out_channels,
            in_channels,
            kernel_d,
            kernel_h,
            kernel_w,
        )))
        .requires_grad();
        let mut bias = Input::new(Tensor::zeros(out_channels)).requires_grad();

        let k = (1. / (in_channels * kernel_d * kernel_h * kernel_w) as f32).sqrt();
        init::uniform(&mut weight, -k, k);
        init::uniform(&mut bias, -k, k);

        Self {
            padding,
            padding_mode,
            stride,
            dilation,
            weight,
            bias,
        }
    }

    /// Computes a 3-dimensional convolution *(cross correlation)*.
    ///
    /// # Arguments
    ///
    /// `input` - signal to convolve.
    ///
    /// The **input** must be of shape *(N, Cin, D, H, W)*
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **D** is the **depth** of the input
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape *(Cout, Cin, Dk,  Hk, Wk)*
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Dk** is the **depth** of the kernel
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be *(N, Cout, Dout, Hout, Wout)*
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix5> + Forward, impl Gradient<Dim = Ix5> + Overwrite + Backward>
    where
        I: Convolve<I, Learnable<Ix5>, Pad>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix5>,
        U: Gradient<Dim = Ix5> + Overwrite,
    {
        let (stride_d, stride_h, stride_w) = self.stride;
        let (padding_d, padding_h, padding_w) = self.padding;
        let (dilation_d, dilation_h, dilation_w) = self.dilation;

        I::convolve(
            input,
            self.weight.clone(),
            &[stride_d, stride_h, stride_w],
            &[dilation_d, dilation_h, dilation_w],
            &[padding_d, padding_h, padding_w],
            self.padding_mode.clone(),
        )
        .into()
            + self.bias.clone()
    }
}

impl<Pad: PaddingMode> Register for Conv3d<Pad> {
    /// Registers the weight and the bias of this `Conv3d` instance.
    fn register_params(&self, params: &mut Vec<Param>) {
        params.extend(self.weight.parameters());
        params.extend(self.bias.parameters());
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}

/// Applies a **grouped volumetric convolution** over an input signal composed of several input
/// planes.
pub struct GroupedConv3d<Pad: PaddingMode> {
    pub padding: (usize, usize, usize),
    pub padding_mode: Pad,
    pub stride: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub groups: usize,
    pub weight: Learnable<Ix5>,
    pub bias: Learnable<Ix1>,
}

impl<Pad: PaddingMode> GroupedConv3d<Pad> {
    /// Creates a new GroupedConv3d.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - number of planes in the input signal.
    ///
    /// * `out_channels` - number of planes in the output signal.
    ///
    /// * `kernel_size` - size of the kernel, a 3-tuple  for this three-dimensional case.
    ///
    /// * `padding` - padding to be applied to the input, a 3-tuple  for this three-dimensional case.
    ///
    /// * `padding_mode` - padding mode, it can be: [`Zero`], [`Constant`], [`Reflective`] or
    /// [`Replicative`].
    ///
    /// * `stride` - stride of the convolution, a 3-tuple  for this three-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 3-tuple  for this
    /// three-dimensional case.
    ///
    /// * `groups` -  controls the connections between inputs and outputs. `in_channels` and
    /// `out_channels` must both be divisible by groups.
    ///
    /// For example:
    /// * at `groups = 1`, all inputs are convolved to all outputs.
    /// *  at `groups = 2`, the operation becomes equivalent to having two convolutional layers
    /// side by side, each seeing half the input channels and producing half the output channels,
    /// and both subsequently concatenated.
    /// * at `groups = in_channels`, each input channel is convolved with its own set of filters.
    ///
    /// The weight and the bias are initialised from *U(-k, k)* where
    /// `k = (groups /(in_channels * kernel_d * kernel_h * kernel_w) as f32).sqrt()`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        padding: (usize, usize, usize),
        padding_mode: Pad,
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Self {
        let (kernel_d, kernel_h, kernel_w) = kernel_size;
        let mut weight = Input::new(Tensor::zeros((
            out_channels,
            in_channels,
            kernel_d,
            kernel_h,
            kernel_w,
        )))
        .requires_grad();
        let mut bias = Input::new(Tensor::zeros(out_channels)).requires_grad();

        let k = (1. / (in_channels * kernel_d * kernel_h * kernel_w) as f32).sqrt();
        init::uniform(&mut weight, -k, k);
        init::uniform(&mut bias, -k, k);

        Self {
            padding,
            padding_mode,
            stride,
            dilation,
            groups,
            weight,
            bias,
        }
    }

    /// Computes a 3-dimensional grouped convolution *(cross correlation)*.
    ///
    /// `input` - the signal to convolve.
    ///
    /// The **input** must be of shape *(N, Cin, D, H, W)*
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **D** is the **depth** of the input
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape *(Cout, Cin, Dk,  Hk, Wk)*
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Dk** is the **depth** of the kernel
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be *(N, Cout, Dout, Hout, Wout)*
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix5> + Forward, impl Gradient<Dim = Ix5> + Overwrite + Backward>
    where
        I: ConvolveWithGroups<I, Learnable<Ix5>, Pad>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix5>,
        U: Gradient<Dim = Ix5> + Overwrite,
    {
        let (stride_d, stride_h, stride_w) = self.stride;
        let (padding_d, padding_h, padding_w) = self.padding;
        let (dilation_d, dilation_h, dilation_w) = self.dilation;

        I::convolve_with_groups(
            input,
            self.weight.clone(),
            &[stride_d, stride_h, stride_w],
            &[dilation_d, dilation_h, dilation_w],
            &[padding_d, padding_h, padding_w],
            self.padding_mode.clone(),
            self.groups,
        )
        .into()
            + self.bias.clone()
    }
}

impl<Pad: PaddingMode> Register for GroupedConv3d<Pad> {
    /// Registers the weight and the bias of this `GroupedConv3d` instance.
    fn register_params(&self, params: &mut Vec<Param>) {
        params.extend(self.weight.parameters());
        params.extend(self.bias.parameters());
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}
