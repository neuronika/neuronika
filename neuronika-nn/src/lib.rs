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

use ndarray::{Ix1, Ix2, Ix3, Ix4, Ix5};

use neuronika_core::{Convolution, MatMatMulT};

use neuronika_variable::{PaddingMode, VarDiff};

pub mod init;

#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

/// Applies a **linear transformation** to the incoming data.
///
/// ```text
/// ʏ = xAᵀ + b
/// ```
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Linear {
    pub weight: VarDiff<Ix2>,
    pub bias: VarDiff<Ix1>,
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
    /// The values for both the weight and bias are initialized from *U(-k, k)* where
    /// `k = (1. / in_features as f32).sqrt()`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let weight = neuronika_variable::zeros((out_features, in_features)).requires_grad();
        let bias = neuronika_variable::zeros(out_features).requires_grad();
        let k = (1. / (in_features as f32)).sqrt();
        init::uniform(&weight, -k, k);
        init::uniform(&bias, -k, k);

        Self { weight, bias }
    }

    /// Applies the linear transformation *y = xA^T + b* to the incoming data.
    ///
    /// # Arguments
    ///
    /// `data` - a variable of shape *(N, in_features)*, the output's shape will be
    /// *(N, out_features)*.
    pub fn forward<I>(&self, input: I) -> VarDiff<Ix2>
    where
        I: MatMatMulT<VarDiff<Ix2>>,
        I::Output: Into<VarDiff<Ix2>>,
    {
        input.mm_t(self.weight.clone()).into() + self.bias.clone()
    }
}

/// A **long short-term memory (LSTM)** cell.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[allow(clippy::upper_case_acronyms)]
pub struct LSTMCell {
    pub weight_ih: VarDiff<Ix2>,
    pub weight_hh: VarDiff<Ix2>,
    pub bias_ih: VarDiff<Ix1>,
    pub bias_hh: VarDiff<Ix1>,
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
    /// All the weight and biases are initialized from *U(-k, k)* where
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
        let weight_ih = neuronika_variable::zeros(weight_ih_shape).requires_grad();
        let weight_hh = neuronika_variable::zeros(weight_hh_shape).requires_grad();
        let bias_ih = neuronika_variable::zeros(bias_shape).requires_grad();
        let bias_hh = neuronika_variable::zeros(bias_shape).requires_grad();

        let k = 1. / (hidden_size as f32).sqrt();
        init::uniform(&weight_ih, -k, k);
        init::uniform(&weight_hh, -k, k);
        init::uniform(&bias_ih, -k, k);
        init::uniform(&bias_hh, -k, k);

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
    pub fn forward<I>(
        &self,
        state: (VarDiff<Ix2>, VarDiff<Ix2>),
        input: I,
    ) -> (VarDiff<Ix2>, VarDiff<Ix2>)
    where
        I: MatMatMulT<VarDiff<Ix2>>,
        I::Output: Into<VarDiff<Ix2>>,
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

/// A **gated recurrent unit (GRU)** cell.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[allow(clippy::upper_case_acronyms)]
pub struct GRUCell {
    pub weight_ih: VarDiff<Ix2>,
    pub weight_hh: VarDiff<Ix2>,
    pub bias_ih: VarDiff<Ix1>,
    pub bias_hh: VarDiff<Ix1>,
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
    /// All the weight and biases are initialized from *U(-k, k)* where
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
        let weight_ih = neuronika_variable::zeros(weight_ih_shape).requires_grad();
        let weight_hh = neuronika_variable::zeros(weight_hh_shape).requires_grad();
        let bias_ih = neuronika_variable::zeros(bias_shape).requires_grad();
        let bias_hh = neuronika_variable::zeros(bias_shape).requires_grad();

        let k = 1. / (hidden_size as f32).sqrt();
        init::uniform(&weight_ih, -k, k);
        init::uniform(&weight_hh, -k, k);
        init::uniform(&bias_ih, -k, k);
        init::uniform(&bias_hh, -k, k);

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    /// Computes a single GRU step.
    ///
    /// * `hidden` - a variable of shape *(batch, hidden_size)*, containing the initial hidden state
    /// for each element in the batch.
    ///
    /// * `input` - a variable containing the input features of shape *(batch, input_size)*.
    ///
    /// The output is a variable made of the next hidden state for each element in
    /// the batch, of shape *(batch, hidden_size)*.
    pub fn forward<I>(&self, hidden: VarDiff<Ix2>, input: I) -> VarDiff<Ix2>
    where
        I: MatMatMulT<VarDiff<Ix2>>,
        I::Output: Into<VarDiff<Ix2>>,
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

/// Applies a temporal convolution over an input signal composed of several input planes.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Conv1d<T>
where
    T: PaddingMode<Ix3>,
{
    pub padding: usize,
    pub padding_mode: T,
    pub stride: usize,
    pub dilation: usize,
    pub weight: VarDiff<Ix3>,
    pub bias: VarDiff<Ix2>,
}

impl<T> Conv1d<T>
where
    T: PaddingMode<Ix3>,
{
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
    /// The weight and the bias of the layer are initialized from *U(-k, k)* where
    /// `k = (1. /(in_channels * kernel_size) as f32).sqrt()`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        padding: usize,
        padding_mode: T,
        stride: usize,
        dilation: usize,
    ) -> Self {
        let weight =
            neuronika_variable::zeros((out_channels, in_channels, kernel_size)).requires_grad();
        let bias = neuronika_variable::zeros((out_channels, 1)).requires_grad();

        let k = (1. / (in_channels * kernel_size) as f32).sqrt();
        init::uniform(&weight, -k, k);
        init::uniform(&bias, -k, k);

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
    pub fn forward<I>(&self, input: I) -> VarDiff<Ix3>
    where
        VarDiff<Ix3>: Convolution<I, Ix3>,
    {
        todo!()
    }
}

/// Applies a **spatial convolution** over an input signal composed of several input planes.
///
/// See also [`GroupedConv2d`].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Conv2d<T>
where
    T: PaddingMode<Ix4>,
{
    pub padding: (usize, usize),
    pub padding_mode: T,
    pub stride: (usize, usize),
    pub dilation: (usize, usize),
    pub weight: VarDiff<Ix4>,
    pub bias: VarDiff<Ix3>,
}

impl<T> Conv2d<T>
where
    T: PaddingMode<Ix4>,
{
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
    /// The weight and the bias are initialized from *U(-k, k)* where
    /// `k = (1. /(in_channels * kernel_w * kernel_h) as f32).sqrt()`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        padding: (usize, usize),
        padding_mode: T,
        stride: (usize, usize),
        dilation: (usize, usize),
    ) -> Self {
        let (kernel_h, kernel_w) = kernel_size;
        let weight = neuronika_variable::zeros((out_channels, in_channels, kernel_h, kernel_w))
            .requires_grad();
        let bias = neuronika_variable::zeros((out_channels, 1, 1)).requires_grad();

        let k = (1. / (in_channels * kernel_h * kernel_w) as f32).sqrt();
        init::uniform(&weight, -k, k);
        init::uniform(&bias, -k, k);

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
    pub fn forward<I>(&self, input: I) -> VarDiff<Ix4>
    where
        VarDiff<Ix4>: Convolution<I, Ix4>,
    {
        todo!()
    }
}

/// Applies a **volumetric convolution** over an input signal composed of several input planes.
///
/// See also [`GroupedConv3d`].
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Conv3d<T>
where
    T: PaddingMode<Ix5>,
{
    pub padding: (usize, usize, usize),
    pub padding_mode: T,
    pub stride: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub weight: VarDiff<Ix5>,
    pub bias: VarDiff<Ix4>,
}

impl<T> Conv3d<T>
where
    T: PaddingMode<Ix5>,
{
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
    /// The weight and the bias of the layer are initialized from *U(-k, k)* where
    /// `k = (1. /(in_channels * kernel_d * kernel_w * kernel_h) as f32).sqrt()`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        padding: (usize, usize, usize),
        padding_mode: T,
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Self {
        let (kernel_d, kernel_h, kernel_w) = kernel_size;
        let weight =
            neuronika_variable::zeros((out_channels, in_channels, kernel_d, kernel_h, kernel_w))
                .requires_grad();
        let bias = neuronika_variable::zeros((out_channels, 1, 1, 1)).requires_grad();

        let k = (1. / (in_channels * kernel_d * kernel_h * kernel_w) as f32).sqrt();
        init::uniform(&weight, -k, k);
        init::uniform(&bias, -k, k);

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
    pub fn forward<I>(&self, input: I) -> VarDiff<Ix5>
    where
        VarDiff<Ix5>: Convolution<I, Ix5>,
        <VarDiff<Ix5> as Convolution<I, Ix5>>::Output: Into<VarDiff<Ix5>>,
    {
        todo!()
    }
}
