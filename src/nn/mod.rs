use convolution::{Convolve, ConvolveWithGroups, PaddingMode};

use super::{Input, InputBackward};
use crate::variable::{
    self,
    node::{Backward, Data, Forward, Gradient, Overwrite},
    MatMatMulT, Tensor, Var, VarDiff,
};
use ndarray::{Ix1, Ix2, Ix3, Ix4, Ix5};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ init module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub mod init {
    use super::super::{variable::VarDiff, Input, InputBackward};
    use ndarray::{Axis, Dimension, Ix2};
    use rand::thread_rng;
    use rand_distr::{Distribution, Normal, Uniform};

    /// Returns the recommended gain value for the given nonlinearity function.
    pub fn calculate_gain(non_linearity: &str) -> f32 {
        match non_linearity {
            "linear" | "sigmoid" => 1.0,
            "tanh" => 5.0 / 3.0,
            "relu" => 2.0_f32.sqrt(),
            "leaky_relu" => (2.0 / (1.0 + 0.01_f32.powi(2))).sqrt(),
            _ => panic!("error: unsupported nonlinearity: {}", non_linearity),
        }
    }

    /// For MLPs `fan_in` and `fan_out` are respectively the number of
    /// inputs and outputs to an hidden unit of the layer.
    /// For CNNs however, the number of input feature maps and
    /// the size of the receptive field must be taken into account .
    fn calculate_fan_in_fan_out<D: Dimension>(
        param: &VarDiff<Input<D>, InputBackward<D>>,
    ) -> (f32, f32) {
        let data = param.data();
        let shape = data.shape();

        let num_input_fmaps = shape[1] as f32;
        let num_output_fmaps = shape[0] as f32;
        let mut receptive_field_size = 1.;

        let no_dim = data.ndim();
        let mut num_el = 0;
        if no_dim > 2 {
            for dim in 2..no_dim {
                num_el += data.len_of(Axis(dim));
            }
            receptive_field_size = num_el as f32;
        }

        let fan_in = num_input_fmaps * receptive_field_size;
        let fan_out = num_output_fmaps * receptive_field_size;
        (fan_in, fan_out)
    }

    /// Fills the input with `value`.
    pub fn constant<D: Dimension>(param: &mut VarDiff<Input<D>, InputBackward<D>>, value: f32) {
        param.data_mut().map_inplace(|el| *el = value);
    }

    /// Fills the input with `0.0`.
    pub fn zeros<D: Dimension>(param: &mut VarDiff<Input<D>, InputBackward<D>>) {
        param.data_mut().map_inplace(|el| *el = 0.);
    }

    /// Fills the input with `1.0`.
    pub fn ones<D: Dimension>(param: &mut VarDiff<Input<D>, InputBackward<D>>) {
        param.data_mut().map_inplace(|el| *el = 1.0);
    }

    /// Fills the 2-dimensional input with the identity matrix.
    /// Preserves the identity of the inputs in Linear layers, where as
    /// many inputs are preserved as possible.
    pub fn eye(param: &mut VarDiff<Input<Ix2>, InputBackward<Ix2>>) {
        for ((x, y), el) in param.data_mut().indexed_iter_mut() {
            if x == y {
                *el = 1.
            } else {
                *el = 0.
            }
        }
    }

    /// Fills the *{3, 4, 5}-dimensional* parameter with the Dirac delta function. Preserves the
    /// identity of the inputs in convolutional layers, where as many input channels
    /// are preserved as possible. In case of `groups` > 1, each group of channels preserves
    /// identity.
    pub fn dirac<D: Dimension>(param: &mut VarDiff<Input<D>, InputBackward<D>>, groups: usize) {
        let mut data = param.data_mut();
        let shape = data.shape().to_vec();
        let no_dim = shape.len();

        if !(3..=5).contains(&no_dim) {
            panic!("error: only 3, 4 and 5 dimensional parameters are supported.");
        }
        assert_eq!(
            shape[0].rem_euclid(groups),
            0,
            "error: output channels must be divisible by groups."
        );
        let out_channels_per_groups = shape[0] / groups;
        let min_dim = out_channels_per_groups.min(shape[1]);

        for g in 0..groups {
            for d in 0..min_dim {
                let mut index = D::zeros(no_dim);
                index[0] = g * out_channels_per_groups + d;
                index[1] = d;
                index
                    .slice_mut()
                    .iter_mut()
                    .skip(2)
                    .zip(shape.iter().skip(2))
                    .for_each(|(el, sh)| *el = sh / 2);
                data[index] = 1.
            }
        }
    }

    /// Fills the input with elements drawn from
    /// the uniform distribution U(low, high).
    pub fn uniform<D: Dimension>(
        param: &mut VarDiff<Input<D>, InputBackward<D>>,
        low: f32,
        high: f32,
    ) {
        let unif_dstr = Uniform::new(low, high);
        let mut t_rng = thread_rng();
        param
            .data_mut()
            .map_inplace(|el| *el = unif_dstr.sample(&mut t_rng));
    }

    /// Fills the input with elements drawn from
    /// the normal distribution N(mean, std^2).
    pub fn normal<D: Dimension>(
        param: &mut VarDiff<Input<D>, InputBackward<D>>,
        mean: f32,
        std: f32,
    ) {
        let norm_dstr = Normal::new(mean, std).unwrap();
        let mut t_rng = thread_rng();
        param
            .data_mut()
            .map_inplace(|el| *el = norm_dstr.sample(&mut t_rng));
    }

    /// Fills the input with values according to the method
    /// described in Understanding the difficulty of training deep feedforward
    /// neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform
    /// distribution.
    pub fn xavier_uniform<D: Dimension>(
        param: &mut VarDiff<Input<D>, InputBackward<D>>,
        gain: f32,
    ) {
        let (fan_in, fan_out) = calculate_fan_in_fan_out(param);
        let std = gain * (2. / ((fan_in + fan_out) as f32)).sqrt();
        let a = 3.0_f32.sqrt() * std;
        let unif_distr = Uniform::new(-a, a);
        let mut t_rng = thread_rng();
        param
            .data_mut()
            .map_inplace(|el| *el = unif_distr.sample(&mut t_rng));
    }

    /// Fills the input with values according to the method
    /// described in Understanding the difficulty of training deep feedforward
    /// neural networks - Glorot, X. & Bengio, Y. (2010), using a normal
    /// distribution.
    ///
    /// Also known as Glorot initialization.
    pub fn xavier_normal<D: Dimension>(param: &mut VarDiff<Input<D>, InputBackward<D>>, gain: f32) {
        let (fan_in, fan_out) = calculate_fan_in_fan_out(param);
        let std = gain * (2. / ((fan_in + fan_out) as f32)).sqrt();
        let norm_distr = Normal::new(0., std).unwrap();
        let mut t_rng = thread_rng();
        param
            .data_mut()
            .map_inplace(|el| *el = norm_distr.sample(&mut t_rng));
    }
}

pub mod convolution;
pub mod loss;

/// Applies a **linear transformation** to the incoming data.
///
/// ```text
/// ʏ = xAᵀ + b
/// ```
pub struct Linear {
    pub weight: VarDiff<Input<Ix2>, InputBackward<Ix2>>,
    pub bias: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
}

impl Linear {
    /// Creates a linear layer.
    ///
    /// `in_features` – size of each input sample.
    ///
    /// `out_features` – size of each output sample.
    ///
    /// The learnable weight of the layer is of shape `(out_features, in_features)`. The values
    /// are initialised from **U(-k, k)** where `k = (1. / in_features as f32).sqrt()`.
    ///
    /// The learnable bias of the layer is of shape `out_features`. The values
    /// are initialised from **U(-k, k)** where `k = (1. / in_features as f32).sqrt()`.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weight = Input::new(Tensor::zeros((out_features, in_features))).requires_grad();
        let mut bias = Input::new(Tensor::zeros(out_features)).requires_grad();
        let k = (1. / (in_features as f32)).sqrt();
        init::uniform(&mut weight, -k, k);
        init::uniform(&mut bias, -k, k);

        Self { weight, bias }
    }

    /// Applies the linear transformation **y = xA^T + b** to the incoming data.
    ///
    /// `data` - `(N, in_features)`, the output will be `(N, out_features)`.
    pub fn forward<W, T, U>(
        &self,
        input: W,
    ) -> VarDiff<impl Data<Dim = Ix2>, impl Gradient<Dim = Ix2> + Overwrite>
    where
        W: MatMatMulT<VarDiff<Input<Ix2>, InputBackward<Ix2>>>,
        W::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix2>,
        U: Gradient<Dim = Ix2> + Overwrite,
    {
        input.mm_mul_t(self.weight.clone()).into() + self.bias.clone()
    }
}

/// A **long short-term memory (LSTM)** cell.
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct LSTMCell {
    pub weight_ih: VarDiff<Input<Ix2>, InputBackward<Ix2>>,
    pub weight_hh: VarDiff<Input<Ix2>, InputBackward<Ix2>>,
    pub bias_ih: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
    pub bias_hh: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
}

impl LSTMCell {
    /// Creates a new LSTMCell.
    ///
    /// `input_size` - The number of expected features in the input.
    ///
    /// `hidden_size` - The number of features in the hidden state.
    ///
    /// All the weight and biases are initialised from **U(-k, k)** where
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
    /// `state` - a tuple of tensors, both of shape `(batch, hidden_size)`, containing the
    /// initial hidden state for each element in the batch and the initial cell's state for
    /// each element in the batch.
    ///
    /// `input` - a tensor containing the input features of shape `(batch, input_size)`.
    ///
    /// The **output** is a tuple of tensors made of the next hidden state for each element in
    /// the batch, of shape `(batch, hidden_size)` and the next cell's state for each element in
    /// the batch, of shape `(batch, hidden_size)`.
    pub fn forward<Cf, Cb, Hf, Hb, I, T, U>(
        &self,
        state: (VarDiff<Cf, Cb>, VarDiff<Hf, Hb>),
        input: I,
    ) -> (
        VarDiff<impl Data<Dim = Ix2>, impl Gradient<Dim = Ix2> + Overwrite>,
        VarDiff<impl Data<Dim = Ix2>, impl Gradient<Dim = Ix2> + Overwrite>,
    )
    where
        Cf: Data<Dim = Ix2>,
        Cb: Gradient<Dim = Ix2> + Overwrite,
        Hf: Data<Dim = Ix2>,
        Hb: Gradient<Dim = Ix2> + Overwrite,
        I: MatMatMulT<VarDiff<Input<Ix2>, InputBackward<Ix2>>>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix2>,
        U: Gradient<Dim = Ix2> + Overwrite,
    {
        let (cell_state, hidden) = state;
        let gates = hidden.mm_mul_t(self.weight_hh.clone())
            + self.bias_hh.clone()
            + input.mm_mul_t(self.weight_ih.clone()).into()
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
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct GRUCell {
    pub weight_ih: VarDiff<Input<Ix2>, InputBackward<Ix2>>,
    pub weight_hh: VarDiff<Input<Ix2>, InputBackward<Ix2>>,
    pub bias_ih: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
    pub bias_hh: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
}

impl GRUCell {
    /// Creates a new GRUCell.
    ///
    /// `input_size` - The number of expected features in the input.
    ///
    /// `hidden_size` - The number of features in the hidden state.
    ///
    /// All the weight and biases are initialised from **U(-k, k)** where
    /// `k = (1. /hidden_size as f32).sqrt()`.
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
    /// `hidden` - a tensor of shape `(batch, hidden_size)`, containing the initial hidden state
    /// for each element in the batch.
    ///
    /// `input` - a tensor containing the input features of shape `(batch, input_size)`.
    ///
    /// The **output** is  a tensor made of the next hidden state for each element in
    /// the batch, of shape `(batch, hidden_size)`.
    pub fn forward<Hf, Hb, I, T, U>(
        &self,
        hidden: VarDiff<Hf, Hb>,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix2>, impl Gradient<Dim = Ix2> + Overwrite>
    where
        Hf: Data<Dim = Ix2>,
        Hb: Gradient<Dim = Ix2> + Overwrite,
        I: MatMatMulT<VarDiff<Input<Ix2>, InputBackward<Ix2>>>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix2>,
        U: Gradient<Dim = Ix2> + Overwrite,
    {
        let (igates, hgates) = {
            (
                input.mm_mul_t(self.weight_ih.clone()).into() + self.bias_ih.clone(),
                hidden.clone().mm_mul_t(self.weight_hh.clone()) + self.bias_hh.clone(),
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

/// Applies a **temporal convolution** over an input signal composed of several input planes.
pub struct Conv1d<Pad: PaddingMode> {
    padding: usize,
    padding_mode: Pad,
    stride: usize,
    dilation: usize,
    weight: VarDiff<Input<Ix3>, InputBackward<Ix3>>,
    bias: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
}

impl<Pad: PaddingMode> Conv1d<Pad> {
    /// Creates a new Conv1d.
    ///
    /// * `in_channels` - the number of planes in the input signal.
    ///
    /// * `out_channels` - the number of planes in the output signal.
    ///
    /// * `kernel_size` - the size of the kernel, a number for this one-dimensional case.
    ///
    /// * `padding` - the padding to be applied to the input, a number for this one-dimensional
    /// case.
    ///
    /// * `padding_mode` - the padding mode, it can be: **zeros**, **constant**, **reflective** or
    /// **replicative**.
    ///
    /// * `stride` - the stride of the convolution, a number for this one-dimensional case
    ///
    /// * `dilation` - controls the spacing between the kernel points, a number for this
    /// one-dimensional case
    ///
    /// The weight and the bias of the layer are initialised from **U(-k, k)** where
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
    /// * `input` - the signal to convolve.
    ///
    /// The **input** must be of shape **(N, Cin, L)**
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **L** is the **length** of the input
    ///
    /// The **kernel** must be of shape **(Cout, Cin, Lk)**
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Lk** is the **length** of the kernel
    ///
    /// The resulting output shape will be **(N, Cout, Lout)**
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix3> + Forward, impl Gradient<Dim = Ix3> + Backward + Overwrite>
    where
        I: Convolve<I, VarDiff<Input<Ix3>, InputBackward<Ix3>>, Pad>,
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

/// Applies a **grouped temporal convolution** over an input signal composed of several input
/// planes.
pub struct GroupedConv1d<Pad: PaddingMode> {
    padding: usize,
    padding_mode: Pad,
    stride: usize,
    dilation: usize,
    groups: usize,
    weight: VarDiff<Input<Ix3>, InputBackward<Ix3>>,
    bias: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
}

impl<Pad: PaddingMode> GroupedConv1d<Pad> {
    /// Creates a new GroupedConv1d.
    ///
    /// * `in_channels` - the number of planes in the input signal.
    ///
    /// * `out_channels` - the number of planes in the output signal.
    ///
    /// * `kernel_size` - the size of the kernel, a number for this one-dimensional case.
    ///
    /// * `padding` - the padding to be applied to the input, a number for this one-dimensional
    /// case.
    ///
    /// * `padding_mode` - the padding mode, it can be: **zeros**, **constant**, **reflective** or
    /// **replicative**.
    ///
    /// * `stride` - the stride of the convolution, a number for this one-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a number for this
    /// one-dimensional case.
    ///
    /// * `groups` -  controls the connections between inputs and outputs. `in_channels` and
    /// `out_channels` must both be **divisible by groups**.
    ///
    ///For example:
    /// * at **groups = 1**, all inputs are convolved to all outputs.
    /// * at **groups = 2**, the operation becomes equivalent to having two convolutional layers side
    /// by side, each seeing half the input channels and producing half the output channels, and
    /// both subsequently concatenated.
    ///* at **groups = in_channels**, each input channel is convolved with its own set of filters.
    ///
    /// The weight and the bias of the layer are initialised from **U(-k, k)** where
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
    /// * `input` - the signal to convolve.
    ///
    /// The **input** must be of shape **(N, Cin, L)**
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **L** is the **length** of the input
    ///
    /// The **kernel** must be of shape **(Cout, Cin, Lk)**
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Lk** is the **length** of the kernel
    ///
    /// The resulting output shape will be **(N, Cout, Lout)**
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix3> + Forward, impl Gradient<Dim = Ix3> + Backward + Overwrite>
    where
        I: ConvolveWithGroups<I, VarDiff<Input<Ix3>, InputBackward<Ix3>>, Pad>,
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

/// Applies a **spatial convolution** over an input signal composed of several input planes.
pub struct Conv2d<Pad: PaddingMode> {
    padding: (usize, usize),
    padding_mode: Pad,
    stride: (usize, usize),
    dilation: (usize, usize),
    weight: VarDiff<Input<Ix4>, InputBackward<Ix4>>,
    bias: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
}

impl<Pad: PaddingMode> Conv2d<Pad> {
    /// Creates a new Conv2d.
    ///
    /// * `in_channels` - the number of planes in the input signal.
    ///
    /// * `out_channels` - the number of planes in the output signal.
    ///
    /// * `kernel_size` - the size of the kernel, a 2-tuple for this two-dimensional case.
    ///
    /// * `padding` - the padding to be applied to the input, a 2-tuple for this two-dimensional
    /// case.
    ///
    /// * `padding_mode` - the padding mode, it can be: **zeros**, **constant**, **reflective** or
    /// **replicative**.
    ///
    /// * `stride` - the stride of the convolution, a 2-tuple for this two-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 2-tuple for this
    /// two-dimensional case.
    ///
    /// The weight and the bias are initialised from **U(-k, k)** where
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
    /// * `input` - the signal to convolve.
    ///
    /// The **input** must be of shape **(N, Cin, H, W)**
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape **(Cout, Cin, Hk, Wk)**
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be **(N, Cout, Hout, Wout)**
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix4> + Forward, impl Gradient<Dim = Ix4> + Backward + Overwrite>
    where
        I: Convolve<I, VarDiff<Input<Ix4>, InputBackward<Ix4>>, Pad>,
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

/// Applies a **spatial grouped convolution** over an input signal composed of several input planes.
pub struct GroupedConv2d<Pad: PaddingMode> {
    padding: (usize, usize),
    padding_mode: Pad,
    stride: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    weight: VarDiff<Input<Ix4>, InputBackward<Ix4>>,
    bias: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
}

impl<Pad: PaddingMode> GroupedConv2d<Pad> {
    /// Creates a new GroupedConv2d.
    ///
    /// * `in_channels` - the number of planes in the input signal.
    ///
    /// * `out_channels` - the number of planes in the output signal.
    ///
    /// * `kernel_size` - the size of the kernel, a 2-tuple  for this two-dimensional case.
    ///
    /// * `padding` - the padding to be applied to the input, a 2-tuple  for this two-dimensional
    /// case.
    ///
    /// * `padding_mode` - the padding mode, it can be: **zeros**, **constant**, **reflective** or
    /// **replicative**.
    ///
    /// * `stride` - the stride of the convolution, a 2-tuple  for this two-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 2-tuple  for this
    /// two-dimensional case.
    ///
    /// * `groups` -  controls the connections between inputs and outputs. `in_channels` and
    /// `out_channels` must both be divisible by groups.
    ///
    /// For example:
    /// * at **groups = 1**, all inputs are convolved to all outputs.
    /// *  at **groups = 2**, the operation becomes equivalent to having two convolutional layers
    /// side by side, each seeing half the input channels and producing half the output channels,
    /// and both subsequently concatenated.
    /// * at groups = in_channels, each input channel is convolved with its own set of filters.
    ///
    /// The weight and the bias of the layer are initialised from **U(-k, k)** where
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
    /// * `input` - the signal to convolve.
    ///
    /// The **input** must be of shape **(N, Cin, H, W)**
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape **(Cout, Cin, Hk, Wk)**
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be **(N, Cout, Hout, Wout)**
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix4> + Forward, impl Gradient<Dim = Ix4> + Backward + Overwrite>
    where
        I: ConvolveWithGroups<I, VarDiff<Input<Ix4>, InputBackward<Ix4>>, Pad>,
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

/// Applies a **volumetric convolution** over an input signal composed of several input planes.
pub struct Conv3d<Pad: PaddingMode> {
    padding: (usize, usize, usize),
    padding_mode: Pad,
    stride: (usize, usize, usize),
    dilation: (usize, usize, usize),
    weight: VarDiff<Input<Ix5>, InputBackward<Ix5>>,
    bias: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
}

impl<Pad: PaddingMode> Conv3d<Pad> {
    /// Creates a new Conv3d.
    ///
    /// * `in_channels` - the number of planes in the input signal.
    ///
    /// * `out_channels` - the number of planes in the output signal.
    ///
    /// * `kernel_size` - the size of the kernel, a 3-tuple for this three-dimensional case.
    ///
    /// * `padding` - the padding to be applied to the input, a 3-tuple for this three-dimensional
    /// case.
    ///
    /// * `padding_mode` - the padding mode, it can be: **zeros**, **constant**, **reflective** or
    /// **replicative**.
    ///
    /// * `stride` - the stride of the convolution, a 3-tuple for this three-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 3-tuple for this
    /// three-dimensional case.
    ///
    /// The weight and the bias of the layer are initialised from **U(-k, k)** where
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
    /// * `input` - the signal to convolve.
    ///
    /// The **input** must be of shape **(N, Cin, D, H, W)**
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **D** is the **depth** of the input
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape **(Cout, Cin, Dk,  Hk, Wk)**
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Dk** is the **depth** of the kernel
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be **(N, Cout, Dout, Hout, Wout)**
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix5> + Forward, impl Gradient<Dim = Ix5> + Backward + Overwrite>
    where
        I: Convolve<I, VarDiff<Input<Ix5>, InputBackward<Ix5>>, Pad>,
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

/// Applies a **grouped volumetric convolution** over an input signal composed of several input
/// planes.
pub struct GroupedConv3d<Pad: PaddingMode> {
    padding: (usize, usize, usize),
    padding_mode: Pad,
    stride: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
    weight: VarDiff<Input<Ix5>, InputBackward<Ix5>>,
    bias: VarDiff<Input<Ix1>, InputBackward<Ix1>>,
}

impl<Pad: PaddingMode> GroupedConv3d<Pad> {
    /// Creates a new GroupedConv3d.
    ///
    /// * `in_channels` - the number of planes in the input signal.
    ///
    /// * `out_channels` - the number of planes in the output signal.
    ///
    /// * `kernel_size` - the size of the kernel, a 3-tuple  for this three-dimensional case.
    ///
    /// * `padding` - the padding to be applied to the input, a 3-tuple  for this three-dimensional
    /// case.
    ///
    /// * `padding_mode` - the padding mode, it can be: **zeros**, **constant**, **reflective** or
    /// **replicative**.
    ///
    /// * `stride` - the stride of the convolution, a 3-tuple  for this three-dimensional case.
    ///
    /// * `dilation` - controls the spacing between the kernel points, a 3-tuple  for this
    /// three-dimensional case.
    ///
    /// * `groups` -  controls the connections between inputs and outputs. `in_channels` and
    /// `out_channels` must both be divisible by groups.
    ///
    /// For example:
    /// * at **groups = 1**, all inputs are convolved to all outputs.
    /// *  at **groups = 2**, the operation becomes equivalent to having two convolutional layers
    /// side by side, each seeing half the input channels and producing half the output channels,
    /// and both subsequently concatenated.
    /// * at **groups = in_channels**, each input channel is convolved with its own set of filters.
    ///
    /// The weight and the bias are initialised from **U(-k, k)** where
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
    /// * `input` - the signal to convolve.
    ///
    /// The **input** must be of shape **(N, Cin, D, H, W)**
    /// * **N** is the batch size
    /// * **Cin** is the number of input channels
    /// * **D** is the **depth** of the input
    /// * **H** is the **height** of the input
    /// * **W** is the **width** of the input
    ///
    /// The **kernel** must be of shape **(Cout, Cin, Dk,  Hk, Wk)**
    /// * **Cout** is the number of output channels
    /// * **Cin** is the number of input channels
    /// * **Dk** is the **depth** of the kernel
    /// * **Hk** is the **height** of the kernel
    /// * **Wk** is the **width** of the kernel
    ///
    /// The resulting output shape will be **(N, Cout, Dout, Hout, Wout)**
    pub fn forward<I, T, U>(
        &self,
        input: I,
    ) -> VarDiff<impl Data<Dim = Ix5> + Forward, impl Gradient<Dim = Ix5> + Backward + Overwrite>
    where
        I: ConvolveWithGroups<I, VarDiff<Input<Ix5>, InputBackward<Ix5>>, Pad>,
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
