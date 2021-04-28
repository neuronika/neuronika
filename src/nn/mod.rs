use super::{Input, InputBackward};
use crate::graph::{
    self,
    node::{Backward, Data, Forward, Gradient, Transpose, TransposeBackward},
    MatMatMul, Tensor, Var, VarDiff,
};
use ndarray::{Ix1, Ix2};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ init module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub mod init {
    use super::super::{graph::ParamDim, graph::VarDiff, Input, InputBackward};
    use ndarray::{Axis, Ix2};
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
    fn calculate_fan_in_fan_out<D: ParamDim>(
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
    pub fn constant<D: ParamDim>(param: &mut VarDiff<Input<D>, InputBackward<D>>, value: f32) {
        param.data_mut().map_inplace(|el| *el = value);
    }

    /// Fills the input with `0.0`.
    pub fn zeros<D: ParamDim>(param: &mut VarDiff<Input<D>, InputBackward<D>>) {
        param.data_mut().map_inplace(|el| *el = 0.);
    }

    /// Fills the input with `1.0`.
    pub fn ones<D: ParamDim>(param: &mut VarDiff<Input<D>, InputBackward<D>>) {
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

    /// Fills the input with elements drawn from
    /// the uniform distribution U(low, high).
    pub fn uniform<D: ParamDim>(
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
    pub fn normal<D: ParamDim>(
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
    pub fn xavier_uniform<D: ParamDim>(param: &mut VarDiff<Input<D>, InputBackward<D>>, gain: f32) {
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
    pub fn xavier_normal<D: ParamDim>(param: &mut VarDiff<Input<D>, InputBackward<D>>, gain: f32) {
        let (fan_in, fan_out) = calculate_fan_in_fan_out(param);
        let std = gain * (2. / ((fan_in + fan_out) as f32)).sqrt();
        let norm_distr = Normal::new(0., std).unwrap();
        let mut t_rng = thread_rng();
        param
            .data_mut()
            .map_inplace(|el| *el = norm_distr.sample(&mut t_rng));
    }
}

pub mod loss;
pub mod utils;

/// Applies a linear transformation to the incoming data.
///
/// **y = xA^T + b**
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
    /// are initialised from **U(-k, k)** where `k = 1. /(in_features as f32).sqrt()`.
    ///
    /// The learnable bias of the layer is of shape `out_features`. The values
    /// are initialised from **U(-k, k)** where `k = 1. /(in_features as f32).sqrt()`.
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
    ) -> VarDiff<impl Data + Forward, impl Backward + Gradient>
    where
        W: MatMatMul<VarDiff<Transpose<Input<Ix2>>, TransposeBackward<InputBackward<Ix2>>>>,
        W::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix2> + Forward,
        U: Gradient<Dim = Ix2> + Backward,
    {
        input.mm_mul(self.weight.clone().t()).into() + self.bias.clone()
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
    /// `k = 1. /(`hidden_size` as f32).sqrt()`.
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
        VarDiff<impl Data<Dim = Ix2> + Forward, impl Gradient<Dim = Ix2> + Backward>,
        VarDiff<impl Data<Dim = Ix2> + Forward, impl Gradient<Dim = Ix2> + Backward>,
    )
    where
        Cf: Data<Dim = Ix2> + Forward,
        Cb: Gradient<Dim = Ix2> + Backward,
        Hf: Data<Dim = Ix2> + Forward,
        Hb: Gradient<Dim = Ix2> + Backward,
        I: MatMatMul<VarDiff<Transpose<Input<Ix2>>, TransposeBackward<InputBackward<Ix2>>>>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix2> + Forward,
        U: Gradient<Dim = Ix2> + Backward,
    {
        let (cell_state, hidden) = state;
        let gates = hidden.mm_mul(self.weight_hh.clone().t())
            + self.bias_hh.clone()
            + input.mm_mul(self.weight_ih.clone().t()).into()
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
    /// `k = 1. /(`hidden_size` as f32).sqrt()`.
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
    ) -> VarDiff<impl Data<Dim = Ix2> + Forward, impl Gradient<Dim = Ix2> + Backward>
    where
        Hf: Data<Dim = Ix2> + Forward,
        Hb: Gradient<Dim = Ix2> + Backward,
        I: MatMatMul<VarDiff<Transpose<Input<Ix2>>, TransposeBackward<InputBackward<Ix2>>>>,
        I::Output: Into<VarDiff<T, U>>,
        T: Data<Dim = Ix2> + Forward,
        U: Gradient<Dim = Ix2> + Backward,
    {
        let (igates, hgates) = {
            (
                input.mm_mul(self.weight_ih.clone().t()).into() + self.bias_ih.clone(),
                hidden.clone().mm_mul(self.weight_hh.clone().t()) + self.bias_hh.clone(),
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
