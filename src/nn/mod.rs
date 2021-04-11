use super::Parameter;
use crate::graph::{node::Node, GraphBuilder, Tensor};
use ndarray::{Array, ArrayView, Dimension, Ix1, Ix2, IxDyn, ShapeBuilder, Slice};

/// Computes the shape of the output map.
///
/// `input_shape` - the shape of the input.
///
/// `kernel_shape` - the shape of the kernel.
///
/// `padding` - the padding around the input.
fn compute_out_shape<D: Dimension>(
    input_shape: &[usize],
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> D {
    let mut map_shape = D::zeros(input_shape.len());
    itertools::izip!(
        map_shape.slice_mut(),
        input_shape,
        kernel_shape,
        padding,
        stride,
        dilation
    )
    .for_each(|(map_s, in_s, k_s, pd, str, dil)| {
        *map_s = (in_s + 2 * pd - dil * (k_s - 1) - 1) / str + 1
    });
    map_shape
}

/// Returns a **rolling window view** of the input array.
///
/// `input` - input array.
///
/// `window_shape` - the shape of each of the windows.
///
/// `padding` - the padding around `input`.
///
/// `stride` - the stride.
///
/// `dilation` - the spacing between each element of the windows.
fn as_windows<'a, D: Dimension>(
    input: &Array<f32, D>,
    window_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayView<'a, f32, IxDyn> {
    let ndim = input.ndim();
    let input_shape = input.shape();

    let mut indexing_strides = vec![0; ndim];
    {
        let view =
            input.slice_each_axis(|ax| Slice::new(0, None, stride[ax.axis.index()] as isize));
        indexing_strides
            .iter_mut()
            .zip(view.strides())
            .for_each(|(is, vs)| *is = *vs);
    }

    let mut window_strides = vec![0; ndim];
    itertools::izip!(window_strides.iter_mut(), input.strides(), dilation)
        .for_each(|(ws, is, dil)| *ws = *is * (*dil as isize));

    let win_indices_shape =
        compute_out_shape::<D>(input_shape, window_shape, padding, stride, dilation);

    let mut new_shape = IxDyn::zeros(win_indices_shape.ndim() + window_shape.len());
    let mut strides = IxDyn::zeros(win_indices_shape.ndim() + window_shape.len());

    new_shape
        .slice_mut()
        .iter_mut()
        .zip(win_indices_shape.slice().iter().chain(window_shape.iter()))
        .for_each(|(ns, _s)| *ns = *_s as usize);

    strides
        .slice_mut()
        .iter_mut()
        .zip(indexing_strides.iter().chain(window_strides.iter()))
        .for_each(|(s, _s)| *s = *_s as usize);

    unsafe { ArrayView::from_shape_ptr(new_shape.strides(strides), input.as_ptr()) }
}

pub mod init {
    use super::super::{graph::GraphBuilder, graph::ParamDim, Parameter};
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
    fn calculate_fan_in_fan_out<D: ParamDim>(param: &GraphBuilder<Parameter<D>>) -> (f32, f32) {
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

    /// Fills the input `Parameter` with `value`.
    pub fn constant<D: ParamDim>(param: &mut GraphBuilder<Parameter<D>>, value: f32) {
        param.data_mut().map_inplace(|el| *el = value);
    }

    /// Fills the input `Parameter` with `0.0`.
    pub fn zeros<D: ParamDim>(param: &mut GraphBuilder<Parameter<D>>) {
        param.data_mut().map_inplace(|el| *el = 0.);
    }

    /// Fills the input `Parameter` with `1.0`.
    pub fn ones<D: ParamDim>(param: &mut GraphBuilder<Parameter<D>>) {
        param.data_mut().map_inplace(|el| *el = 1.0);
    }

    /// Fills the 2-dimensional input `Parameter` with the identity matrix.
    /// Preserves the identity of the inputs in Linear layers, where as
    /// many inputs are preserved as possible.
    pub fn eye(param: &mut GraphBuilder<Parameter<Ix2>>) {
        for ((x, y), el) in param.data_mut().indexed_iter_mut() {
            if x == y {
                *el = 1.
            } else {
                *el = 0.
            }
        }
    }

    /// Fills the input `Parameter` with elements drawn from
    /// the uniform distribution U(low, high).
    pub fn uniform<D: ParamDim>(param: &mut GraphBuilder<Parameter<D>>, low: f32, high: f32) {
        let unif_dstr = Uniform::new(low, high);
        let mut t_rng = thread_rng();
        param
            .data_mut()
            .map_inplace(|el| *el = unif_dstr.sample(&mut t_rng));
    }

    /// Fills the input `Parameter` with elements drawn from
    /// the normal distribution N(mean, std^2).
    pub fn normal<D: ParamDim>(param: &mut GraphBuilder<Parameter<D>>, mean: f32, std: f32) {
        let norm_dstr = Normal::new(mean, std).unwrap();
        let mut t_rng = thread_rng();
        param
            .data_mut()
            .map_inplace(|el| *el = norm_dstr.sample(&mut t_rng));
    }

    /// Fills the input `Parameter` with values according to the method
    /// described in Understanding the difficulty of training deep feedforward
    /// neural networks - Glorot, X. & Bengio, Y. (2010), using a uniform
    /// distribution.
    pub fn xavier_uniform<D: ParamDim>(param: &mut GraphBuilder<Parameter<D>>, gain: f32) {
        let (fan_in, fan_out) = calculate_fan_in_fan_out(param);
        let std = gain * (2. / ((fan_in + fan_out) as f32)).sqrt();
        let a = 3.0_f32.sqrt() * std;
        let unif_distr = Uniform::new(-a, a);
        let mut t_rng = thread_rng();
        param
            .data_mut()
            .map_inplace(|el| *el = unif_distr.sample(&mut t_rng));
    }

    /// Fills the input `Parameter` with values according to the method
    /// described in Understanding the difficulty of training deep feedforward
    /// neural networks - Glorot, X. & Bengio, Y. (2010), using a normal
    /// distribution.
    ///
    /// Also known as Glorot initialization.
    pub fn xavier_normal<D: ParamDim>(param: &mut GraphBuilder<Parameter<D>>, gain: f32) {
        let (fan_in, fan_out) = calculate_fan_in_fan_out(param);
        let std = gain * (2. / ((fan_in + fan_out) as f32)).sqrt();
        let norm_distr = Normal::new(0., std).unwrap();
        let mut t_rng = thread_rng();
        param
            .data_mut()
            .map_inplace(|el| *el = norm_distr.sample(&mut t_rng));
    }
}
/// Applies a linear transformation to the incoming data.
///
/// **y = xA^T + b**
pub struct Linear {
    pub weight: GraphBuilder<Parameter<Ix2>>,
    pub bias: GraphBuilder<Parameter<Ix1>>,
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
        let mut weight = Parameter::new(Tensor::zeros((out_features, in_features)));
        let mut bias = Parameter::new(Tensor::zeros(out_features));
        let k = (1. / (in_features as f32)).sqrt();
        init::uniform(&mut weight, -k, k);
        init::uniform(&mut bias, -k, k);

        Self { weight, bias }
    }

    /// Applies the linear transformation **y = xA^T + b** to the incoming data.
    ///
    /// `data` - `(N, in_features)`, the output will be `(N, out_features)`.
    pub fn forward(
        &self,
        input: &GraphBuilder<impl Node<Dim = Ix2>>,
    ) -> GraphBuilder<impl Node<Dim = Ix2>> {
        input.mm(&self.weight.t()) + self.bias.clone()
    }
}
