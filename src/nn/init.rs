// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ init module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//! Layers' parameters initialization functions.
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
use super::Learnable;
use ndarray::{Axis, Dimension, Ix2};
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};

/// Returns the recommended gain value for the given non-linearity function.
///
/// Supported non-linearities are:
/// * linear
/// * sigmoid
/// * tanh
/// * relu
/// * leaky_relu
///
/// # Arguments
///
/// `non_linearity` - a non-linearity function's name.
///
/// # Panics
///
/// If `non_linearity` is not among those listed above.
pub fn calculate_gain(non_linearity: &str) -> f32 {
    match non_linearity {
        "linear" | "sigmoid" => 1.0,
        "tanh" => 5.0 / 3.0,
        "relu" => 2.0_f32.sqrt(),
        "leaky_relu" => (2.0 / (1.0 + 0.01_f32.powi(2))).sqrt(),
        _ => panic!("error: unsupported nonlinearity: {}", non_linearity),
    }
}

/// Returns the *fan_in* and the *fan_out*.
///
/// For *MLPs* *fan_in* and *fan_out* are respectively the number of inputs and outputs to an
/// hidden unit of the layer. For *CNNs* however, the number of input feature maps and the size
/// of the receptive field must be taken into account .
///
/// # Arguments
///
/// `param` - differentiable variable for which the *fan in* and the *fan out* must be
/// calculated.
pub fn calculate_fan_in_fan_out<D: Dimension>(param: &Learnable<D>) -> (f32, f32) {
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

/// Fills the differentiable leaf variable with a constant value.
///
/// # Arguments
///
/// * `param` - differentiable variable to initialize.
///
/// * `value` - value to fill the variable with.
pub fn constant<D: Dimension>(param: &Learnable<D>, value: f32) {
    param.data_mut().map_inplace(|el| *el = value);
}

/// Fills the differentiable leaf variable with zeros.
///
/// # Arguments
///
/// `param` - differentiable variable to initialize.
pub fn zeros<D: Dimension>(param: &Learnable<D>) {
    param.data_mut().map_inplace(|el| *el = 0.);
}

/// Fills the differentiable leaf variable with ones.
///
/// # Arguments
///
/// `param` - differentiable variable to initialize.
pub fn ones<D: Dimension>(param: &Learnable<D>) {
    param.data_mut().map_inplace(|el| *el = 1.0);
}

/// Fills the matrix differentiable leaf variable with the identity matrix.
///
/// Preserves the identity of the inputs in Linear layers, where as
/// many inputs are preserved as possible.
///
/// # Arguments
///
/// `param` - differentiable variable to initialize.
pub fn eye(param: &Learnable<Ix2>) {
    for ((x, y), el) in param.data_mut().indexed_iter_mut() {
        if x == y {
            *el = 1.
        } else {
            *el = 0.
        }
    }
}

/// Fills the {3, 4, 5}-dimensional differentiable leaf variable with the Dirac delta function.
///
/// Preserves the identity of the inputs in convolutional layers, where as many input channels
/// are preserved as possible. In case of `groups > 1`, each group of channels preserves
/// identity.
///
/// # Arguments
///
/// * `param` - differentiable variable to initialize.
///
/// * `groups` - number of groups.
///
/// # Panics
///
/// If the differentiable variable is not {3, 4, 5}-dimensional and the number of output
/// channels is not divisible by `groups`. The number of output channels is equal to the length
/// of the first axis of `param`'s data.
pub fn dirac<D: Dimension>(param: &Learnable<D>, groups: usize) {
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

/// Fills the differentiable leaf variable with elements drawn from the uniform distribution
/// *U(low, high)*.
///
/// # Arguments
///
/// * `param` - differentiable variable to initialize.
///
/// * `low` - lower bound of the uniform distribution.
///
/// * `high` - upper bound of the uniform distribution.
///
/// # Panics
///
/// If `low` >= `high`.
pub fn uniform<D: Dimension>(param: &Learnable<D>, low: f32, high: f32) {
    let unif_dstr = Uniform::new(low, high);
    let mut t_rng = thread_rng();
    param
        .data_mut()
        .map_inplace(|el| *el = unif_dstr.sample(&mut t_rng));
}

/// Fills the differentiable leaf variable with elements drawn from the normal distribution
/// *N(mean, std^2)*.
///
/// # Arguments
///
/// * `param` - differentiable variable to initialize.
///
/// * `mean` - mean of the normal distribution.
///
/// * `std` - standard deviation of the normal distribution.
pub fn normal<D: Dimension>(param: &Learnable<D>, mean: f32, std: f32) {
    let norm_dstr = Normal::new(mean, std).unwrap();
    let mut t_rng = thread_rng();
    param
        .data_mut()
        .map_inplace(|el| *el = norm_dstr.sample(&mut t_rng));
}

/// Fills the differentiable leaf variable with values according to the method described in
/// [Understanding the difficulty of training deep feedforward
/// neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) - Glorot, X. &
/// Bengio, Y. (2010), using a uniform distribution.
///
/// # Arguments
///
/// * `param` - differentiable variable to initialize.
///
/// * `gain` - optional scaling factor. See also [`calculate_gain`](function@calculate_gain).
pub fn xavier_uniform<D: Dimension>(param: &Learnable<D>, gain: f32) {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(param);
    let std = gain * (2. / ((fan_in + fan_out) as f32)).sqrt();
    let a = 3.0_f32.sqrt() * std;
    let unif_distr = Uniform::new(-a, a);
    let mut t_rng = thread_rng();
    param
        .data_mut()
        .map_inplace(|el| *el = unif_distr.sample(&mut t_rng));
}

/// Fills the differentiable leaf variable with values according to the method described in
/// [Understanding the difficulty of training deep feedforward
/// neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) - Glorot, X. &
/// Bengio, Y. (2010), using a normal distribution.
///
/// Also known as **Glorot initialization**.
///
/// # Arguments
///
/// * `param` - differentiable variable to initialize.
///
/// * `gain` - optional scaling factor. See also [`calculate_gain`](function@calculate_gain).
pub fn xavier_normal<D: Dimension>(param: &Learnable<D>, gain: f32) {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(param);
    let std = gain * (2. / ((fan_in + fan_out) as f32)).sqrt();
    let norm_distr = Normal::new(0., std).unwrap();
    let mut t_rng = thread_rng();
    param
        .data_mut()
        .map_inplace(|el| *el = norm_distr.sample(&mut t_rng));
}
