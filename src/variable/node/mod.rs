mod absolute_error;
mod addition;
mod bce;
mod bce_with_logits;
mod chunk;
mod concatenate;
mod convolution;
mod division;
mod dropout;
mod exp;
mod kldiv;
mod leaky_relu;
mod logn;
mod logsoftmax;
mod matrix_matrix_mul;
mod matrix_matrix_mul_t;
mod matrix_vector_mul;
mod mean;
mod multi_concatenate;
mod multi_stack;
mod multiplication;
mod negation;
mod nll;
mod pad;
mod power;
mod relu;
mod sigmoid;
mod softmax;
mod softplus;
mod sqrt;
mod squared_error;
mod stack;
mod subtraction;
mod sum;
mod tanh;
mod transpose;
mod unsqueeze;
mod vector_matrix_mul;
mod vector_vector_mul;

pub(crate) use absolute_error::*;
pub(crate) use addition::*;
pub(crate) use bce::*;
pub(crate) use bce_with_logits::*;
pub(crate) use chunk::*;
pub(crate) use concatenate::*;
pub(crate) use convolution::*;
pub(crate) use division::*;
pub(crate) use dropout::*;
pub(crate) use exp::*;
pub(crate) use kldiv::*;
pub(crate) use leaky_relu::*;
pub(crate) use logn::*;
pub(crate) use logsoftmax::*;
pub(crate) use matrix_matrix_mul::*;
pub(crate) use matrix_matrix_mul_t::*;
pub(crate) use matrix_vector_mul::*;
pub(crate) use mean::*;
pub(crate) use multi_concatenate::*;
pub(crate) use multi_stack::*;
pub(crate) use multiplication::*;
pub(crate) use negation::*;
pub(crate) use nll::*;
pub(crate) use pad::*;
pub(crate) use power::*;
pub(crate) use relu::*;
pub(crate) use sigmoid::*;
pub(crate) use softmax::*;
pub(crate) use softplus::*;
pub(crate) use sqrt::*;
pub(crate) use squared_error::*;
pub(crate) use stack::*;
pub(crate) use subtraction::*;
pub(crate) use sum::*;
pub(crate) use tanh::*;
pub(crate) use transpose::*;
pub(crate) use unsqueeze::*;
pub(crate) use vector_matrix_mul::*;
pub(crate) use vector_vector_mul::*;

/// Forward-propagation behavior.
///
/// This trait is implemented by all the internal forward components of `Var` and `VarDiff`.
///
/// The main method it provides is the `.forward()` method that is used to propagate computations
/// from the leaf variables to the graph's root.
pub(crate) trait Forward {
    /// Propagates the computations forwards.
    ///
    /// It also defines the logic for the computation of the node.
    fn forward(&self);
}

/// Back-propagation behavior.
///
/// This trait is implemented by all the internal backward components of `VarDiff`.
///
/// The main method it provides is the `.backward()` method that is used to back-propagate gradients
/// from the root variables to the graph's leaves.
pub(crate) trait Backward {
    /// Propagates the computations backwards.
    ///
    /// It also defines the logic for the back-propagation of the node.
    fn backward(&self);
}

// #[cfg(test)]
// const F16_EPSILON: f32 = 9.77e-04;

// #[cfg(test)]
// /// Checks element-wise whether `array` is within `F16_EPSILON` of `target`.
// ///
// /// # Arguments
// ///
// /// * `array` - array to check.
// ///
// /// * `target` - target to check against.
// ///
// /// # Panics
// ///
// /// If `array` is not within `F16_EPSILON` of `target`.
// fn assert_almost_equals<D: Dimension>(array: &Tensor<D>, target: &Tensor<D>) {
//     assert!(
//         Zip::from(array).and(target).all(|l, r| {
//             (*l == 0. && *r == 0.)
//                 || (!l.is_finite() && !r.is_finite())
//                 || ((1. - r / l).abs() <= F16_EPSILON)
//         }),
//         "\nLeft:\n{}\nRight:\n{}",
//         array,
//         target
//     );
// }

// #[cfg(test)]
// /// Creates a new tensor with shape `shape` and elements `elements`.
// ///
// /// # Arguments
// ///
// /// * `shape` - shape.
// ///
// /// * `elements` - elements.
// fn new_tensor<D, Sh>(shape: Sh, elements: Vec<f32>) -> Rc<RefCell<Tensor<D>>>
// where
//     D: Dimension + 'static,
//     Sh: Into<ndarray::StrideShape<D>>,
// {
//     Rc::new(RefCell::new(
//         Tensor::from_shape_vec(shape, elements).unwrap(),
//     ))
// }

// fn new_opt_tensor<D, Sh>(shape: Sh, elements: Vec<f32>) -> Rc<RefCell<Option<Tensor<D>>>>
// where
//     D: Dimension + 'static,
//     Sh: Into<ndarray::StrideShape<D>>,
// {
//     Rc::new(RefCell::new(Some(
//         Tensor::from_shape_vec(shape, elements).unwrap(),
//     )))
// }
