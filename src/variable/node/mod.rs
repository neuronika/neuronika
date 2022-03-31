// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Nodes' Modules ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mod addition;
mod bce_loss;
mod bce_with_logits_loss;
mod chunk;
mod concatenate;
mod division;
mod dropout;
mod exp;
mod kldiv_loss;
mod leaky_relu;
mod logn;
mod logsoftmax;
mod mae_loss;
mod matrix_matrix_mul;
mod matrix_matrix_mul_t;
mod matrix_vector_mul;
mod mean;
mod mse_loss;
mod multi_concatenate;
mod multi_stack;
mod multiplication;
mod negation;
mod nll_loss;
mod pad;
mod power;
mod relu;
mod sigmoid;
mod softmax;
mod softplus;
mod sqrt;
mod stack;
mod subtraction;
mod sum;
mod tanh;
mod transpose;
mod unsqueeze;
mod vector_matrix_mul;
mod vector_vector_mul;

use ndarray::{Array, ArrayD, Axis, DimMax, Dimension, IntoDimension, Ix0, Ix1, Ix2, Zip};
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

pub(crate) use addition::*;
pub(crate) use bce_loss::*;
pub(crate) use bce_with_logits_loss::*;
pub(crate) use chunk::*;
pub(crate) use concatenate::*;
pub(crate) use division::*;
pub(crate) use dropout::*;
pub(crate) use exp::*;
pub(crate) use kldiv_loss::*;
pub(crate) use kldiv_loss::*;
pub(crate) use leaky_relu::*;
pub(crate) use leaky_relu::*;
pub(crate) use logn::*;
pub(crate) use logn::*;
pub(crate) use logsoftmax::*;
pub(crate) use logsoftmax::*;
pub(crate) use mae_loss::*;
pub(crate) use mae_loss::*;
pub(crate) use matrix_matrix_mul::*;
pub(crate) use matrix_matrix_mul_t::*;
pub(crate) use matrix_vector_mul::*;
pub(crate) use mean::*;
pub(crate) use mse_loss::*;
pub(crate) use multi_concatenate::*;
pub(crate) use multi_stack::*;
pub(crate) use multiplication::*;
pub(crate) use negation::*;
pub(crate) use nll_loss::*;
pub(crate) use pad::*;
pub(crate) use power::*;
pub(crate) use relu::*;
pub(crate) use sigmoid::*;
pub(crate) use softmax::*;
pub(crate) use softplus::*;
pub(crate) use sqrt::*;
pub(crate) use stack::*;
pub(crate) use subtraction::*;
pub(crate) use sum::*;
pub(crate) use tanh::*;
pub(crate) use transpose::*;
pub(crate) use unsqueeze::*;
pub(crate) use vector_matrix_mul::*;
pub(crate) use vector_vector_mul::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Type Aliases ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;
pub(crate) type BroadTensor<Lhs, Rhs> = Tensor<Broadcasted<Lhs, Rhs>>;
pub(crate) type DynTensor = ArrayD<f32>;
pub(crate) type Tensor<D> = Array<f32, D>;
pub(crate) type SharedTensor<D> = Rc<RefCell<Tensor<D>>>;

/// Specifies the reduction to apply to the *loss* output.
#[derive(Copy, Clone, Debug)]
pub enum Reduction {
    /// The output will be summed.
    Sum,
    /// The sum of the output will be divided by the batch size for the [`kldiv_loss`] and the
    /// [`nll_loss`]. For all other losses the output will be divided by the number of elements.
    Mean,
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Computational Nodes` Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Forward-propagation behavior.
///
/// This trait is implemented by all the internal forward components of `Var` and `VarDiff`.
///
/// The main method it provides is the `.forward()` method that is used to propagate computations
/// from the leaf variables to the graph's root.
///
/// The other two methods this trait provides, namely `.was_computed()` and `.reset_computation()`,
/// are used to perform caching during the forward pass.
///
/// Caching is critical to avoid recomputing paths and to achieve good performance when a
/// computational graph has more than one root, like the one, for instance, of a recurrent neural
/// network.
pub trait Forward {
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
///
/// The other two methods, namely `.no_grad()` and `.with_grad()` are used to shut down
/// gradients' computation.
pub trait Backward {
    /// Propagates the computations backwards.
    ///
    /// It also defines the logic for the back-propagation of the node.
    fn backward(&self);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DotDim ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Utility trait useful to compute the dimensionality of algebraic operations' results.
pub(crate) trait DotDim<Rhs>
where
    Self: Dimension,
    Rhs: Dimension,
{
    /// Dimension of the resulting variable.
    type Output: Dimension;

    /// Does the actual computation of the shape.
    fn shape(lhs: Self, rhs: Rhs) -> <Self as DotDim<Rhs>>::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DotDim implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl DotDim<Ix1> for Ix1 {
    type Output = Ix0;

    fn shape(_: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        ().into_dimension()
    }
}

impl DotDim<Ix2> for Ix1 {
    type Output = Ix1;

    fn shape(_: Self, rhs: Ix2) -> <Self as DotDim<Ix2>>::Output {
        let mut result = Ix1::zeros(1);
        result[0] = rhs.last_elem();
        result
    }
}

impl DotDim<Ix1> for Ix2 {
    type Output = Ix1;

    fn shape(lhs: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        let mut result = Ix1::zeros(1);
        result[0] = lhs[0];
        result
    }
}

impl DotDim<Ix2> for Ix2 {
    type Output = Ix2;

    fn shape(lhs: Self, rhs: Ix2) -> <Self as DotDim<Ix2>>::Output {
        let mut result = Ix2::zeros(2);
        result[0] = lhs[0];
        result[1] = rhs[1];
        result
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gradient Accumulation Utilities  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Shrinks `array` summing in-place along `axis`.
///
/// # Arguments
///
/// * `array` - array to reduce.
///
/// * `axis` - axis to sum along to.
fn sum_axis_inplace(array: &mut DynTensor, axis: Axis) {
    let (first, rest) = array.view_mut().split_at(axis, 1);
    Zip::from(first.remove_axis(axis))
        .and(rest.lanes(axis))
        .for_each(|dst, src| *dst += src.sum());
    array.index_axis_inplace(axis, 0);
}

/// Reduces `src` to the desired `dim` dimension, reverting the broadcasting.
///
/// # Arguments
///
/// * `dim` - desired dimension for the source tensor.
///
/// * `src` - tensor to reduce.
pub fn reduce<D: Dimension, E: Dimension>(dim: D, src: &Tensor<E>) -> Tensor<D> {
    let mut src = src.clone().into_dyn();

    while src.ndim() > dim.ndim() {
        sum_axis_inplace(&mut src, Axis(0));
    }

    for (axis, size) in dim.slice().iter().enumerate() {
        if *size == 1 {
            sum_axis_inplace(&mut src, Axis(axis));
            src.insert_axis_inplace(Axis(axis));
        }
    }

    debug_assert_eq!(
        src.raw_dim(),
        dim.into_dyn(),
        "Dimension mismatch in gradient reduction."
    );

    if src.is_standard_layout() {
        src.into_dimensionality::<D>().unwrap()
    } else {
        src.clone().into_dimensionality::<D>().unwrap()
    }
}

pub trait Switch {
    fn deallocate(&self);

    fn allocate(&self);
}

pub struct SwitchableTensor<D>
where
    D: Dimension,
{
    tensor: RefCell<Option<Tensor<D>>>,
    shape: D,
}

impl<D> SwitchableTensor<D>
where
    D: Dimension,
{
    pub fn zeros(shape: D) -> Self {
        Self {
            tensor: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
        }
    }

    pub fn from_ndarray(tensor: Tensor<D>) -> Self {
        let shape = tensor.raw_dim();

        Self {
            shape,
            tensor: RefCell::new(Some(tensor)),
        }
    }

    pub fn array(&self) -> Ref<Tensor<D>> {
        debug_assert!(self.tensor.borrow().is_some(), "Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`");

        Ref::map(self.tensor.borrow(), |b| b.as_ref().unwrap())
    }

    pub fn array_mut(&self) -> RefMut<Tensor<D>> {
        debug_assert!(self.tensor.borrow().is_some(), "Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`");

        RefMut::map(self.tensor.borrow_mut(), |b| b.as_mut().unwrap())
    }

    pub fn shape(&self) -> D {
        self.shape.clone()
    }
}

impl<D> Switch for SwitchableTensor<D>
where
    D: Dimension,
{
    fn deallocate(&self) {
        *self.tensor.borrow_mut() = None;
    }

    fn allocate(&self) {
        let mut tensor = self.tensor.borrow_mut();
        if tensor.is_none() {
            *tensor = Some(Tensor::zeros(self.shape()));
        }
    }
}

pub struct SwitchableBufferedTensor<D>
where
    D: Dimension,
{
    switchable: Rc<SwitchableTensor<D>>,
    buffer: RefCell<Option<Tensor<D>>>,
}

impl<D> SwitchableBufferedTensor<D>
where
    D: Dimension,
{
    pub fn from_switchable(switchable: Rc<SwitchableTensor<D>>) -> Self {
        Self {
            buffer: RefCell::new(Some(Array::zeros(switchable.shape()))),
            switchable,
        }
    }

    pub fn array(&self) -> Ref<Tensor<D>> {
        self.switchable.array()
    }

    pub fn array_mut(&self) -> RefMut<Tensor<D>> {
        self.switchable.array_mut()
    }

    pub fn buffer(&self) -> Ref<Tensor<D>> {
        debug_assert!(self.buffer.borrow().is_some(), "Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`");

        Ref::map(self.buffer.borrow(), |b| b.as_ref().unwrap())
    }

    pub fn buffer_mut(&self) -> RefMut<Tensor<D>> {
        debug_assert!(self.buffer.borrow().is_some(), "Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`");

        RefMut::map(self.buffer.borrow_mut(), |b| b.as_mut().unwrap())
    }

    pub fn shape(&self) -> D {
        self.switchable.shape()
    }
}

impl<D> Switch for SwitchableBufferedTensor<D>
where
    D: Dimension,
{
    fn deallocate(&self) {
        self.switchable.deallocate();

        *self.buffer.borrow_mut() = None;
    }

    fn allocate(&self) {
        self.switchable.allocate();

        let mut buffer = self.buffer.borrow_mut();
        if buffer.is_none() {
            *buffer = Some(Tensor::zeros(self.shape()));
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tensor Utilities ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates an empty tensor whose shape is the result of broadcasting between those of `left` and
/// `right`.
///
/// # Arguments
///
/// * `left` - left operand in the binary operations that admits broadcasting.
///
/// * `right` - right operand in the binary operations that admits broadcasting.
pub(crate) fn cobroadcasted_zeros<Lhs, Rhs>(
    left: &Tensor<Lhs>,
    right: &Tensor<Rhs>,
) -> BroadTensor<Lhs, Rhs>
where
    Lhs: Dimension + DimMax<Rhs>,
    Rhs: Dimension,
{
    let (bigger, smaller) = if left.ndim() >= right.ndim() {
        (left.shape(), right.shape())
    } else {
        (right.shape(), left.shape())
    };
    let mut out = <Lhs as DimMax<Rhs>>::Output::zeros(bigger.len());
    out.slice_mut()
        .iter_mut()
        .zip(bigger.iter())
        .for_each(|(l, r)| *l = *r);
    let k = bigger.len() - smaller.len();
    out.slice_mut()
        .iter_mut()
        .skip(k)
        .zip(smaller.iter())
        .for_each(|(l, r)| {
            if *l != *r {
                if *l == 1 {
                    *l = *r
                } else if *r != 1 {
                    panic!("The two tensors have incompatible shape.")
                }
            }
        });
    Tensor::zeros(out)
}

/// Returns a `Ref` to `tensor`. This function is used to access gradients.
///
/// # Arguments
///
/// `tensor` - gradient.
///
/// # Panics
///
/// If the gradient has been de-allocated.
pub(crate) fn expect_tensor<D>(tensor: &Rc<RefCell<Option<Tensor<D>>>>) -> Ref<Tensor<D>>
where
    D: Dimension,
{
    Ref::map(tensor.borrow(), |b| {
        b.as_ref().expect(
            "Trying to get a de-allocated gradient. Switch on the gradients first by using with_grad().",
        )
    })
}

/// Returns a `RefMut` to `tensor`. This function is used to access gradients.
///
/// # Arguments
///
/// `tensor` - gradient.
///
/// # Panics
///
/// If the gradient has been de-allocated.
pub(crate) fn expect_tensor_mut<D>(tensor: &Rc<RefCell<Option<Tensor<D>>>>) -> RefMut<Tensor<D>>
where
    D: Dimension,
{
    RefMut::map(tensor.borrow_mut(), |b| {
        b.as_mut().expect(
            "Trying to get a de-allocated gradient. Switch on the gradients first by using with_grad().",
        )
    })
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Testing Utilities ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
const F16_EPSILON: f32 = 9.77e-04;

#[cfg(test)]
/// Checks element-wise whether `array` is within `F16_EPSILON` of `target`.
///
/// # Arguments
///
/// * `array` - array to check.
///
/// * `target` - target to check against.
///
/// # Panics
///
/// If `array` is not within `F16_EPSILON` of `target`.
fn assert_almost_equals<D: Dimension>(array: &Tensor<D>, target: &Tensor<D>) {
    assert!(
        Zip::from(array).and(target).all(|l, r| {
            (*l == 0. && *r == 0.)
                || (!l.is_finite() && !r.is_finite())
                || ((1. - r / l).abs() <= F16_EPSILON)
        }),
        "\nLeft:\n{}\nRight:\n{}",
        array,
        target
    );
}

#[cfg(test)]
/// Creates a new tensor with shape `shape` and elements `elements`.
///
/// # Arguments
///
/// * `shape` - shape.
///
/// * `elements` - elements.
fn new_tensor<D, Sh>(shape: Sh, elements: Vec<f32>) -> Rc<RefCell<Tensor<D>>>
where
    D: Dimension + 'static,
    Sh: Into<ndarray::StrideShape<D>>,
{
    Rc::new(RefCell::new(
        Tensor::from_shape_vec(shape, elements).unwrap(),
    ))
}

fn new_opt_tensor<D, Sh>(shape: Sh, elements: Vec<f32>) -> Rc<RefCell<Option<Tensor<D>>>>
where
    D: Dimension + 'static,
    Sh: Into<ndarray::StrideShape<D>>,
{
    Rc::new(RefCell::new(Some(
        Tensor::from_shape_vec(shape, elements).unwrap(),
    )))
}
