use ndarray::{
    linalg::{general_mat_mul, general_mat_vec_mul},
    Array, ArrayBase, ArrayD, ArrayView, Axis, DimMax, Dimension, IntoNdProducer, Ix1, Ix2, Zip,
};
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

pub(crate) use binary::*;
pub use binary::{
    Constant, Convolve, ConvolveWithGroups, PaddingMode, Reflective, Replicative, Zero,
};
pub use input::{Input, InputBackward};
pub(crate) use nary::*;
pub(crate) use unary::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Nodes' Modules ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mod binary;
mod input;
mod nary;
mod unary;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Type Aliases ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;
pub(crate) type BroadTensor<Lhs, Rhs> = Tensor<Broadcasted<Lhs, Rhs>>;
pub(crate) type DynTensor = ArrayD<f32>;
pub(crate) type Tensor<D> = Array<f32, D>;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Computational Nodes` Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Data representation.
///
/// This trait is implemented by all the internal forward components of `Var` and `VarDiff`.
///
/// It provides the `.data()` method that is used to retrieve a [`Ref`] to the data stored inside
/// the node.
pub trait Data {
    /// The data's dimensionality.
    type Dim: Dimension;

    /// Returns an immutable reference to the data inside `self`.
    fn data(&self) -> Ref<Tensor<Self::Dim>>;

    /// Returns a mutable reference to the data inside `self`.
    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>>;
}

/// Forward-propagation behavior.
///
/// This trait is implemented by all the internal forward components of `Var` and `VarDiff`.
///
/// The main method it provides is the `.forward()` method that is used to propagate computations
/// from the leaf variables to the graph's root.
///
/// The other two methods, namely `.was_computed()` and `.reset_computation()`, are used to perform
/// caching during the forward pass. Caching is critical to avoid recomputing paths and to achieve
/// good performance when a computational graph has more than one root, like the one, for instance,
/// of a recurrent neural network.
pub trait Forward {
    /// Propagates the computations forwards.
    ///
    /// It also defines the logic for the computation of the node.
    fn forward(&self);

    /// Returns `true` if the node was computed, `false` otherwise.
    fn was_computed(&self) -> bool;

    /// Reset the node's flag, making it computable again.
    fn reset_computation(&self);
}

/// Gradient representation.
///
/// This trait is implemented by all the internal backward components of `VarDiff`.
///
/// It provides the `.gradient()` method that is used to get a [`Ref`] to the data stored inside
/// the node.
pub trait Gradient {
    /// The gradient's dimensionality.
    type Dim: Dimension;

    /// Returns an immutable reference to the gradient inside `self`.
    fn gradient(&self) -> Ref<Tensor<Self::Dim>>;

    /// Returns a mutable reference to the gradient inside `self`.
    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>>;
}

/// Gradient accumulation's mode.
///
/// This trait is used to keep track of the gradient status. It specifies whether the gradient
/// must be overwritten or accumulated with `+=`.
pub trait Overwrite {
    /// Returns `true` is the gradient of the node `self` can be overwritten.
    fn can_overwrite(&self) -> bool;

    /// Set the status of `self` as an overwritable node.
    fn set_overwrite(&self, state: bool);
}

/// The union of the Gradient and Overwrite.
pub trait GradientOverwrite<D>: Gradient<Dim = D> + Overwrite {}

impl<T> Overwrite for Rc<T>
where
    T: Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.as_ref().can_overwrite()
    }

    fn set_overwrite(&self, state: bool) {
        self.as_ref().set_overwrite(state)
    }
}

impl<T, D> Gradient for Rc<T>
where
    T: Gradient<Dim = D>,
    D: Dimension,
{
    type Dim = D;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.as_ref().gradient()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.as_ref().gradient_mut()
    }
}

impl<D: Dimension, T> GradientOverwrite<D> for T where T: Gradient<Dim = D> + Overwrite {}

/// Back-propagation behavior.
///
/// This trait is implemented by all the internal backward components of `VarDiff`.
///
/// The main method it provides is the `.backward()` method that is used to back-propagate gradients
/// from the root variables to the graph's leaves.
///
/// The other two methods, namely `.no_grad()` and `.with_grad()` are used to shut down
/// gradients' computation.
pub trait Backward: Overwrite {
    /// Propagates the computations backwards.
    ///
    /// It also defines the logic for the back-propagation of the node.
    fn backward(&self);

    /// Shuts down the computation of the gradient for the node `self` and de-allocates its gradient.
    fn no_grad(&self);

    /// Switches back on the computation of the gradient for the node `self` and re-allocates its
    ///gradient.
    fn with_grad(&self);
}

/// Eval mode behavior.
///
/// This trait is implemented by all the variables and all the components that admit multiple
/// behaviors during training and evaluation.
///
/// It provides two methods, namely `.train()` and `.eval()`, that are used respectively to set
/// the entity in training mode and in evaluation mode.
pub trait Eval {
    /// Sets `self` in training mode.
    fn train(&self);

    /// Sets `self` in evaluation mode.
    fn eval(&self);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DotDim ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Utility trait useful to compute the dimensionality of algebraic operations' results.
trait DotDim<Rhs>
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
    type Output = Ix1;

    fn shape(_: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        let mut res_shape = Ix1::zeros(1);
        res_shape[0] = 1;
        res_shape
    }
}

impl DotDim<Ix2> for Ix1 {
    type Output = Ix1;

    fn shape(_: Self, rhs: Ix2) -> <Self as DotDim<Ix1>>::Output {
        let mut res_shape = Ix1::zeros(1);
        res_shape[0] = rhs.last_elem();
        res_shape
    }
}

impl DotDim<Ix1> for Ix2 {
    type Output = Ix1;

    fn shape(lhs: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        let mut res_shape = Ix1::zeros(1);
        res_shape[0] = lhs[0];
        res_shape
    }
}

impl DotDim<Ix2> for Ix2 {
    type Output = Ix2;

    fn shape(lhs: Self, rhs: Ix2) -> <Self as DotDim<Ix2>>::Output {
        let mut res_shape = Ix2::zeros(2);
        res_shape[0] = lhs[0];
        res_shape[1] = rhs[1];
        res_shape
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

/// Reduces `src` to the desired `dim`ension, reverting the broadcasting mechanic.
///
/// # Arguments
///
/// * `dim` - Desired dimension for the source tensor
/// * `src` - Tensor to reduce
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

    debug_assert_eq!(src.raw_dim(), dim.into_dyn());
    debug_assert!(src.is_standard_layout());

    src.into_dimensionality::<D>().unwrap()
}

/// Performs gradient accumulation of `gradient` into `destination_node`.
///
/// # Arguments
///
/// * `destination_node` - a node of the computational graph.
///
/// * `gradient` - incoming gradient.
pub fn push_gradient<'a, T, P, D>(destination_node: &T, gradient: P)
where
    T: Gradient + Overwrite + ?Sized,
    P: IntoNdProducer<Dim = D, Output = ArrayView<'a, f32, D>, Item = &'a f32>,
    D: Dimension,
{
    let mut destination_gradient = destination_node.gradient_mut();
    let zip = Zip::from(&mut *destination_gradient).and_broadcast(gradient);
    if destination_node.can_overwrite() {
        zip.for_each(|d, s| *d = *s);
        destination_node.set_overwrite(false);
    } else {
        zip.for_each(|d, s| *d += *s);
    }
}

/// Performs gradient accumulation into `destination_node`.
///
/// This functions accumulates the gradient of the matrix multiplication operation.
///
/// # Arguments
///
/// * `destination_node` - a node of the computational graph.
///
/// * `first` - two-dimensional array.
///
/// * `second` - two-dimensional array.
pub fn push_mat_mat_gradient<T, S1, S2>(
    destination_node: &T,
    first: &ArrayBase<S1, Ix2>,
    second: &ArrayBase<S2, Ix2>,
) where
    T: Gradient<Dim = Ix2> + Overwrite,
    S1: ndarray::Data<Elem = f32>,
    S2: ndarray::Data<Elem = f32>,
{
    if destination_node.can_overwrite() {
        general_mat_mul(1., first, second, 0., &mut destination_node.gradient_mut());
        destination_node.set_overwrite(false);
    } else {
        general_mat_mul(1., first, second, 1., &mut destination_node.gradient_mut());
    }
}

/// Performs gradient accumulation into `destination_node`.
///
/// This functions accumulates the gradient of the matrix-vector multiplication operation.
///
/// # Arguments
///
/// * `destination_node` - a node of the computational graph.
///
/// * `first` - two-dimensional array.
///
/// * `second` - one-dimensional array.
pub fn push_mat_vec_gradient<T, S1, S2>(
    destination_node: &T,
    first: &ArrayBase<S1, Ix2>,
    second: &ArrayBase<S2, Ix1>,
) where
    T: Gradient<Dim = Ix2> + Overwrite,
    S1: ndarray::Data<Elem = f32>,
    S2: ndarray::Data<Elem = f32>,
{
    let mut destination_gradient = destination_node.gradient_mut();
    let zip = Zip::from(&mut *destination_gradient)
        .and_broadcast(first)
        .and_broadcast(second);
    if destination_node.can_overwrite() {
        zip.for_each(|d, f, s| *d = f * s);
        destination_node.set_overwrite(false);
    } else {
        zip.for_each(|d, f, s| *d += f * s);
    }
}

/// Performs gradient accumulation into `destination_node`.
///
/// This functions accumulates the gradient of the vector-matrix multiplication operation.
///
/// # Arguments
///
/// * `destination_node` - a node of the computational graph.
///
/// * `first` - two-dimensional array.
///
/// * `second` - one-dimensional array.
pub fn push_vec_mat_gradient<T, S1, S2>(
    destination_node: &T,
    first: &ArrayBase<S1, Ix2>,
    second: &ArrayBase<S2, Ix1>,
) where
    T: Gradient<Dim = Ix1> + Overwrite,
    S1: ndarray::Data<Elem = f32>,
    S2: ndarray::Data<Elem = f32>,
{
    if destination_node.can_overwrite() {
        general_mat_vec_mul(1., first, second, 0., &mut destination_node.gradient_mut());
        destination_node.set_overwrite(false);
    } else {
        general_mat_vec_mul(1., first, second, 1., &mut destination_node.gradient_mut());
    }
}

/// Performs gradient accumulation into `destination_node`.
///
/// This functions accumulates the gradient of the vector-vector multiplication operation.
///
/// # Arguments
///
/// * `destination_node` - a node of the computational graph.
///
/// * `first` - two-dimensional array.
///
/// * `second` - one-dimensional array.
pub fn push_vec_vec_gradient<T, S>(destination_node: &T, first: &ArrayBase<S, Ix1>, second: &f32)
where
    T: Gradient<Dim = Ix1> + Overwrite,
    S: ndarray::Data<Elem = f32>,
{
    let mut destination_gradient = destination_node.gradient_mut();
    let zip = Zip::from(&mut *destination_gradient).and_broadcast(first);
    if destination_node.can_overwrite() {
        zip.for_each(|d, f| *d = f * second);
        destination_node.set_overwrite(false);
    } else {
        zip.for_each(|d, f| *d += f * second);
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
pub(crate) fn broadcasted_zeros<Lhs, Rhs>(
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
    let mut broadcasted_dim = <Lhs as DimMax<Rhs>>::Output::zeros(bigger.len());
    broadcasted_dim
        .slice_mut()
        .iter_mut()
        .zip(bigger.iter())
        .for_each(|(l, r)| *l = *r);
    broadcasted_dim
        .slice_mut()
        .iter_mut()
        .rev()
        .zip(smaller.iter().rev())
        .for_each(|(l, r)| *l = std::cmp::max(*l, *r));
    Tensor::zeros(broadcasted_dim)
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
pub(crate) fn expect_tensor<D: Dimension>(tensor: &RefCell<Option<Tensor<D>>>) -> Ref<Tensor<D>> {
    Ref::map(tensor.borrow(), |b| {
        b.as_ref().expect(
            "error: trying to get a de-allocated gradient. 
        Switch on the gradients first by using with_grad().",
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
pub(crate) fn expect_tensor_mut<D: Dimension>(
    tensor: &RefCell<Option<Tensor<D>>>,
) -> RefMut<Tensor<D>> {
    RefMut::map(tensor.borrow_mut(), |b| {
        b.as_mut().expect(
            "error: trying to get a de-allocated gradient. 
        Switch on the gradients first by using with_grad().",
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
/// Creates a new input node whose data will have shape `shape` and elements `elements`.
///
/// # Arguments
///
/// * `shape` - shape.
///
/// * `elements` - elements.
fn new_input<D, Sh>(shape: Sh, elements: Vec<f32>) -> Rc<Input<D>>
where
    D: Dimension + 'static,
    Sh: Into<ndarray::StrideShape<D>>,
{
    Input::new(new_tensor(shape, elements)).node
}

#[cfg(test)]
/// Creates a new backward input node whose gradient will have shape `shape` and elements
/// `elements`.
///
/// # Arguments
///
/// * `shape` - shape.
///
/// * `elements` - elements.
fn new_backward_input<D, Sh>(shape: Sh, elements: Vec<f32>) -> Rc<InputBackward<D>>
where
    D: Dimension + 'static,
    Sh: Into<ndarray::StrideShape<D>>,
{
    Rc::new(
        Input::new(new_tensor(shape, elements))
            .node
            .differentiable(),
    )
}

#[cfg(test)]
/// Creates a new tensor with shape `shape` and elements `elements`.
///
/// # Arguments
///
/// * `shape` - shape.
///
/// * `elements` - elements.
fn new_tensor<D, Sh>(shape: Sh, elements: Vec<f32>) -> Tensor<D>
where
    D: Dimension + 'static,
    Sh: Into<ndarray::StrideShape<D>>,
{
    Tensor::from_shape_vec(shape, elements).unwrap()
}
