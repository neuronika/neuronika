use super::Tensor;
use ndarray::{Dimension, Ix1, Ix2};
use std::cell::{Ref, RefMut};

pub mod backward;
pub mod forward;

pub use backward::*;
pub use forward::*;

/// Data representation.
///
/// This trait is implemented by all the internal forward components of `Var` and `VarDiff`.
///
/// It provides the `.data()` method that is used to retrive a [`Ref`] to the data stored inside
/// the node.
pub trait Data {
    /// The data's dimensionality.
    type Dim: Dimension;

    /// Returns an immutable reference to the data inside `self`.
    fn data(&self) -> Ref<Tensor<Self::Dim>>;

    /// Returns a mutable reference to the data inside `self`.
    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>>;
}

/// Forward-propagation behaviour.
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
/// It provides the `.gradient()` method that is used to retrive a [`Ref`] to the data stored inside
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

/// Back-propagation behaviour.
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

    /// Shuts down the computation of the gradient for the node `self` and deallocates its gradient.
    fn no_grad(&self);

    /// Switches back on the computation of the gradient for the node `self` and re-allocates its
    ///gradient.
    fn with_grad(&self);
}

/// Specifies the nodes that can be made differentiable.
pub(crate) trait Differentiable {
    type Output: Gradient + Overwrite;

    /// Returns the differentiable counterpart of `self`.
    fn differentiable(&self) -> Self::Output;
}

/// Eval mode behaviour.
///
/// This trait is implemented by all the variables and all the components that admit multiple
/// behaviours during training and evaluation.
///
/// It provides two methods, namely `.train()` and `.eval()`, that are used respectively to set
/// the entity in training mode and in evaluation mode.
pub trait Eval {
    /// Sets `self` in training mode.
    fn train(&self);

    /// Sets `self` in evaluation mode.
    fn eval(&self);
}

trait DotDim<Rhs>
where
    Self: Dimension,
    Rhs: Dimension,
{
    type Output: Dimension;

    fn shape(lhs: Self, rhs: Rhs) -> <Self as DotDim<Rhs>>::Output;
}

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
