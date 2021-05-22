use super::Tensor;
use ndarray::{Dimension, Ix1, Ix2};
use std::cell::{Ref, RefMut};

pub mod backward;
pub mod forward;

pub use backward::*;
pub use forward::*;

/// Data representation.
pub trait Data {
    type Dim: Dimension;

    fn data(&self) -> Ref<Tensor<Self::Dim>>;
}

/// Forward-propagation behaviour.
pub trait Forward {
    fn forward(&self);

    fn was_computed(&self) -> bool;

    fn reset_computation(&self);
}

/// Gradient representation.
pub trait Gradient {
    type Dim: Dimension;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>>;

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>>;
}

/// Gradient accumulation's modes.
pub trait Overwrite {
    fn can_overwrite(&self) -> bool;

    fn set_overwrite(&self, state: bool);
}

/// Back-propagation behaviour.
pub trait Backward: Overwrite {
    fn backward(&self);

    fn no_grad(&self);

    fn with_grad(&self);
}

pub(crate) trait Differentiable {
    type Output: Gradient + Overwrite;

    fn differentiable(&self) -> Self::Output;
}

/// Eval mode behaviour.
pub trait Eval {
    fn train(&self);

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
