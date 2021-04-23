use super::{BroadTensor, Tensor};
use ndarray::{DimMax, Dimension, Ix1, Ix2};

pub mod backward;
pub mod forward;

pub use backward::*;
pub use forward::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub(super) fn broadcasted_zeros<Lhs, Rhs>(
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
    let b_dim = {
        let mut empty_d = <Lhs as DimMax<Rhs>>::Output::zeros(bigger.len());
        let empty_d_slice = empty_d.slice_mut();
        empty_d_slice
            .iter_mut()
            .zip(bigger.iter())
            .for_each(|(e_el, b_el)| *e_el = *b_el);
        empty_d_slice
            .iter_mut()
            .rev()
            .zip(smaller.iter().rev())
            .for_each(|(l, r)| *l = std::cmp::max(*l, *r));
        empty_d
    };
    Tensor::zeros(b_dim)
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
