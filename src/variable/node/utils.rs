use ndarray::{Array, ArrayD, Axis, DimMax, Dimension, Ix0, Ix1, Ix2, Zip};
use std::{cell::RefCell, rc::Rc};

/// Shorthand for `Rc<RefCell<T>>`.
pub(crate) type Shared<T> = Rc<RefCell<T>>;
/// A broadcasted ndarray's dimension.
pub(crate) type Broadcast<D, E> = <D as DimMax<E>>::Output;

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

impl DotDim<Ix1> for Ix1 {
    type Output = Ix0;

    fn shape(_: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        Ix0::zeros(0)
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

/// Creates an empty tensor whose shape is the result of broadcasting between those of `left` and
/// `right`.
///
/// # Arguments
///
/// * `left` - left operand in the binary operations that admits broadcasting.
///
/// * `right` - right operand in the binary operations that admits broadcasting.
pub(crate) fn cobroadcasted_zeros<D, E>(
    left: &Array<f32, D>,
    right: &Array<f32, E>,
) -> Array<f32, Broadcast<D, E>>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    let (bigger, smaller) = if left.ndim() >= right.ndim() {
        (left.shape(), right.shape())
    } else {
        (right.shape(), left.shape())
    };
    let mut out = <D as DimMax<E>>::Output::zeros(bigger.len());
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

    Array::zeros(out)
}

/// Shrinks `array` summing in-place along `axis`.
///
/// # Arguments
///
/// * `array` - array to reduce.
///
/// * `axis` - axis to sum along to.
fn sum_axis_inplace(array: &mut ArrayD<f32>, axis: Axis) {
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
pub fn reduce<D, E>(dim: D, src: &Array<f32, E>) -> Array<f32, D>
where
    D: Dimension,
    E: Dimension,
{
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
