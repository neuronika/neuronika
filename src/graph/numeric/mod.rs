use ndarray::{
    arr1,
    linalg::{general_mat_mul, general_mat_vec_mul},
    Array, Axis, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, RemoveAxis, ShapeBuilder, Zip,
};
use std::fmt::{Display, Formatter, Result};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// **Private** local module that groups together all the stuff needed in this file and that should not be
/// exposed elsewere.
///
/// This module mimics the behaviour of the `C++ anonymous namespace`.
mod local {
    use ndarray::{Ix, Zip};

    /// Computes the broadcasted shape of two slices of [dimensions] inplace.
    ///
    /// This function must be used *only* with those dimensions' slice of same length that need
    /// to be compared, since the result of the extra ones is always a copy of them.
    ///
    /// This function is needed in order to avoid some duplicated code that would appear into the
    /// various [Broadcast trait] implementations.
    ///
    /// [Broadcast trait]: neuronika::Broadcast
    /// [dimensions]: ndarray::Dimension
    ///
    /// ## Parameters
    /// * `fst` - Mutable slice of dimensions in which to store the resulting shape
    /// * `snd` - Other slice of dimensions required for the comparison
    ///
    /// ## Panics
    /// If `fst` and `snd` have unbroadcastable shapes, that is when they differ and neither is `1`, then
    /// this function panics.
    ///
    /// ## Warnings
    /// This function is intended to be used only for interal purposes, thus should not be exported.
    #[inline]
    pub(super) fn compute_broadcasting(fst: &mut [Ix], snd: &[Ix]) {
        debug_assert_eq!(fst.len(), snd.len(), "incompatible lengths");

        Zip::from(fst).and(snd).apply(|l, r| {
            if l != r && *l != 1 && *r != 1 {
                panic!("cannot broadcast");
            }
            *l = std::cmp::max(*l, *r)
        });
    }
}

// =============================================== Tensor Type ===============================================

/// A *n*-dimensional [tensor] of *real* values that support efficient [broadcasting].
///
/// All the standard mathematic binary operators like `+`, `-`, `*` and `/`, exploit **SIMD** computation
/// and are also executed in multiple threads whenever possible.
///
/// [tensor]: https://en.wikipedia.org/wiki/Tensor
/// [broadcasting]: https://numpy.org/devdocs/user/theory.broadcasting.html
#[derive(Debug, PartialEq)]
pub struct Tensor<D>
where
    D: Dimension,
{
    /// Content of the tensor
    pub array: Array<f32, D>, // pub for now otherwise tests won't run
}

impl<D> Display for Tensor<D>
where
    D: Dimension,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.array)
    }
}

// ============================================= Types Relations =============================================

/// Broadcast relation among two [Dimension]s.
///
/// This trait is needed in order to use the *broadcasting* semantic of the standard algebraic operations
/// among [tensor]s.
///
/// [tensor]: neuronika::Tensor
pub trait Broadcast<R>
where
    Self: Dimension + RemoveAxis,
    R: Dimension + RemoveAxis,
{
    type Output: Dimension + RemoveAxis + Broadcast<R> + Broadcast<Self>;

    fn broadcast(self, rhs: R) -> Broadcasted<Self, R>;
}

/// The [Broadcast] result of two [dimension]s.
///
/// [Broadcast]: neuronica::Broadcast
/// [Dimension]: ndarray::Dimension
pub type Broadcasted<L, R> = <L as Broadcast<R>>::Output;

impl Broadcast<IxDyn> for IxDyn {
    type Output = IxDyn;

    fn broadcast(self, rhs: IxDyn) -> IxDyn {
        let lhs_dim = self.ndim();
        let rhs_dim = rhs.ndim();
        let mut res;
        if lhs_dim < rhs_dim {
            res = rhs;
            local::compute_broadcasting(&mut res.slice_mut()[rhs_dim - lhs_dim..], self.slice());
            return res;
        }
        res = self;
        if lhs_dim == rhs_dim {
            local::compute_broadcasting(res.slice_mut(), rhs.slice());
        } else {
            local::compute_broadcasting(&mut res.slice_mut()[lhs_dim - rhs_dim..], rhs.slice());
        }
        res
    }
}

/// Automatically implements all the trivial cases for the [Broadcast] relation.
///
/// [Broadcast]: neuronika::Broadcast
macro_rules! unary_broadcast_impls {
    ($($dim: ty),* $(,)*) => {
        $(
            impl Broadcast<IxDyn> for $dim {
                type Output = IxDyn;

                fn broadcast(self, rhs: IxDyn) -> IxDyn {
                    let lhs_dim = self.ndim();
                    let rhs_dim = rhs.ndim();
                    let mut res;
                    if lhs_dim < rhs_dim {
                        res = rhs.clone();
                        local::compute_broadcasting(&mut res.slice_mut()[rhs_dim - lhs_dim..], self.slice());
                        return res;
                    }
                    res = self.clone().into_dyn();
                    if lhs_dim == rhs_dim {
                        local::compute_broadcasting(res.slice_mut(), rhs.slice());
                    } else {
                        local::compute_broadcasting(&mut res.slice_mut()[lhs_dim - rhs_dim..], rhs.slice());
                    }
                    res
                }
            }

            impl Broadcast<$dim> for IxDyn {
                type Output = IxDyn;

                fn broadcast(self, rhs: $dim) -> IxDyn {
                    let lhs_dim = self.ndim();
                    let rhs_dim = rhs.ndim();
                    let mut res;
                    if lhs_dim < rhs_dim {
                        res = rhs.clone().into_dyn();
                        local::compute_broadcasting(&mut res.slice_mut()[rhs_dim - lhs_dim..], self.slice());
                        return res;
                    }
                    res = self.clone();
                    if lhs_dim == rhs_dim {
                        local::compute_broadcasting(res.slice_mut(), rhs.slice());
                    } else {
                        local::compute_broadcasting(&mut res.slice_mut()[lhs_dim - rhs_dim..], rhs.slice());
                    }
                    res
                }
            }

            impl Broadcast<$dim> for $dim {
                type Output = $dim;

                fn broadcast(self, rhs: $dim) -> $dim {
                    let mut res = self.clone();
                    local::compute_broadcasting(res.slice_mut(), rhs.slice());
                    res
                }
            }
        )*
    };
}

/// Automatically implements all the other cases of the [Broadcast] relation accordingly.
///
/// [Broadcast]: neuronika::Broadcast
macro_rules! binary_broadcast_impls {
    ($small: ty, $big: ty) => {
        impl Broadcast<$small> for $big {
            type Output = $big;

            fn broadcast(self, rhs: $small) -> $big {
                let mut res = self.clone();
                local::compute_broadcasting(&mut res.slice_mut()[(self.ndim() - rhs.ndim())..], rhs.slice());
                res
            }
        }

        impl Broadcast<$big> for $small {
            type Output = $big;

            fn broadcast(self, rhs: $big) -> $big {
                let mut res = rhs.clone();
                local::compute_broadcasting(&mut res.slice_mut()[(rhs.ndim() - self.ndim())..], self.slice());
                res
            }
        }
    };

    ($(($small: ty, $big: ty)),* $(,)*) => {
        $(binary_broadcast_impls!{$small, $big })*
    };
}

#[rustfmt::skip]
unary_broadcast_impls!(Ix1, Ix2, Ix3, Ix4, Ix5, Ix6);

#[rustfmt::skip]
binary_broadcast_impls! {
    (Ix1, Ix6), (Ix1, Ix5), (Ix1, Ix4), (Ix1, Ix3), (Ix1, Ix2),
    (Ix2, Ix6), (Ix2, Ix5), (Ix2, Ix4), (Ix2, Ix3),
    (Ix3, Ix6), (Ix3, Ix5), (Ix3, Ix4),
    (Ix4, Ix6), (Ix4, Ix5),
    (Ix5, Ix6)
}

// =========================================== Operators Overload ===========================================

/// Automatically implements the overload of the `+`, `-`, `*` and `/` binary algebraic operators for
/// [tensor]s.
///
/// [tensor]: crate::Tensor
macro_rules! operators_overload_impls {
    ($(($op: ident, $fun: ident, $sym: tt)),+ $(,)*) => {
        $(
            impl<L, R> $op<&Tensor<R>> for &Tensor<L>
            where
                L: Dimension + RemoveAxis + Broadcast<R>,
                R: Dimension + RemoveAxis,
            {
                type Output = Tensor<Broadcasted<L, R>>;

                fn $fun(self, rhs: &Tensor<R>) -> Self::Output {
                    let shape = self.array.raw_dim().broadcast(rhs.array.raw_dim());
                    let mut array = Array::<f32, Broadcasted<L, R>>::zeros(shape);
                    Zip::from(&mut array)
                        .and_broadcast(&self.array)
                        .and_broadcast(&rhs.array)
                        .par_apply(|res, l, r| *res = l $sym r);

                    Self::Output { array }
                }
            }
        )*
    };
}

operators_overload_impls!((Add, add, +), (Sub, sub, -), (Mul, mul, *), (Div, div, /));

impl<D> Neg for &Tensor<D>
where
    D: Dimension + RemoveAxis,
{
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        Self::Output {
            array: -&self.array,
        }
    }
}

// ============================================ Impl for Tensor Type ============================================

// Methods specific to the two dimensional Tensor.
impl Tensor<Ix2> {
    pub(super) fn mat_mul(
        &self,
        rhs: &Self,
        target: &mut Self,
        alpha: f32,
        beta: f32,
        t_lhs: bool,
        t_rhs: bool,
    ) {
        match (t_lhs, t_rhs) {
            (true, true) => general_mat_mul(
                alpha,
                &self.array.t(),
                &rhs.array.t(),
                beta,
                &mut target.array,
            ),
            (true, false) => {
                general_mat_mul(alpha, &self.array.t(), &rhs.array, beta, &mut target.array)
            }
            (false, true) => {
                general_mat_mul(alpha, &self.array, &rhs.array.t(), beta, &mut target.array)
            }
            (false, false) => {
                general_mat_mul(alpha, &self.array, &rhs.array, beta, &mut target.array)
            }
        }
    }

    pub(super) fn mat_vec_mul(
        &self,
        rhs: &Tensor<Ix1>,
        target: &mut Tensor<Ix1>,
        alpha: f32,
        beta: f32,
        t: bool,
    ) {
        match t {
            true => {
                general_mat_vec_mul(alpha, &self.array.t(), &rhs.array, beta, &mut target.array)
            }
            false => general_mat_vec_mul(alpha, &self.array, &rhs.array, beta, &mut target.array),
        }
    }
}

// Methods for all dimensional Tensors.
impl<D> Tensor<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn new(array: Array<f32, D>) -> Self {
        Tensor { array }
    }

    pub fn array(&self) -> &Array<f32, D> {
        &self.array
    }

    pub fn array_mut(&mut self) -> &mut Array<f32, D> {
        &mut self.array
    }

    // Gets the number of elements stored in this
    // Tensor.
    pub fn len(&self) -> usize {
        self.array.len()
    }

    // Gets the shape of the array stored in this
    // Tensor.
    pub fn shape(&self) -> &[usize] {
        self.array.shape()
    }

    // Initializes another Tensor with the same
    // dimensionsality as this with all its
    // values set to zero.
    pub fn zeros_from(&self) -> Self {
        Self {
            array: <Array<f32, D>>::zeros(self.array.raw_dim()),
        }
    }

    pub fn set_zero(&mut self) {
        self.array.map_inplace(|el| *el = 0.0);
    }

    // Creates another Tensor with transposed array.
    pub fn t(&self) -> Self {
        Self {
            array: self.array.t().to_owned(),
        }
    }

    // Creates another Tensor whose array is the sum of
    // all of self's elements.
    pub fn sum(&self) -> Tensor<Ix1> {
        Tensor {
            array: arr1(&[self.array.sum()]),
        }
    }

    /// Stacks a slice of [Tensor]s of [dimension] `D` upon a new axis, generating a new one with dimension `D+1`.
    ///
    /// Please, note that the new axis will have index `0`.
    ///
    /// [Tensor]: neuronika::Tensor
    /// [dimension]: ndarray::Dimension
    ///
    /// ## Parameters
    /// `tensors` - Slice of tensors to stack together.
    ///
    /// ## Panics
    /// If `tensors` is empty, or their shapes differ, then this function panics.
    ///
    /// ## Examples
    /// ```
    /// use neuronika::Tensor;
    ///
    /// let fst = Tensor::zeros(3); // 3 rows
    /// let snd = Tensor::zeros(3); // 3 rows
    ///
    /// // So, we expect that the result has 2 rows, and 3 columns
    /// assert_eq!(Tensor::stack(&[fst, snd]), Tensor::zeros((2, 3)));
    /// ```
    pub fn stack(tensors: &[Tensor<D>]) -> Tensor<D::Larger>
    where
        D::Larger: RemoveAxis,
    {
        let arrays: Vec<_> = tensors.iter().map(|t| t.array.view()).collect();

        Tensor {
            array: ndarray::stack_new_axis(Axis(0), &arrays[..]).unwrap(),
        }
    }

    /// Concatenates a slice of `D` [dimensional] [Tensor]s on an existing axis, generating a new
    /// one with same dimension and owning a copy of their content arranged properly.
    ///
    /// [Tensor]: neuronika::Tensor
    /// [dimension]: ndarray::Dimension
    ///
    /// ## Parameters
    /// `axis` - Axis upon which to perform the concatenation
    /// `tensors` - Slice of tensors to concatenate together.
    ///
    /// ## Panics
    /// If `axis` is out of bounds, `tensors` is empty, or they have mismatching shapes apart from
    /// along `axis`, then this function panics.
    ///
    /// ## Examples
    /// ```
    /// use neuronika::Tensor;
    /// use ndarray::Axis;
    ///
    /// let fst = Tensor::zeros((3, 1)); // 3 rows, 1 column
    /// let snd = Tensor::zeros((3, 4)); // 3 rows, 4 columns
    ///
    /// // So, we expect that the result of concatenation along
    /// // axis `1` has 3 rows and 5 columns
    /// assert_eq!(Tensor::concatenate(Axis(1), &[fst, snd]), Tensor::zeros((3, 5)));
    /// ```
    pub fn concatenate(axis: Axis, tensors: &[Tensor<D>]) -> Tensor<D> {
        let arrays: Vec<_> = tensors.iter().map(|t| t.array.view()).collect();

        Tensor {
            array: ndarray::concatenate(axis, &arrays[..]).unwrap(),
        }
    }

    /// Creates a zeroed tensor of the desired shape.
    pub fn zeros<S>(shape: S) -> Self
    where
        S: ShapeBuilder<Dim = D>,
    {
        Self {
            array: Array::zeros(shape),
        }
    }

    pub fn softmax(&self, axis: usize) -> Self {
        let mut new = self.zeros_from();
        Zip::from(self.array.lanes(Axis(axis)))
            .and(new.array.lanes_mut(Axis(axis)))
            .apply(|lane_self, lane_new| {
                let max = lane_self.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = &lane_self.map(|el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_new)
                    .and(num)
                    .apply(|lane_new_el, num_el| *lane_new_el = *num_el / den);
            });
        new
    }
}

#[cfg(test)]
mod tests;
