mod gradient;
mod history;
mod node;
mod utils;
mod var;
mod vardiff;

#[cfg(feature = "serialize")]
mod serde;

pub use var::Var;
pub use vardiff::VarDiff;

// pub use node::{
/*Constant, Convolve, ConvolveWithGroups,*/
/*PaddingMode, Reflective, Replicative, Zero,*/
//};

/// Matrix-matrix multiplication.
pub trait MatMatMul<Rhs> {
    /// The type of the matrix-matrix multiplication's result. See the
    /// [*differentiability arithmetic*] for more details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Computes the matrix-matrix multiplication between `self` and `other`.
    fn mm(self, other: Rhs) -> Self::Output;
}

/// Matrix-matrix multiplication with transposed right hand side operand.
///
/// This fused operation is marginally faster than performing the matrix-matrix multiplication
/// and transposition separately.
pub trait MatMatMulT<Rhs> {
    /// The type of the matrix-matrix multiplication with transposed right hand side operand's
    /// result. See the [*differentiability arithmetic*] for more details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Computes the matrix-matrix multiplication between `self` and transposed `other`.
    fn mm_t(self, other: Rhs) -> Self::Output;
}

/// Matrix-vector multiplication.
pub trait MatVecMul<Rhs> {
    /// The type of the matrix-vector multiplication's result. See the
    /// [*differentiability arithmetic*] for more details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Computes the matrix-vector multiplication between `self` and `other`.
    fn mv(self, other: Rhs) -> Self::Output;
}

/// Vector-matrix multiplication.
pub trait VecMatMul<Rhs> {
    /// The type of the vector-matrix multiplication's result. See the
    /// [*differentiability arithmetic*] for more details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Computes the vector-matrix multiplication between `self` and `other`.
    fn vm(self, other: Rhs) -> Self::Output;
}

/// Vector-vector multiplication, *a.k.a. dot product or inner product*.
pub trait VecVecMul<Rhs> {
    /// The type of the dot product's result. See the [*differentiability arithmetic*] for
    /// more details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Computes the dot product between `self` and `other`.
    fn vv(self, other: Rhs) -> Self::Output;
}

/// Concatenation.
pub trait Cat<Rhs> {
    /// The type of the concatenation's result. See the [*differentiability arithmetic*] for
    /// more details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Concatenates variables along the given axis.
    fn cat(self, other: Rhs, axis: usize) -> Self::Output;
}

/// Stacking.
pub trait Stack<Rhs> {
    /// The type of the stacking's result. See the [*differentiability arithmetic*] for
    /// more details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Stacks variables along the given axis.
    fn stack(self, other: Rhs, axis: usize) -> Self::Output;
}

/// Convolution.
pub trait Convolve<Rhs> {
    /// The type of the convolution's result. See the [*differentiability arithmetic*] for more
    /// details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Applies a *n*-dimensional convolution with the given parameters. *n* can be either 1, 2 or
    /// 3.
    fn convolve<T>(
        self,
        kernel: Rhs,
        stride: T,
        dilation: T,
        padding: T,
        groups: usize,
    ) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
