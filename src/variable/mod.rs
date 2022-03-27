mod history;
mod node;
mod var;
mod vardiff;

pub use var::Var;
pub use vardiff::VarDiff;

pub(crate) use node::*;
pub use node::{
    Backward,
    /*Constant, Convolve, ConvolveWithGroups,*/
    Forward, /*PaddingMode, Reflective, Replicative, Zero,*/
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition ~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Vector Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Vector Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Vector Vector Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cat and Stack traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
