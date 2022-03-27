mod node;
mod var;
mod vardiff;

use std::{
    cell::{RefCell, RefMut},
    collections::BTreeMap,
};

pub use var::Var;
pub use vardiff::VarDiff;

pub(crate) use node::*;
pub use node::{
    Backward,
    /*Constant, Convolve, ConvolveWithGroups,*/
    Forward, /*PaddingMode, Reflective, Replicative, Zero,*/
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ History ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Id of an operation in the tape. The first component is the address of the struct and the second
/// is the size of the history at insertion. The former is unique, the latter enforces order.
#[derive(Copy, Clone, Eq)]
struct HistoryId((usize, usize));

impl HistoryId {
    fn new(ptr: usize, order: usize) -> Self {
        Self((ptr, order))
    }
}

impl PartialEq for HistoryId {
    fn eq(&self, other: &Self) -> bool {
        let Self((lhs_ptr, _)) = self;
        let Self((rhs_ptr, _)) = other;

        lhs_ptr == rhs_ptr
    }
}

impl Ord for HistoryId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let Self((lhs_ptr, lhs_order)) = self;
        let Self((rhs_ptr, rhs_order)) = other;

        // equal only if the pointer is the same
        if lhs_ptr == rhs_ptr {
            return std::cmp::Ordering::Equal;
        }

        // same ordering but not equal is ok
        if lhs_order == rhs_order {
            return std::cmp::Ordering::Less;
        }

        lhs_order.cmp(&rhs_order)
    }
}

impl PartialOrd for HistoryId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
pub(crate) struct History<T>
where
    T: Clone,
{
    path: BTreeMap<HistoryId, T>,
    buffer: RefCell<Vec<T>>,
}

impl<T> History<T>
where
    T: Clone,
{
    /// Performs the merge between this history and another one.
    ///
    /// # Arguments
    ///
    /// `other` - other history.
    pub(crate) fn merge(&mut self, mut other: Self) {
        self.path.append(&mut other.path);
    }

    /// Appends a new computation to the history.
    ///
    /// # Arguments
    ///
    /// * `ptr` - address of the new node.
    ///
    /// * `op` - computation to append.
    pub(crate) fn insert(&mut self, ptr: usize, op: T) {
        let id = HistoryId::new(ptr, self.path.len());

        self.path.insert(id, op);
        self.buffer.borrow_mut().truncate(0);
    }

    /// Returns the length of the history.
    pub(crate) fn len(&self) -> usize {
        self.path.len()
    }

    /// Returns the length of the buffered history.
    pub(crate) fn buffer_len(&self) -> usize {
        self.buffer.borrow().len()
    }

    /// Returns the content of the tape in a vector.
    pub(crate) fn to_vec(&self) -> Vec<T> {
        self.path.values().cloned().collect()
    }

    /// Returns a reference to the buffer.
    pub(crate) fn buffer_mut(&self) -> RefMut<Vec<T>> {
        self.buffer.borrow_mut()
    }
}

impl<T> Default for History<T>
where
    T: Clone,
{
    fn default() -> Self {
        let path = BTreeMap::new();
        let buffer = RefCell::new(Vec::new());

        Self { path, buffer }
    }
}

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
