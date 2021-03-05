use super::Borrow;
use itertools::izip;
use ndarray::linalg::{general_mat_mul, general_mat_vec_mul};
use ndarray::{concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis, Ix, Zip};
use num_traits::pow;
use rand::thread_rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::cell::{Cell, RefMut};
use std::ops::{Add, Div, Mul, Neg, Sub};

pub(crate) type Vector = Array1<f32>; // One dimensional array.
pub(crate) type Matrix = Array2<f32>; // Two dimensional array.

// Abstraction over scalars, vectors and matrices.
#[derive(Debug)]
pub enum DataRepr {
    Scalar(f32),
    Vector(Vector),
    Matrix(Matrix),
}

pub trait Tensor {
    type Conte;
}

impl DataRepr {
    // Creates a vector of val.
    pub(super) fn constant_vec(shape: [usize; 1], val: f32) -> Self {
        Self::Vector(Vector::from_elem(shape, val))
    }

    // Creates a matrix of val.
    pub(super) fn constant_mat(shape: [usize; 2], val: f32) -> Self {
        Self::Matrix(Matrix::from_elem(shape, val))
    }

    // Wrapper around ndarray's eye method.
    pub(super) fn eye(n: Ix) -> Self {
        Self::Matrix(Matrix::eye(n))
    }

    // Creates a vector whose elements are drawn from
    // the uniform distribution U(low, high) -> [low, high).
    pub(super) fn uniform_vec(shape: [usize; 1], low: f32, high: f32) -> Self {
        let unif_dstr = Uniform::new(low, high);
        Self::Vector(Vector::from_shape_simple_fn(shape, || {
            unif_dstr.sample(&mut thread_rng())
        }))
    }

    // Creates a matrix whose elements are drawn from
    // the uniform distribution U(low, high) -> [low, high).
    pub(super) fn uniform_mat(shape: [usize; 2], low: f32, high: f32) -> Self {
        let unif_dstr = Uniform::new(low, high);
        Self::Matrix(Matrix::from_shape_simple_fn(shape, || {
            unif_dstr.sample(&mut thread_rng())
        }))
    }

    // Creates a vector whose elements are sampled from
    // the normal distribution N(mean, std^2).
    pub(super) fn normal_vec(shape: [usize; 1], mean: f32, std: f32) -> Self {
        let norm_dstr = Normal::new(mean, std).unwrap();
        Self::Vector(Vector::from_shape_simple_fn(shape, || {
            norm_dstr.sample(&mut thread_rng())
        }))
    }

    // Creates a matrix whose elements are sampled from
    // the normal distribution N(mean, std^2).
    pub(super) fn normal_mat(shape: [usize; 2], mean: f32, std: f32) -> Self {
        let norm_dstr = Normal::new(mean, std).unwrap();
        Self::Matrix(Matrix::from_shape_simple_fn(shape, || {
            norm_dstr.sample(&mut thread_rng())
        }))
    }

    // Used to extract a scalar from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn scalar(&self) -> f32 {
        match self {
            Self::Scalar(val) => *val,
            _ => panic!("error: not a scalar."),
        }
    }

    // Used to extract a vector from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn vector(&self) -> &Vector {
        match self {
            Self::Vector(val) => val,
            _ => panic!("error: not a vector."),
        }
    }

    // Used to extract a matrix from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn matrix(&self) -> &Matrix {
        match self {
            Self::Matrix(val) => val,
            _ => panic!("error: not a matrix."),
        }
    }

    // Used to extract a vector from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn vector_mut(&mut self) -> &mut Vector {
        match self {
            Self::Vector(val) => val,
            _ => panic!("error: not a vector."),
        }
    }

    // Used to extract a matrix from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn matrix_mut(&mut self) -> &mut Matrix {
        match self {
            Self::Matrix(val) => val,
            _ => panic!("error: not a matrix."),
        }
    }

    // Wrapper for ndarray's len method.
    // Returns the number of elements contained in
    // this DataRepr.
    fn len(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Vector(val) => val.len(),
            Self::Matrix(val) => val.len(),
        }
    }

    // Wrapper for ndarray's shape method.
    // Returns the shape of the underlying data.
    pub(super) fn shape(&self) -> &[usize] {
        match self {
            Self::Scalar(_) => &[1],
            Self::Vector(val) => val.shape(),
            Self::Matrix(val) => val.shape(),
        }
    }

    // Returns a new DataRepr struct corresponding to
    // the transposed self.
    pub(super) fn t(&self) -> Self {
        match self {
            // The transposition of a scalar is the
            // scalar.
            Self::Scalar(val) => Self::Scalar(*val),
            // The transposition of the vector is the vector,
            // because it has only one axis.
            Self::Vector(val) => {
                let mut new = Vector::zeros(val.raw_dim());
                Zip::from(&mut new)
                    .and(val)
                    .par_apply(|new_el, val_el| *new_el = *val_el);
                Self::Vector(new)
            }
            // The transposition of a matrix is a matrix
            // with swapped axes.
            Self::Matrix(val) => {
                let mut new = Matrix::zeros((val.ncols(), val.nrows()));
                Zip::from(&mut new)
                    .and(val.t())
                    .par_apply(|new_el, val_el| *new_el = *val_el);
                Self::Matrix(new)
            }
        }
    }

    // Initializes a DataRepr struct with the corresponding
    // zeroed value.
    pub(super) fn zeros(&self) -> Self {
        match self {
            Self::Scalar(_) => Self::Scalar(0.0),
            Self::Vector(val) => Self::Vector(Vector::zeros(val.raw_dim())),
            Self::Matrix(val) => Self::Matrix(Matrix::zeros(val.raw_dim())),
        }
    }

    // Computes the sum of this DataRepr struct's elements
    // and returns it in a scalar.
    pub(super) fn sum(&self) -> Self {
        match self {
            Self::Scalar(val) => Self::Scalar(*val),
            Self::Vector(val) => Self::Scalar(val.sum()),
            Self::Matrix(val) => Self::Scalar(val.sum()),
        }
    }

    // Set this DataRepr's value to 0. Should the value be a
    // vector or a matrix then all the elements are set to 0.
    pub(super) fn set_zero(&mut self) {
        match self {
            Self::Scalar(val) => *val = 0.0,
            Self::Vector(val) => val.par_map_inplace(|val| *val = 0.0),
            Self::Matrix(val) => val.par_map_inplace(|val| *val = 0.0),
        }
    }

    // A wrapper for the ndarray's map method.
    // Returns a new DataRepr identical in size,
    // whose elements are a function of self's
    // elements.
    pub(super) fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        match self {
            Self::Scalar(val) => Self::Scalar(f(*val)),
            Self::Vector(val) => Self::Vector(val.map(|val| f(*val))),
            Self::Matrix(val) => Self::Matrix(val.map(|val| f(*val))),
        }
    }

    // A wrapper for the ndarray's map_inplace method.
    // Modifies self inplace so that each element is a
    // function of itself.
    pub(super) fn map_inplace<F: Sync + Send>(&mut self, f: F)
    where
        F: Fn(f32) -> f32,
    {
        match self {
            Self::Scalar(val) => *val = f(*val),
            Self::Vector(val) => val.par_map_inplace(|val| *val = f(*val)),
            Self::Matrix(val) => val.par_map_inplace(|val| *val = f(*val)),
        }
    }

    // Concatenates the DataReprs along the given axis.
    pub(super) fn cat(slice: &[&Self], axis: usize) -> Self {
        // The underlying data must be of the same type.
        match slice[0] {
            // Scalars can only be concatenated with other scalars.
            Self::Scalar(_) => {
                let unboxed: Vec<f32> = slice.iter().map(|repr| repr.scalar()).collect();
                Self::Vector(Vector::from(unboxed))
            }
            // Vectors can only be concatenated with other vectors.
            Self::Vector(_) => {
                let unboxed: Vec<ArrayView1<f32>> =
                    slice.iter().map(|repr| repr.vector().view()).collect();
                Self::Vector(concatenate(Axis(axis), &unboxed).ok().unwrap())
            }
            // Matrixes can only be concatenated with other matrices.
            Self::Matrix(_) => {
                let unboxed: Vec<ArrayView2<f32>> =
                    slice.iter().map(|repr| repr.matrix().view()).collect();
                Self::Matrix(concatenate(Axis(axis), &unboxed).ok().unwrap())
            }
        }
    }

    // Computes the Softmax along an axis of self.
    pub(super) fn softmax(&self, axis: usize) -> Self {
        match (self, axis) {
            (Self::Vector(op_val), _) => {
                let max = op_val.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = op_val.map(|el| (el - max).exp());
                let den = num.sum();
                Self::Vector(num / den)
            }
            (Self::Matrix(op_val), 0) => {
                let mut new = Matrix::zeros(op_val.raw_dim());
                Zip::from(op_val.gencolumns())
                    .and(new.gencolumns_mut())
                    .apply(|col_op, mut col_new| {
                        let max = col_op.fold(std::f32::MIN, |x, y| x.max(*y));
                        let num = &col_op.map(|el| (el - max).exp());
                        let den = num.sum();
                        col_new.assign(&(num / den))
                    });
                Self::Matrix(new)
            }
            (Self::Matrix(op_val), 1) => {
                let mut new = Matrix::zeros(op_val.raw_dim());
                Zip::from(op_val.genrows())
                    .and(new.genrows_mut())
                    .apply(|row_op, mut row_new| {
                        let max = row_op.fold(std::f32::MIN, |x, y| x.max(*y));
                        let num = &row_op.map(|el| (el - max).exp());
                        let den = num.sum();
                        row_new.assign(&(num / den))
                    });
                Self::Matrix(new)
            }
            (_, _) => panic!("error: softmax is undefined for scalar inputs."),
        }
    }
}

// Forward action tracker. Ensures
// that the actual computation only happens
// when the node is fully accumulated.
#[derive(Debug, PartialEq)]
pub enum ForwardAction {
    Evaluate,
    Cached,
}

// Backward action tracker. Keeps track
// of the gradient accumulation.
#[derive(Debug, PartialEq)]
pub enum BackwardAction {
    // Set the gradient.
    Set,
    // Accumulates the gradient.
    Increment,
}

// Keeps track of the number of times
// that a node in the computational graph
// has been evaluated during either the forward
// or the backward pass.
#[derive(Debug, Default)]
pub struct PassCounter {
    forward_count: Cell<usize>,
    backward_count: Cell<usize>,
}

impl PassCounter {
    pub fn clear(&self) {
        self.forward_count.set(0);
        self.backward_count.set(0);
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.forward_count.get() == 0
    }

    pub fn recurse_backward(&self) -> bool {
        let backward_count = self.backward_count.get();
        let forward_count = self.forward_count.get();

        if backward_count == forward_count {
            self.clear();
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn forward_action(&self) -> ForwardAction {
        let count = self.forward_count.get();
        self.forward_count.set(count + 1);

        match count {
            0 => ForwardAction::Evaluate,
            _ => ForwardAction::Cached,
        }
    }

    #[inline(always)]
    pub fn backward_action(&self) -> BackwardAction {
        let backward_count = self.backward_count.get();

        let action = match backward_count {
            0 => BackwardAction::Set,
            _ => BackwardAction::Increment,
        };

        self.backward_count.set(backward_count + 1);
        action
    }
}

// Implements fast, parallelized and simd-enabled
// arithmetic operations between scalars and other
// DataRepr values. It mimics ndarray's arithmetic
// operations, as a consequence, the macro implemets
// two functions: scalar-others and others-scalar.
//
// Types: Scalar <op> Scalar -> Scalar
//        Scalar <op> Vector -> Vector
//        Scalar <op> Matrix -> Matrix
//        Vector <op> Scalar -> Vector
//        Matrix <op> Scalar -> Matrix
macro_rules! impl_ops_scal {
    ($fun_:ident, $_fun:ident, $type:ty, $op:tt) => {
        fn $fun_(
            res: &mut $type,
            lhs: f32,
            rhs: &$type,
        ) {
            Zip::from(res)
                .and(rhs)
                .par_apply(|res, rhs| *res = lhs $op *rhs);
        }
        fn $_fun(
            res: &mut $type,
            lhs: &$type,
            rhs: f32
        ) {
            Zip::from(res)
                .and(lhs)
                .par_apply(|res, lhs| *res = *lhs $op rhs);
        }
    };
}

// Scalar-vector and vector-scalar addition.
impl_ops_scal!(add_sv, add_vs, Vector, +);
// Scalar-vector and vector-scalar subtraction.
impl_ops_scal!(sub_sv, sub_vs, Vector, -);
// Scalar-vector and vector-scalar multiplication.
impl_ops_scal!(mul_sv, mul_vs, Vector, *);
// Scalar-vector and vector-scalar division.
impl_ops_scal!(div_sv, div_vs, Vector, /);

// Scalar-matrix and matrix-scalar addition.
impl_ops_scal!(add_sm, add_ms, Matrix, +);
// Scalar-matrix and matrix-scalar subtraction.
impl_ops_scal!(sub_sm, sub_ms, Matrix, -);
// Scalar-matrix and matrix-scalar multiplication.
impl_ops_scal!(mul_sm, mul_ms, Matrix, *);
// Scalar-matrix and matrix-scalar division.
impl_ops_scal!(div_sm, div_ms, Matrix, /);

// Implements fast, parallelized and simd-enabled
// arithmetic operations between non scalar DataRepr
// values. It mimics ndarray's arithmetic operations
// including shape broadcasting when needed.
//
// If two shapes are not broadcastable together
// ndarray's broadcasting panic error happens.
macro_rules! impl_ops {
    ($fun:ident,
        $lhs_type:ty,
        $rhs_type:ty,
        $res_type:ty,
        $op:tt) => {
        fn $fun
        (
            res: &mut $res_type,
            lhs: &$lhs_type,
            rhs: &$rhs_type
        ) {
            Zip::from(res)
                .and_broadcast(lhs)
                .and_broadcast(rhs)
                .par_apply(|res, lhs, rhs| *res = *lhs $op *rhs);
        }
    };
}

// Broadcasting only happens if one of the two operands
// is a singleton, in this case the singleton one
// behaves like a scalar and, as such, is commutative.
//
// Vector-vector addition.
impl_ops!(add_vv, Vector, Vector, Vector, +);
// Vector-vector subtraction.
impl_ops!(sub_vv, Vector, Vector, Vector, -);
// Vector-vector multiplication.
impl_ops!(mul_vv, Vector, Vector, Vector, *);
// Vector-vector division.
impl_ops!(div_vv, Vector, Vector, Vector, /);

// numpy's style operations. ndarray currently
// doesn't support this.
//
impl_ops!(add_vm, Vector, Matrix, Matrix, +);
// Vector-matrix subtraction.
impl_ops!(sub_vm, Vector, Matrix, Matrix, -);
// Vector-matrix multiplication.
impl_ops!(mul_vm, Vector, Matrix, Matrix,*);
// Vector-matrix division.
impl_ops!(div_vm, Vector, Matrix, Matrix, /);

// Broadcasting will only take place if the
// matrix has the same number of columns as
// the vector's elements.
//
// Matrix-vector addition.
impl_ops!(add_mv, Matrix, Vector, Matrix, +);
// Matrix-vector subtraction.
impl_ops!(sub_mv, Matrix, Vector, Matrix, -);
// Matrix-vector element-wise multiplication.
impl_ops!(mul_mv, Matrix, Vector, Matrix, *);
// Matrix-vector division.
impl_ops!(div_mv, Matrix, Vector, Matrix, /);

// In order for the broadcasting to take place
// one of the two following things must happen:
// either one of the two matrix is a singleton
// or, if it's not, it must have the same
// number of columns of the other but
// exactly one row.
//
// Matrix-matrix addition.
impl_ops!(add_mm, Matrix, Matrix, Matrix, +);
// Matrix-matrix subtraction.
impl_ops!(sub_mm, Matrix, Matrix, Matrix, -);
// Matrix-matrix element-wise multiplication.
impl_ops!(mul_mm, Matrix, Matrix, Matrix, *);
// Matrix-matrix division.
impl_ops!(div_mm, Matrix, Matrix, Matrix, /);

// Implements the Add, Sub, Mul and Div traits
// using the previously defined ops.
//
// The +, -, *, / operations all create a new
// DataRepr struct.
macro_rules! impl_arithmetic_ops {
    ($trait:ident,
        $fun:ident,
        $op:tt,
        $sv_op:ident,
        $sm_op:ident,
        $vs_op:ident,
        $vv_op:ident,
        $vm_op:ident,
        $ms_op:ident,
        $mv_op:ident,
        $mm_op:ident
    ) => {
        impl<'a> $trait<&'a DataRepr> for &'a DataRepr
        {
            type Output = DataRepr;

            fn $fun(self, rhs: Self) -> DataRepr {
                match (self, rhs)  {
                    (DataRepr::Scalar(lhs_val), DataRepr::Scalar(rhs_val)) => {
                        // Just sums the lhs and the rhs.
                        DataRepr::Scalar(*lhs_val $op *rhs_val)
                    },
                    (DataRepr::Scalar(lhs_val), DataRepr::Vector(rhs_val)) => {
                        // Performs the scalar-vector op.
                        let mut new = Vector::zeros(rhs_val.raw_dim());
                        $sv_op(&mut new, *lhs_val, rhs_val);
                        DataRepr::Vector(new)
                    },
                    (DataRepr::Scalar(lhs_val),  DataRepr::Matrix(rhs_val)) => {
                        // Performs the scalar-matrix op.
                        let mut new = Matrix::zeros(rhs_val.raw_dim());
                        $sm_op(&mut new, *lhs_val, rhs_val);
                        DataRepr::Matrix(new)
                    },
                    (DataRepr::Vector(lhs_val), DataRepr::Scalar(rhs_val)) => {
                        // Performs the vector-scalar op.
                        let mut new = Vector::zeros(lhs_val.raw_dim());
                        $vs_op(&mut new, lhs_val, *rhs_val);
                        DataRepr::Vector(new)
                    },
                    (DataRepr::Vector(lhs_val),   DataRepr::Vector(rhs_val)) => {
                        // Performs the vector-vector op.
                        let mut new = Vector::zeros(lhs_val.raw_dim());
                        $vv_op(&mut new, lhs_val, rhs_val);
                        DataRepr::Vector(new)
                    },
                    (DataRepr::Vector(lhs_val), DataRepr::Matrix(rhs_val)) => {
                        // Performs the vector-matrix op.
                        let mut new_dim = rhs_val.raw_dim();
                        // numpy's broadcasting rules.
                        new_dim[1] = if new_dim[1] == 1 {
                            lhs_val.len()
                        } else {
                            new_dim[1]
                        };
                        let mut new = Matrix::zeros(new_dim);
                        $vm_op(&mut new, lhs_val, rhs_val);
                        DataRepr::Matrix(new)
                    },
                    (DataRepr::Matrix(lhs_val) ,DataRepr::Scalar(rhs_val)) => {
                        // Performs the matrix-scalar op.
                        let mut new = Matrix::zeros(lhs_val.raw_dim());
                        $ms_op(&mut new, lhs_val, *rhs_val);
                        DataRepr::Matrix(new)
                    },
                    (DataRepr::Matrix(lhs_val), DataRepr::Vector(rhs_val)) => {
                        // Performs the matrix-scalar op.
                        let mut new_dim = lhs_val.raw_dim();
                        // numpy's broadcasting rules.
                        new_dim[1] = if new_dim[1] == 1 {
                            rhs_val.len()
                        } else {
                            new_dim[1]
                        };
                        let mut new = Matrix::zeros(new_dim);
                        $mv_op(&mut new, lhs_val, rhs_val);
                        DataRepr::Matrix(new)
                    },
                    (DataRepr::Matrix(lhs_val), DataRepr::Matrix(rhs_val)) => {
                        // Performs the matrix-matrix op.
                        let mut new_dim = lhs_val.raw_dim();
                        // numpy's broadcasting rules.
                        new_dim[1] = if new_dim[1] == 1 {
                            rhs_val.raw_dim()[1]
                        } else {
                            new_dim[1]
                        };
                        new_dim[0] = if new_dim[0] == 1 {
                            rhs_val.raw_dim()[0]
                        } else {
                            new_dim[0]
                        };
                        let mut new = Matrix::zeros(new_dim);
                        $mm_op(&mut new, lhs_val, rhs_val);
                        DataRepr::Matrix(new)
                    },
                }
            }
        }
    };
}

// Add trait implementation for the &DataRepr type.
impl_arithmetic_ops!(Add, add, +, add_sv, add_sm, add_vs, add_vv, add_vm, add_ms, add_mv, add_mm);
// Sub trait implementation for the &DataRepr type.
impl_arithmetic_ops!(Sub, sub, -, sub_sv, sub_sm, sub_vs, sub_vv, sub_vm, sub_ms, sub_mv, sub_mm);
// Mul trait implementation for the &DataRepr type.
impl_arithmetic_ops!(Mul, mul, *, mul_sv, mul_sm, mul_vs, mul_vv, mul_vm, mul_ms, mul_mv, mul_mm);
// Div trait implementation for the &DataRepr type.
impl_arithmetic_ops!(Div, div, /, div_sv, div_sm, div_vs, div_vv, div_vm, div_ms, div_mv, div_mm);

// Neg trait implementation.
impl Neg for &DataRepr {
    type Output = DataRepr;
    fn neg(self) -> DataRepr {
        self.map(|el| -el)
    }
}

// Implements the arithmetic operations used
// during the forward pass in the computational
// graph. Those operation are performed
// in place as the data field of the downstream
// node is just updated with the forwarded value/s.
macro_rules! impl_forward_arithmetic_ops {
    ($fun:ident,
        $op:tt,
        $sv_op:ident,
        $sm_op:ident,
        $vs_op:ident,
        $vv_op:ident,
        $vm_op:ident,
        $ms_op:ident,
        $mv_op:ident,
        $mm_op:ident
    ) => {
        pub(super) fn $fun(trgt: &mut DataRepr, lhs: &DataRepr, rhs: &DataRepr) {
            match trgt {
                // We already know the operands' types.
                // A scalar can only be the result of a scalar-scalar
                // operations.
                DataRepr::Scalar(_) => {
                    *trgt = DataRepr::Scalar(rhs.scalar() $op lhs.scalar());
                }
                // We already know the operands' types.
                // A vector can be the result of a scalar-vector or
                // vector-scalar or vector-vector operation.
                DataRepr::Vector(trgt_val) => {
                    // If lhs is a scalar then rhs is a vector.
                    if let DataRepr::Scalar(lhs_val) = lhs {
                        $sv_op(trgt_val, *lhs_val, rhs.vector());
                        // If rhs is a scalar then lhs is a vector.
                    } else if let DataRepr::Scalar(rhs_val) = rhs {
                        $vs_op(trgt_val, lhs.vector(), *rhs_val);
                        // At this point they must necessarily be
                        // both vectors.
                    } else {
                        $vv_op(trgt_val, lhs.vector(), rhs.vector())
                    }
                }
                // We already know the operands' types.
                // A matrix can be the result of a scalar-matrix or
                // matrix-scalar or vector-matrix or matrix-vector or
                // matrix-matrix operation.
                DataRepr::Matrix(trgt_val) => {
                    // If lhs is a scalar then rhs is a matrix.
                    if let DataRepr::Scalar(lhs_val) = lhs {
                        $sm_op(trgt_val, *lhs_val, rhs.matrix());
                        // If lhs is a vector then rhs is a vector.
                    } else if let DataRepr::Vector(lhs_val) = lhs {
                        $vm_op(trgt_val, lhs_val, rhs.matrix());
                        // If rhs is a scalar then lhs is a matrix.
                    } else if let DataRepr::Scalar(rhs_val) = rhs {
                        $ms_op(trgt_val, lhs.matrix(), *rhs_val);
                        // If rhs is a vector then lhs is a matrix.
                    } else if let DataRepr::Vector(rhs_val) = rhs {
                        $mv_op(trgt_val, lhs.matrix(), rhs_val);
                    } else {
                        // At this point they must necessarily be
                        // two matrices.
                        $mm_op(trgt_val, lhs.matrix(), rhs.matrix())
                    };
                }
            }
        }
    };
}

// Implements the add operation used in forward passes.
impl_forward_arithmetic_ops!(add, +, add_sv, add_sm, add_vs, add_vv, add_vm, add_ms, add_mv, add_mm);
// Implements the sub operation used in forward passes.
impl_forward_arithmetic_ops!(sub, -, sub_sv, sub_sm, sub_vs, sub_vv, sub_vm, sub_ms, sub_mv, sub_mm);
// Implements the mul operation used in forward passes.
impl_forward_arithmetic_ops!(mul, *, mul_sv, mul_sm, mul_vs, mul_vv, mul_vm, mul_ms, mul_mv, mul_mm);
// Implements the div operation used in forward passes.
impl_forward_arithmetic_ops!(div, /, div_sv, div_sm, div_vs, div_vv, div_vm, div_ms, div_mv, div_mm);

// Implements the gradient accumulation functions.
macro_rules! impl_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun(
            trgt: &mut DataRepr,
            src: &DataRepr,
        ) {
            match (trgt, src) {
                (DataRepr::Scalar(trgt_val), DataRepr::Scalar(src_val)) =>{
                    *trgt_val $op *src_val;
                },
                (DataRepr::Scalar(trgt_val), DataRepr::Vector(src_val)) => {
                    // Reduces src_val.
                    *trgt_val $op src_val.sum()
                },
                (DataRepr::Scalar(trgt_val), DataRepr::Matrix(src_val)) => {
                    // Reduces src_val.
                    *trgt_val $op src_val.sum()
                },
                (DataRepr::Vector(trgt_val), DataRepr::Scalar(src_val)) => {
                    Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op *src_val)
                }
                (DataRepr::Vector(trgt_val), DataRepr::Vector(src_val)) => {
                    // Checks wether one of the two vectors is bigger.
                    // The only admissible case is when either or both
                    // are a singletons.
                    if trgt_val.len() >= src_val.len() {
                        // This broadcast fails if src and trgt don't have
                        // the same shape.
                        Zip::from(trgt_val)
                            .and_broadcast(src_val)
                            .par_apply(|trgt_val, src_val| *trgt_val $op *src_val);
                    } else {
                        // trgt could be a singleton. Broadcasting fails if src
                        // and trgt don't have the same shape.
                        Zip::from(trgt_val)
                            .and_broadcast(&src_val.sum_axis(Axis(0)))
                            .par_apply(|trgt_val, src_val| *trgt_val $op *src_val);
                    }
                }
                (DataRepr::Vector(trgt_val), DataRepr::Matrix(src_val)) => {
                    // Reduces src.
                    let reduced_src = src_val.sum_axis(Axis(0));
                    // Checks wether one of the tow vectors is bigger.
                    // The only admissible case is when either or both
                    // are singletons.
                    if trgt_val.len() >= reduced_src.len() {
                        // This broadcast fails if src and trgt don't have
                        // the same shape.
                        Zip::from(trgt_val)
                            .and_broadcast(&reduced_src)
                            .par_apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src);
                    } else {
                        // trgt could be a singleton. Broadcasting fails if src
                        // and trgt don't have the same shape.
                        Zip::from(trgt_val)
                            .and_broadcast(&reduced_src.sum_axis(Axis(0)))
                            .apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src);
                    }
                },
                (DataRepr::Matrix(trgt_val), DataRepr::Scalar(src_val)) => {
                    Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op *src_val)
                }
                (DataRepr::Matrix(trgt_val), DataRepr::Vector(src_val)) => {
                    Zip::from(trgt_val)
                        .and_broadcast(src_val)
                        .apply(|trgt_val, src_val| *trgt_val $op *src_val)
                },
                (DataRepr::Matrix(trgt_val), DataRepr::Matrix(src_val)) => {
                    Zip::from(trgt_val)
                        .and_broadcast(src_val)
                        .par_apply(|trgt_val, src_val| *trgt_val $op *src_val)
                },
            }
        }
    };
}

// Accumulation assignment.
impl_accumulation_ops!(assign, =);
// Accumulation add assignemnt.
impl_accumulation_ops!(add_assign, +=);
// Accumulation sub assignment.
impl_accumulation_ops!(sub_assign, -=);

// Implements the gradient accumulation
// functions; the src is passed by value
// and thus consumed.
macro_rules! impl_accumulation_ops_v {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun(
            trgt: &mut DataRepr,
            src: DataRepr,
        ) {
            match (trgt, src) {
                (DataRepr::Scalar(trgt_val), DataRepr::Scalar(src_val)) => {
                    *trgt_val $op src_val;
                },
                (DataRepr::Scalar(trgt_val), DataRepr::Vector(src_val)) => {
                    // Reduces src_val.
                    *trgt_val $op src_val.sum()
                }
                (DataRepr::Scalar(trgt_val), DataRepr::Matrix(src_val)) => {
                    // Reduces src_val.
                    *trgt_val $op src_val.sum()
                },
                (DataRepr::Vector(trgt_val), DataRepr::Scalar(src_val)) => {
                        Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op src_val)
                }
                (DataRepr::Vector(trgt_val), DataRepr::Vector(src_val)) => {
                    // Checks wether one of the two vectors is bigger.
                    // The only admissible case is when either or both
                    // are a singletons.
                    if trgt_val.len() >= src_val.len() {
                        // This broadcast fails if src and trgt don't have
                        // the same shape.
                        Zip::from(trgt_val)
                            .and_broadcast(&src_val)
                            .par_apply(|trgt_val, src_val| *trgt_val $op *src_val);
                    } else {
                        // trgt could be a singleton. Broadcasting fails if src
                        // and trgt don't have the same shape.
                        Zip::from(trgt_val)
                            .and_broadcast(&src_val.sum_axis(Axis(0)))
                            .par_apply(|trgt_val, src_val| *trgt_val $op *src_val);
                    }
                },
                (DataRepr::Vector(trgt_val), DataRepr::Matrix(src_val)) => {
                    // Reduces src_val.
                    let reduced_src = src_val.sum_axis(Axis(0));
                    // Checks wether one of the two vectors is bigger.
                    // The only admissible case is when either or both
                    // are a singletons.
                    if trgt_val.len() >= reduced_src.len() {
                        // This broadcast fails if src and trgt don't have
                        // the same shape.
                        Zip::from(trgt_val)
                            .and_broadcast(&reduced_src)
                            .par_apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src);
                    } else {
                        // trgt could be a singleton. Broadcasting fails if src
                        // and trgt don't have the same shape.
                        Zip::from(trgt_val)
                            .and_broadcast(&reduced_src.sum_axis(Axis(0)))
                            .apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src);
                    }
                },
                (DataRepr::Matrix(trgt_val), DataRepr::Scalar(src_val)) => {
                        Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op src_val)
                },
                (DataRepr::Matrix(trgt_val), DataRepr::Vector(src_val)) => {
                    Zip::from(trgt_val)
                        .and_broadcast(&src_val)
                        .apply(|trgt_val, src_val| *trgt_val $op *src_val)
                },
                (DataRepr::Matrix(trgt_val), DataRepr::Matrix(src_val)) => {
                    Zip::from(trgt_val)
                        .and_broadcast(&src_val)
                        .par_apply(|trgt_val, src_val| *trgt_val $op *src_val)
                }
            }
        }
    };
}

// Accumulation assignment by value.
impl_accumulation_ops_v!(assign_v, =);
// Accumulation add assignemnt by value.
impl_accumulation_ops_v!(add_assign_v, +=);

// Used in the computation of the
// gradient of the division operation.
pub(super) fn div_assign_pow(trgt: &mut DataRepr, src: &DataRepr, exp: usize) {
    match (trgt, src) {
        (DataRepr::Scalar(trgt_val), DataRepr::Scalar(src_val)) => {
            *trgt_val /= pow(*src_val, exp);
        }
        (DataRepr::Scalar(trgt_val), DataRepr::Vector(src_val)) => {
            // Reduces the source but first applies the pow fun.
            let reduced_src = {
                let pow_res = src_val.mapv(|x| pow(x, exp));
                let sum_pow_res = pow_res.sum();
                sum_pow_res
            };
            *trgt_val /= reduced_src
        }
        (DataRepr::Scalar(trgt_val), DataRepr::Matrix(src_val)) => {
            // Reduces the source but first applies the pow fun.
            // Reduces the source but first applies the pow fun.
            let reduced_src = {
                let pow_res = src_val.mapv(|x| pow(x, exp));
                let sum_pow_res = pow_res.sum();
                sum_pow_res
            };
            *trgt_val /= reduced_src
        }
        (DataRepr::Vector(trgt_val), DataRepr::Scalar(src_val)) => {
            Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val /= pow(*src_val, exp))
        }
        (DataRepr::Vector(trgt_val), DataRepr::Vector(src_val)) => {
            if trgt_val.len() >= src_val.len() {
                Zip::from(trgt_val)
                    .and_broadcast(src_val)
                    .par_apply(|trgt_val, src_val| *trgt_val /= pow(*src_val, exp));
            } else {
                let reduced_src = src_val.mapv(|x| pow(x, exp)).sum_axis(Axis(0));
                Zip::from(trgt_val)
                    .and_broadcast(&reduced_src)
                    .par_apply(|trgt_val, src_val| *trgt_val /= *src_val);
            }
        }
        (DataRepr::Vector(trgt_val), DataRepr::Matrix(src_val)) => {
            let reduced_src = src_val.mapv(|x| pow(x, exp)).sum_axis(Axis(0));
            if trgt_val.len() >= reduced_src.len() {
                Zip::from(trgt_val)
                    .and_broadcast(&reduced_src)
                    .par_apply(|trgt_val, reduced_src| *trgt_val /= *reduced_src);
            } else {
                Zip::from(trgt_val)
                    .and_broadcast(&reduced_src.sum_axis(Axis(0)))
                    .apply(|trgt_val, reduced_src| *trgt_val /= *reduced_src);
            }
        }
        (DataRepr::Matrix(trgt_val), DataRepr::Scalar(src_val)) => {
            Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val /= pow(*src_val, exp))
        }
        (DataRepr::Matrix(trgt_val), DataRepr::Vector(src_val)) => Zip::from(trgt_val)
            .and_broadcast(src_val)
            .apply(|trgt_val, src_val| *trgt_val /= pow(*src_val, exp)),
        (DataRepr::Matrix(trgt_val), DataRepr::Matrix(src_val)) => Zip::from(trgt_val)
            .and_broadcast(src_val)
            .par_apply(|trgt_val, src_val| *trgt_val /= pow(*src_val, exp)),
    }
}

// Computes the power of the incoming
// data during the forward pass.
pub(super) fn pow_forward(data: &mut DataRepr, src: &DataRepr, exp: u16) {
    match (data, src) {
        (DataRepr::Scalar(data_val), DataRepr::Scalar(src_val)) => {
            *data_val = pow(*src_val, exp as usize)
        }
        (DataRepr::Vector(data_val), DataRepr::Vector(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| *data_el = pow(*src_el, exp as usize)),
        (DataRepr::Matrix(data_val), DataRepr::Matrix(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| *data_el = pow(*src_el, exp as usize)),
        _ => panic!("error: the two operands should have the same size."),
    }
}

// Used in the accumulation of the
// gradient of the pow op. Implements the
// necessary accumulation operations.
// Computes: d/dx x^n = n * x^(n-1).
macro_rules! impl_pow_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun (
            grad: &mut DataRepr,
            downstream_grad: &DataRepr,
            data: &DataRepr,
            exp: u16,
        )
        {
            match (grad, downstream_grad, data) {
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1 ) * exp as f32,
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1) * exp as f32
                    }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1) * exp as f32
                    }),
                _ => panic!("error: gradient and data should have the same size."),
            }
        }
    };
}

// Accumulation op for the set action.
impl_pow_accumulation_ops!(pow_diff_assign, =);
// Accumulation op for the increment action.
impl_pow_accumulation_ops!(pow_diff_add_assign, +=);

// Computes the ReLU of the incoming
// data during the forward pass.
pub(super) fn relu_forward(data: &mut DataRepr, src: &DataRepr) {
    match (data, src) {
        (DataRepr::Scalar(data_val), DataRepr::Scalar(src_val)) => {
            *data_val = if *src_val < 0.0 { 0.0 } else { *src_val }
        }
        (DataRepr::Vector(data_val), DataRepr::Vector(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| *data_el = if *src_el < 0.0 { 0.0 } else { *src_el }),
        (DataRepr::Matrix(data_val), DataRepr::Matrix(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| *data_el = if *src_el < 0.0 { 0.0 } else { *src_el }),
        _ => panic!("error: the two operands should have the same size."),
    }
}

// Used in the accumulation of the gradient
// of the ReLU op. Implements the necessary
// accumulation operations.
macro_rules! impl_relu_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun(
            grad: &mut DataRepr,
            downstream_grad: &DataRepr,
            data: &DataRepr,
        ) {
            match (grad, downstream_grad, data) {
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op if *data_val > 0.0 { *down_grad_val } else { 0.0 },
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op if *data_val > 0.0 { *down_grad_val } else { 0.0 }
                    }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op if *data_val > 0.0 { *down_grad_val } else { 0.0 }
                    }),
                _ => panic!("error: gradient and data should have the same size."),
            }
        }
    };
}

// Accumulation op for the set action.
impl_relu_accumulation_ops!(relu_diff_assign, =);
// Accumulation op for the increment action.
impl_relu_accumulation_ops!(relu_diff_add_assign, +=);

// Computes the LeakyReLU of the incoming
// data during the forward pass.
pub(super) fn leakyrelu_forward(data: &mut DataRepr, src: &DataRepr, slope: f32) {
    match (data, src) {
        (DataRepr::Scalar(data_val), DataRepr::Scalar(src_val)) => {
            *data_val = if *src_val < 0.0 {
                *src_val * slope
            } else {
                *src_val
            }
        }
        (DataRepr::Vector(data_val), DataRepr::Vector(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| {
                *data_el = if *src_el < 0.0 {
                    *src_el * slope
                } else {
                    *src_el
                }
            }),
        (DataRepr::Matrix(data_val), DataRepr::Matrix(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| {
                *data_el = if *src_el < 0.0 {
                    *src_el * slope
                } else {
                    *src_el
                }
            }),
        _ => panic!("error: the two operands should have the same size."),
    }
}

// Used in the accumulation of the gradient
// of the LeakyReLU op. Implements the necessary
// accumulation operations.
macro_rules! impl_leakyrelu_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun(
            grad: &mut DataRepr,
            downstream_grad: &DataRepr,
            data: &DataRepr,
            slope:f32
        ) {
            match (grad, downstream_grad, data) {
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op if *data_val > 0.0 { *down_grad_val } else { slope },
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op if *data_val > 0.0 { *down_grad_val } else { slope }
                    }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op if *data_val > 0.0 { *down_grad_val } else { slope }
                    }),
                _ => panic!("error: gradient and data should have the same size."),
            }
        }
    };
}

// Accumulation op for the set action.
impl_leakyrelu_accumulation_ops!(leakyrelu_diff_assign, =);
// Accumulation op for the increment action.
impl_leakyrelu_accumulation_ops!(leakyrelu_diff_add_assign, +=);

// Computes the Sigmoid of the incoming
// data during the forward pass. The sigmoid
// is clipped at +15, -15.
pub(super) fn sigmoid_forward(data: &mut DataRepr, src: &DataRepr) {
    match (data, src) {
        (DataRepr::Scalar(data_val), DataRepr::Scalar(src_val)) => {
            *data_val = if *src_val >= 15.0 {
                1.0
            } else if *src_val <= -15.0 {
                0.0
            } else {
                1.0 / (1.0 + (-*src_val).exp())
            }
        }
        (DataRepr::Vector(data_val), DataRepr::Vector(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| {
                *data_el = if *src_el >= 15.0 {
                    1.0
                } else if *src_el <= -15.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-*src_el).exp())
                }
            }),
        (DataRepr::Matrix(data_val), DataRepr::Matrix(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| {
                *data_el = if *src_el >= 15.0 {
                    1.0
                } else if *src_el <= -15.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-*src_el).exp())
                }
            }),
        _ => panic!("error: the two operands should have the same size."),
    }
}

// Used in the accumulation of the gradient
// of the Sigmoid op. Implements the necessary
// accumulation operations.
macro_rules! impl_sigmoid_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun(
            grad: &mut DataRepr,
            downstream_grad: &DataRepr,
            data: &DataRepr,
        ) {
            match (grad, downstream_grad, data) {
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op *down_grad_val * *data_val * (1.0 - *data_val),
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * *data_val * (1.0 - *data_val)
                    }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * *data_val * (1.0 - *data_val)
                    }),
                _ => panic!("erorr: gradient and data should have the same size."),
            }
        }
    };
}

// Accumulation op for the set action.
impl_sigmoid_accumulation_ops!(sigmoid_diff_assign, =);
// Accumulation op for the increment action.
impl_sigmoid_accumulation_ops!(sigmoid_diff_add_assign, +=);

// Computes the Exponential of the incoming
// data during the forward pass.
pub(super) fn exp_forward(data: &mut DataRepr, src: &DataRepr) {
    match (data, src) {
        (DataRepr::Scalar(data_val), DataRepr::Scalar(src_val)) => *data_val = src_val.exp(),
        (DataRepr::Vector(data_val), DataRepr::Vector(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| *data_el = src_el.exp()),
        (DataRepr::Matrix(data_val), DataRepr::Matrix(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| *data_el = src_el.exp()),
        _ => panic!("error: the two operands should have the same size."),
    }
}

// Used in the accumulation of the
// gradient of the Exp op. Implements
// the necessary accumulation operations.
macro_rules! impl_exp_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun(
            grad: &mut DataRepr,
            downstream_grad: &DataRepr,
            data: &DataRepr,
        ) {
            match (grad, downstream_grad, data) {
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op *down_grad_val * *data_val,
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * *data_val
                    }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * *data_val
                    }),
                _ => panic!("error: gradient and data should have the same size."),
            }
        }
    };
}

// Accumulation op for the set action.
impl_exp_accumulation_ops!(exp_diff_assign, =);
// Accumulation op for the increment action.
impl_exp_accumulation_ops!(exp_diff_add_assign, +=);

// Computes the Tanh of the incoming
// data during the forward pass.
pub(super) fn tanh_forward(data: &mut DataRepr, src: &DataRepr) {
    match (data, src) {
        (DataRepr::Scalar(data_val), DataRepr::Scalar(src_val)) => *data_val = src_val.tanh(),
        (DataRepr::Vector(data_val), DataRepr::Vector(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| *data_el = src_el.tanh()),
        (DataRepr::Matrix(data_val), DataRepr::Matrix(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| *data_el = src_el.tanh()),
        _ => panic!("error: the two operands should have the same size."),
    }
}

// Used in the accumulation of the
// gradient of the Tanh op. Implements
// the necessary accumulation operations.
macro_rules! impl_tanh_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun(
            grad: &mut DataRepr,
            downstream_grad: &DataRepr,
            data: &DataRepr,
        ) {
            match (grad, downstream_grad, data) {
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op *down_grad_val * (1.0 - pow(*data_val, 2)),
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * (1.0 - pow(*data_val, 2))
                    }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * (1.0 - pow(*data_val, 2))
                    }),
                _ => panic!("error: gradient and data should have the same size."),
            }
        }
    };
}

// Accumulation op for the set action.
impl_tanh_accumulation_ops!(tanh_diff_assign, =);
// Accumulation op for the increment action.
impl_tanh_accumulation_ops!(tanh_diff_add_assign, +=);

// Implements scalar-scaled accumulation
// operations.
macro_rules! impl_scaled_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun(
            trgt: &mut DataRepr,
            src: &DataRepr,
            scalar:f32
        ) {
            match (trgt, src) {
                (DataRepr::Scalar(trgt_val), DataRepr::Scalar(src_val)) => {
                    *trgt_val $op *src_val * scalar;
                },
                (DataRepr::Scalar(trgt_val), DataRepr::Vector(src_val)) => {
                    *trgt_val $op src_val.sum() * scalar
                }
                (DataRepr::Scalar(trgt_val), DataRepr::Matrix(src_val)) => {
                    *trgt_val $op src_val.sum() * scalar
                },
                (DataRepr::Vector(trgt_val), DataRepr::Scalar(src_val)) => {
                    Zip::from(trgt_val)
                        .par_apply(|trgt_val| *trgt_val $op *src_val * scalar)
                },
                (DataRepr::Vector(trgt_val), DataRepr::Vector(src_val)) => {
                    if trgt_val.len() >= src_val.len() {
                        Zip::from(trgt_val)
                            .and_broadcast(src_val)
                            .par_apply(|trgt_val, src_val| *trgt_val $op *src_val * scalar);
                    } else {
                        Zip::from(trgt_val)
                            .and_broadcast(&src_val.sum_axis(Axis(0)))
                            .par_apply(|trgt_val, src_val| *trgt_val $op *src_val * scalar);
                    }
                },
                (DataRepr::Vector(trgt_val), DataRepr::Matrix(src_val)) => {
                    let reduced_src = src_val.sum_axis(Axis(0));
                    if trgt_val.len() >= reduced_src.len() {
                        Zip::from(trgt_val)
                            .and_broadcast(&reduced_src)
                            .par_apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src * scalar);
                    } else {
                        Zip::from(trgt_val)
                            .and_broadcast(&reduced_src.sum_axis(Axis(0)))
                            .apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src * scalar);
                    }
                },
                (DataRepr::Matrix(trgt_val), DataRepr::Scalar(src_val))=> {
                        Zip::from(trgt_val)
                            .par_apply(|trgt_val| *trgt_val $op *src_val * scalar)
                },
                (DataRepr::Matrix(trgt_val), DataRepr::Vector(src_val)) => {
                    Zip::from(trgt_val)
                        .and_broadcast(src_val)
                        .apply(|trgt_val, src_val| *trgt_val $op *src_val * scalar)
                },
                (DataRepr::Matrix(trgt_val), DataRepr::Matrix(src_val)) => {
                    Zip::from(trgt_val)
                        .and_broadcast(src_val)
                        .par_apply(|trgt_val, src_val| *trgt_val $op *src_val * scalar)
                }
            }
        }
    };
}

// Scaled accumulation assignment, permforms a = b * c.
impl_scaled_accumulation_ops!(scaled_assign, =);
// Scaled accumulation add-assignment, performs a += b * c.
impl_scaled_accumulation_ops!(scaled_add_assign, +=);

// Wrapper for the ndarray's gen_mat_mul method.
// Computes trgt = alpha * lhs x rhs + beta * trgt.
pub(super) fn mat_mat_mul(
    trgt: &mut DataRepr,
    alpha: f32,
    lhs: &DataRepr,
    rhs: &DataRepr,
    beta: f32,
    t_lhs: bool, // When true trasposes lhs.
    t_rhs: bool, // When true transposes rhs.
) {
    if t_lhs {
        general_mat_mul(
            alpha,
            &lhs.matrix().t(),
            &rhs.matrix(),
            beta,
            trgt.matrix_mut(),
        );
        return;
    }

    if t_rhs {
        general_mat_mul(
            alpha,
            &lhs.matrix(),
            &rhs.matrix().t(),
            beta,
            trgt.matrix_mut(),
        );
        return;
    }

    general_mat_mul(alpha, &lhs.matrix(), &rhs.matrix(), beta, trgt.matrix_mut());
}

// Wrapper for the ndarray's gen_mat_vec_mul method.
// Computes trgt = alpha * lhs x rhs + beta * trgt.
pub(super) fn mat_vec_mul(
    trgt: &mut DataRepr,
    alpha: f32,
    lhs: &DataRepr,
    rhs: &DataRepr,
    beta: f32,
    transpose: bool,
) {
    if transpose {
        general_mat_vec_mul(
            alpha,
            &lhs.matrix().t(),
            rhs.vector(),
            beta,
            &mut trgt.vector_mut(),
        );
        return;
    }
    general_mat_vec_mul(
        alpha,
        lhs.matrix(),
        rhs.vector(),
        beta,
        &mut trgt.vector_mut(),
    );
}

// Backpropagates the gradient for the lhs operand
// in the mat_vec_mul op.
pub(super) fn mat_vec_mul_backward_lhs(lhs_grad: &mut DataRepr, grad: &Vector, rhs: &Vector) {
    lhs_grad
        .matrix_mut()
        .genrows_mut()
        .into_iter()
        .zip(grad.into_iter())
        .for_each(|(mut row, grad_el)| {
            row.assign(&rhs.map(|el| el * grad_el));
        });
}

// Computes the binary concatenation of lhs
// and rhs during the forward pass.
pub(super) fn cat_forward(dest: &mut DataRepr, lhs: &DataRepr, rhs: &DataRepr, axis: usize) {
    match (dest, axis) {
        (DataRepr::Vector(dest_val), _) => {
            // Chains lhs and rhs, zips the result
            // to dest and then performs
            // the assignment.
            dest_val
                .iter_mut()
                .zip(lhs.vector().iter().chain(rhs.vector().iter()))
                .for_each(|(dest_el, fused_el)| *dest_el = *fused_el);
        }
        (DataRepr::Matrix(dest_val), 0) => {
            // Chains lhs and rhs, zips the result
            // to dest and then performs
            // the assignment.
            dest_val
                .iter_mut()
                .zip(lhs.matrix().iter().chain(rhs.matrix().iter()))
                .for_each(|(dest_el, fused_el)| *dest_el = *fused_el);
        }
        (DataRepr::Matrix(dest_val), 1) => {
            // The concatenation must take place along the
            // columns.

            // Zips the genrows iterators.
            for (mut dest_row, lhs_row, rhs_row) in izip!(
                dest_val.genrows_mut().into_iter(),
                lhs.matrix().genrows().into_iter(),
                rhs.matrix().genrows().into_iter()
            ) {
                let dest_row = dest_row.as_slice_mut().unwrap();
                let lhs_row = lhs_row.as_slice().unwrap();
                let rhs_row = rhs_row.as_slice().unwrap();

                // Splits each dest row in two parts accordingly
                // to the lhs row lenght and it assigns to the
                // resulting slices the lhs row and the rhs row.
                let (dest_left, dest_right) = dest_row.split_at_mut(lhs_row.len());
                Zip::from(dest_left)
                    .and(lhs_row)
                    .apply(|left_el, lhs_el| *left_el = *lhs_el);
                Zip::from(dest_right)
                    .and(rhs_row)
                    .apply(|right_el, rhs_el| *right_el = *rhs_el);
            }
        }
        _ => panic!("erorr: too large of an axis or scalar result."),
    }
}

// Implements the binary concatenation
// accumulation operations.
macro_rules! impl_cat_acc_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun(
            lhs_grad: &mut DataRepr,
            rhs_grad: &mut DataRepr,
            grad: &DataRepr,
            axis: usize,
        ) {
            match (lhs_grad, rhs_grad, axis, grad) {
                (
                    DataRepr::Vector(lhs_grad_val),
                    DataRepr::Vector(rhs_grad_val),
                    _,
                    DataRepr::Vector(grad_val)
                ) => {
                        // Splits the incoming gradient
                        // accordingly to lhs_grad_val len, then
                        // it performs the assignment.
                        let (grad_val_left, grad_val_right) =
                            grad_val.as_slice().unwrap().split_at(lhs_grad_val.len());
                        Zip::from(lhs_grad_val)
                            .and_broadcast(grad_val_left)
                            .apply(|left_el, lhs_el| *left_el $op *lhs_el);
                        Zip::from(rhs_grad_val)
                            .and_broadcast(grad_val_right)
                            .apply(|right_el, rhs_el| *right_el $op *rhs_el);
                    },
                (
                    DataRepr::Matrix(lhs_grad_val),
                    DataRepr::Matrix(rhs_grad_val),
                    0,
                    DataRepr::Matrix(grad_val)
                ) => {
                        // Chains the lhs and rhs genrows
                        // iterators then zips the result
                        // to the grad_val genrows iterator.
                        // At that point, it performs the element
                        // wise assignment.
                        grad_val
                            .genrows()
                            .into_iter()
                            .zip(
                                lhs_grad_val
                                    .genrows_mut()
                                    .into_iter()
                                    .chain(rhs_grad_val.genrows_mut()),
                            )
                            .for_each(|(grad_row, mut dest_row)| {
                                let grad_row = grad_row.as_slice().unwrap();
                                let dest_row = dest_row.as_slice_mut().unwrap();

                                dest_row
                                    .iter_mut()
                                    .zip(grad_row.iter())
                                    .for_each(|(dest_el, grad_el)| {
                                        *dest_el $op *grad_el;
                                    });
                            });
                },
                (
                    DataRepr::Matrix(lhs_grad_val),
                    DataRepr::Matrix(rhs_grad_val),
                    1,
                    DataRepr::Matrix(grad_val)
                ) => {
                    // Zips the genrows iterators, then splits each of
                    // the gradient rows and performs the assignment.
                    izip!(
                        grad_val.genrows().into_iter(),
                        lhs_grad_val.genrows_mut().into_iter(),
                        rhs_grad_val.genrows_mut().into_iter()
                    )
                    .for_each(|(grad_row, mut lhs_row, mut rhs_row)| {
                        let grad_row = grad_row.as_slice().unwrap();
                        let lhs_row = lhs_row.as_slice_mut().unwrap();
                        let rhs_row = rhs_row.as_slice_mut().unwrap();

                        let (grad_val_left, grad_val_right) = grad_row.split_at(lhs_row.len());

                        lhs_row
                            .iter_mut()
                            .zip(grad_val_left.iter())
                            .for_each(|(lhs, grad)| {
                                *lhs $op *grad;
                            });

                        rhs_row
                            .iter_mut()
                            .zip(grad_val_right.iter())
                            .for_each(|(rhs, grad)| {
                                *rhs $op *grad;
                            });
                        });
                },
                _ => panic!("error: operands' or gradinents' type mismatch in concatenation."),
            }
        }
    };
}

// Accumulation op for the set action.
impl_cat_acc_ops!(cat_backward_assign, =);
// Accumulation op for the increment action.
impl_cat_acc_ops!(cat_backward_add_assign, +=);

// Computes the multiple concatenation of the
// incoming data during the forward pass.
pub(super) fn multicat_forward(dest: &mut DataRepr, srcs: Vec<Borrow<DataRepr>>, axis: usize) {
    let mut offset: usize = 0;
    match (&dest, axis) {
        (DataRepr::Vector(_), _) => {
            // If dest is a vector then the operands' data are all vectors,
            // otherwise panics. Gets their lengths.
            let lens: Vec<usize> = srcs.iter().map(|bor| bor.len()).collect();

            // Zips the lengths and the operands' DataReprs.
            lens.iter().zip(srcs.iter()).for_each(|(len, op_data)| {
                // Zips a mutable view of appropriate length of
                // dest with the matching operand's data
                // and performs the assignment.
                Zip::from(dest.vector_mut().slice_mut(s![offset..offset + len]))
                    .and(op_data.vector())
                    .par_apply(|data_el, op_data_el| *data_el = *op_data_el);
                // Increases the offset.
                offset += len;
            });
        }
        (DataRepr::Matrix(_), 0) => {
            // If the data is a matrix then the operands' data are all matrices,
            // otherwise panics. Gets their shapes.
            let shapes: Vec<&[usize]> = srcs.iter().map(|bor| bor.shape()).collect();

            // Zips the shapes and the operands' DataReprs.
            shapes.iter().zip(srcs.iter()).for_each(|(shape, op_data)| {
                // Zips a mutable view of appropriate size of the
                // data with the matching operand's data
                // and performs the assignment.
                Zip::from(
                    dest.matrix_mut()
                        .slice_mut(s![offset..offset + shape[0], ..]),
                )
                .and(op_data.matrix())
                .par_apply(|data_el, op_data_el| *data_el = *op_data_el);
                // Increases the offset along axis 0.
                offset += shape[0];
            });
        }
        (DataRepr::Matrix(_), 1) => {
            // If the data is a matrix then the operands' data are all matrices,
            // otherwise panics. Gets their shapes.
            let shapes: Vec<&[usize]> = srcs.iter().map(|bor| bor.shape()).collect();

            // Zips the shapes and the operands' DataReprs.
            shapes.iter().zip(srcs.iter()).for_each(|(shape, op_data)| {
                // Zips a mutable view of appropriate size of the
                // data with the matching operand's data
                // and performs the assignment.
                Zip::from(
                    dest.matrix_mut()
                        .slice_mut(s![.., offset..offset + shape[1]]),
                )
                .and(op_data.matrix())
                .par_apply(|data_el, op_data_el| *data_el = *op_data_el);
                // Increases the offset along axis 1.
                offset += shape[1];
            });
        }
        (_, _) => panic!("erorr: too large of an axis or scalar result."),
    }
}

// Implements the binary concatenation
// accumulation operations.
macro_rules! impl_multi_cat_acc_ops {
    ($fn:ident, $op:tt, $acc_fun:ident) => {
        pub(super) fn $fn(
            data: &DataRepr,
            ops_data: Vec<Borrow<DataRepr>>,
            grad: &DataRepr,
            dest_grads: &mut [RefMut<DataRepr>],
            axis: usize,
        ) {
            match (data, grad, axis) {
                (DataRepr::Vector(_), DataRepr::Vector(grad_val), _) => {
                    // If the downstream gradient is a vector gets the lenght of all
                    // the components, which are also vectors.
                    let lens: Vec<usize> = ops_data.iter().map(|bor| bor.len()).collect();
                    // The offset is used to sum the corresponding components of the
                    // downstream gradient.
                    let mut offset: usize = 0;

                    // Zips the lenghts and the component's gradient so that
                    // when a gradient has been fully accumulated the next
                    // one will recieve the right portion of the downstream
                    // gradient.
                    lens.iter().zip(dest_grads).for_each(|(len, op_grad)|{
                        Zip::from(op_grad.vector_mut())
                            .and_broadcast(grad_val.slice(s![offset..offset + *len]))
                            .par_apply(|op_grad_el, grad_vec_el| *op_grad_el $op *grad_vec_el);
                        // Inrements the offset.
                        offset += len;
                    });
                },
                (DataRepr::Matrix(_), DataRepr::Matrix(grad_val), 0) => {
                    // Gets the shapes of the components, whic are matrices.
                    let shapes: Vec<&[usize]> = ops_data.iter().map(|bor| bor.shape()).collect();
                    let mut offset: usize = 0;

                    // Sums to each component's grad the corresponing subview
                    // of the incoming gradient.
                    shapes.iter().zip(dest_grads).for_each(|(shape, op_grad)| {
                        Zip::from(op_grad.matrix_mut())
                            .and_broadcast(grad_val.slice(s![offset..offset + shape[0], ..]))
                            .par_apply(|op_grad_el, grad_vec_el| *op_grad_el $op *grad_vec_el);
                        // Increments the offset with the shape component
                        // corresponding to the rows axis.
                        offset += shape[0];
                    });
                },
                (DataRepr::Matrix(_), DataRepr::Matrix(grad_val), 1) => {
                    // Gets the shapes of the components, which are matrices.
                    let shapes: Vec<&[usize]> = ops_data.iter().map(|bor| bor.shape()).collect();
                    let mut offset: usize = 0;

                    // Sums to each component's grad the corresponing subview
                    // of the incoming gradient.
                    shapes.iter().zip(dest_grads).for_each(|(shape, op_grad)| {
                        Zip::from(op_grad.matrix_mut())
                            .and_broadcast(grad_val.slice(s![.., offset..offset + shape[1]]))
                            .par_apply(|op_grad_el, grad_vec_el| *op_grad_el $op *grad_vec_el);
                        // Increments the offset with the shape component
                        // corresponding to the columns axis.
                        offset += shape[1];
                    });
                },
                _ => panic!("error: operands's type mismatch in concatenation."),
            }
        }
    };
}

// Accumulation op for the set action.
impl_multi_cat_acc_ops!(multicat_backward_assign, =, assign);
// Accumulation op for the increment action.
impl_multi_cat_acc_ops!(multicat_backward_add_assign, +=, add_assign);

// Softmax forward operation.
pub(super) fn softmax_forward(dest: &mut DataRepr, src: &DataRepr, axis: usize) {
    match (dest, axis) {
        (DataRepr::Vector(dest_val), _) => {
            let src_val = src.vector();
            let max = src_val.fold(std::f32::MIN, |x, y| x.max(*y));
            let num = src_val.map(|el| (el - max).exp());
            let den = num.sum();
            Zip::from(dest_val)
                .and(&num)
                .apply(|src_el, num_el| *src_el = num_el / den);
        }
        (DataRepr::Matrix(dest_val), 0) => {
            let src_val = src.matrix();
            Zip::from(src_val.gencolumns())
                .and(dest_val.gencolumns_mut())
                .apply(|col_src, mut col_dest| {
                    let max = col_src.fold(std::f32::MIN, |x, y| x.max(*y));
                    let num = &col_src.map(|el| (el - max).exp());
                    let den = num.sum();
                    col_dest.assign(&(num / den))
                });
        }
        (DataRepr::Matrix(dest_val), 1) => {
            let src_val = src.matrix();
            Zip::from(src_val.genrows())
                .and(dest_val.genrows_mut())
                .apply(|row_src, mut row_dest| {
                    let max = row_src.fold(std::f32::MIN, |x, y| x.max(*y));
                    let num = &row_src.map(|el| (el - max).exp());
                    let den = num.sum();
                    row_dest.assign(&(num / den))
                });
        }
        (_, _) => panic!("error: invalid arguments."),
    }
}

pub(super) fn softmax_backward(
    dest_grad: &mut DataRepr,
    input_grad: &DataRepr,
    data: &DataRepr,
    jacobian: &mut Matrix,
    action: f32,
    axis: usize,
) {
    // Fills the Jacobian J of the softmax s such that
    // J[i][k] = s[i] * (1- s[k]) if i == k.
    // J[i][k] = - s[i] * s[k] if i != k.
    // The Jacobian of the Softmax is symmetric.
    fn fill_jacobian(jacobian: &mut Matrix, data: &ArrayView1<f32>) {
        for (row_idx, (mut row, row_val)) in jacobian
            .genrows_mut()
            .into_iter()
            .zip(data.iter())
            .enumerate()
        {
            for (col_idx, (grad, col_val)) in row
                .as_slice_mut()
                .unwrap()
                .iter_mut()
                .zip(data.iter())
                .enumerate()
            {
                if row_idx == col_idx {
                    *grad = row_val * (1.0 - col_val);
                } else {
                    *grad = -row_val * col_val;
                }
            }
        }
    };

    match (dest_grad, input_grad, axis) {
        (DataRepr::Vector(dest_grad_val), DataRepr::Vector(input_grad_val), _) => {
            fill_jacobian(jacobian, &data.vector().view());
            general_mat_vec_mul(1.0, &jacobian, &input_grad_val, action, dest_grad_val);
        }
        (DataRepr::Matrix(dest_grad_val), DataRepr::Matrix(input_grad_val), 0) => {
            Zip::from(dest_grad_val.gencolumns_mut())
                .and(data.matrix().gencolumns())
                .and(input_grad_val.gencolumns())
                .apply(|mut d_g_col, data_col, grad_col| {
                    fill_jacobian(jacobian, &data_col);
                    general_mat_vec_mul(1.0, &jacobian, &grad_col, action, &mut d_g_col);
                })
        }
        (DataRepr::Matrix(dest_grad_val), DataRepr::Matrix(input_grad_val), 1) => {
            Zip::from(dest_grad_val.genrows_mut())
                .and(data.matrix().genrows())
                .and(input_grad_val.genrows())
                .apply(|mut d_g_row, data_row, grad_row| {
                    fill_jacobian(jacobian, &data_row);
                    general_mat_vec_mul(1.0, &jacobian, &grad_row, action, &mut d_g_row);
                })
        }
        (_, _, _) => panic!("error: operands's type mismatch in softmax."),
    }
}

#[cfg(test)]
mod tests;
