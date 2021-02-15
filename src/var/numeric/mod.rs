use ndarray::linalg::{general_mat_mul, general_mat_vec_mul};
use ndarray::{Array1, Array2, Axis, Zip};
use num_traits::{pow, Float, One, Zero};
use std::cell::Cell;
use std::convert::TryFrom;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub(crate) type Vector<A> = Array1<A>; // One dimensional array.
pub(crate) type Matrix<A> = Array2<A>; // Two dimensional array.

// Abstraction over scalars, vectors and matrices.
#[derive(Debug)]
pub enum DataRepr<A>
where
    A: Copy,
{
    Scalar(A),
    Vector(Vector<A>),
    Matrix(Matrix<A>),
}

impl<A> DataRepr<A>
where
    A: Copy + Send + Sync,
{
    // Used to extract a scalar from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn scalar(&self) -> A {
        match self {
            Self::Scalar(val) => *val,
            _ => panic!("error: not a scalar."),
        }
    }

    // Used to extract a vector from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn vector(&self) -> &Vector<A> {
        match self {
            Self::Vector(val) => &val,
            _ => panic!("error: not a vector."),
        }
    }

    // Used to extract a matrix from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn matrix(&self) -> &Matrix<A> {
        match self {
            Self::Matrix(val) => &val,
            _ => panic!("error: not a matrix."),
        }
    }

    // Used to extract a vector from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn vector_mut(&mut self) -> &mut Vector<A> {
        match self {
            Self::Vector(val) => val,
            _ => panic!("error: not a vector."),
        }
    }

    // Used to extract a matrix from the DataRepr
    // struct when the value's type can be determined
    // with certainty.
    fn matrix_mut(&mut self) -> &mut Matrix<A> {
        match self {
            Self::Matrix(val) => val,
            _ => panic!("error: not a matrix."),
        }
    }
}

impl<A> DataRepr<A>
where
    A: Copy + Send + Sync + Zero + One,
{
    // Initializes a DataRepr struct with the corresponding
    // zeroed type value.
    pub(super) fn zeros(&self) -> Self {
        match self {
            Self::Scalar(_) => Self::Scalar(A::zero()),
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
            Self::Scalar(val) => *val = A::zero(),
            Self::Vector(val) => val.par_map_inplace(|val| *val = A::zero()),
            Self::Matrix(val) => val.par_map_inplace(|val| *val = A::zero()),
        }
    }

    // A wrapper for the ndarray's map method.
    pub(super) fn map<F, B: Copy>(&self, f: F) -> DataRepr<B>
    where
        F: Fn(&A) -> B,
    {
        match self {
            Self::Scalar(val) => DataRepr::Scalar(f(val)),
            Self::Vector(val) => DataRepr::Vector(val.map(|val| f(val))),
            Self::Matrix(val) => DataRepr::Matrix(val.map(|val| f(val))),
        }
    }

    // A wrapper for the ndarray's map_inplace method.
    pub(super) fn map_inplace<F: Sync + Send>(&mut self, f: F)
    where
        F: Fn(&A) -> A,
    {
        match self {
            Self::Scalar(val) => *val = f(&val),
            Self::Vector(val) => val.par_map_inplace(|val| *val = f(&val)),
            Self::Matrix(val) => val.par_map_inplace(|val| *val = f(&val)),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ForwardAction {
    Evaluate,
    Cached,
}

#[derive(Debug, PartialEq)]
pub enum BackwardAction {
    Set,
    Increment,
}

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
        debug_assert!(self.recurse_backward(), "Not fully backpropagated.");
        self.forward_count.get() == 0
    }

    pub fn recurse_backward(&self) -> bool {
        let backward_count = self.backward_count.get();
        let forward_count = self.forward_count.get();

        assert!(backward_count <= forward_count);

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

// Implements fast, parallelized and simd-compatible arithmetic operations
// between scalars and other DataRepr values. It mimics ndarray's arithmetic
// operations, as a consequence, the macro implemets two functions to ensure
// commutativity.
//
// Types: Scalar <op> Scalar -> Scalar
//        Scalar <op> Vector -> Vector
//        Scalar <op> Matrix -> Matrix
//        Vector <op> Scalar -> Vector
//        Matrix <op> Scalar -> Matrix
macro_rules! impl_ops_scal {
    ($fun_:ident, $_fun:ident, $type:ty, $op:tt) => {
        fn $fun_<
            A: Copy
                + Zero
                + One
                + Neg
                + Send
                + Sync
                + Add<Output = A>
                + AddAssign
                + Div<Output = A>
                + Mul<Output = A>
                + Sub<Output = A>
        >(  res: &mut $type,
            lhs: &A,
            rhs: &$type,
        ) {
            Zip::from(res)
                .and(rhs)
                .par_apply(|res, rhs| *res = *lhs $op *rhs);
        }
        fn $_fun<
            A: Copy
                + Zero
                + One
                + Neg
                + Send
                + Sync
                + Add<Output = A>
                + AddAssign
                + Div<Output = A>
                + Mul<Output = A>
                + Sub<Output = A>
        >(  res: &mut $type,
            lhs: &$type,
            rhs: &A
        ) {
            Zip::from(res)
                .and(lhs)
                .par_apply(|res, lhs| *res = *lhs $op *rhs);
        }
    };
}

// Scalar-vector and vector-scalar addition.
impl_ops_scal!(add_sv, add_vs, Vector::<A>, +);
// Scalar-vector and vector-scalar subtraction.
impl_ops_scal!(sub_sv, sub_vs, Vector::<A>, -);
// Scalar-vector and vector-scalar multiplication.
impl_ops_scal!(mul_sv, mul_vs, Vector::<A>, *);
// Scalar-vector and vector-scalar division.
impl_ops_scal!(div_sv, div_vs, Vector::<A>, /);

// Scalar-matrix and matrix-scalar addition.
impl_ops_scal!(add_sm, add_ms, Matrix::<A>, +);
// Scalar-matrix and matrix-scalar subtraction.
impl_ops_scal!(sub_sm, sub_ms, Matrix::<A>, -);
// Scalar-matrix and matrix-scalar multiplication.
impl_ops_scal!(mul_sm, mul_ms, Matrix::<A>, *);
// Scalar-matrix and matrix-scalar division.
impl_ops_scal!(div_sm, div_ms, Matrix::<A>, /);

// Implements fast, parallelized and simd-enabled arithmetic operations
// between non scalar DataRepr values. It mimics ndarray's arithmetic
// operations including shape broadcasting when needed. If two shapes
// are not broadcastable together ndarray's broadcasting error is returned.
//
// Types: the result's type in determined by the left hand side operand.
macro_rules! impl_ops {
    ($fun:ident, $lhs_type:ty, $rhs_type:ty, $op:tt) => {
        fn $fun<
            A: Copy
                + Zero
                + One
                + Neg
                + Send
                + Sync
                + Add<Output = A>
                + Div<Output = A>
                + Mul<Output = A>
                + Sub<Output = A>
        > (
            res: &mut $lhs_type,
            lhs: &$lhs_type,
            rhs: &$rhs_type
        ) {
            Zip::from(res)
                .and(lhs)
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
impl_ops!(add_vv, Vector<A>, Vector<A>, +);
// Vector-vector subtraction.
impl_ops!(sub_vv, Vector<A>, Vector<A>, -);
// Vector-vector multiplication.
impl_ops!(mul_vv, Vector<A>, Vector<A>, *);
// Vector-vector division.
impl_ops!(div_vv, Vector<A>, Vector<A>, /);

// N.B. this functions will all result in a ndarray's
// error as a matrix cannot be broadcasted to a vector.
// They act as graceful points of failure.
//
// Vector-matrix addition.
impl_ops!(add_vm, Vector<A>, Matrix<A>, +);
// Vector-matrix subtraction.
impl_ops!(sub_vm, Vector<A>, Matrix<A>, -);
// Vector-matrix multiplication.
impl_ops!(mul_vm, Vector<A>, Matrix<A>, *);
// Vector-matrix division.
impl_ops!(div_vm, Vector<A>, Matrix<A>, /);

// Broadcasting will only take place if the matrix has the
// same number of columns as the vector's elements.
//
// Matrix-vector addition.
impl_ops!(add_mv, Matrix<A>, Vector<A>, +);
// Matrix-vector subtraction.
impl_ops!(sub_mv, Matrix<A>, Vector<A>, -);
// Matrix-vector element-wise multiplication.
impl_ops!(mul_mv, Matrix<A>, Vector<A>, *);
// Matrix-vector division.
impl_ops!(div_mv, Matrix<A>, Vector<A>, /);

// In order for the broadcasting to take place one of the two
// following things must happen: either one of the two matrix
// must be a singleton or, if it's not, it must have the same
// number of columns of the other one but exactly one row.
//
// Matrix-matrix addition.
impl_ops!(add_mm, Matrix<A>, Matrix<A>, +);
// Matrix-matrix subtraction.
impl_ops!(sub_mm, Matrix<A>, Matrix<A>, -);
// Matrix-matrix element-wise multiplication.
impl_ops!(mul_mm, Matrix<A>, Matrix<A>, *);
// Matrix-matrix division.
impl_ops!(div_mm, Matrix<A>, Matrix<A>, /);

// Implements the Add, Sub, Mul and Div traits using the previously
// defined ops.
//
// The +, -, *, / operations all create a new DataRepr struct
// with its value determined by the operands' ones.
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
        impl<'a, A> $trait<&'a DataRepr<A>> for &'a DataRepr<A>
        where
            A: Copy
            + Zero
            + One
            + Neg
            + Send
            + Sync
            + Add<Output=A>
            + AddAssign
            + Div<Output=A>
            + Mul<Output=A>
            + Sub<Output=A>
        {
            type Output = DataRepr<A>;

            fn $fun(self, rhs: Self) -> DataRepr<A> {
                match self {
                    DataRepr::Scalar(lhs_val) => match rhs {
                        DataRepr::Scalar(rhs_val) => DataRepr::Scalar(*lhs_val $op *rhs_val),
                        DataRepr::Vector(rhs_val) => {
                            let mut new = Vector::zeros(rhs_val.raw_dim());
                            $sv_op(&mut new, lhs_val, rhs_val);
                            DataRepr::Vector(new)
                            },
                        DataRepr::Matrix(rhs_val) => {
                            let mut new = Matrix::zeros(rhs_val.raw_dim());
                            $sm_op(&mut new, lhs_val, rhs_val);
                            DataRepr::Matrix(new)
                        }
                    },
                    DataRepr::Vector(lhs_val) => match rhs {
                        DataRepr::Scalar(rhs_val) => {
                            let mut new = Vector::zeros(lhs_val.raw_dim());
                            $vs_op(&mut new, lhs_val, rhs_val);
                            DataRepr::Vector(new)
                        },
                        DataRepr::Vector(rhs_val) => {
                            let mut new = Vector::zeros(lhs_val.raw_dim());
                            $vv_op(&mut new, lhs_val, rhs_val);
                            DataRepr::Vector(new)
                        },
                        DataRepr::Matrix(rhs_val) => {
                            let mut new = Vector::zeros(lhs_val.raw_dim());
                            $vm_op(&mut new, lhs_val, rhs_val);
                            DataRepr::Vector(new)
                        },
                    },
                    DataRepr::Matrix(lhs_val) => match rhs {
                        DataRepr::Scalar(rhs_val) => {
                            let mut new = Matrix::zeros(lhs_val.raw_dim());
                            $ms_op(&mut new, lhs_val, rhs_val);
                            DataRepr::Matrix(new)
                        },
                        DataRepr::Vector(rhs_val) => {
                            let mut new = Matrix::zeros(lhs_val.raw_dim());
                            $mv_op(&mut new, lhs_val, rhs_val);
                            DataRepr::Matrix(new)
                        },
                        DataRepr::Matrix(rhs_val) => {
                            let mut new = Matrix::zeros(lhs_val.raw_dim());
                            $mm_op(&mut new, lhs_val, rhs_val);
                            DataRepr::Matrix(new)
                        },
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
impl<A: Copy + Neg<Output = A> + Zero + One + Send + Sync> Neg for &DataRepr<A> {
    type Output = DataRepr<A>;
    fn neg(self) -> DataRepr<A> {
        self.map(|el| *el * A::one().neg())
    }
}

// Implements the arithmetic operation used during the forward
// pass in the computational graph.
macro_rules! impl_forward_arithmetic_ops {
    ($fun:ident,
        $op:tt,
        $sv_op:ident,
        $vs_op:ident,
        $vv_op:ident,
        $sm_op:ident,
        $ms_op:ident,
        $mv_op:ident,
        $mm_op:ident
    ) => {
        pub(super) fn $fun<A>(trgt: &mut DataRepr<A>, lhs: &DataRepr<A>, rhs: &DataRepr<A>)
        where
            A: Copy
                + Zero
                + One
                + Neg
                + Send
                + Sync
                + Add<Output = A>
                + AddAssign
                + Div<Output = A>
                + Mul<Output = A>
                + Sub<Output = A>
        {
            match trgt {
                // We already know the operands' types.
                DataRepr::Scalar(_) => {
                    *trgt = DataRepr::Scalar(rhs.scalar() $op lhs.scalar());
                }
                // We already know the operands' types.
                DataRepr::Vector(trgt_val) => {
                    if let DataRepr::Scalar(lhs_val) = lhs {
                        $sv_op(trgt_val, lhs_val, rhs.vector());
                    } else if let DataRepr::Scalar(rhs_val) = rhs {
                        $vs_op(trgt_val, lhs.vector(), &rhs_val);
                    } else {
                        $vv_op(trgt_val, lhs.vector(), rhs.vector())
                    }
                }
                // We already know the operands' types.
                DataRepr::Matrix(trgt_val) => {
                    if let DataRepr::Scalar(lhs_val) = lhs {
                        $sm_op(trgt_val, lhs_val, rhs.matrix());
                    } else if let DataRepr::Scalar(rhs_val) = rhs {
                        $ms_op(trgt_val, lhs.matrix(), &rhs_val);
                    } else if let DataRepr::Vector(rhs_val) = rhs {
                        $mv_op(trgt_val, lhs.matrix(), rhs_val);
                    } else {
                        $mm_op(trgt_val, lhs.matrix(), rhs.matrix())
                    };
                }
            }
        }
    };
}

// Implements the add operation used in forward passes.
impl_forward_arithmetic_ops!(add, +, add_sv, add_vs, add_vv, add_sm, add_ms, add_mv, add_mm);
// Implements the sub operation used in forward passes.
impl_forward_arithmetic_ops!(sub, -, sub_sv, sub_vs, sub_vv, sub_sm, sub_ms, sub_mv, sub_mm);
// Implements the mul operation used in forward passes.
impl_forward_arithmetic_ops!(mul, *, mul_sv, mul_vs, mul_vv, mul_sm, mul_ms, mul_mv, mul_mm);
// Implements the div operation used in forward passes.
impl_forward_arithmetic_ops!(div, /, div_sv, div_vs, div_vv, div_sm, div_ms, div_mv, div_mm);

// Implements the gradient accumulation functions.
macro_rules! impl_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun<A> (
            trgt: &mut DataRepr<A>,
            src: &DataRepr<A>,
        )
        where
            A:Copy
                + Add<Output = A>
                + AddAssign
                + Sub<Output = A>
                + SubAssign
                + Mul<Output=A>
                + MulAssign
                + DivAssign
                + Zero
                + One
                + Send
                + Sync
        {
            match trgt {
                DataRepr::Scalar(trgt_val) => match src {
                    DataRepr::Scalar(src_val) => {
                        *trgt_val $op *src_val;
                    },
                    DataRepr::Vector(src_val) => {
                        *trgt_val $op src_val.sum()
                    }
                    DataRepr::Matrix(src_val) => {
                        *trgt_val $op src_val.sum()
                    }
                },
                DataRepr::Vector(trgt_val) => match src {
                    DataRepr::Scalar(src_val) => {
                        Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op *src_val)
                    }
                    DataRepr::Vector(src_val) => {
                        if trgt_val.dim() >= src_val.dim() {
                            Zip::from(trgt_val)
                                .and_broadcast(src_val)
                                .par_apply(|trgt_val, src_val| *trgt_val $op *src_val);
                        } else {
                            Zip::from(trgt_val)
                                .and_broadcast(&src_val.sum_axis(Axis(0)))
                                .par_apply(|trgt_val, src_val| *trgt_val $op *src_val);
                        }
                    }
                    DataRepr::Matrix(src_val) => {
                        let reduced_src = src_val.sum_axis(Axis(0));
                        if trgt_val.dim() < reduced_src.dim() {
                            Zip::from(trgt_val)
                                .and_broadcast(&reduced_src.sum_axis(Axis(0)))
                                .apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src);
                        } else {
                            Zip::from(trgt_val)
                                .and(&reduced_src)
                                .par_apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src);
                        }
                    }
                },
                DataRepr::Matrix(trgt_val) => match src {
                    DataRepr::Scalar(src_val) => {
                        Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op *src_val)
                    }
                    DataRepr::Vector(src_val) => Zip::from(trgt_val)
                        .and_broadcast(src_val)
                        .apply(|trgt_val, src_val| *trgt_val $op *src_val),
                    DataRepr::Matrix(src_val) => Zip::from(trgt_val)
                        .and_broadcast(src_val)
                        .par_apply(|trgt_val, src_val| *trgt_val $op *src_val),
                },
            }
        }
    };
}

// Implements the gradient accumulation functions; the src
// is passed by value.
macro_rules! impl_accumulation_ops_v {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun<A> (
            trgt: &mut DataRepr<A>,
            src: DataRepr<A>,
        )
        where
            A:Copy
                + Add<Output = A>
                + AddAssign
                + Sub<Output = A>
                + SubAssign
                + Mul<Output = A>
                + MulAssign
                + DivAssign
                + Zero
                + One
                + Send
                + Sync
        {
            match trgt {
                DataRepr::Scalar(trgt_val) => match src {
                    DataRepr::Scalar(src_val) => {
                        *trgt_val $op src_val;
                    },
                    DataRepr::Vector(src_val) => {
                        *trgt_val $op src_val.sum()
                    }
                    DataRepr::Matrix(src_val) => {
                        *trgt_val $op src_val.sum()
                    }
                },
                DataRepr::Vector(trgt_val) => match src {
                    DataRepr::Scalar(src_val) => {
                        Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op src_val)
                    }
                    DataRepr::Vector(src_val) => {
                        if trgt_val.dim() >= src_val.dim() {
                            Zip::from(trgt_val)
                                .and_broadcast(&src_val)
                                .par_apply(|trgt_val, src_val| *trgt_val $op *src_val);
                        } else {
                            Zip::from(trgt_val)
                                .and_broadcast(&src_val.sum_axis(Axis(0)))
                                .par_apply(|trgt_val, src_val| *trgt_val $op *src_val);
                        }
                    }
                    DataRepr::Matrix(src_val) => {
                        let reduced_src = src_val.sum_axis(Axis(0));
                        if trgt_val.dim() < reduced_src.dim() {
                            Zip::from(trgt_val)
                                .and_broadcast(&reduced_src.sum_axis(Axis(0)))
                                .apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src);
                        } else {
                            Zip::from(trgt_val)
                                .and(&reduced_src)
                                .par_apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src);
                        }
                    }
                },
                DataRepr::Matrix(trgt_val) => match src {
                    DataRepr::Scalar(src_val) => {
                        Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op src_val)
                    }
                    DataRepr::Vector(src_val) => Zip::from(trgt_val)
                        .and_broadcast(&src_val)
                        .apply(|trgt_val, src_val| *trgt_val $op *src_val),
                    DataRepr::Matrix(src_val) => Zip::from(trgt_val)
                        .and_broadcast(&src_val)
                        .par_apply(|trgt_val, src_val| *trgt_val $op *src_val),
                },
            }
        }
    };
}

// Accumulation assignment.
impl_accumulation_ops!(assign, =);
// Accumulation assignment by value.
impl_accumulation_ops_v!(assign_v, =);
// Accumulation add-assignemnt.
impl_accumulation_ops!(add_assign, +=);
// Accumulation add-assignemnt by value.
impl_accumulation_ops_v!(add_assign_v, +=);
// Accumulation sub assignment.
impl_accumulation_ops!(sub_assign, -=);

// Used in the computation of the first order derivative
// of the division operation.
pub(super) fn div_assign_pow<A>(trgt: &mut DataRepr<A>, src: &DataRepr<A>, exp: usize)
where
    A: Copy
        + Add<Output = A>
        + AddAssign
        + Sub<Output = A>
        + SubAssign
        + Mul<Output = A>
        + MulAssign
        + DivAssign
        + Zero
        + One
        + Send
        + Sync,
{
    match trgt {
        DataRepr::Scalar(trgt_val) => match src {
            DataRepr::Scalar(src_val) => {
                *trgt_val /= pow(*src_val, exp);
            }
            DataRepr::Vector(src_val) => *trgt_val /= src_val.mapv(|x| pow(x, exp)).sum(),
            DataRepr::Matrix(src_val) => *trgt_val /= src_val.mapv(|x| pow(x, exp)).sum(),
        },
        DataRepr::Vector(trgt_val) => match src {
            DataRepr::Scalar(src_val) => {
                Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val /= pow(*src_val, exp))
            }
            DataRepr::Vector(src_val) => {
                if trgt_val.dim() >= src_val.dim() {
                    Zip::from(trgt_val)
                        .and_broadcast(src_val)
                        .par_apply(|trgt_val, src_val| *trgt_val /= pow(*src_val, exp));
                } else {
                    Zip::from(trgt_val)
                        .and_broadcast(&src_val.mapv(|x| pow(x, exp)).sum_axis(Axis(0)))
                        .par_apply(|trgt_val, src_val| *trgt_val /= *src_val);
                }
            }
            DataRepr::Matrix(src_val) => {
                let reduced_src = src_val.mapv(|x| pow(x, exp)).sum_axis(Axis(0));
                if trgt_val.dim() < reduced_src.dim() {
                    Zip::from(trgt_val)
                        .and_broadcast(&reduced_src.sum_axis(Axis(0)))
                        .apply(|trgt_val, reduced_src| *trgt_val /= *reduced_src);
                } else {
                    Zip::from(trgt_val)
                        .and(&reduced_src)
                        .par_apply(|trgt_val, reduced_src| *trgt_val /= *reduced_src);
                }
            }
        },
        DataRepr::Matrix(trgt_val) => match src {
            DataRepr::Scalar(src_val) => {
                Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val /= pow(*src_val, exp))
            }
            DataRepr::Vector(src_val) => Zip::from(trgt_val)
                .and_broadcast(src_val)
                .apply(|trgt_val, src_val| *trgt_val /= pow(*src_val, exp)),
            DataRepr::Matrix(src_val) => Zip::from(trgt_val)
                .and_broadcast(src_val)
                .par_apply(|trgt_val, src_val| *trgt_val /= pow(*src_val, exp)),
        },
    }
}

// Used in the accumulation of the gradient of the pow op.
// Implements the necessary accumulation operations.
// Computes: d/dx x^n = n * x^(n-1).
macro_rules! impl_pow_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun<A>(
            grad: &mut DataRepr<A>,
            downstream_grad: &DataRepr<A>,
            data: &DataRepr<A>,
            exp: u16,
        ) where
            A: Copy
                + Zero
                + TryFrom<u16>
                + One
                + Add<Output = A>
                + AddAssign
                + Mul<Output = A>
                + Send
                + Sync,
        {
            match (grad, downstream_grad, data) {
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1 ) * A::try_from(exp).ok().unwrap(),
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op down_grad_val.sum() * pow(*data_val, exp as usize - 1 ) * A::try_from(exp).ok().unwrap(),
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op down_grad_val.sum() * pow(*data_val, exp as usize - 1 ) * A::try_from(exp).ok().unwrap(),
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, data_val| {
                            *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1) * A::try_from(exp).ok().unwrap()
                    }),
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and_broadcast(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1) * A::try_from(exp).ok().unwrap()
                    }),
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and_broadcast(&down_grad_val.sum_axis(Axis(0)))
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1) * A::try_from(exp).ok().unwrap()
                        }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, data_val| {
                            *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1) * A::try_from(exp).ok().unwrap()
                        }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and_broadcast(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1) * A::try_from(exp).ok().unwrap()
                    }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and_broadcast(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op *down_grad_val * pow(*data_val, exp as usize - 1) * A::try_from(exp).ok().unwrap()
                    }),
                _ => panic!("erorr: gradient and data should have the same size."),
            }
        }
    };
}

impl_pow_accumulation_ops!(pow_diff_assign, =);
impl_pow_accumulation_ops!(pow_diff_add_assign, +=);

pub(super) fn relu_forward<A>(data: &mut DataRepr<A>, src: &DataRepr<A>)
where
    A: Sync + Send + PartialOrd + Copy + Zero + One,
{
    match (data, src) {
        (DataRepr::Scalar(data_val), DataRepr::Scalar(src_val)) => {
            *data_val = if *src_val < A::zero() {
                A::zero()
            } else {
                *src_val
            }
        }
        (DataRepr::Vector(data_val), DataRepr::Vector(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| {
                *data_el = if *src_el < A::zero() {
                    A::zero()
                } else {
                    *src_el
                }
            }),
        (DataRepr::Matrix(data_val), DataRepr::Matrix(src_val)) => Zip::from(data_val)
            .and(src_val)
            .par_apply(|data_el, src_el| {
                *data_el = if *src_el < A::zero() {
                    A::zero()
                } else {
                    *src_el
                }
            }),
        _ => panic!("error: the two operands should have the same size."),
    }
}

// Used in the accumulation of the gradient of the ReLU op.
// Implements the necessary accumulation operations.
macro_rules! impl_relu_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun<A>(
            grad: &mut DataRepr<A>,
            downstream_grad: &DataRepr<A>,
            data: &DataRepr<A>,
        ) where
            A: Copy
                + Zero
                + One
                + Add<Output = A>
                + AddAssign
                + Mul<Output = A>
                + Send
                + Sync
                + PartialOrd,
        {
            match (grad, downstream_grad, data) {
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op if *data_val > A::zero() { *down_grad_val } else { A::zero() },
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => {
                    *grad_val $op if *data_val > A::zero() { down_grad_val.sum() } else { A::zero() }
                }
                (
                    DataRepr::Scalar(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Scalar(data_val),
                ) => *grad_val $op if *data_val > A::zero() { down_grad_val.sum() } else { A::zero() },
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, data_val| {
                            *grad_val $op if *data_val > A::zero() { *down_grad_val } else { A::zero() }
                    }),
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and_broadcast(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op if *data_val > A::zero() { *down_grad_val } else { A::zero() }
                    }),
                (
                    DataRepr::Vector(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Vector(data_val),
                ) => Zip::from(grad_val)
                        .and_broadcast(&down_grad_val.sum_axis(Axis(0)))
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op if *data_val > A::zero() { *down_grad_val } else { A::zero() }
                        }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Scalar(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, data_val| {
                            *grad_val $op if *data_val > A::zero() { *down_grad_val } else { A::zero() }
                        }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Vector(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and_broadcast(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op if *data_val > A::zero() { *down_grad_val } else { A::zero() }
                    }),
                (
                    DataRepr::Matrix(grad_val),
                    DataRepr::Matrix(down_grad_val),
                    DataRepr::Matrix(data_val),
                ) => Zip::from(grad_val)
                        .and_broadcast(down_grad_val)
                        .and(data_val)
                        .par_apply(|grad_val, down_grad_val, data_val| {
                            *grad_val $op if *data_val > A::zero() { *down_grad_val } else { A::zero() }
                    }),
                _ => panic!("erorr: gradient and data should have the same size."),
            }
        }
    };
}

impl_relu_accumulation_ops!(relu_diff_assign, =);
impl_relu_accumulation_ops!(relu_diff_add_assign, +=);

// Implements scalar-scaled accumulation operations.
macro_rules! impl_scaled_accumulation_ops {
    ($fun:ident, $op:tt) => {
        pub(super) fn $fun<A>(
            trgt: &mut DataRepr<A>,
            src: &DataRepr<A>,
            scalar:A
        )
        where
            A: Copy
                + Add<Output = A>
                + AddAssign
                + Sub<Output = A>
                + SubAssign
                + Mul<Output=A>
                + Zero
                + One
                + Send
                + Sync
        {
            match trgt {
                DataRepr::Scalar(trgt_val) => match src {
                    DataRepr::Scalar(src_val) => {
                        *trgt_val $op *src_val * scalar;
                    },
                    DataRepr::Vector(src_val) => {
                        *trgt_val $op src_val.sum() * scalar
                    }
                    DataRepr::Matrix(src_val) => {
                        *trgt_val $op src_val.sum() * scalar
                    }
                },
                DataRepr::Vector(trgt_val) => match src {
                    DataRepr::Scalar(src_val) => {
                        Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op *src_val * scalar)
                    }
                    DataRepr::Vector(src_val) => {
                        if trgt_val.dim() >= src_val.dim() {
                            Zip::from(trgt_val)
                                .and_broadcast(src_val)
                                .par_apply(|trgt_val, src_val| *trgt_val $op *src_val * scalar);
                        } else {
                            Zip::from(trgt_val)
                                .and_broadcast(&src_val.sum_axis(Axis(0)))
                                .par_apply(|trgt_val, src_val| *trgt_val $op *src_val * scalar);
                        }
                    }
                    DataRepr::Matrix(src_val) => {
                        let reduced_src = src_val.sum_axis(Axis(0));
                        if trgt_val.dim() < reduced_src.dim() {
                            Zip::from(trgt_val)
                                .and_broadcast(&reduced_src.sum_axis(Axis(0)))
                                .apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src * scalar);
                        } else {
                            Zip::from(trgt_val)
                                .and(&reduced_src)
                                .par_apply(|trgt_val, reduced_src| *trgt_val $op *reduced_src * scalar);
                        }
                    }
                },
                DataRepr::Matrix(trgt_val) => match src {
                    DataRepr::Scalar(src_val) => {
                        Zip::from(trgt_val).par_apply(|trgt_val| *trgt_val $op *src_val * scalar)
                    }
                    DataRepr::Vector(src_val) => Zip::from(trgt_val)
                        .and_broadcast(src_val)
                        .apply(|trgt_val, src_val| *trgt_val $op *src_val * scalar),
                    DataRepr::Matrix(src_val) => Zip::from(trgt_val)
                        .and(src_val)
                        .par_apply(|trgt_val, src_val| *trgt_val $op *src_val * scalar),
                },
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
pub(super) fn mat_mat_mul<
    A: 'static
        + Copy
        + Send
        + Sync
        + Zero
        + One
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>,
>(
    trgt: &mut DataRepr<A>,
    alpha: A,
    lhs: &DataRepr<A>,
    rhs: &DataRepr<A>,
    beta: A,
    t_lhs: bool,
    t_rhs: bool,
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
pub(super) fn mat_vec_mul<
    A: 'static
        + Copy
        + Send
        + Sync
        + Zero
        + One
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>,
>(
    trgt: &mut DataRepr<A>,
    alpha: A,
    lhs: &DataRepr<A>,
    rhs: &DataRepr<A>,
    beta: A,
) {
    general_mat_vec_mul(
        alpha,
        lhs.matrix(),
        rhs.vector(),
        beta,
        &mut trgt.vector_mut(),
    );
}

#[cfg(test)]
mod tests;
