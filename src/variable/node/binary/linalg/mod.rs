mod matrix_matrix_mul;
mod matrix_matrix_mul_t;
mod matrix_vector_mul;
mod vector_matrix_mul;
mod vector_vector_mul;

use super::{expect_tensor, expect_tensor_mut, Backward, Forward, OptionalTensor, Tensor};

#[cfg(test)]
use super::{assert_almost_equals, new_tensor};

pub(crate) use matrix_matrix_mul::{
    MatrixMatrixMul, MatrixMatrixMulBackward, MatrixMatrixMulBackwardLeft,
    MatrixMatrixMulBackwardRight,
};

pub(crate) use matrix_matrix_mul_t::{
    MatrixMatrixMulT, MatrixMatrixMulTBackward, MatrixMatrixMulTBackwardLeft,
    MatrixMatrixMulTBackwardRight,
};
pub(crate) use matrix_vector_mul::{
    MatrixVectorMul, MatrixVectorMulBackward, MatrixVectorMulBackwardLeft,
    MatrixVectorMulBackwardRight,
};
pub(crate) use vector_matrix_mul::{
    VectorMatrixMul, VectorMatrixMulBackward, VectorMatrixMulBackwardLeft,
    VectorMatrixMulBackwardRight,
};
pub(crate) use vector_vector_mul::{
    VectorVectorMul, VectorVectorMulBackward, VectorVectorMulBackwardUnary,
};
