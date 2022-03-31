use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{linalg::general_mat_mul, Ix2};
use std::rc::Rc;

pub struct MatrixMatrixMulT {
    left_data: SharedTensor<Ix2>,
    right_data: SharedTensor<Ix2>,
    data: SharedTensor<Ix2>,
}

impl MatrixMatrixMulT {
    pub fn new(
        left_data: SharedTensor<Ix2>,
        right_data: SharedTensor<Ix2>,
        data: SharedTensor<Ix2>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
        }
    }
}

impl Forward for MatrixMatrixMulT {
    fn forward(&self) {
        general_mat_mul(
            1.,
            &*self.left_data.borrow(),
            &self.right_data.borrow().t(),
            0.,
            &mut *self.data.borrow_mut(),
        );
    }
}

pub struct MatrixMatrixMulTBackwardLeft {
    left_gradient: Rc<SwitchableTensor<Ix2>>,
    right_data: SharedTensor<Ix2>,
    gradient: Rc<SwitchableTensor<Ix2>>,
}

impl MatrixMatrixMulTBackwardLeft {
    pub fn new(
        left_gradient: Rc<SwitchableTensor<Ix2>>,
        right_data: SharedTensor<Ix2>,
        gradient: Rc<SwitchableTensor<Ix2>>,
    ) -> Self {
        Self {
            left_gradient,
            right_data,
            gradient,
        }
    }
}

impl Backward for MatrixMatrixMulTBackwardLeft {
    fn backward(&self) {
        general_mat_mul(
            1.,
            &*self.gradient.array(),
            &self.right_data.borrow(),
            1.,
            &mut *self.left_gradient.array_mut(),
        );
    }
}

pub struct MatrixMatrixMulTBackwardRight {
    left_data: SharedTensor<Ix2>,
    right_gradient: Rc<SwitchableTensor<Ix2>>,
    gradient: Rc<SwitchableTensor<Ix2>>,
}

impl MatrixMatrixMulTBackwardRight {
    pub fn new(
        left_data: SharedTensor<Ix2>,
        right_gradient: Rc<SwitchableTensor<Ix2>>,
        gradient: Rc<SwitchableTensor<Ix2>>,
    ) -> Self {
        Self {
            left_data,
            right_gradient,
            gradient,
        }
    }
}

impl Backward for MatrixMatrixMulTBackwardRight {
    fn backward(&self) {
        general_mat_mul(
            1.,
            &self.gradient.array().t(),
            &self.left_data.borrow(),
            1.,
            &mut *self.right_gradient.array_mut(),
        )
    }
}

pub struct MatrixMatrixMulTBackward {
    left: MatrixMatrixMulTBackwardLeft,
    right: MatrixMatrixMulTBackwardRight,
}

impl MatrixMatrixMulTBackward {
    pub fn new(left: MatrixMatrixMulTBackwardLeft, right: MatrixMatrixMulTBackwardRight) -> Self {
        Self { left, right }
    }
}

impl Backward for MatrixMatrixMulTBackward {
    fn backward(&self) {
        self.left.backward();
        self.right.backward();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
