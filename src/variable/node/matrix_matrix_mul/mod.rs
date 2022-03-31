use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{linalg::general_mat_mul, Ix2};
use std::rc::Rc;

pub struct MatrixMatrixMul {
    left_data: SharedTensor<Ix2>,
    right_data: SharedTensor<Ix2>,
    data: SharedTensor<Ix2>,
}

impl MatrixMatrixMul {
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

impl Forward for MatrixMatrixMul {
    fn forward(&self) {
        general_mat_mul(
            1.,
            &*self.left_data.borrow(),
            &*self.right_data.borrow(),
            0.,
            &mut *self.data.borrow_mut(),
        );
    }
}

pub struct MatrixMatrixMulBackwardLeft {
    left_gradient: Rc<SwitchableTensor<Ix2>>,
    right_data: SharedTensor<Ix2>,
    gradient: Rc<SwitchableTensor<Ix2>>,
}

impl MatrixMatrixMulBackwardLeft {
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

impl Backward for MatrixMatrixMulBackwardLeft {
    fn backward(&self) {
        general_mat_mul(
            1.,
            &*self.gradient.array(),
            &self.right_data.borrow().t(),
            1.,
            &mut *self.left_gradient.array_mut(),
        );
    }
}

pub struct MatrixMatrixMulBackwardRight {
    left_data: SharedTensor<Ix2>,
    right_gradient: Rc<SwitchableTensor<Ix2>>,
    gradient: Rc<SwitchableTensor<Ix2>>,
}

impl MatrixMatrixMulBackwardRight {
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

impl Backward for MatrixMatrixMulBackwardRight {
    fn backward(&self) {
        general_mat_mul(
            1.,
            &self.left_data.borrow().t(),
            &*self.gradient.array(),
            1.,
            &mut *self.right_gradient.array_mut(),
        )
    }
}

pub struct MatrixMatrixMulBackward {
    left: MatrixMatrixMulBackwardLeft,
    right: MatrixMatrixMulBackwardRight,
}

impl MatrixMatrixMulBackward {
    pub fn new(left: MatrixMatrixMulBackwardLeft, right: MatrixMatrixMulBackwardRight) -> Self {
        Self { left, right }
    }
}

impl Backward for MatrixMatrixMulBackward {
    fn backward(&self) {
        self.left.backward();
        self.right.backward();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
