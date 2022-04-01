use super::{Backward, Forward, Gradient, Shared};
use ndarray::{linalg::general_mat_mul, Array2, Ix2};
use std::rc::Rc;

pub(crate) struct MatrixMatrixMul {
    left_data: Shared<Array2<f32>>,
    right_data: Shared<Array2<f32>>,
    data: Shared<Array2<f32>>,
}

impl MatrixMatrixMul {
    pub(crate) fn new(
        left_data: Shared<Array2<f32>>,
        right_data: Shared<Array2<f32>>,
        data: Shared<Array2<f32>>,
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

pub(crate) struct MatrixMatrixMulBackwardLeft {
    left_gradient: Rc<Gradient<Ix2>>,
    right_data: Shared<Array2<f32>>,
    gradient: Rc<Gradient<Ix2>>,
}

impl MatrixMatrixMulBackwardLeft {
    pub(crate) fn new(
        left_gradient: Rc<Gradient<Ix2>>,
        right_data: Shared<Array2<f32>>,
        gradient: Rc<Gradient<Ix2>>,
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
            &*self.gradient.borrow(),
            &self.right_data.borrow().t(),
            1.,
            &mut *self.left_gradient.borrow_mut(),
        );
    }
}

pub(crate) struct MatrixMatrixMulBackwardRight {
    left_data: Shared<Array2<f32>>,
    right_gradient: Rc<Gradient<Ix2>>,
    gradient: Rc<Gradient<Ix2>>,
}

impl MatrixMatrixMulBackwardRight {
    pub(crate) fn new(
        left_data: Shared<Array2<f32>>,
        right_gradient: Rc<Gradient<Ix2>>,
        gradient: Rc<Gradient<Ix2>>,
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
            &*self.gradient.borrow(),
            1.,
            &mut *self.right_gradient.borrow_mut(),
        )
    }
}

pub(crate) struct MatrixMatrixMulBackward {
    left: MatrixMatrixMulBackwardLeft,
    right: MatrixMatrixMulBackwardRight,
}

impl MatrixMatrixMulBackward {
    pub(crate) fn new(
        left: MatrixMatrixMulBackwardLeft,
        right: MatrixMatrixMulBackwardRight,
    ) -> Self {
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