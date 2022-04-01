use super::{Backward, Forward, Gradient, Shared};
use ndarray::{linalg::general_mat_mul, Array2, Ix2};
use std::rc::Rc;

pub(crate) struct MatrixMatrixMulT {
    left_data: Shared<Array2<f32>>,
    right_data: Shared<Array2<f32>>,
    data: Shared<Array2<f32>>,
}

impl MatrixMatrixMulT {
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

pub(crate) struct MatrixMatrixMulTBackwardLeft {
    left_gradient: Rc<Gradient<Ix2>>,
    right_data: Shared<Array2<f32>>,
    gradient: Rc<Gradient<Ix2>>,
}

impl MatrixMatrixMulTBackwardLeft {
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

impl Backward for MatrixMatrixMulTBackwardLeft {
    fn backward(&self) {
        general_mat_mul(
            1.,
            &*self.gradient.borrow(),
            &self.right_data.borrow(),
            1.,
            &mut *self.left_gradient.borrow_mut(),
        );
    }
}

pub(crate) struct MatrixMatrixMulTBackwardRight {
    left_data: Shared<Array2<f32>>,
    right_gradient: Rc<Gradient<Ix2>>,
    gradient: Rc<Gradient<Ix2>>,
}

impl MatrixMatrixMulTBackwardRight {
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

impl Backward for MatrixMatrixMulTBackwardRight {
    fn backward(&self) {
        general_mat_mul(
            1.,
            &self.gradient.borrow().t(),
            &self.left_data.borrow(),
            1.,
            &mut *self.right_gradient.borrow_mut(),
        )
    }
}

pub(crate) struct MatrixMatrixMulTBackward {
    left: MatrixMatrixMulTBackwardLeft,
    right: MatrixMatrixMulTBackwardRight,
}

impl MatrixMatrixMulTBackward {
    pub(crate) fn new(
        left: MatrixMatrixMulTBackwardLeft,
        right: MatrixMatrixMulTBackwardRight,
    ) -> Self {
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
