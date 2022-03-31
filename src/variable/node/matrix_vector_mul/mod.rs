use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{linalg::general_mat_vec_mul, s, Ix1, Ix2, NewAxis, Zip};
use std::rc::Rc;

pub struct MatrixVectorMul {
    left_data: SharedTensor<Ix2>,
    right_data: SharedTensor<Ix1>,
    data: SharedTensor<Ix1>,
}

impl MatrixVectorMul {
    pub fn new(
        left_data: SharedTensor<Ix2>,
        right_data: SharedTensor<Ix1>,
        data: SharedTensor<Ix1>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
        }
    }
}

impl Forward for MatrixVectorMul {
    fn forward(&self) {
        general_mat_vec_mul(
            1.,
            &*self.left_data.borrow(),
            &*self.right_data.borrow(),
            0.,
            &mut *self.data.borrow_mut(),
        );
    }
}

pub struct MatrixVectorMulBackwardLeft {
    left_gradient: Rc<SwitchableTensor<Ix2>>,
    right_data: SharedTensor<Ix1>,
    gradient: Rc<SwitchableTensor<Ix1>>,
}

impl MatrixVectorMulBackwardLeft {
    pub fn new(
        left_gradient: Rc<SwitchableTensor<Ix2>>,
        right_data: SharedTensor<Ix1>,
        gradient: Rc<SwitchableTensor<Ix1>>,
    ) -> Self {
        Self {
            left_gradient,
            right_data,
            gradient,
        }
    }
}

impl Backward for MatrixVectorMulBackwardLeft {
    fn backward(&self) {
        Zip::from(&mut *self.left_gradient.array_mut())
            .and_broadcast(&self.gradient.array().slice(s![.., NewAxis]))
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|d, &f, &s| *d += f * s);
    }
}

pub struct MatrixVectorMulBackwardRight {
    left_data: SharedTensor<Ix2>,
    right_gradient: Rc<SwitchableTensor<Ix1>>,
    gradient: Rc<SwitchableTensor<Ix1>>,
}

impl MatrixVectorMulBackwardRight {
    pub fn new(
        left_data: SharedTensor<Ix2>,
        right_gradient: Rc<SwitchableTensor<Ix1>>,
        gradient: Rc<SwitchableTensor<Ix1>>,
    ) -> Self {
        Self {
            left_data,
            right_gradient,
            gradient,
        }
    }
}

impl Backward for MatrixVectorMulBackwardRight {
    fn backward(&self) {
        general_mat_vec_mul(
            1.,
            &self.left_data.borrow().t(),
            &*self.gradient.array(),
            1.,
            &mut *self.right_gradient.array_mut(),
        );
    }
}

pub struct MatrixVectorMulBackward {
    left: MatrixVectorMulBackwardLeft,
    right: MatrixVectorMulBackwardRight,
}

impl MatrixVectorMulBackward {
    pub fn new(left: MatrixVectorMulBackwardLeft, right: MatrixVectorMulBackwardRight) -> Self {
        Self { left, right }
    }
}

impl Backward for MatrixVectorMulBackward {
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
