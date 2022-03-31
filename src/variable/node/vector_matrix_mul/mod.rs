use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{linalg::general_mat_vec_mul, s, Ix1, Ix2, NewAxis, Zip};
use std::rc::Rc;

pub struct VectorMatrixMul {
    left_data: SharedTensor<Ix1>,
    right_data: SharedTensor<Ix2>,
    data: SharedTensor<Ix1>,
}

impl VectorMatrixMul {
    pub fn new(
        left_data: SharedTensor<Ix1>,
        right_data: SharedTensor<Ix2>,
        data: SharedTensor<Ix1>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
        }
    }
}

impl Forward for VectorMatrixMul {
    fn forward(&self) {
        general_mat_vec_mul(
            1.,
            &self.right_data.borrow().t(),
            &*self.left_data.borrow(),
            0.,
            &mut *self.data.borrow_mut(),
        );
    }
}

pub struct VectorMatrixMulBackwardLeft {
    left_gradient: Rc<SwitchableTensor<Ix1>>,
    right_data: SharedTensor<Ix2>,
    gradient: Rc<SwitchableTensor<Ix1>>,
}

impl VectorMatrixMulBackwardLeft {
    pub fn new(
        left_gradient: Rc<SwitchableTensor<Ix1>>,
        right_data: SharedTensor<Ix2>,
        gradient: Rc<SwitchableTensor<Ix1>>,
    ) -> Self {
        Self {
            left_gradient,
            right_data,
            gradient,
        }
    }
}

impl Backward for VectorMatrixMulBackwardLeft {
    fn backward(&self) {
        general_mat_vec_mul(
            1.,
            &self.right_data.borrow(),
            &*self.gradient.array(),
            1.,
            &mut *self.left_gradient.array_mut(),
        );
    }
}

pub struct VectorMatrixMulBackwardRight {
    left_data: SharedTensor<Ix1>,
    right_gradient: Rc<SwitchableTensor<Ix2>>,
    gradient: Rc<SwitchableTensor<Ix1>>,
}

impl VectorMatrixMulBackwardRight {
    pub fn new(
        left_data: SharedTensor<Ix1>,
        right_gradient: Rc<SwitchableTensor<Ix2>>,
        gradient: Rc<SwitchableTensor<Ix1>>,
    ) -> Self {
        Self {
            left_data,
            right_gradient,
            gradient,
        }
    }
}

impl Backward for VectorMatrixMulBackwardRight {
    fn backward(&self) {
        Zip::from(&mut *self.right_gradient.array_mut())
            .and_broadcast(&self.left_data.borrow().slice(s![.., NewAxis]))
            .and_broadcast(&*self.gradient.array())
            .for_each(|d, &f, &s| *d += f * s);
    }
}

pub struct VectorMatrixMulBackward {
    left: VectorMatrixMulBackwardLeft,
    right: VectorMatrixMulBackwardRight,
}

impl VectorMatrixMulBackward {
    pub fn new(left: VectorMatrixMulBackwardLeft, right: VectorMatrixMulBackwardRight) -> Self {
        Self { left, right }
    }
}

impl Backward for VectorMatrixMulBackward {
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
