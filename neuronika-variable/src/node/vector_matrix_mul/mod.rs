use std::rc::Rc;

use ndarray::{linalg::general_mat_vec_mul, s, Array1, Array2, Ix1, Ix2, NewAxis, Zip};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
};

pub(crate) struct VectorMatrixMul {
    left_data: Shared<Array1<f32>>,
    right_data: Shared<Array2<f32>>,
    data: Shared<Array1<f32>>,
}

impl VectorMatrixMul {
    pub(crate) fn new(
        left_data: Shared<Array1<f32>>,
        right_data: Shared<Array2<f32>>,
        data: Shared<Array1<f32>>,
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

pub(crate) struct VectorMatrixMulBackwardLeft {
    left_gradient: Rc<Gradient<Array1<f32>, Ix1>>,
    right_data: Shared<Array2<f32>>,
    gradient: Rc<Gradient<Array1<f32>, Ix1>>,
}

impl VectorMatrixMulBackwardLeft {
    pub(crate) fn new(
        left_gradient: Rc<Gradient<Array1<f32>, Ix1>>,
        right_data: Shared<Array2<f32>>,
        gradient: Rc<Gradient<Array1<f32>, Ix1>>,
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
            &*self.gradient.borrow(),
            1.,
            &mut *self.left_gradient.borrow_mut(),
        );
    }
}

pub(crate) struct VectorMatrixMulBackwardRight {
    left_data: Shared<Array1<f32>>,
    right_gradient: Rc<Gradient<Array2<f32>, Ix2>>,
    gradient: Rc<Gradient<Array1<f32>, Ix1>>,
}

impl VectorMatrixMulBackwardRight {
    pub(crate) fn new(
        left_data: Shared<Array1<f32>>,
        right_gradient: Rc<Gradient<Array2<f32>, Ix2>>,
        gradient: Rc<Gradient<Array1<f32>, Ix1>>,
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
        Zip::from(&mut *self.right_gradient.borrow_mut())
            .and_broadcast(&self.left_data.borrow().slice(s![.., NewAxis]))
            .and_broadcast(&*self.gradient.borrow())
            .for_each(|d, &f, &s| *d += f * s);
    }
}

pub(crate) struct VectorMatrixMulBackward {
    left: VectorMatrixMulBackwardLeft,
    right: VectorMatrixMulBackwardRight,
}

impl VectorMatrixMulBackward {
    pub(crate) fn new(
        left: VectorMatrixMulBackwardLeft,
        right: VectorMatrixMulBackwardRight,
    ) -> Self {
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
