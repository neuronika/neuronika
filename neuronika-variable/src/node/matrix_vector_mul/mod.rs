use std::rc::Rc;

use ndarray::{linalg::general_mat_vec_mul, s, Array1, Array2, Ix1, Ix2, NewAxis, Zip};

use crate::{gradient::Gradient, utils::Shared};

use super::{Backward, Forward};

pub(crate) struct MatrixVectorMul {
    left_data: Shared<Array2<f32>>,
    right_data: Shared<Array1<f32>>,
    data: Shared<Array1<f32>>,
}

impl MatrixVectorMul {
    pub(crate) fn new(
        left_data: Shared<Array2<f32>>,
        right_data: Shared<Array1<f32>>,
        data: Shared<Array1<f32>>,
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

pub(crate) struct MatrixVectorMulBackwardLeft {
    left_gradient: Rc<Gradient<Ix2>>,
    right_data: Shared<Array1<f32>>,
    gradient: Rc<Gradient<Ix1>>,
}

impl MatrixVectorMulBackwardLeft {
    pub(crate) fn new(
        left_gradient: Rc<Gradient<Ix2>>,
        right_data: Shared<Array1<f32>>,
        gradient: Rc<Gradient<Ix1>>,
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
        Zip::from(&mut *self.left_gradient.borrow_mut())
            .and_broadcast(&self.gradient.borrow().slice(s![.., NewAxis]))
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|d, &f, &s| *d += f * s);
    }
}

pub(crate) struct MatrixVectorMulBackwardRight {
    left_data: Shared<Array2<f32>>,
    right_gradient: Rc<Gradient<Ix1>>,
    gradient: Rc<Gradient<Ix1>>,
}

impl MatrixVectorMulBackwardRight {
    pub(crate) fn new(
        left_data: Shared<Array2<f32>>,
        right_gradient: Rc<Gradient<Ix1>>,
        gradient: Rc<Gradient<Ix1>>,
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
            &*self.gradient.borrow(),
            1.,
            &mut *self.right_gradient.borrow_mut(),
        );
    }
}

pub(crate) struct MatrixVectorMulBackward {
    left: MatrixVectorMulBackwardLeft,
    right: MatrixVectorMulBackwardRight,
}

impl MatrixVectorMulBackward {
    pub(crate) fn new(
        left: MatrixVectorMulBackwardLeft,
        right: MatrixVectorMulBackwardRight,
    ) -> Self {
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
