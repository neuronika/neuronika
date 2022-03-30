use super::{Backward, Forward, OptionalTensor, Tensor};
use ndarray::{linalg::general_mat_mul, Ix2};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct MatrixMatrixMul {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    data: Rc<RefCell<Tensor<Ix2>>>,
    computed: Cell<bool>,
}

impl MatrixMatrixMul {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix2>>>,
        right_data: Rc<RefCell<Tensor<Ix2>>>,
        data: Rc<RefCell<Tensor<Ix2>>>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
            computed: Cell::default(),
        }
    }
}

impl Forward for MatrixMatrixMul {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        general_mat_mul(
            1.0,
            &*self.left_data.borrow(),
            &*self.right_data.borrow(),
            0.0,
            &mut *self.data.borrow_mut(),
        );
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct MatrixMatrixMulBackwardLeft {
    left_gradient: Rc<OptionalTensor<Ix2>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    gradient: Rc<OptionalTensor<Ix2>>,
}

impl MatrixMatrixMulBackwardLeft {
    pub fn new(
        left_gradient: Rc<OptionalTensor<Ix2>>,
        right_data: Rc<RefCell<Tensor<Ix2>>>,
        gradient: Rc<OptionalTensor<Ix2>>,
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
            &*self.gradient.content(),
            &self.right_data.borrow().t(),
            1.,
            &mut *self.left_gradient.content_mut(),
        );
    }
}

pub struct MatrixMatrixMulBackwardRight {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    right_gradient: Rc<OptionalTensor<Ix2>>,
    gradient: Rc<OptionalTensor<Ix2>>,
}

impl MatrixMatrixMulBackwardRight {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix2>>>,
        right_gradient: Rc<OptionalTensor<Ix2>>,
        gradient: Rc<OptionalTensor<Ix2>>,
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
            &*self.gradient.content(),
            1.,
            &mut *self.right_gradient.content_mut(),
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
