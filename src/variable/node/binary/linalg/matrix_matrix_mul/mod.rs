#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
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

pub struct MatrixMatrixMulBackward {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    shape: Ix2,
}

impl MatrixMatrixMulBackward {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix2>>>,
        left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        right_data: Rc<RefCell<Tensor<Ix2>>>,
        right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        shape: Ix2,
    ) -> Self {
        Self {
            left_data,
            left_gradient,
            right_data,
            right_gradient,
            gradient,
            shape,
        }
    }
}

impl Backward for MatrixMatrixMulBackward {
    fn backward(&self) {
        let gradient = expect_tensor(&self.gradient);

        {
            let right_data = self.right_data.borrow();
            let mut left_gradient = expect_tensor_mut(&self.left_gradient);

            general_mat_mul(1.0, &*gradient, &right_data.t(), 1.0, &mut *left_gradient);
        }

        {
            let left_data = self.left_data.borrow();
            let mut right_gradient = expect_tensor_mut(&self.right_gradient);

            general_mat_mul(1.0, &left_data.t(), &*gradient, 1.0, &mut *right_gradient)
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

pub struct MatrixMatrixMulBackwardLeft {
    left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    shape: Ix2,
}

impl MatrixMatrixMulBackwardLeft {
    pub fn new(
        left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        right_data: Rc<RefCell<Tensor<Ix2>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        shape: Ix2,
    ) -> Self {
        Self {
            left_gradient,
            right_data,
            gradient,
            shape,
        }
    }
}

impl Backward for MatrixMatrixMulBackwardLeft {
    fn backward(&self) {
        let mut left_gradient = expect_tensor_mut(&self.left_gradient);
        let gradient = expect_tensor(&self.gradient);
        let right_data = self.right_data.borrow();

        general_mat_mul(1.0, &*gradient, &right_data.t(), 1.0, &mut *left_gradient);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

pub struct MatrixMatrixMulBackwardRight {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    shape: Ix2,
}

impl MatrixMatrixMulBackwardRight {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix2>>>,
        right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        shape: Ix2,
    ) -> Self {
        Self {
            left_data,
            right_gradient,
            gradient,
            shape,
        }
    }
}

impl Backward for MatrixMatrixMulBackwardRight {
    fn backward(&self) {
        let mut right_gradient = expect_tensor_mut(&self.right_gradient);
        let gradient = expect_tensor(&self.gradient);
        let left_data = self.left_data.borrow();

        general_mat_mul(1.0, &left_data.t(), &*gradient, 1.0, &mut *right_gradient)
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
