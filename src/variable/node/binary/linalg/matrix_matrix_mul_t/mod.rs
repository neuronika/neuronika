#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{linalg::general_mat_mul, Ix2};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct MatrixMatrixMulT {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    data: Rc<RefCell<Tensor<Ix2>>>,
    computed: Cell<bool>,
}

impl MatrixMatrixMulT {
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

impl Forward for MatrixMatrixMulT {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        general_mat_mul(
            1.0,
            &*self.left_data.borrow(),
            &self.right_data.borrow().t(),
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

pub struct MatrixMatrixMulTBackward {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    shape: Ix2,
}

impl MatrixMatrixMulTBackward {
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

impl Backward for MatrixMatrixMulTBackward {
    fn backward(&self) {
        let gradient = expect_tensor(&self.gradient);
        {
            {
                let mut left_gradient = expect_tensor_mut(&self.left_gradient);
                let right_data = self.right_data.borrow();

                general_mat_mul(1.0, &*gradient, &right_data, 1.0, &mut *left_gradient);
            }

            {
                let mut right_gradient = expect_tensor_mut(&self.right_gradient);
                let left_data = self.left_data.borrow();

                general_mat_mul(1.0, &gradient.t(), &left_data, 1.0, &mut *right_gradient)
            }
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

pub struct MatrixMatrixMulTBackwardLeft {
    left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    shape: Ix2,
}

impl MatrixMatrixMulTBackwardLeft {
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

impl Backward for MatrixMatrixMulTBackwardLeft {
    fn backward(&self) {
        let mut left_gradient = expect_tensor_mut(&self.left_gradient);
        let right_data = self.right_data.borrow();
        let gradient = expect_tensor(&self.gradient);

        general_mat_mul(1.0, &*gradient, &right_data, 1.0, &mut *left_gradient);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

pub struct MatrixMatrixMulTBackwardRight {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    shape: Ix2,
}

impl MatrixMatrixMulTBackwardRight {
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

impl Backward for MatrixMatrixMulTBackwardRight {
    fn backward(&self) {
        let mut right_gradient = expect_tensor_mut(&self.right_gradient);
        let gradient = expect_tensor(&self.gradient);
        let left_data = self.left_data.borrow();

        general_mat_mul(1.0, &gradient.t(), &left_data, 1.0, &mut *right_gradient)
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
