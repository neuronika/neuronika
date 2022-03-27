#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{linalg::general_mat_vec_mul, s, Ix1, Ix2, NewAxis, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct VectorMatrixMul {
    left_data: Rc<RefCell<Tensor<Ix1>>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    data: Rc<RefCell<Tensor<Ix1>>>,
    computed: Cell<bool>,
}

impl VectorMatrixMul {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix1>>>,
        right_data: Rc<RefCell<Tensor<Ix2>>>,
        data: Rc<RefCell<Tensor<Ix1>>>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
            computed: Cell::default(),
        }
    }
}

impl Forward for VectorMatrixMul {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        general_mat_vec_mul(
            1.0,
            &self.right_data.borrow().t(),
            &*self.left_data.borrow(),
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

pub struct VectorMatrixMulBackward {
    left_data: Rc<RefCell<Tensor<Ix1>>>,
    left_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    shape: Ix1,
}

impl VectorMatrixMulBackward {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix1>>>,
        left_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
        right_data: Rc<RefCell<Tensor<Ix2>>>,
        right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
        shape: Ix1,
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

impl Backward for VectorMatrixMulBackward {
    fn backward(&self) {
        let gradient = expect_tensor(&self.gradient);

        {
            let mut left_gradient = expect_tensor_mut(&self.left_gradient);
            let right_data = self.right_data.borrow();

            general_mat_vec_mul(1.0, &right_data, &*gradient, 1.0, &mut *left_gradient);
        }

        {
            let mut right_gradient = expect_tensor_mut(&self.right_gradient);
            let left_data = self.left_data.borrow();

            Zip::from(&mut *right_gradient)
                .and_broadcast(&left_data.slice(s![.., NewAxis]))
                .and_broadcast(&*gradient)
                .for_each(|d, f, s| *d += f * s);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

pub struct VectorMatrixMulBackwardLeft {
    left_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    shape: Ix1,
}

impl VectorMatrixMulBackwardLeft {
    pub fn new(
        left_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
        right_data: Rc<RefCell<Tensor<Ix2>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
        shape: Ix1,
    ) -> Self {
        Self {
            left_gradient,
            right_data,
            gradient,
            shape,
        }
    }
}

impl Backward for VectorMatrixMulBackwardLeft {
    fn backward(&self) {
        let mut left_gradient = expect_tensor_mut(&self.left_gradient);
        let gradient = expect_tensor(&self.gradient);
        let right_data = self.right_data.borrow();

        general_mat_vec_mul(1.0, &right_data, &*gradient, 1.0, &mut *left_gradient);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

pub struct VectorMatrixMulBackwardRight {
    left_data: Rc<RefCell<Tensor<Ix1>>>,
    right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    shape: Ix1,
}

impl VectorMatrixMulBackwardRight {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix1>>>,
        right_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
        shape: Ix1,
    ) -> Self {
        Self {
            left_data,
            right_gradient,
            gradient,
            shape,
        }
    }
}

impl Backward for VectorMatrixMulBackwardRight {
    fn backward(&self) {
        let mut right_gradient = expect_tensor_mut(&self.right_gradient);
        let gradient = expect_tensor(&self.gradient);
        let left_data = self.left_data.borrow();

        Zip::from(&mut *right_gradient)
            .and_broadcast(&left_data.slice(s![.., NewAxis]))
            .and_broadcast(&*gradient)
            .for_each(|d, f, s| *d += f * s);
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
