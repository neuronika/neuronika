#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{linalg::general_mat_vec_mul, s, Ix1, Ix2, NewAxis, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct MatrixVectorMul {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    right_data: Rc<RefCell<Tensor<Ix1>>>,
    data: Rc<RefCell<Tensor<Ix1>>>,
    computed: Cell<bool>,
}

impl MatrixVectorMul {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix2>>>,
        right_data: Rc<RefCell<Tensor<Ix1>>>,
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

impl Forward for MatrixVectorMul {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        general_mat_vec_mul(
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

pub struct MatrixVectorMulBackward {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    right_data: Rc<RefCell<Tensor<Ix1>>>,
    right_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    shape: Ix1,
}

impl MatrixVectorMulBackward {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix2>>>,
        left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        right_data: Rc<RefCell<Tensor<Ix1>>>,
        right_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
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

impl Backward for MatrixVectorMulBackward {
    fn backward(&self) {
        let gradient = expect_tensor(&self.gradient);

        {
            let mut left_gradient = expect_tensor_mut(&self.left_gradient);
            let right_data = self.right_data.borrow();

            Zip::from(&mut *left_gradient)
                .and_broadcast(&gradient.slice(s![.., NewAxis]))
                .and_broadcast(&*right_data)
                .for_each(|d, f, s| *d += f * s);
        }

        {
            let mut right_gradient = expect_tensor_mut(&self.right_gradient);
            let left_data = self.left_data.borrow();

            general_mat_vec_mul(1.0, &left_data.t(), &*gradient, 1.0, &mut *right_gradient);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

pub struct MatrixVectorMulBackwardLeft {
    left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
    right_data: Rc<RefCell<Tensor<Ix1>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    shape: Ix1,
}

impl MatrixVectorMulBackwardLeft {
    pub fn new(
        left_gradient: Rc<RefCell<Option<Tensor<Ix2>>>>,
        right_data: Rc<RefCell<Tensor<Ix1>>>,
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

impl Backward for MatrixVectorMulBackwardLeft {
    fn backward(&self) {
        let mut left_gradient = expect_tensor_mut(&self.left_gradient);
        let right_data = self.right_data.borrow();
        let gradient = expect_tensor(&self.gradient);

        {
            Zip::from(&mut *left_gradient)
                .and_broadcast(&gradient.slice(s![.., NewAxis]))
                .and_broadcast(&*right_data)
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

pub struct MatrixVectorMulBackwardRight {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    right_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    shape: Ix1,
}

impl MatrixVectorMulBackwardRight {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix2>>>,
        right_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
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

impl Backward for MatrixVectorMulBackwardRight {
    fn backward(&self) {
        let gradient = expect_tensor(&self.gradient);
        let mut right_gradient = expect_tensor_mut(&self.right_gradient);
        let left_data = self.left_data.borrow();

        general_mat_vec_mul(1.0, &left_data.t(), &*gradient, 1.0, &mut *right_gradient);
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
