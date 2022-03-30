use super::{Backward, Forward, OptionalTensor, Tensor};
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

pub struct VectorMatrixMulBackwardLeft {
    left_gradient: Rc<OptionalTensor<Ix1>>,
    right_data: Rc<RefCell<Tensor<Ix2>>>,
    gradient: Rc<OptionalTensor<Ix1>>,
}

impl VectorMatrixMulBackwardLeft {
    pub fn new(
        left_gradient: Rc<OptionalTensor<Ix1>>,
        right_data: Rc<RefCell<Tensor<Ix2>>>,
        gradient: Rc<OptionalTensor<Ix1>>,
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
            &*self.gradient.content(),
            1.,
            &mut *self.left_gradient.content_mut(),
        );
    }
}

pub struct VectorMatrixMulBackwardRight {
    left_data: Rc<RefCell<Tensor<Ix1>>>,
    right_gradient: Rc<OptionalTensor<Ix2>>,
    gradient: Rc<OptionalTensor<Ix1>>,
}

impl VectorMatrixMulBackwardRight {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix1>>>,
        right_gradient: Rc<OptionalTensor<Ix2>>,
        gradient: Rc<OptionalTensor<Ix1>>,
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
        Zip::from(&mut *self.right_gradient.content_mut())
            .and_broadcast(&self.left_data.borrow().slice(s![.., NewAxis]))
            .and_broadcast(&*self.gradient.content())
            .for_each(|d, f, s| *d += f * s);
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
