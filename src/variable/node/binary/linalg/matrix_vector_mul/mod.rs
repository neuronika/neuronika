use super::{Backward, Forward, OptionalTensor, Tensor};
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

pub struct MatrixVectorMulBackwardLeft {
    left_gradient: Rc<OptionalTensor<Ix2>>,
    right_data: Rc<RefCell<Tensor<Ix1>>>,
    gradient: Rc<OptionalTensor<Ix1>>,
}

impl MatrixVectorMulBackwardLeft {
    pub fn new(
        left_gradient: Rc<OptionalTensor<Ix2>>,
        right_data: Rc<RefCell<Tensor<Ix1>>>,
        gradient: Rc<OptionalTensor<Ix1>>,
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
        Zip::from(&mut *self.left_gradient.content_mut())
            .and_broadcast(&self.gradient.content().slice(s![.., NewAxis]))
            .and_broadcast(&*self.right_data.borrow())
            .for_each(|d, f, s| *d += f * s);
    }
}

pub struct MatrixVectorMulBackwardRight {
    left_data: Rc<RefCell<Tensor<Ix2>>>,
    right_gradient: Rc<OptionalTensor<Ix1>>,
    gradient: Rc<OptionalTensor<Ix1>>,
}

impl MatrixVectorMulBackwardRight {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix2>>>,
        right_gradient: Rc<OptionalTensor<Ix1>>,
        gradient: Rc<OptionalTensor<Ix1>>,
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
            &*self.gradient.content(),
            1.,
            &mut *self.right_gradient.content_mut(),
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
