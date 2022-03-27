#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{arr0, Ix0, Ix1, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct VectorVectorMul {
    left_data: Rc<RefCell<Tensor<Ix1>>>,
    right_data: Rc<RefCell<Tensor<Ix1>>>,
    data: Rc<RefCell<Tensor<Ix0>>>,
    computed: Cell<bool>,
}

impl VectorVectorMul {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix1>>>,
        right_data: Rc<RefCell<Tensor<Ix1>>>,
        data: Rc<RefCell<Tensor<Ix0>>>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
            computed: Cell::default(),
        }
    }
}

impl Forward for VectorVectorMul {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        *self.data.borrow_mut() = arr0(self.left_data.borrow().dot(&*self.right_data.borrow()));
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct VectorVectorMulBackward {
    left_data: Rc<RefCell<Tensor<Ix1>>>,
    left_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    right_data: Rc<RefCell<Tensor<Ix1>>>,
    right_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix0>>>>,
}

impl VectorVectorMulBackward {
    pub fn new(
        left_data: Rc<RefCell<Tensor<Ix1>>>,
        left_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
        right_data: Rc<RefCell<Tensor<Ix1>>>,
        right_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix0>>>>,
    ) -> Self {
        Self {
            left_data,
            left_gradient,
            right_data,
            right_gradient,
            gradient,
        }
    }
}

impl Backward for VectorVectorMulBackward {
    fn backward(&self) {
        let gradient = expect_tensor(&self.gradient);

        {
            let mut left_gradient = expect_tensor_mut(&self.left_gradient);
            let right_data = self.right_data.borrow();

            Zip::from(&mut *left_gradient)
                .and(&*right_data)
                .and_broadcast(&*gradient)
                .for_each(|left_grad_el, right_data_el, grad_el| {
                    *left_grad_el += right_data_el * grad_el
                });
        }

        {
            let mut right_gradient = expect_tensor_mut(&self.right_gradient);
            let left_data = self.left_data.borrow();

            Zip::from(&mut *right_gradient)
                .and(&*left_data)
                .and_broadcast(&*gradient)
                .for_each(|right_grad_el, left_data_el, grad_el| {
                    *right_grad_el += left_data_el * grad_el
                });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(arr0(0.));
    }
}

pub struct VectorVectorMulBackwardUnary {
    operand_data: Rc<RefCell<Tensor<Ix1>>>,
    operand_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix0>>>>,
}

impl VectorVectorMulBackwardUnary {
    pub fn new(
        operand_data: Rc<RefCell<Tensor<Ix1>>>,
        operand_gradient: Rc<RefCell<Option<Tensor<Ix1>>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix0>>>>,
    ) -> Self {
        Self {
            operand_gradient,
            operand_data,
            gradient,
        }
    }
}

impl Backward for VectorVectorMulBackwardUnary {
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);
        let data = self.operand_data.borrow();

        Zip::from(&mut *operand_gradient)
            .and(&*data)
            .and_broadcast(&*gradient)
            .for_each(|op_grad_el, data_el, grad_el| *op_grad_el += data_el * grad_el);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(arr0(0.));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
