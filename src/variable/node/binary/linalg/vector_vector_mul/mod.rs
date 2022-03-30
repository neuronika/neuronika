use super::{Backward, Forward, OptionalTensor, Tensor};
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

pub struct VectorVectorMulBackwardUnary {
    operand_data: Rc<RefCell<Tensor<Ix1>>>,
    operand_gradient: Rc<OptionalTensor<Ix1>>,
    gradient: Rc<OptionalTensor<Ix0>>,
}

impl VectorVectorMulBackwardUnary {
    pub fn new(
        operand_data: Rc<RefCell<Tensor<Ix1>>>,
        operand_gradient: Rc<OptionalTensor<Ix1>>,
        gradient: Rc<OptionalTensor<Ix0>>,
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
        Zip::from(&mut *self.operand_gradient.content_mut())
            .and(&*self.operand_data.borrow())
            .and_broadcast(&*self.gradient.content())
            .for_each(|op_grad_el, data_el, grad_el| *op_grad_el += data_el * grad_el);
    }
}

pub struct VectorVectorMulBackward {
    left: VectorVectorMulBackwardUnary,
    right: VectorVectorMulBackwardUnary,
}

impl VectorVectorMulBackward {
    pub fn new(left: VectorVectorMulBackwardUnary, right: VectorVectorMulBackwardUnary) -> Self {
        Self { left, right }
    }
}

impl Backward for VectorVectorMulBackward {
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
