use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::{arr0, Ix0, Ix1, Zip};
use std::rc::Rc;

pub struct VectorVectorMul {
    left_data: SharedTensor<Ix1>,
    right_data: SharedTensor<Ix1>,
    data: SharedTensor<Ix0>,
}

impl VectorVectorMul {
    pub fn new(
        left_data: SharedTensor<Ix1>,
        right_data: SharedTensor<Ix1>,
        data: SharedTensor<Ix0>,
    ) -> Self {
        Self {
            left_data,
            right_data,
            data,
        }
    }
}

impl Forward for VectorVectorMul {
    fn forward(&self) {
        *self.data.borrow_mut() = arr0(self.left_data.borrow().dot(&*self.right_data.borrow()));
    }
}

pub struct VectorVectorMulBackwardUnary {
    operand_data: SharedTensor<Ix1>,
    operand_gradient: Rc<SwitchableTensor<Ix1>>,
    gradient: Rc<SwitchableTensor<Ix0>>,
}

impl VectorVectorMulBackwardUnary {
    pub fn new(
        operand_data: SharedTensor<Ix1>,
        operand_gradient: Rc<SwitchableTensor<Ix1>>,
        gradient: Rc<SwitchableTensor<Ix0>>,
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
        Zip::from(&mut *self.operand_gradient.array_mut())
            .and(&*self.operand_data.borrow())
            .and_broadcast(&*self.gradient.array())
            .for_each(|op_grad_el, &data_el, &grad_el| *op_grad_el += data_el * grad_el);
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
