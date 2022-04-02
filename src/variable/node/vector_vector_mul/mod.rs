use std::rc::Rc;

use ndarray::{arr0, Array, Ix0, Ix1, Zip};

use crate::variable::{gradient::Gradient, utils::Shared};

use super::{Backward, Forward};

pub(crate) struct VectorVectorMul {
    left_data: Shared<Array<f32, Ix1>>,
    right_data: Shared<Array<f32, Ix1>>,
    data: Shared<Array<f32, Ix0>>,
}

impl VectorVectorMul {
    pub(crate) fn new(
        left_data: Shared<Array<f32, Ix1>>,
        right_data: Shared<Array<f32, Ix1>>,
        data: Shared<Array<f32, Ix0>>,
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

pub(crate) struct VectorVectorMulBackwardUnary {
    operand_data: Shared<Array<f32, Ix1>>,
    operand_gradient: Rc<Gradient<Ix1>>,
    gradient: Rc<Gradient<Ix0>>,
}

impl VectorVectorMulBackwardUnary {
    pub(crate) fn new(
        operand_data: Shared<Array<f32, Ix1>>,
        operand_gradient: Rc<Gradient<Ix1>>,
        gradient: Rc<Gradient<Ix0>>,
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
        Zip::from(&mut *self.operand_gradient.borrow_mut())
            .and(&*self.operand_data.borrow())
            .and_broadcast(&*self.gradient.borrow())
            .for_each(|op_grad_el, &data_el, &grad_el| *op_grad_el += data_el * grad_el);
    }
}

pub(crate) struct VectorVectorMulBackward {
    left: VectorVectorMulBackwardUnary,
    right: VectorVectorMulBackwardUnary,
}

impl VectorVectorMulBackward {
    pub(crate) fn new(
        left: VectorVectorMulBackwardUnary,
        right: VectorVectorMulBackwardUnary,
    ) -> Self {
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
