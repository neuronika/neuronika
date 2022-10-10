use std::rc::Rc;

use ndarray::{Array, Dimension};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
};

pub(crate) struct Unsqueeze<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D::Larger>>,
}

impl<D> Unsqueeze<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_data: Shared<Array<f32, D>>,
        data: Shared<Array<f32, D::Larger>>,
    ) -> Self {
        Self { operand_data, data }
    }
}

impl<D> Forward for Unsqueeze<D>
where
    D: Dimension,
{
    fn forward(&self) {
        let mut data = self.data.borrow_mut();
        let operand_data = self.operand_data.borrow();
        let mut unsqueezed = data.view_mut().into_shape(operand_data.raw_dim()).unwrap();
        unsqueezed.assign(&operand_data);
    }
}

pub(crate) struct UnsqueezeBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
    gradient: Rc<Gradient<Array<f32, D::Larger>, D::Larger>>,
}

impl<D> UnsqueezeBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
        gradient: Rc<Gradient<Array<f32, D::Larger>, D::Larger>>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
        }
    }
}

impl<D> Backward for UnsqueezeBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let gradient = self.gradient.borrow();
        let view = gradient
            .view()
            .into_shape(self.operand_gradient.shape())
            .unwrap();

        *self.operand_gradient.borrow_mut() += &view;
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
