use super::{Backward, Forward, SharedTensor, SwitchableTensor};
use ndarray::Dimension;
use std::rc::Rc;

pub struct Unsqueeze<D>
where
    D: Dimension,
{
    operand_data: SharedTensor<D>,
    data: SharedTensor<D::Larger>,
}

impl<D> Unsqueeze<D>
where
    D: Dimension,
{
    pub fn new(operand_data: SharedTensor<D>, data: SharedTensor<D::Larger>) -> Self {
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

pub struct UnsqueezeBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<D::Larger>>,
}

impl<D> UnsqueezeBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<D::Larger>>,
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
        let gradient = self.gradient.array();
        let view = gradient
            .view()
            .into_shape(self.operand_gradient.shape())
            .unwrap();

        *self.operand_gradient.array_mut() += &view;
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
