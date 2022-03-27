#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::Dimension;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Unsqueeze<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D::Larger>>>,
    computed: Cell<bool>,
}

impl<D> Unsqueeze<D>
where
    D: Dimension,
{
    pub fn new(operand_data: Rc<RefCell<Tensor<D>>>, data: Rc<RefCell<Tensor<D::Larger>>>) -> Self {
        Self {
            operand_data,
            data,
            computed: Cell::default(),
        }
    }
}

impl<D> Forward for Unsqueeze<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let mut data = self.data.borrow_mut();
        let operand_data = self.operand_data.borrow();
        let mut unsqueezed = data.view_mut().into_shape(operand_data.raw_dim()).unwrap();
        unsqueezed.assign(&operand_data);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct UnsqueezeBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
    shape: D::Larger,
}

impl<D> UnsqueezeBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D::Larger>>>>,
        shape: D::Larger,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            shape,
        }
    }
}

impl<D> Backward for UnsqueezeBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        let view = gradient
            .view()
            .into_shape(operand_gradient.raw_dim())
            .unwrap();

        *operand_gradient += &view;
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
