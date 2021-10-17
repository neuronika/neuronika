use super::{
    expect_tensor, expect_tensor_mut, push_mat_mat_gradient, Backward, Data, DotDim, Forward,
    Gradient, Overwrite, Tensor,
};
use ndarray::{linalg::general_mat_mul, Ix2};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

pub struct MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix2>>,
    computed: Cell<bool>,
}

impl<Lhs, Rhs> MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(left.data().raw_dim(), right.data().raw_dim());
        let data = RefCell::new(Tensor::zeros((shape[0], shape[1])));

        Self {
            left,
            right,
            data,
            computed: Cell::new(false),
        }
    }
}

impl<Lhs, Rhs> Data for MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    type Dim = Ix2;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        general_mat_mul(
            1.0,
            &*self.left.data(),
            &*self.right.data(),
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

pub struct MatrixMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix2>>>,
    shape: Ix2,
    overwrite: Cell<bool>,
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, LhsG, RhsD, RhsG> MatrixMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    pub fn new(
        left_data: Rc<LhsD>,
        left_grad: Rc<LhsG>,
        right_data: Rc<RhsD>,
        right_grad: Rc<RhsG>,
    ) -> Self {
        let shape = DotDim::shape(
            left_grad.gradient().raw_dim(),
            right_grad.gradient().raw_dim(),
        );

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape))),
            shape,
            overwrite: Cell::new(true),
            left_data,
            left_grad,
            right_data,
            right_grad,
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Gradient for MatrixMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    type Dim = Ix2;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Overwrite for MatrixMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for MatrixMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn backward(&self) {
        let gradient = self.gradient();
        push_mat_mat_gradient(&*self.left_grad, &gradient, &self.right_data.data().t());
        push_mat_mat_gradient(&*self.right_grad, &self.left_data.data().t(), &gradient);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

pub struct MatrixMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix2>>>,
    shape: Ix2,
    overwrite: Cell<bool>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
}

impl<LhsG, RhsD> MatrixMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    pub fn new(left_grad: Rc<LhsG>, right_data: Rc<RhsD>) -> Self {
        let shape = DotDim::shape(left_grad.gradient().raw_dim(), right_data.data().raw_dim());

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape))),
            shape,
            overwrite: Cell::new(true),
            left_grad,
            right_data,
        }
    }
}

impl<LhsG, RhsD> Gradient for MatrixMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    type Dim = Ix2;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsG, RhsD> Overwrite for MatrixMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsG, RhsD> Backward for MatrixMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn backward(&self) {
        push_mat_mat_gradient(
            &*self.left_grad,
            &self.gradient(),
            &self.right_data.data().t(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

pub struct MatrixMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix2>>>,
    shape: Ix2,
    overwrite: Cell<bool>,
    left_data: Rc<LhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, RhsG> MatrixMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    pub fn new(left_data: Rc<LhsD>, right_grad: Rc<RhsG>) -> Self {
        let shape = DotDim::shape(left_data.data().raw_dim(), right_grad.gradient().raw_dim());

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape))),
            shape,
            overwrite: Cell::new(true),
            left_data,
            right_grad,
        }
    }
}

impl<LhsD, RhsG> Gradient for MatrixMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    type Dim = Ix2;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, RhsG> Overwrite for MatrixMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, RhsG> Backward for MatrixMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn backward(&self) {
        push_mat_mat_gradient(
            &*self.right_grad,
            &self.left_data.data().t(),
            &self.gradient(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

#[cfg(test)]
mod test;
