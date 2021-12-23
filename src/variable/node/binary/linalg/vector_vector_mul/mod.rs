#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, push_vec_vec_gradient, Backward, Data, Forward, Gradient,
    Overwrite, Tensor,
};
use ndarray::{arr0, Ix0, Ix1};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix0>>,
    computed: Cell<bool>,
}

impl<Lhs, Rhs> VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(arr0(0.));

        Self {
            left,
            right,
            data,
            computed: Cell::new(false),
        }
    }
}

impl<Lhs, Rhs> Data for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    type Dim = Ix0;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        *self.data.borrow_mut() = arr0(self.left.data().dot(&*self.right.data()));
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<Lhs, Rhs> Debug for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorVectorMul")
            .field("data", &self.data.borrow())
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<Lhs, Rhs> Display for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMulBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix0>>>,
    overwrite: Cell<bool>,
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, LhsG, RhsD, RhsG> VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    pub fn new(
        left_data: Rc<LhsD>,
        left_grad: Rc<LhsG>,
        right_data: Rc<RhsD>,
        right_grad: Rc<RhsG>,
    ) -> Self {
        Self {
            gradient: RefCell::new(Some(arr0(0.))),
            overwrite: Cell::new(true),
            left_data,
            left_grad,
            right_data,
            right_grad,
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Gradient for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    type Dim = Ix0;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Overwrite for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    fn backward(&self) {
        let gradient = self.gradient();
        push_vec_vec_gradient(
            &*self.left_grad,
            &self.right_data.data(),
            &gradient.clone().into_scalar(),
        );
        push_vec_vec_gradient(
            &*self.right_grad,
            &self.left_data.data(),
            &gradient.clone().into_scalar(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(arr0(0.));
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Debug for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorVectorMulBackward")
            .field("gradient", &self.gradient.borrow())
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Display for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMulBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Overwrite,
    U: Data<Dim = Ix1>,
{
    gradient: RefCell<Option<Tensor<Ix0>>>,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Overwrite,
    U: Data<Dim = Ix1>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        Self {
            gradient: RefCell::new(Some(arr0(0.))),
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Overwrite,
    U: Data<Dim = Ix1>,
{
    type Dim = Ix0;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Overwrite,
    U: Data<Dim = Ix1>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Overwrite,
    U: Data<Dim = Ix1>,
{
    fn backward(&self) {
        push_vec_vec_gradient(
            &*self.diff_operand,
            &self.no_diff_operand.data(),
            &self.gradient().clone().into_scalar(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(arr0(0.));
    }
}

impl<T, U> Debug for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Overwrite,
    U: Data<Dim = Ix1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorVectorMulBackwardUnary")
            .field("gradient", &self.gradient.borrow())
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T, U> Display for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Overwrite,
    U: Data<Dim = Ix1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
