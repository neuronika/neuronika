#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, push_vec_vec_gradient, Backward, Cache, Data, Forward,
    Gradient, Overwrite, Tensor,
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
pub struct VectorVectorMul<Lhs: ?Sized, Rhs: ?Sized>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix0>>,
    computed: Cell<bool>,
}

impl<Lhs: ?Sized, Rhs: ?Sized> VectorVectorMul<Lhs, Rhs>
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

impl<Lhs: ?Sized, Rhs: ?Sized> Data for VectorVectorMul<Lhs, Rhs>
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
impl<Lhs: ?Sized, Rhs: ?Sized> Cache for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<Lhs: ?Sized, Rhs: ?Sized> Forward for VectorVectorMul<Lhs, Rhs>
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
}

impl<Lhs: ?Sized, Rhs: ?Sized> Debug for VectorVectorMul<Lhs, Rhs>
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

impl<Lhs: ?Sized, Rhs: ?Sized> Display for VectorVectorMul<Lhs, Rhs>
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
pub struct VectorVectorMulBackward<LhsD: ?Sized, LhsG: ?Sized, RhsD: ?Sized, RhsG: ?Sized>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix1>,
{
    gradient: RefCell<Option<Tensor<Ix0>>>,
    overwrite: Cell<bool>,
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD: ?Sized, LhsG: ?Sized, RhsD: ?Sized, RhsG: ?Sized>
    VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix1>,
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

impl<LhsD: ?Sized, LhsG: ?Sized, RhsD: ?Sized, RhsG: ?Sized> Gradient
    for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix1>,
{
    type Dim = Ix0;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD: ?Sized, LhsG: ?Sized, RhsD: ?Sized, RhsG: ?Sized> Overwrite
    for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix1>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD: ?Sized, LhsG: ?Sized, RhsD: ?Sized, RhsG: ?Sized> Backward
    for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix1>,
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

impl<LhsD: ?Sized, LhsG: ?Sized, RhsD: ?Sized, RhsG: ?Sized> Debug
    for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorVectorMulBackward")
            .field("gradient", &self.gradient.borrow())
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<LhsD: ?Sized, LhsG: ?Sized, RhsD: ?Sized, RhsG: ?Sized> Display
    for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix1>,
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
pub struct VectorVectorMulBackwardUnary<T: ?Sized, U: ?Sized>
where
    T: Gradient<Dim = Ix1>,
    U: Data<Dim = Ix1>,
{
    gradient: RefCell<Option<Tensor<Ix0>>>,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T: ?Sized, U: ?Sized> VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1>,
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

impl<T: ?Sized, U: ?Sized> Gradient for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1>,
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

impl<T: ?Sized, U: ?Sized> Overwrite for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1>,
    U: Data<Dim = Ix1>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: ?Sized, U: ?Sized> Backward for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1>,
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

impl<T: ?Sized, U: ?Sized> Debug for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1>,
    U: Data<Dim = Ix1>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorVectorMulBackwardUnary")
            .field("gradient", &self.gradient.borrow())
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T: ?Sized, U: ?Sized> Display for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1>,
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
