use super::{
    super::{BroadTensor, Broadcasted, Tensor, Var},
    broadcasted_zeros, DotDim,
};
use ndarray::{
    concatenate,
    linalg::{general_mat_mul, general_mat_vec_mul},
    stack, Axis, DimMax, Dimension, Ix1, Ix2, RemoveAxis, Zip,
};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub trait Data {
    type Dim: Dimension;

    fn data(&self) -> Ref<Tensor<Self::Dim>>;
}

pub trait Forward {
    fn forward(&self);

    fn was_computed(&self) -> bool;

    fn reset_computation(&self);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Input<D: Dimension> {
    data: RefCell<Tensor<D>>,
}

impl<D: Dimension> Input<D> {
    pub fn new(data: Tensor<D>) -> Var<Self> {
        let input = Self {
            data: RefCell::new(data),
        };

        Var::new(input)
    }

    pub(crate) fn data_mut(&self) -> RefMut<Tensor<D>> {
        self.data.borrow_mut()
    }
}

impl<D: Dimension> Data for Input<D> {
    type Dim = D;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<D: Dimension> Forward for Input<D> {
    fn forward(&self) {
        // Nothing
    }

    fn was_computed(&self) -> bool {
        false
    }

    fn reset_computation(&self) {
        // Nothing
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Negation<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> Negation<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());

        Self {
            op,
            data: RefCell::new(data),
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Negation<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = -o);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Negation<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Transpose ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Transpose<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> Transpose<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().t().raw_dim());

        Self {
            op,
            data: RefCell::new(data),
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Transpose<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(self.op.data().t())
            .par_for_each(|v, o| *v = *o);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Transpose<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Addition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Addition<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> Addition<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(broadcasted_zeros(&left.data(), &right.data()));

        Self {
            left,
            right,
            data,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for Addition<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Addition<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .par_for_each(|v, l, r| *v = l + r);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Subtraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Subtraction<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> Subtraction<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(broadcasted_zeros(&left.data(), &right.data()));

        Self {
            left,
            right,
            data,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for Subtraction<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Subtraction<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .par_for_each(|v, l, r| *v = l - r);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Multiplication<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> Multiplication<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(broadcasted_zeros(&left.data(), &right.data()));

        Self {
            left,
            right,
            data,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for Multiplication<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Multiplication<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .par_for_each(|v, l, r| *v = l * r);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Division ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Division<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> Division<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(broadcasted_zeros(&left.data(), &right.data()));

        Self {
            left,
            right,
            data,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for Division<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Division<Lhs, Rhs>
where
    Lhs: Data + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .par_for_each(|v, l, r| *v = l / r);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix2> + Forward,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix2>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix2> + Forward,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(left.data().raw_dim(), right.data().raw_dim());
        let data = RefCell::new(Tensor::zeros((shape[0], shape[1])));

        Self {
            left,
            right,
            data,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix2> + Forward,
{
    type Dim = Ix2;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix2> + Forward,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        general_mat_mul(
            1.0,
            &*self.left.data(),
            &*self.right.data(),
            0.0,
            &mut *self.data.borrow_mut(),
        );
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix1>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(left.data().raw_dim(), right.data().raw_dim());
        let data = RefCell::new(Tensor::zeros(shape[0]));

        Self {
            left,
            right,
            data,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        general_mat_vec_mul(
            1.0,
            &*self.left.data(),
            &*self.right.data(),
            0.0,
            &mut *self.data.borrow_mut(),
        );
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1> + Forward,
    Rhs: Data<Dim = Ix2> + Forward,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix1>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1> + Forward,
    Rhs: Data<Dim = Ix2> + Forward,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(left.data().raw_dim(), right.data().raw_dim());
        let data = RefCell::new(Tensor::zeros(shape[0]));

        Self {
            left,
            right,
            data,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1> + Forward,
    Rhs: Data<Dim = Ix2> + Forward,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1> + Forward,
    Rhs: Data<Dim = Ix2> + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        general_mat_vec_mul(
            1.0,
            &self.right.data().t(),
            &*self.left.data(),
            0.0,
            &mut *self.data.borrow_mut(),
        );
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix1>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(left.data().raw_dim(), right.data().raw_dim());
        let data = RefCell::new(Tensor::zeros(shape[0]));

        Self {
            left,
            right,
            data,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        self.data.borrow_mut()[0] = self.left.data().dot(&*self.right.data());
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Power ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Power<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    exp: i32,
    state: Cell<bool>,
}

impl<T: Data + Forward> Power<T> {
    pub fn new(op: Rc<T>, exp: i32) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());

        Self {
            op,
            data: RefCell::new(data),
            exp,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Power<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        let exp = self.exp;
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.powi(exp));
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Power<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sum ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Sum<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<Ix1>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> Sum<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(1));

        Self {
            op,
            data,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Sum<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        self.data.borrow_mut()[0] = self.op.data().sum();
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Sum<T> {
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logn ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Logn<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> Logn<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(op.data().raw_dim()));

        Self {
            op,
            data,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Logn<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.ln());
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Logn<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct ReLU<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> ReLU<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(op.data().raw_dim()));

        Self {
            op,
            data,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for ReLU<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.max(0.));
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for ReLU<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct LeakyReLU<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> LeakyReLU<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(op.data().raw_dim()));

        Self {
            op,
            data,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for LeakyReLU<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = if *o > 0.0 { *o } else { 0.01 * o });
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for LeakyReLU<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftPlus ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SoftPlus<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> SoftPlus<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(op.data().raw_dim()));

        Self {
            op,
            data,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for SoftPlus<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| {
                *v = if *o < -15.0 {
                    0.0
                } else if *o > 15.0 {
                    *o
                } else {
                    (1.0 + o.exp()).ln()
                }
            });
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for SoftPlus<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Sigmoid<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> Sigmoid<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(op.data().raw_dim()));

        Self {
            op,
            data,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Sigmoid<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| {
                *v = if *o >= 15.0 {
                    1.0
                } else if *o <= -15.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-*o).exp())
                }
            });
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Sigmoid<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TanH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct TanH<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> TanH<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(op.data().raw_dim()));

        Self {
            op,
            data,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for TanH<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.tanh());
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for TanH<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exp ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Exp<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> Exp<T> {
    pub fn new(op: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(op.data().raw_dim()));

        Self {
            op,
            data,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Exp<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.exp());
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Exp<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Softmax<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    axis: usize,
    state: Cell<bool>,
}

impl<T: Data + Forward> Softmax<T> {
    pub fn new(op: Rc<T>, axis: usize) -> Self {
        let data = RefCell::new(Tensor::zeros(op.data().raw_dim()));

        Self {
            op,
            data,
            axis,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Softmax<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        let axis = self.axis;
        Zip::from(self.data.borrow_mut().lanes_mut(Axis(axis)))
            .and(self.op.data().lanes(Axis(axis)))
            .for_each(|lane_v, lane_o| {
                let max = lane_o.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = &lane_o.map(|el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_v)
                    .and(num)
                    .for_each(|lane_v_el, num_el| *lane_v_el = *num_el / den);
            });
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Softmax<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSoftmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct LogSoftmax<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    axis: usize,
    state: Cell<bool>,
}

impl<T: Data + Forward> LogSoftmax<T> {
    pub fn new(op: Rc<T>, axis: usize) -> Self {
        let data = RefCell::new(Tensor::zeros(op.data().raw_dim()));

        Self {
            op,
            data,
            axis,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for LogSoftmax<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        let axis = self.axis;
        Zip::from(self.data.borrow_mut().lanes_mut(Axis(axis)))
            .and(self.op.data().lanes(Axis(axis)))
            .for_each(|lane_v, lane_o| {
                let max = lane_o.fold(std::f32::MIN, |x, y| x.max(*y));
                let exp = &lane_o.map(|el| (el - max).exp());
                let log_sum_exp = exp.sum().ln();
                Zip::from(lane_v)
                    .and(lane_o)
                    .for_each(|lane_v_el, lane_o_el| *lane_v_el = lane_o_el - log_sum_exp - max);
            });
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for LogSoftmax<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim> + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    data: RefCell<Tensor<Lhs::Dim>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim> + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let data = RefCell::new(
            concatenate(
                Axis(axis),
                &[
                    Tensor::zeros(left.data().raw_dim()).view(),
                    Tensor::zeros(right.data().raw_dim()).view(),
                ],
            )
            .unwrap(),
        );

        Self {
            left,
            right,
            data,
            axis,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim> + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: RemoveAxis,
{
    type Dim = Lhs::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim> + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: RemoveAxis,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        let lhs_data = self.left.data();
        let rhs_data = self.right.data();
        let mut data = self.data.borrow_mut();
        let axis = self.axis;
        let (mut lhs_portion, mut rhs_portion) = data
            .view_mut()
            .split_at(Axis(axis), lhs_data.len_of(Axis(axis)));
        Zip::from(&*lhs_data)
            .and(&mut lhs_portion)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
        Zip::from(&*rhs_data)
            .and(&mut rhs_portion)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim> + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    data: RefCell<Tensor<<Lhs::Dim as Dimension>::Larger>>,
    state: Cell<bool>,
}

impl<Lhs, Rhs> Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim> + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let data = RefCell::new(
            stack(
                Axis(axis),
                &[
                    Tensor::zeros(left.data().raw_dim()).view(),
                    Tensor::zeros(right.data().raw_dim()).view(),
                ],
            )
            .unwrap(),
        );

        Self {
            left,
            right,
            data,
            axis,
            state: Cell::new(false),
        }
    }

    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }

    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs> Data for Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim> + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: RemoveAxis,
{
    type Dim = <Lhs::Dim as Dimension>::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim> + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: RemoveAxis,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        let lhs_data = self.left.data();
        let rhs_data = self.right.data();
        let mut data = self.data.borrow_mut();
        let axis = self.axis;
        let mut subview_iter = data.axis_iter_mut(Axis(axis));

        let mut subview_left = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<Lhs::Dim>()
            .unwrap();
        Zip::from(&*lhs_data)
            .and(&mut subview_left)
            .for_each(|single_el, fused_el| *fused_el = *single_el);

        let mut subview_right = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<Rhs::Dim>()
            .unwrap();
        Zip::from(&*rhs_data)
            .and(&mut subview_right)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Unsqueeze ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Unsqueeze<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<<<T as Data>::Dim as Dimension>::Larger>>,
    axis: usize,
    state: Cell<bool>,
}

impl<T: Data + Forward> Unsqueeze<T> {
    pub fn new(op: Rc<T>, axis: usize) -> Self {
        let shape = op.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.insert_axis(Axis(axis))));

        Self {
            op,
            data,
            axis,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Unsqueeze<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        let mut data = self.data.borrow_mut();
        let mut unsqueezed = data
            .axis_iter_mut(Axis(self.axis))
            .next()
            .unwrap()
            .into_dimensionality::<T::Dim>()
            .unwrap();
        let operand_data = self.op.data();
        Zip::from(&mut unsqueezed)
            .and(&*operand_data)
            .par_for_each(|unsqueezed_el, operand_data_el| *unsqueezed_el = *operand_data_el);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Unsqueeze<T> {
    type Dim = <T::Dim as Dimension>::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Chunk ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Chunk<T: Data + Forward> {
    op: Rc<T>,
    chunk_no: usize,
    chunk_shape: T::Dim,
    data: RefCell<Tensor<T::Dim>>,
    state: Cell<bool>,
}

impl<T: Data + Forward> Chunk<T> {
    pub fn new(op: Rc<T>, chunk: Tensor<T::Dim>, chunk_no: usize) -> Self {
        Self {
            op,
            chunk_shape: chunk.raw_dim(),
            data: RefCell::new(chunk),
            chunk_no,
            state: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T: Data + Forward> Forward for Chunk<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
        let mut data = self.data.borrow_mut();
        let op_data = self.op.data();
        let (chunk_shape, chunk_no) = (&self.chunk_shape, self.chunk_no);
        let operand_data_chunk = op_data
            .exact_chunks(chunk_shape.clone())
            .into_iter()
            .skip(chunk_no)
            .take(1)
            .next()
            .unwrap();
        Zip::from(&mut *data)
            .and(&operand_data_chunk)
            .par_for_each(|chunk_el, operand_data_chunk_el| *chunk_el = *operand_data_chunk_el);
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        debug_assert_eq!(self.state.get(), true);

        self.state.set(false);
    }
}

impl<T: Data + Forward> Data for Chunk<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::StrideShape;

    const F16_EPSILON: f32 = 9.77e-04;

    fn assert_is_precise_enough<D: Dimension>(our: &Tensor<D>, their: &Tensor<D>) {
        assert!(
            Zip::from(our).and(their).all(|l, r| {
                (*l == 0. && *r == 0.)
                    || (!l.is_finite() && !r.is_finite())
                    || ((1. - r / l).abs() <= F16_EPSILON)
            }),
            "\nLeft:\n{}\nRight:\n{}",
            our,
            their
        );
    }

    fn make_me_an_input<D, Sh>(shape: Sh, elems: Vec<f32>) -> Rc<Input<D>>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Input::new(Tensor::from_shape_vec(shape, elems).unwrap()).forward
    }

    mod negation {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let negation = Negation::new(input.clone());
            assert_eq!(*negation.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(negation.was_computed(), false);
            assert!(Rc::ptr_eq(&negation.operand(), &input));

            negation.forward();
            assert_is_precise_enough(
                &*negation.data(),
                &Tensor::from_shape_vec((3, 3), vec![-1., -2., -3., -4., -5., -6., -7., -8., -9.])
                    .unwrap(),
            );
            assert_eq!(negation.was_computed(), true);

            negation.reset_computation();
            assert_eq!(negation.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let first = Rc::new(Negation::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Negation::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![-1., -2., -3., -4., -5., -6., -7., -8., -9.])
                    .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![-1., -2., -3., -4., -5., -6., -7., -8., -9.])
                    .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod transpose {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let transpose = Transpose::new(input.clone());
            assert_eq!(*transpose.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(transpose.was_computed(), false);
            assert!(Rc::ptr_eq(&transpose.operand(), &input));

            transpose.forward();
            assert_is_precise_enough(
                &*transpose.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]).unwrap(),
            );
            assert_eq!(transpose.was_computed(), true);

            transpose.reset_computation();
            assert_eq!(transpose.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let first = Rc::new(Transpose::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Transpose::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]).unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]).unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod addition {
        use super::*;

        #[test]
        fn single_node() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![1., 1., 1., 1., 1., 1., 1., 1., 1.]);

            let addition = Addition::new(left.clone(), right.clone());
            assert_eq!(*addition.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(addition.was_computed(), false);
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.forward();
            assert_is_precise_enough(
                &*addition.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]).unwrap(),
            );
            assert_eq!(addition.was_computed(), true);
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.reset_computation();
            assert_eq!(addition.was_computed(), false);
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![1.; 9]);

            let first = Rc::new(Addition::new(left.clone(), right.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second = Addition::new(first.clone(), right.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]).unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]).unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![3., 4., 5., 6., 7., 8., 9., 10., 11.])
                    .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }

        #[test]
        fn left_broadcast() {
            let left = make_me_an_input((1, 3), vec![1., 2., 3.]);
            let right = make_me_an_input((2, 2, 3), vec![1.; 12]);

            let addition = Addition::new(left.clone(), right.clone());
            assert_eq!(*addition.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert_eq!(addition.was_computed(), false);
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.forward();
            assert_is_precise_enough(
                &*addition.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.],
                )
                .unwrap(),
            );
            assert_eq!(addition.was_computed(), true);

            addition.reset_computation();
            assert_eq!(addition.was_computed(), false);
        }

        #[test]
        fn right_broadcast() {
            let left = make_me_an_input((2, 2, 3), vec![1.; 12]);
            let right = make_me_an_input((1, 3), vec![1., 2., 3.]);

            let addition = Addition::new(left.clone(), right.clone());
            assert_eq!(*addition.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert_eq!(addition.was_computed(), false);
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.forward();
            assert_is_precise_enough(
                &*addition.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.],
                )
                .unwrap(),
            );
            assert_eq!(addition.was_computed(), true);

            addition.reset_computation();
            assert_eq!(addition.was_computed(), false);
        }
    }

    mod subtraction {
        use super::*;

        #[test]
        fn single_node() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![1.; 9]);

            let subtraction = Subtraction::new(left.clone(), right.clone());
            assert_eq!(*subtraction.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(subtraction.was_computed(), false);
            assert!(Rc::ptr_eq(&subtraction.left_operand(), &left));
            assert!(Rc::ptr_eq(&subtraction.right_operand(), &right));

            subtraction.forward();
            assert_is_precise_enough(
                &*subtraction.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 1., 2., 3., 4., 5., 6., 7., 8.]).unwrap(),
            );
            assert_eq!(subtraction.was_computed(), true);

            subtraction.reset_computation();
            assert_eq!(subtraction.was_computed(), false);
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![1.; 9]);

            let first = Rc::new(Subtraction::new(left.clone(), right.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second = Subtraction::new(first.clone(), right.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 1., 2., 3., 4., 5., 6., 7., 8.]).unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 1., 2., 3., 4., 5., 6., 7., 8.]).unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![-1., 0., 1., 2., 3., 4., 5., 6., 7.]).unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }

        #[test]
        fn left_broadcast() {
            let left = make_me_an_input((1, 3), vec![1., 2., 3.]);
            let right = make_me_an_input((2, 2, 3), vec![1.; 12]);

            let subtraction = Subtraction::new(left.clone(), right.clone());
            assert_eq!(*subtraction.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert_eq!(subtraction.was_computed(), false);
            assert!(Rc::ptr_eq(&subtraction.left_operand(), &left));
            assert!(Rc::ptr_eq(&subtraction.right_operand(), &right));

            subtraction.forward();
            assert_is_precise_enough(
                &*subtraction.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2.],
                )
                .unwrap(),
            );
            assert_eq!(subtraction.was_computed(), true);
            assert!(Rc::ptr_eq(&subtraction.left_operand(), &left));
            assert!(Rc::ptr_eq(&subtraction.right_operand(), &right));

            subtraction.reset_computation();
            assert_eq!(subtraction.was_computed(), false);
        }

        #[test]
        fn right_broadcast() {
            let left = make_me_an_input((2, 2, 3), vec![1.; 12]);
            let right = make_me_an_input((1, 3), vec![1., 2., 3.]);

            let subtraction = Subtraction::new(left.clone(), right.clone());
            assert_eq!(*subtraction.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert_eq!(subtraction.was_computed(), false);
            assert!(Rc::ptr_eq(&subtraction.left_operand(), &left));
            assert!(Rc::ptr_eq(&subtraction.right_operand(), &right));

            subtraction.forward();
            assert_is_precise_enough(
                &*subtraction.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![0., -1., -2., 0., -1., -2., 0., -1., -2., 0., -1., -2.],
                )
                .unwrap(),
            );
            assert_eq!(subtraction.was_computed(), true);
            assert!(Rc::ptr_eq(&subtraction.left_operand(), &left));
            assert!(Rc::ptr_eq(&subtraction.right_operand(), &right));

            subtraction.reset_computation();
            assert_eq!(subtraction.was_computed(), false);
        }
    }

    mod multiplication {
        use super::*;

        #[test]
        fn single_node() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![2.; 9]);

            let multiplication = Multiplication::new(left.clone(), right.clone());
            assert_eq!(*multiplication.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(multiplication.was_computed(), false);
            assert!(Rc::ptr_eq(&multiplication.left_operand(), &left));
            assert!(Rc::ptr_eq(&multiplication.right_operand(), &right));

            multiplication.forward();
            assert_is_precise_enough(
                &*multiplication.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 4., 6., 8., 10., 12., 14., 16., 18.])
                    .unwrap(),
            );
            assert_eq!(multiplication.was_computed(), true);
            assert!(Rc::ptr_eq(&multiplication.left_operand(), &left));
            assert!(Rc::ptr_eq(&multiplication.right_operand(), &right));

            multiplication.reset_computation();
            assert_eq!(multiplication.was_computed(), false);
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![2.; 9]);

            let first = Rc::new(Multiplication::new(left.clone(), right.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second = Multiplication::new(first.clone(), right.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 4., 6., 8., 10., 12., 14., 16., 18.])
                    .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 4., 6., 8., 10., 12., 14., 16., 18.])
                    .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![4., 8., 12., 16., 20., 24., 28., 32., 36.])
                    .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }

        #[test]
        fn left_broadcast() {
            let left = make_me_an_input((1, 3), vec![1., 2., 3.]);
            let right = make_me_an_input((2, 2, 3), vec![2.; 12]);

            let multiplication = Multiplication::new(left.clone(), right.clone());
            assert_eq!(*multiplication.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert_eq!(multiplication.was_computed(), false);
            assert!(Rc::ptr_eq(&multiplication.left_operand(), &left));
            assert!(Rc::ptr_eq(&multiplication.right_operand(), &right));

            multiplication.forward();
            assert_is_precise_enough(
                &*multiplication.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![2., 4., 6., 2., 4., 6., 2., 4., 6., 2., 4., 6.],
                )
                .unwrap(),
            );
            assert_eq!(multiplication.was_computed(), true);
            assert!(Rc::ptr_eq(&multiplication.left_operand(), &left));
            assert!(Rc::ptr_eq(&multiplication.right_operand(), &right));

            multiplication.reset_computation();
            assert_eq!(multiplication.was_computed(), false);
        }

        #[test]
        fn right_broadcast() {
            let left = make_me_an_input((2, 2, 3), vec![2.; 12]);
            let right = make_me_an_input((1, 3), vec![1., 2., 3.]);

            let multiplication = Multiplication::new(left.clone(), right.clone());
            assert_eq!(*multiplication.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert_eq!(multiplication.was_computed(), false);
            assert!(Rc::ptr_eq(&multiplication.left_operand(), &left));
            assert!(Rc::ptr_eq(&multiplication.right_operand(), &right));

            multiplication.forward();
            assert_is_precise_enough(
                &*multiplication.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![2., 4., 6., 2., 4., 6., 2., 4., 6., 2., 4., 6.],
                )
                .unwrap(),
            );
            assert_eq!(multiplication.was_computed(), true);
            assert!(Rc::ptr_eq(&multiplication.left_operand(), &left));
            assert!(Rc::ptr_eq(&multiplication.right_operand(), &right));

            multiplication.reset_computation();
            assert_eq!(multiplication.was_computed(), false);
        }
    }

    mod division {
        use super::*;

        #[test]
        fn single_node() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![2.; 9]);

            let division = Division::new(left.clone(), right.clone());
            assert_eq!(*division.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(division.was_computed(), false);
            assert!(Rc::ptr_eq(&division.left_operand(), &left));
            assert!(Rc::ptr_eq(&division.right_operand(), &right));

            division.forward();
            assert_is_precise_enough(
                &*division.data(),
                &Tensor::from_shape_vec((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])
                    .unwrap(),
            );
            assert_eq!(division.was_computed(), true);
            assert!(Rc::ptr_eq(&division.left_operand(), &left));
            assert!(Rc::ptr_eq(&division.right_operand(), &right));

            division.reset_computation();
            assert_eq!(division.was_computed(), false);
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![2.; 9]);

            let first = Rc::new(Division::new(left.clone(), right.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second = Division::new(first.clone(), right.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])
                    .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])
                    .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![0.25, 0.50, 0.75, 1., 1.25, 1.50, 1.75, 2., 2.25],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }

        #[test]
        fn left_broadcast() {
            let left = make_me_an_input((1, 3), vec![1., 2., 3.]);
            let right = make_me_an_input((2, 2, 3), vec![2.; 12]);

            let division = Division::new(left.clone(), right.clone());
            assert_eq!(*division.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert_eq!(division.was_computed(), false);
            assert!(Rc::ptr_eq(&division.left_operand(), &left));
            assert!(Rc::ptr_eq(&division.right_operand(), &right));

            division.forward();
            assert_is_precise_enough(
                &*division.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![0.5, 1., 1.5, 0.5, 1., 1.5, 0.5, 1., 1.5, 0.5, 1., 1.5],
                )
                .unwrap(),
            );
            assert_eq!(division.was_computed(), true);
            assert!(Rc::ptr_eq(&division.left_operand(), &left));
            assert!(Rc::ptr_eq(&division.right_operand(), &right));

            division.reset_computation();
            assert_eq!(division.was_computed(), false);
        }

        #[test]
        fn right_broadcast() {
            let left = make_me_an_input((2, 2, 3), vec![2.; 12]);
            let right = make_me_an_input((1, 3), vec![1., 2., 3.]);

            let division = Division::new(left.clone(), right.clone());
            assert_eq!(*division.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert_eq!(division.was_computed(), false);
            assert!(Rc::ptr_eq(&division.left_operand(), &left));
            assert!(Rc::ptr_eq(&division.right_operand(), &right));

            division.forward();
            assert_is_precise_enough(
                &*division.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![
                        2., 1., 0.6667, 2., 1., 0.6667, 2., 1., 0.6667, 2., 1., 0.6667,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(division.was_computed(), true);
            assert!(Rc::ptr_eq(&division.left_operand(), &left));
            assert!(Rc::ptr_eq(&division.right_operand(), &right));

            division.reset_computation();
            assert_eq!(division.was_computed(), false);
        }
    }

    mod power {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let power = Power::new(input.clone(), 3);
            assert_eq!(*power.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(power.was_computed(), false);
            assert!(Rc::ptr_eq(&power.operand(), &input));

            power.forward();
            assert_is_precise_enough(
                &*power.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![1., 8., 27., 64., 125., 216., 343., 512., 729.],
                )
                .unwrap(),
            );
            assert_eq!(power.was_computed(), true);
            assert!(Rc::ptr_eq(&power.operand(), &input));

            power.reset_computation();
            assert_eq!(power.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let first = Rc::new(Power::new(input.clone(), 3));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Power::new(first.clone(), 5);
            assert_eq!(second.was_computed(), false);
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![1., 8., 27., 64., 125., 216., 343., 512., 729.],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![1., 8., 27., 64., 125., 216., 343., 512., 729.],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        1.0000e+00, 3.2768e+04, 1.4349e+07, 1.0737e+09, 3.0518e+10, 4.7018e+11,
                        4.7476e+12, 3.5184e+13, 2.0589e+14,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod sum {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((1, 3, 1, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let sum = Sum::new(input.clone());
            assert_eq!(*sum.data(), Tensor::from_elem(1, 0.));
            assert_eq!(sum.was_computed(), false);
            assert!(Rc::ptr_eq(&sum.operand(), &input));

            sum.forward();
            assert_is_precise_enough(&*sum.data(), &Tensor::from_elem(1, 45.));
            assert_eq!(sum.was_computed(), true);
            assert!(Rc::ptr_eq(&sum.operand(), &input));

            sum.reset_computation();
            assert_eq!(sum.was_computed(), false);
        }
    }

    mod logn {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let logn = Logn::new(input.clone());
            assert_eq!(*logn.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(logn.was_computed(), false);
            assert!(Rc::ptr_eq(&logn.operand(), &input));

            logn.forward();
            assert_is_precise_enough(
                &*logn.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.0000, 0.6931, 1.0986, 1.3863, 1.6094, 1.7918, 1.9459, 2.0794, 2.1972,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(logn.was_computed(), true);
            assert!(Rc::ptr_eq(&logn.operand(), &input));

            logn.reset_computation();
            assert_eq!(logn.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let first = Rc::new(Logn::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Logn::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.0000, 0.6931, 1.0986, 1.3863, 1.6094, 1.7918, 1.9459, 2.0794, 2.1972,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.0000, 0.6931, 1.0986, 1.3863, 1.6094, 1.7918, 1.9459, 2.0794, 2.1972,
                    ],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        f32::NEG_INFINITY,
                        -0.3665,
                        0.0940,
                        0.3266,
                        0.4759,
                        0.5832,
                        0.6657,
                        0.7321,
                        0.7872,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod relu {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let relu = ReLU::new(input.clone());
            assert_eq!(*relu.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(relu.was_computed(), false);
            assert!(Rc::ptr_eq(&relu.operand(), &input));

            relu.forward();
            assert_is_precise_enough(
                &*relu.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 0., 0., 0., 0., 1., 2., 3., 4.]).unwrap(),
            );
            assert_eq!(relu.was_computed(), true);
            assert!(Rc::ptr_eq(&relu.operand(), &input));

            relu.reset_computation();
            assert_eq!(relu.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let first = Rc::new(ReLU::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = ReLU::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 0., 0., 0., 0., 1., 2., 3., 4.]).unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 0., 0., 0., 0., 1., 2., 3., 4.]).unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 0., 0., 0., 0., 1., 2., 3., 4.]).unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod leakyrelu {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let leaky_relu = LeakyReLU::new(input.clone());
            assert_eq!(*leaky_relu.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(leaky_relu.was_computed(), false);
            assert!(Rc::ptr_eq(&leaky_relu.operand(), &input));

            leaky_relu.forward();
            assert_is_precise_enough(
                &*leaky_relu.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![-0.04, -0.03, -0.02, -0.01, 0., 1., 2., 3., 4.],
                )
                .unwrap(),
            );
            assert_eq!(leaky_relu.was_computed(), true);
            assert!(Rc::ptr_eq(&leaky_relu.operand(), &input));

            leaky_relu.reset_computation();
            assert_eq!(leaky_relu.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let first = Rc::new(LeakyReLU::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = LeakyReLU::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![-0.04, -0.03, -0.02, -0.01, 0., 1., 2., 3., 4.],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![-0.04, -0.03, -0.02, -0.01, 0., 1., 2., 3., 4.],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![-4.0e-04, -3.0e-04, -2.0e-04, -1.0e-04, 0., 1., 2., 3., 4.],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod softplus {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let softplus = SoftPlus::new(input.clone());
            assert_eq!(*softplus.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(softplus.was_computed(), false);
            assert!(Rc::ptr_eq(&softplus.operand(), &input));

            softplus.forward();
            assert_is_precise_enough(
                &*softplus.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.01815, 0.04859, 0.12693, 0.31326, 0.69315, 1.31326, 2.12693, 3.04859,
                        4.01815,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(softplus.was_computed(), true);
            assert!(Rc::ptr_eq(&softplus.operand(), &input));

            softplus.reset_computation();
            assert_eq!(softplus.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let first = Rc::new(SoftPlus::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = SoftPlus::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.01815, 0.04859, 0.12693, 0.31326, 0.69315, 1.31326, 2.12693, 3.04859,
                        4.01815,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.01815, 0.04859, 0.12693, 0.31326, 0.69315, 1.31326, 2.12693, 3.04859,
                        4.01815,
                    ],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.70226, 0.71774, 0.75862, 0.86199, 1.09861, 1.55144, 2.23954, 3.09492,
                        4.03598,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod sigmoid {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let sigmoid = Sigmoid::new(input.clone());
            assert_eq!(*sigmoid.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(sigmoid.was_computed(), false);
            assert!(Rc::ptr_eq(&sigmoid.operand(), &input));

            sigmoid.forward();
            assert_is_precise_enough(
                &*sigmoid.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.01799, 0.04743, 0.11920, 0.26894, 0.5, 0.73106, 0.88080, 0.95257, 0.98201,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(sigmoid.was_computed(), true);
            assert!(Rc::ptr_eq(&sigmoid.operand(), &input));

            sigmoid.reset_computation();
            assert_eq!(sigmoid.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let first = Rc::new(Sigmoid::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Sigmoid::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.01799, 0.04743, 0.11920, 0.26894, 0.5, 0.73106, 0.88080, 0.95257, 0.98201,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.01799, 0.04743, 0.11920, 0.26894, 0.5, 0.73106, 0.88080, 0.95257, 0.98201,
                    ],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.50450, 0.51185, 0.52977, 0.56683, 0.62246, 0.67504, 0.70699, 0.72163,
                        0.72751,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod tanh {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let tanh = TanH::new(input.clone());
            assert_eq!(*tanh.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(tanh.was_computed(), false);
            assert!(Rc::ptr_eq(&tanh.operand(), &input));

            tanh.forward();
            assert_is_precise_enough(
                &*tanh.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        -0.99933, -0.99505, -0.96403, -0.76159, 0., 0.76159, 0.96403, 0.99505,
                        0.99933,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(tanh.was_computed(), true);
            assert!(Rc::ptr_eq(&tanh.operand(), &input));

            tanh.reset_computation();
            assert_eq!(tanh.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let first = Rc::new(TanH::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = TanH::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        -0.99933, -0.99505, -0.96403, -0.76159, 0., 0.76159, 0.96403, 0.99505,
                        0.99933,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        -0.99933, -0.99505, -0.96403, -0.76159, 0., 0.76159, 0.96403, 0.99505,
                        0.99933,
                    ],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        -0.76131, -0.75951, -0.74607, -0.64201, 0., 0.64201, 0.74607, 0.75951,
                        0.76131,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod exp {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let exp = Exp::new(input.clone());
            assert_eq!(*exp.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(exp.was_computed(), false);
            assert!(Rc::ptr_eq(&exp.operand(), &input));

            exp.forward();
            assert_is_precise_enough(
                &*exp.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        1.83156e-02,
                        4.97871e-02,
                        1.35335e-01,
                        3.67879e-01,
                        1.00000e+00,
                        2.71828e+00,
                        7.38906e+00,
                        2.00855e+01,
                        5.45981e+01,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(exp.was_computed(), true);
            assert!(Rc::ptr_eq(&exp.operand(), &input));

            exp.reset_computation();
            assert_eq!(exp.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let first = Rc::new(Exp::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Exp::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        1.83156e-02,
                        4.97871e-02,
                        1.35335e-01,
                        3.67879e-01,
                        1.00000e+00,
                        2.71828e+00,
                        7.38906e+00,
                        2.00855e+01,
                        5.45981e+01,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        1.83156e-02,
                        4.97871e-02,
                        1.35335e-01,
                        3.67879e-01,
                        1.00000e+00,
                        2.71828e+00,
                        7.38906e+00,
                        2.00855e+01,
                        5.45981e+01,
                    ],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        1.01848e+00,
                        1.05105e+00,
                        1.14492e+00,
                        1.44467e+00,
                        2.71828e+00,
                        1.51543e+01,
                        1.61818e+03,
                        5.28491e+08,
                        5.14843e+23,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod softmax {
        use super::*;

        #[test]
        fn single_node_rows() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let softmax = Softmax::new(input.clone(), 0);
            assert_eq!(*softmax.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(softmax.was_computed(), false);
            assert!(Rc::ptr_eq(&softmax.operand(), &input));

            softmax.forward();
            assert_is_precise_enough(
                &*softmax.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330,
                        0.950330, 0.950330,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(softmax.was_computed(), true);
            assert!(Rc::ptr_eq(&softmax.operand(), &input));

            softmax.reset_computation();
            assert_eq!(softmax.was_computed(), false);
        }

        #[test]
        fn single_node_columns() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let softmax = Softmax::new(input.clone(), 1);
            assert_eq!(*softmax.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(softmax.was_computed(), false);
            assert!(Rc::ptr_eq(&softmax.operand(), &input));

            softmax.forward();
            assert_is_precise_enough(
                &*softmax.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.09003, 0.24473, 0.66524, 0.09003, 0.24473, 0.66524, 0.09003, 0.24473,
                        0.66524,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(softmax.was_computed(), true);
            assert!(Rc::ptr_eq(&softmax.operand(), &input));

            softmax.reset_computation();
            assert_eq!(softmax.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let first = Rc::new(Softmax::new(input.clone(), 0));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Softmax::new(first.clone(), 1);
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330,
                        0.950330, 0.950330,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330,
                        0.950330, 0.950330,
                    ],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        0.333333, 0.333333, 0.333333, 0.333333, 0.333333, 0.333333, 0.333333,
                        0.333333, 0.333333,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod logsoftmax {
        use super::*;

        #[test]
        fn single_node_rows() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let log_softmax = LogSoftmax::new(input.clone(), 0);
            assert_eq!(*log_softmax.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(log_softmax.was_computed(), false);
            assert!(Rc::ptr_eq(&log_softmax.operand(), &input));

            log_softmax.forward();
            assert_is_precise_enough(
                &*log_softmax.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        -6.050946, -6.050946, -6.050946, -3.050946, -3.050946, -3.050946,
                        -0.050946, -0.050946, -0.050946,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(log_softmax.was_computed(), true);
            assert!(Rc::ptr_eq(&log_softmax.operand(), &input));

            log_softmax.reset_computation();
            assert_eq!(log_softmax.was_computed(), false);
        }

        #[test]
        fn single_node_columns() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let softmax = LogSoftmax::new(input.clone(), 1);
            assert_eq!(*softmax.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(softmax.was_computed(), false);
            assert!(Rc::ptr_eq(&softmax.operand(), &input));

            softmax.forward();
            assert_is_precise_enough(
                &*softmax.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        -2.407606, -1.407606, -0.407606, -2.407606, -1.407606, -0.407606,
                        -2.407606, -1.407606, -0.407606,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(softmax.was_computed(), true);
            assert!(Rc::ptr_eq(&softmax.operand(), &input));

            softmax.reset_computation();
            assert_eq!(softmax.was_computed(), false);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let first = Rc::new(LogSoftmax::new(input.clone(), 0));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = LogSoftmax::new(first.clone(), 1);
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        -6.050946, -6.050946, -6.050946, -3.050946, -3.050946, -3.050946,
                        -0.050946, -0.050946, -0.050946,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        -6.050946, -6.050946, -6.050946, -3.050946, -3.050946, -3.050946,
                        -0.050946, -0.050946, -0.050946,
                    ],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![
                        -1.098612, -1.098612, -1.098612, -1.098612, -1.098612, -1.098612,
                        -1.098612, -1.098612, -1.098612,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod concatenate {
        use super::*;

        #[test]
        fn single_node_rows() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((2, 3), vec![1.; 6]);

            let concatenate = Concatenate::new(left.clone(), right.clone(), 0);
            assert_eq!(*concatenate.data(), Tensor::from_elem((5, 3), 0.));
            assert_eq!(concatenate.was_computed(), false);
            assert!(Rc::ptr_eq(&concatenate.left_operand(), &left));
            assert!(Rc::ptr_eq(&concatenate.right_operand(), &right));

            concatenate.forward();
            assert_is_precise_enough(
                &*concatenate.data(),
                &Tensor::from_shape_vec(
                    (5, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1.,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(concatenate.was_computed(), true);
            assert!(Rc::ptr_eq(&concatenate.left_operand(), &left));
            assert!(Rc::ptr_eq(&concatenate.right_operand(), &right));

            concatenate.reset_computation();
            assert_eq!(concatenate.was_computed(), false);
        }

        #[test]
        fn single_node_columns() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((3, 2), vec![1.; 6]);

            let concatenate = Concatenate::new(left.clone(), right.clone(), 1);
            assert_eq!(*concatenate.data(), Tensor::from_elem((3, 5), 0.));
            assert_eq!(concatenate.was_computed(), false);
            assert!(Rc::ptr_eq(&concatenate.left_operand(), &left));
            assert!(Rc::ptr_eq(&concatenate.right_operand(), &right));

            concatenate.forward();
            assert_is_precise_enough(
                &*concatenate.data(),
                &Tensor::from_shape_vec(
                    (3, 5),
                    vec![
                        -4., -3., -2., 1., 1., -1., 0., 1., 1., 1., 2., 3., 4., 1., 1.,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(concatenate.was_computed(), true);
            assert!(Rc::ptr_eq(&concatenate.left_operand(), &left));
            assert!(Rc::ptr_eq(&concatenate.right_operand(), &right));

            concatenate.reset_computation();
            assert_eq!(concatenate.was_computed(), false);
        }

        #[test]
        #[should_panic]
        fn cannot_concatenate_by_rows() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((3, 2), vec![1.; 6]);

            Concatenate::new(left, right, 0);
        }

        #[test]
        #[should_panic]
        fn cannot_concatenate_by_columns() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((2, 3), vec![1.; 6]);

            Concatenate::new(left, right, 1);
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((2, 3), vec![1.; 6]);

            let first = Rc::new(Concatenate::new(left.clone(), right.clone(), 0));
            assert_eq!(*first.data(), Tensor::from_elem((5, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second_right = make_me_an_input((5, 2), vec![1.; 10]);

            let second = Concatenate::new(first.clone(), second_right.clone(), 1);
            assert_eq!(*second.data(), Tensor::from_elem((5, 5), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &second_right));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (5, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1.,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((5, 5), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &second_right));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (5, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1.,
                    ],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (5, 5),
                    vec![
                        -4., -3., -2., 1., 1., -1., 0., 1., 1., 1., 2., 3., 4., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1.,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &second_right));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod stack {
        use super::*;

        #[test]
        fn single_node_rows() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((3, 3), vec![0.; 9]);

            let stack = Stack::new(left.clone(), right.clone(), 0);
            assert_eq!(*stack.data(), Tensor::from_elem((2, 3, 3), 0.));
            assert_eq!(stack.was_computed(), false);
            assert!(Rc::ptr_eq(&stack.left_operand(), &left));
            assert!(Rc::ptr_eq(&stack.right_operand(), &right));

            stack.forward();
            assert_is_precise_enough(
                &*stack.data(),
                &Tensor::from_shape_vec(
                    (2, 3, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(stack.was_computed(), true);
            assert!(Rc::ptr_eq(&stack.left_operand(), &left));
            assert!(Rc::ptr_eq(&stack.right_operand(), &right));

            stack.reset_computation();
            assert_eq!(stack.was_computed(), false);
        }

        #[test]
        fn single_node_columns() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((3, 3), vec![0.; 9]);

            let stack = Stack::new(left.clone(), right.clone(), 1);
            assert_eq!(*stack.data(), Tensor::from_elem((3, 2, 3), 0.));
            assert_eq!(stack.was_computed(), false);
            assert!(Rc::ptr_eq(&stack.left_operand(), &left));
            assert!(Rc::ptr_eq(&stack.right_operand(), &right));

            stack.forward();
            assert_is_precise_enough(
                &*stack.data(),
                &Tensor::from_shape_vec(
                    (3, 2, 3),
                    vec![
                        -4., -3., -2., 0., 0., 0., -1., 0., 1., 0., 0., 0., 2., 3., 4., 0., 0., 0.,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(stack.was_computed(), true);
            assert!(Rc::ptr_eq(&stack.left_operand(), &left));
            assert!(Rc::ptr_eq(&stack.right_operand(), &right));

            stack.reset_computation();
            assert_eq!(stack.was_computed(), false);
        }

        #[test]
        #[should_panic]
        fn cannot_stack_by_rows() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((3, 2), vec![1.; 6]);

            Stack::new(left, right, 0);
        }

        #[test]
        #[should_panic]
        fn cannot_stack_by_columns() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((2, 3), vec![1.; 6]);

            Stack::new(left, right, 1);
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = make_me_an_input((3, 3), vec![0.; 9]);

            let first = Rc::new(Stack::new(left.clone(), right.clone(), 0));
            assert_eq!(*first.data(), Tensor::from_elem((2, 3, 3), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second_right = make_me_an_input((2, 3, 3), vec![0.; 18]);

            let second = Stack::new(first.clone(), second_right.clone(), 2);
            assert_eq!(*second.data(), Tensor::from_elem((2, 3, 2, 3), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &second_right));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (2, 3, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((2, 3, 2, 3), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &second_right));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec(
                    (2, 3, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                )
                .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec(
                    (2, 3, 2, 3),
                    vec![
                        -4., -3., -2., 0., 0., 0., -1., 0., 1., 0., 0., 0., 2., 3., 4., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                )
                .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &second_right));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }

    mod unsqueeze {
        use super::*;

        #[test]
        fn single_node_rows() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let unsqueeze = Unsqueeze::new(input.clone(), 0);
            assert_eq!(*unsqueeze.data(), Tensor::from_elem((1, 3, 3), 0.));
            assert_eq!(unsqueeze.was_computed(), false);
            assert!(Rc::ptr_eq(&unsqueeze.operand(), &input));

            unsqueeze.forward();
            assert_is_precise_enough(
                &*unsqueeze.data(),
                &Tensor::from_shape_vec((1, 3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.])
                    .unwrap(),
            );
            assert_eq!(unsqueeze.was_computed(), true);
            assert!(Rc::ptr_eq(&unsqueeze.operand(), &input));

            unsqueeze.reset_computation();
            assert_eq!(unsqueeze.was_computed(), false);
        }

        #[test]
        fn single_node_columns() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let unsqueeze = Unsqueeze::new(input.clone(), 1);
            assert_eq!(*unsqueeze.data(), Tensor::from_elem((3, 1, 3), 0.));
            assert_eq!(unsqueeze.was_computed(), false);
            assert!(Rc::ptr_eq(&unsqueeze.operand(), &input));

            unsqueeze.forward();
            assert_is_precise_enough(
                &*unsqueeze.data(),
                &Tensor::from_shape_vec((3, 1, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.])
                    .unwrap(),
            );
            assert_eq!(unsqueeze.was_computed(), true);
            assert!(Rc::ptr_eq(&unsqueeze.operand(), &input));

            unsqueeze.reset_computation();
            assert_eq!(unsqueeze.was_computed(), false);
        }

        #[test]
        #[should_panic]
        fn cannot_unsqueeze() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            Unsqueeze::new(input, 3);
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);

            let first = Rc::new(Unsqueeze::new(input.clone(), 2));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3, 1), 0.));
            assert_eq!(first.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Unsqueeze::new(first.clone(), 0);
            assert_eq!(*second.data(), Tensor::from_elem((1, 3, 3, 1), 0.));
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3, 1), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.])
                    .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((1, 3, 3, 1), 0.));
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), false);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3, 1), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.])
                    .unwrap(),
            );
            assert_is_precise_enough(
                &*second.data(),
                &Tensor::from_shape_vec((1, 3, 3, 1), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.])
                    .unwrap(),
            );
            assert_eq!(first.was_computed(), true);
            assert_eq!(second.was_computed(), true);
            assert!(Rc::ptr_eq(&first.operand(), &input));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.reset_computation();
            second.reset_computation();
            assert_eq!(first.was_computed(), false);
            assert_eq!(second.was_computed(), false);
        }
    }
}
