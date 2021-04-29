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

    fn assert_almost_equals<D: Dimension>(our: &Tensor<D>, their: &Tensor<D>) {
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

    fn new_input<D, Sh>(shape: Sh, elems: Vec<f32>) -> Rc<Input<D>>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Input::new(new_tensor(shape, elems)).forward
    }

    fn new_tensor<D, Sh>(shape: Sh, elems: Vec<f32>) -> Tensor<D>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Tensor::from_shape_vec(shape, elems).unwrap()
    }

    mod negation {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Negation::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Negation::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Negation::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![3., 2., 1., 0., -1., -2., -3., -4., -5.]),
            );
        }
    }

    mod transpose {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Transpose::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Transpose::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Transpose::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![2., 5., 8., 3., 6., 9., 4., 7., 10.]),
            );
        }
    }

    mod addition {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = Addition::new(left.clone(), right.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = Addition::new(left.clone(), right.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn forward() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = Addition::new(left.clone(), right);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = left.data_mut();
                *data = &*data + &Tensor::from_elem(1, 10.);
            }
            assert_almost_equals(
                &*left.data(),
                &new_tensor((3, 3), vec![11., 12., 13., 14., 15., 16., 17., 18., 19.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![12., 13., 14., 15., 16., 17., 18., 19., 20.]),
            );
        }

        #[test]
        fn left_broadcast_forward() {
            let left = new_input((1, 3), vec![1., 2., 3.]);
            let right = new_input((2, 2, 3), vec![1.; 12]);
            let node = Addition::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 2, 3),
                    vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.],
                ),
            );
        }

        #[test]
        fn right_broadcast_forward() {
            let left = new_input((2, 2, 3), vec![1.; 12]);
            let right = new_input((1, 3), vec![1., 2., 3.]);
            let node = Addition::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 2, 3),
                    vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.],
                ),
            );
        }
    }

    mod subtraction {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = Subtraction::new(left.clone(), right.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = Subtraction::new(left.clone(), right.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn forward() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = Subtraction::new(left.clone(), right);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![-5., -4., -3., -2., -1., 0., 1., 2., 3.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = left.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*left.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![-5., -4., -3., -2., -1., 0., 1., 2., 3.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );
        }

        #[test]
        fn left_broadcast_forward() {
            let left = new_input((1, 3), vec![-1., 0., 1.]);
            let right = new_input((2, 2, 3), vec![1.; 12]);
            let node = Subtraction::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 2, 3),
                    vec![-2., -1., 0., -2., -1., 0., -2., -1., 0., -2., -1., 0.],
                ),
            );
        }

        #[test]
        fn right_broadcast_forward() {
            let left = new_input((2, 2, 3), vec![1.; 12]);
            let right = new_input((1, 3), vec![-1., 0., 1.]);
            let node = Subtraction::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 2, 3),
                    vec![2., 1., 0., 2., 1., 0., 2., 1., 0., 2., 1., 0.],
                ),
            );
        }
    }

    mod multiplication {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![-1.; 9]);
            let node = Multiplication::new(left.clone(), right.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![-1.; 9]);
            let node = Multiplication::new(left.clone(), right.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn forward() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![-1.; 9]);
            let node = Multiplication::new(left, right.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            *right.data_mut() = new_tensor((3, 3), vec![2.; 9]);
            assert_almost_equals(&*right.data(), &new_tensor((3, 3), vec![2.; 9]));

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![-8., -6., -4., -2., 0., 2., 4., 6., 8.]),
            );
        }

        #[test]
        fn left_broadcast_forward() {
            let left = new_input((1, 3), vec![-1., 0., 1.]);
            let right = new_input((2, 2, 3), vec![-2.; 12]);
            let node = Multiplication::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 2, 3),
                    vec![2., 0., -2., 2., 0., -2., 2., 0., -2., 2., 0., -2.],
                ),
            );
        }

        #[test]
        fn right_broadcast_forward() {
            let left = new_input((2, 2, 3), vec![-2.; 12]);
            let right = new_input((1, 3), vec![-1., 0., 1.]);
            let node = Multiplication::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 2, 3),
                    vec![2., 0., -2., 2., 0., -2., 2., 0., -2., 2., 0., -2.],
                ),
            );
        }
    }

    mod division {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![2.; 9]);
            let node = Division::new(left.clone(), right.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![2.; 9]);
            let node = Division::new(left.clone(), right.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn forward() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![2.; 9]);
            let node = Division::new(left, right.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            *right.data_mut() = new_tensor((3, 3), vec![-2.; 9]);
            assert_almost_equals(&*right.data(), &new_tensor((3, 3), vec![-2.; 9]));

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![-0.5, -1., -1.5, -2., -2.5, -3., -3.5, -4., -4.5],
                ),
            );
        }

        #[test]
        fn left_broadcast_forward() {
            let left = new_input((1, 3), vec![1., 2., 3.]);
            let right = new_input((2, 2, 3), vec![2.; 12]);
            let node = Division::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 2, 3),
                    vec![0.5, 1., 1.5, 0.5, 1., 1.5, 0.5, 1., 1.5, 0.5, 1., 1.5],
                ),
            );
        }

        #[test]
        fn right_broadcast_forward() {
            let left = new_input((2, 2, 3), vec![2.; 12]);
            let right = new_input((1, 3), vec![1., 2., 3.]);
            let node = Division::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 2, 3),
                    vec![
                        2., 1., 0.6667, 2., 1., 0.6667, 2., 1., 0.6667, 2., 1., 0.6667,
                    ],
                ),
            );
        }
    }

    mod matrixmatrixmul {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = MatrixMatrixMul::new(left.clone(), right.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = MatrixMatrixMul::new(left.clone(), right.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn forward() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = MatrixMatrixMul::new(left, right.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![6., 6., 6., 15., 15., 15., 24., 24., 24.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            *right.data_mut() = new_tensor((3, 3), vec![-2.; 9]);
            assert_almost_equals(&*right.data(), &new_tensor((3, 3), vec![-2.; 9]));

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![6., 6., 6., 15., 15., 15., 24., 24., 24.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![-12., -12., -12., -30., -30., -30., -48., -48., -48.],
                ),
            );
        }
    }

    mod matrixvectormul {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input(3, vec![1.; 3]);
            let node = MatrixVectorMul::new(left.clone(), right.clone());

            assert_eq!(*node.data(), Tensor::from_elem(3, 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input(3, vec![1.; 3]);
            let node = MatrixVectorMul::new(left.clone(), right.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn forward() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input(3, vec![1.; 3]);
            let node = MatrixVectorMul::new(left, right.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(3, vec![6., 15., 24.]));

            // ----------------------------------- No Second Evaluation -----------------------------------
            *right.data_mut() = new_tensor(3, vec![-2.; 3]);
            assert_almost_equals(&*right.data(), &new_tensor(3, vec![-2.; 3]));

            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(3, vec![6., 15., 24.]));

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(3, vec![-12., -30., -48.]));
        }
    }

    mod vectormatrixmul {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input(3, vec![1.; 3]);
            let right = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = VectorMatrixMul::new(left.clone(), right.clone());

            assert_eq!(*node.data(), Tensor::from_elem(3, 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input(3, vec![1.; 3]);
            let right = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = VectorMatrixMul::new(left.clone(), right.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn forward() {
            let left = new_input(3, vec![1.; 3]);
            let right = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = VectorMatrixMul::new(left.clone(), right);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(3, vec![12., 15., 18.]));

            // ----------------------------------- No Second Evaluation -----------------------------------
            *left.data_mut() = new_tensor(3, vec![-2.; 3]);
            assert_almost_equals(&*left.data(), &new_tensor(3, vec![-2.; 3]));

            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(3, vec![12., 15., 18.]));

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(3, vec![-24., -30., -36.]));
        }
    }

    mod vectorvectormul {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input(3, vec![2.; 3]);
            let right = new_input(3, vec![1., 2., 3.]);
            let node = VectorVectorMul::new(left.clone(), right.clone());

            assert_eq!(*node.data(), Tensor::from_elem(1, 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input(3, vec![2.; 3]);
            let right = new_input(3, vec![1., 2., 3.]);
            let node = VectorVectorMul::new(left.clone(), right.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn forward() {
            let left = new_input(3, vec![2.; 3]);
            let right = new_input(3, vec![1., 2., 3.]);
            let node = VectorVectorMul::new(left.clone(), right);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![12.]));

            // ----------------------------------- No Second Evaluation -----------------------------------
            *left.data_mut() = new_tensor(3, vec![-2.; 3]);
            assert_almost_equals(&*left.data(), &new_tensor(3, vec![-2.; 3]));

            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![12.]));

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![-12.]));
        }
    }

    mod power {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Power::new(input.clone(), 2);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Power::new(input.clone(), 2);

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Power::new(input.clone(), 3);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![1., 8., 27., 64., 125., 216., 343., 512., 729.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![1., 8., 27., 64., 125., 216., 343., 512., 729.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![8., 27., 64., 125., 216., 343., 512., 729., 1_000.],
                ),
            );
        }
    }

    mod sum {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Sum::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem(1, 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Sum::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Sum::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![45.]));

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
            );

            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![45.]));

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![54.]));
        }
    }

    mod logn {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Logn::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Logn::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Logn::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0., 0.6931, 1.0986, 1.3863, 1.6094, 1.7918, 1.9459, 2.0794, 2.1972,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0., 0.6931, 1.0986, 1.3863, 1.6094, 1.7918, 1.9459, 2.0794, 2.1972,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.6931, 1.0986, 1.3863, 1.6094, 1.7918, 1.9459, 2.0794, 2.1972, 2.3026,
                    ],
                ),
            );
        }
    }

    mod relu {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = ReLU::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = ReLU::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = ReLU::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![0., 0., 0., 0., 0., 1., 2., 3., 4.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![0., 0., 0., 0., 0., 1., 2., 3., 4.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![0., 0., 0., 0., 1., 2., 3., 4., 5.]),
            );
        }
    }

    mod leakyrelu {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = LeakyReLU::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = LeakyReLU::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = LeakyReLU::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![-0.04, -0.03, -0.02, -0.01, 0., 1., 2., 3., 4.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![-0.04, -0.03, -0.02, -0.01, 0., 1., 2., 3., 4.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![-0.03, -0.02, -0.01, 0., 1., 2., 3., 4., 5.]),
            );
        }
    }

    mod softplus {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = SoftPlus::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = SoftPlus::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = SoftPlus::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.01815, 0.04859, 0.12693, 0.31326, 0.69315, 1.31326, 2.12693, 3.04859,
                        4.01815,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.01815, 0.04859, 0.12693, 0.31326, 0.69315, 1.31326, 2.12693, 3.04859,
                        4.01815,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.048587, 0.126928, 0.313262, 0.693147, 1.313262, 2.126928, 3.048587,
                        4.01815, 5.006715,
                    ],
                ),
            );
        }
    }

    mod sigmoid {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Sigmoid::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Sigmoid::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Sigmoid::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.01799, 0.04743, 0.1192, 0.26894, 0.5, 0.73106, 0.8808, 0.95257, 0.98201,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.01799, 0.04743, 0.1192, 0.26894, 0.5, 0.73106, 0.8808, 0.95257, 0.98201,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.04743, 0.1192, 0.26894, 0.5, 0.73106, 0.8808, 0.95257, 0.98201, 0.993307,
                    ],
                ),
            );
        }
    }

    mod tanh {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = TanH::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = TanH::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = TanH::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.99933, -0.99505, -0.96403, -0.76159, 0., 0.76159, 0.96403, 0.99505,
                        0.99933,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.99933, -0.99505, -0.96403, -0.76159, 0., 0.76159, 0.96403, 0.99505,
                        0.99933,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.99505, -0.96403, -0.76159, 0., 0.76159, 0.96403, 0.99505, 0.99933,
                        0.999909,
                    ],
                ),
            );
        }
    }

    mod exp {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Exp::new(input.clone());

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Exp::new(input.clone());

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Exp::new(input.clone());

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
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
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
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
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        4.97871e-02,
                        1.35335e-01,
                        3.67879e-01,
                        1.00000e+00,
                        2.71828e+00,
                        7.38906e+00,
                        2.00855e+01,
                        5.45981e+01,
                        1.48413e+02,
                    ],
                ),
            );
        }
    }

    mod softmax {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Softmax::new(input.clone(), 0);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Softmax::new(input.clone(), 0);

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward_rows() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Softmax::new(input.clone(), 0);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330,
                        0.950330, 0.950330,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330,
                        0.950330, 0.950330,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330,
                        0.950330, 0.950330,
                    ],
                ),
            );
        }

        #[test]
        fn forward_columns() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Softmax::new(input.clone(), 1);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241, 0.090031,
                        0.244728, 0.665241,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241, 0.090031,
                        0.244728, 0.665241,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241, 0.090031,
                        0.244728, 0.665241,
                    ],
                ),
            );
        }
    }

    mod logsoftmax {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = LogSoftmax::new(input.clone(), 0);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = LogSoftmax::new(input.clone(), 0);

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn forward_rows() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = LogSoftmax::new(input.clone(), 0);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -6.050946, -6.050946, -6.050946, -3.050946, -3.050946, -3.050946,
                        -0.050946, -0.050946, -0.050946,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -6.050946, -6.050946, -6.050946, -3.050946, -3.050946, -3.050946,
                        -0.050946, -0.050946, -0.050946,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -6.0509, -6.0509, -6.0509, -3.0509, -3.0509, -3.0509, -0.0509, -0.0509,
                        -0.0509,
                    ],
                ),
            );
        }

        #[test]
        fn forward_columns() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = LogSoftmax::new(input.clone(), 1);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -2.407606, -1.407606, -0.407606, -2.407606, -1.407606, -0.407606,
                        -2.407606, -1.407606, -0.407606,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -2.407606, -1.407606, -0.407606, -2.407606, -1.407606, -0.407606,
                        -2.407606, -1.407606, -0.407606,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -2.4076, -1.4076, -0.4076, -2.4076, -1.4076, -0.4076, -2.4076, -1.4076,
                        -0.4076,
                    ],
                ),
            );
        }
    }

    mod concatenate {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((2, 3), vec![1.; 6]);
            let node = Concatenate::new(left.clone(), right.clone(), 0);

            assert_eq!(*node.data(), Tensor::from_elem((5, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((2, 3), vec![1.; 6]);
            let node = Concatenate::new(left.clone(), right.clone(), 0);

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        #[should_panic]
        fn fail_by_rows() {
            Concatenate::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                new_input((3, 2), vec![1.; 6]),
                0,
            );
        }

        #[test]
        #[should_panic]
        fn fail_by_columns() {
            Concatenate::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                new_input((2, 3), vec![1.; 6]),
                1,
            );
        }

        #[test]
        fn forward_rows() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((2, 3), vec![1.; 6]);
            let node = Concatenate::new(left.clone(), right, 0);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (5, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1.,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = left.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*left.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (5, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1.,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (5, 3),
                    vec![
                        -3., -2., -1., 0., 1., 2., 3., 4., 5., 1., 1., 1., 1., 1., 1.,
                    ],
                ),
            );
        }

        #[test]
        fn forward_columns() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 2), vec![1.; 6]);
            let node = Concatenate::new(left.clone(), right, 1);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 5),
                    vec![
                        -4., -3., -2., 1., 1., -1., 0., 1., 1., 1., 2., 3., 4., 1., 1.,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = left.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*left.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 5),
                    vec![
                        -4., -3., -2., 1., 1., -1., 0., 1., 1., 1., 2., 3., 4., 1., 1.,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 5),
                    vec![
                        -3., -2., -1., 1., 1., 0., 1., 2., 1., 1., 3., 4., 5., 1., 1.,
                    ],
                ),
            );
        }
    }

    mod stack {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![0.; 9]);
            let node = Stack::new(left.clone(), right.clone(), 0);

            assert_eq!(*node.data(), Tensor::from_elem((2, 3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        fn computation_state_transition() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![0.; 9]);
            let node = Stack::new(left.clone(), right.clone(), 0);

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.left_operand(), &left));
            assert!(Rc::ptr_eq(&node.right_operand(), &right));
        }

        #[test]
        #[should_panic]
        fn fail_by_rows() {
            Stack::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                new_input((3, 2), vec![0.; 6]),
                0,
            );
        }

        #[test]
        #[should_panic]
        fn fail_by_columns() {
            Stack::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                new_input((2, 3), vec![0.; 6]),
                1,
            );
        }

        #[test]
        fn forward_rows() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![0.; 9]);
            let node = Stack::new(left.clone(), right, 0);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 3, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = left.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*left.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 3, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 3, 3),
                    vec![
                        -3., -2., -1., 0., 1., 2., 3., 4., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                ),
            );
        }

        #[test]
        fn forward_columns() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![0.; 9]);
            let node = Stack::new(left.clone(), right, 1);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 2, 3),
                    vec![
                        -4., -3., -2., 0., 0., 0., -1., 0., 1., 0., 0., 0., 2., 3., 4., 0., 0., 0.,
                    ],
                ),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = left.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*left.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 2, 3),
                    vec![
                        -4., -3., -2., 0., 0., 0., -1., 0., 1., 0., 0., 0., 2., 3., 4., 0., 0., 0.,
                    ],
                ),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 2, 3),
                    vec![
                        -3., -2., -1., 0., 0., 0., 0., 1., 2., 0., 0., 0., 3., 4., 5., 0., 0., 0.,
                    ],
                ),
            );
        }
    }

    mod unsqueeze {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Unsqueeze::new(input.clone(), 0);

            assert_eq!(*node.data(), Tensor::from_elem((1, 3, 3), 0.));
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        fn computation_state_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Unsqueeze::new(input.clone(), 0);

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.forward();
            assert_eq!(node.was_computed(), true);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));

            node.reset_computation();
            assert_eq!(node.was_computed(), false);
            assert!(Rc::ptr_eq(&node.operand(), &input));
        }

        #[test]
        #[should_panic]
        fn fail() {
            Unsqueeze::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                3,
            );
        }

        #[test]
        fn forward_rows() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Unsqueeze::new(input.clone(), 0);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((1, 3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((1, 3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((1, 3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );
        }

        #[test]
        fn forward_columns() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Unsqueeze::new(input.clone(), 1);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 1, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 1, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 1, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );
        }

        #[test]
        fn forward_depths() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Unsqueeze::new(input.clone(), 2);

            // ------------------------------------- First Evaluation -------------------------------------
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3, 1), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );

            // ----------------------------------- No Second Evaluation -----------------------------------
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3, 1), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );

            // ------------------------------------- Second Evaluation -------------------------------------
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3, 1), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );
        }
    }
}
