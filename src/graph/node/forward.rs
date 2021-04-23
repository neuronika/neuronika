use super::{
    super::{BroadTensor, Broadcasted, Tensor, Var},
    broadcasted_zeros, DotDim,
};
use ndarray::{
    concatenate,
    linalg::{general_mat_mul, general_mat_vec_mul},
    stack, Axis, DimMax, Dimension, Ix1, Ix2, RemoveAxis, Zip,
};
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::rc::Rc;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub trait Data {
    type Dim: Dimension;
    fn data(&self) -> Ref<Tensor<Self::Dim>>;
}

pub trait Forward {
    fn forward(&self) -> bool;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Input<D>
where
    D: Dimension,
{
    data: RefCell<Tensor<D>>,
}

impl<D> Input<D>
where
    D: Dimension,
{
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
    fn forward(&self) -> bool {
        false
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Negation<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T> Negation<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for Negation<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = -o);
        true
    }
}

impl<T> Data for Negation<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Transpose ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Transpose<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T> Transpose<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().t().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for Transpose<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(self.op.data().t())
            .par_for_each(|v, o| *v = *o);
        true
    }
}

impl<T> Data for Transpose<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Addition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Addition<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> Addition<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(broadcasted_zeros(&left.data(), &right.data()));
        Self {
            left,
            right,
            data,
            was_computed: Cell::new(false),
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
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Addition<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .par_for_each(|v, l, r| *v = l + r);
        true
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Subtraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Subtraction<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> Subtraction<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(broadcasted_zeros(&left.data(), &right.data()));
        Self {
            left,
            right,
            data,
            was_computed: Cell::new(false),
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
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Subtraction<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .par_for_each(|v, l, r| *v = l - r);
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Multiplication<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> Multiplication<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(broadcasted_zeros(&left.data(), &right.data()));
        Self {
            left,
            right,
            data,
            was_computed: Cell::new(false),
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
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Multiplication<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .par_for_each(|v, l, r| *v = l * r);
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Division ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Division<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> Division<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let data = RefCell::new(broadcasted_zeros(&left.data(), &right.data()));
        Self {
            left,
            right,
            data,
            was_computed: Cell::new(false),
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
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for Division<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .par_for_each(|v, l, r| *v = l / r);
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix2>>,
    was_computed: Cell<bool>,
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
            was_computed: Cell::new(false),
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
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    type Dim = Ix2;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        general_mat_mul(
            1.0,
            &*self.left.data(),
            &*self.right.data(),
            0.0,
            &mut *self.data.borrow_mut(),
        );
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix1>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix1>>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix1>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(left.data().raw_dim(), right.data().raw_dim());
        let data = RefCell::new(Tensor::zeros(shape[0]));
        Self {
            left,
            right,
            data,
            was_computed: Cell::new(false),
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
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix1>,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix1>,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        general_mat_vec_mul(
            1.0,
            &*self.left.data(),
            &*self.right.data(),
            0.0,
            &mut *self.data.borrow_mut(),
        );
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix1>>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(left.data().raw_dim(), right.data().raw_dim());
        let data = RefCell::new(Tensor::zeros(shape[0]));
        Self {
            left,
            right,
            data,
            was_computed: Cell::new(false),
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
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs> Forward for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        self.data.borrow_mut()[0] = self.left.data().dot(&*self.right.data());
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Power ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Power<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    exp: i32,
    was_computed: Cell<bool>,
}

impl<T> Power<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>, exp: i32) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            exp,
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for Power<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        let exp = self.exp;
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.powi(exp));
        true
    }
}

impl<T> Data for Power<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sum ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Sum<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<Ix1>>,
    was_computed: Cell<bool>,
}

impl<T> Sum<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(1);
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for Sum<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        self.data.borrow_mut()[0] = self.op.data().sum();
        true
    }
}

impl<T> Data for Sum<T>
where
    T: Data,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logn ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Logn<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T> Logn<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for Logn<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.ln());
        true
    }
}

impl<T> Data for Logn<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ReLU<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T> ReLU<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for ReLU<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = if *o > 0.0 { *o } else { 0.0 });
        true
    }
}

impl<T> Data for ReLU<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct LeakyReLU<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T> LeakyReLU<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for LeakyReLU<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = if *o > 0.0 { *o } else { 0.01 * o });
        true
    }
}

impl<T> Data for LeakyReLU<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftPlus ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SoftPlus<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T> SoftPlus<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for SoftPlus<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
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
        true
    }
}

impl<T> Data for SoftPlus<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Sigmoid<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T> Sigmoid<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for Sigmoid<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
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
        true
    }
}

impl<T> Data for Sigmoid<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TanH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct TanH<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T> TanH<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for TanH<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.tanh());
        true
    }
}

impl<T> Data for TanH<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Exp ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Exp<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T> Exp<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for Exp<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.exp());
        true
    }
}

impl<T> Data for Exp<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Softmax<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    axis: usize,
    was_computed: Cell<bool>,
}

impl<T> Softmax<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>, axis: usize) -> Self {
        let data = Tensor::zeros(op.data().raw_dim());
        Self {
            op,
            data: RefCell::new(data),
            axis,
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for Softmax<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
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
        true
    }
}

impl<T> Data for Softmax<T>
where
    T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Concatenate<Lhs, Rhs, D>
where
    Lhs: Data<Dim = D>,
    Rhs: Data<Dim = D>,
    D: Dimension + RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    data: RefCell<Tensor<D>>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs, D> Concatenate<Lhs, Rhs, D>
where
    Lhs: Data<Dim = D>,
    Rhs: Data<Dim = D>,
    D: Dimension + RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let data = RefCell::new(
            concatenate(Axis(axis), &[left.data().view(), right.data().view()]).unwrap(),
        );
        Self {
            left,
            right,
            data,
            axis,
            was_computed: Cell::new(false),
        }
    }
    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }
    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs, D> Data for Concatenate<Lhs, Rhs, D>
where
    Lhs: Data<Dim = D>,
    Rhs: Data<Dim = D>,
    D: Dimension + RemoveAxis,
{
    type Dim = D;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs, D> Forward for Concatenate<Lhs, Rhs, D>
where
    Lhs: Data<Dim = D>,
    Rhs: Data<Dim = D>,
    D: Dimension + RemoveAxis,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);

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
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Stack<Lhs, Rhs, D>
where
    Lhs: Data<Dim = D>,
    Rhs: Data<Dim = D>,
    D: Dimension + RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    data: RefCell<Tensor<D::Larger>>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs, D> Stack<Lhs, Rhs, D>
where
    Lhs: Data<Dim = D>,
    Rhs: Data<Dim = D>,
    D: Dimension + RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let data =
            RefCell::new(stack(Axis(axis), &[left.data().view(), right.data().view()]).unwrap());
        Self {
            left,
            right,
            data,
            axis,
            was_computed: Cell::new(false),
        }
    }
    pub fn left_operand(&self) -> Rc<Lhs> {
        self.left.clone()
    }
    pub fn right_operand(&self) -> Rc<Rhs> {
        self.right.clone()
    }
}

impl<Lhs, Rhs, D> Data for Stack<Lhs, Rhs, D>
where
    Lhs: Data<Dim = D>,
    Rhs: Data<Dim = D>,
    D: Dimension + RemoveAxis,
{
    type Dim = D::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Lhs, Rhs, D> Forward for Stack<Lhs, Rhs, D>
where
    Lhs: Data<Dim = D>,
    Rhs: Data<Dim = D>,
    D: Dimension + RemoveAxis,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);

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
        let mut subview_right = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<Rhs::Dim>()
            .unwrap();
        Zip::from(&*lhs_data)
            .and(&mut subview_left)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
        Zip::from(&*rhs_data)
            .and(&mut subview_right)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Unsqueeze ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Unsqueeze<T>
where
    T: Data,
{
    op: Rc<T>,
    data: RefCell<Tensor<<<T as Data>::Dim as Dimension>::Larger>>,
    axis: usize,
    was_computed: Cell<bool>,
}

impl<T> Unsqueeze<T>
where
    T: Data,
{
    pub fn new(op: Rc<T>, axis: usize) -> Self {
        let shape = op.data().raw_dim();
        let data = Tensor::zeros(shape.insert_axis(Axis(axis)));
        Self {
            op,
            data: RefCell::new(data),
            axis,
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.op.clone()
    }
}

impl<T> Forward for Unsqueeze<T>
where
    T: Data,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
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
        true
    }
}

impl<T> Data for Unsqueeze<T>
where
    T: Data,
{
    type Dim = <T::Dim as Dimension>::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~