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
    fn forward(&self) -> bool;
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
    fn forward(&self) -> bool {
        false
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Negation<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> Negation<T> {
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

impl<T: Data + Forward> Forward for Negation<T> {
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
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> Transpose<T> {
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

impl<T: Data + Forward> Forward for Transpose<T> {
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
    was_computed: Cell<bool>,
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
    was_computed: Cell<bool>,
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
    was_computed: Cell<bool>,
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
    was_computed: Cell<bool>,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix2> + Forward,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix2>>,
    was_computed: Cell<bool>,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix1>>,
    was_computed: Cell<bool>,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1> + Forward,
    Rhs: Data<Dim = Ix1> + Forward,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix1>>,
    was_computed: Cell<bool>,
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
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        self.data.borrow_mut()[0] = self.left.data().dot(&*self.right.data());

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Power ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Power<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    exp: i32,
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> Power<T> {
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

impl<T: Data + Forward> Forward for Power<T> {
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
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> Sum<T> {
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

impl<T: Data + Forward> Forward for Sum<T> {
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        self.data.borrow_mut()[0] = self.op.data().sum();

        true
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
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> Logn<T> {
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

impl<T: Data + Forward> Forward for Logn<T> {
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

impl<T: Data + Forward> Data for Logn<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ReLU<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> ReLU<T> {
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

impl<T: Data + Forward> Forward for ReLU<T> {
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.op.data())
            .par_for_each(|v, o| *v = o.max(0.));

        true
    }
}

impl<T: Data + Forward> Data for ReLU<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct LeakyReLU<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> LeakyReLU<T> {
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

impl<T: Data + Forward> Forward for LeakyReLU<T> {
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
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> SoftPlus<T> {
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

impl<T: Data + Forward> Forward for SoftPlus<T> {
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
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> Sigmoid<T> {
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

impl<T: Data + Forward> Forward for Sigmoid<T> {
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
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> TanH<T> {
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

impl<T: Data + Forward> Forward for TanH<T> {
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
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> Exp<T> {
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

impl<T: Data + Forward> Forward for Exp<T> {
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
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> Softmax<T> {
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

impl<T: Data + Forward> Forward for Softmax<T> {
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
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> LogSoftmax<T> {
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

impl<T: Data + Forward> Forward for LogSoftmax<T> {
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
                let exp = &lane_o.map(|el| (el - max).exp());
                let log_sum_exp = exp.sum().ln();
                Zip::from(lane_v)
                    .and(lane_o)
                    .for_each(|lane_v_el, lane_o_el| *lane_v_el = lane_o_el - log_sum_exp - max);
            });

        true
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
    was_computed: Cell<bool>,
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
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim> + Forward,
    Rhs: Data + Forward,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let data = stack(
            Axis(axis),
            &[
                Tensor::zeros(left.data().raw_dim()).view(),
                Tensor::zeros(right.data().raw_dim()).view(),
            ],
        )
        .unwrap();

        Self {
            left,
            right,
            data: RefCell::new(data),
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Unsqueeze ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Unsqueeze<T: Data + Forward> {
    op: Rc<T>,
    data: RefCell<Tensor<<<T as Data>::Dim as Dimension>::Larger>>,
    axis: usize,
    was_computed: Cell<bool>,
}

impl<T: Data + Forward> Unsqueeze<T> {
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

impl<T: Data + Forward> Forward for Unsqueeze<T> {
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

impl<T: Data + Forward> Data for Unsqueeze<T> {
    type Dim = <T::Dim as Dimension>::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

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
            assert!(Rc::ptr_eq(&negation.operand(), &input));

            negation.forward();
            assert_is_precise_enough(
                &*negation.data(),
                &Tensor::from_shape_vec((3, 3), vec![-1., -2., -3., -4., -5., -6., -7., -8., -9.])
                    .unwrap(),
            );
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let first = Rc::new(Negation::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Negation::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![-1., -2., -3., -4., -5., -6., -7., -8., -9.])
                    .unwrap()
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![-1., -2., -3., -4., -5., -6., -7., -8., -9.])
                    .unwrap()
            );
            assert_eq!(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap()
            );
        }
    }

    mod transpose {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let transpose = Transpose::new(input.clone());
            assert_eq!(*transpose.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&transpose.operand(), &input));

            transpose.forward();
            assert_is_precise_enough(
                &*transpose.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]).unwrap(),
            );
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let first = Rc::new(Transpose::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Transpose::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            first.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]).unwrap()
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&second.operand(), &first));

            second.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]).unwrap()
            );
            assert_eq!(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap()
            );
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
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.forward();
            assert_eq!(
                &*addition.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]).unwrap()
            );
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![1.; 9]);

            let first = Rc::new(Addition::new(left.clone(), right.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second = Addition::new(first.clone(), right.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]).unwrap()
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));

            second.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]).unwrap()
            );
            assert_eq!(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![3., 4., 5., 6., 7., 8., 9., 10., 11.])
                    .unwrap()
            );
        }

        #[test]
        fn left_broadcast() {
            let left = make_me_an_input((1, 3), vec![1., 2., 3.]);
            let right = make_me_an_input((2, 2, 3), vec![1.; 12]);

            let addition = Addition::new(left.clone(), right.clone());
            assert_eq!(*addition.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.forward();
            assert_eq!(
                &*addition.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.]
                )
                .unwrap()
            );
        }

        #[test]
        fn right_broadcast() {
            let left = make_me_an_input((2, 2, 3), vec![1.; 12]);
            let right = make_me_an_input((1, 3), vec![1., 2., 3.]);

            let addition = Addition::new(left.clone(), right.clone());
            assert_eq!(*addition.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.forward();
            assert_eq!(
                &*addition.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.]
                )
                .unwrap()
            );
        }
    }

    mod subtraction {
        use super::*;

        #[test]
        fn single_node() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![1., 1., 1., 1., 1., 1., 1., 1., 1.]);

            let addition = Subtraction::new(left.clone(), right.clone());
            assert_eq!(*addition.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.forward();
            assert_eq!(
                &*addition.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 1., 2., 3., 4., 5., 6., 7., 8.]).unwrap()
            );
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![1.; 9]);

            let first = Rc::new(Subtraction::new(left.clone(), right.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second = Subtraction::new(first.clone(), right.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 1., 2., 3., 4., 5., 6., 7., 8.]).unwrap()
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));

            second.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![0., 1., 2., 3., 4., 5., 6., 7., 8.]).unwrap()
            );
            assert_eq!(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![-1., 0., 1., 2., 3., 4., 5., 6., 7.]).unwrap()
            );
        }

        #[test]
        fn left_broadcast() {
            let left = make_me_an_input((1, 3), vec![1., 2., 3.]);
            let right = make_me_an_input((2, 2, 3), vec![1.; 12]);

            let addition = Subtraction::new(left.clone(), right.clone());
            assert_eq!(*addition.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.forward();
            assert_eq!(
                &*addition.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2.]
                )
                .unwrap()
            );
        }

        #[test]
        fn right_broadcast() {
            let left = make_me_an_input((2, 2, 3), vec![1.; 12]);
            let right = make_me_an_input((1, 3), vec![1., 2., 3.]);

            let addition = Subtraction::new(left.clone(), right.clone());
            assert_eq!(*addition.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert!(Rc::ptr_eq(&addition.left_operand(), &left));
            assert!(Rc::ptr_eq(&addition.right_operand(), &right));

            addition.forward();
            assert_eq!(
                &*addition.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![0., -1., -2., 0., -1., -2., 0., -1., -2., 0., -1., -2.]
                )
                .unwrap()
            );
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
            assert!(Rc::ptr_eq(&multiplication.left_operand(), &left));
            assert!(Rc::ptr_eq(&multiplication.right_operand(), &right));

            multiplication.forward();
            assert_eq!(
                &*multiplication.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 4., 6., 8., 10., 12., 14., 16., 18.])
                    .unwrap()
            );
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![2.; 9]);

            let first = Rc::new(Multiplication::new(left.clone(), right.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second = Multiplication::new(first.clone(), right.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 4., 6., 8., 10., 12., 14., 16., 18.])
                    .unwrap()
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));

            second.forward();
            assert_eq!(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![2., 4., 6., 8., 10., 12., 14., 16., 18.])
                    .unwrap()
            );
            assert_eq!(
                &*second.data(),
                &Tensor::from_shape_vec((3, 3), vec![4., 8., 12., 16., 20., 24., 28., 32., 36.])
                    .unwrap()
            );
        }

        #[test]
        fn left_broadcast() {
            let left = make_me_an_input((1, 3), vec![1., 2., 3.]);
            let right = make_me_an_input((2, 2, 3), vec![2.; 12]);

            let multiplication = Multiplication::new(left.clone(), right.clone());
            assert_eq!(*multiplication.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert!(Rc::ptr_eq(&multiplication.left_operand(), &left));
            assert!(Rc::ptr_eq(&multiplication.right_operand(), &right));

            multiplication.forward();
            assert_eq!(
                &*multiplication.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![2., 4., 6., 2., 4., 6., 2., 4., 6., 2., 4., 6.]
                )
                .unwrap()
            );
        }

        #[test]
        fn right_broadcast() {
            let left = make_me_an_input((2, 2, 3), vec![2.; 12]);
            let right = make_me_an_input((1, 3), vec![1., 2., 3.]);

            let multiplication = Multiplication::new(left.clone(), right.clone());
            assert_eq!(*multiplication.data(), Tensor::from_elem((2, 2, 3), 0.));
            assert!(Rc::ptr_eq(&multiplication.left_operand(), &left));
            assert!(Rc::ptr_eq(&multiplication.right_operand(), &right));

            multiplication.forward();
            assert_eq!(
                &*multiplication.data(),
                &Tensor::from_shape_vec(
                    (2, 2, 3),
                    vec![2., 4., 6., 2., 4., 6., 2., 4., 6., 2., 4., 6.]
                )
                .unwrap()
            );
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
            assert!(Rc::ptr_eq(&division.left_operand(), &left));
            assert!(Rc::ptr_eq(&division.right_operand(), &right));

            division.forward();
            assert_is_precise_enough(
                &*division.data(),
                &Tensor::from_shape_vec((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])
                    .unwrap(),
            );
        }

        #[test]
        fn chained() {
            let left = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = make_me_an_input((3, 3), vec![2.; 9]);

            let first = Rc::new(Division::new(left.clone(), right.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&first.left_operand(), &left));
            assert!(Rc::ptr_eq(&first.right_operand(), &right));

            let second = Division::new(first.clone(), right.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&second.left_operand(), &first));
            assert!(Rc::ptr_eq(&second.right_operand(), &right));

            first.forward();
            assert_is_precise_enough(
                &*first.data(),
                &Tensor::from_shape_vec((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5])
                    .unwrap(),
            );
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));

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
        }

        #[test]
        fn left_broadcast() {
            let left = make_me_an_input((1, 3), vec![1., 2., 3.]);
            let right = make_me_an_input((2, 2, 3), vec![2.; 12]);

            let division = Division::new(left.clone(), right.clone());
            assert_eq!(*division.data(), Tensor::from_elem((2, 2, 3), 0.));
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
        }

        #[test]
        fn right_broadcast() {
            let left = make_me_an_input((2, 2, 3), vec![2.; 12]);
            let right = make_me_an_input((1, 3), vec![1., 2., 3.]);

            let division = Division::new(left.clone(), right.clone());
            assert_eq!(*division.data(), Tensor::from_elem((2, 2, 3), 0.));
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
        }
    }

    mod power {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let logn = Power::new(input.clone(), 3);
            assert_eq!(*logn.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&logn.operand(), &input));

            logn.forward();
            assert_is_precise_enough(
                &*logn.data(),
                &Tensor::from_shape_vec(
                    (3, 3),
                    vec![1., 8., 27., 64., 125., 216., 343., 512., 729.],
                )
                .unwrap(),
            );
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let first = Rc::new(Power::new(input.clone(), 3));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Power::new(first.clone(), 5);
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
        }
    }

    mod logn {
        use super::*;

        #[test]
        fn single_node() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let logn = Logn::new(input.clone());
            assert_eq!(*logn.data(), Tensor::from_elem((3, 3), 0.));
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
        }

        #[test]
        fn chained() {
            let input = make_me_an_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);

            let first = Rc::new(Logn::new(input.clone()));
            assert_eq!(*first.data(), Tensor::from_elem((3, 3), 0.));
            assert!(Rc::ptr_eq(&first.operand(), &input));

            let second = Logn::new(first.clone());
            assert_eq!(*second.data(), Tensor::from_elem((3, 3), 0.));
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
        }
    }
}
