use super::{
    super::{
        broadcasted_zeros, expect_tensor, expect_tensor_mut, BroadTensor, Broadcasted, DynTensor,
        Tensor,
    },
    Backward, Data, Differentiable, DotDim, Dropout, Gradient, Input, Overwrite,
};
use ndarray::{
    concatenate,
    linalg::{general_mat_mul, general_mat_vec_mul},
    s, stack, ArrayBase, ArrayView, Axis, DimMax, Dimension, IntoNdProducer, Ix1, Ix2, NewAxis,
    RemoveAxis, Zip,
};

use std::cell::{Cell, Ref, RefCell, RefMut};
use std::rc::Rc;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Utility Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fn sum_axis_inplace(arr: &mut DynTensor, axis: Axis) {
    let (first, rest) = arr.view_mut().split_at(axis, 1);
    Zip::from(first.remove_axis(axis))
        .and(rest.lanes(axis))
        .for_each(|dst, src| *dst += src.sum());
    arr.index_axis_inplace(axis, 0);
}

pub fn reduce<D: Dimension, E: Dimension>(dest: &Tensor<D>, src: &Tensor<E>) -> DynTensor {
    let mut dyn_rhs = src.clone().into_dyn();

    unsafe {
        while (*(&dyn_rhs as *const DynTensor)).ndim() > dest.ndim() {
            sum_axis_inplace(&mut dyn_rhs, Axis(0));
        }
    }

    for (axis, size) in dest.shape().iter().enumerate() {
        if *size == 1 {
            sum_axis_inplace(&mut dyn_rhs, ndarray::Axis(axis));
            dyn_rhs.insert_axis_inplace(ndarray::Axis(axis));
        }
    }

    dyn_rhs
}

pub fn push_gradient<'a, T, P, D>(node: &T, src: P)
where
    T: Gradient + Overwrite,
    P: IntoNdProducer<Dim = D, Output = ArrayView<'a, f32, D>, Item = &'a f32>,
    D: Dimension,
{
    let mut dest = node.gradient_mut();
    let zip = Zip::from(&mut *dest).and_broadcast(src);
    if node.can_overwrite() {
        zip.for_each(|d, s| *d = *s);
        node.set_overwrite(false);
    } else {
        zip.for_each(|d, s| *d += *s);
    }
}

pub fn push_mat_mat_gradient<T, S1, S2>(
    dest: &T,
    fst: &ArrayBase<S1, Ix2>,
    snd: &ArrayBase<S2, Ix2>,
) where
    T: Gradient<Dim = Ix2> + Overwrite,
    S1: ndarray::Data<Elem = f32>,
    S2: ndarray::Data<Elem = f32>,
{
    if dest.can_overwrite() {
        general_mat_mul(1., fst, snd, 0., &mut dest.gradient_mut());
        dest.set_overwrite(false);
    } else {
        general_mat_mul(1., fst, snd, 1., &mut dest.gradient_mut());
    }
}

pub fn push_mat_vec_gradient<T, S1, S2>(
    node: &T,
    fst: &ArrayBase<S1, Ix2>,
    snd: &ArrayBase<S2, Ix1>,
) where
    T: Gradient<Dim = Ix2> + Overwrite,
    S1: ndarray::Data<Elem = f32>,
    S2: ndarray::Data<Elem = f32>,
{
    let mut dest = node.gradient_mut();
    let zip = Zip::from(&mut *dest).and_broadcast(fst).and_broadcast(snd);
    if node.can_overwrite() {
        zip.for_each(|d, f, s| *d = f * s);
        node.set_overwrite(false);
    } else {
        zip.for_each(|d, f, s| *d += f * s);
    }
}

pub fn push_vec_mat_gradient<T, S1, S2>(
    dest: &T,
    fst: &ArrayBase<S1, Ix2>,
    snd: &ArrayBase<S2, Ix1>,
) where
    T: Gradient<Dim = Ix1> + Overwrite,
    S1: ndarray::Data<Elem = f32>,
    S2: ndarray::Data<Elem = f32>,
{
    if dest.can_overwrite() {
        general_mat_vec_mul(1., fst, snd, 0., &mut dest.gradient_mut());
        dest.set_overwrite(false);
    } else {
        general_mat_vec_mul(1., fst, snd, 1., &mut dest.gradient_mut());
    }
}

pub fn push_vec_vec_gradient<T, S>(node: &T, fst: &ArrayBase<S, Ix1>, snd: &f32)
where
    T: Gradient<Dim = Ix1> + Overwrite,
    S: ndarray::Data<Elem = f32>,
{
    let mut dest = node.gradient_mut();
    let zip = Zip::from(&mut *dest).and_broadcast(fst);
    if node.can_overwrite() {
        zip.for_each(|d, f| *d = f * snd);
        node.set_overwrite(false);
    } else {
        zip.for_each(|d, f| *d += f * snd);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InputBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// The backward component of a differentiable leaf of the computational graph.
pub struct InputBackward<D: Dimension> {
    gradient: RefCell<Option<Tensor<D>>>,
    overwrite: Cell<bool>,
}

impl<D: Dimension> InputBackward<D> {
    pub fn zero_grad(&self) {
        expect_tensor_mut(&self.gradient).fill(0.);
    }
}

impl<D: Dimension> Differentiable for Input<D> {
    type Output = InputBackward<D>;

    fn differentiable(&self) -> Self::Output {
        Self::Output {
            gradient: RefCell::new(Some(Tensor::zeros(self.data().raw_dim()))),
            overwrite: Cell::new(true),
        }
    }
}

impl<D: Dimension> Gradient for InputBackward<D> {
    type Dim = D;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<D: Dimension> Overwrite for InputBackward<D> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NegationBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct NegationBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T: Gradient + Overwrite> NegationBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            operand,
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for NegationBackward<T> {
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for NegationBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for NegationBackward<T> {
    fn backward(&self) {
        push_gradient(&*self.operand, &-(&*self.gradient()));
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransposeBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct TransposeBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T: Gradient + Overwrite> TransposeBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.gradient().t().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            operand,
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for TransposeBackward<T> {
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for TransposeBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for TransposeBackward<T> {
    fn backward(&self) {
        push_gradient(&*self.operand, self.gradient().t());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdditionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    gradient: RefCell<Option<BroadTensor<Lhs::Dim, Rhs::Dim>>>,
    shape: Broadcasted<Lhs::Dim, Rhs::Dim>,
    overwrite: Cell<bool>,
    left: Rc<Lhs>,
    right: Rc<Rhs>,
}

impl<Lhs, Rhs> AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let gradient = broadcasted_zeros(&left.gradient(), &right.gradient());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            left,
            right,
        }
    }
}

impl<Lhs, Rhs> Gradient for AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<Lhs, Rhs> Overwrite for AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<Lhs, Rhs> Backward for AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn backward(&self) {
        let reduced = reduce(&self.left.gradient_mut(), &self.gradient());
        push_gradient(&*self.left, &reduced.as_standard_layout());

        let reduced = reduce(&self.right.gradient_mut(), &self.gradient());
        push_gradient(&*self.right, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdditionBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    gradient: RefCell<Option<BroadTensor<T::Dim, U::Dim>>>,
    shape: Broadcasted<T::Dim, U::Dim>,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T, U> AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    pub fn new(diff: Rc<T>, no_diff: Rc<U>) -> Self {
        let gradient = broadcasted_zeros(&diff.gradient(), &no_diff.data());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            operand: diff,
            overwrite: Cell::new(true),
        }
    }
}

impl<T, U> Gradient for AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    type Dim = Broadcasted<T::Dim, U::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for AdditionBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn backward(&self) {
        let reduced = reduce(&self.operand.gradient(), &self.gradient());
        push_gradient(&*self.operand, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SubtractionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    gradient: RefCell<Option<BroadTensor<Lhs::Dim, Rhs::Dim>>>,
    shape: Broadcasted<Lhs::Dim, Rhs::Dim>,
    overwrite: Cell<bool>,
    left: Rc<Lhs>,
    right: Rc<Rhs>,
}

impl<Lhs, Rhs> SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let gradient = broadcasted_zeros(&left.gradient(), &right.gradient());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            left,
            right,
        }
    }
}

impl<Lhs, Rhs> Gradient for SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<Lhs, Rhs> Overwrite for SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<Lhs, Rhs> Backward for SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient + Overwrite,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn backward(&self) {
        let reduced = reduce(&self.left.gradient_mut(), &self.gradient());
        push_gradient(&*self.left, &reduced.as_standard_layout());

        let reduced = -reduce(&self.right.gradient_mut(), &self.gradient());
        push_gradient(&*self.right, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SubtractionBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SubtractionBackwardLeft<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    gradient: RefCell<Option<BroadTensor<T::Dim, U::Dim>>>,
    shape: Broadcasted<T::Dim, U::Dim>,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T, U> SubtractionBackwardLeft<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    pub fn new(diff: Rc<T>, no_diff: Rc<U>) -> Self {
        let gradient = broadcasted_zeros(&diff.gradient(), &no_diff.data());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            operand: diff,
        }
    }
}

impl<T, U> Gradient for SubtractionBackwardLeft<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    type Dim = Broadcasted<T::Dim, U::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for SubtractionBackwardLeft<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for SubtractionBackwardLeft<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn backward(&self) {
        let reduced = reduce(&self.operand.gradient_mut(), &self.gradient());
        push_gradient(&*self.operand, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SubtractionBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SubtractionBackwardRight<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    gradient: RefCell<Option<BroadTensor<T::Dim, U::Dim>>>,
    shape: Broadcasted<T::Dim, U::Dim>,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T, U> SubtractionBackwardRight<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    pub fn new(diff: Rc<T>, no_diff: Rc<U>) -> Self {
        let gradient = broadcasted_zeros(&diff.gradient(), &no_diff.data());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            operand: diff,
        }
    }
}

impl<T, U> Gradient for SubtractionBackwardRight<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    type Dim = Broadcasted<T::Dim, U::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for SubtractionBackwardRight<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for SubtractionBackwardRight<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn backward(&self) {
        let reduced = -reduce(&*self.operand.gradient_mut(), &self.gradient());
        push_gradient(&*self.operand, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiplicationBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    gradient: RefCell<Option<BroadTensor<LhsG::Dim, RhsG::Dim>>>,
    shape: Broadcasted<LhsG::Dim, RhsG::Dim>,
    overwrite: Cell<bool>,
    buffer: RefCell<Option<BroadTensor<LhsG::Dim, RhsG::Dim>>>,
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, LhsG, RhsD, RhsG> MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    pub fn new(
        left_data: Rc<LhsD>,
        left_grad: Rc<LhsG>,
        right_data: Rc<RhsD>,
        right_grad: Rc<RhsG>,
    ) -> Self {
        let gradient = broadcasted_zeros(&left_grad.gradient(), &right_grad.gradient());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape: shape.clone(),
            overwrite: Cell::new(true),
            buffer: RefCell::new(Some(Tensor::zeros(shape))),
            left_data,
            left_grad,
            right_data,
            right_grad,
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Gradient for MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    type Dim = Broadcasted<LhsG::Dim, RhsG::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Overwrite for MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let mut buffer = expect_tensor_mut(&self.buffer);
        Zip::from(&mut *buffer)
            .and(&*gradient)
            .and_broadcast(&*self.right_data.data())
            .for_each(|d, g, r| *d = g * r);
        let reduced = reduce(&self.left_grad.gradient(), &buffer);
        push_gradient(&*self.left_grad, &reduced.as_standard_layout());

        Zip::from(&mut *buffer)
            .and(&*gradient)
            .and_broadcast(&*self.left_data.data())
            .for_each(|d, g, l| *d = g * l);
        let reduced = reduce(&self.right_grad.gradient(), &buffer);
        push_gradient(&*self.right_grad, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiplicationBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    gradient: RefCell<Option<BroadTensor<T::Dim, U::Dim>>>,
    shape: Broadcasted<T::Dim, U::Dim>,
    overwrite: Cell<bool>,
    buffer: RefCell<Option<BroadTensor<T::Dim, U::Dim>>>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let gradient = broadcasted_zeros(&diff_operand.gradient(), &no_diff_operand.data());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape: shape.clone(),
            overwrite: Cell::new(true),
            buffer: RefCell::new(Some(Tensor::zeros(shape))),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    type Dim = Broadcasted<T::Dim, U::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Overwrite,
    U: Data,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let mut buffer = expect_tensor_mut(&self.buffer);

        Zip::from(&mut *buffer)
            .and(&*gradient)
            .and_broadcast(&*self.no_diff_operand.data())
            .for_each(|d, g, v| *d = g * v);
        let reduced = reduce(&self.diff_operand.gradient_mut(), &buffer);
        push_gradient(&*self.diff_operand, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DivisionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    gradient: RefCell<Option<BroadTensor<LhsG::Dim, RhsG::Dim>>>,
    shape: Broadcasted<LhsG::Dim, RhsG::Dim>,
    overwrite: Cell<bool>,
    buffer: RefCell<Option<BroadTensor<LhsG::Dim, RhsG::Dim>>>,
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, LhsG, RhsD, RhsG> DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    pub fn new(
        left_data: Rc<LhsD>,
        left_grad: Rc<LhsG>,
        right_data: Rc<RhsD>,
        right_grad: Rc<RhsG>,
    ) -> Self {
        let gradient = broadcasted_zeros(&left_grad.gradient(), &right_grad.gradient());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape: shape.clone(),
            overwrite: Cell::new(true),
            buffer: RefCell::new(Some(Tensor::zeros(shape))),
            left_data,
            left_grad,
            right_data,
            right_grad,
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Gradient for DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    type Dim = Broadcasted<LhsG::Dim, RhsG::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Overwrite for DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let mut buffer = expect_tensor_mut(&self.buffer);

        Zip::from(&mut *buffer)
            .and(&*gradient)
            .and_broadcast(&*self.right_data.data())
            .for_each(|d, g, r| *d = g / r);
        let reduced = reduce(&self.left_grad.gradient(), &buffer);
        push_gradient(&*self.left_grad, &reduced.as_standard_layout());

        Zip::from(&mut *buffer)
            .and(&*gradient)
            .and_broadcast(&*self.left_data.data())
            .and_broadcast(&*self.right_data.data())
            .for_each(|d, g, l, r| *d = -g * l / r.powi(2));
        let reduced = reduce(&self.right_grad.gradient(), &buffer);
        push_gradient(&*self.right_grad, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DivisionBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    gradient: RefCell<Option<BroadTensor<LhsG::Dim, RhsD::Dim>>>,
    shape: Broadcasted<LhsG::Dim, RhsD::Dim>,
    overwrite: Cell<bool>,
    buffer: RefCell<Option<BroadTensor<LhsG::Dim, RhsD::Dim>>>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
}

impl<LhsG, RhsD> DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    pub fn new(left_grad: Rc<LhsG>, right_data: Rc<RhsD>) -> Self {
        let gradient = broadcasted_zeros(&left_grad.gradient(), &right_data.data());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape: shape.clone(),
            overwrite: Cell::new(true),
            buffer: RefCell::new(Some(Tensor::zeros(shape))),
            left_grad,
            right_data,
        }
    }
}

impl<LhsG, RhsD> Gradient for DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    type Dim = Broadcasted<LhsG::Dim, RhsD::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsG, RhsD> Overwrite for DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsG, RhsD> Backward for DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data,
    LhsG: Gradient + Overwrite,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let mut buffer = expect_tensor_mut(&self.buffer);

        Zip::from(&mut *buffer)
            .and(&*gradient)
            .and_broadcast(&*self.right_data.data())
            .for_each(|d, g, r| *d = g / r);
        let reduced = reduce(&self.left_grad.gradient(), &buffer);
        push_gradient(&*self.left_grad, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DivisionBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    gradient: RefCell<Option<BroadTensor<LhsD::Dim, RhsG::Dim>>>,
    shape: Broadcasted<LhsD::Dim, RhsG::Dim>,
    overwrite: Cell<bool>,
    buffer: RefCell<Option<BroadTensor<LhsD::Dim, RhsG::Dim>>>,
    left_data: Rc<LhsD>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, RhsD, RhsG> DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    pub fn new(left_data: Rc<LhsD>, right_data: Rc<RhsD>, right_grad: Rc<RhsG>) -> Self {
        let gradient = broadcasted_zeros(&left_data.data(), &right_grad.gradient());
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape: shape.clone(),
            overwrite: Cell::new(true),
            buffer: RefCell::new(Some(Tensor::zeros(shape))),
            left_data,
            right_data,
            right_grad,
        }
    }
}

impl<LhsD, RhsD, RhsG> Gradient for DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    type Dim = Broadcasted<LhsD::Dim, RhsG::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, RhsD, RhsG> Overwrite for DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, RhsD, RhsG> Backward for DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    RhsG: Gradient + Overwrite,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let mut buffer = expect_tensor_mut(&self.buffer);

        Zip::from(&mut *buffer)
            .and(&*gradient)
            .and_broadcast(&*self.left_data.data())
            .and_broadcast(&*self.right_data.data())
            .for_each(|d, g, l, r| *d = -g * l / r.powi(2));
        let reduced = reduce(&self.right_grad.gradient(), &buffer);
        push_gradient(&*self.right_grad, &reduced.as_standard_layout());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MattrixMatrixMulBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixMatrixMulBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixMatrixMulBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MattrixMatrixMulTBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMulTBackward<LhsD, LhsG, RhsD, RhsG>
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

impl<LhsD, LhsG, RhsD, RhsG> MatrixMatrixMulTBackward<LhsD, LhsG, RhsD, RhsG>
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
            right_grad.gradient().t().raw_dim(),
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

impl<LhsD, LhsG, RhsD, RhsG> Gradient for MatrixMatrixMulTBackward<LhsD, LhsG, RhsD, RhsG>
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

impl<LhsD, LhsG, RhsD, RhsG> Overwrite for MatrixMatrixMulTBackward<LhsD, LhsG, RhsD, RhsG>
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

impl<LhsD, LhsG, RhsD, RhsG> Backward for MatrixMatrixMulTBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn backward(&self) {
        let gradient = self.gradient();
        push_mat_mat_gradient(&*self.left_grad, &gradient, &self.right_data.data());
        push_mat_mat_gradient(&*self.right_grad, &gradient.t(), &self.left_data.data());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixMatrixMulTBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMulTBackwardLeft<LhsG, RhsD>
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

impl<LhsG, RhsD> MatrixMatrixMulTBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    pub fn new(left_grad: Rc<LhsG>, right_data: Rc<RhsD>) -> Self {
        let shape = DotDim::shape(
            left_grad.gradient().raw_dim(),
            right_data.data().t().raw_dim(),
        );

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape))),
            shape,
            overwrite: Cell::new(true),
            left_grad,
            right_data,
        }
    }
}

impl<LhsG, RhsD> Gradient for MatrixMatrixMulTBackwardLeft<LhsG, RhsD>
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

impl<LhsG, RhsD> Overwrite for MatrixMatrixMulTBackwardLeft<LhsG, RhsD>
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

impl<LhsG, RhsD> Backward for MatrixMatrixMulTBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn backward(&self) {
        push_mat_mat_gradient(&*self.left_grad, &self.gradient(), &self.right_data.data());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixMatrixMulBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMulTBackwardRight<LhsD, RhsG>
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

impl<LhsD, RhsG> MatrixMatrixMulTBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    pub fn new(left_data: Rc<LhsD>, right_grad: Rc<RhsG>) -> Self {
        let shape = DotDim::shape(
            left_data.data().raw_dim(),
            right_grad.gradient().t().raw_dim(),
        );

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape))),
            shape,
            overwrite: Cell::new(true),
            left_data,
            right_grad,
        }
    }
}

impl<LhsD, RhsG> Gradient for MatrixMatrixMulTBackwardRight<LhsD, RhsG>
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

impl<LhsD, RhsG> Overwrite for MatrixMatrixMulTBackwardRight<LhsD, RhsG>
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

impl<LhsD, RhsG> Backward for MatrixMatrixMulTBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn backward(&self) {
        push_mat_mat_gradient(
            &*self.right_grad,
            &self.gradient().t(),
            &self.left_data.data(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMulBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    shape: Ix1,
    overwrite: Cell<bool>,
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, LhsG, RhsD, RhsG> MatrixVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
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

impl<LhsD, LhsG, RhsD, RhsG> Gradient for MatrixVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Overwrite for MatrixVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for MatrixVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    fn backward(&self) {
        let gradient = self.gradient();
        push_mat_vec_gradient(
            &*self.left_grad,
            &gradient.slice(s![.., NewAxis]),
            &self.right_data.data(),
        );
        push_vec_mat_gradient(&*self.right_grad, &self.left_data.data().t(), &gradient);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMulBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixVectorMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    shape: Ix1,
    overwrite: Cell<bool>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
}

impl<LhsG, RhsD> MatrixVectorMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix1>,
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

impl<LhsG, RhsD> Gradient for MatrixVectorMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsG, RhsD> Overwrite for MatrixVectorMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsG, RhsD> Backward for MatrixVectorMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn backward(&self) {
        push_mat_vec_gradient(
            &*self.left_grad,
            &self.gradient().slice(s![.., NewAxis]),
            &self.right_data.data(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMulBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixVectorMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    shape: Ix1,
    overwrite: Cell<bool>,
    left_data: Rc<LhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, RhsG> MatrixVectorMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
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

impl<LhsD, RhsG> Gradient for MatrixVectorMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, RhsG> Overwrite for MatrixVectorMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, RhsG> Backward for MatrixVectorMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2>,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    fn backward(&self) {
        push_vec_mat_gradient(
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMulBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct VectorMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    shape: Ix1,
    overwrite: Cell<bool>,
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, LhsG, RhsD, RhsG> VectorMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
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

impl<LhsD, LhsG, RhsD, RhsG> Gradient for VectorMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Overwrite for VectorMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for VectorMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn backward(&self) {
        let gradient = self.gradient();
        push_vec_mat_gradient(&*self.left_grad, &self.right_data.data(), &gradient);
        push_mat_vec_gradient(
            &*self.right_grad,
            &self.left_data.data().slice(s![.., NewAxis]),
            &gradient,
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMulBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorMatrixMulBackwardLeft<LhsG, RhsD>
where
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsD: Data<Dim = Ix2>,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    shape: Ix1,
    overwrite: Cell<bool>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
}

impl<LhsG, RhsD> VectorMatrixMulBackwardLeft<LhsG, RhsD>
where
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsD: Data<Dim = Ix2>,
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

impl<LhsG, RhsD> Gradient for VectorMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsG, RhsD> Overwrite for VectorMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsG, RhsD> Backward for VectorMatrixMulBackwardLeft<LhsG, RhsD>
where
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsD: Data<Dim = Ix2>,
{
    fn backward(&self) {
        push_vec_mat_gradient(&*self.left_grad, &self.right_data.data(), &self.gradient());
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMulBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    shape: Ix1,
    overwrite: Cell<bool>,
    left_data: Rc<LhsD>,
    right_grad: Rc<RhsG>,
}

impl<LhsD, RhsG> VectorMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
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

impl<LhsD, RhsG> Gradient for VectorMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<LhsD, RhsG> Overwrite for VectorMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<LhsD, RhsG> Backward for VectorMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsG: Gradient<Dim = Ix2> + Overwrite,
{
    fn backward(&self) {
        push_mat_vec_gradient(
            &*self.right_grad,
            &self.left_data.data().slice(s![.., NewAxis]),
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMulBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    shape: Ix1,
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

impl<LhsD, LhsG, RhsD, RhsG> Gradient for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1>,
    RhsD: Data<Dim = Ix1>,
    LhsG: Gradient<Dim = Ix1> + Overwrite,
    RhsG: Gradient<Dim = Ix1> + Overwrite,
{
    type Dim = Ix1;

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
        push_vec_vec_gradient(&*self.left_grad, &self.right_data.data(), &gradient[0]);
        push_vec_vec_gradient(&*self.right_grad, &self.left_data.data(), &gradient[0]);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMulBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Overwrite,
    U: Data<Dim = Ix1>,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    shape: Ix1,
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
        let shape = DotDim::shape(
            diff_operand.gradient().raw_dim(),
            no_diff_operand.data().raw_dim(),
        );

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape))),
            shape,
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
    type Dim = Ix1;

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
            &self.gradient()[0],
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PowerBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct PowerBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
    exp: i32,
}

impl<T, U> PowerBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>, exp: i32) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            exp,
        }
    }
}

impl<T, U> Gradient for PowerBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for PowerBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for PowerBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let op_data = self.no_diff_operand.data();
        let grad = self.gradient();
        let exp = self.exp;

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = grad_el * op_data_el.powi(exp - 1) * exp as f32
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += grad_el * op_data_el.powi(exp - 1) * exp as f32
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SqrtBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct SqrtBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> SqrtBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for SqrtBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for SqrtBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for SqrtBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, data_el| {
                *op_grad_el = *grad_el * 0.5 / (data_el + f32::EPSILON)
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, data_el| {
                *op_grad_el += *grad_el * 0.5 / (data_el + f32::EPSILON)
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SumBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SumBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<Ix1>>>,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T: Gradient + Overwrite> SumBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        Self {
            operand,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            overwrite: Cell::new(true),
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for SumBackward<T> {
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for SumBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for SumBackward<T> {
    fn backward(&self) {
        let mut op_grad = self.operand.gradient_mut();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and_broadcast(&*grad);
        if self.operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el| *op_grad_el = *grad_el);
            self.operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MeanBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MeanBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<Ix1>>>,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T: Gradient + Overwrite> MeanBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        Self {
            operand,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            overwrite: Cell::new(true),
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for MeanBackward<T> {
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for MeanBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for MeanBackward<T> {
    fn backward(&self) {
        let numel = self.operand.gradient().len() as f32;
        let mut op_grad = self.operand.gradient_mut();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and_broadcast(&*grad);
        if self.operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el| *op_grad_el = *grad_el / numel);
            self.operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el / numel);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LognBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct LognBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> LognBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for LognBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for LognBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for LognBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let op_data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, op_data_el| *op_grad_el = grad_el / op_data_el);
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, op_data_el| *op_grad_el += grad_el / op_data_el);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLUBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct ReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> ReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for ReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for ReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for ReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let op_data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = ((*op_data_el > 0.0) as usize as f32) * grad_el
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += ((*op_data_el > 0.0) as usize as f32) * grad_el
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLUBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct LeakyReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> LeakyReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for LeakyReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for LeakyReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for LeakyReLUBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let op_data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = ((*op_data_el > 0.0) as usize as f32) * grad_el
                    + ((*op_data_el <= 0.0) as usize as f32) * 0.01
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += ((*op_data_el > 0.0) as usize as f32) * grad_el
                    + ((*op_data_el <= 0.0) as usize as f32) * 0.01
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftPlusBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~S

pub struct SoftPlusBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> SoftPlusBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for SoftPlusBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for SoftPlusBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for SoftPlusBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let op_data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = grad_el / (1.0 + (-*op_data_el).exp())
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += grad_el / (1.0 + (-*op_data_el).exp())
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SigmoidBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, data_el| {
                *op_grad_el = *grad_el * *data_el * (1.0 - *data_el)
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, data_el| {
                *op_grad_el += *grad_el * *data_el * (1.0 - *data_el)
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TanHBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct TanHBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> TanHBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for TanHBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for TanHBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for TanHBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, data_el| {
                *op_grad_el = *grad_el * (1.0 - data_el.powi(2))
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, data_el| {
                *op_grad_el += *grad_el * (1.0 - data_el.powi(2))
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ExpBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for ExpBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, data_el| *op_grad_el = *grad_el * data_el);
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, data_el| *op_grad_el += *grad_el * data_el);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftmaxBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
    axis: usize,
}

impl<T, U> SoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>, axis: usize) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            axis,
        }
    }
}

impl<T, U> Gradient for SoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for SoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for SoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();
        let axis = self.axis;
        let zip = Zip::from(op_grad.lanes_mut(Axis(axis)))
            .and(grad.lanes(Axis(axis)))
            .and(data.lanes(Axis(axis)));

        if self.diff_operand.can_overwrite() {
            zip.for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let sum = Zip::from(grad_lane)
                    .and(data_lane)
                    .fold(0., |acc, grad_el, data_el| acc + grad_el * data_el);
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, grad_el, data_el| {
                        *op_grad_el = data_el * (grad_el - sum)
                    })
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let sum = Zip::from(grad_lane)
                    .and(data_lane)
                    .fold(0., |acc, grad_el, data_el| acc + grad_el * data_el);
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, grad_el, data_el| {
                        *op_grad_el += data_el * (grad_el - sum)
                    })
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSoftmaxBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
    axis: usize,
}

impl<T, U> LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>, axis: usize) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            axis,
        }
    }
}

impl<T, U> Gradient for LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();
        let axis = self.axis;

        let zip = Zip::from(op_grad.lanes_mut(Axis(axis)))
            .and(grad.lanes(Axis(axis)))
            .and(data.lanes(Axis(axis)));
        if self.diff_operand.can_overwrite() {
            zip.for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let gradient_sum = grad_lane.sum();
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, grad_el, data_el| {
                        *op_grad_el = grad_el - data_el.exp() * gradient_sum
                    })
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let gradient_sum = grad_lane.sum();
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, grad_el, data_el| {
                        *op_grad_el += grad_el - data_el.exp() * gradient_sum
                    })
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConcatenateBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<Lhs::Dim>>>,
    shape: Lhs::Dim,
    overwrite: Cell<bool>,
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
}

impl<Lhs, Rhs> ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let gradient = concatenate(
            Axis(axis),
            &[left.gradient().view(), right.gradient().view()],
        )
        .unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            left,
            right,
            axis,
        }
    }
}

impl<Lhs, Rhs> Gradient for ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    type Dim = Lhs::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<Lhs, Rhs> Overwrite for ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<Lhs, Rhs> Backward for ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let (lhs_part, rhs_part) = gradient.view().split_at(
            Axis(self.axis),
            self.left.gradient_mut().len_of(Axis(self.axis)),
        );

        push_gradient(&*self.left, lhs_part);
        push_gradient(&*self.right, rhs_part);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConcatenateBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    left: Rc<T>,
    axis: usize,
}

impl<T> ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    pub fn new<U: Data<Dim = T::Dim>>(left: Rc<T>, right: Rc<U>, axis: usize) -> Self {
        let gradient =
            concatenate(Axis(axis), &[left.gradient().view(), right.data().view()]).unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            left,
            axis,
        }
    }
}

impl<T> Gradient for ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T> Overwrite for ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T> Backward for ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let (lhs_part, _) = gradient.view().split_at(
            Axis(self.axis),
            self.left.gradient_mut().len_of(Axis(self.axis)),
        );

        push_gradient(&*self.left, lhs_part);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConcatenateBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    offset: usize,
    right: Rc<T>,
    axis: usize,
}

impl<T> ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    pub fn new<U: Data<Dim = T::Dim>>(left: Rc<U>, right: Rc<T>, axis: usize) -> Self {
        let gradient =
            concatenate(Axis(axis), &[left.data().view(), right.gradient().view()]).unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            right,
            offset: left.data().len_of(Axis(axis)),
            axis,
        }
    }
}

impl<T> Gradient for ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T> Overwrite for ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T> Backward for ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let (_, rhs_part) = gradient.view().split_at(Axis(self.axis), self.offset);
        push_gradient(&*self.right, rhs_part);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StackBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct StackBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<<Lhs::Dim as Dimension>::Larger>>>,
    shape: <Lhs::Dim as Dimension>::Larger,
    overwrite: Cell<bool>,
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
}

impl<Lhs, Rhs> StackBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let gradient = stack(
            Axis(axis),
            &[left.gradient().view(), right.gradient().view()],
        )
        .unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            left,
            right,
            axis,
        }
    }
}

impl<Lhs, Rhs> Gradient for StackBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    type Dim = <Lhs::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<Lhs, Rhs> Overwrite for StackBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<Lhs, Rhs> Backward for StackBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let mut subviews = gradient.axis_iter(Axis(self.axis));
        push_gradient(
            &*self.left,
            subviews
                .next()
                .unwrap()
                .into_dimensionality::<Lhs::Dim>()
                .unwrap(),
        );
        push_gradient(
            &*self.right,
            subviews
                .next()
                .unwrap()
                .into_dimensionality::<Rhs::Dim>()
                .unwrap(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StackBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct StackBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<<T::Dim as Dimension>::Larger>>>,
    shape: <T::Dim as Dimension>::Larger,
    overwrite: Cell<bool>,
    left: Rc<T>,
    axis: usize,
}

impl<T> StackBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    pub fn new<U: Data<Dim = T::Dim>>(left: Rc<T>, right: Rc<U>, axis: usize) -> Self {
        let gradient = stack(Axis(axis), &[left.gradient().view(), right.data().view()]).unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            left,
            axis,
        }
    }
}

impl<T> Gradient for StackBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    type Dim = <T::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T> Overwrite for StackBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T> Backward for StackBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn backward(&self) {
        push_gradient(
            &*self.left,
            self.gradient()
                .axis_iter(Axis(self.axis))
                .next()
                .unwrap()
                .into_dimensionality::<T::Dim>()
                .unwrap(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StackBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct StackBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<<T::Dim as Dimension>::Larger>>>,
    shape: <T::Dim as Dimension>::Larger,
    overwrite: Cell<bool>,
    right: Rc<T>,
    axis: usize,
}

impl<T> StackBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    pub fn new<U: Data<Dim = T::Dim>>(left: Rc<U>, right: Rc<T>, axis: usize) -> Self {
        let gradient = stack(Axis(axis), &[left.data().view(), right.gradient().view()]).unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            right,
            axis,
        }
    }
}

impl<T> Gradient for StackBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    type Dim = <T::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T> Overwrite for StackBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T> Backward for StackBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn backward(&self) {
        push_gradient(
            &*self.right,
            self.gradient()
                .axis_iter(Axis(self.axis))
                .nth(1)
                .unwrap()
                .into_dimensionality::<T::Dim>()
                .unwrap(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ UnsqueezeBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct UnsqueezeBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<<T::Dim as Dimension>::Larger>>>,
    shape: <T::Dim as Dimension>::Larger,
    overwrite: Cell<bool>,
    operand: Rc<T>,
    axis: usize,
}

impl<T: Gradient + Overwrite> UnsqueezeBackward<T> {
    pub fn new(operand: Rc<T>, axis: usize) -> Self {
        let gradient = Tensor::zeros(operand.gradient().raw_dim().insert_axis(Axis(axis)));
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            operand,
            axis,
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for UnsqueezeBackward<T> {
    type Dim = <T::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for UnsqueezeBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for UnsqueezeBackward<T> {
    fn backward(&self) {
        push_gradient(
            &*self.operand,
            self.gradient()
                .axis_iter(Axis(self.axis))
                .next()
                .unwrap()
                .into_dimensionality::<T::Dim>()
                .unwrap(),
        );
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ChunkBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ChunkBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    operand: Rc<T>,
    chunk_id: usize,
}

impl<T: Gradient + Overwrite> ChunkBackward<T> {
    pub fn new(operand: Rc<T>, grad_chunk: Tensor<T::Dim>, chunk_id: usize) -> Self {
        let shape = grad_chunk.raw_dim();

        Self {
            gradient: RefCell::new(Some(grad_chunk)),
            shape,
            overwrite: Cell::new(true),
            operand,
            chunk_id,
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for ChunkBackward<T> {
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for ChunkBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for ChunkBackward<T> {
    fn backward(&self) {
        let mut diff_operand = self.operand.gradient_mut();
        let grad = self.gradient();
        let mut op_gradient_chunk = diff_operand
            .exact_chunks_mut(self.shape.clone())
            .into_iter()
            .skip(self.chunk_id)
            .take(1)
            .next()
            .unwrap();

        let zip = Zip::from(&mut op_gradient_chunk).and(&*grad);
        if self.operand.can_overwrite() {
            zip.for_each(|dest, src| *dest = *src);
            self.operand.set_overwrite(false);
        } else {
            zip.for_each(|dest, src| *dest += src);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DropoutBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<Dropout<U>>,
    p: f64,
    train: Rc<Cell<bool>>,
}

impl<T, U> DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(
        diff_operand: Rc<T>,
        no_diff_operand: Rc<Dropout<U>>,
        p: f64,
        forward_status: Rc<Cell<bool>>,
    ) -> DropoutBackward<T, U> {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            p,
            train: forward_status,
        }
    }
}

impl<T, U> Gradient for DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        if self.train.get() {
            let mut op_grad = self.diff_operand.gradient_mut();
            let grad = self.gradient();
            let p = &self.p;
            if (*p - 1.).abs() <= f64::EPSILON {
                if self.diff_operand.can_overwrite() {
                    Zip::from(&mut *op_grad).for_each(|op_grad_el| *op_grad_el = 0.);
                    self.diff_operand.set_overwrite(false);
                }
            } else if *p <= f64::EPSILON {
                let zip = Zip::from(&mut *op_grad).and(&*grad);
                if self.diff_operand.can_overwrite() {
                    zip.for_each(|op_grad_el, grad_el| *op_grad_el = *grad_el);
                    self.diff_operand.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el);
                }
            } else {
                let noise = self.no_diff_operand.noise();
                let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*noise);
                if self.diff_operand.can_overwrite() {
                    zip.for_each(|op_grad_el, grad_el, noise_el| *op_grad_el = *grad_el * noise_el);
                    self.diff_operand.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad_el, grad_el, noise_el| {
                        *op_grad_el += *grad_el * noise_el
                    });
                }
            }
        } else {
            self.diff_operand.gradient_mut().assign(&*self.gradient());
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::StrideShape;
    #[cfg(feature = "blas")]
    extern crate blas_src;

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
        Input::new(new_tensor(shape, elems)).node
    }

    fn new_backward_input<D, Sh>(shape: Sh, elems: Vec<f32>) -> Rc<InputBackward<D>>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Rc::new(Input::new(new_tensor(shape, elems)).node.differentiable())
    }

    fn new_tensor<D, Sh>(shape: Sh, elems: Vec<f32>) -> Tensor<D>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Tensor::from_shape_vec(shape, elems).unwrap()
    }

    mod backward_input {
        use super::*;

        #[test]
        fn creation() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            assert_eq!(*input.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*input.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(input.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let input = new_backward_input((3, 3), vec![0.; 9]);

            input.set_overwrite(true);
            assert_eq!(input.can_overwrite(), true);

            input.set_overwrite(true);
            assert_eq!(input.can_overwrite(), true);

            input.set_overwrite(false);
            assert_eq!(input.can_overwrite(), false);

            input.set_overwrite(false);
            assert_eq!(input.can_overwrite(), false);
        }
    }

    mod backward_negation {
        use super::*;

        #[test]
        fn creation() {
            let node = NegationBackward::new(new_backward_input((3, 3), vec![0.; 9]));

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = NegationBackward::new(input.clone());

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(input.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(input.can_overwrite(), false);

            input.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(input.can_overwrite(), true);

            input.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(input.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), false);

            input.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), false);

            input.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = NegationBackward::new(input.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![-1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![-2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            input.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![-1.; 9]));
        }
    }

    mod backward_addition {
        use super::*;

        #[test]
        fn creation() {
            let node = AdditionBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                new_backward_input((3, 3), vec![0.; 9]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = AdditionBackward::new(lhs.clone(), rhs.clone());

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = AdditionBackward::new(lhs.clone(), rhs.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        }

        #[test]
        fn backward_broadcast_left() {
            let lhs = new_backward_input(3, vec![0.; 3]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = AdditionBackward::new(lhs.clone(), rhs.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![3.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![6.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![3.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        }

        #[test]
        fn backward_broadcast_right() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input(3, vec![0.; 3]);
            let node = AdditionBackward::new(lhs.clone(), rhs.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![3.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![6.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![3.; 3]));
        }

        #[test]
        fn backward_unary() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let not_diff = new_input((3, 3), vec![0.; 9]);
            let node = AdditionBackwardUnary::new(diff.clone(), not_diff);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        }

        #[test]
        fn backward_unary_broadcast() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let not_diff = new_input((3, 3), vec![0.; 9]);
            let node = AdditionBackwardUnary::new(diff.clone(), not_diff);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![6.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3.; 3]));
        }
    }

    mod backward_subtraction {
        use super::*;

        #[test]
        fn creation() {
            let node = SubtractionBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                new_backward_input((3, 3), vec![0.; 9]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = SubtractionBackward::new(lhs.clone(), rhs.clone());

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = SubtractionBackward::new(lhs.clone(), rhs.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-1.; 9]));
        }

        #[test]
        fn backward_broadcast_left() {
            let lhs = new_backward_input(3, vec![0.; 3]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = SubtractionBackward::new(lhs.clone(), rhs.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![3.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![6.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![3.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-1.; 9]));
        }

        #[test]
        fn backward_broadcast_right() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input(3, vec![0.; 3]);
            let node = SubtractionBackward::new(lhs.clone(), rhs.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![-3.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![-6.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![-3.; 3]));
        }

        #[test]
        fn backward_left() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = SubtractionBackwardLeft::new(diff.clone(), new_input((3, 3), vec![0.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        }

        #[test]
        fn backward_left_broadcast() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = SubtractionBackwardLeft::new(diff.clone(), new_input((3, 3), vec![0.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![6.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3.; 3]));
        }

        #[test]
        fn backward_right() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = SubtractionBackwardRight::new(diff.clone(), new_input((3, 3), vec![0.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-1.; 9]));
        }

        #[test]
        fn backward_right_broadcast() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = SubtractionBackwardRight::new(diff.clone(), new_input((3, 3), vec![0.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-3.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-6.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-3.; 3]));
        }
    }

    mod backward_multiplication {
        use super::*;

        #[test]
        fn creation() {
            let node = MultiplicationBackward::new(
                new_input((3, 3), vec![3.; 9]),
                new_backward_input((3, 3), vec![0.; 9]),
                new_input((3, 3), vec![5.; 9]),
                new_backward_input((3, 3), vec![0.; 9]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = MultiplicationBackward::new(
                new_input((3, 3), vec![3.; 9]),
                lhs.clone(),
                new_input((3, 3), vec![5.; 9]),
                rhs.clone(),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = MultiplicationBackward::new(
                new_input((3, 3), vec![3.; 9]),
                lhs.clone(),
                new_input((3, 3), vec![5.; 9]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![5.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![3.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![10.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![6.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![5.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![3.; 9]));
        }

        #[test]
        fn backward_broadcast_left() {
            let lhs = new_backward_input(3, vec![0.; 3]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = MultiplicationBackward::new(
                new_input(3, vec![3.; 3]),
                lhs.clone(),
                new_input((3, 3), vec![5.; 9]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![15.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![3.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![30.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![6.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![15.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![3.; 9]));
        }

        #[test]
        fn backward_broadcast_right() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input(3, vec![0.; 3]);
            let node = MultiplicationBackward::new(
                new_input((3, 3), vec![3.; 9]),
                lhs.clone(),
                new_input(3, vec![5.; 3]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![5.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![9.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![10.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![18.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![5.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![9.; 3]));
        }

        #[test]
        fn backward_unary() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node =
                MultiplicationBackwardUnary::new(diff.clone(), new_input((3, 3), vec![5.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![5.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![10.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![5.; 9]));
        }

        #[test]
        fn backward_unary_broadcast() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node =
                MultiplicationBackwardUnary::new(diff.clone(), new_input((3, 3), vec![5.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![15.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![30.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![15.; 3]));
        }
    }

    mod backward_division {
        use super::*;

        #[test]
        fn creation() {
            let node = DivisionBackward::new(
                new_input((3, 3), vec![3.; 9]),
                new_backward_input((3, 3), vec![0.; 9]),
                new_input((3, 3), vec![5.; 9]),
                new_backward_input((3, 3), vec![0.; 9]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = DivisionBackward::new(
                new_input((3, 3), vec![3.; 9]),
                lhs.clone(),
                new_input((3, 3), vec![5.; 9]),
                rhs.clone(),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = DivisionBackward::new(
                new_input((3, 3), vec![3.; 9]),
                lhs.clone(),
                new_input((3, 3), vec![5.; 9]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.4; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.24; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));
        }

        #[test]
        fn backward_broadcast_left() {
            let lhs = new_backward_input(3, vec![0.; 3]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = DivisionBackward::new(
                new_input(3, vec![3.; 3]),
                lhs.clone(),
                new_input((3, 3), vec![5.; 9]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![0.6; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![1.2; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.24; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![0.6; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));
        }

        #[test]
        fn backward_broadcast_right() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input(3, vec![0.; 3]);
            let node = DivisionBackward::new(
                new_input((3, 3), vec![3.; 9]),
                lhs.clone(),
                new_input(3, vec![5.; 3]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![-0.36; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.4; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![-0.72; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![-0.36; 3]));
        }

        #[test]
        fn backward_left() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = DivisionBackwardLeft::new(diff.clone(), new_input((3, 3), vec![5.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![0.2; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![0.4; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
        }

        #[test]
        fn backward_left_broadcast() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = DivisionBackwardLeft::new(diff.clone(), new_input((3, 3), vec![5.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0.6; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![1.2; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0.6; 3]));
        }

        #[test]
        fn backward_right() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = DivisionBackwardRight::new(
                new_input((3, 3), vec![3.; 9]),
                new_input((3, 3), vec![5.; 9]),
                diff.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-0.24; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));
        }

        #[test]
        fn backward_right_broadcast() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = DivisionBackwardRight::new(
                new_input((3, 3), vec![3.; 9]),
                new_input((3, 3), vec![5.; 9]),
                diff.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-0.36; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-0.72; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-0.36; 3]));
        }
    }

    mod backward_matrix_matrix_mul {
        use super::*;

        #[test]
        fn creation() {
            let node = MatrixMatrixMulBackward::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                new_backward_input((3, 3), vec![0.; 9]),
                new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]),
                new_backward_input((3, 3), vec![0.; 9]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = MatrixMatrixMulBackward::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                lhs.clone(),
                new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]),
                rhs.clone(),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = MatrixMatrixMulBackward::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                lhs.clone(),
                new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![33., 42., 51., 33., 42., 51., 33., 42., 51.]),
            );
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((3, 3), vec![12., 12., 12., 15., 15., 15., 18., 18., 18.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![66., 84., 102., 66., 84., 102., 66., 84., 102.]),
            );
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((3, 3), vec![24., 24., 24., 30., 30., 30., 36., 36., 36.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![33., 42., 51., 33., 42., 51., 33., 42., 51.]),
            );
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((3, 3), vec![12., 12., 12., 15., 15., 15., 18., 18., 18.]),
            );
        }

        #[test]
        fn backward_left() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = MatrixMatrixMulBackwardLeft::new(
                diff.clone(),
                new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![33., 42., 51., 33., 42., 51., 33., 42., 51.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![66., 84., 102., 66., 84., 102., 66., 84., 102.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![33., 42., 51., 33., 42., 51., 33., 42., 51.]),
            );
        }

        #[test]
        fn backward_right() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = MatrixMatrixMulBackwardRight::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                diff.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![12., 12., 12., 15., 15., 15., 18., 18., 18.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![24., 24., 24., 30., 30., 30., 36., 36., 36.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![12., 12., 12., 15., 15., 15., 18., 18., 18.]),
            );
        }
    }

    mod backward_matrix_matrix_mul_t {
        use super::*;

        #[test]
        fn creation() {
            let node = MatrixMatrixMulTBackward::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                new_backward_input((3, 3), vec![0.; 9]),
                new_input((2, 3), vec![10., 11., 12., 13., 14., 15.]),
                new_backward_input((2, 3), vec![0.; 6]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 2), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 2), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((2, 3), vec![0.; 6]);
            let node = MatrixMatrixMulTBackward::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                lhs.clone(),
                new_input((2, 3), vec![10., 11., 12., 13., 14., 15.]),
                rhs.clone(),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((2, 3), vec![0.; 6]);
            let node = MatrixMatrixMulTBackward::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                lhs.clone(),
                new_input((2, 3), vec![10., 11., 12., 13., 14., 15.]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 2), vec![1.; 6]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 2), vec![1.; 6]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![23., 25., 27., 23., 25., 27., 23., 25., 27.]),
            );
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((2, 3), vec![12., 15., 18., 12., 15., 18.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![46., 50., 54., 46., 50., 54., 46., 50., 54.]),
            );
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((2, 3), vec![24., 30., 36., 24., 30., 36.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![23., 25., 27., 23., 25., 27., 23., 25., 27.]),
            );
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((2, 3), vec![12., 15., 18., 12., 15., 18.]),
            );
        }

        #[test]
        fn backward_left() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = MatrixMatrixMulTBackwardLeft::new(
                diff.clone(),
                new_input((2, 3), vec![10., 11., 12., 13., 14., 15.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 2), vec![1.; 6]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 2), vec![1.; 6]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![23., 25., 27., 23., 25., 27., 23., 25., 27.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![46., 50., 54., 46., 50., 54., 46., 50., 54.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![23., 25., 27., 23., 25., 27., 23., 25., 27.]),
            );
        }

        #[test]
        fn backward_right() {
            let diff = new_backward_input((2, 3), vec![0.; 6]);
            let node = MatrixMatrixMulTBackwardRight::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                diff.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 2), vec![1.; 6]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 2), vec![1.; 6]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((2, 3), vec![12., 15., 18., 12., 15., 18.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((2, 3), vec![24., 30., 36., 24., 30., 36.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((2, 3), vec![12., 15., 18., 12., 15., 18.]),
            );
        }
    }

    mod backward_matrix_vector_mul {
        use super::*;

        #[test]
        fn creation() {
            let node = MatrixVectorMulBackward::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                new_backward_input((3, 3), vec![0.; 9]),
                new_input(3, vec![1., 2., 3.]),
                new_backward_input(3, vec![0.; 3]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input(3, vec![0.; 3]);
            let node = MatrixVectorMulBackward::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                lhs.clone(),
                new_input(3, vec![1., 2., 3.]),
                rhs.clone(),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input(3, vec![0.; 3]);
            let node = MatrixVectorMulBackward::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                lhs.clone(),
                new_input(3, vec![1., 2., 3.]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]),
            );
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![12., 15., 18.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![2., 4., 6., 2., 4., 6., 2., 4., 6.]),
            );
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![24., 30., 36.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]),
            );
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![12., 15., 18.]));
        }

        #[test]
        fn backward_left() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node =
                MatrixVectorMulBackwardLeft::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![2., 4., 6., 2., 4., 6., 2., 4., 6.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 1., 2., 3., 1., 2., 3.]),
            );
        }

        #[test]
        fn backward_right() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = MatrixVectorMulBackwardRight::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                diff.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![12., 15., 18.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![24., 30., 36.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![12., 15., 18.]));
        }
    }

    mod backward_vector_matrix_mul {
        use super::*;

        #[test]
        fn creation() {
            let node = VectorMatrixMulBackward::new(
                new_input(3, vec![1., 2., 3.]),
                new_backward_input(3, vec![0.; 3]),
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                new_backward_input((3, 3), vec![0.; 9]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input(3, vec![0.; 3]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = VectorMatrixMulBackward::new(
                new_input(3, vec![1., 2., 3.]),
                lhs.clone(),
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                rhs.clone(),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let lhs = new_backward_input(3, vec![0.; 3]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = VectorMatrixMulBackward::new(
                new_input(3, vec![1., 2., 3.]),
                lhs.clone(),
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![6., 15., 24.]));
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((3, 3), vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![12., 30., 48.]));
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((3, 3), vec![2., 2., 2., 4., 4., 4., 6., 6., 6.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![6., 15., 24.]));
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((3, 3), vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]),
            );
        }

        #[test]
        fn backward_left() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = VectorMatrixMulBackwardLeft::new(
                diff.clone(),
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![6., 15., 24.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![12., 30., 48.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![6., 15., 24.]));
        }

        #[test]
        fn backward_right() {
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node =
                VectorMatrixMulBackwardRight::new(new_input(3, vec![1., 2., 3.]), diff.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![2., 2., 2., 4., 4., 4., 6., 6., 6.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]),
            );
        }
    }

    mod backward_vector_vector_mul {
        use super::*;

        #[test]
        fn creation() {
            let node = VectorVectorMulBackward::new(
                new_input(3, vec![1., 2., 3.]),
                new_backward_input(3, vec![0.; 3]),
                new_input(3, vec![4., 5., 6.]),
                new_backward_input(3, vec![0.; 3]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(1, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(1, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input(3, vec![0.; 3]);
            let rhs = new_backward_input(3, vec![0.; 3]);
            let node = VectorVectorMulBackward::new(
                new_input(3, vec![1., 2., 3.]),
                lhs.clone(),
                new_input(3, vec![4., 5., 6.]),
                rhs.clone(),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let lhs = new_backward_input(3, vec![0.; 3]);
            let rhs = new_backward_input(3, vec![0.; 3]);
            let node = VectorVectorMulBackward::new(
                new_input(3, vec![1., 2., 3.]),
                lhs.clone(),
                new_input(3, vec![4., 5., 6.]),
                rhs.clone(),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![4., 5., 6.]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![1., 2., 3.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![8., 10., 12.]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![2., 4., 6.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![4., 5., 6.]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![1., 2., 3.]));
        }

        #[test]
        fn backward_unary() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node =
                VectorVectorMulBackwardUnary::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![1., 2., 3.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![2., 4., 6.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![1., 2., 3.]));
        }
    }

    mod backward_power {
        use super::*;

        #[test]
        fn creation() {
            let node = PowerBackward::new(
                new_backward_input(3, vec![0.; 3]),
                new_input(3, vec![1., 2., 3.]),
                3,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = PowerBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]), 3);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = PowerBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]), 3);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3., 12., 27.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![6., 24., 54.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3., 12., 27.]));
        }

        #[test]
        fn backward_negative_exp() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = PowerBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]), -3);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![-3., -0.1875, -0.037037]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![-6., -0.375, -0.074075]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![-3., -0.1875, -0.037037]),
            );
        }
    }

    mod backward_sqrt {
        use super::*;

        #[test]
        fn creation() {
            let node = SqrtBackward::new(
                new_backward_input(3, vec![0.; 3]),
                new_input(3, vec![1., 2., 3.]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = SqrtBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = SqrtBackward::new(diff.clone(), new_input(3, vec![1., 1.4142, 1.7321]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0.5, 0.3536, 0.2887]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![1., 0.7071, 0.5774]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0.5, 0.3536, 0.2887]));
        }
    }

    mod backward_sum {
        use super::*;

        #[test]
        fn creation() {
            let node = SumBackward::new(new_backward_input((10, 10), vec![0.; 100]));

            assert_eq!(*node.gradient(), Tensor::from_elem(1, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(1, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input((10, 10), vec![0.; 100]);
            let node = SumBackward::new(diff.clone());

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input((10, 10), vec![0.; 100]);
            let node = SumBackward::new(diff.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(1, vec![1.]);
            assert_almost_equals(&*node.gradient(), &new_tensor(1, vec![1.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![1.; 100]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![2.; 100]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![1.; 100]));
        }
    }

    mod backward_mean {
        use super::*;

        #[test]
        fn creation() {
            let node = MeanBackward::new(new_backward_input((10, 10), vec![0.; 100]));

            assert_eq!(*node.gradient(), Tensor::from_elem(1, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(1, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input((10, 10), vec![0.; 100]);
            let node = MeanBackward::new(diff.clone());

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input((10, 10), vec![0.; 100]);
            let node = MeanBackward::new(diff.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(1, vec![1.]);
            assert_almost_equals(&*node.gradient(), &new_tensor(1, vec![1.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![0.01; 100]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![0.02; 100]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![0.01; 100]));
        }
    }

    mod backward_logn {
        use super::*;

        #[test]
        fn creation() {
            let node = LognBackward::new(
                new_backward_input(3, vec![0.; 3]),
                new_input(3, vec![1., 2., 3.]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = LognBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = LognBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![1., 0.5, 0.33333]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![2., 1., 0.66667]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![1., 0.5, 0.33333]));
        }
    }

    mod backward_relu {
        use super::*;

        #[test]
        fn creation() {
            let node = ReLUBackward::new(
                new_backward_input(3, vec![0.; 3]),
                new_input(3, vec![-1., 2., -3.]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = ReLUBackward::new(diff.clone(), new_input(3, vec![-1., 2., -3.]));

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = ReLUBackward::new(diff.clone(), new_input(3, vec![-1., 2., -3.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0., 1., 0.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0., 2., 0.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0., 1., 0.]));
        }
    }

    mod backward_leaky_relu {
        use super::*;

        #[test]
        fn creation() {
            let node = LeakyReLUBackward::new(
                new_backward_input(3, vec![0.; 3]),
                new_input(3, vec![-1., 2., -3.]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = LeakyReLUBackward::new(diff.clone(), new_input(3, vec![-1., 2., -3.]));

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = LeakyReLUBackward::new(diff.clone(), new_input(3, vec![-1., 2., -3.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0.01, 1., 0.01]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0.02, 2., 0.02]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0.01, 1., 0.01]));
        }
    }

    mod backward_softplus {
        use super::*;

        #[test]
        fn creation() {
            let node = SoftPlusBackward::new(
                new_backward_input(3, vec![0.; 3]),
                new_input(3, vec![1., 2., 3.]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = SoftPlusBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = SoftPlusBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.7311, 0.8808, 0.9526]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![1.4622, 1.7616, 1.9052]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.7311, 0.8808, 0.9526]),
            );
        }
    }

    mod backward_sigmoid {
        use super::*;
        use crate::variable::node::{Forward, Sigmoid};

        #[test]
        fn creation() {
            let node = SigmoidBackward::new(
                new_backward_input(3, vec![0.; 3]),
                Rc::new(Sigmoid::new(new_input(3, vec![1., 2., 3.]))),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = SigmoidBackward::new(
                diff.clone(),
                Rc::new(Sigmoid::new(new_input(3, vec![1., 2., 3.]))),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let not_diff = Rc::new(Sigmoid::new(new_input(3, vec![1., 2., 3.])));
            not_diff.forward();
            let node = SigmoidBackward::new(diff.clone(), not_diff);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.1966, 0.105, 0.0452]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.3932, 0.21, 0.0904]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.1966, 0.105, 0.0452]),
            );
        }
    }

    mod backward_tanh {
        use super::*;
        use crate::variable::node::{Forward, TanH};

        #[test]
        fn creation() {
            let node = TanHBackward::new(
                new_backward_input(3, vec![0.; 3]),
                Rc::new(TanH::new(new_input(3, vec![1., 2., 3.]))),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = TanHBackward::new(
                diff.clone(),
                Rc::new(TanH::new(new_input(3, vec![1., 2., 3.]))),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let not_diff = Rc::new(TanH::new(new_input(3, vec![1., 2., 3.])));
            not_diff.forward();
            let node = TanHBackward::new(diff.clone(), not_diff);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.4199, 0.07065, 0.009865]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.8398, 0.1413, 0.01973]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.4199, 0.07065, 0.009865]),
            );
        }
    }

    mod backward_exp {
        use super::*;
        use crate::variable::node::{Exp, Forward};

        #[test]
        fn creation() {
            let node = ExpBackward::new(
                new_backward_input(3, vec![0.; 3]),
                Rc::new(Exp::new(new_input(3, vec![1., 2., 3.]))),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = ExpBackward::new(
                diff.clone(),
                Rc::new(Exp::new(new_input(3, vec![1., 2., 3.]))),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[allow(clippy::clippy::approx_constant)]
        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let not_diff = Rc::new(Exp::new(new_input(3, vec![1., 2., 3.])));
            not_diff.forward();
            let node = ExpBackward::new(diff.clone(), not_diff);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![2.7183, 7.3891, 20.0855]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![5.4366, 14.7782, 40.171]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![2.7183, 7.3891, 20.0855]),
            );
        }
    }

    mod backward_softmax {
        use super::*;
        use crate::variable::node::{Forward, Softmax};

        #[test]
        fn creation() {
            let axis = 0;
            let node = SoftmaxBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                Rc::new(Softmax::new(
                    new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                    axis,
                )),
                axis,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let axis = 0;
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = SoftmaxBackward::new(
                diff.clone(),
                Rc::new(Softmax::new(
                    new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                    axis,
                )),
                axis,
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward_rows() {
            let axis = 0;
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let not_diff = Rc::new(Softmax::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                axis,
            ));
            not_diff.forward();
            let node_b = SoftmaxBackward::new(diff.clone(), not_diff, axis);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node_b.gradient_mut() = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            assert_almost_equals(
                &*node_b.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node_b.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.01376, -0.01376, -0.01376, -0.13455, -0.13455, -0.13455, 0.148323,
                        0.148323, 0.148323,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node_b.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.02752, -0.02752, -0.02752, -0.2691, -0.2691, -0.2691, 0.296646,
                        0.296646, 0.296646,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node_b.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.01376, -0.01376, -0.01376, -0.13455, -0.13455, -0.13455, 0.148323,
                        0.148323, 0.148323,
                    ],
                ),
            );
        }

        #[test]
        fn backward_columns() {
            let axis = 1;
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let not_diff = Rc::new(Softmax::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                axis,
            ));
            not_diff.forward();
            let node = SoftmaxBackward::new(diff.clone(), not_diff, axis);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            assert_almost_equals(
                &*node.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.1418, -0.1408, 0.2826, -0.1418, -0.1408, 0.2826, -0.1418, -0.1408,
                        0.2826,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.2836, -0.2815, 0.5652, -0.2836, -0.2815, 0.5652, -0.2836, -0.2815,
                        0.5652,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.1418, -0.1408, 0.2826, -0.1418, -0.1408, 0.2826, -0.1418, -0.1408,
                        0.2826,
                    ],
                ),
            );
        }
    }

    mod backward_logsoftmax {
        use super::*;
        use crate::variable::node::{Forward, LogSoftmax};

        #[test]
        fn creation() {
            let axis = 0;
            let node = LogSoftmaxBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                Rc::new(LogSoftmax::new(
                    new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                    axis,
                )),
                axis,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let axis = 0;
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = LogSoftmaxBackward::new(
                diff.clone(),
                Rc::new(LogSoftmax::new(
                    new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                    axis,
                )),
                axis,
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward_rows() {
            let axis = 0;
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let not_diff = Rc::new(LogSoftmax::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                axis,
            ));
            not_diff.forward();
            let node = LogSoftmaxBackward::new(diff.clone(), not_diff, axis);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            assert_almost_equals(
                &*node.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.9717, 1.9647, 2.9576, 3.4322, 4.2903, 5.1483, -4.4040, -6.2550, -8.1059,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        1.9435, 3.9293, 5.9152, 6.8645, 8.5806, 10.2967, -8.8079, -12.5099,
                        -16.2119,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.9717, 1.9647, 2.9576, 3.4322, 4.2903, 5.1483, -4.4040, -6.2550, -8.1059,
                    ],
                ),
            );
        }

        #[test]
        fn backward_columns() {
            let axis = 1;
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let not_diff = Rc::new(LogSoftmax::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                axis,
            ));
            not_diff.forward();
            let node = LogSoftmaxBackward::new(diff.clone(), not_diff, axis);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            assert_almost_equals(
                &*node.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.4598, 0.5316, -0.9914, 2.6495, 1.3291, -3.9786, 4.8393, 2.1265, -6.9658,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.9196, 1.0633, -1.9829, 5.2991, 2.6581, -7.9572, 9.6785, 4.2530, -13.9316,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.4598, 0.5316, -0.9914, 2.6495, 1.3291, -3.9786, 4.8393, 2.1265, -6.9658,
                    ],
                ),
            );
        }
    }

    mod backward_transpose {
        use super::*;

        #[test]
        fn creation() {
            let node = TransposeBackward::new(new_backward_input((4, 3), vec![0.; 12]));

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 4), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 4), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = TransposeBackward::new(diff.clone());

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = TransposeBackward::new(diff.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 4), vec![1.; 12]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 4), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }
    }

    mod backward_chunks {
        use super::*;

        #[test]
        fn creation() {
            let node = ChunkBackward::new(
                new_backward_input((4, 3), vec![0.; 12]),
                Tensor::zeros((1, 3)),
                0,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((1, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((1, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = ChunkBackward::new(diff.clone(), Tensor::zeros((1, 3)), 0);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let chunk_0 = ChunkBackward::new(diff.clone(), Tensor::zeros((1, 3)), 0);
            let chunk_1 = ChunkBackward::new(diff.clone(), Tensor::zeros((1, 3)), 1);
            let chunk_2 = ChunkBackward::new(diff.clone(), Tensor::zeros((1, 3)), 2);
            let chunk_3 = ChunkBackward::new(diff.clone(), Tensor::zeros((1, 3)), 3);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *chunk_0.gradient_mut() = new_tensor((1, 3), vec![1.; 3]);
            assert_almost_equals(&*chunk_0.gradient(), &new_tensor((1, 3), vec![1.; 3]));

            *chunk_1.gradient_mut() = new_tensor((1, 3), vec![2.; 3]);
            assert_almost_equals(&*chunk_1.gradient(), &new_tensor((1, 3), vec![2.; 3]));

            *chunk_2.gradient_mut() = new_tensor((1, 3), vec![3.; 3]);
            assert_almost_equals(&*chunk_2.gradient(), &new_tensor((1, 3), vec![3.; 3]));

            *chunk_3.gradient_mut() = new_tensor((1, 3), vec![4.; 3]);
            assert_almost_equals(&*chunk_3.gradient(), &new_tensor((1, 3), vec![4.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            chunk_0.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            );

            chunk_1.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![1., 1., 1., 2., 2., 2., 0., 0., 0., 0., 0., 0.]),
            );

            chunk_2.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![1., 1., 1., 2., 2., 2., 3., 3., 3., 0., 0., 0.]),
            );

            chunk_3.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            chunk_0.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![2., 2., 2., 2., 2., 2., 3., 3., 3., 4., 4., 4.]),
            );

            chunk_1.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![2., 2., 2., 4., 4., 4., 3., 3., 3., 4., 4., 4.]),
            );

            chunk_2.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![2., 2., 2., 4., 4., 4., 6., 6., 6., 4., 4., 4.]),
            );

            chunk_3.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![2., 2., 2., 4., 4., 4., 6., 6., 6., 8., 8., 8.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            chunk_0.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![1., 1., 1., 4., 4., 4., 6., 6., 6., 8., 8., 8.]),
            );

            diff.set_overwrite(true);
            chunk_1.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![1., 1., 1., 2., 2., 2., 6., 6., 6., 8., 8., 8.]),
            );

            diff.set_overwrite(true);
            chunk_2.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![1., 1., 1., 2., 2., 2., 3., 3., 3., 8., 8., 8.]),
            );

            diff.set_overwrite(true);
            chunk_3.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((4, 3), vec![1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]),
            );
        }
    }

    mod backward_unsqueeze {
        use super::*;

        #[test]
        fn creation() {
            let node = UnsqueezeBackward::new(new_backward_input((4, 3), vec![0.; 12]), 0);

            assert_eq!(*node.gradient(), Tensor::from_elem((1, 4, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((1, 4, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = UnsqueezeBackward::new(diff.clone(), 0);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), false);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            diff.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(diff.can_overwrite(), false);
        }

        #[test]
        fn backward_rows() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = UnsqueezeBackward::new(diff.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((1, 4, 3), vec![1.; 12]);
            assert_almost_equals(&*node.gradient(), &new_tensor((1, 4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_columns() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = UnsqueezeBackward::new(diff.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 1, 3), vec![1.; 12]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 1, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_depths() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = UnsqueezeBackward::new(diff.clone(), 2);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 3, 1), vec![1.; 12]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 3, 1), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }
    }

    mod backward_cat {
        use super::*;

        #[test]
        fn creation() {
            let node = ConcatenateBackward::new(
                new_backward_input((4, 3), vec![0.; 12]),
                new_backward_input((4, 2), vec![0.; 8]),
                1,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((4, 5), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((4, 5), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 2), vec![0.; 8]);
            let node = ConcatenateBackward::new(lhs.clone(), rhs.clone(), 1);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward_rows() {
            let lhs = new_backward_input((3, 4), vec![0.; 12]);
            let rhs = new_backward_input((2, 4), vec![0.; 8]);
            let node = ConcatenateBackward::new(lhs.clone(), rhs.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![2.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![2.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));
        }

        #[test]
        fn backward_columns() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 2), vec![0.; 8]);
            let node = ConcatenateBackward::new(lhs.clone(), rhs.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 5), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 5), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![2.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));
        }

        #[test]
        fn backward_left_rows() {
            let lhs = new_backward_input((3, 4), vec![0.; 12]);
            let node = ConcatenateBackwardLeft::new(lhs.clone(), new_input((2, 4), vec![0.; 8]), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));
        }

        #[test]
        fn backward_left_columns() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = ConcatenateBackwardLeft::new(lhs.clone(), new_input((4, 2), vec![0.; 8]), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 5), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 5), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_right_rows() {
            let rhs = new_backward_input((2, 4), vec![0.; 8]);
            let node =
                ConcatenateBackwardRight::new(new_input((3, 4), vec![0.; 12]), rhs.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![2.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));
        }

        #[test]
        fn backward_right_columns() {
            let rhs = new_backward_input((4, 2), vec![0.; 8]);
            let node =
                ConcatenateBackwardRight::new(new_input((4, 3), vec![0.; 12]), rhs.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 5), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 5), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![2.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));
        }
    }

    mod backward_stack {
        use super::*;

        #[test]
        fn creation() {
            let node = StackBackward::new(
                new_backward_input((4, 3), vec![0.; 12]),
                new_backward_input((4, 3), vec![0.; 12]),
                0,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((2, 4, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((2, 4, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackward::new(lhs.clone(), rhs.clone(), 0);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            lhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), false);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            rhs.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), true);
            assert_eq!(rhs.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(lhs.can_overwrite(), false);
            assert_eq!(rhs.can_overwrite(), false);
        }

        #[test]
        fn backward_rows() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackward::new(lhs.clone(), rhs.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_columns() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackward::new(lhs.clone(), rhs.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 2, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 2, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_left_rows() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackwardLeft::new(lhs.clone(), new_input((4, 3), vec![0.; 12]), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_left_columns() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackwardLeft::new(lhs.clone(), new_input((4, 3), vec![0.; 12]), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 2, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 2, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_right_rows() {
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackwardRight::new(new_input((4, 3), vec![0.; 12]), rhs.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_right_columns() {
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackwardRight::new(new_input((4, 3), vec![0.; 12]), rhs.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 2, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 2, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }
    }

    mod backward_dropout {
        use super::*;

        #[test]
        fn creation() {
            let node = DropoutBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                Rc::new(Dropout::new(
                    new_input((3, 3), vec![1.; 9]),
                    0.5,
                    Rc::new(Cell::new(true)),
                )),
                0.5,
                Rc::new(Cell::new(true)),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(node.can_overwrite(), true);
        }

        #[test]
        fn computation_state_transition() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = DropoutBackward::new(
                input.clone(),
                Rc::new(Dropout::new(
                    new_input((3, 3), vec![1.; 9]),
                    0.5,
                    Rc::new(Cell::new(true)),
                )),
                0.5,
                Rc::new(Cell::new(true)),
            );

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(input.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(input.can_overwrite(), false);

            input.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(input.can_overwrite(), true);

            input.set_overwrite(true);
            assert_eq!(node.can_overwrite(), true);
            assert_eq!(input.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), true);

            node.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), true);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), false);

            node.backward();
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), false);

            input.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), false);

            input.set_overwrite(false);
            assert_eq!(node.can_overwrite(), false);
            assert_eq!(input.can_overwrite(), false);
        }

        #[test]
        fn backward_p_one() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = DropoutBackward::new(
                input.clone(),
                Rc::new(Dropout::new(
                    new_input((3, 3), vec![1.; 9]),
                    1.,
                    Rc::new(Cell::new(true)),
                )),
                1.,
                Rc::new(Cell::new(true)),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![0.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![0.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            input.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![0.; 9]));
        }

        #[test]
        fn backward_p_zero() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = DropoutBackward::new(
                input.clone(),
                Rc::new(Dropout::new(
                    new_input((3, 3), vec![1.; 9]),
                    0.,
                    Rc::new(Cell::new(true)),
                )),
                0.,
                Rc::new(Cell::new(true)),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            input.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        }
    }
}
