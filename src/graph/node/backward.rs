use super::{
    super::{BroadTensor, Broadcasted, Tensor},
    broadcasted_zeros,
    forward::{Data, Forward, Input},
    DotDim,
};
use ndarray::{
    concatenate,
    linalg::{general_mat_mul, general_mat_vec_mul},
    stack, ArrayView1, Axis, DimMax, Dimension, Ix1, Ix2, RemoveAxis, Zip,
};
use std::cell::{Cell, Ref, RefCell, RefMut};
use std::rc::Rc;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Utility Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

fn sum_axis_inplace(arr: &mut ndarray::ArrayD<f32>, axis: ndarray::Axis) {
    let (first, rest) = arr.view_mut().split_at(axis, 1);
    ndarray::Zip::from(first.remove_axis(axis))
        .and(rest.lanes(axis))
        .for_each(|dst, src| *dst += src.sum());
    arr.index_axis_inplace(axis, 0);
}

pub fn reduce<D: ndarray::Dimension, E: ndarray::Dimension>(
    dest: &ndarray::Array<f32, D>,
    src: &ndarray::Array<f32, E>,
) -> ndarray::ArrayD<f32> {
    let mut dyn_rhs = src.clone().into_dyn();
    unsafe {
        while (*(&dyn_rhs as *const ndarray::ArrayD<f32>)).ndim() > dest.ndim() {
            sum_axis_inplace(&mut dyn_rhs, ndarray::Axis(0));
        }
        for (axis, size) in dest.shape().iter().enumerate() {
            if *size == 1 {
                sum_axis_inplace(&mut dyn_rhs, ndarray::Axis(axis));
                dyn_rhs.insert_axis_inplace(ndarray::Axis(axis));
            }
        }
        dyn_rhs
    }
}

fn fill_softmax_jacobian(jacobian: &mut Tensor<Ix2>, array: &ArrayView1<f32>) {
    for (row_idx, (mut row, row_val)) in jacobian
        .rows_mut()
        .into_iter()
        .zip(array.iter())
        .enumerate()
    {
        for (col_idx, (grad, col_val)) in row
            .as_slice_mut()
            .unwrap()
            .iter_mut()
            .zip(array.iter())
            .enumerate()
        {
            if row_idx == col_idx {
                *grad = row_val * (1.0 - col_val);
            } else {
                *grad = -row_val * col_val;
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub trait Gradient {
    type Dim: Dimension;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>>;

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>>;

    fn can_overwrite(&self) -> bool;

    fn was_overwritten(&self);
}

pub trait Backward {
    fn backward(&self) -> bool;
}

pub trait Differentiable {
    type Output: Backward + Gradient;

    fn differentiable(&self) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InputBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct InputBackward<D: Dimension> {
    gradient: RefCell<Tensor<D>>,
    can_overwrite: Cell<bool>,
}

impl<D: Dimension> InputBackward<D> {
    pub fn zero_grad(&self) {
        self.gradient.borrow_mut().map_inplace(|el| *el = 0.0);
    }
}

impl<D: Dimension> Gradient for InputBackward<D> {
    type Dim = D;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<D: Dimension> Backward for InputBackward<D> {
    fn backward(&self) -> bool {
        false
    }
}

impl<D: Dimension> Differentiable for Input<D> {
    type Output = InputBackward<D>;

    fn differentiable(&self) -> Self::Output {
        Self::Output {
            gradient: RefCell::new(Tensor::zeros(self.data().raw_dim())),
            can_overwrite: Cell::new(true),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NegationBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct NegationBackward<T: Gradient + Backward> {
    operand: Rc<T>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T: Gradient + Backward> NegationBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand.gradient().raw_dim()));
        Self {
            operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T: Gradient + Backward> Gradient for NegationBackward<T> {
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T: Gradient + Backward> Backward for NegationBackward<T> {
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut operand_grad = self.operand.gradient_mut();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *operand_grad).and(&*grad);
        if self.operand.can_overwrite() {
            zip.par_for_each(|dest, src| *dest = -src);
            self.operand.was_overwritten();
        } else {
            zip.par_for_each(|dest, src| *dest += -src);
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransposeBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct TransposeBackward<T: Gradient + Backward> {
    operand: Rc<T>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T: Gradient + Backward> TransposeBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand.gradient().raw_dim()));

        Self {
            operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T: Gradient + Backward> Gradient for TransposeBackward<T> {
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T: Gradient + Backward> Backward for TransposeBackward<T> {
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut operand_grad = self.operand.gradient_mut();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *operand_grad).and(grad.t());
        if self.operand.can_overwrite() {
            zip.par_for_each(|dest, src| *dest = *src);
            self.operand.was_overwritten();
        } else {
            zip.par_for_each(|dest, src| *dest += *src);
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdditionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    gradient: RefCell<Tensor<Broadcasted<Lhs::Dim, Rhs::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(&left.gradient(), &right.gradient()));

        Self {
            left,
            right,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<Lhs, Rhs> Backward for AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut lhs_grad, mut rhs_grad) =
            { (self.left.gradient_mut(), self.right.gradient_mut()) };
        let (gradient_lhs, gradient_rhs) = {
            let grad = self.gradient.borrow();
            (reduce(&*lhs_grad, &grad), reduce(&*rhs_grad, &grad))
        };

        if self.left.can_overwrite() {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&gradient_lhs.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.left.was_overwritten();
        } else {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&gradient_lhs.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        if self.right.can_overwrite() {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&gradient_rhs.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.left.was_overwritten();
        } else {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&gradient_rhs.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        true
    }
}

impl<Lhs, Rhs> Gradient for AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdditionBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct AdditionBackwardUnary<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    diff_operand: Rc<T>,
    gradient: RefCell<BroadTensor<T::Dim, U::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> AdditionBackwardUnary<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(
            &*diff_operand.gradient(),
            &*no_diff_operand.data(),
        ));

        Self {
            diff_operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Backward for AdditionBackwardUnary<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut operand_grad = self.diff_operand.gradient_mut();
        let gradient = reduce(&operand_grad, &*self.gradient.borrow());

        if self.diff_operand.can_overwrite() {
            Zip::from(&mut *operand_grad)
                .and_broadcast(&gradient.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.diff_operand.was_overwritten();
        } else {
            Zip::from(&mut *operand_grad)
                .and_broadcast(&gradient.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        true
    }
}

impl<T, U> Gradient for AdditionBackwardUnary<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    type Dim = Broadcasted<T::Dim, U::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }
    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SubtractionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    gradient: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(&left.gradient(), &right.gradient()));

        Self {
            left,
            right,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<Lhs, Rhs> Backward for SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut lhs_grad, mut rhs_grad) =
            { (self.left.gradient_mut(), self.right.gradient_mut()) };
        let (gradient_lhs, gradient_rhs) = {
            let grad = self.gradient.borrow();
            (reduce(&*lhs_grad, &grad), reduce(&*rhs_grad, &grad))
        };

        if self.left.can_overwrite() {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&gradient_lhs.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.left.was_overwritten();
        } else {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&gradient_lhs.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        if self.right.can_overwrite() {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&gradient_rhs.as_standard_layout())
                .par_for_each(|dest, src| *dest = -src);
            self.left.was_overwritten();
        } else {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&gradient_rhs.as_standard_layout())
                .par_for_each(|dest, src| *dest += -src);
        }

        true
    }
}

impl<Lhs, Rhs> Gradient for SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SubtractionBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SubtractionBackwardLeft<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    diff_operand: Rc<T>,
    gradient: RefCell<BroadTensor<T::Dim, U::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> SubtractionBackwardLeft<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    pub fn new(diff_operand: Rc<T>, operand: Rc<U>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(
            &*diff_operand.gradient(),
            &*operand.data(),
        ));

        Self {
            diff_operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Backward for SubtractionBackwardLeft<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut operand_grad = self.diff_operand.gradient_mut();
        let gradient = reduce(&operand_grad, &*self.gradient.borrow());

        if self.diff_operand.can_overwrite() {
            Zip::from(&mut *operand_grad)
                .and_broadcast(&gradient)
                .par_for_each(|dest, src| *dest = *src);
            self.diff_operand.was_overwritten();
        } else {
            Zip::from(&mut *operand_grad)
                .and_broadcast(&gradient)
                .par_for_each(|dest, src| *dest += *src);
        }

        true
    }
}

impl<T, U> Gradient for SubtractionBackwardLeft<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    type Dim = Broadcasted<T::Dim, U::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SubtractionBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SubtractionBackwardRight<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    diff_operand: Rc<T>,
    gradient: RefCell<BroadTensor<T::Dim, U::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> SubtractionBackwardRight<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    pub fn new(diff_operand: Rc<T>, operand: Rc<U>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(
            &*diff_operand.gradient(),
            &*operand.data(),
        ));

        Self {
            diff_operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Backward for SubtractionBackwardRight<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut operand_grad = self.diff_operand.gradient_mut();
        let gradient = reduce(&operand_grad, &*self.gradient.borrow());

        if self.diff_operand.can_overwrite() {
            Zip::from(&mut *operand_grad)
                .and_broadcast(&gradient)
                .par_for_each(|dest, src| *dest = -src);
            self.diff_operand.was_overwritten();
        } else {
            Zip::from(&mut *operand_grad)
                .and_broadcast(&gradient)
                .par_for_each(|dest, src| *dest += -src);
        }

        true
    }
}

impl<T, U> Gradient for SubtractionBackwardRight<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    type Dim = Broadcasted<T::Dim, U::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiplicationBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Broadcasted<LhsG::Dim, RhsG::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsD, LhsG, RhsD, RhsG> MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    pub fn new(
        left_data: Rc<LhsD>,
        left_grad: Rc<LhsG>,
        right_data: Rc<RhsD>,
        right_grad: Rc<RhsG>,
    ) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(
            &left_grad.gradient(),
            &right_grad.gradient(),
        ));

        Self {
            left_data,
            left_grad,
            right_data,
            right_grad,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut lhs_grad, mut rhs_grad) = {
            (
                self.left_grad.gradient_mut(),
                self.right_grad.gradient_mut(),
            )
        };
        let grad = self.gradient.borrow();

        let mut tmp = Tensor::zeros(grad.raw_dim());
        Zip::from(&mut tmp)
            .and(&*grad)
            .and_broadcast(&*self.right_data.data())
            .par_for_each(|tmp_el, grad_el, rhs_data_el| *tmp_el = grad_el * rhs_data_el);

        let to_left_grad = reduce(&*lhs_grad, &tmp);
        if self.left_grad.can_overwrite() {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&to_left_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.left_grad.was_overwritten();
        } else {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&to_left_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        Zip::from(&mut tmp)
            .and(&*grad)
            .and_broadcast(&*self.left_data.data())
            .par_for_each(|tmp_el, grad_el, lhs_data_el| *tmp_el = grad_el * lhs_data_el);

        let to_right_grad = reduce(&*rhs_grad, &tmp);
        if self.right_grad.can_overwrite() {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&to_right_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.right_grad.was_overwritten();
        } else {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&to_right_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        true
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Gradient for MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    type Dim = Broadcasted<LhsG::Dim, RhsG::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiplicationBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
    gradient: RefCell<BroadTensor<T::Dim, U::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(
            &*diff_operand.gradient(),
            &*no_diff_operand.data(),
        ));

        Self {
            diff_operand,
            no_diff_operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Backward for MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut operand_grad = self.diff_operand.gradient_mut();
        let grad = self.gradient.borrow();

        let mut tmp = Tensor::zeros(grad.raw_dim());
        Zip::from(&mut tmp)
            .and(&*grad)
            .and_broadcast(&*self.no_diff_operand.data())
            .par_for_each(|tmp_el, grad_el, no_diff_operand_el| {
                *tmp_el = grad_el * no_diff_operand_el
            });

        let gradient = reduce(&operand_grad, &tmp);
        if self.diff_operand.can_overwrite() {
            Zip::from(&mut *operand_grad)
                .and_broadcast(&gradient.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.diff_operand.was_overwritten();
        } else {
            Zip::from(&mut *operand_grad)
                .and_broadcast(&gradient.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        true
    }
}

impl<T, U> Gradient for MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Backward,
    U: Data + Forward,
    T::Dim: Dimension + DimMax<U::Dim>,
{
    type Dim = Broadcasted<T::Dim, U::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DivisionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Broadcasted<LhsG::Dim, RhsG::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsD, LhsG, RhsD, RhsG> DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    pub fn new(
        left_data: Rc<LhsD>,
        left_grad: Rc<LhsG>,
        right_data: Rc<RhsD>,
        right_grad: Rc<RhsG>,
    ) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(
            &left_grad.gradient(),
            &right_grad.gradient(),
        ));

        Self {
            left_data,
            left_grad,
            right_data,
            right_grad,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut lhs_grad, mut rhs_grad) = {
            (
                self.left_grad.gradient_mut(),
                self.right_grad.gradient_mut(),
            )
        };
        let grad = self.gradient.borrow();

        let mut tmp = Tensor::zeros(grad.raw_dim());
        Zip::from(&mut tmp)
            .and(&*grad)
            .and_broadcast(&*self.right_data.data())
            .par_for_each(|tmp_el, grad_el, rhs_data_el| *tmp_el = grad_el / rhs_data_el);

        let to_left_grad = reduce(&*lhs_grad, &tmp);
        if self.left_grad.can_overwrite() {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&to_left_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.left_grad.was_overwritten();
        } else {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&to_left_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        Zip::from(&mut tmp)
            .and(&*grad)
            .and_broadcast(&*self.left_data.data())
            .and_broadcast(&*self.right_data.data())
            .par_for_each(|tmp_el, grad_el, lhs_data_el, rhs_data_el| {
                *tmp_el = grad_el * lhs_data_el / rhs_data_el.powi(2)
            });

        let to_right_grad = reduce(&*rhs_grad, &tmp);
        if self.right_grad.can_overwrite() {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&to_right_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.right_grad.was_overwritten();
        } else {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&to_right_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        true
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Gradient for DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsD::Dim>,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    type Dim = Broadcasted<LhsG::Dim, RhsG::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DivisionBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    gradient: RefCell<Tensor<Broadcasted<LhsG::Dim, RhsD::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsG, RhsD> DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    pub fn new(left_grad: Rc<LhsG>, right_data: Rc<RhsD>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(&left_grad.gradient(), &right_data.data()));

        Self {
            left_grad,
            right_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsG, RhsD> Backward for DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut lhs_grad = { self.left_grad.gradient_mut() };
        let grad = self.gradient.borrow();

        let mut tmp = Tensor::zeros(grad.raw_dim());
        Zip::from(&mut tmp)
            .and(&*grad)
            .and_broadcast(&*self.right_data.data())
            .par_for_each(|tmp_el, grad_el, rhs_data_el| *tmp_el = grad_el / rhs_data_el);

        let to_left_grad = reduce(&*lhs_grad, &tmp);
        if self.left_grad.can_overwrite() {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&to_left_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.left_grad.was_overwritten();
        } else {
            Zip::from(&mut *lhs_grad)
                .and_broadcast(&to_left_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        true
    }
}

impl<LhsG, RhsD> Gradient for DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data + Forward,
    LhsG: Gradient + Backward,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    type Dim = Broadcasted<LhsG::Dim, RhsD::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DivisionBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    left_data: Rc<LhsD>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Broadcasted<LhsD::Dim, RhsG::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsD, RhsD, RhsG> DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    pub fn new(left_data: Rc<LhsD>, right_data: Rc<RhsD>, right_grad: Rc<RhsG>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(&left_data.data(), &right_grad.gradient()));

        Self {
            left_data,
            right_data,
            right_grad,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsD, RhsD, RhsG> Backward for DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut rhs_grad = self.right_grad.gradient_mut();
        let grad = self.gradient.borrow();

        let mut tmp = Tensor::zeros(grad.raw_dim());
        Zip::from(&mut tmp)
            .and(&*grad)
            .and_broadcast(&*self.left_data.data())
            .and_broadcast(&*self.right_data.data())
            .par_for_each(|tmp_el, grad_el, lhs_data_el, rhs_data_el| {
                *tmp_el = grad_el * lhs_data_el / rhs_data_el.powi(2)
            });

        let to_right_grad = reduce(&*rhs_grad, &tmp);
        if self.right_grad.can_overwrite() {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&to_right_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest = *src);
            self.right_grad.was_overwritten();
        } else {
            Zip::from(&mut *rhs_grad)
                .and_broadcast(&to_right_grad.as_standard_layout())
                .par_for_each(|dest, src| *dest += *src);
        }

        true
    }
}

impl<LhsD, RhsD, RhsG> Gradient for DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data + Forward,
    RhsD: Data + Forward,
    RhsG: Gradient + Backward,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    type Dim = Broadcasted<LhsD::Dim, RhsG::Dim>;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MattrixMatrixMulBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsD: Data<Dim = Ix2> + Forward,
    LhsG: Gradient<Dim = Ix2> + Gradient,
    RhsG: Gradient<Dim = Ix2> + Gradient,
{
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Ix2>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsD, LhsG, RhsD, RhsG> MatrixMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsD: Data<Dim = Ix2> + Forward,
    LhsG: Gradient<Dim = Ix2> + Gradient,
    RhsG: Gradient<Dim = Ix2> + Gradient,
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
        let gradient = RefCell::new(Tensor::zeros((shape[0], shape[1])));

        Self {
            left_data,
            left_grad,
            right_data,
            right_grad,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for MatrixMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsD: Data<Dim = Ix2> + Forward,
    LhsG: Gradient<Dim = Ix2> + Gradient,
    RhsG: Gradient<Dim = Ix2> + Gradient,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut lhs_grad, lhs_data, mut rhs_grad, rhs_data) = {
            (
                self.left_grad.gradient_mut(),
                self.left_data.data(),
                self.right_grad.gradient_mut(),
                self.right_data.data(),
            )
        };
        let grad = self.gradient.borrow();

        if self.left_grad.can_overwrite() {
            general_mat_mul(1.0, &grad, &rhs_data.t(), 0., &mut lhs_grad);
            self.left_grad.was_overwritten();
        } else {
            general_mat_mul(1.0, &grad, &rhs_data.t(), 1., &mut lhs_grad);
        }

        if self.right_grad.can_overwrite() {
            general_mat_mul(1.0, &lhs_data.t(), &grad, 0., &mut rhs_grad);
            self.right_grad.was_overwritten();
        } else {
            general_mat_mul(1.0, &lhs_data.t(), &grad, 1., &mut rhs_grad);
        }

        true
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Gradient for MatrixMatrixMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsD: Data<Dim = Ix2> + Forward,
    LhsG: Gradient<Dim = Ix2> + Gradient,
    RhsG: Gradient<Dim = Ix2> + Gradient,
{
    type Dim = Ix2;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixMatrixMulBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
{
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    gradient: RefCell<Tensor<Ix2>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsG, RhsD> MatrixMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
{
    pub fn new(left_grad: Rc<LhsG>, right_data: Rc<RhsD>) -> Self {
        let shape = DotDim::shape(left_grad.gradient().raw_dim(), right_data.data().raw_dim());
        let gradient = RefCell::new(Tensor::zeros((shape[0], shape[1])));

        Self {
            left_grad,
            right_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsG, RhsD> Backward for MatrixMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut lhs_grad, rhs_data) = { (self.left_grad.gradient_mut(), self.right_data.data()) };
        let grad = self.gradient.borrow();

        if self.left_grad.can_overwrite() {
            general_mat_mul(1.0, &grad, &rhs_data.t(), 0., &mut lhs_grad);
            self.left_grad.was_overwritten();
        } else {
            general_mat_mul(1.0, &grad, &rhs_data.t(), 1., &mut lhs_grad);
        }

        true
    }
}

impl<LhsG, RhsD> Gradient for MatrixMatrixMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix2> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
{
    type Dim = Ix2;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixMatrixMulBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsG: Gradient<Dim = Ix2> + Backward,
{
    left_data: Rc<LhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Ix2>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsD, RhsG> MatrixMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsG: Gradient<Dim = Ix2> + Backward,
{
    pub fn new(left_data: Rc<LhsD>, right_grad: Rc<RhsG>) -> Self {
        let shape = DotDim::shape(left_data.data().raw_dim(), right_grad.gradient().raw_dim());
        let gradient = RefCell::new(Tensor::zeros((shape[0], shape[1])));

        Self {
            left_data,
            right_grad,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsD, RhsG> Backward for MatrixMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsG: Gradient<Dim = Ix2> + Backward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (lhs_data, mut rhs_grad) = { (self.left_data.data(), self.right_grad.gradient_mut()) };
        let grad = self.gradient.borrow();

        if self.right_grad.can_overwrite() {
            general_mat_mul(1.0, &lhs_data.t(), &grad, 0., &mut rhs_grad);
            self.right_grad.was_overwritten();
        } else {
            general_mat_mul(1.0, &lhs_data.t(), &grad, 1., &mut rhs_grad);
        }

        true
    }
}

impl<LhsD, RhsG> Gradient for MatrixMatrixMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsG: Gradient<Dim = Ix2> + Backward,
{
    type Dim = Ix2;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMulBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MatrixVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Ix1>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsD, LhsG, RhsD, RhsG> MatrixVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
    RhsG: Gradient<Dim = Ix1> + Backward,
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
        let gradient = RefCell::new(Tensor::zeros(shape[0]));

        Self {
            left_data,
            left_grad,
            right_data,
            right_grad,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for MatrixVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut lhs_grad, lhs_data, mut rhs_grad, rhs_data) = {
            (
                self.left_grad.gradient_mut(),
                self.left_data.data(),
                self.right_grad.gradient_mut(),
                self.right_data.data(),
            )
        };
        let grad = self.gradient.borrow();

        if self.left_grad.can_overwrite() {
            Zip::from(lhs_grad.rows_mut())
                .and(&*grad)
                .for_each(|row, grad_el| {
                    Zip::from(row)
                        .and(&*rhs_data)
                        .for_each(|row_el, rhs_data_el| *row_el = *rhs_data_el * *grad_el);
                });
            self.left_grad.was_overwritten();
        } else {
            Zip::from(lhs_grad.rows_mut())
                .and(&*grad)
                .for_each(|row, grad_el| {
                    Zip::from(row)
                        .and(&*rhs_data)
                        .for_each(|row_el, rhs_data_el| *row_el += *rhs_data_el * *grad_el);
                });
        }

        if self.right_grad.can_overwrite() {
            general_mat_vec_mul(1.0, &lhs_data.t(), &grad, 0., &mut rhs_grad);
            self.right_grad.was_overwritten();
        } else {
            general_mat_vec_mul(1.0, &lhs_data.t(), &grad, 1., &mut rhs_grad);
        }

        true
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Gradient for MatrixVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMulBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixVectorMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
{
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    gradient: RefCell<Tensor<Ix1>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsG, RhsD> MatrixVectorMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
{
    pub fn new(left_grad: Rc<LhsG>, right_data: Rc<RhsD>) -> Self {
        let shape = DotDim::shape(left_grad.gradient().raw_dim(), right_data.data().raw_dim());
        let gradient = RefCell::new(Tensor::zeros(shape[0]));

        Self {
            left_grad,
            right_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsG, RhsD> Backward for MatrixVectorMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut lhs_grad, rhs_data) = { (self.left_grad.gradient_mut(), self.right_data.data()) };
        let grad = self.gradient.borrow();

        if self.left_grad.can_overwrite() {
            Zip::from(lhs_grad.rows_mut())
                .and(&*grad)
                .for_each(|row, grad_el| {
                    Zip::from(row)
                        .and(&*rhs_data)
                        .for_each(|row_el, rhs_data_el| *row_el = *rhs_data_el * *grad_el);
                });
            self.left_grad.was_overwritten();
        } else {
            Zip::from(lhs_grad.rows_mut())
                .and(&*grad)
                .for_each(|row, grad_el| {
                    Zip::from(row)
                        .and(&*rhs_data)
                        .for_each(|row_el, rhs_data_el| *row_el += *rhs_data_el * *grad_el);
                });
        }

        true
    }
}

impl<LhsG, RhsD> Gradient for MatrixVectorMulBackwardLeft<LhsG, RhsD>
where
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix2> + Backward,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMulBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixVectorMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    left_data: Rc<LhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Ix1>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsD, RhsG> MatrixVectorMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    pub fn new(left_data: Rc<LhsD>, right_grad: Rc<RhsG>) -> Self {
        let shape = DotDim::shape(left_data.data().raw_dim(), right_grad.gradient().raw_dim());
        let gradient = RefCell::new(Tensor::zeros(shape[0]));

        Self {
            left_data,
            right_grad,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsD, RhsG> Backward for MatrixVectorMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (lhs_data, mut rhs_grad) = { (self.left_data.data(), self.right_grad.gradient_mut()) };

        let grad = self.gradient.borrow();
        if self.right_grad.can_overwrite() {
            general_mat_vec_mul(1.0, &lhs_data.t(), &grad, 0., &mut rhs_grad);
            self.right_grad.was_overwritten();
        } else {
            general_mat_vec_mul(1.0, &lhs_data.t(), &grad, 1., &mut rhs_grad);
        }

        true
    }
}

impl<LhsD, RhsG> Gradient for MatrixVectorMulBackwardRight<LhsD, RhsG>
where
    LhsD: Data<Dim = Ix2> + Forward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMulBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1> + Forward,
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix1> + Backward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Ix1>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<LhsD, LhsG, RhsD, RhsG> VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1> + Forward,
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix1> + Backward,
    RhsG: Gradient<Dim = Ix1> + Backward,
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
        let gradient = RefCell::new(Tensor::zeros(shape[0]));

        Self {
            left_data,
            left_grad,
            right_data,
            right_grad,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1> + Forward,
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix1> + Backward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut lhs_grad, lhs_data, mut rhs_grad, rhs_data) = {
            (
                self.left_grad.gradient_mut(),
                self.left_data.data(),
                self.right_grad.gradient_mut(),
                self.right_data.data(),
            )
        };
        let grad = self.gradient.borrow();

        let left_zip = Zip::from(&mut *lhs_grad).and(&*rhs_data);
        if self.left_grad.can_overwrite() {
            left_zip.for_each(|lhs_grad_el, rhs_data_el| *lhs_grad_el = rhs_data_el * grad[0]);
            self.left_grad.was_overwritten();
        } else {
            left_zip.for_each(|lhs_grad_el, rhs_data_el| *lhs_grad_el += rhs_data_el * grad[0]);
        }

        let right_zip = Zip::from(&mut *rhs_grad).and(&*lhs_data);
        if self.right_grad.can_overwrite() {
            right_zip.for_each(|rhs_grad_el, lhs_data_el| *rhs_grad_el = lhs_data_el * grad[0]);
            self.right_grad.was_overwritten();
        } else {
            right_zip.for_each(|rhs_grad_el, lhs_data_el| *rhs_grad_el += lhs_data_el * grad[0]);
        }

        true
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Gradient for VectorVectorMulBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data<Dim = Ix1> + Forward,
    RhsD: Data<Dim = Ix1> + Forward,
    LhsG: Gradient<Dim = Ix1> + Backward,
    RhsG: Gradient<Dim = Ix1> + Backward,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMulBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Backward,
    U: Data<Dim = Ix1> + Forward,
{
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
    gradient: RefCell<Tensor<Ix1>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Backward,
    U: Data<Dim = Ix1> + Forward,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = DotDim::shape(
            diff_operand.gradient().raw_dim(),
            no_diff_operand.data().raw_dim(),
        );
        let gradient = RefCell::new(Tensor::zeros(shape[0]));

        Self {
            diff_operand,
            no_diff_operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Backward for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Backward,
    U: Data<Dim = Ix1> + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut diff_op_grad, no_diff_op_data) = {
            (
                self.diff_operand.gradient_mut(),
                self.no_diff_operand.data(),
            )
        };
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *diff_op_grad).and(&*no_diff_op_data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|diff_op_grad_el, no_diff_op_data_el| {
                *diff_op_grad_el = no_diff_op_data_el * grad[0]
            });
            self.diff_operand.was_overwritten();
        } else {
            zip.for_each(|diff_op_grad_el, no_diff_op_data_el| {
                *diff_op_grad_el += no_diff_op_data_el * grad[0]
            });
        }

        true
    }
}

impl<T, U> Gradient for VectorVectorMulBackwardUnary<T, U>
where
    T: Gradient<Dim = Ix1> + Backward,
    U: Data<Dim = Ix1> + Forward,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PowerBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct PowerBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    exp: i32,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> PowerBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>, exp: i32) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            exp,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for PowerBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T, U> Backward for PowerBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand_grad.gradient_mut();
        let op_data = self.operand_data.data();
        let grad = self.gradient.borrow();
        let exp = self.exp;

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.operand_grad.can_overwrite() {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = grad_el * op_data_el.powi(exp - 1) * exp as f32
            });
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += grad_el * op_data_el.powi(exp - 1) * exp as f32
            });
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SumBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SumBackward<T: Gradient + Backward> {
    operand: Rc<T>,
    gradient: RefCell<Tensor<Ix1>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T: Gradient + Backward> SumBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(1));

        Self {
            operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T: Gradient + Backward> Gradient for SumBackward<T> {
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T: Gradient + Backward> Backward for SumBackward<T> {
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand.gradient_mut();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *op_grad).and_broadcast(&*grad);
        if self.operand.can_overwrite() {
            zip.par_for_each(|op_grad_el, grad_el| *op_grad_el = *grad_el);
            self.operand.was_overwritten();
        } else {
            zip.par_for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el);
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LognBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct LognBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> LognBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for LognBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T, U> Backward for LognBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand_grad.gradient_mut();
        let op_data = self.operand_data.data();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.operand_grad.can_overwrite() {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| *op_grad_el = grad_el / op_data_el);
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| *op_grad_el += grad_el / op_data_el);
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLUBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ReLUBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> ReLUBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for ReLUBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T, U> Backward for ReLUBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand_grad.gradient_mut();
        let op_data = self.operand_data.data();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.operand_grad.can_overwrite() {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = if *op_data_el > 0.0 { *grad_el } else { 0.0 }
            });
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += if *op_data_el > 0.0 { *grad_el } else { 0.0 }
            });
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLUBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct LeakyReLUBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> LeakyReLUBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for LeakyReLUBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T, U> Backward for LeakyReLUBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand_grad.gradient_mut();
        let op_data = self.operand_data.data();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.operand_grad.can_overwrite() {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = if *op_data_el > 0.0 { *grad_el } else { 0.01 }
            });
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += if *op_data_el > 0.0 { *grad_el } else { 0.01 }
            });
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftPlusBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~S

pub struct SoftPlusBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> SoftPlusBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for SoftPlusBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T, U> Backward for SoftPlusBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand_grad.gradient_mut();
        let op_data = self.operand_data.data();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.operand_grad.can_overwrite() {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = if *op_data_el >= 15.0 {
                    *grad_el
                } else if *op_data_el <= -15.0 {
                    0.0
                } else {
                    grad_el / (1.0 + (-*op_data_el).exp())
                }
            });
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += if *op_data_el >= 15.0 {
                    *grad_el
                } else if *op_data_el <= -15.0 {
                    0.0
                } else {
                    grad_el / (1.0 + (-*op_data_el).exp())
                }
            });
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SigmoidBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SigmoidBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> SigmoidBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for SigmoidBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T, U> Backward for SigmoidBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand_grad.gradient_mut();
        let op_data = self.operand_data.data();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.operand_grad.can_overwrite() {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = *grad_el * *op_data_el * (1.0 - *op_data_el)
            });
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += *grad_el * *op_data_el * (1.0 - *op_data_el)
            });
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TanHBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct TanHBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> TanHBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for TanHBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T, U> Backward for TanHBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand_grad.gradient_mut();
        let op_data = self.operand_data.data();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.operand_grad.can_overwrite() {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = *grad_el * (1.0 - op_data_el.powi(2))
            });
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += *grad_el * (1.0 - op_data_el.powi(2))
            });
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ExpBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ExpBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> ExpBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for ExpBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T, U> Backward for ExpBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand_grad.gradient_mut();
        let op_data = self.operand_data.data();
        let grad = self.gradient.borrow();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.operand_grad.can_overwrite() {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| *op_grad_el = *grad_el * op_data_el);
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += *grad_el * op_data_el
            });
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftmaxBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SoftmaxBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    axis: usize,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U> SoftmaxBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>, axis: usize) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            axis,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for SoftmaxBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T, U> Backward for SoftmaxBackward<T, U>
where
    T: Gradient<Dim = U::Dim> + Backward,
    U: Data + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut op_grad = self.operand_grad.gradient_mut();
        let op_data = self.operand_data.data();
        let grad = self.gradient.borrow();
        let axis = self.axis;
        let shape = op_data.raw_dim();
        let jacobian_shape = (shape[axis], shape[axis]);

        let zip = Zip::from(op_grad.lanes_mut(Axis(axis)))
            .and(grad.lanes(Axis(axis)))
            .and(op_data.lanes(Axis(axis)));
        if self.operand_grad.can_overwrite() {
            zip.par_for_each(|mut op_grad_lane, grad_lane, op_data_lane| {
                let mut jacobian = Tensor::<Ix2>::zeros(jacobian_shape);
                fill_softmax_jacobian(&mut jacobian, &op_data_lane);
                general_mat_vec_mul(1.0, &jacobian, &grad_lane, 0., &mut op_grad_lane);
            });
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|mut op_grad_lane, grad_lane, op_data_lane| {
                let mut jacobian = Tensor::<Ix2>::zeros(jacobian_shape);
                fill_softmax_jacobian(&mut jacobian, &op_data_lane);
                general_mat_vec_mul(1.0, &jacobian, &grad_lane, 1., &mut op_grad_lane);
            });
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConcatenateBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient<Dim = Rhs::Dim> + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    gradient: RefCell<Tensor<Lhs::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient<Dim = Rhs::Dim> + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let gradient = RefCell::new(
            concatenate(
                Axis(axis),
                &[left.gradient().view(), right.gradient().view()],
            )
            .unwrap(),
        );

        Self {
            left,
            right,
            gradient,
            axis,
            can_overwrite: Cell::new(true),
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

impl<Lhs, Rhs> Gradient for ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient<Dim = Rhs::Dim> + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: RemoveAxis,
{
    type Dim = Lhs::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<Lhs, Rhs> Backward for ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient<Dim = Rhs::Dim> + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: RemoveAxis,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let grad = self.gradient.borrow();
        let mut lhs_grad = self.left.gradient_mut();
        let mut rhs_grad = self.right.gradient_mut();
        let axis = self.axis;
        let (lhs_portion, rhs_portion) = grad
            .view()
            .split_at(Axis(axis), lhs_grad.len_of(Axis(axis)));

        let zip_lhs = Zip::from(&mut *lhs_grad).and(&lhs_portion);
        if self.left.can_overwrite() {
            zip_lhs.par_for_each(|lhs_grad_el, lhs_portion_el| *lhs_grad_el = *lhs_portion_el);
            self.left.was_overwritten();
        } else {
            zip_lhs.par_for_each(|lhs_grad_el, lhs_portion_el| *lhs_grad_el += *lhs_portion_el);
        }

        let zip_rhs = Zip::from(&mut *rhs_grad).and(&rhs_portion);
        if self.right.can_overwrite() {
            zip_rhs.par_for_each(|rhs_grad_el, rhs_portion_el| *rhs_grad_el = *rhs_portion_el);
            self.right.was_overwritten();
        } else {
            zip_rhs.par_for_each(|rhs_grad_el, rhs_portion_el| *rhs_grad_el += *rhs_portion_el);
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConcatenateBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct ConcatenateBackwardLeft<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    left: Rc<T>,
    axis: usize,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T> ConcatenateBackwardLeft<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    pub fn new<U>(left: Rc<T>, right: Rc<U>, axis: usize) -> Self
    where
        U: Data<Dim = T::Dim>,
    {
        let gradient = RefCell::new(
            concatenate(Axis(axis), &[left.gradient().view(), right.data().view()]).unwrap(),
        );

        Self {
            left,
            axis,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.left.clone()
    }
}

impl<T> Gradient for ConcatenateBackwardLeft<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T> Backward for ConcatenateBackwardLeft<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let grad = self.gradient.borrow();
        let mut lhs_grad = self.left.gradient_mut();
        let axis = self.axis;
        let (lhs_portion, _) = grad
            .view()
            .split_at(Axis(axis), lhs_grad.len_of(Axis(axis)));

        let zip_lhs = Zip::from(&mut *lhs_grad).and(&lhs_portion);
        if self.left.can_overwrite() {
            zip_lhs.par_for_each(|lhs_grad_el, lhs_portion_el| *lhs_grad_el = *lhs_portion_el);
            self.left.was_overwritten();
        } else {
            zip_lhs.par_for_each(|lhs_grad_el, lhs_portion_el| *lhs_grad_el += *lhs_portion_el);
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConcatenateBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ConcatenateBackwardRight<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    offset: usize,
    right: Rc<T>,
    axis: usize,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T> ConcatenateBackwardRight<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    pub fn new<U>(left: Rc<U>, right: Rc<T>, axis: usize) -> Self
    where
        U: Data<Dim = T::Dim>,
    {
        let gradient = RefCell::new(
            concatenate(Axis(axis), &[left.data().view(), right.gradient().view()]).unwrap(),
        );

        Self {
            right,
            gradient,
            offset: left.data().len_of(Axis(axis)),
            axis,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
    pub fn operand(&self) -> Rc<T> {
        self.right.clone()
    }
}

impl<T> Gradient for ConcatenateBackwardRight<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T> Backward for ConcatenateBackwardRight<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let grad = self.gradient.borrow();
        let mut rhs_grad = self.right.gradient_mut();
        let axis = self.axis;
        let (_, rhs_portion) = grad.view().split_at(Axis(axis), self.offset);

        let zip_rhs = Zip::from(&mut *rhs_grad).and(&rhs_portion);
        if self.right.can_overwrite() {
            zip_rhs.par_for_each(|rhs_grad_el, rhs_portion_el| *rhs_grad_el = *rhs_portion_el);
            self.right.was_overwritten();
        } else {
            zip_rhs.par_for_each(|rhs_grad_el, rhs_portion_el| *rhs_grad_el += *rhs_portion_el);
        }

        true
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StackBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct StackBackward<Lhs, Rhs>
where
    Lhs: Gradient<Dim = Rhs::Dim> + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    gradient: RefCell<Tensor<<Lhs::Dim as Dimension>::Larger>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<Lhs, Rhs> StackBackward<Lhs, Rhs>
where
    Lhs: Gradient<Dim = Rhs::Dim> + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let gradient = RefCell::new(
            stack(
                Axis(axis),
                &[left.gradient().view(), right.gradient().view()],
            )
            .unwrap(),
        );

        Self {
            left,
            right,
            gradient,
            axis,
            can_overwrite: Cell::new(true),
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

impl<Lhs, Rhs> Gradient for StackBackward<Lhs, Rhs>
where
    Lhs: Gradient<Dim = Rhs::Dim> + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: RemoveAxis,
{
    type Dim = <Lhs::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<Lhs, Rhs> Backward for StackBackward<Lhs, Rhs>
where
    Lhs: Gradient<Dim = Rhs::Dim> + Backward,
    Rhs: Gradient + Backward,
    Lhs::Dim: RemoveAxis,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let grad = self.gradient.borrow();
        let mut lhs_grad = self.left.gradient_mut();
        let mut rhs_grad = self.right.gradient_mut();
        let axis = self.axis;
        let mut subview_iter = grad.axis_iter(Axis(axis));
        let (lhs_portion, rhs_portion) = {
            (
                subview_iter
                    .next()
                    .unwrap()
                    .into_dimensionality::<Lhs::Dim>()
                    .unwrap(),
                subview_iter
                    .next()
                    .unwrap()
                    .into_dimensionality::<Rhs::Dim>()
                    .unwrap(),
            )
        };

        let zip_lhs = Zip::from(&mut *lhs_grad).and(&lhs_portion);
        if self.left.can_overwrite() {
            zip_lhs.par_for_each(|lhs_grad_el, lhs_portion_el| *lhs_grad_el = *lhs_portion_el);
            self.left.was_overwritten();
        } else {
            zip_lhs.par_for_each(|lhs_grad_el, lhs_portion_el| *lhs_grad_el += *lhs_portion_el);
        }

        let zip_rhs = Zip::from(&mut *rhs_grad).and(&rhs_portion);
        if self.right.can_overwrite() {
            zip_rhs.par_for_each(|rhs_grad_el, rhs_portion_el| *rhs_grad_el = *rhs_portion_el);
            self.right.was_overwritten();
        } else {
            zip_rhs.par_for_each(|rhs_grad_el, rhs_portion_el| *rhs_grad_el += *rhs_portion_el);
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StackBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct StackBackwardLeft<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    left: Rc<T>,
    axis: usize,
    gradient: RefCell<Tensor<<T::Dim as Dimension>::Larger>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T> StackBackwardLeft<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    pub fn new<U>(left: Rc<T>, right: Rc<U>, axis: usize) -> Self
    where
        U: Data<Dim = T::Dim>,
    {
        let gradient = RefCell::new(
            stack(Axis(axis), &[left.gradient().view(), right.data().view()]).unwrap(),
        );

        Self {
            left,
            gradient,
            axis,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.left.clone()
    }
}

impl<T> Gradient for StackBackwardLeft<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    type Dim = <T::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T> Backward for StackBackwardLeft<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let grad = self.gradient.borrow();
        let mut lhs_grad = self.left.gradient_mut();
        let axis = self.axis;
        let mut subview_iter = grad.axis_iter(Axis(axis));
        let lhs_portion = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<T::Dim>()
            .unwrap();

        let zip_lhs = Zip::from(&mut *lhs_grad).and(&lhs_portion);
        if self.left.can_overwrite() {
            zip_lhs.par_for_each(|lhs_grad_el, lhs_portion_el| *lhs_grad_el = *lhs_portion_el);
            self.left.was_overwritten();
        } else {
            zip_lhs.par_for_each(|lhs_grad_el, lhs_portion_el| *lhs_grad_el += *lhs_portion_el);
        }

        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StackBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct StackBackwardRight<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    right: Rc<T>,
    axis: usize,
    gradient: RefCell<Tensor<<T::Dim as Dimension>::Larger>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T> StackBackwardRight<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    pub fn new<U>(left: Rc<U>, right: Rc<T>, axis: usize) -> Self
    where
        U: Data<Dim = T::Dim>,
    {
        let gradient = RefCell::new(
            stack(Axis(axis), &[left.data().view(), right.gradient().view()]).unwrap(),
        );

        Self {
            right,
            gradient,
            axis,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }

    pub fn operand(&self) -> Rc<T> {
        self.right.clone()
    }
}

impl<T> Gradient for StackBackwardRight<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    type Dim = <T::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T> Backward for StackBackwardRight<T>
where
    T: Gradient + Backward,
    T::Dim: RemoveAxis,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let grad = self.gradient.borrow();
        let mut rhs_grad = self.right.gradient_mut();
        let axis = self.axis;
        let mut subview_iter = grad.axis_iter(Axis(axis));
        let rhs_portion = subview_iter
            .skip(1)
            .next()
            .unwrap()
            .into_dimensionality::<T::Dim>()
            .unwrap();

        let zip_rhs = Zip::from(&mut *rhs_grad).and(&rhs_portion);
        if self.right.can_overwrite() {
            zip_rhs.par_for_each(|rhs_grad_el, rhs_portion_el| *rhs_grad_el = *rhs_portion_el);
            self.right.was_overwritten();
        } else {
            zip_rhs.par_for_each(|rhs_grad_el, rhs_portion_el| *rhs_grad_el += *rhs_portion_el);
        }
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ UnsqueezeBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct UnsqueezeBackward<T: Gradient + Backward> {
    operand: Rc<T>,
    axis: usize,
    gradient: RefCell<Tensor<<T::Dim as Dimension>::Larger>>,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T: Gradient + Backward> UnsqueezeBackward<T> {
    pub fn new(operand: Rc<T>, axis: usize) -> Self {
        let shape = operand.gradient().raw_dim();
        let gradient = RefCell::new(Tensor::zeros(shape.insert_axis(Axis(axis))));

        Self {
            operand,
            axis,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T: Gradient + Backward> Gradient for UnsqueezeBackward<T> {
    type Dim = <T::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }

    fn was_overwritten(&self) {
        debug_assert_eq!(self.can_overwrite.get(), true);
        self.can_overwrite.set(false);
    }
}

impl<T: Gradient + Backward> Backward for UnsqueezeBackward<T> {
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let mut operand_grad = self.operand.gradient_mut();
        let axis = self.axis;
        let grad = self.gradient.borrow();
        let unsqueezed_gradient = grad
            .axis_iter(Axis(axis))
            .next()
            .unwrap()
            .into_dimensionality::<T::Dim>()
            .unwrap();

        let zip = Zip::from(&mut *operand_grad).and(&unsqueezed_gradient);
        if self.operand.can_overwrite() {
            zip.par_for_each(|dest, src| *dest = *src);
            self.operand.was_overwritten();
        } else {
            zip.par_for_each(|dest, src| *dest += src);
        }

        true
    }
}
