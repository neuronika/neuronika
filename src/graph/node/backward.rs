use super::super::{BroadTensor, Broadcasted, Tensor};
use super::broadcasted_zeros;
use super::forward::{Data, Forward, Input};
use ndarray::{linalg::general_mat_vec_mul, ArrayView1, Axis, DimMax, Dimension, Ix1, Ix2, Zip};
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

fn fill_jacobian(jacobian: &mut Tensor<Ix2>, array: &ArrayView1<f32>) {
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait Gradient {
    type Dim: Dimension;
    fn gradient(&self) -> Ref<Tensor<Self::Dim>>;
    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>>;
    fn can_overwrite(&self) -> bool;
    fn was_overwritten(&self);
}

pub trait Backward {
    fn backward(&mut self) -> bool;
}

pub trait Differentiable {
    type Output: Backward + Gradient;

    fn differentiable(&self) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ InputBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct InputBackward<D: Dimension> {
    gradient: RefCell<Tensor<D>>,
    can_overwrite: Cell<bool>,
}

impl<D> InputBackward<D>
where
    D: Dimension,
{
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
    fn backward(&mut self) -> bool {
        false
    }
}

impl<D> Differentiable for Input<D>
where
    D: Dimension,
{
    type Output = InputBackward<D>;

    fn differentiable(&self) -> Self::Output {
        Self::Output {
            gradient: RefCell::new(Tensor::zeros(self.data().raw_dim())),
            can_overwrite: Cell::new(true),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NegationBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct NegationBackward<T>
where
    T: Gradient,
{
    operand: Rc<T>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T> NegationBackward<T>
where
    T: Gradient,
{
    pub fn new(operand: Rc<T>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand.gradient().raw_dim()));
        Self {
            operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T> Gradient for NegationBackward<T>
where
    T: Gradient,
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

impl<T> Backward for NegationBackward<T>
where
    T: Gradient,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TransposeBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct TransposeBackward<T>
where
    T: Gradient,
{
    operand: Rc<T>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T> TransposeBackward<T>
where
    T: Gradient,
{
    pub fn new(operand: Rc<T>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand.gradient().raw_dim()));
        Self {
            operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T> Gradient for TransposeBackward<T>
where
    T: Gradient,
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

impl<T> Backward for TransposeBackward<T>
where
    T: Gradient,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdditionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient,
    Rhs: Gradient,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    gradient: RefCell<Tensor<Broadcasted<Lhs::Dim, Rhs::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<Lhs, Rhs> AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient,
    Rhs: Gradient,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(&left.gradient(), &right.gradient()));

        Self {
            left,
            right,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<Lhs, Rhs> Backward for AdditionBackward<Lhs, Rhs>
where
    Lhs: Gradient,
    Rhs: Gradient,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    Lhs: Gradient,
    Rhs: Gradient,
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
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AdditionBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct AdditionBackwardUnary<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
{
    diff_operand: Rc<T>,
    gradient: RefCell<BroadTensor<T::Dim, U::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U> AdditionBackwardUnary<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
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
            was_computed: false,
        }
    }
}

impl<T, U> Backward for AdditionBackwardUnary<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    T: Gradient + Backward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
    U: Data + Forward + 'static,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SubtractionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient,
    Rhs: Gradient,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    gradient: RefCell<Tensor<Broadcasted<Lhs::Dim, Rhs::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<Lhs, Rhs> SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient,
    Rhs: Gradient,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(&left.gradient(), &right.gradient()));

        Self {
            left,
            right,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<Lhs, Rhs> Backward for SubtractionBackward<Lhs, Rhs>
where
    Lhs: Gradient,
    Rhs: Gradient,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    Lhs: Gradient,
    Rhs: Gradient,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SubtractionBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SubtractionBackwardLeft<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
{
    diff_operand: Rc<T>,
    gradient: RefCell<BroadTensor<T::Dim, U::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U> SubtractionBackwardLeft<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
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
            was_computed: false,
        }
    }
}

impl<T, U> Backward for SubtractionBackwardLeft<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SubtractionBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SubtractionBackwardRight<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
{
    diff_operand: Rc<T>,
    gradient: RefCell<BroadTensor<T::Dim, U::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U> SubtractionBackwardRight<T, U>
where
    T: Gradient + Backward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
    U: Data + Forward + 'static,
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
            was_computed: false,
        }
    }
}

impl<T, U> Backward for SubtractionBackwardRight<T, U>
where
    T: Gradient + Backward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
    U: Data + Forward + 'static,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiplicationBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient,
    RhsG: Gradient,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Broadcasted<LhsG::Dim, RhsG::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<LhsD, LhsG, RhsD, RhsG> MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient,
    RhsG: Gradient,
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
            was_computed: false,
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for MultiplicationBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient,
    RhsG: Gradient,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient,
    RhsG: Gradient,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiplicationBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
{
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
    gradient: RefCell<BroadTensor<T::Dim, U::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U> MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
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
            was_computed: false,
        }
    }
}

impl<T, U> Backward for MultiplicationBackwardUnary<T, U>
where
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    T: Gradient + Backward + 'static,
    U: Data + Forward + 'static,
    <T as Gradient>::Dim: DimMax<<U as Data>::Dim>,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DivisionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient,
    RhsG: Gradient,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    left_data: Rc<LhsD>,
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Broadcasted<LhsG::Dim, RhsG::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<LhsD, LhsG, RhsD, RhsG> DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient,
    RhsG: Gradient,
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
            was_computed: false,
        }
    }
}

impl<LhsD, LhsG, RhsD, RhsG> Backward for DivisionBackward<LhsD, LhsG, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient,
    RhsG: Gradient,
    LhsG::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    LhsD: Data,
    RhsD: Data,
    LhsG: Gradient,
    RhsG: Gradient,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DivisionBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data,
    LhsG: Gradient,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    left_grad: Rc<LhsG>,
    right_data: Rc<RhsD>,
    gradient: RefCell<Tensor<Broadcasted<LhsG::Dim, RhsD::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<LhsG, RhsD> DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data,
    LhsG: Gradient,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    pub fn new(left_grad: Rc<LhsG>, right_data: Rc<RhsD>) -> Self {
        let gradient = RefCell::new(broadcasted_zeros(&left_grad.gradient(), &right_data.data()));

        Self {
            left_grad,
            right_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<LhsG, RhsD> Backward for DivisionBackwardLeft<LhsG, RhsD>
where
    RhsD: Data,
    LhsG: Gradient,
    LhsG::Dim: Dimension + DimMax<RhsD::Dim>,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    LhsG: Gradient,
    RhsD: Data,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DivisionBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    RhsG: Gradient,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    left_data: Rc<LhsD>,
    right_data: Rc<RhsD>,
    right_grad: Rc<RhsG>,
    gradient: RefCell<Tensor<Broadcasted<LhsD::Dim, RhsG::Dim>>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<LhsD, RhsD, RhsG> DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    RhsG: Gradient,
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
            was_computed: false,
        }
    }
}

impl<LhsD, RhsD, RhsG> Backward for DivisionBackwardRight<LhsD, RhsD, RhsG>
where
    LhsD: Data,
    RhsD: Data,
    RhsG: Gradient,
    LhsD::Dim: Dimension + DimMax<RhsG::Dim>,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;
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
    LhsD: Data,
    RhsD: Data,
    RhsG: Gradient,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PowerBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct PowerBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    exp: i32,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U, D> PowerBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>, exp: i32) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));
        Self {
            operand_grad,
            operand_data,
            exp,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T, U, D> Gradient for PowerBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
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

impl<T, U, D> Backward for PowerBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SumBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SumBackward<T>
where
    T: Gradient,
{
    operand: Rc<T>,
    gradient: RefCell<Tensor<Ix1>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T> SumBackward<T>
where
    T: Gradient,
{
    pub fn new(operand: Rc<T>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(1));
        Self {
            operand,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T> Gradient for SumBackward<T>
where
    T: Gradient,
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

impl<T> Backward for SumBackward<T>
where
    T: Gradient,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LognBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct LognBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U, D> LognBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));
        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T, U, D> Gradient for LognBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
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

impl<T, U, D> Backward for LognBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReLUBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ReLUBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U, D> ReLUBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));
        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T, U, D> Gradient for ReLUBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
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

impl<T, U, D> Backward for ReLUBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LeakyReLUBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct LeakyReLUBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U, D> LeakyReLUBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));
        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T, U, D> Gradient for LeakyReLUBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
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

impl<T, U, D> Backward for LeakyReLUBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftPlusBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~S

pub struct SoftPlusBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U, D> SoftPlusBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));
        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T, U, D> Gradient for SoftPlusBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
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

impl<T, U, D> Backward for SoftPlusBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SigmoidBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SigmoidBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U, D> SigmoidBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));
        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T, U, D> Gradient for SigmoidBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
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

impl<T, U, D> Backward for SigmoidBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TanHBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct TanHBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U, D> TanHBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));
        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T, U, D> Gradient for TanHBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
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

impl<T, U, D> Backward for TanHBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ExpBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ExpBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U, D> ExpBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));
        Self {
            operand_grad,
            operand_data,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T, U, D> Gradient for ExpBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
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

impl<T, U, D> Backward for ExpBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftmaxBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SoftmaxBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    operand_grad: Rc<T>,
    operand_data: Rc<U>,
    axis: usize,
    gradient: RefCell<Tensor<T::Dim>>,
    can_overwrite: Cell<bool>,
    was_computed: bool,
}

impl<T, U, D> SoftmaxBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    pub fn new(operand_grad: Rc<T>, operand_data: Rc<U>, axis: usize) -> Self {
        let gradient = RefCell::new(Tensor::zeros(operand_grad.gradient().raw_dim()));

        Self {
            operand_grad,
            operand_data,
            axis,
            gradient,
            can_overwrite: Cell::new(true),
            was_computed: false,
        }
    }
}

impl<T, U, D> Gradient for SoftmaxBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
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

impl<T, U, D> Backward for SoftmaxBackward<T, U, D>
where
    T: Gradient<Dim = D>,
    U: Data<Dim = D>,
    D: Dimension,
{
    fn backward(&mut self) -> bool {
        if self.was_computed {
            return false;
        }
        self.was_computed = true;

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
                fill_jacobian(&mut jacobian, &op_data_lane);
                general_mat_vec_mul(1.0, &jacobian, &grad_lane, 0., &mut op_grad_lane);
            });
            self.operand_grad.was_overwritten();
        } else {
            zip.par_for_each(|mut op_grad_lane, grad_lane, op_data_lane| {
                let mut jacobian = Tensor::<Ix2>::zeros(jacobian_shape);
                fill_jacobian(&mut jacobian, &op_data_lane);
                general_mat_vec_mul(1.0, &jacobian, &grad_lane, 1., &mut op_grad_lane);
            });
        }
        true
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//
