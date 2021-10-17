use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient, Overwrite,
    Tensor,
};
use ndarray::{concatenate, Axis, RemoveAxis, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

pub struct Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    data: RefCell<Tensor<Lhs::Dim>>,
    computed: Cell<bool>,
}

impl<Lhs, Rhs> Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
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
            computed: Cell::new(false),
        }
    }
}

impl<Lhs, Rhs> Data for Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    type Dim = Lhs::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
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
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

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

#[cfg(test)]
mod test;
