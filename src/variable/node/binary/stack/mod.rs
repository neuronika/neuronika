use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient, Overwrite,
    Tensor,
};
use ndarray::{stack, Axis, Dimension, RemoveAxis, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

pub struct Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    data: RefCell<Tensor<<Lhs::Dim as Dimension>::Larger>>,
    computed: Cell<bool>,
}

impl<Lhs, Rhs> Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
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
            computed: Cell::new(false),
        }
    }
}

impl<Lhs, Rhs> Data for Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    type Dim = <Lhs::Dim as Dimension>::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for Stack<Lhs, Rhs>
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
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

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

#[cfg(test)]
mod test;
