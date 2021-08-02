use super::{
    broadcasted_zeros, expect_tensor, expect_tensor_mut, push_gradient, reduce, Backward,
    BroadTensor, Broadcasted, Data, Forward, Gradient, Overwrite, Tensor,
};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{DimMax, Dimension, Zip};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
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
    computed: Cell<bool>,
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
            computed: Cell::new(false),
        }
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

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for Subtraction<Lhs, Rhs>
where
    Lhs: Data,
    Rhs: Data,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.data())
            .and_broadcast(&*self.right.data())
            .for_each(|v, l, r| *v = l - r);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
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

        let reduced = reduce(&self.right.gradient_mut(), &self.gradient());
        let mut right_grad = self.right.gradient_mut();
        let zip = Zip::from(&mut *right_grad).and_broadcast(&reduced);
        if self.right.can_overwrite() {
            self.right.set_overwrite(false);
            zip.for_each(|right_el, reduced_el| *right_el = -reduced_el);
        } else {
            zip.for_each(|right_el, reduced_el| *right_el += -reduced_el);
        }
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
        let mut grad = self.operand.gradient_mut();
        let reduced = reduce(&*grad, &self.gradient());
        let zip = Zip::from(&mut *grad).and_broadcast(&reduced);
        if self.operand.can_overwrite() {
            self.operand.set_overwrite(false);
            zip.for_each(|operand_el, reduced_el| *operand_el = -reduced_el);
        } else {
            zip.for_each(|operand_el, reduced_el| *operand_el += -reduced_el);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod test {
    use super::*;

    mod forward {

        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = Subtraction::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = Subtraction::new(left, right);

            node.forward();
            assert!(node.was_computed());

            node.forward();
            assert!(node.was_computed());

            node.reset_computation();
            assert!(!node.was_computed());

            node.reset_computation();
            assert!(!node.was_computed());
        }

        #[test]
        fn forward() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = Subtraction::new(left.clone(), right);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![-5., -4., -3., -2., -1., 0., 1., 2., 3.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = SubtractionBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                new_backward_input((3, 3), vec![0.; 9]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((3, 3), vec![0.; 9]);
            let rhs = new_backward_input((3, 3), vec![0.; 9]);
            let node = SubtractionBackward::new(lhs.clone(), rhs.clone());

            node.backward();
            assert!(node.can_overwrite());
            assert!(!lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            node.backward();
            assert!(node.can_overwrite());
            assert!(!lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            lhs.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            lhs.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            rhs.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(rhs.can_overwrite());

            rhs.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(rhs.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(rhs.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(rhs.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!lhs.can_overwrite());
            assert!(!rhs.can_overwrite());
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-6.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-3.; 3]));
        }
    }
}
