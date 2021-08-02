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
    computed: Cell<bool>,
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
            computed: Cell::new(false),
        }
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

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for Multiplication<Lhs, Rhs>
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
            .for_each(|v, l, r| *v = l * r);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod test {
    use super::*;

    mod forward {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![-1.; 9]);
            let node = Multiplication::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![-1.; 9]);
            let node = Multiplication::new(left, right);

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
            let right = new_input((3, 3), vec![-1.; 9]);
            let node = Multiplication::new(left, right.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *right.data_mut() = new_tensor((3, 3), vec![2.; 9]);
            assert_almost_equals(&*right.data(), &new_tensor((3, 3), vec![2.; 9]));

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    mod backward {
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
            assert!(node.can_overwrite());
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![10.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![6.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![30.; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![6.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![10.; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![18.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![10.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![30.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![15.; 3]));
        }
    }
}
