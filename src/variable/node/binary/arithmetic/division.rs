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
    computed: Cell<bool>,
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
            computed: Cell::new(false),
        }
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

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for Division<Lhs, Rhs>
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
            .for_each(|v, l, r| *v = l / r);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod test {
    use super::*;

    mod forward {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![2.; 9]);
            let node = Division::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![2.; 9]);
            let node = Division::new(left, right);

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
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![2.; 9]);
            let node = Division::new(left, right.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *right.data_mut() = new_tensor((3, 3), vec![-2.; 9]);
            assert_almost_equals(&*right.data(), &new_tensor((3, 3), vec![-2.; 9]));

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    mod backward {
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
            assert!(node.can_overwrite());
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.4; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.24; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![1.2; 3]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.24; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.4; 9]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![-0.72; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![0.4; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![1.2; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-0.24; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-0.72; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-0.36; 3]));
        }
    }

    #[test]
    fn no_grad() {
        // DivisionBackward
        let node = DivisionBackward::new(
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // DivisionBackwardLeft
        let node = DivisionBackwardLeft::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // DivisionBackwardRight
        let node = DivisionBackwardRight::new(
            new_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
