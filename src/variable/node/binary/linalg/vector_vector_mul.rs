use super::{
    expect_tensor, expect_tensor_mut, push_vec_vec_gradient, Backward, Data, DotDim, Forward,
    Gradient, Overwrite, Tensor,
};
use ndarray::Ix1;
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix1>>,
    computed: Cell<bool>,
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
            computed: Cell::new(false),
        }
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

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix1>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        self.data.borrow_mut()[0] = self.left.data().dot(&*self.right.data());
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod test {

    use super::*;

    #[cfg(feature = "blas")]
    extern crate blas_src;

    mod forward {
        use super::*;

        #[test]
        fn creation() {
            let left = new_input(3, vec![2.; 3]);
            let right = new_input(3, vec![1., 2., 3.]);
            let node = VectorVectorMul::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem(1, 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let left = new_input(3, vec![2.; 3]);
            let right = new_input(3, vec![1., 2., 3.]);
            let node = VectorVectorMul::new(left, right);

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
            let left = new_input(3, vec![2.; 3]);
            let right = new_input(3, vec![1., 2., 3.]);
            let node = VectorVectorMul::new(left.clone(), right);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![12.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *left.data_mut() = new_tensor(3, vec![-2.; 3]);
            assert_almost_equals(&*left.data(), &new_tensor(3, vec![-2.; 3]));

            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![12.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![-12.]));
        }
    }

    mod backward {
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
            assert!(node.can_overwrite());
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![8., 10., 12.]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor(3, vec![2., 4., 6.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![2., 4., 6.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![1., 2., 3.]));
        }
    }
}
