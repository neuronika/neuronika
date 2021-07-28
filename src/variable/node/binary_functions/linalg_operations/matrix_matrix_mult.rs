use super::*;
use ndarray::linalg::general_mat_mul;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix2>>,
    computed: Cell<bool>,
}

impl<Lhs, Rhs> MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(left.data().raw_dim(), right.data().raw_dim());
        let data = RefCell::new(Tensor::zeros((shape[0], shape[1])));

        Self {
            left,
            right,
            data,
            computed: Cell::new(false),
        }
    }
}

impl<Lhs, Rhs> Data for MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    type Dim = Ix2;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix2>,
    Rhs: Data<Dim = Ix2>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        general_mat_mul(
            1.0,
            &*self.left.data(),
            &*self.right.data(),
            0.0,
            &mut *self.data.borrow_mut(),
        );
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
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
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = MatrixMatrixMul::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let right = new_input((3, 3), vec![1.; 9]);
            let node = MatrixMatrixMul::new(left, right);

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
            let right = new_input((3, 3), vec![1.; 9]);
            let node = MatrixMatrixMul::new(left, right.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![6., 6., 6., 15., 15., 15., 24., 24., 24.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *right.data_mut() = new_tensor((3, 3), vec![-2.; 9]);
            assert_almost_equals(&*right.data(), &new_tensor((3, 3), vec![-2.; 9]));

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![6., 6., 6., 15., 15., 15., 24., 24., 24.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![-12., -12., -12., -30., -30., -30., -48., -48., -48.],
                ),
            );
        }
    }

    mod backward {
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
            assert!(node.can_overwrite());
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*lhs.gradient(),
                &new_tensor((3, 3), vec![66., 84., 102., 66., 84., 102., 66., 84., 102.]),
            );
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((3, 3), vec![24., 24., 24., 30., 30., 30., 36., 36., 36.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![66., 84., 102., 66., 84., 102., 66., 84., 102.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![24., 24., 24., 30., 30., 30., 36., 36., 36.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![12., 12., 12., 15., 15., 15., 18., 18., 18.]),
            );
        }
    }
}
