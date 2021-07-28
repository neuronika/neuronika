use super::*;
use ndarray::{linalg::general_mat_vec_mul, s, NewAxis};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix2>,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    data: RefCell<Tensor<Ix1>>,
    computed: Cell<bool>,
}

impl<Lhs, Rhs> VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix2>,
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

impl<Lhs, Rhs> Data for VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix2>,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Data<Dim = Ix1>,
    Rhs: Data<Dim = Ix2>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        general_mat_vec_mul(
            1.0,
            &self.right.data().t(),
            &*self.left.data(),
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
            let left = new_input(3, vec![1.; 3]);
            let right = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = VectorMatrixMul::new(left, right);

            assert_eq!(*node.data(), Tensor::from_elem(3, 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let left = new_input(3, vec![1.; 3]);
            let right = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = VectorMatrixMul::new(left, right);

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
            let left = new_input(3, vec![1.; 3]);
            let right = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = VectorMatrixMul::new(left.clone(), right);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(3, vec![12., 15., 18.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *left.data_mut() = new_tensor(3, vec![-2.; 3]);
            assert_almost_equals(&*left.data(), &new_tensor(3, vec![-2.; 3]));

            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(3, vec![12., 15., 18.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(3, vec![-24., -30., -36.]));
        }
    }

    mod backward {
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
            assert!(node.can_overwrite());
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![12., 30., 48.]));
            assert_almost_equals(
                &*rhs.gradient(),
                &new_tensor((3, 3), vec![2., 2., 2., 4., 4., 4., 6., 6., 6.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![12., 30., 48.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![2., 2., 2., 4., 4., 4., 6., 6., 6.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor((3, 3), vec![1., 1., 1., 2., 2., 2., 3., 3., 3.]),
            );
        }
    }
}
