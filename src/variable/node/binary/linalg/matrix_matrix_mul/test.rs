use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, MatrixMatrixMul, MatrixMatrixMulBackward, MatrixMatrixMulBackwardLeft,
    MatrixMatrixMulBackwardRight, Overwrite, Tensor,
};

#[cfg(feature = "blas")]
extern crate blas_src;

mod forward {
    use super::{
        assert_almost_equals, new_input, new_tensor, Data, Forward, MatrixMatrixMul, Tensor,
    };

    #[test]
    fn creation() {
        let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right = new_input((3, 3), vec![1.; 9]);
        let node = MatrixMatrixMul::new(left, right);

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((3, 3), 0.));
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
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Gradient,
        MatrixMatrixMulBackward, MatrixMatrixMulBackwardLeft, MatrixMatrixMulBackwardRight,
        Overwrite, Tensor,
    };

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

    #[test]
    fn no_grad() {
        // MatrixMatrixMulBackward
        let node = MatrixMatrixMulBackward::new(
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // MatrixMatrixMulBackwardLeft
        let node = MatrixMatrixMulBackwardLeft::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // MatrixMatrixMulBackwardRight
        let node = MatrixMatrixMulBackwardRight::new(
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
