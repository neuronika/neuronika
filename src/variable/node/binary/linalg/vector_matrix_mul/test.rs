use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, Overwrite, Tensor, VectorMatrixMul, VectorMatrixMulBackward,
    VectorMatrixMulBackwardLeft, VectorMatrixMulBackwardRight,
};

#[cfg(feature = "blas")]
extern crate blas_src;

mod forward {
    use super::{
        assert_almost_equals, new_input, new_tensor, Data, Forward, Tensor, VectorMatrixMul,
    };

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
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Gradient,
        Overwrite, Tensor, VectorMatrixMulBackward, VectorMatrixMulBackwardLeft,
        VectorMatrixMulBackwardRight,
    };

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
        let node = VectorMatrixMulBackwardRight::new(new_input(3, vec![1., 2., 3.]), diff.clone());

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

    #[test]
    fn no_grad() {
        // VectorMatrixMulBackward
        let node = VectorMatrixMulBackward::new(
            new_input(3, vec![0.; 3]),
            new_backward_input(3, vec![0.; 3]),
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // VectorMatrixMulBackwardLeft
        let node = VectorMatrixMulBackwardLeft::new(
            new_backward_input(3, vec![0.; 3]),
            new_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // VectorMatrixMulBackwardRight
        let node = VectorMatrixMulBackwardRight::new(
            new_input(3, vec![0.; 3]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
