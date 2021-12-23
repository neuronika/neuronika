use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, MatrixMatrixMulT, MatrixMatrixMulTBackward, MatrixMatrixMulTBackwardLeft,
    MatrixMatrixMulTBackwardRight, Overwrite, Tensor,
};

#[cfg(feature = "blas")]
extern crate blas_src;

mod forward {
    use super::{
        assert_almost_equals, new_input, new_tensor, Data, Forward, MatrixMatrixMulT, Tensor,
    };

    #[test]
    fn creation() {
        let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right = new_input((2, 3), vec![1.; 6]);
        let node = MatrixMatrixMulT::new(left, right);

        assert_eq!(*node.data(), Tensor::from_elem((3, 2), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((3, 2), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right = new_input((2, 3), vec![1.; 6]);
        let node = MatrixMatrixMulT::new(left, right);

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
        let right = new_input((2, 3), vec![1.; 6]);
        let node = MatrixMatrixMulT::new(left, right.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 2), vec![6., 6., 15., 15., 24., 24.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *right.data_mut() = new_tensor((2, 3), vec![-2.; 6]);
        assert_almost_equals(&*right.data(), &new_tensor((2, 3), vec![-2.; 6]));

        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 2), vec![6., 6., 15., 15., 24., 24.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 2), vec![-12., -12., -30., -30., -48., -48.]),
        );
    }

    #[test]
    fn debug() {
        let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right = new_input((3, 3), vec![1.; 9]);
        let node = MatrixMatrixMulT::new(left, right);

        let output = "MatrixMatrixMulT { data: [[0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right = new_input((3, 3), vec![1.; 9]);
        let node = MatrixMatrixMulT::new(left, right);

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}
mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Gradient,
        MatrixMatrixMulTBackward, MatrixMatrixMulTBackwardLeft, MatrixMatrixMulTBackwardRight,
        Overwrite, Tensor,
    };

    #[test]
    fn creation() {
        let node = MatrixMatrixMulTBackward::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((2, 3), vec![10., 11., 12., 13., 14., 15.]),
            new_backward_input((2, 3), vec![0.; 6]),
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((3, 2), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 2), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let lhs = new_backward_input((3, 3), vec![0.; 9]);
        let rhs = new_backward_input((2, 3), vec![0.; 6]);
        let node = MatrixMatrixMulTBackward::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            lhs.clone(),
            new_input((2, 3), vec![10., 11., 12., 13., 14., 15.]),
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
        let rhs = new_backward_input((2, 3), vec![0.; 6]);
        let node = MatrixMatrixMulTBackward::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            lhs.clone(),
            new_input((2, 3), vec![10., 11., 12., 13., 14., 15.]),
            rhs.clone(),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 2), vec![1.; 6]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 2), vec![1.; 6]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*lhs.gradient(),
            &new_tensor((3, 3), vec![23., 25., 27., 23., 25., 27., 23., 25., 27.]),
        );
        assert_almost_equals(
            &*rhs.gradient(),
            &new_tensor((2, 3), vec![12., 15., 18., 12., 15., 18.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*lhs.gradient(),
            &new_tensor((3, 3), vec![46., 50., 54., 46., 50., 54., 46., 50., 54.]),
        );
        assert_almost_equals(
            &*rhs.gradient(),
            &new_tensor((2, 3), vec![24., 30., 36., 24., 30., 36.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(
            &*lhs.gradient(),
            &new_tensor((3, 3), vec![23., 25., 27., 23., 25., 27., 23., 25., 27.]),
        );
        assert_almost_equals(
            &*rhs.gradient(),
            &new_tensor((2, 3), vec![12., 15., 18., 12., 15., 18.]),
        );
    }

    #[test]
    fn debug() {
        let lhs = new_backward_input((3, 3), vec![0.; 9]);
        let rhs = new_backward_input((3, 3), vec![0.; 9]);
        let node = MatrixMatrixMulTBackward::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            lhs,
            new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]),
            rhs,
        );

        let output = "MatrixMatrixMulTBackward { gradient: Some([[0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2), overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let lhs = new_backward_input((3, 3), vec![0.; 9]);
        let rhs = new_backward_input((3, 3), vec![0.; 9]);
        let node = MatrixMatrixMulTBackward::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            lhs,
            new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]),
            rhs,
        );

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }

    #[test]
    fn backward_left() {
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = MatrixMatrixMulTBackwardLeft::new(
            diff.clone(),
            new_input((2, 3), vec![10., 11., 12., 13., 14., 15.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 2), vec![1.; 6]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 2), vec![1.; 6]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor((3, 3), vec![23., 25., 27., 23., 25., 27., 23., 25., 27.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor((3, 3), vec![46., 50., 54., 46., 50., 54., 46., 50., 54.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor((3, 3), vec![23., 25., 27., 23., 25., 27., 23., 25., 27.]),
        );
    }

    #[test]
    fn debug_left() {
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = MatrixMatrixMulTBackwardLeft::new(
            diff.clone(),
            new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]),
        );

        let output = "MatrixMatrixMulTBackwardLeft { gradient: Some([[0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2), overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display_left() {
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = MatrixMatrixMulTBackwardLeft::new(
            diff.clone(),
            new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]),
        );
        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }

    #[test]
    fn backward_right() {
        let diff = new_backward_input((2, 3), vec![0.; 6]);
        let node = MatrixMatrixMulTBackwardRight::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            diff.clone(),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 2), vec![1.; 6]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 2), vec![1.; 6]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor((2, 3), vec![12., 15., 18., 12., 15., 18.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor((2, 3), vec![24., 30., 36., 24., 30., 36.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor((2, 3), vec![12., 15., 18., 12., 15., 18.]),
        );
    }

    #[test]
    fn debug_right() {
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = MatrixMatrixMulTBackwardRight::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            diff.clone(),
        );

        let output = "MatrixMatrixMulTBackwardRight { gradient: Some([[0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2), overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display_right() {
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = MatrixMatrixMulTBackwardRight::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            diff.clone(),
        );

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }

    #[test]
    fn no_grad() {
        // MatrixMatrixMulTBackward
        let node = MatrixMatrixMulTBackward::new(
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // MatrixMatrixMulTBackwardLeft
        let node = MatrixMatrixMulTBackwardLeft::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // MatrixMatrixMulTBackwardRight
        let node = MatrixMatrixMulTBackwardRight::new(
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
