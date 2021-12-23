use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, Overwrite, VectorVectorMul, VectorVectorMulBackward, VectorVectorMulBackwardUnary,
};
use ndarray::arr0;

#[cfg(feature = "blas")]
extern crate blas_src;

mod forward {
    use super::{
        arr0, assert_almost_equals, new_input, new_tensor, Data, Forward, VectorVectorMul,
    };

    #[test]
    fn creation() {
        let left = new_input(3, vec![2.; 3]);
        let right = new_input(3, vec![1., 2., 3.]);
        let node = VectorVectorMul::new(left, right);

        assert_eq!(*node.data(), arr0(0.));
        assert_eq!(*node.data_mut(), arr0(0.));
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
        assert_almost_equals(&*node.data(), &arr0(12.0));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *left.data_mut() = new_tensor(3, vec![-2.; 3]);
        assert_almost_equals(&*left.data(), &new_tensor(3, vec![-2.; 3]));

        node.forward();
        assert_almost_equals(&*node.data(), &arr0(12.0));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(&*node.data(), &arr0(-12.));
    }

    #[test]
    fn debug() {
        let left = new_input(3, vec![2.; 3]);
        let right = new_input(3, vec![1., 2., 3.]);
        let node = VectorVectorMul::new(left, right);

        let output =
            "VectorVectorMul { data: 0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let left = new_input(3, vec![2.; 3]);
        let right = new_input(3, vec![1., 2., 3.]);
        let node = VectorVectorMul::new(left, right);

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use super::{
        arr0, assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Gradient,
        Overwrite, VectorVectorMulBackward, VectorVectorMulBackwardUnary,
    };

    #[test]
    fn creation() {
        let node = VectorVectorMulBackward::new(
            new_input(3, vec![1., 2., 3.]),
            new_backward_input(3, vec![0.; 3]),
            new_input(3, vec![4., 5., 6.]),
            new_backward_input(3, vec![0.; 3]),
        );

        assert_eq!(*node.gradient(), arr0(0.));
        assert_eq!(*node.gradient_mut(), arr0(0.));
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
        *node.gradient_mut() = arr0(1.);
        assert_almost_equals(&*node.gradient(), &arr0(1.));

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
    fn debug() {
        let lhs = new_backward_input(3, vec![0.; 3]);
        let rhs = new_backward_input(3, vec![0.; 3]);
        let node = VectorVectorMulBackward::new(
            new_input(3, vec![1., 2., 3.]),
            lhs,
            new_input(3, vec![4., 5., 6.]),
            rhs,
        );

        let output = "VectorVectorMulBackward { gradient: Some(0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0), overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let lhs = new_backward_input(3, vec![0.; 3]);
        let rhs = new_backward_input(3, vec![0.; 3]);
        let node = VectorVectorMulBackward::new(
            new_input(3, vec![1., 2., 3.]),
            lhs,
            new_input(3, vec![4., 5., 6.]),
            rhs,
        );

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }

    #[test]
    fn backward_unary() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let node = VectorVectorMulBackwardUnary::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = arr0(1.);
        assert_almost_equals(&*node.gradient(), &arr0(1.));

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

    #[test]
    fn debug_unary() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let node = VectorVectorMulBackwardUnary::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

        let output = "VectorVectorMulBackwardUnary { gradient: Some(0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0), overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display_unary() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let node = VectorVectorMulBackwardUnary::new(diff.clone(), new_input(3, vec![1., 2., 3.]));

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }

    #[test]
    fn no_grad() {
        // VectorVectorMulBackward
        let node = VectorVectorMulBackward::new(
            new_input(3, vec![0.; 3]),
            new_backward_input(3, vec![0.; 3]),
            new_input(3, vec![0.; 3]),
            new_backward_input(3, vec![0.; 3]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), arr0(0.));

        // VectorVectorMulBackwardUnary
        let node = VectorVectorMulBackwardUnary::new(
            new_backward_input(3, vec![0.; 3]),
            new_input(3, vec![0.; 3]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), arr0(0.));
    }
}
