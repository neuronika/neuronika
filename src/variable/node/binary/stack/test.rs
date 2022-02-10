use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Cache, Data,
    Forward, Gradient, Overwrite, Stack, StackBackward, StackBackwardLeft, StackBackwardRight,
    Tensor,
};
mod forward {
    use super::{assert_almost_equals, new_input, new_tensor, Cache, Data, Forward, Stack, Tensor};

    #[test]
    fn creation() {
        let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let right = new_input((3, 3), vec![0.; 9]);
        let node = Stack::new(left, right, 0);

        assert_eq!(*node.data(), Tensor::from_elem((2, 3, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((2, 3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let right = new_input((3, 3), vec![0.; 9]);
        let node = Stack::new(left, right, 0);

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
    #[should_panic]
    fn fail_by_rows() {
        Stack::new(
            new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            new_input((3, 2), vec![0.; 6]),
            0,
        );
    }

    #[test]
    #[should_panic]
    fn fail_by_columns() {
        Stack::new(
            new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            new_input((2, 3), vec![0.; 6]),
            1,
        );
    }

    #[test]
    fn forward_rows() {
        let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let right = new_input((3, 3), vec![0.; 9]);
        let node = Stack::new(left.clone(), right, 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (2, 3, 3),
                vec![
                    -4., -3., -2., -1., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                ],
            ),
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
            &new_tensor(
                (2, 3, 3),
                vec![
                    -4., -3., -2., -1., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (2, 3, 3),
                vec![
                    -3., -2., -1., 0., 1., 2., 3., 4., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                ],
            ),
        );
    }

    #[test]
    fn forward_columns() {
        let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let right = new_input((3, 3), vec![0.; 9]);
        let node = Stack::new(left.clone(), right, 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 2, 3),
                vec![
                    -4., -3., -2., 0., 0., 0., -1., 0., 1., 0., 0., 0., 2., 3., 4., 0., 0., 0.,
                ],
            ),
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
            &new_tensor(
                (3, 2, 3),
                vec![
                    -4., -3., -2., 0., 0., 0., -1., 0., 1., 0., 0., 0., 2., 3., 4., 0., 0., 0.,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 2, 3),
                vec![
                    -3., -2., -1., 0., 0., 0., 0., 1., 2., 0., 0., 0., 3., 4., 5., 0., 0., 0.,
                ],
            ),
        );
    }

    #[test]
    fn debug() {
        let left = new_input(1, vec![0.]);
        let right = new_input(1, vec![0.]);
        let node = Stack::new(left, right, 0);

        let output = "Stack { data: [[0.0],\n [0.0]], shape=[2, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2, axis: 0, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let left = new_input(1, vec![0.]);
        let right = new_input(1, vec![0.]);
        let node = Stack::new(left, right, 0);

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Gradient,
        Overwrite, StackBackward, StackBackwardLeft, StackBackwardRight, Tensor,
    };

    #[test]
    fn creation() {
        let node = StackBackward::new(
            new_backward_input((4, 3), vec![0.; 12]),
            new_backward_input((4, 3), vec![0.; 12]),
            0,
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((2, 4, 3), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((2, 4, 3), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let lhs = new_backward_input((4, 3), vec![0.; 12]);
        let rhs = new_backward_input((4, 3), vec![0.; 12]);
        let node = StackBackward::new(lhs.clone(), rhs.clone(), 0);

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
    fn backward_rows() {
        let lhs = new_backward_input((4, 3), vec![0.; 12]);
        let rhs = new_backward_input((4, 3), vec![0.; 12]);
        let node = StackBackward::new(lhs.clone(), rhs.clone(), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
        assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
    }

    #[test]
    fn backward_columns() {
        let lhs = new_backward_input((4, 3), vec![0.; 12]);
        let rhs = new_backward_input((4, 3), vec![0.; 12]);
        let node = StackBackward::new(lhs.clone(), rhs.clone(), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((4, 2, 3), vec![1.; 24]);
        assert_almost_equals(&*node.gradient(), &new_tensor((4, 2, 3), vec![1.; 24]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
    }

    #[test]
    fn backward_left_rows() {
        let lhs = new_backward_input((4, 3), vec![0.; 12]);
        let node = StackBackwardLeft::new(lhs.clone(), new_input((4, 3), vec![0.; 12]), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
        assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
    }

    #[test]
    fn backward_left_columns() {
        let lhs = new_backward_input((4, 3), vec![0.; 12]);
        let node = StackBackwardLeft::new(lhs.clone(), new_input((4, 3), vec![0.; 12]), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((4, 2, 3), vec![1.; 24]);
        assert_almost_equals(&*node.gradient(), &new_tensor((4, 2, 3), vec![1.; 24]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
    }

    #[test]
    fn backward_right_rows() {
        let rhs = new_backward_input((4, 3), vec![0.; 12]);
        let node = StackBackwardRight::new(new_input((4, 3), vec![0.; 12]), rhs.clone(), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
        assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
    }

    #[test]
    fn backward_right_columns() {
        let rhs = new_backward_input((4, 3), vec![0.; 12]);
        let node = StackBackwardRight::new(new_input((4, 3), vec![0.; 12]), rhs.clone(), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((4, 2, 3), vec![1.; 24]);
        assert_almost_equals(&*node.gradient(), &new_tensor((4, 2, 3), vec![1.; 24]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
    }

    #[test]
    fn no_grad() {
        // StackBackward
        let node = StackBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
            0,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // StackBackwardLeft
        let node = StackBackwardLeft::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            0,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // StackBackwardRight
        let node = StackBackwardRight::new(
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
            0,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }

    #[test]
    fn debug() {
        {
            let node = StackBackward::new(
                new_backward_input(1, vec![0.]),
                new_backward_input(1, vec![0.]),
                0,
            );

            let output = "StackBackward { gradient: Some([[0.0],\n [0.0]], shape=[2, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2), axis: 0, overwrite: true }";
            assert_eq!(output, format!("{:?}", node));
        }
    }

    #[test]
    fn debug_left() {
        let node =
            StackBackwardLeft::new(new_backward_input(1, vec![0.]), new_input(1, vec![0.]), 0);

        let output = "StackBackwardLeft { gradient: Some([[0.0],\n [0.0]], shape=[2, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2), axis: 0, overwrite: true }";
        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn debug_right() {
        let node =
            StackBackwardRight::new(new_input(1, vec![0.]), new_backward_input(1, vec![0.]), 0);

        let output = "StackBackwardRight { gradient: Some([[0.0],\n [0.0]], shape=[2, 1], strides=[1, 1], layout=CFcf (0xf), const ndim=2), axis: 0, overwrite: true }";
        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        {
            let node = StackBackward::new(
                new_backward_input(1, vec![0.]),
                new_backward_input(1, vec![0.]),
                0,
            );
            assert_eq!(format!("{}", node.gradient()), format!("{}", node));
        }
    }

    #[test]
    fn display_left() {
        {
            let node =
                StackBackwardLeft::new(new_backward_input(1, vec![0.]), new_input(1, vec![0.]), 0);
            assert_eq!(format!("{}", node.gradient()), format!("{}", node));
        }
    }

    #[test]
    fn display_right() {
        {
            let node =
                StackBackwardRight::new(new_input(1, vec![0.]), new_backward_input(1, vec![0.]), 0);

            assert_eq!(format!("{}", node.gradient()), format!("{}", node));
        }
    }
}
