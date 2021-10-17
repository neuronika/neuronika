use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Concatenate,
    ConcatenateBackward, ConcatenateBackwardLeft, ConcatenateBackwardRight, Data, Forward,
    Gradient, Overwrite, Tensor,
};

mod forward {
    use super::{assert_almost_equals, new_input, new_tensor, Concatenate, Data, Forward, Tensor};

    #[test]
    fn creation() {
        let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let right = new_input((2, 3), vec![1.; 6]);
        let node = Concatenate::new(left, right, 0);

        assert_eq!(*node.data(), Tensor::from_elem((5, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let right = new_input((2, 3), vec![1.; 6]);
        let node = Concatenate::new(left, right, 0);

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
        Concatenate::new(
            new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            new_input((3, 2), vec![1.; 6]),
            0,
        );
    }

    #[test]
    #[should_panic]
    fn fail_by_columns() {
        Concatenate::new(
            new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            new_input((2, 3), vec![1.; 6]),
            1,
        );
    }

    #[test]
    fn forward_rows() {
        let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let right = new_input((2, 3), vec![1.; 6]);
        let node = Concatenate::new(left.clone(), right, 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (5, 3),
                vec![
                    -4., -3., -2., -1., 0., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1.,
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
                (5, 3),
                vec![
                    -4., -3., -2., -1., 0., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1.,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (5, 3),
                vec![
                    -3., -2., -1., 0., 1., 2., 3., 4., 5., 1., 1., 1., 1., 1., 1.,
                ],
            ),
        );
    }

    #[test]
    fn forward_columns() {
        let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let right = new_input((3, 2), vec![1.; 6]);
        let node = Concatenate::new(left.clone(), right, 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 5),
                vec![
                    -4., -3., -2., 1., 1., -1., 0., 1., 1., 1., 2., 3., 4., 1., 1.,
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
                (3, 5),
                vec![
                    -4., -3., -2., 1., 1., -1., 0., 1., 1., 1., 2., 3., 4., 1., 1.,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 5),
                vec![
                    -3., -2., -1., 1., 1., 0., 1., 2., 1., 1., 3., 4., 5., 1., 1.,
                ],
            ),
        );
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward,
        ConcatenateBackward, ConcatenateBackwardLeft, ConcatenateBackwardRight, Gradient,
        Overwrite, Tensor,
    };

    #[test]
    fn creation() {
        let node = ConcatenateBackward::new(
            new_backward_input((4, 3), vec![0.; 12]),
            new_backward_input((4, 2), vec![0.; 8]),
            1,
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((4, 5), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((4, 5), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let lhs = new_backward_input((4, 3), vec![0.; 12]);
        let rhs = new_backward_input((4, 2), vec![0.; 8]);
        let node = ConcatenateBackward::new(lhs.clone(), rhs.clone(), 1);

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
        let lhs = new_backward_input((3, 4), vec![0.; 12]);
        let rhs = new_backward_input((2, 4), vec![0.; 8]);
        let node = ConcatenateBackward::new(lhs.clone(), rhs.clone(), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
        assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![2.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![2.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));
    }

    #[test]
    fn backward_columns() {
        let lhs = new_backward_input((4, 3), vec![0.; 12]);
        let rhs = new_backward_input((4, 2), vec![0.; 8]);
        let node = ConcatenateBackward::new(lhs.clone(), rhs.clone(), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((4, 5), vec![1.; 20]);
        assert_almost_equals(&*node.gradient(), &new_tensor((4, 5), vec![1.; 20]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![2.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));
    }

    #[test]
    fn backward_left_rows() {
        let lhs = new_backward_input((3, 4), vec![0.; 12]);
        let node = ConcatenateBackwardLeft::new(lhs.clone(), new_input((2, 4), vec![0.; 8]), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
        assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));
    }

    #[test]
    fn backward_left_columns() {
        let lhs = new_backward_input((4, 3), vec![0.; 12]);
        let node = ConcatenateBackwardLeft::new(lhs.clone(), new_input((4, 2), vec![0.; 8]), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((4, 5), vec![1.; 20]);
        assert_almost_equals(&*node.gradient(), &new_tensor((4, 5), vec![1.; 20]));

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
        let rhs = new_backward_input((2, 4), vec![0.; 8]);
        let node = ConcatenateBackwardRight::new(new_input((3, 4), vec![0.; 12]), rhs.clone(), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
        assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![2.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));
    }

    #[test]
    fn backward_right_columns() {
        let rhs = new_backward_input((4, 2), vec![0.; 8]);
        let node = ConcatenateBackwardRight::new(new_input((4, 3), vec![0.; 12]), rhs.clone(), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((4, 5), vec![1.; 20]);
        assert_almost_equals(&*node.gradient(), &new_tensor((4, 5), vec![1.; 20]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![2.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));
    }

    #[test]
    fn no_grad() {
        // ConcatenateBackward
        let node = ConcatenateBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
            0,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // ConcatenateBackwardLeft
        let node = ConcatenateBackwardLeft::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            0,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        // ConcatenateBackwardRight
        let node = ConcatenateBackwardRight::new(
            new_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
            0,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
