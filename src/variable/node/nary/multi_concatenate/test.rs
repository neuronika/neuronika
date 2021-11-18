use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, MultiConcatenate, MultiConcatenateBackward, Overwrite, Tensor,
};

mod forward {
    use super::{
        assert_almost_equals, new_input, new_tensor, Data, Forward, MultiConcatenate, Tensor,
    };

    #[test]
    fn creation() {
        let first = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let second = new_input((3, 3), vec![0.; 9]);
        let node = MultiConcatenate::new(vec![first, second], 0, new_tensor((6, 3), vec![0.; 18]));

        assert_eq!(*node.data(), Tensor::from_elem((6, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((6, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let first = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let second = new_input((3, 3), vec![0.; 9]);
        let node = MultiConcatenate::new(vec![first, second], 0, new_tensor((6, 3), vec![0.; 18]));

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
        let first = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let second = new_input((2, 3), vec![1.; 6]);
        let node = MultiConcatenate::new(
            vec![first.clone(), second],
            0,
            new_tensor((5, 3), vec![0.; 15]),
        );

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
            let mut data = first.data_mut();
            *data = &*data + &Tensor::from_elem(1, 1.);
        }
        assert_almost_equals(
            &*first.data(),
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
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_tensor, Backward, Gradient,
        MultiConcatenateBackward, Overwrite, Tensor,
    };

    #[test]
    fn creation() {
        let node = MultiConcatenateBackward::new(
            vec![
                new_backward_input((4, 3), vec![0.; 12]),
                new_backward_input((4, 2), vec![0.; 8]),
            ],
            1,
            ndarray::Dim([4, 5]),
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((4, 5), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((4, 5), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let first = new_backward_input((4, 3), vec![0.; 12]);
        let second = new_backward_input((4, 2), vec![0.; 8]);
        let node = MultiConcatenateBackward::new(
            vec![first.clone(), second.clone()],
            1,
            ndarray::Dim([4, 5]),
        );

        node.backward();
        assert!(node.can_overwrite());
        assert!(!first.can_overwrite());
        assert!(!second.can_overwrite());

        node.backward();
        assert!(node.can_overwrite());
        assert!(!first.can_overwrite());
        assert!(!second.can_overwrite());

        first.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(first.can_overwrite());
        assert!(!second.can_overwrite());

        first.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(first.can_overwrite());
        assert!(!second.can_overwrite());

        second.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(first.can_overwrite());
        assert!(second.can_overwrite());

        second.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(first.can_overwrite());
        assert!(second.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(first.can_overwrite());
        assert!(second.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(first.can_overwrite());
        assert!(second.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!first.can_overwrite());
        assert!(!second.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!first.can_overwrite());
        assert!(!second.can_overwrite());
    }

    #[test]
    fn backward() {
        let first = new_backward_input((3, 4), vec![0.; 12]);
        let second = new_backward_input((2, 4), vec![0.; 8]);
        let node = MultiConcatenateBackward::new(
            vec![first.clone(), second.clone()],
            0,
            ndarray::Dim([5, 4]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
        assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*first.gradient(), &new_tensor((3, 4), vec![1.; 12]));
        assert_almost_equals(&*second.gradient(), &new_tensor((2, 4), vec![1.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*first.gradient(), &new_tensor((3, 4), vec![2.; 12]));
        assert_almost_equals(&*second.gradient(), &new_tensor((2, 4), vec![2.; 8]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        first.set_overwrite(true);
        second.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*first.gradient(), &new_tensor((3, 4), vec![1.; 12]));
        assert_almost_equals(&*second.gradient(), &new_tensor((2, 4), vec![1.; 8]));
    }

    #[test]
    fn no_grad() {
        // MultiConcatenateBackward
        let node = MultiConcatenateBackward::new(
            vec![
                new_backward_input((3, 3), vec![0.; 9]),
                new_backward_input((3, 3), vec![0.; 9]),
            ],
            0,
            ndarray::Dim([6, 3]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
