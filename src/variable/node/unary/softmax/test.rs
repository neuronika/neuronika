use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, Overwrite, Rc, Softmax, SoftmaxBackward, Tensor,
};

mod forward {
    use super::{assert_almost_equals, new_input, new_tensor, Data, Forward, Softmax, Tensor};

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Softmax::new(input, 0);

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Softmax::new(input, 0);

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
    fn forward_rows() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Softmax::new(input.clone(), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![
                    0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330, 0.950330,
                    0.950330,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        {
            let mut data = input.data_mut();
            *data = &*data + &Tensor::from_elem(1, 1.);
        }
        assert_almost_equals(
            &*input.data(),
            &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
        );

        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![
                    0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330, 0.950330,
                    0.950330,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![
                    0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330, 0.950330,
                    0.950330,
                ],
            ),
        );
    }

    #[test]
    fn forward_columns() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Softmax::new(input.clone(), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![
                    0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241, 0.090031, 0.244728,
                    0.665241,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        {
            let mut data = input.data_mut();
            *data = &*data + &Tensor::from_elem(1, 1.);
        }
        assert_almost_equals(
            &*input.data(),
            &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
        );

        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![
                    0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241, 0.090031, 0.244728,
                    0.665241,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![
                    0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241, 0.090031, 0.244728,
                    0.665241,
                ],
            ),
        );
    }

    #[test]
    fn debug() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Softmax::new(input.clone(), 0);

        let output = "Softmax { data: [[0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2, axis: 0, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Softmax::new(input.clone(), 0);

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Forward,
        Gradient, Overwrite, Rc, Softmax, SoftmaxBackward, Tensor,
    };

    #[test]
    fn creation() {
        let axis = 0;
        let node = SoftmaxBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            Rc::new(Softmax::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                axis,
            )),
            axis,
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let axis = 0;
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = SoftmaxBackward::new(
            diff.clone(),
            Rc::new(Softmax::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                axis,
            )),
            axis,
        );

        node.backward();
        assert!(node.can_overwrite());
        assert!(!diff.can_overwrite());

        node.backward();
        assert!(node.can_overwrite());
        assert!(!diff.can_overwrite());

        diff.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(diff.can_overwrite());

        diff.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(diff.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(diff.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(diff.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!diff.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!diff.can_overwrite());
    }

    #[test]
    fn backward_rows() {
        let axis = 0;
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let not_diff = Rc::new(Softmax::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            axis,
        ));
        not_diff.forward();
        let node_b = SoftmaxBackward::new(diff.clone(), not_diff, axis);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node_b.gradient_mut() = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        assert_almost_equals(
            &*node_b.gradient(),
            &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node_b.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.01376, -0.01376, -0.01376, -0.13455, -0.13455, -0.13455, 0.148323, 0.148323,
                    0.148323,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node_b.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.02752, -0.02752, -0.02752, -0.2691, -0.2691, -0.2691, 0.296646, 0.296646,
                    0.296646,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node_b.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.01376, -0.01376, -0.01376, -0.13455, -0.13455, -0.13455, 0.148323, 0.148323,
                    0.148323,
                ],
            ),
        );
    }

    #[test]
    fn backward_columns() {
        let axis = 1;
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let not_diff = Rc::new(Softmax::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            axis,
        ));
        not_diff.forward();
        let node = SoftmaxBackward::new(diff.clone(), not_diff, axis);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        assert_almost_equals(
            &*node.gradient(),
            &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.1418, -0.1408, 0.2826, -0.1418, -0.1408, 0.2826, -0.1418, -0.1408, 0.2826,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.2836, -0.2815, 0.5652, -0.2836, -0.2815, 0.5652, -0.2836, -0.2815, 0.5652,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.1418, -0.1408, 0.2826, -0.1418, -0.1408, 0.2826, -0.1418, -0.1408, 0.2826,
                ],
            ),
        );
    }

    #[test]
    fn debug() {
        let node = SoftmaxBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            1,
        );

        let output = "SoftmaxBackward { gradient: Some([[0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2), axis: 1, overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let node = SoftmaxBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            1,
        );

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }

    #[test]
    fn no_grad() {
        // SoftmaxBackward
        let node = SoftmaxBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            0,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
