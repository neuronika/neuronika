use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, LogSoftmax, LogSoftmaxBackward, Overwrite, Rc, Tensor,
};

mod forward {
    use super::{assert_almost_equals, new_input, new_tensor, Data, Forward, LogSoftmax, Tensor};

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = LogSoftmax::new(input, 0);

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = LogSoftmax::new(input, 0);

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
        let node = LogSoftmax::new(input.clone(), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![
                    -6.050946, -6.050946, -6.050946, -3.050946, -3.050946, -3.050946, -0.050946,
                    -0.050946, -0.050946,
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
                    -6.050946, -6.050946, -6.050946, -3.050946, -3.050946, -3.050946, -0.050946,
                    -0.050946, -0.050946,
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
                    -6.0509, -6.0509, -6.0509, -3.0509, -3.0509, -3.0509, -0.0509, -0.0509, -0.0509,
                ],
            ),
        );
    }

    #[test]
    fn forward_columns() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = LogSoftmax::new(input.clone(), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![
                    -2.407606, -1.407606, -0.407606, -2.407606, -1.407606, -0.407606, -2.407606,
                    -1.407606, -0.407606,
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
                    -2.407606, -1.407606, -0.407606, -2.407606, -1.407606, -0.407606, -2.407606,
                    -1.407606, -0.407606,
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
                    -2.4076, -1.4076, -0.4076, -2.4076, -1.4076, -0.4076, -2.4076, -1.4076, -0.4076,
                ],
            ),
        );
    }

    #[test]
    fn debug() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = LogSoftmax::new(input.clone(), 0);

        let output = "LogSoftmax { data: [[0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2, axis: 0, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = LogSoftmax::new(input.clone(), 0);

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Forward,
        Gradient, LogSoftmax, LogSoftmaxBackward, Overwrite, Rc, Tensor,
    };

    #[test]
    fn creation() {
        let axis = 0;
        let node = LogSoftmaxBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            Rc::new(LogSoftmax::new(
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
        let node = LogSoftmaxBackward::new(
            diff.clone(),
            Rc::new(LogSoftmax::new(
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
        let not_diff = Rc::new(LogSoftmax::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            axis,
        ));
        not_diff.forward();
        let node = LogSoftmaxBackward::new(diff.clone(), not_diff, axis);

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
                    0.9717, 1.9647, 2.9576, 3.4322, 4.2903, 5.1483, -4.4040, -6.2550, -8.1059,
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
                    1.9435, 3.9293, 5.9152, 6.8645, 8.5806, 10.2967, -8.8079, -12.5099, -16.2119,
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
                    0.9717, 1.9647, 2.9576, 3.4322, 4.2903, 5.1483, -4.4040, -6.2550, -8.1059,
                ],
            ),
        );
    }

    #[test]
    fn backward_columns() {
        let axis = 1;
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let not_diff = Rc::new(LogSoftmax::new(
            new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            axis,
        ));
        not_diff.forward();
        let node = LogSoftmaxBackward::new(diff.clone(), not_diff, axis);

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
                    0.4598, 0.5316, -0.9914, 2.6495, 1.3291, -3.9786, 4.8393, 2.1265, -6.9658,
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
                    0.9196, 1.0633, -1.9829, 5.2991, 2.6581, -7.9572, 9.6785, 4.2530, -13.9316,
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
                    0.4598, 0.5316, -0.9914, 2.6495, 1.3291, -3.9786, 4.8393, 2.1265, -6.9658,
                ],
            ),
        );
    }

    #[test]
    fn no_grad() {
        // LogSoftmaxBackward
        let node = LogSoftmaxBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            1,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }

    #[test]
    fn debug() {
        let node = LogSoftmaxBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            1,
        );

        let output = "LogSoftmaxBackward { gradient: Some([[0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0],\n [0.0, 0.0, 0.0]], shape=[3, 3], strides=[3, 1], layout=Cc (0x5), const ndim=2), axis: 1, overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let node = LogSoftmaxBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            1,
        );

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }
}
