use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, Overwrite, Tensor, Unsqueeze, UnsqueezeBackward,
};

mod forward {
    use super::{assert_almost_equals, new_input, new_tensor, Data, Forward, Tensor, Unsqueeze};

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Unsqueeze::new(input, 0);

        assert_eq!(*node.data(), Tensor::from_elem((1, 3, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((1, 3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Unsqueeze::new(input, 0);

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
    fn fail() {
        Unsqueeze::new(
            new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            3,
        );
    }

    #[test]
    fn forward_rows() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Unsqueeze::new(input.clone(), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((1, 3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
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
            &new_tensor((1, 3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((1, 3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
        );
    }

    #[test]
    fn forward_columns() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Unsqueeze::new(input.clone(), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 1, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
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
            &new_tensor((3, 1, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 1, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
        );
    }

    #[test]
    fn forward_depths() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Unsqueeze::new(input.clone(), 2);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3, 1), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
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
            &new_tensor((3, 3, 1), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3, 1), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
        );
    }

    #[test]
    fn debug() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Unsqueeze::new(input.clone(), 2);

        let output = "Unsqueeze { data: [[[0.0],\n  [0.0],\n  [0.0]],\n\n [[0.0],\n  [0.0],\n  [0.0]],\n\n [[0.0],\n  [0.0],\n  [0.0]]], shape=[3, 3, 1], strides=[3, 1, 1], layout=Cc (0x5), const ndim=3, axis: 2, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Unsqueeze::new(input.clone(), 2);

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_tensor, Backward, Gradient, Overwrite,
        Tensor, UnsqueezeBackward,
    };

    #[test]
    fn creation() {
        let node = UnsqueezeBackward::new(new_backward_input((4, 3), vec![0.; 12]), 0);

        assert_eq!(*node.gradient(), Tensor::from_elem((1, 4, 3), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((1, 4, 3), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let diff = new_backward_input((4, 3), vec![0.; 12]);
        let node = UnsqueezeBackward::new(diff.clone(), 0);

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
        let diff = new_backward_input((4, 3), vec![0.; 12]);
        let node = UnsqueezeBackward::new(diff.clone(), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((1, 4, 3), vec![1.; 12]);
        assert_almost_equals(&*node.gradient(), &new_tensor((1, 4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
    }

    #[test]
    fn backward_columns() {
        let diff = new_backward_input((4, 3), vec![0.; 12]);
        let node = UnsqueezeBackward::new(diff.clone(), 1);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((4, 1, 3), vec![1.; 12]);
        assert_almost_equals(&*node.gradient(), &new_tensor((4, 1, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
    }

    #[test]
    fn backward_depths() {
        let diff = new_backward_input((4, 3), vec![0.; 12]);
        let node = UnsqueezeBackward::new(diff.clone(), 2);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((4, 3, 1), vec![1.; 12]);
        assert_almost_equals(&*node.gradient(), &new_tensor((4, 3, 1), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
    }

    #[test]
    fn debug() {
        let diff = new_backward_input((4, 3), vec![0.; 12]);
        let node = UnsqueezeBackward::new(diff.clone(), 2);

        let output = "UnsqueezeBackward { gradient: Some([[[0.0],\n  [0.0],\n  [0.0]],\n\n [[0.0],\n  [0.0],\n  [0.0]],\n\n [[0.0],\n  [0.0],\n  [0.0]],\n\n [[0.0],\n  [0.0],\n  [0.0]]], shape=[4, 3, 1], strides=[3, 1, 1], layout=Cc (0x5), const ndim=3), axis: 2, overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let diff = new_backward_input((4, 3), vec![0.; 12]);
        let node = UnsqueezeBackward::new(diff.clone(), 2);

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }

    #[test]
    fn no_grad() {
        // UnsqueezeBackward
        let node = UnsqueezeBackward::new(new_backward_input((3, 3), vec![0.; 9]), 0);

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
