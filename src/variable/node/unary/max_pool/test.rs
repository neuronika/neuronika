use super::{
    Backward, Cache, Data, Forward, Gradient, MaxPool,
    MaxPoolBackward, new_backward_input, new_input, new_tensor, Overwrite, Rc, Tensor,
};

mod forward {
    use super::{
        Cache, Data, Forward, MaxPool, new_input, new_tensor,
        Tensor,
    };

    #[test]
    fn creation() {
        let input = new_input((4, 4, 6, 5), vec![0.; 4 * 4 * 6 * 5]);
        let node = MaxPool::new(input, &[2, 3], &[2, 1]);

        assert_eq!(*node.data(), Tensor::zeros((4, 4, 3, 3)));
        assert_eq!(*node.data_mut(), Tensor::zeros((4, 4, 3, 3)));
        assert!(!node.was_computed());
    }

    #[test]
    #[should_panic(
    expected = "error: invalid pool shape [2, 3, 5] for 2d input."
    )]
    fn creation_dims_dont_match() {
        let input = new_input((4, 4, 6, 5), vec![0.; 4 * 4 * 6 * 5]);
        MaxPool::new(input, &[2, 3, 5], &[2, 1]);
    }

    #[test]
    #[should_panic(
    expected = "Pool shape [2, 6] doesn't fit in input shape [4, 4, 6, 5]."
    )]
    fn creation_pool_doesnt_fit() {
        let input = new_input((4, 4, 6, 5), vec![0.; 4 * 4 * 6 * 5]);
        MaxPool::new(input, &[2, 6], &[2, 1]);
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((4, 4, 6, 5), vec![0.; 4 * 4 * 6 * 5]);
        let node = MaxPool::new(input, &[2, 3], &[2, 1]);

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
        let input = new_input((2, 2, 2, 2), vec![1., 4., 2., 4., 5., 4., 12., 4., 7., 6., 8., 4., 11., 7., 7., 6.]);
        let node = MaxPool::new(input, &[2, 2], &[1, 1]);

        node.forward();
        assert_eq!(*node.data(), new_tensor((2, 2, 1, 1), vec![4., 12., 8., 11.]));
    }

    #[test]
    fn debug() {
        let input = new_input((2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let node = MaxPool::new(input, &[2], &[1]);

        let output = "MaxPool { data: [[[0.0],\n  [0.0]],\n\n [[0.0],\n  [0.0]]], shape=[2, 2, 1], strides=[2, 1, 1], layout=Cc (0x5), const ndim=3, pool_shape: [2], stride: [1], computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let input = new_input((2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let node = MaxPool::new(input, &[2], &[11]);

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use crate::Forward;

    use super::{
        Backward, Gradient, MaxPool, MaxPoolBackward, new_backward_input,
        new_input, new_tensor, Overwrite, Rc, Tensor,
    };

    #[test]
    fn creation() {
        let node = MaxPoolBackward::new(
            new_backward_input((4, 4, 6, 5), vec![0.; 4 * 4 * 6 * 5]),
            Rc::new(MaxPool::new(
                new_input((4, 4, 6, 5), vec![0.; 4 * 4 * 6 * 5]),
                &[2, 2],
                &[2, 2],
            )),
            &[2, 2],
            &[2, 2],
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((4, 4, 3, 2), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((4, 4, 3, 2), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let diff = new_backward_input((4, 4, 6, 5), vec![0.; 4 * 4 * 6 * 5]);
        let node = MaxPoolBackward::new(
            diff.clone(),
            Rc::new(MaxPool::new(
                new_input((4, 4, 6, 5), vec![0.; 4 * 4 * 6 * 5]),
                &[2, 2],
                &[2, 2],
            )),
            &[2, 2],
            &[2, 2],
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

        diff.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(!diff.can_overwrite());

        diff.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(!diff.can_overwrite());
    }

    #[test]
    fn backward() {
        let diff = new_backward_input((2, 2, 2, 2), vec![1.; 16]);
        let no_diff = Rc::new(MaxPool::new(
            new_input((2, 2, 2, 2), vec![1., 4., 2., 4., 5., 4., 12., 4., 7., 6., 8., 4., 11., 7., 7., 6.]),
            &[2, 2],
            &[1, 1],
        ));
        no_diff.forward();
        let node = MaxPoolBackward::new(
            diff.clone(),
            no_diff,
            &[2, 2],
            &[2, 2],
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((2, 2, 1, 1), vec![2., 3., 5., 4.]);
        assert_eq!(*node.gradient(), new_tensor((2, 2, 1, 1), vec![2., 3., 5., 4.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_eq!(*diff.gradient(), new_tensor((2, 2, 2, 2), vec![0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 5., 0., 4., 0., 0., 0.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_eq!(*diff.gradient(), new_tensor((2, 2, 2, 2), vec![0., 4., 0., 0., 0., 0., 6., 0., 0., 0., 10., 0., 8., 0., 0., 0.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_eq!(*diff.gradient(), new_tensor((2, 2, 2, 2), vec![0., 2., 0., 0., 0., 0., 3., 0., 0., 0., 5., 0., 4., 0., 0., 0.]));
    }

    #[test]
    fn debug() {
        let diff = new_backward_input((2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let no_diff = Rc::new(MaxPool::new(
            new_input((2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]),
            &[2],
            &[1],
        ));
        let node = MaxPoolBackward::new(
            diff,
            no_diff,
            &[2],
            &[1],
        );


        let output = "MaxPoolBackward { gradient: Some([[[0.0],\n  [0.0]],\n\n [[0.0],\n  [0.0]]], shape=[2, 2, 1], strides=[2, 1, 1], layout=Cc (0x5), const ndim=3), pool_shape: [2], stride: [1], overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let diff = new_backward_input((2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let no_diff = Rc::new(MaxPool::new(
            new_input((2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]),
            &[2],
            &[1],
        ));
        let node = MaxPoolBackward::new(
            diff,
            no_diff,
            &[2],
            &[1],
        );

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }

    #[test]
    fn no_grad() {
        let diff = new_backward_input((2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]);
        let no_diff = Rc::new(MaxPool::new(
            new_input((2, 2, 2), vec![1., 2., 3., 4., 5., 6., 7., 8.]),
            &[2],
            &[1],
        ));
        let node = MaxPoolBackward::new(
            diff,
            no_diff,
            &[2],
            &[1],
        );


        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}