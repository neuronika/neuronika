use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Addition, AdditionBackward,
    AdditionBackwardUnary, Backward, Cache, Data, Forward, Gradient, Overwrite, Tensor,
};

mod forward {
    use super::{
        assert_almost_equals, new_input, new_tensor, Addition, Cache, Data, Forward, Tensor,
    };

    #[test]
    fn creation() {
        let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right = new_input((3, 3), vec![1.; 9]);
        let node = Addition::new(left, right);

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right = new_input((3, 3), vec![1.; 9]);
        let node = Addition::new(left, right);

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
        let right = new_input((3, 3), vec![1.; 9]);
        let node = Addition::new(left.clone(), right);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        {
            let mut data = left.data_mut();
            *data = &*data + &Tensor::from_elem(1, 10.);
        }
        assert_almost_equals(
            &*left.data(),
            &new_tensor((3, 3), vec![11., 12., 13., 14., 15., 16., 17., 18., 19.]),
        );

        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![12., 13., 14., 15., 16., 17., 18., 19., 20.]),
        );
    }

    #[test]
    fn left_broadcast_forward() {
        let left = new_input((1, 3), vec![1., 2., 3.]);
        let right = new_input((2, 2, 3), vec![1.; 12]);
        let node = Addition::new(left, right);

        assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (2, 2, 3),
                vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.],
            ),
        );
    }

    #[test]
    fn right_broadcast_forward() {
        let left = new_input((2, 2, 3), vec![1.; 12]);
        let right = new_input((1, 3), vec![1., 2., 3.]);
        let node = Addition::new(left, right);

        assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (2, 2, 3),
                vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.],
            ),
        );
    }

    #[test]
    fn debug() {
        let left = new_input(1, vec![0.]);
        let right = new_input(1, vec![0.]);
        let node = Addition::new(left, right);

        let output = "Addition { data: [0.0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let left = new_input(1, vec![0.]);
        let right = new_input(1, vec![0.]);
        let node = Addition::new(left, right);

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, AdditionBackward,
        AdditionBackwardUnary, Backward, Gradient, Overwrite, Tensor,
    };

    #[test]
    fn creation() {
        let node = AdditionBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let lhs = new_backward_input((3, 3), vec![0.; 9]);
        let rhs = new_backward_input((3, 3), vec![0.; 9]);
        let node = AdditionBackward::new(lhs.clone(), rhs.clone());

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
        let rhs = new_backward_input((3, 3), vec![0.; 9]);
        let node = AdditionBackward::new(lhs.clone(), rhs.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
    }

    #[test]
    fn backward_broadcast_left() {
        let lhs = new_backward_input(3, vec![0.; 3]);
        let rhs = new_backward_input((3, 3), vec![0.; 9]);
        let node = AdditionBackward::new(lhs.clone(), rhs.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![3.; 3]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![6.; 3]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![3.; 3]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
    }

    #[test]
    fn backward_broadcast_right() {
        let lhs = new_backward_input((3, 3), vec![0.; 9]);
        let rhs = new_backward_input((1, 3), vec![0.; 3]);
        let node = AdditionBackward::new(lhs.clone(), rhs.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((1, 3), vec![3.; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![2.; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((1, 3), vec![6.; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((1, 3), vec![3.; 3]));
    }

    #[test]
    fn backward_unary() {
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let not_diff = new_input((3, 3), vec![0.; 9]);
        let node = AdditionBackwardUnary::new(diff.clone(), not_diff);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![2.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![1.; 9]));
    }

    #[test]
    fn backward_unary_broadcast() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let not_diff = new_input((3, 3), vec![0.; 9]);
        let node = AdditionBackwardUnary::new(diff.clone(), not_diff);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3.; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![6.; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3.; 3]));
    }

    #[test]
    fn no_grad() {
        let node = AdditionBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

        let node = AdditionBackwardUnary::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }

    #[test]
    fn debug() {
        {
            let node = AdditionBackward::new(
                new_backward_input(1, vec![0.]),
                new_backward_input(1, vec![0.]),
            );

            let output = "AdditionBackward { gradient: Some([0.0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1), overwrite: true }";
            assert_eq!(output, format!("{:?}", node));
        }
    }

    #[test]
    fn debug_unary() {
        let node =
            AdditionBackwardUnary::new(new_backward_input(1, vec![0.]), new_input(1, vec![0.]));

        let output = "AdditionBackwardUnary { gradient: Some([0.0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1), overwrite: true }";
        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        {
            let node = AdditionBackward::new(
                new_backward_input(1, vec![0.]),
                new_backward_input(1, vec![0.]),
            );
            assert_eq!(format!("{}", node.gradient()), format!("{}", node));
        }
    }

    #[test]
    fn display_unary() {
        let node =
            AdditionBackwardUnary::new(new_backward_input(1, vec![0.]), new_input(1, vec![0.]));
        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }
}
