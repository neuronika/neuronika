use super::{
    assert_almost_equals, new_tensor, Addition, AdditionBackward, Backward, Cache, Forward, Tensor,
};

mod forward {
    use super::{assert_almost_equals, new_tensor, Addition, Cache, Forward, Tensor};

    #[test]
    fn creation() {
        let left_data = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right_data = new_tensor((3, 3), vec![1.; 9]);
        let data = new_tensor((3, 3), vec![0.0; 9]);
        let op = Addition::new(left_data, right_data, data);

        assert!(!op.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let left_data = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right_data = new_tensor((3, 3), vec![1.; 9]);
        let data = new_tensor((3, 3), vec![0.0; 9]);
        let op = Addition::new(left_data, right_data, data);

        op.forward();
        assert!(op.was_computed());

        op.forward();
        assert!(op.was_computed());

        op.reset_computation();
        assert!(!op.was_computed());

        op.reset_computation();
        assert!(!op.was_computed());
    }

    #[test]
    fn forward() {
        let left_data = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right_data = new_tensor((3, 3), vec![1.; 9]);
        let data = new_tensor((3, 3), vec![0.0; 9]);
        let op = Addition::new(left_data.clone(), right_data, data);

        let target = new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        op.forward();
        assert_almost_equals(&*op.data.borrow(), &*target.borrow());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        {
            let mut data = left_data.borrow_mut();
            *data = &*data + &Tensor::from_elem(1, 10.);
        }
        assert_almost_equals(
            &*left_data.borrow(),
            &*new_tensor((3, 3), vec![11., 12., 13., 14., 15., 16., 17., 18., 19.]).borrow(),
        );

        op.forward();
        assert_almost_equals(
            &*op.data.borrow(),
            &*new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]).borrow(),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        op.reset_computation();
        op.forward();
        assert_almost_equals(
            &*op.data.borrow(),
            &*new_tensor((3, 3), vec![12., 13., 14., 15., 16., 17., 18., 19., 20.]).borrow(),
        );
    }

    #[test]
    fn left_broadcast_forward() {
        let left_data = new_tensor((1, 3), vec![1., 2., 3.]);
        let right_data = new_tensor((2, 2, 3), vec![1.; 12]);
        let data = new_tensor((2, 2, 3), vec![0.0; 12]);
        let op = Addition::new(left_data, right_data, data);

        op.forward();
        assert_almost_equals(
            &*op.data.borrow(),
            &*new_tensor(
                (2, 2, 3),
                vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.],
            )
            .borrow(),
        );
    }

    #[test]
    fn right_broadcast_forward() {
        let left_data = new_tensor((2, 2, 3), vec![1.; 12]);
        let right_data = new_tensor((1, 3), vec![1., 2., 3.]);
        let data = new_tensor((2, 2, 3), vec![0.0; 12]);
        let op = Addition::new(left_data, right_data, data);

        op.forward();
        assert_almost_equals(
            &*op.data.borrow(),
            &new_tensor(
                (2, 2, 3),
                vec![2., 3., 4., 2., 3., 4., 2., 3., 4., 2., 3., 4.],
            )
            .borrow(),
        );
    }
}

mod backward {
    use super::{assert_almost_equals, new_tensor, AdditionBackward, Backward, Tensor};

    #[test]
    fn creation() {
        let node = AdditionBackward::new(
            new_opt_tensor((3, 3), vec![0.; 9]),
            new_opt_tensor((3, 3), vec![0.; 9]),
            new_opt_tensor((3, 3), vec![0.; 9]),
            (3, 3).into_dimension(),
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(node.can_overwrite());
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
}
