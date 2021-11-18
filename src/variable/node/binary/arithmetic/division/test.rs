use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Division,
    DivisionBackward, DivisionBackwardLeft, DivisionBackwardRight, Forward, Gradient, Overwrite,
    Tensor,
};

mod forward {
    use super::{assert_almost_equals, new_input, new_tensor, Data, Division, Forward, Tensor};

    #[test]
    fn creation() {
        let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right = new_input((3, 3), vec![2.; 9]);
        let node = Division::new(left, right);

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let left = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let right = new_input((3, 3), vec![2.; 9]);
        let node = Division::new(left, right);

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
        let right = new_input((3, 3), vec![2.; 9]);
        let node = Division::new(left, right.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *right.data_mut() = new_tensor((3, 3), vec![-2.; 9]);
        assert_almost_equals(&*right.data(), &new_tensor((3, 3), vec![-2.; 9]));

        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![-0.5, -1., -1.5, -2., -2.5, -3., -3.5, -4., -4.5],
            ),
        );
    }

    #[test]
    fn left_broadcast_forward() {
        let left = new_input((1, 3), vec![1., 2., 3.]);
        let right = new_input((2, 2, 3), vec![2.; 12]);
        let node = Division::new(left, right);

        assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (2, 2, 3),
                vec![0.5, 1., 1.5, 0.5, 1., 1.5, 0.5, 1., 1.5, 0.5, 1., 1.5],
            ),
        );
    }

    #[test]
    fn right_broadcast_forward() {
        let left = new_input((2, 2, 3), vec![2.; 12]);
        let right = new_input((1, 3), vec![1., 2., 3.]);
        let node = Division::new(left, right);

        assert_eq!(*node.data(), Tensor::from_elem((2, 2, 3), 0.));
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (2, 2, 3),
                vec![
                    2., 1., 0.6667, 2., 1., 0.6667, 2., 1., 0.6667, 2., 1., 0.6667,
                ],
            ),
        );
    }
}
mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward,
        DivisionBackward, DivisionBackwardLeft, DivisionBackwardRight, Gradient, Overwrite, Tensor,
    };

    #[test]
    fn creation() {
        let node = DivisionBackward::new(
            new_input((3, 3), vec![3.; 9]),
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![5.; 9]),
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
        let node = DivisionBackward::new(
            new_input((3, 3), vec![3.; 9]),
            lhs.clone(),
            new_input((3, 3), vec![5.; 9]),
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
        let lhs = new_backward_input((3, 3), vec![0.; 9]);
        let rhs = new_backward_input((3, 3), vec![0.; 9]);
        let node = DivisionBackward::new(
            new_input((3, 3), vec![3.; 9]),
            lhs.clone(),
            new_input((3, 3), vec![5.; 9]),
            rhs.clone(),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.4; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.24; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));
    }

    #[test]
    fn backward_broadcast_left() {
        let lhs = new_backward_input(3, vec![0.; 3]);
        let rhs = new_backward_input((3, 3), vec![0.; 9]);
        let node = DivisionBackward::new(
            new_input(3, vec![3.; 3]),
            lhs.clone(),
            new_input((3, 3), vec![5.; 9]),
            rhs.clone(),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![0.6; 3]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![1.2; 3]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.24; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor(3, vec![0.6; 3]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));
    }

    #[test]
    fn backward_broadcast_right() {
        let lhs = new_backward_input((3, 3), vec![0.; 9]);
        let rhs = new_backward_input((1, 3), vec![0.; 3]);
        let node = DivisionBackward::new(
            new_input((3, 3), vec![3.; 9]),
            lhs.clone(),
            new_input((1, 3), vec![5.; 3]),
            rhs.clone(),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((1, 3), vec![-0.36; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.4; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((1, 3), vec![-0.72; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        lhs.set_overwrite(true);
        rhs.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
        assert_almost_equals(&*rhs.gradient(), &new_tensor((1, 3), vec![-0.36; 3]));
    }

    #[test]
    fn backward_left() {
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = DivisionBackwardLeft::new(diff.clone(), new_input((3, 3), vec![5.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![0.2; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![0.4; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![0.2; 9]));
    }

    #[test]
    fn backward_left_broadcast() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let node = DivisionBackwardLeft::new(diff.clone(), new_input((3, 3), vec![5.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0.6; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![1.2; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![0.6; 3]));
    }

    #[test]
    fn backward_right() {
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = DivisionBackwardRight::new(
            new_input((3, 3), vec![3.; 9]),
            new_input((3, 3), vec![5.; 9]),
            diff.clone(),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-0.24; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((3, 3), vec![-0.12; 9]));
    }

    #[test]
    fn backward_right_broadcast() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let node = DivisionBackwardRight::new(
            new_input((3, 3), vec![3.; 9]),
            new_input((3, 3), vec![5.; 9]),
            diff.clone(),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-0.36; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-0.72; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![-0.36; 3]));
    }
}

#[test]
fn no_grad() {
    // DivisionBackward
    let node = DivisionBackward::new(
        new_input((3, 3), vec![0.; 9]),
        new_backward_input((3, 3), vec![0.; 9]),
        new_input((3, 3), vec![0.; 9]),
        new_backward_input((3, 3), vec![0.; 9]),
    );

    node.no_grad();
    assert!(node.gradient.borrow().is_none());

    node.with_grad();
    assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

    // DivisionBackwardLeft
    let node = DivisionBackwardLeft::new(
        new_backward_input((3, 3), vec![0.; 9]),
        new_input((3, 3), vec![0.; 9]),
    );

    node.no_grad();
    assert!(node.gradient.borrow().is_none());

    node.with_grad();
    assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));

    // DivisionBackwardRight
    let node = DivisionBackwardRight::new(
        new_input((3, 3), vec![0.; 9]),
        new_input((3, 3), vec![0.; 9]),
        new_backward_input((3, 3), vec![0.; 9]),
    );

    node.no_grad();
    assert!(node.gradient.borrow().is_none());

    node.with_grad();
    assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
}
