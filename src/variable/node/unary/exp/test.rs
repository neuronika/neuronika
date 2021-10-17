use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Exp,
    ExpBackward, Forward, Gradient, Overwrite, Rc, Tensor,
};

mod forward {
    use super::{assert_almost_equals, new_input, new_tensor, Data, Exp, Forward, Tensor};

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Exp::new(input);

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Exp::new(input);

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
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Exp::new(input.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![
                    1.83156e-02,
                    4.97871e-02,
                    1.35335e-01,
                    3.67879e-01,
                    1.00000e+00,
                    2.71828e+00,
                    7.38906e+00,
                    2.00855e+01,
                    5.45981e+01,
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
                    1.83156e-02,
                    4.97871e-02,
                    1.35335e-01,
                    3.67879e-01,
                    1.00000e+00,
                    2.71828e+00,
                    7.38906e+00,
                    2.00855e+01,
                    5.45981e+01,
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
                    4.97871e-02,
                    1.35335e-01,
                    3.67879e-01,
                    1.00000e+00,
                    2.71828e+00,
                    7.38906e+00,
                    2.00855e+01,
                    5.45981e+01,
                    1.48413e+02,
                ],
            ),
        );
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Exp,
        ExpBackward, Forward, Gradient, Overwrite, Rc, Tensor,
    };

    #[test]
    fn creation() {
        let node = ExpBackward::new(
            new_backward_input(3, vec![0.; 3]),
            Rc::new(Exp::new(new_input(3, vec![1., 2., 3.]))),
        );

        assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let node = ExpBackward::new(
            diff.clone(),
            Rc::new(Exp::new(new_input(3, vec![1., 2., 3.]))),
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

    #[allow(clippy::approx_constant)]
    #[test]
    fn backward() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let not_diff = Rc::new(Exp::new(new_input(3, vec![1., 2., 3.])));
        not_diff.forward();
        let node = ExpBackward::new(diff.clone(), not_diff);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
        assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(3, vec![2.7183, 7.3891, 20.0855]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(3, vec![5.4366, 14.7782, 40.171]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(3, vec![2.7183, 7.3891, 20.0855]),
        );
    }

    #[test]
    fn no_grad() {
        // ExpBackward
        let node = ExpBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
