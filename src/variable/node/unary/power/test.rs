use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, Overwrite, Power, PowerBackward, Tensor,
};

mod forward {
    use super::{assert_almost_equals, new_input, new_tensor, Data, Forward, Power, Tensor};

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Power::new(input, 2);

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Power::new(input, 2);

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
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Power::new(input.clone(), 3);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![1., 8., 27., 64., 125., 216., 343., 512., 729.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        {
            let mut data = input.data_mut();
            *data = &*data + &Tensor::from_elem(1, 1.);
        }
        assert_almost_equals(
            &*input.data(),
            &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        );

        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![1., 8., 27., 64., 125., 216., 343., 512., 729.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor(
                (3, 3),
                vec![8., 27., 64., 125., 216., 343., 512., 729., 1_000.],
            ),
        );
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Gradient,
        Overwrite, PowerBackward, Tensor,
    };

    #[test]
    fn creation() {
        let node = PowerBackward::new(
            new_backward_input(3, vec![0.; 3]),
            new_input(3, vec![1., 2., 3.]),
            3,
        );

        assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let node = PowerBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]), 3);

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
    fn backward() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let node = PowerBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]), 3);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
        assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3., 12., 27.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![6., 24., 54.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3., 12., 27.]));
    }

    #[test]
    fn backward_negative_exp() {
        let diff = new_backward_input(3, vec![0.; 3]);
        let node = PowerBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]), -3);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
        assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(3, vec![-3., -0.1875, -0.037037]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(3, vec![-6., -0.375, -0.074075]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor(3, vec![-3., -0.1875, -0.037037]),
        );
    }

    #[test]
    fn no_grad() {
        // PowerBackward
        let node = PowerBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            new_input((3, 3), vec![0.; 9]),
            2,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
