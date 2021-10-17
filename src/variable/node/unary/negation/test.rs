use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, Negation, NegationBackward, Overwrite, Tensor,
};

mod forward {

    use super::{assert_almost_equals, new_input, new_tensor, Data, Forward, Negation, Tensor};

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Negation::new(input);

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Negation::new(input);

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
        let node = Negation::new(input.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
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
            &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![3., 2., 1., 0., -1., -2., -3., -4., -5.]),
        );
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_tensor, Backward, Gradient, NegationBackward,
        Overwrite, Tensor,
    };

    #[test]
    fn creation() {
        let node = NegationBackward::new(new_backward_input((3, 3), vec![0.; 9]));

        assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let input = new_backward_input((3, 3), vec![0.; 9]);
        let node = NegationBackward::new(input.clone());

        node.backward();
        assert!(node.can_overwrite());
        assert!(!input.can_overwrite());

        node.backward();
        assert!(node.can_overwrite());
        assert!(!input.can_overwrite());

        input.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input.can_overwrite());

        input.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(input.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(input.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!input.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!input.can_overwrite());

        input.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(!input.can_overwrite());

        input.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(!input.can_overwrite());
    }

    #[test]
    fn backward() {
        let input = new_backward_input((3, 3), vec![0.; 9]);
        let node = NegationBackward::new(input.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![-1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![-2.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        input.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![-1.; 9]));
    }

    #[test]
    fn no_grad() {
        // NegationBackward
        let node = NegationBackward::new(new_backward_input((3, 3), vec![0.; 9]));

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
