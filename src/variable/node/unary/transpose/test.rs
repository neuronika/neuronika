use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, Overwrite, Tensor, Transpose, TransposeBackward,
};
mod forward {

    use super::{assert_almost_equals, new_input, new_tensor, Data, Forward, Tensor, Transpose};

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Transpose::new(input);

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Transpose::new(input);

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
        let node = Transpose::new(input.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]),
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
            &new_tensor((3, 3), vec![1., 4., 7., 2., 5., 8., 3., 6., 9.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![2., 5., 8., 3., 6., 9., 4., 7., 10.]),
        );
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_tensor, Backward, Gradient, Overwrite,
        Tensor, TransposeBackward,
    };

    #[test]
    fn creation() {
        let node = TransposeBackward::new(new_backward_input((4, 3), vec![0.; 12]));

        assert_eq!(*node.gradient(), Tensor::from_elem((3, 4), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 4), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let diff = new_backward_input((4, 3), vec![0.; 12]);
        let node = TransposeBackward::new(diff.clone());

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
        let diff = new_backward_input((4, 3), vec![0.; 12]);
        let node = TransposeBackward::new(diff.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 4), vec![1.; 12]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 4), vec![1.; 12]));

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
    fn no_grad() {
        // TransposeBackward
        let node = TransposeBackward::new(new_backward_input((3, 3), vec![0.; 9]));

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
