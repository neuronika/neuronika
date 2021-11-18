use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Cell, Data, Dropout,
    DropoutBackward, Forward, Gradient, Overwrite, Rc, Tensor,
};

mod forward {
    use super::{
        assert_almost_equals, new_input, new_tensor, Cell, Data, Dropout, Forward, Rc, Tensor,
    };

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Dropout::new(input, 0.5, Rc::new(Cell::new(true)));

        assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    #[should_panic(
        expected = "error: dropout probability has to be between 0 and 1, but got -0.5."
    )]
    fn creation_less_than_zero() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let _ = Dropout::new(input, -0.5, Rc::new(Cell::new(true)));
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Dropout::new(input, 0.5, Rc::new(Cell::new(true)));

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
    fn forward_p_one() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Dropout::new(input.clone(), 1., Rc::new(Cell::new(true)));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(&*node.data(), &new_tensor((3, 3), vec![0.; 9]));

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
        assert_almost_equals(&*node.data(), &new_tensor((3, 3), vec![0.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(&*node.data(), &new_tensor((3, 3), vec![0.; 9]));
    }

    #[test]
    fn forward_scaling() {
        let input = new_input((3, 3), vec![3.; 9]);
        let node = Dropout::new(input, 0.5, Rc::new(Cell::new(true)));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        node.data()
            .iter()
            .all(|el| *el <= f32::EPSILON || (el - 6.).abs() <= f32::EPSILON);
    }

    #[test]
    fn forward_p_zero() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Dropout::new(input.clone(), 0., Rc::new(Cell::new(true)));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
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
            &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(
            &*node.data(),
            &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
        );
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Cell, Dropout,
        DropoutBackward, Gradient, Overwrite, Rc, Tensor,
    };

    #[test]
    fn creation() {
        let node = DropoutBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            Rc::new(Dropout::new(
                new_input((3, 3), vec![1.; 9]),
                0.5,
                Rc::new(Cell::new(true)),
            )),
            0.5,
            Rc::new(Cell::new(true)),
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let input = new_backward_input((3, 3), vec![0.; 9]);
        let node = DropoutBackward::new(
            input.clone(),
            Rc::new(Dropout::new(
                new_input((3, 3), vec![1.; 9]),
                0.5,
                Rc::new(Cell::new(true)),
            )),
            0.5,
            Rc::new(Cell::new(true)),
        );

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
    fn backward_p_one() {
        let input = new_backward_input((3, 3), vec![0.; 9]);
        let node = DropoutBackward::new(
            input.clone(),
            Rc::new(Dropout::new(
                new_input((3, 3), vec![1.; 9]),
                1.,
                Rc::new(Cell::new(true)),
            )),
            1.,
            Rc::new(Cell::new(true)),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![0.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![0.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        input.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![0.; 9]));
    }

    #[test]
    fn backward_p_zero() {
        let input = new_backward_input((3, 3), vec![0.; 9]);
        let node = DropoutBackward::new(
            input.clone(),
            Rc::new(Dropout::new(
                new_input((3, 3), vec![1.; 9]),
                0.,
                Rc::new(Cell::new(true)),
            )),
            0.,
            Rc::new(Cell::new(true)),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
        assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![2.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        input.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![1.; 9]));
    }

    #[test]
    fn no_grad() {
        // DropoutBackward
        let node = DropoutBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            Rc::new(Dropout::new(
                new_input((3, 3), vec![0.; 9]),
                0.5,
                Rc::new(Cell::new(true)),
            )),
            0.5,
            Rc::new(Cell::new(true)),
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }
}
