use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, Overwrite, Sum, SumBackward, Tensor,
};
use ndarray::arr0;

mod forward {
    use super::{arr0, assert_almost_equals, new_input, new_tensor, Data, Forward, Sum, Tensor};

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Sum::new(input);

        assert_eq!(*node.data(), arr0(0.));
        assert_eq!(*node.data_mut(), arr0(0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Sum::new(input);

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
        let node = Sum::new(input.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(&*node.data(), &arr0(45.));

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
        assert_almost_equals(&*node.data(), &arr0(45.));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(&*node.data(), &arr0(54.));
    }

    #[test]
    fn debug() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Sum::new(input.clone());

        let output = "Sum { data: 0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let node = Sum::new(input.clone());

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use super::{
        arr0, assert_almost_equals, new_backward_input, new_tensor, Backward, Gradient, Overwrite,
        SumBackward,
    };

    #[test]
    fn creation() {
        let node = SumBackward::new(new_backward_input((10, 10), vec![0.; 100]));

        assert_eq!(*node.gradient(), arr0(0.));
        assert_eq!(*node.gradient_mut(), arr0(0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let diff = new_backward_input((10, 10), vec![0.; 100]);
        let node = SumBackward::new(diff.clone());

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
    #[allow(clippy::approx_constant)]
    fn backward() {
        let diff = new_backward_input((10, 10), vec![0.; 100]);
        let node = SumBackward::new(diff.clone());

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = arr0(1.);
        assert_almost_equals(&*node.gradient(), &arr0(1.));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![1.; 100]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![2.; 100]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![1.; 100]));
    }

    #[test]
    fn debug() {
        let diff = new_backward_input((10, 10), vec![0.; 100]);
        let node = SumBackward::new(diff.clone());

        let output = "SumBackward { gradient: Some(0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0), overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let diff = new_backward_input((10, 10), vec![0.; 100]);
        let node = SumBackward::new(diff.clone());

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }

    #[test]
    fn no_grad() {
        // SumBackward
        let node = SumBackward::new(new_backward_input((3, 3), vec![0.; 9]));

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), arr0(0.));
    }
}
