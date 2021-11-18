use super::{
    conv_out_shape, new_backward_input, new_input, Backward, Convolution, ConvolutionBackward,
    Forward, Gradient, GroupedConvolution, GroupedConvolutionBackward, NData, Overwrite, Tensor,
    Zero,
};

mod forward {
    use super::{conv_out_shape, new_input, Convolution, Forward, NData, Tensor, Zero};

    #[test]
    fn creation() {
        let input = new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
        let kernel = new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);
        let node = Convolution::new(input, kernel, &[1, 1], &[1, 1], &[0, 0], Zero);

        let outshape: ndarray::Ix4 =
            conv_out_shape(&[4, 4, 6, 6], &[4, 4, 2, 2], &[0, 0], &[1, 1], &[1, 1]);
        assert_eq!(*node.data(), Tensor::from_elem(outshape, 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
        let kernel = new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);
        let node = Convolution::new(input, kernel, &[1, 1], &[1, 1], &[0, 0], Zero);

        node.forward();
        assert!(node.was_computed());

        node.forward();
        assert!(node.was_computed());

        node.reset_computation();
        assert!(!node.was_computed());

        node.reset_computation();
        assert!(!node.was_computed());
    }
}

mod forward_grouped {
    use super::*;

    #[test]
    fn creation() {
        let input = new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
        let kernel = new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);
        let node = GroupedConvolution::new(input, kernel, &[1, 1], &[1, 1], &[0, 0], Zero, 2);

        let outshape: ndarray::Ix4 =
            conv_out_shape(&[4, 4, 6, 6], &[4, 4, 2, 2], &[0, 0], &[1, 1], &[1, 1]);
        assert_eq!(*node.data(), Tensor::from_elem(outshape, 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
        let kernel = new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);
        let node = GroupedConvolution::new(input, kernel, &[1, 1], &[1, 1], &[0, 0], Zero, 2);

        node.forward();
        assert!(node.was_computed());

        node.forward();
        assert!(node.was_computed());

        node.reset_computation();
        assert!(!node.was_computed());

        node.reset_computation();
        assert!(!node.was_computed());
    }
}

mod backward {
    use super::{
        conv_out_shape, new_backward_input, new_input, Backward, ConvolutionBackward, Gradient,
        Overwrite, Tensor, Zero,
    };

    #[test]
    fn creation() {
        let node = ConvolutionBackward::new(
            new_backward_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
            new_backward_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
            new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
            new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
            &[1, 1],
            &[1, 1],
            &[0, 0],
            Zero,
        );

        let outshape: ndarray::Ix4 =
            conv_out_shape(&[4, 4, 6, 6], &[4, 4, 2, 2], &[0, 0], &[1, 1], &[1, 1]);

        assert_eq!(*node.gradient(), Tensor::from_elem(outshape, 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem(outshape, 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let input_grad = new_backward_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
        let kernel_grad = new_backward_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);

        let node = ConvolutionBackward::new(
            input_grad.clone(),
            kernel_grad.clone(),
            new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
            new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
            &[1, 1],
            &[1, 1],
            &[0, 0],
            Zero,
        );

        node.backward();
        assert!(node.can_overwrite());
        assert!(!input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        node.backward();
        assert!(node.can_overwrite());
        assert!(!input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        input_grad.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        input_grad.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        kernel_grad.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(kernel_grad.can_overwrite());

        kernel_grad.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(kernel_grad.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(kernel_grad.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(kernel_grad.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());
    }
}

mod backward_grouped {
    use super::{
        conv_out_shape, new_backward_input, new_input, Backward, Gradient,
        GroupedConvolutionBackward, Overwrite, Tensor, Zero,
    };

    #[test]
    fn creation() {
        let node = GroupedConvolutionBackward::new(
            new_backward_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
            new_backward_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
            new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
            new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
            &[1, 1],
            &[1, 1],
            &[0, 0],
            Zero,
            2,
        );

        let outshape: ndarray::Ix4 =
            conv_out_shape(&[4, 4, 6, 6], &[4, 4, 2, 2], &[0, 0], &[1, 1], &[1, 1]);

        assert_eq!(*node.gradient(), Tensor::from_elem(outshape, 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem(outshape, 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let input_grad = new_backward_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
        let kernel_grad = new_backward_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);

        let node = GroupedConvolutionBackward::new(
            input_grad.clone(),
            kernel_grad.clone(),
            new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
            new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
            &[1, 1],
            &[1, 1],
            &[0, 0],
            Zero,
            2,
        );

        node.backward();
        assert!(node.can_overwrite());
        assert!(!input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        node.backward();
        assert!(node.can_overwrite());
        assert!(!input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        input_grad.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        input_grad.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        kernel_grad.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(kernel_grad.can_overwrite());

        kernel_grad.set_overwrite(true);
        assert!(node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(kernel_grad.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(kernel_grad.can_overwrite());

        node.set_overwrite(false);
        assert!(!node.can_overwrite());
        assert!(input_grad.can_overwrite());
        assert!(kernel_grad.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());

        node.backward();
        assert!(!node.can_overwrite());
        assert!(!input_grad.can_overwrite());
        assert!(!kernel_grad.can_overwrite());
    }
}
