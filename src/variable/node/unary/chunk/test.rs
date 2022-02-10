use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Cache, Chunk,
    ChunkBackward, Data, Forward, Gradient, Overwrite, Tensor,
};

mod forward {

    use super::{assert_almost_equals, new_input, new_tensor, Cache, Chunk, Data, Forward, Tensor};

    #[test]
    fn creation() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Chunk::new(input, Tensor::zeros((1, 3)), 0);

        assert_eq!(*node.data(), Tensor::from_elem((1, 3), 0.));
        assert_eq!(*node.data_mut(), Tensor::from_elem((1, 3), 0.));
        assert!(!node.was_computed());
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
        let node = Chunk::new(input, Tensor::zeros((1, 3)), 0);

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
        let node = Chunk::new(input.clone(), Tensor::zeros((1, 3)), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.forward();
        assert_almost_equals(&*node.data(), &new_tensor((1, 3), vec![-4., -3., -2.]));

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
        assert_almost_equals(&*node.data(), &new_tensor((1, 3), vec![-4., -3., -2.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.reset_computation();
        node.forward();
        assert_almost_equals(&*node.data(), &new_tensor((1, 3), vec![-3., -2., -1.]));
    }

    #[test]
    fn debug() {
        let input = new_input(3, vec![0.; 3]);
        let node = Chunk::new(input, ndarray::arr1(&[0.]), 1);

        let output = "Chunk { data: [0.0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1, chunk_no: 1, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let input = new_input(3, vec![0.; 3]);
        let node = Chunk::new(input, ndarray::arr1(&[0.]), 1);

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use super::{
        assert_almost_equals, new_backward_input, new_tensor, Backward, ChunkBackward, Gradient,
        Overwrite, Tensor,
    };

    #[test]
    fn creation() {
        let node = ChunkBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            Tensor::zeros((1, 3)),
            0,
        );

        assert_eq!(*node.gradient(), Tensor::from_elem((1, 3), 0.));
        assert_eq!(*node.gradient_mut(), Tensor::from_elem((1, 3), 0.));
        assert!(node.can_overwrite());
    }

    #[test]
    fn computation_state_transition() {
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = ChunkBackward::new(diff.clone(), Tensor::zeros((1, 3)), 0);

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
        let diff = new_backward_input((3, 3), vec![0.; 9]);
        let node = ChunkBackward::new(diff.clone(), Tensor::zeros((1, 3)), 0);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *node.gradient_mut() = new_tensor((1, 3), vec![1.; 3]);
        assert_almost_equals(&*node.gradient(), &new_tensor((1, 3), vec![1.; 3]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor((3, 3), vec![1., 1., 1., 0., 0., 0., 0., 0., 0.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor((3, 3), vec![2., 2., 2., 0., 0., 0., 0., 0., 0.]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        diff.set_overwrite(true);
        node.backward();
        assert_almost_equals(
            &*diff.gradient(),
            &new_tensor((3, 3), vec![1., 1., 1., 0., 0., 0., 0., 0., 0.]),
        );
    }

    #[test]
    fn no_grad() {
        let node = ChunkBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            Tensor::zeros((1, 3)),
            0,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
    }

    #[test]
    fn debug() {
        let node = ChunkBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            Tensor::zeros((1, 3)),
            0,
        );

        let output = "ChunkBackward { gradient: Some([[0.0, 0.0, 0.0]], shape=[1, 3], strides=[3, 1], layout=CFcf (0xf), const ndim=2), chunk_no: 0, overwrite: true }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let node = ChunkBackward::new(
            new_backward_input((3, 3), vec![0.; 9]),
            Tensor::zeros((1, 3)),
            0,
        );

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }
}
