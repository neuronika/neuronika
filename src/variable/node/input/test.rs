use super::{Data, Gradient, Input, InputBackward, Overwrite, Tensor};
use std::cell::{Cell, RefCell};

mod forward {

    use super::{Cell, Data, Input, RefCell, Tensor};

    #[test]
    fn creation() {
        let input = Input {
            data: RefCell::new(Tensor::zeros((3, 3))),
            computed: Cell::new(false),
        };
        assert_eq!(*input.data(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*input.data_mut(), Tensor::from_elem((3, 3), 0.));
    }

    #[test]
    fn debug() {
        let node = Input {
            data: RefCell::new(Tensor::zeros(1)),
            computed: Cell::new(false),
        };
        let output =
            "Input { data: [0.0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1, computed: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let node = Input {
            data: RefCell::new(Tensor::zeros(1)),
            computed: Cell::new(false),
        };

        assert_eq!(format!("{}", node.data()), format!("{}", node));
    }
}

mod backward {
    use super::{Cell, Gradient, InputBackward, Overwrite, RefCell, Tensor};

    #[test]
    fn creation() {
        let input = InputBackward {
            gradient: RefCell::new(Some(Tensor::zeros((3, 3)))),
            overwrite: Cell::new(true),
        };
        assert!(input.can_overwrite());
        assert_eq!(*input.gradient(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*input.gradient_mut(), Tensor::from_elem((3, 3), 0.));
    }

    #[test]
    fn computation_state_transition() {
        let input = InputBackward {
            gradient: RefCell::new(Some(Tensor::zeros((3, 3)))),
            overwrite: Cell::new(true),
        };

        assert!(input.can_overwrite());
        input.set_overwrite(false);
        assert!(!input.can_overwrite());
        input.set_overwrite(true);
        assert!(input.can_overwrite());
    }

    #[test]
    fn zero_grad() {
        let input = InputBackward {
            gradient: RefCell::new(Some(Tensor::ones((3, 3)))),
            overwrite: Cell::new(true),
        };

        input.zero_grad();
        assert_eq!(*input.gradient(), Tensor::zeros((3, 3)));
    }

    #[test]
    fn debug() {
        let node = InputBackward {
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            overwrite: Cell::new(false),
        };
        let output =
            "InputBackward { gradient: Some([0.0], shape=[1], strides=[1], layout=CFcf (0xf), const ndim=1), overwrite: false }";

        assert_eq!(output, format!("{:?}", node));
    }

    #[test]
    fn display() {
        let node = InputBackward {
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            overwrite: Cell::new(false),
        };

        assert_eq!(format!("{}", node.gradient()), format!("{}", node));
    }
}
