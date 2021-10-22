use super::{Data, Forward, Gradient, Input, InputBackward, Overwrite, Tensor};
use std::cell::{Cell, RefCell};

mod forward {

    use super::{Cell, Data, Forward, Input, RefCell, Tensor};

    #[test]
    fn creation() {
        let input = Input {
            data: RefCell::new(Tensor::zeros((3, 3))),
            computed: Cell::new(false),
        };
        assert!(!input.was_computed());
        assert_eq!(*input.data(), Tensor::from_elem((3, 3), 0.));
        assert_eq!(*input.data(), Tensor::from_elem((3, 3), 0.));
    }

    #[test]
    fn computation_was_computed_transition() {
        let input = Input {
            data: RefCell::new(Tensor::zeros((3, 3))),
            computed: Cell::new(false),
        };

        input.forward();
        assert!(input.was_computed());

        input.forward();
        assert!(input.was_computed());

        input.reset_computation();
        assert!(!input.was_computed());

        input.reset_computation();
        assert!(!input.was_computed());
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
}
