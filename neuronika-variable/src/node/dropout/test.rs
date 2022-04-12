use std::{cell::Cell, error::Error, rc::Rc};

use ndarray::Array;

use crate::utils::{are_similar, new_shared};

mod forward {
    use super::super::{Dropout, Forward};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let operand_data = Array::zeros((3, 3));
        let data = Array::ones((3, 3));
        let noise = Array::from_elem((3, 3), 2.);
        let status = Rc::new(Cell::new(true));
        let op = Dropout::new(
            new_shared(operand_data.clone()),
            new_shared(data.clone()),
            0.,
            new_shared(noise.clone()),
            status.clone(),
        );

        assert_eq!(op.status.get(), status.get());
        assert_eq!(op.p, 0.);
        are_similar(op.operand_data.borrow(), &operand_data)?;
        are_similar(op.data.borrow(), &data)?;
        are_similar(op.noise.borrow(), &noise)
    }

    #[test]
    #[should_panic]
    fn too_low_probability() {
        let _ = Dropout::new(
            new_shared(Array::zeros((3, 3))),
            new_shared(Array::zeros((3, 3))),
            -0.5,
            new_shared(Array::zeros((3, 3))),
            Rc::new(Cell::new(true)),
        );
    }

    #[test]
    #[should_panic]
    fn too_high_probability() {
        let _ = Dropout::new(
            new_shared(Array::zeros((3, 3))),
            new_shared(Array::zeros((3, 3))),
            1.5,
            new_shared(Array::zeros((3, 3))),
            Rc::new(Cell::new(true)),
        );
    }

    #[test]
    fn one_probability() -> Result<(), Box<dyn Error>> {
        let op = Dropout::new(
            new_shared(Array::linspace(1., 9., 9).into_shape((3, 3))?),
            new_shared(Array::zeros((3, 3))),
            1.,
            new_shared(Array::zeros((3, 3))),
            Rc::new(Cell::new(true)),
        );

        op.forward();
        are_similar(op.data.borrow(), &Array::zeros((3, 3)))
    }

    #[test]
    fn zero_probability() -> Result<(), Box<dyn Error>> {
        let op = Dropout::new(
            new_shared(Array::linspace(1., 9., 9).into_shape((3, 3))?),
            new_shared(Array::zeros((3, 3))),
            0.,
            new_shared(Array::zeros((3, 3))),
            Rc::new(Cell::new(true)),
        );

        op.forward();
        are_similar(
            op.data.borrow(),
            &Array::linspace(1., 9., 9).into_shape((3, 3))?,
        )
    }

    #[test]
    fn base_case() {
        let op = Dropout::new(
            new_shared(Array::linspace(1., 9., 9).into_shape((3, 3)).unwrap()),
            new_shared(Array::zeros((3, 3))),
            0.5,
            new_shared(Array::zeros((3, 3))),
            Rc::new(Cell::new(true)),
        );

        op.forward();
        assert!(op
            .data
            .borrow()
            .iter()
            .zip(&Array::linspace(2.0_f32, 18., 9).into_shape((3, 3)).unwrap())
            .all(|(&l, &r)| l <= r));
    }
}

mod backward {
    use super::super::{Backward, DropoutBackward, Gradient};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let operand_gradient = Array::zeros((3, 3));
        let gradient = Array::ones((3, 3));
        let noise = Array::from_elem((3, 3), 2.);
        let status = Rc::new(Cell::new(true));
        let op = DropoutBackward::new(
            Rc::new(Gradient::from_ndarray(operand_gradient.clone())),
            Rc::new(Gradient::from_ndarray(gradient.clone())),
            0.,
            new_shared(noise.clone()),
            status.clone(),
        );

        assert_eq!(op.status.get(), status.get());
        assert_eq!(op.p, 0.);
        are_similar(op.operand_gradient.borrow(), &operand_gradient)?;
        are_similar(op.gradient.borrow(), &gradient)?;
        are_similar(op.noise.borrow(), &noise)
    }

    #[test]
    fn one_probability() -> Result<(), Box<dyn Error>> {
        let op = DropoutBackward::new(
            Rc::new(Gradient::zeros((3, 3))),
            Rc::new(Gradient::from_ndarray(Array::ones((3, 3)))),
            1.,
            new_shared(Array::zeros((3, 3))),
            Rc::new(Cell::new(true)),
        );

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::zeros((3, 3)))
    }

    #[test]
    fn zero_probability() -> Result<(), Box<dyn Error>> {
        let op = DropoutBackward::new(
            Rc::new(Gradient::zeros((3, 3))),
            Rc::new(Gradient::from_ndarray(Array::ones((3, 3)))),
            0.,
            new_shared(Array::zeros((3, 3))),
            Rc::new(Cell::new(true)),
        );

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::ones((3, 3)))
    }
}
