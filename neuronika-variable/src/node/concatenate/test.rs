use std::error::Error;

use ndarray::{Array, Axis};

use crate::utils::{are_similar, new_shared};

mod forward {
    use super::super::{Concatenate, Forward};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let left = Array::zeros((2, 3));
        let right = Array::ones((1, 3));
        let data = Array::from_elem((3, 3), 2.);
        let op = Concatenate::new(
            new_shared(left.clone()),
            new_shared(right.clone()),
            new_shared(data.clone()),
            0,
        );

        assert_eq!(op.axis, Axis(0));
        are_similar(op.left.borrow(), &left)?;
        are_similar(op.right.borrow(), &right)?;
        are_similar(op.data.borrow(), &data)
    }

    #[test]
    fn rows() -> Result<(), Box<dyn Error>> {
        let op = Concatenate::new(
            new_shared(Array::linspace(-4., 1., 6).into_shape((2, 3))?),
            new_shared(Array::linspace(2., 4., 3).into_shape((1, 3))?),
            new_shared(Array::zeros((3, 3))),
            0,
        );

        op.forward();
        are_similar(
            op.data.borrow(),
            &Array::linspace(-4., 4., 9).into_shape((3, 3))?,
        )
    }

    #[test]
    fn columns() -> Result<(), Box<dyn Error>> {
        let op = Concatenate::new(
            new_shared(Array::linspace(-4., 1., 6).into_shape((3, 2))?),
            new_shared(Array::linspace(2., 4., 3).into_shape((3, 1))?),
            new_shared(Array::zeros((3, 3))),
            1,
        );

        op.forward();
        are_similar(
            op.data.borrow(),
            &Array::from_shape_vec((3, 3), vec![-4., -3., 2., -2., -1., 3., 0., 1., 4.])?,
        )
    }
}

#[cfg(test)]
mod backward {
    use std::rc::Rc;

    use super::super::{
        Backward, ConcatenateBackward, ConcatenateBackwardLeft, ConcatenateBackwardRight, Gradient,
    };
    use super::*;

    #[test]
    fn left_creation() -> Result<(), Box<dyn Error>> {
        let operand_gradient = Array::zeros((2, 3));
        let gradient = Array::ones((3, 3));
        let op = ConcatenateBackwardLeft::new(
            Rc::new(Gradient::from_ndarray(operand_gradient.clone())),
            Rc::new(Gradient::from_ndarray(gradient.clone())),
            0,
        );

        assert_eq!(op.axis, Axis(0));
        are_similar(op.operand_gradient.borrow(), &operand_gradient)?;
        are_similar(op.gradient.borrow(), &gradient)
    }

    #[test]
    fn left_rows() -> Result<(), Box<dyn Error>> {
        let op = ConcatenateBackwardLeft::new(
            Rc::new(Gradient::zeros((2, 3))),
            Rc::new(Gradient::from_ndarray(Array::ones((3, 3)))),
            0,
        );

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::ones((2, 3)))?;

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::from_elem((2, 3), 2.))
    }

    #[test]
    fn left_columns() -> Result<(), Box<dyn Error>> {
        let op = ConcatenateBackwardLeft::new(
            Rc::new(Gradient::zeros((3, 2))),
            Rc::new(Gradient::from_ndarray(Array::ones((3, 3)))),
            1,
        );

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::ones((3, 2)))?;

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::from_elem((3, 2), 2.))
    }

    #[test]
    fn right_creation() -> Result<(), Box<dyn Error>> {
        let operand_gradient = Array::zeros((1, 3));
        let gradient = Array::ones((3, 3));
        let op = ConcatenateBackwardRight::new(
            Rc::new(Gradient::from_ndarray(operand_gradient.clone())),
            Rc::new(Gradient::from_ndarray(gradient.clone())),
            0,
            6,
        );

        assert_eq!(op.axis, Axis(0));
        are_similar(op.operand_gradient.borrow(), &operand_gradient)?;
        are_similar(op.gradient.borrow(), &gradient)
    }

    #[test]
    fn right_rows() -> Result<(), Box<dyn Error>> {
        let op = ConcatenateBackwardRight::new(
            Rc::new(Gradient::zeros((1, 3))),
            Rc::new(Gradient::from_ndarray(Array::ones((3, 3)))),
            0,
            2,
        );

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::ones((1, 3)))?;

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::from_elem((1, 3), 2.))
    }

    #[test]
    fn right_columns() -> Result<(), Box<dyn Error>> {
        let op = ConcatenateBackwardRight::new(
            Rc::new(Gradient::zeros((3, 1))),
            Rc::new(Gradient::from_ndarray(Array::ones((3, 3)))),
            1,
            2,
        );

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::ones((3, 1)))?;

        op.backward();
        are_similar(op.operand_gradient.borrow(), &Array::from_elem((3, 1), 2.))
    }

    #[test]
    fn base_case() -> Result<(), Box<dyn Error>> {
        let shared_grad = Rc::new(Gradient::from_ndarray(Array::ones((3, 3))));
        let op = ConcatenateBackward::new(
            ConcatenateBackwardLeft::new(Rc::new(Gradient::zeros((2, 3))), shared_grad.clone(), 0),
            ConcatenateBackwardRight::new(Rc::new(Gradient::zeros((1, 3))), shared_grad, 0, 2),
        );

        op.backward();
        are_similar(op.left.operand_gradient.borrow(), &Array::ones((2, 3)))?;
        are_similar(op.right.operand_gradient.borrow(), &Array::ones((1, 3)))?;

        op.backward();
        are_similar(
            op.left.operand_gradient.borrow(),
            &Array::from_elem((2, 3), 2.),
        )?;
        are_similar(
            op.right.operand_gradient.borrow(),
            &Array::from_elem((1, 3), 2.),
        )
    }
}
