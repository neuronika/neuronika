use std::error::Error;

use ndarray::Array;

use crate::utils::{are_similar, new_shared};

#[cfg(test)]
mod forward {
    use super::super::{Forward, Logn};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let operand_data = Array::zeros((3, 3));
        let data = Array::ones((3, 3));
        let op = Logn::new(new_shared(operand_data.clone()), new_shared(data.clone()));

        are_similar(op.operand_data.borrow(), &operand_data)?;
        are_similar(op.data.borrow(), &data)
    }

    #[test]
    fn base_case() -> Result<(), Box<dyn Error>> {
        let op = Logn::new(
            new_shared(Array::linspace(1., 9., 9).into_shape((3, 3))?),
            new_shared(Array::zeros((3, 3))),
        );

        op.forward();
        are_similar(
            op.data.borrow(),
            &Array::from_shape_fn(9, |i| (1. + i as f32).ln()).into_shape((3, 3))?,
        )
    }
}

#[cfg(test)]
mod backward {
    use std::rc::Rc;

    use super::super::{Backward, Gradient, LognBackward};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let operand_gradient = Array::zeros((3, 3));
        let operand_data = Array::ones((3, 3));
        let gradient = Array::from_elem((3, 3), 2.);
        let op = LognBackward::new(
            Rc::new(Gradient::from_ndarray(operand_gradient.clone())),
            new_shared(operand_data.clone()),
            Rc::new(Gradient::from_ndarray(gradient.clone())),
        );

        are_similar(op.operand_gradient.borrow(), &operand_gradient)?;
        are_similar(op.operand_data.borrow(), &operand_data)?;
        are_similar(op.gradient.borrow(), &gradient)
    }

    #[test]
    fn base_case() -> Result<(), Box<dyn Error>> {
        let op = LognBackward::new(
            Rc::new(Gradient::ndarray_zeros((3, 3))),
            new_shared(Array::linspace(1., 9., 9).into_shape((3, 3))?),
            Rc::new(Gradient::from_ndarray(Array::ones((3, 3)))),
        );

        op.backward();
        are_similar(
            op.operand_gradient.borrow(),
            &Array::from_shape_fn(9, |i| 1. / (1. + i as f32)).into_shape((3, 3))?,
        )?;

        op.backward();
        are_similar(
            op.operand_gradient.borrow(),
            &Array::from_shape_fn(9, |i| 2. / (1. + i as f32)).into_shape((3, 3))?,
        )
    }
}
