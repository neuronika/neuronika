use std::error::Error;

use ndarray::Array;

use crate::utils::{are_similar, new_shared};

#[cfg(test)]
mod forward {
    use super::super::{Exp, Forward};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let operand_data = Array::zeros((3, 3));
        let data = Array::ones((3, 3));
        let op = Exp::new(new_shared(operand_data.clone()), new_shared(data.clone()));

        are_similar(op.operand_data.borrow(), &operand_data)?;
        are_similar(op.data.borrow(), &data)
    }

    #[test]
    fn base_case() -> Result<(), Box<dyn Error>> {
        let op = Exp::new(
            new_shared(Array::linspace(-4., 4., 9).into_shape((3, 3))?),
            new_shared(Array::zeros((3, 3))),
        );

        op.forward();
        are_similar(
            op.data.borrow(),
            &Array::from_shape_fn(9, |i| (-4. + i as f32).exp()).into_shape((3, 3))?,
        )
    }
}

#[cfg(test)]
mod backward {
    use std::rc::Rc;

    use super::super::{Backward, ExpBackward, Gradient};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let operand_gradient = Array::zeros((3, 3));
        let data = Array::ones((3, 3));
        let gradient = Array::from_elem((3, 3), 2.);
        let op = ExpBackward::new(
            Rc::new(Gradient::from_ndarray(operand_gradient.clone())),
            new_shared(data.clone()),
            Rc::new(Gradient::from_ndarray(gradient.clone())),
        );

        are_similar(op.operand_gradient.borrow(), &operand_gradient)?;
        are_similar(op.data.borrow(), &data)?;
        are_similar(op.gradient.borrow(), &gradient)
    }

    #[test]
    fn base_case() -> Result<(), Box<dyn Error>> {
        let data = Array::from_shape_fn(9, |i| (-4. + i as f32).exp()).into_shape((3, 3))?;
        let op = ExpBackward::new(
            Rc::new(Gradient::ndarray_zeros((3, 3))),
            new_shared(data.clone()),
            Rc::new(Gradient::from_ndarray(Array::ones((3, 3)))),
        );

        op.backward();
        are_similar(op.operand_gradient.borrow(), &data)?;

        op.backward();
        are_similar(op.operand_gradient.borrow(), &(data * 2.))
    }
}
