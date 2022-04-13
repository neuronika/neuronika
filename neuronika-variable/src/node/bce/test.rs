use std::error::Error;

use ndarray::{arr0, Array};

use crate::utils::{are_similar, new_shared};

#[cfg(test)]
mod forward {
    use super::super::{BinaryCrossEntropy, Forward, Reduction};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let input_data = Array::linspace(0., 1., 9).into_shape((3, 3))?;
        let target_data = Array::linspace(1., 0., 9).into_shape((3, 3))?;
        let data = Array::zeros(());
        let op = BinaryCrossEntropy::new(
            new_shared(input_data.clone()),
            new_shared(target_data.clone()),
            new_shared(data.clone()),
            Reduction::Mean,
        );

        are_similar(op.input_data.borrow(), &input_data)?;
        are_similar(op.target_data.borrow(), &target_data)?;
        are_similar(op.data.borrow(), &data)
    }

    #[test]
    fn base_case_mean() -> Result<(), Box<dyn Error>> {
        let op = BinaryCrossEntropy::new(
            new_shared(Array::linspace(0., 1., 9).into_shape((3, 3))?),
            new_shared(Array::linspace(1., 0., 9).into_shape((3, 3))?),
            new_shared(arr0(0.)),
            Reduction::Mean,
        );

        op.forward();
        are_similar(op.data.borrow(), &arr0(23.12971))
    }

    #[test]
    fn base_case_sum() -> Result<(), Box<dyn Error>> {
        let op = BinaryCrossEntropy::new(
            new_shared(Array::linspace(0., 1., 9).into_shape((3, 3))?),
            new_shared(Array::linspace(1., 0., 9).into_shape((3, 3))?),
            new_shared(arr0(0.)),
            Reduction::Sum,
        );

        op.forward();
        are_similar(op.data.borrow(), &arr0(208.16739))
    }
}

#[cfg(test)]
mod backward {
    use std::rc::Rc;

    use super::super::{Backward, BinaryCrossEntropyBackward, Gradient, Reduction};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let input_data = Array::linspace(0., 1., 9).into_shape((3, 3))?;
        let target_data = Array::linspace(1., 0., 9).into_shape((3, 3))?;
        let input_gradient = Array::zeros((3, 3));
        let gradient = arr0(0.);
        let op = BinaryCrossEntropyBackward::new(
            new_shared(input_data.clone()),
            new_shared(target_data.clone()),
            Rc::new(Gradient::from_ndarray(input_gradient.clone())),
            Rc::new(Gradient::from_ndarray(gradient.clone())),
            Reduction::Sum,
        );

        are_similar(op.input_data.borrow(), &input_data)?;
        are_similar(op.target_data.borrow(), &target_data)?;
        are_similar(op.input_gradient.borrow(), &input_gradient)?;
        are_similar(op.gradient.borrow(), &gradient)
    }

    #[test]
    fn base_case_mean() -> Result<(), Box<dyn Error>> {
        let op = BinaryCrossEntropyBackward::new(
            new_shared(Array::linspace(0., 1., 9).into_shape((3, 3))?),
            new_shared(Array::linspace(1., 0., 9).into_shape((3, 3))?),
            Rc::new(Gradient::ndarray_zeros((3, 3))),
            Rc::new(Gradient::from_ndarray(arr0(1.))),
            Reduction::Mean,
        );

        op.backward();
        are_similar(
            op.input_gradient.borrow(),
            &Array::from_shape_vec(
                (3, 3),
                vec![
                    -932067.56, -0.761905, -0.296296, -0.118519, 0., 0.118519, 0.296296, 0.761905,
                    932067.56,
                ],
            )?,
        )
    }

    #[test]
    fn base_case_sum() -> Result<(), Box<dyn Error>> {
        let op = BinaryCrossEntropyBackward::new(
            new_shared(Array::linspace(0., 1., 9).into_shape((3, 3))?),
            new_shared(Array::linspace(1., 0., 9).into_shape((3, 3))?),
            Rc::new(Gradient::ndarray_zeros((3, 3))),
            Rc::new(Gradient::from_ndarray(arr0(1.))),
            Reduction::Sum,
        );

        op.backward();
        are_similar(
            op.input_gradient.borrow(),
            &Array::from_shape_vec(
                (3, 3),
                vec![
                    -8388608., -6.857143, -2.666667, -1.066667, 0., 1.066667, 2.666667, 6.857143,
                    8388608.,
                ],
            )?,
        )
    }
}
