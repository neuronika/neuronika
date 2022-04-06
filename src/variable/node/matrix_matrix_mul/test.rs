use std::{error::Error, rc::Rc};

use ndarray::Array;

use crate::variable::utils::{are_similar, new_shared};

mod forward {
    use super::super::{Forward, MatrixMatrixMul};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let left = Array::linspace(1., 9., 9).into_shape((3, 3))?;
        let right = Array::ones((3, 3));
        let data = Array::zeros((3, 3));
        let op = MatrixMatrixMul::new(
            new_shared(left.clone()),
            new_shared(right.clone()),
            new_shared(data.clone()),
        );

        are_similar(op.left_data.borrow(), &left)?;
        are_similar(op.right_data.borrow(), &right)?;
        are_similar(op.data.borrow(), &data)
    }

    #[test]
    fn base_case() -> Result<(), Box<dyn Error>> {
        let left = Array::linspace(1., 9., 9).into_shape((3, 3))?;
        let right = Array::zeros((3, 3));
        let data = Array::zeros((3, 3));
        let op = MatrixMatrixMul::new(
            new_shared(left.clone()),
            new_shared(right.clone()),
            new_shared(data),
        );

        op.forward();
        are_similar(op.data.borrow(), &(left.dot(&right)))
    }
}

mod backward {
    use super::super::{
        Backward, Gradient, MatrixMatrixMulBackward, MatrixMatrixMulBackwardLeft,
        MatrixMatrixMulBackwardRight,
    };
    use super::*;

    #[test]
    fn left_creation() -> Result<(), Box<dyn Error>> {
        let right_data = Array::zeros((3, 3));
        let left_grad = Array::ones((3, 3));
        let grad = Array::from_elem((3, 3), 2.);
        let op = MatrixMatrixMulBackwardLeft::new(
            new_shared(right_data.clone()),
            Rc::new(Gradient::from_ndarray(left_grad.clone())),
            Rc::new(Gradient::from_ndarray(grad.clone())),
        );

        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.left_gradient.borrow(), &left_grad)?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn left_base_case() -> Result<(), Box<dyn Error>> {
        let right_data = Array::linspace(10., 18., 9).into_shape((3, 3))?;
        let left_grad = Array::zeros((3, 3));
        let grad = Array::ones((3, 3));
        let op = MatrixMatrixMulBackwardLeft::new(
            new_shared(right_data.clone()),
            Rc::new(Gradient::from_ndarray(left_grad)),
            Rc::new(Gradient::from_ndarray(grad.clone())),
        );

        op.backward();
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(
            op.left_gradient.borrow(),
            &(Array::from_shape_vec((3, 3), vec![33., 42., 51., 33., 42., 51., 33., 42., 51.]))?,
        )?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(
            op.left_gradient.borrow(),
            &(Array::from_shape_vec((3, 3), vec![66., 84., 102., 66., 84., 102., 66., 84., 102.]))?,
        )?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn right_creation() -> Result<(), Box<dyn Error>> {
        let left_data = Array::zeros((3, 3));
        let right_grad = Array::ones((3, 3));
        let grad = Array::from_elem((3, 3), 2.);
        let op = MatrixMatrixMulBackwardRight::new(
            new_shared(left_data.clone()),
            Rc::new(Gradient::from_ndarray(right_grad.clone())),
            Rc::new(Gradient::from_ndarray(grad.clone())),
        );

        are_similar(op.left_data.borrow(), &left_data)?;
        are_similar(op.right_gradient.borrow(), &right_grad)?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn right_base_case() -> Result<(), Box<dyn Error>> {
        let left_data = Array::linspace(1., 9., 9).into_shape((3, 3))?;
        let right_grad = Array::zeros((3, 3));
        let grad = Array::ones((3, 3));
        let op = MatrixMatrixMulBackwardRight::new(
            new_shared(left_data.clone()),
            Rc::new(Gradient::from_ndarray(right_grad)),
            Rc::new(Gradient::from_ndarray(grad.clone())),
        );

        op.backward();
        are_similar(op.left_data.borrow(), &left_data)?;
        are_similar(
            op.right_gradient.borrow(),
            &(Array::from_shape_vec((3, 3), vec![12., 12., 12., 15., 15., 15., 18., 18., 18.]))?,
        )?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.left_data.borrow(), &left_data)?;
        are_similar(
            op.right_gradient.borrow(),
            &(Array::from_shape_vec((3, 3), vec![24., 24., 24., 30., 30., 30., 36., 36., 36.]))?,
        )?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn backward() -> Result<(), Box<dyn Error>> {
        let left_data = Array::linspace(1., 9., 9).into_shape((3, 3))?;
        let right_data = Array::linspace(10., 18., 9).into_shape((3, 3))?;
        let left_grad = Array::zeros((3, 3));
        let right_grad = Array::zeros((3, 3));
        let grad = Array::ones((3, 3));
        let op = MatrixMatrixMulBackward::new(
            MatrixMatrixMulBackwardLeft::new(
                new_shared(right_data.clone()),
                Rc::new(Gradient::from_ndarray(left_grad)),
                Rc::new(Gradient::from_ndarray(grad.clone())),
            ),
            MatrixMatrixMulBackwardRight::new(
                new_shared(left_data.clone()),
                Rc::new(Gradient::from_ndarray(right_grad)),
                Rc::new(Gradient::from_ndarray(grad.clone())),
            ),
        );

        op.backward();
        are_similar(op.left.right_data.borrow(), &right_data)?;
        are_similar(
            op.left.left_gradient.borrow(),
            &(Array::from_shape_vec((3, 3), vec![33., 42., 51., 33., 42., 51., 33., 42., 51.]))?,
        )?;
        are_similar(op.left.gradient.borrow(), &grad)?;
        are_similar(op.right.left_data.borrow(), &left_data)?;
        are_similar(
            op.right.right_gradient.borrow(),
            &(Array::from_shape_vec((3, 3), vec![12., 12., 12., 15., 15., 15., 18., 18., 18.]))?,
        )?;
        are_similar(op.right.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.left.right_data.borrow(), &right_data)?;
        are_similar(
            op.left.left_gradient.borrow(),
            &(Array::from_shape_vec((3, 3), vec![66., 84., 102., 66., 84., 102., 66., 84., 102.]))?,
        )?;
        are_similar(op.left.gradient.borrow(), &grad)?;
        are_similar(op.right.left_data.borrow(), &left_data)?;
        are_similar(
            op.right.right_gradient.borrow(),
            &(Array::from_shape_vec((3, 3), vec![24., 24., 24., 30., 30., 30., 36., 36., 36.]))?,
        )?;
        are_similar(op.right.gradient.borrow(), &grad)
    }
}
