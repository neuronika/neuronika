use std::{error::Error, rc::Rc};

use ndarray::Array;

use crate::utils::{are_similar, new_shared};

mod forward {
    use super::super::{Forward, Multiplication};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let left = Array::linspace(1., 9., 9).into_shape((3, 3))?;
        let right = Array::from_elem((3, 3), 2.);
        let data = Array::zeros((3, 3));
        let op = Multiplication::new(
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
        let right = Array::from_elem((3, 3), 2.);
        let op = Multiplication::new(
            new_shared(left.clone()),
            new_shared(right.clone()),
            new_shared(Array::zeros((3, 3))),
        );

        op.forward();
        are_similar(op.data.borrow(), &(left * right))
    }

    #[test]
    fn left_broadcast() -> Result<(), Box<dyn Error>> {
        let left = Array::linspace(1., 3., 3).into_shape((1, 3))?;
        let right = Array::from_elem((2, 2, 3), 2.);
        let op = Multiplication::new(
            new_shared(left.clone()),
            new_shared(right.clone()),
            new_shared(Array::zeros((2, 2, 3))),
        );

        op.forward();
        are_similar(op.data.borrow(), &(left * right))
    }

    #[test]
    fn right_broadcast() -> Result<(), Box<dyn Error>> {
        let left = Array::from_elem((2, 2, 3), 2.);
        let right = Array::linspace(1., 3., 3).into_shape((1, 3))?;
        let op = Multiplication::new(
            new_shared(left.clone()),
            new_shared(right.clone()),
            new_shared(Array::zeros((2, 2, 3))),
        );

        op.forward();
        are_similar(op.data.borrow(), &(left * right))
    }
}

mod backward {
    use super::super::{
        Backward, BufferedGradient, Gradient, MultiplicationBackward, MultiplicationBackwardLeft,
        MultiplicationBackwardRight,
    };
    use super::*;

    #[test]
    fn left_creation() -> Result<(), Box<dyn Error>> {
        let left_grad = Array::ones((3, 3));
        let right_data = Array::from_elem((3, 3), 2.);
        let grad = Array::from_elem((3, 3), 3.);
        let buff = Array::zeros((3, 3));
        let op = MultiplicationBackwardLeft::new(
            new_shared(right_data.clone()),
            Rc::new(Gradient::from_ndarray(left_grad.clone())),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        are_similar(op.left_gradient.borrow(), &left_grad)?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.gradient.borrow(), &grad)?;
        are_similar(op.gradient.buffer(), &buff)
    }

    #[test]
    fn left_base_case() -> Result<(), Box<dyn Error>> {
        let right_data = Array::from_elem((3, 3), 5.);
        let grad = Array::from_elem((3, 3), 1.);
        let op = MultiplicationBackwardLeft::new(
            new_shared(right_data.clone()),
            Rc::new(Gradient::zeros((3, 3))),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        op.backward();
        are_similar(op.left_gradient.borrow(), &Array::from_elem((3, 3), 5.))?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.left_gradient.borrow(), &Array::from_elem((3, 3), 10.))?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn left_reduction() -> Result<(), Box<dyn Error>> {
        let right_data = Array::from_elem((3, 3), 5.);
        let grad = Array::from_elem((3, 3), 1.);
        let op = MultiplicationBackwardLeft::new(
            new_shared(right_data.clone()),
            Rc::new(Gradient::zeros(3)),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        op.backward();
        are_similar(op.left_gradient.borrow(), &Array::from_elem(3, 15.))?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.left_gradient.borrow(), &Array::from_elem(3, 30.))?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn right_creation() -> Result<(), Box<dyn Error>> {
        let left_data = Array::ones((3, 3));
        let right_grad = Array::from_elem((3, 3), 2.);
        let grad = Array::from_elem((3, 3), 3.);
        let buff = Array::zeros((3, 3));
        let op = MultiplicationBackwardRight::new(
            new_shared(left_data.clone()),
            Rc::new(Gradient::from_ndarray(right_grad.clone())),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        are_similar(op.left_data.borrow(), &left_data)?;
        are_similar(op.right_gradient.borrow(), &right_grad)?;
        are_similar(op.gradient.borrow(), &grad)?;
        are_similar(op.gradient.buffer(), &buff)
    }

    #[test]
    fn right_base_case() -> Result<(), Box<dyn Error>> {
        let grad = Array::ones((3, 3));
        let op = MultiplicationBackwardRight::new(
            new_shared(Array::from_elem((3, 3), 5.)),
            Rc::new(Gradient::zeros((3, 3))),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        op.backward();
        are_similar(op.right_gradient.borrow(), &Array::from_elem((3, 3), 5.))?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.right_gradient.borrow(), &Array::from_elem((3, 3), 10.))?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn right_reduction() -> Result<(), Box<dyn Error>> {
        let grad = Array::ones((3, 3));
        let op = MultiplicationBackwardRight::new(
            new_shared(Array::from_elem((3, 3), 5.)),
            Rc::new(Gradient::zeros(3)),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        op.backward();
        are_similar(op.right_gradient.borrow(), &Array::from_elem(3, 15.))?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.right_gradient.borrow(), &Array::from_elem(3, 30.))?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn backward() -> Result<(), Box<dyn Error>> {
        let left_data = Array::from_elem((3, 3), 3.);
        let right_data = Array::from_elem((3, 3), 5.);
        let grad = Array::from_elem((3, 3), 1.);
        let shared_grad = Rc::new(Gradient::from_ndarray(grad.clone()));
        let op = MultiplicationBackward::new(
            MultiplicationBackwardLeft::new(
                new_shared(right_data.clone()),
                Rc::new(Gradient::zeros((3, 3))),
                Rc::new(BufferedGradient::new(shared_grad.clone())),
            ),
            MultiplicationBackwardRight::new(
                new_shared(left_data.clone()),
                Rc::new(Gradient::zeros((3, 3))),
                Rc::new(BufferedGradient::new(shared_grad)),
            ),
        );

        op.backward();
        are_similar(
            op.left.left_gradient.borrow(),
            &Array::from_elem((3, 3), 5.),
        )?;
        are_similar(op.left.right_data.borrow(), &right_data)?;
        are_similar(op.left.gradient.borrow(), &grad)?;
        are_similar(
            op.right.right_gradient.borrow(),
            &Array::from_elem((3, 3), 3.),
        )?;
        are_similar(op.right.left_data.borrow(), &left_data)?;
        are_similar(op.right.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(
            op.left.left_gradient.borrow(),
            &Array::from_elem((3, 3), 10.),
        )?;
        are_similar(op.left.right_data.borrow(), &right_data)?;
        are_similar(op.left.gradient.borrow(), &grad)?;
        are_similar(
            op.right.right_gradient.borrow(),
            &Array::from_elem((3, 3), 6.),
        )?;
        are_similar(op.right.left_data.borrow(), &left_data)?;
        are_similar(op.right.gradient.borrow(), &grad)
    }
}
