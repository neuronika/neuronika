use std::{error::Error, rc::Rc};

use ndarray::Array;

use crate::variable::utils::{are_similar, new_shared};

mod forward {
    use super::super::{Division, Forward};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let left = Array::linspace(1., 9., 9).into_shape((3, 3))?;
        let right = Array::from_elem((3, 3), 2.);
        let data = Array::zeros((3, 3));
        let op = Division::new(
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
        let data = Array::zeros((3, 3));
        let op = Division::new(
            new_shared(left.clone()),
            new_shared(right.clone()),
            new_shared(data),
        );

        op.forward();
        are_similar(op.data.borrow(), &(left / right))
    }

    #[test]
    fn left_broadcast() -> Result<(), Box<dyn Error>> {
        let left = Array::linspace(1., 3., 3).into_shape((1, 3))?;
        let right = Array::from_elem((2, 2, 3), 2.);
        let data = Array::zeros((2, 2, 3));
        let op = Division::new(
            new_shared(left.clone()),
            new_shared(right.clone()),
            new_shared(data),
        );

        op.forward();
        are_similar(op.data.borrow(), &(left / right))
    }

    #[test]
    #[should_panic]
    fn wrong_left_broadcast() {
        let left = Array::zeros((3, 3));
        let right = Array::zeros((2, 2, 3));
        let data = Array::zeros((2, 2, 3));
        let op = Division::new(new_shared(left), new_shared(right), new_shared(data));

        op.forward();
    }

    #[test]
    fn right_broadcast() -> Result<(), Box<dyn Error>> {
        let left = Array::from_elem((2, 2, 3), 2.);
        let right = Array::linspace(1., 3., 3).into_shape((1, 3))?;
        let data = Array::zeros((2, 2, 3));
        let op = Division::new(
            new_shared(left.clone()),
            new_shared(right.clone()),
            new_shared(data),
        );

        op.forward();
        are_similar(op.data.borrow(), &(left / right))
    }

    #[test]
    #[should_panic]
    fn wrong_right_broadcast() {
        let left = Array::zeros((2, 2, 3));
        let right = Array::zeros((3, 3));
        let data = Array::zeros((2, 2, 3));
        let op = Division::new(new_shared(left), new_shared(right), new_shared(data));

        op.forward();
    }
}

mod backward {
    use super::super::{
        Backward, BufferedGradient, DivisionBackward, DivisionBackwardLeft, DivisionBackwardRight,
        Gradient,
    };
    use super::*;

    #[test]
    fn left_creation() -> Result<(), Box<dyn Error>> {
        let left_grad = Array::ones((3, 3));
        let right_data = Array::from_elem((3, 3), 2.);
        let grad = Array::from_elem((3, 3), 3.);
        let buff = Array::zeros((3, 3));
        let op = DivisionBackwardLeft::new(
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
        let left_grad = Array::zeros((3, 3));
        let right_data = Array::from_elem((3, 3), 5.);
        let grad = Array::from_elem((3, 3), 1.);
        let op = DivisionBackwardLeft::new(
            new_shared(right_data.clone()),
            Rc::new(Gradient::from_ndarray(left_grad)),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        op.backward();
        are_similar(op.left_gradient.borrow(), &Array::from_elem((3, 3), 0.2))?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.left_gradient.borrow(), &Array::from_elem((3, 3), 0.4))?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn left_reduction() -> Result<(), Box<dyn Error>> {
        let left_grad = Array::zeros(3);
        let right_data = Array::from_elem((3, 3), 5.);
        let grad = Array::from_elem((3, 3), 1.);
        let op = DivisionBackwardLeft::new(
            new_shared(right_data.clone()),
            Rc::new(Gradient::from_ndarray(left_grad)),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        op.backward();
        are_similar(op.left_gradient.borrow(), &Array::from_elem(3, 0.6))?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.left_gradient.borrow(), &Array::from_elem(3, 1.2))?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    #[should_panic]
    fn wrong_left_reduction() {
        let left_grad = Array::zeros((3, 3));
        let right_data = Array::zeros((2, 2, 3));
        let grad = Array::zeros((2, 2, 3));
        let op = DivisionBackwardLeft::new(
            new_shared(right_data),
            Rc::new(Gradient::from_ndarray(left_grad)),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(grad)))),
        );

        op.backward();
    }

    #[test]
    fn right_creation() -> Result<(), Box<dyn Error>> {
        let left_data = Array::ones((3, 3));
        let right_data = Array::from_elem((3, 3), 2.);
        let right_grad = Array::from_elem((3, 3), 3.);
        let grad = Array::from_elem((3, 3), 4.);
        let buff = Array::zeros((3, 3));
        let op = DivisionBackwardRight::new(
            new_shared(left_data.clone()),
            new_shared(right_data.clone()),
            Rc::new(Gradient::from_ndarray(right_grad.clone())),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        are_similar(op.left_data.borrow(), &left_data)?;
        are_similar(op.right_data.borrow(), &right_data)?;
        are_similar(op.right_gradient.borrow(), &right_grad)?;
        are_similar(op.gradient.borrow(), &grad)?;
        are_similar(op.gradient.buffer(), &buff)
    }

    #[test]
    fn right_base_case() -> Result<(), Box<dyn Error>> {
        let left_data = Array::from_elem((3, 3), 3.);
        let right_data = Array::from_elem((3, 3), 5.);
        let right_grad = Array::zeros((3, 3));
        let grad = Array::ones((3, 3));
        let op = DivisionBackwardRight::new(
            new_shared(left_data),
            new_shared(right_data),
            Rc::new(Gradient::from_ndarray(right_grad)),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        op.backward();
        are_similar(op.right_gradient.borrow(), &Array::from_elem((3, 3), -0.12))?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.right_gradient.borrow(), &Array::from_elem((3, 3), -0.24))?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    fn right_reduction() -> Result<(), Box<dyn Error>> {
        let left_data = Array::from_elem((3, 3), 3.);
        let right_data = Array::from_elem(3, 5.);
        let right_grad = Array::zeros(3);
        let grad = Array::ones((3, 3));
        let op = DivisionBackwardRight::new(
            new_shared(left_data),
            new_shared(right_data),
            Rc::new(Gradient::from_ndarray(right_grad)),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(
                grad.clone(),
            )))),
        );

        op.backward();
        are_similar(op.right_gradient.borrow(), &Array::from_elem(3, -0.36))?;
        are_similar(op.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(op.right_gradient.borrow(), &Array::from_elem(3, -0.72))?;
        are_similar(op.gradient.borrow(), &grad)
    }

    #[test]
    #[should_panic]
    fn wrong_right_reduction() {
        let left_data = Array::zeros((2, 2, 3));
        let right_data = Array::zeros((3, 3));
        let right_grad = Array::zeros((3, 3));
        let grad = Array::zeros((2, 2, 3));
        let op = DivisionBackwardRight::new(
            new_shared(left_data),
            new_shared(right_data),
            Rc::new(Gradient::from_ndarray(right_grad)),
            Rc::new(BufferedGradient::new(Rc::new(Gradient::from_ndarray(grad)))),
        );

        op.backward();
    }

    #[test]
    fn backward() -> Result<(), Box<dyn Error>> {
        let left_data = Array::from_elem((3, 3), 3.);
        let left_grad = Array::zeros((3, 3));
        let right_data = Array::from_elem((3, 3), 5.);
        let right_grad = Array::zeros((3, 3));
        let grad = Array::from_elem((3, 3), 1.);
        let shared_grad = Rc::new(Gradient::from_ndarray(grad.clone()));
        let op = DivisionBackward::new(
            DivisionBackwardLeft::new(
                new_shared(right_data.clone()),
                Rc::new(Gradient::from_ndarray(left_grad)),
                Rc::new(BufferedGradient::new(shared_grad.clone())),
            ),
            DivisionBackwardRight::new(
                new_shared(left_data),
                new_shared(right_data.clone()),
                Rc::new(Gradient::from_ndarray(right_grad)),
                Rc::new(BufferedGradient::new(shared_grad)),
            ),
        );

        op.backward();
        are_similar(
            op.left.left_gradient.borrow(),
            &Array::from_elem((3, 3), 0.2),
        )?;
        are_similar(op.left.right_data.borrow(), &right_data)?;
        are_similar(op.left.gradient.borrow(), &grad)?;
        are_similar(
            op.right.right_gradient.borrow(),
            &Array::from_elem((3, 3), -0.12),
        )?;
        are_similar(op.right.right_data.borrow(), &right_data)?;
        are_similar(op.right.gradient.borrow(), &grad)?;

        op.backward();
        are_similar(
            op.left.left_gradient.borrow(),
            &Array::from_elem((3, 3), 0.4),
        )?;
        are_similar(op.left.right_data.borrow(), &right_data)?;
        are_similar(op.left.gradient.borrow(), &grad)?;
        are_similar(
            op.right.right_gradient.borrow(),
            &Array::from_elem((3, 3), -0.24),
        )?;
        are_similar(op.right.right_data.borrow(), &right_data)?;
        are_similar(op.right.gradient.borrow(), &grad)
    }
}
