use std::error::Error;

use ndarray::Array;

use crate::utils::{are_similar, new_shared};

mod forward {
    use super::super::{Chunk, Forward};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let input_data = Array::linspace(-4., 4., 9).into_shape((3, 3))?;
        let data = Array::zeros((1, 3));
        let op = Chunk::new(new_shared(input_data.clone()), new_shared(data.clone()), 0);

        are_similar(op.operand_data.borrow(), &input_data)?;
        are_similar(op.data.borrow(), &data)
    }

    #[test]
    fn base_case() -> Result<(), Box<dyn Error>> {
        let input_data = Array::linspace(-4., 4., 9).into_shape((3, 3))?;
        let data = Array::zeros((1, 3));

        let op = Chunk::new(new_shared(input_data.clone()), new_shared(data.clone()), 0);
        op.forward();
        are_similar(
            op.data.borrow(),
            &Array::from_shape_vec((1, 3), vec![-4., -3., -2.])?,
        )?;

        let op = Chunk::new(new_shared(input_data.clone()), new_shared(data.clone()), 1);
        op.forward();
        are_similar(
            op.data.borrow(),
            &Array::from_shape_vec((1, 3), vec![-1., 0., 1.])?,
        )?;

        let op = Chunk::new(new_shared(input_data), new_shared(data), 2);
        op.forward();
        are_similar(
            op.data.borrow(),
            &Array::from_shape_vec((1, 3), vec![2., 3., 4.])?,
        )
    }
}

#[cfg(test)]
mod backward {
    use std::rc::Rc;

    use super::super::{Backward, ChunkBackward, Gradient};
    use super::*;

    #[test]
    fn creation() -> Result<(), Box<dyn Error>> {
        let operand_gradient = Array::zeros((3, 3));
        let gradient = Array::ones((1, 3));
        let op = ChunkBackward::new(
            Rc::new(Gradient::from_ndarray(operand_gradient.clone())),
            Rc::new(Gradient::from_ndarray(gradient.clone())),
            0,
        );

        are_similar(op.operand_gradient.borrow(), &operand_gradient)?;
        are_similar(op.gradient.borrow(), &gradient)
    }

    #[test]
    fn base_case() -> Result<(), Box<dyn Error>> {
        let op = ChunkBackward::new(
            Rc::new(Gradient::ndarray_zeros((3, 3))),
            Rc::new(Gradient::from_ndarray(Array::ones((1, 3)))),
            0,
        );
        op.backward();
        are_similar(
            op.operand_gradient.borrow(),
            &Array::from_shape_vec((3, 3), vec![1., 1., 1., 0., 0., 0., 0., 0., 0.])?,
        )?;
        op.backward();
        are_similar(
            op.operand_gradient.borrow(),
            &Array::from_shape_vec((3, 3), vec![2., 2., 2., 0., 0., 0., 0., 0., 0.])?,
        )?;

        let op = ChunkBackward::new(
            Rc::new(Gradient::ndarray_zeros((3, 3))),
            Rc::new(Gradient::from_ndarray(Array::ones((1, 3)))),
            1,
        );
        op.backward();
        are_similar(
            op.operand_gradient.borrow(),
            &Array::from_shape_vec((3, 3), vec![0., 0., 0., 1., 1., 1., 0., 0., 0.])?,
        )?;
        op.backward();
        are_similar(
            op.operand_gradient.borrow(),
            &Array::from_shape_vec((3, 3), vec![0., 0., 0., 2., 2., 2., 0., 0., 0.])?,
        )?;

        let op = ChunkBackward::new(
            Rc::new(Gradient::ndarray_zeros((3, 3))),
            Rc::new(Gradient::from_ndarray(Array::ones((1, 3)))),
            2,
        );
        op.backward();
        are_similar(
            op.operand_gradient.borrow(),
            &Array::from_shape_vec((3, 3), vec![0., 0., 0., 0., 0., 0., 1., 1., 1.])?,
        )?;

        op.backward();
        are_similar(
            op.operand_gradient.borrow(),
            &Array::from_shape_vec((3, 3), vec![0., 0., 0., 0., 0., 0., 2., 2., 2.])?,
        )
    }
}
