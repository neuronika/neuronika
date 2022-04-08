use std::rc::Rc;

use ndarray::{Array, Dimension};

use crate::{gradient::Gradient, utils::Shared};

use super::{Backward, Forward};

pub(crate) struct Chunk<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    chunk_no: usize,
    shape: D,
    data: Shared<Array<f32, D>>,
}

impl<D> Chunk<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_data: Shared<Array<f32, D>>,
        data: Shared<Array<f32, D>>,
        chunk_no: usize,
    ) -> Self {
        let shape = data.borrow().raw_dim();

        Self {
            operand_data,
            shape,
            data,
            chunk_no,
        }
    }
}

impl<D> Forward for Chunk<D>
where
    D: Dimension,
{
    fn forward(&self) {
        let (mut data, operand_data, shape, chunk_no) = (
            self.data.borrow_mut(),
            self.operand_data.borrow(),
            &self.shape,
            self.chunk_no,
        );

        let operand_data_chunk = operand_data
            .exact_chunks(shape.clone())
            .into_iter()
            .skip(chunk_no)
            .take(1)
            .next()
            .unwrap();

        data.assign(&operand_data_chunk);
    }
}

pub(crate) struct ChunkBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<D>>,
    chunk_no: usize,
}

impl<D> ChunkBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<D>>,
        chunk_no: usize,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            chunk_no,
        }
    }
}

impl<D> Backward for ChunkBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = self.operand_gradient.borrow_mut();
        let mut operand_gradient_chunk = operand_gradient
            .exact_chunks_mut(self.gradient.shape())
            .into_iter()
            .skip(self.chunk_no)
            .take(1)
            .next()
            .unwrap();

        operand_gradient_chunk += &*self.gradient.borrow();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
