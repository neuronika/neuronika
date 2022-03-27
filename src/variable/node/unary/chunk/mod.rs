#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::Dimension;
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Chunk<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    chunk_no: usize,
    shape: D,
    data: Rc<RefCell<Tensor<D>>>,
    computed: Cell<bool>,
}

impl<D> Chunk<D>
where
    D: Dimension,
{
    pub fn new(
        operand_data: Rc<RefCell<Tensor<D>>>,
        data: Rc<RefCell<Tensor<D>>>,
        chunk_no: usize,
    ) -> Self {
        let shape = data.borrow().raw_dim();

        Self {
            operand_data,
            shape,
            data,
            chunk_no,
            computed: Cell::default(),
        }
    }
}

impl<D> Forward for Chunk<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
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

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct ChunkBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D>>>>,
    shape: D,
    chunk_no: usize,
}

impl<D> ChunkBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D>>>>,
        shape: D,
        chunk_no: usize,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            shape,
            chunk_no,
        }
    }
}

impl<D> Backward for ChunkBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let (mut operand_gradient, gradient, chunk_no, shape) = (
            expect_tensor_mut(&self.operand_gradient),
            expect_tensor(&self.gradient),
            self.chunk_no,
            self.shape.clone(),
        );

        let mut operand_gradient_chunk = operand_gradient
            .exact_chunks_mut(shape)
            .into_iter()
            .skip(chunk_no)
            .take(1)
            .next()
            .unwrap();

        operand_gradient_chunk += &*gradient;
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
