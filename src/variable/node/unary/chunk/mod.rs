#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Tensor,
};
use ndarray::Zip;
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Chunk ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Chunk<T: Data> {
    operand: Rc<T>,
    chunk_no: usize,
    chunk_shape: T::Dim,
    data: RefCell<Tensor<T::Dim>>,
    computed: Cell<bool>,
}

impl<T: Data> Chunk<T> {
    pub fn new(operand: Rc<T>, chunk: Tensor<T::Dim>, chunk_no: usize) -> Self {
        Self {
            operand,
            chunk_shape: chunk.raw_dim(),
            data: RefCell::new(chunk),
            chunk_no,
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Chunk<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (mut data, operand_data, chunk_shape, chunk_no) = (
            self.data.borrow_mut(),
            self.operand.data(),
            &self.chunk_shape,
            self.chunk_no,
        );

        let operand_data_chunk = operand_data
            .exact_chunks(chunk_shape.clone())
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

impl<T: Data> Data for Chunk<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<T: Data> Debug for Chunk<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chunk")
            .field("data", &self.data.borrow())
            .field("chunk_no", &self.chunk_no)
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<T: Data> Display for Chunk<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ChunkBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct ChunkBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    operand: Rc<T>,
    chunk_no: usize,
}

impl<T: Gradient + Overwrite> ChunkBackward<T> {
    pub fn new(operand: Rc<T>, grad_chunk: Tensor<T::Dim>, chunk_no: usize) -> Self {
        let shape = grad_chunk.raw_dim();

        Self {
            gradient: RefCell::new(Some(grad_chunk)),
            shape,
            overwrite: Cell::new(true),
            operand,
            chunk_no,
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for ChunkBackward<T> {
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for ChunkBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for ChunkBackward<T> {
    fn backward(&self) {
        let (mut diff_operand, grad, chunk_no) =
            (self.operand.gradient_mut(), self.gradient(), self.chunk_no);

        let mut op_gradient_chunk = diff_operand
            .exact_chunks_mut(self.shape.clone())
            .into_iter()
            .skip(chunk_no)
            .take(1)
            .next()
            .unwrap();

        let zip = Zip::from(&mut op_gradient_chunk).and(&*grad);
        if self.operand.can_overwrite() {
            zip.for_each(|dest, src| *dest = *src);
            self.operand.set_overwrite(false);
        } else {
            zip.for_each(|dest, src| *dest += src);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

impl<T: Gradient + Overwrite> Debug for ChunkBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkBackward")
            .field("gradient", &self.gradient.borrow())
            .field("chunk_no", &self.chunk_no)
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T: Gradient + Overwrite> Display for ChunkBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
