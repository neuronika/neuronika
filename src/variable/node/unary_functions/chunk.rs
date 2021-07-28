use super::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Chunk ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
        let mut data = self.data.borrow_mut();
        let operand_data = self.operand.data();
        let (chunk_shape, chunk_no) = (&self.chunk_shape, self.chunk_no);
        let operanderand_data_chunk = operand_data
            .exact_chunks(chunk_shape.clone())
            .into_iter()
            .skip(chunk_no)
            .take(1)
            .next()
            .unwrap();
        Zip::from(&mut *data)
            .and(&operanderand_data_chunk)
            .for_each(|chunk_el, operanderand_data_chunk_el| {
                *chunk_el = *operanderand_data_chunk_el
            });
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ChunkBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ChunkBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    operand: Rc<T>,
    chunk_id: usize,
}

impl<T: Gradient + Overwrite> ChunkBackward<T> {
    pub fn new(operand: Rc<T>, grad_chunk: Tensor<T::Dim>, chunk_id: usize) -> Self {
        let shape = grad_chunk.raw_dim();

        Self {
            gradient: RefCell::new(Some(grad_chunk)),
            shape,
            overwrite: Cell::new(true),
            operand,
            chunk_id,
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
        let mut diff_operand = self.operand.gradient_mut();
        let grad = self.gradient();
        let mut op_gradient_chunk = diff_operand
            .exact_chunks_mut(self.shape.clone())
            .into_iter()
            .skip(self.chunk_id)
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod test {}
