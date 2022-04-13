use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

use ndarray::{DimMax, Dimension};

use cudnn::BinaryOp;

use crate::{
    autograd::Forward,
    cuda::cuarray::CuArray,
    cuda::cunode::BinaryOperation,
    history::History,
    utils::{cobroadcast, Shared},
};

/// A non-differentiable variable with data allocated on a CUDA capable device.
pub struct CuVar<D>
where
    D: Dimension,
{
    pub(crate) data: Shared<CuArray<f32, D>>,
    pub(crate) history: History<(Rc<dyn Forward>, Cell<bool>)>,
}

impl<D> CuVar<D>
where
    D: Dimension,
{
    pub(crate) fn leaf(array: CuArray<f32, D>) -> Self {
        Self {
            data: Rc::new(RefCell::new(array)),
            history: History::default(),
        }
    }

    pub(crate) fn node(
        data: Shared<CuArray<f32, D>>,
        op: Rc<dyn Forward>,
        mut history: History<(Rc<dyn Forward>, Cell<bool>)>,
    ) -> Self {
        history.insert(Rc::as_ptr(&op) as *const () as usize, (op, Cell::default()));

        Self { data, history }
    }

    pub(crate) fn binary<E>(
        mut self,
        binary_op: BinaryOp,
        rhs: CuVar<E>,
    ) -> CuVar<<D as DimMax<E>>::Output>
    where
        D: 'static + Dimension + DimMax<E>,
        E: 'static + Dimension,
    {
        self.history.merge(rhs.history);

        let dim = cobroadcast(
            self.data.borrow().dimension(),
            rhs.data.borrow().dimension(),
        );

        let data = Rc::new(RefCell::new(CuArray::zeroed(
            dim.size(),
            dim,
            self.data.borrow().device().clone(),
        )));

        let op = Rc::new(BinaryOperation::new(
            binary_op,
            self.data,
            rhs.data,
            data.clone(),
        ));

        CuVar::node(data, op, self.history)
    }

    /// Propagates the computations forwards and populates all the variables from the leaves of the
    /// graph to `self`.
    pub fn forward(&self) {
        let mut buffer = self.history.buffer_mut(); // Borrows for the scope

        // If the length of the buffer is greater than 0 it means that forward has already been
        // called and the path must be recomputed, else the buffer is empty and must be populated.
        if buffer.is_empty() {
            *buffer = self.history.to_vec()
        } else {
            buffer.iter().for_each(|(_, computed)| computed.set(false));
        }

        buffer
            .iter()
            .filter(|(_, computed)| !computed.get())
            .for_each(|(op, computed)| {
                op.forward();
                computed.set(true)
            });
    }
}

impl<D, E> std::ops::Add<CuVar<E>> for CuVar<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = CuVar<<D as DimMax<E>>::Output>;

    fn add(self, rhs: CuVar<E>) -> Self::Output {
        self.binary(BinaryOp::Add, rhs)
    }
}

impl<D, E> std::ops::Mul<CuVar<E>> for CuVar<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = CuVar<<D as DimMax<E>>::Output>;

    fn mul(self, rhs: CuVar<E>) -> Self::Output {
        self.binary(BinaryOp::Mul, rhs)
    }
}
