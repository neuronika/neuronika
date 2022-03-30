use super::{reduce, Backward, BroadTensor, Broadcasted, Forward, OptionalTensor, Shared, Tensor};
use ndarray::{DimMax, Dimension, Zip};
use std::{cell::Cell, rc::Rc};

pub struct Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left: Shared<Tensor<D>>,
    right: Shared<Tensor<E>>,
    data: Shared<BroadTensor<D, E>>,
    computed: Cell<bool>,
}

impl<D, E> Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left: Shared<Tensor<D>>,
        right: Shared<Tensor<E>>,
        data: Shared<BroadTensor<D, E>>,
    ) -> Self {
        Self {
            left,
            right,
            data,
            computed: Cell::default(),
        }
    }
}

impl<D, E> Forward for Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.borrow())
            .and_broadcast(&*self.right.borrow())
            .for_each(|v, l, r| *v = l + r);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}
pub struct AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand: Rc<OptionalTensor<D>>,
    gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
}

impl<D, E> AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand: Rc<OptionalTensor<D>>,
        gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
    ) -> Self {
        Self { operand, gradient }
    }
}

impl<D, E> Backward for AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let reduced = reduce(self.operand.shape(), &*self.gradient.content());
        *self.operand.content_mut() += &reduced;
    }
}

pub struct AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand: Rc<OptionalTensor<E>>,
    gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
}

impl<D, E> AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand: Rc<OptionalTensor<E>>,
        gradient: Rc<OptionalTensor<Broadcasted<D, E>>>,
    ) -> Self {
        Self { operand, gradient }
    }
}

impl<D, E> Backward for AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        let reduced = reduce(self.operand.shape(), &*self.gradient.content());
        *self.operand.content_mut() += &reduced;
    }
}

pub struct AdditionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left: AdditionBackwardLeft<D, E>,
    right: AdditionBackwardRight<D, E>,
}

impl<D, E> AdditionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(left: AdditionBackwardLeft<D, E>, right: AdditionBackwardRight<D, E>) -> Self {
        Self { left, right }
    }
}

impl<D, E> Backward for AdditionBackward<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn backward(&self) {
        self.left.backward();
        self.right.backward();
    }
}

// #[cfg(test)]
// mod test;
