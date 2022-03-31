use super::{reduce, Backward, Broadcasted, Forward, SharedTensor, SwitchableTensor};
use ndarray::{DimMax, Dimension, Zip};
use std::rc::Rc;

pub struct Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left: SharedTensor<D>,
    right: SharedTensor<E>,
    data: SharedTensor<Broadcasted<D, E>>,
}

impl<D, E> Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        left: SharedTensor<D>,
        right: SharedTensor<E>,
        data: SharedTensor<Broadcasted<D, E>>,
    ) -> Self {
        Self { left, right, data }
    }
}

impl<D, E> Forward for Addition<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn forward(&self) {
        Zip::from(&mut *self.data.borrow_mut())
            .and_broadcast(&*self.left.borrow())
            .and_broadcast(&*self.right.borrow())
            .for_each(|v, &l, &r| *v = l + r);
    }
}
pub struct AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<Broadcasted<D, E>>>,
}

impl<D, E> AdditionBackwardLeft<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<Broadcasted<D, E>>>,
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
        let reduced = reduce(self.operand.shape(), &*self.gradient.array());
        *self.operand.array_mut() += &reduced;
    }
}

pub struct AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    operand: Rc<SwitchableTensor<E>>,
    gradient: Rc<SwitchableTensor<Broadcasted<D, E>>>,
}

impl<D, E> AdditionBackwardRight<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub fn new(
        operand: Rc<SwitchableTensor<E>>,
        gradient: Rc<SwitchableTensor<Broadcasted<D, E>>>,
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
        let reduced = reduce(self.operand.shape(), &*self.gradient.array());
        *self.operand.array_mut() += &reduced;
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
