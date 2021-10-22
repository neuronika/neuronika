use super::{
    expect_tensor, expect_tensor_mut, Data, Differentiable, Dimension, Forward, Gradient,
    Overwrite, Tensor,
};
use std::cell::{Cell, Ref, RefCell, RefMut};

/// The forward component of a leaf of the computational graph.
pub struct Input<D: Dimension> {
    data: RefCell<Tensor<D>>,
    computed: Cell<bool>,
}

impl<D: Dimension> Input<D> {
    pub fn new(data: Tensor<D>) -> super::super::Var<Self> {
        let input = Self {
            data: RefCell::new(data),
            computed: Cell::new(false),
        };

        super::super::Var::new(input)
    }
}

impl<D: Dimension> Data for Input<D> {
    type Dim = D;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<D: Dimension> Forward for Input<D> {
    fn forward(&self) {
        self.computed.set(true);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<D: Dimension> Differentiable for Input<D> {
    type Output = InputBackward<D>;

    fn differentiable(&self) -> Self::Output {
        Self::Output {
            gradient: RefCell::new(Some(Tensor::zeros(self.data().raw_dim()))),
            overwrite: Cell::new(true),
        }
    }
}

/// The backward component of a differentiable leaf of the computational graph.
pub struct InputBackward<D: Dimension> {
    gradient: RefCell<Option<Tensor<D>>>,
    overwrite: Cell<bool>,
}

impl<D: Dimension> InputBackward<D> {
    pub fn zero_grad(&self) {
        expect_tensor_mut(&self.gradient).fill(0.);
    }
}

impl<D: Dimension> Gradient for InputBackward<D> {
    type Dim = D;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<D: Dimension> Overwrite for InputBackward<D> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

#[cfg(test)]
mod test;
