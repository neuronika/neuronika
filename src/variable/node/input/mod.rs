use super::{
    expect_tensor, expect_tensor_mut, Data, Differentiable, Dimension, Forward, Gradient,
    Overwrite, Tensor,
};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
};

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

impl<D: Dimension> Debug for Input<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Input")
            .field("data", &self.data.borrow())
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<D: Dimension> Display for Input<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data.borrow())
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

impl<D: Dimension> Debug for InputBackward<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Input")
            .field("gradient", &self.gradient.borrow())
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<D: Dimension> Display for InputBackward<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

#[cfg(test)]
mod test;
