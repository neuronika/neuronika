#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Cache, Data, Forward, Gradient,
    Overwrite, Tensor,
};
use ndarray::{Axis, Dimension, RemoveAxis, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiStack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    operands: Vec<Rc<dyn Data<Dim = D>>>,
    axis: usize,
    data: RefCell<Tensor<D::Larger>>,
    computed: Cell<bool>,
}

impl<D> MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operands: Vec<Rc<dyn Data<Dim = D>>>,
        axis: usize,
        tensor: Tensor<D::Larger>,
    ) -> Self {
        let (data, computed) = (RefCell::new(tensor), Cell::new(false));

        Self {
            operands,
            axis,
            data,
            computed,
        }
    }
}

impl<D> Data for MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    type Dim = D::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<D> Cache for MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<D> Forward for MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (mut data, axis) = (self.data.borrow_mut(), self.axis);

        self.operands
            .iter()
            .zip(data.axis_iter_mut(Axis(axis)))
            .for_each(|(operand, axis_data)| {
                let operand_data = operand.data();
                Zip::from(&mut axis_data.into_dimensionality::<D>().unwrap())
                    .and(&*operand_data)
                    .for_each(|axis_data_el, operand_data_el| *axis_data_el = *operand_data_el)
            });
    }
}

impl<D> Debug for MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiStack")
            .field("data", &self.data.borrow())
            .field("axis", &self.axis)
            .field("operands", &self.operands.len())
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<D> Display for MultiStack<D>
where
    D: Dimension + RemoveAxis,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiStackBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    gradient: RefCell<Option<Tensor<D::Larger>>>,
    shape: D::Larger,
    overwrite: Cell<bool>,
    operands: Vec<Rc<dyn Gradient<Dim = D>>>,
    axis: usize,
}

impl<D> MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        operands: Vec<Rc<dyn Gradient<Dim = D>>>,
        axis: usize,
        shape: D::Larger,
    ) -> Self {
        let gradient = RefCell::new(Some(Tensor::zeros(shape.clone())));
        let overwrite = Cell::new(true);

        Self {
            gradient,
            shape,
            overwrite,
            operands,
            axis,
        }
    }
}

impl<D> Gradient for MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    type Dim = D::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<D> Overwrite for MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<D> Backward for MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let (axis, grad) = (self.axis, &self.gradient.borrow());

        self.operands
            .iter()
            .zip(grad.as_ref().unwrap().axis_iter(Axis(axis)))
            .for_each(|(operand, grad_view)| {
                push_gradient(operand.as_ref(), &grad_view);
            });
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

impl<D> Debug for MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiStackBackward")
            .field("gradient", &self.gradient.borrow())
            .field("operands", &self.operands.len())
            .field("axis", &self.axis)
            .field("overwrite", &self.overwrite)
            .finish()
    }
}

impl<D> Display for MultiStackBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
