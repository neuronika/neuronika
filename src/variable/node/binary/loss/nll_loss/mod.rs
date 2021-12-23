#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Reduction,
    Tensor,
};
use ndarray::{arr0, Axis, Dimension, IntoDimension, Ix0, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NLLLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[allow(clippy::upper_case_acronyms)]
pub struct NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix0>>,
    reduction: Reduction,
    computed: Cell<bool>,
}

impl<T, U> NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    pub(crate) fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(arr0(0.)),
            reduction,
            computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    type Dim = Ix0;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<T, U> Forward for NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }
        self.computed.set(true);
        let (mut loss_data, input_data, target_data) = {
            (
                self.data.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };
        *loss_data = {
            let total_loss = Zip::indexed(&*input_data)
                .and_broadcast(&target_data.view().insert_axis(Axis(1)))
                .fold(0.0, |loss, idx, log, target| {
                    if idx.into_dimension()[1] == *target as usize {
                        loss + log
                    } else {
                        loss + 0.
                    }
                });
            match self.reduction {
                Reduction::Mean => arr0(-total_loss / input_data.len_of(Axis(0)) as f32),
                Reduction::Sum => arr0(-total_loss),
            }
        };
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T, U> Debug for NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NLLLoss")
            .field("data", &self.data.borrow())
            .field("reduction", &self.reduction)
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<T, U> Display for NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NLLLossBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[allow(clippy::upper_case_acronyms)]
pub struct NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    diff_input: Rc<T>,
    target: Rc<U>,
    gradient: RefCell<Option<Tensor<Ix0>>>,
    reduction: Reduction,
    overwrite: Cell<bool>,
}

impl<T, U> NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    pub(crate) fn new(diff_input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            target,
            gradient: RefCell::new(Some(arr0(0.))),
            reduction,
            overwrite: Cell::new(true),
        }
    }
}

impl<T, U> Gradient for NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    type Dim = Ix0;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    fn backward(&self) {
        let (mut operand_gradient, gradient, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient(),
                self.target.data(),
            )
        };
        let zip = Zip::indexed(&mut *operand_gradient)
            .and_broadcast(&*gradient)
            .and_broadcast(target_data.view().insert_axis(Axis(1)));

        match self.reduction {
            Reduction::Mean => {
                let n = target_data.len() as f32;
                if self.diff_input.can_overwrite() {
                    zip.for_each(|idx, op_grad, grad, target| {
                        if idx.into_dimension().last_elem() == *target as usize {
                            *op_grad = grad * -1. / n
                        } else {
                            *op_grad = 0.;
                        }
                    });
                    self.diff_input.set_overwrite(false);
                } else {
                    zip.for_each(|idx, op_grad, grad, target| {
                        if idx.into_dimension().last_elem() == *target as usize {
                            *op_grad += grad * -1. / n
                        } else {
                            *op_grad += 0.;
                        }
                    });
                }
            }
            Reduction::Sum => {
                if self.diff_input.can_overwrite() {
                    zip.for_each(|idx, op_grad, grad, target| {
                        if idx.into_dimension().last_elem() == *target as usize {
                            *op_grad = grad * -1.
                        } else {
                            *op_grad = 0.
                        }
                    });
                    self.diff_input.set_overwrite(false);
                } else {
                    zip.for_each(|idx, op_grad, grad, target| {
                        if idx.into_dimension().last_elem() == *target as usize {
                            *op_grad += grad * -1.
                        } else {
                            *op_grad += 0.
                        }
                    });
                }
            }
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(arr0(0.));
    }
}

impl<T, U> Debug for NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NLLLossBackward")
            .field("gradient", &self.gradient.borrow())
            .field("reduction", &self.reduction)
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T, U> Display for NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
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
