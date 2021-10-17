use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Reduction,
    Tensor,
};
use ndarray::{Axis, Ix1, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

#[allow(clippy::upper_case_acronyms)]
pub struct KLDivLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    computed: Cell<bool>,
}

impl<T, U> KLDivLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    pub(crate) fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for KLDivLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<T, U> Forward for KLDivLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
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
        loss_data[0] = {
            let total_loss =
                Zip::from(&*input_data)
                    .and(&*target_data)
                    .fold(0.0, |loss, log, target| {
                        if *target > 0. {
                            loss + target * (target.ln() - log)
                        } else {
                            loss + 0.
                        }
                    });
            match self.reduction {
                Reduction::Mean => total_loss / input_data.len_of(Axis(0)) as f32,
                Reduction::Sum => total_loss,
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

#[allow(clippy::upper_case_acronyms)]
pub struct KLDivLossBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    diff_input: Rc<T>,
    target: Rc<U>,
    gradient: RefCell<Option<Tensor<Ix1>>>,
    reduction: Reduction,
    overwrite: Cell<bool>,
}

impl<T, U> KLDivLossBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub(crate) fn new(diff_input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            target,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            reduction,
            overwrite: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for KLDivLossBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for KLDivLossBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for KLDivLossBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let (mut operand_gradient, gradient, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient(),
                self.target.data(),
            )
        };
        let zip = Zip::from(&mut *operand_gradient)
            .and_broadcast(&*gradient)
            .and(&*target_data);

        match self.reduction {
            Reduction::Mean => {
                let n = target_data.len_of(Axis(0)) as f32;
                if self.diff_input.can_overwrite() {
                    zip.for_each(|op_grad, grad, target| *op_grad = -target * grad / n);
                    self.diff_input.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad, grad, target| *op_grad += -target * grad / n);
                }
            }
            Reduction::Sum => {
                if self.diff_input.can_overwrite() {
                    zip.for_each(|op_grad, grad, target| *op_grad = -target * grad);
                    self.diff_input.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad, grad, target| *op_grad += -target * grad);
                }
            }
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
    }
}

#[cfg(test)]
mod test;
