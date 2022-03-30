use super::{reduction::Reduction, Backward, Forward, OptionalTensor, Tensor};
use ndarray::{arr0, Dimension, Ix0, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[allow(clippy::upper_case_acronyms)]
pub struct BCELoss<D>
where
    D: Dimension,
{
    input_data: Rc<RefCell<Tensor<D>>>,
    target_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<Ix0>>>,
    reduction: Reduction,
    computed: Cell<bool>,
}

impl<D> BCELoss<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input_data: Rc<RefCell<Tensor<D>>>,
        target_data: Rc<RefCell<Tensor<D>>>,
        data: Rc<RefCell<Tensor<Ix0>>>,
        reduction: Reduction,
    ) -> Self {
        Self {
            input_data,
            target_data,
            data,
            reduction,
            computed: Cell::default(),
        }
    }
}

impl<D> Forward for BCELoss<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (mut data, input_data, target_data) = {
            (
                self.data.borrow_mut(),
                self.input_data.borrow(),
                self.target_data.borrow(),
            )
        };

        const MIN_LOG: f32 = -100.;
        *data = {
            let total_loss =
                Zip::from(&*input_data)
                    .and(&*target_data)
                    .fold(0.0, |loss, input, target| {
                        loss + (target * input.ln().clamp(MIN_LOG, f32::MAX)
                            + (1. - target) * (1. - input).ln().clamp(MIN_LOG, f32::MAX))
                    });
            match self.reduction {
                Reduction::Mean => arr0(-total_loss / input_data.len() as f32),
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

#[allow(clippy::upper_case_acronyms)]
pub struct BCELossBackward<D>
where
    D: Dimension,
{
    input_data: Rc<RefCell<Tensor<D>>>,
    input_gradient: Rc<OptionalTensor<D>>,
    target_data: Rc<RefCell<Tensor<D>>>,
    gradient: Rc<OptionalTensor<Ix0>>,
    reduction: Reduction,
}

impl<D> BCELossBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input_data: Rc<RefCell<Tensor<D>>>,
        input_gradient: Rc<OptionalTensor<D>>,
        target_data: Rc<RefCell<Tensor<D>>>,
        gradient: Rc<OptionalTensor<Ix0>>,
        reduction: Reduction,
    ) -> Self {
        Self {
            input_data,
            input_gradient,
            target_data,
            gradient,
            reduction,
        }
    }
}

impl<D> Backward for BCELossBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut input_gradient = self.input_gradient.content_mut();
        let gradient = self.gradient.content();
        let target_data = self.target_data.borrow();
        let input_data = self.input_data.borrow();
        let zip = Zip::from(&mut *input_gradient)
            .and_broadcast(&*gradient)
            .and(&*input_data)
            .and(&*target_data);

        match self.reduction {
            Reduction::Mean => {
                let n = input_data.len() as f32;
                zip.for_each(|op_grad, &grad, &input, &target| {
                    *op_grad +=
                        (input - target) / ((1. - input) * input).max(f32::EPSILON) * grad / n
                });
            }
            Reduction::Sum => {
                zip.for_each(|op_grad, &grad, &input, &target| {
                    *op_grad += (input - target) / ((1. - input) * input).max(f32::EPSILON) * grad
                });
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
