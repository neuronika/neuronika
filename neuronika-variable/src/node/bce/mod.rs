use std::rc::Rc;

use ndarray::{arr0, Array, Dimension, Ix0, Zip};

use crate::{gradient::Gradient, utils::Shared, Reduction};

use super::{Backward, Forward};

pub(crate) struct BinaryCrossEntropy<D>
where
    D: Dimension,
{
    input_data: Shared<Array<f32, D>>,
    target_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, Ix0>>,
    reduction: Reduction,
}

impl<D> BinaryCrossEntropy<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input_data: Shared<Array<f32, D>>,
        target_data: Shared<Array<f32, D>>,
        data: Shared<Array<f32, Ix0>>,
        reduction: Reduction,
    ) -> Self {
        Self {
            input_data,
            target_data,
            data,
            reduction,
        }
    }
}

impl<D> Forward for BinaryCrossEntropy<D>
where
    D: Dimension,
{
    fn forward(&self) {
        const LOG_MIN: f32 = 100.;

        let input_data = self.input_data.borrow();
        *self.data.borrow_mut() = {
            let total_loss = Zip::from(&*input_data)
                .and(&*self.target_data.borrow())
                .fold(0., |loss, &input, &target| {
                    loss - target * input.ln().clamp(-LOG_MIN, f32::MAX)
                        + (target - 1.) * (1. - input).ln().clamp(-LOG_MIN, f32::MAX)
                });
            match self.reduction {
                Reduction::Mean => arr0(total_loss / input_data.len() as f32),
                Reduction::Sum => arr0(total_loss),
            }
        };
    }
}

pub(crate) struct BinaryCrossEntropyBackward<D>
where
    D: Dimension,
{
    input_data: Shared<Array<f32, D>>,
    target_data: Shared<Array<f32, D>>,
    input_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<Ix0>>,
    reduction: Reduction,
}

impl<D> BinaryCrossEntropyBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input_data: Shared<Array<f32, D>>,
        target_data: Shared<Array<f32, D>>,
        input_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<Ix0>>,
        reduction: Reduction,
    ) -> Self {
        Self {
            input_data,
            target_data,
            input_gradient,
            gradient,
            reduction,
        }
    }
}

impl<D> Backward for BinaryCrossEntropyBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut input_gradient = self.input_gradient.borrow_mut();
        let gradient = self.gradient.borrow();
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
                        (input - target) / ((1. - input) * input).max(f32::EPSILON) * grad / n;
                });
            }
            Reduction::Sum => {
                zip.for_each(|op_grad, &grad, &input, &target| {
                    *op_grad += (input - target) / ((1. - input) * input).max(f32::EPSILON) * grad;
                });
            }
        }
    }
}

#[cfg(test)]
mod test;
