use std::rc::Rc;

use ndarray::{arr0, Array, Dimension, Ix0, Zip};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
    Reduction,
};

#[allow(clippy::upper_case_acronyms)]
pub(crate) struct BCEWithLogits<D>
where
    D: Dimension,
{
    input_data: Shared<Array<f32, D>>,
    target_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, Ix0>>,
    reduction: Reduction,
}

impl<D> BCEWithLogits<D>
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

impl<D> Forward for BCEWithLogits<D>
where
    D: Dimension,
{
    fn forward(&self) {
        let input_data = self.input_data.borrow();
        let target_data = self.target_data.borrow();

        *self.data.borrow_mut() = {
            let total_loss =
                Zip::from(&*input_data)
                    .and(&*target_data)
                    .fold(0., |loss, &input, &target| {
                        let max = (-input).max(0.);
                        loss + (1. - target) * input
                            + max
                            + ((-max).exp() + (-input - max).exp()).ln()
                    });

            match self.reduction {
                Reduction::Mean => arr0(total_loss / input_data.len() as f32),
                Reduction::Sum => arr0(total_loss),
            }
        };
    }
}

#[allow(clippy::upper_case_acronyms)]
pub(crate) struct BCEWithLogitsBackward<D>
where
    D: Dimension,
{
    input_data: Shared<Array<f32, D>>,
    input_gradient: Rc<Gradient<Array<f32, D>, D>>,
    target_data: Shared<Array<f32, D>>,
    gradient: Rc<Gradient<Array<f32, Ix0>, Ix0>>,
    reduction: Reduction,
}

impl<D> BCEWithLogitsBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input_data: Shared<Array<f32, D>>,
        input_gradient: Rc<Gradient<Array<f32, D>, D>>,
        target_data: Shared<Array<f32, D>>,
        gradient: Rc<Gradient<Array<f32, Ix0>, Ix0>>,
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

impl<D> Backward for BCEWithLogitsBackward<D>
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
                    let input_sigmoid = 1. / (1. + (-input).exp());
                    *op_grad += (input_sigmoid - target) * grad / n
                });
            }
            Reduction::Sum => {
                zip.for_each(|op_grad, &grad, &input, &target| {
                    let input_sigmoid = 1. / (1. + (-input).exp());
                    *op_grad += (input_sigmoid - target) * grad
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
