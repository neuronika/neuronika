use super::{Backward, Forward, Reduction, SharedTensor, SwitchableTensor};
use ndarray::{arr0, Dimension, Ix0, Zip};
use std::rc::Rc;

#[allow(clippy::upper_case_acronyms)]
pub struct MSELoss<D>
where
    D: Dimension,
{
    input_data: SharedTensor<D>,
    target_data: SharedTensor<D>,
    data: SharedTensor<Ix0>,
    reduction: Reduction,
}

impl<D> MSELoss<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input_data: SharedTensor<D>,
        target_data: SharedTensor<D>,
        data: SharedTensor<Ix0>,
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

impl<D> Forward for MSELoss<D>
where
    D: Dimension,
{
    fn forward(&self) {
        let (input_data, target_data) = (self.input_data.borrow(), self.target_data.borrow());
        *self.data.borrow_mut() = {
            let total_loss = Zip::from(&*input_data)
                .and(&*target_data)
                .fold(0., |loss, &input, &target| loss + (input - target).powi(2));

            match self.reduction {
                Reduction::Mean => arr0(total_loss / input_data.len() as f32),
                Reduction::Sum => arr0(total_loss),
            }
        };
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct MSELossBackward<D>
where
    D: Dimension,
{
    input_data: SharedTensor<D>,
    target_data: SharedTensor<D>,
    input_gradient: Rc<SwitchableTensor<D>>,
    gradient: Rc<SwitchableTensor<Ix0>>,
    reduction: Reduction,
}

impl<D> MSELossBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input_data: SharedTensor<D>,
        target_data: SharedTensor<D>,
        input_gradient: Rc<SwitchableTensor<D>>,
        gradient: Rc<SwitchableTensor<Ix0>>,
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

impl<D> Backward for MSELossBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut input_gradient = self.input_gradient.array_mut();
        let gradient = self.gradient.array();
        let input_data = self.input_data.borrow();
        let target_data = self.target_data.borrow();
        let zip = Zip::from(&mut *input_gradient)
            .and_broadcast(&*gradient)
            .and(&*input_data)
            .and(&*target_data);

        match self.reduction {
            Reduction::Mean => {
                let n = input_data.len() as f32;
                zip.for_each(|op_grad, &grad, &input, &target| {
                    *op_grad += (2. * (input - target)) * grad / n
                });
            }
            Reduction::Sum => {
                zip.for_each(|op_grad, &grad, &input, &target| {
                    *op_grad += (2. * (input - target)) * grad
                });
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
