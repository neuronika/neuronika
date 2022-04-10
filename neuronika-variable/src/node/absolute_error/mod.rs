use std::rc::Rc;

use ndarray::{arr0, Array, Dimension, Ix0, Zip};

use crate::{gradient::Gradient, utils::Shared, Reduction};

use super::{Backward, Forward};

pub struct AbsoluteError<D>
where
    D: Dimension,
{
    input_data: Shared<Array<f32, D>>,
    target_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, Ix0>>,
    reduction: Reduction,
}

impl<D> AbsoluteError<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input_data: Shared<Array<f32, D>>,
        target_data: Shared<Array<f32, D>>,
        data: Shared<Array<f32, Ix0>>,
        reduction: Reduction,
    ) -> Self {
        debug_assert_eq!(input_data.borrow().shape(), target_data.borrow().shape());

        Self {
            input_data,
            target_data,
            data,
            reduction,
        }
    }
}

impl<D> Forward for AbsoluteError<D>
where
    D: Dimension,
{
    fn forward(&self) {
        let input_data = self.input_data.borrow();
        *self.data.borrow_mut() = {
            let total_loss = Zip::from(&*input_data)
                .and(&*self.target_data.borrow())
                .fold(0., |loss, &input, &target| loss + (input - target).abs());

            match self.reduction {
                Reduction::Mean => arr0(total_loss / input_data.len() as f32),
                Reduction::Sum => arr0(total_loss),
            }
        };
    }
}

pub struct AbsoluteErrorBackward<D>
where
    D: Dimension,
{
    input_data: Shared<Array<f32, D>>,
    target_data: Shared<Array<f32, D>>,
    input_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<Ix0>>,
    reduction: Reduction,
}

impl<D> AbsoluteErrorBackward<D>
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
        debug_assert_eq!(input_data.borrow().shape(), target_data.borrow().shape());
        debug_assert_eq!(target_data.borrow().shape(), input_gradient.shape().slice());

        Self {
            input_data,
            target_data,
            input_gradient,
            gradient,
            reduction,
        }
    }
}

impl<D> Backward for AbsoluteErrorBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut input_gradient = self.input_gradient.borrow_mut();
        let gradient = self.gradient.borrow();
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
                    let diff = input - target;
                    *op_grad += ((diff != 0.) as u8 as f32) * (diff.signum() * grad / n);
                });
            }
            Reduction::Sum => {
                zip.for_each(|op_grad, &grad, &input, &target| {
                    let diff = input - target;
                    *op_grad += ((diff != 0.) as u8 as f32) * (diff.signum() * grad)
                });
            }
        }
    }
}

#[cfg(test)]
mod test;
