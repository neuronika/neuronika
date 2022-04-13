use std::rc::Rc;

use ndarray::{arr0, Array, Axis, Dimension, Ix0, RemoveAxis, Zip};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
    Reduction,
};

#[allow(clippy::upper_case_acronyms)]
pub(crate) struct NegativeLogLikelihood<D>
where
    D: Dimension + RemoveAxis,
{
    input_data: Shared<Array<f32, D>>,
    target_data: Shared<Array<f32, D::Smaller>>,
    data: Shared<Array<f32, Ix0>>,
    reduction: Reduction,
}

impl<D> NegativeLogLikelihood<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        input_data: Shared<Array<f32, D>>,
        target_data: Shared<Array<f32, D::Smaller>>,
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

impl<D> Forward for NegativeLogLikelihood<D>
where
    D: Dimension + RemoveAxis,
{
    fn forward(&self) {
        let (input_data, target_data) = (self.input_data.borrow(), self.target_data.borrow());

        *self.data.borrow_mut() = {
            let total_loss = input_data
                .outer_iter()
                .enumerate()
                .fold(0., |loss, (idx, logits)| {
                    loss + Zip::from(logits).and(&*target_data).fold(
                        0.,
                        |partial_loss, &logit, &target| {
                            partial_loss + ((target as usize == idx) as u8 as f32) * logit
                        },
                    )
                });

            match self.reduction {
                Reduction::Mean => arr0(-total_loss / input_data.len_of(Axis(0)) as f32),
                Reduction::Sum => arr0(-total_loss),
            }
        };
    }
}

#[allow(clippy::upper_case_acronyms)]
pub(crate) struct NegativeLogLikelihoodBackward<D>
where
    D: Dimension + RemoveAxis,
{
    target_data: Shared<Array<f32, D::Smaller>>,
    input_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<Ix0>>,
    reduction: Reduction,
}

impl<D> NegativeLogLikelihoodBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        target_data: Shared<Array<f32, D::Smaller>>,
        input_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<Ix0>>,
        reduction: Reduction,
    ) -> Self {
        Self {
            target_data,
            input_gradient,
            gradient,
            reduction,
        }
    }
}

impl<D> Backward for NegativeLogLikelihoodBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let mut input_gradient = self.input_gradient.borrow_mut();
        let gradient = self.gradient.borrow()[()];
        let target_data = self.target_data.borrow();

        let iter = input_gradient.outer_iter_mut().enumerate();

        match self.reduction {
            Reduction::Mean => {
                let n = target_data.len() as f32;
                iter.for_each(|(idx, gradient_channel)| {
                    Zip::from(gradient_channel)
                        .and(&*target_data)
                        .for_each(|grad_el, &target| {
                            *grad_el -= gradient * ((target as usize == idx) as u8 as f32) / n
                        })
                });
            }
            Reduction::Sum => {
                iter.for_each(|(idx, gradient_channel)| {
                    Zip::from(gradient_channel)
                        .and(&*target_data)
                        .for_each(|grad_el, &target| {
                            *grad_el -= gradient * (target as usize == idx) as u8 as f32
                        })
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
