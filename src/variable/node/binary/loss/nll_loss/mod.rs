#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, reduction::Reduction, Backward, Forward, Tensor};
use ndarray::{arr0, Axis, Dimension, Ix0, RemoveAxis, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[allow(clippy::upper_case_acronyms)]
pub struct NLLLoss<D>
where
    D: Dimension + RemoveAxis,
{
    input_data: Rc<RefCell<Tensor<D>>>,
    target_data: Rc<RefCell<Tensor<D::Smaller>>>,
    data: Rc<RefCell<Tensor<Ix0>>>,
    reduction: Reduction,
    computed: Cell<bool>,
}

impl<D> NLLLoss<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        input_data: Rc<RefCell<Tensor<D>>>,
        target_data: Rc<RefCell<Tensor<D::Smaller>>>,
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

impl<D> Forward for NLLLoss<D>
where
    D: Dimension + RemoveAxis,
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

        *data = {
            let total_loss =
                input_data
                    .outer_iter()
                    .enumerate()
                    .fold(0.0, |loss, (idx, logits)| {
                        loss + Zip::from(logits).and(&*target_data).fold(
                            0.0,
                            |partial_loss, logit, target| {
                                partial_loss + ((*target as usize == idx) as u8 as f32) * logit
                            },
                        )
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
        self.computed.set(false)
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct NLLLossBackward<D>
where
    D: Dimension + RemoveAxis,
{
    target_data: Rc<RefCell<Tensor<D::Smaller>>>,
    input_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<Ix0>>>>,
    reduction: Reduction,
}

impl<D> NLLLossBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        target_data: Rc<RefCell<Tensor<D::Smaller>>>,
        input_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<Ix0>>>>,
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

impl<D> Backward for NLLLossBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        let (mut input_gradient, gradient, target_data) = {
            (
                expect_tensor_mut(&self.input_gradient),
                expect_tensor(&self.gradient)[()],
                self.target_data.borrow(),
            )
        };

        let iter = input_gradient.outer_iter_mut().enumerate();

        match self.reduction {
            Reduction::Mean => {
                let n = target_data.len() as f32;
                iter.for_each(|(idx, gradient_channel)| {
                    Zip::from(gradient_channel)
                        .and(&*target_data)
                        .for_each(|grad_el, target| {
                            *grad_el -= gradient * ((*target as usize == idx) as u8 as f32) / n
                        })
                });
            }
            Reduction::Sum => {
                iter.for_each(|(idx, gradient_channel)| {
                    Zip::from(gradient_channel)
                        .and(&*target_data)
                        .for_each(|grad_el, target| {
                            *grad_el -= gradient * (*target as usize == idx) as u8 as f32
                        })
                });
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
