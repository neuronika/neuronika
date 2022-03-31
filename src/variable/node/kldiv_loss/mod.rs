use super::{Backward, Forward, Reduction, SharedTensor, SwitchableTensor};
use ndarray::{arr0, Axis, Dimension, Ix0, Zip};
use std::rc::Rc;

#[allow(clippy::upper_case_acronyms)]
pub struct KLDivLoss<D>
where
    D: Dimension,
{
    input_data: SharedTensor<D>,
    target_data: SharedTensor<D>,
    data: SharedTensor<Ix0>,
    reduction: Reduction,
}

impl<D> KLDivLoss<D>
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

impl<D> Forward for KLDivLoss<D>
where
    D: Dimension,
{
    fn forward(&self) {
        let (input_data, target_data) = (self.input_data.borrow(), self.target_data.borrow());
        *self.data.borrow_mut() = {
            let total_loss =
                Zip::from(&*input_data)
                    .and(&*target_data)
                    .fold(0., |loss, &log, &target| {
                        loss + target * (target.ln() - log) * (target > 0.) as u8 as f32
                    });

            match self.reduction {
                Reduction::Mean => arr0(total_loss / input_data.len_of(Axis(0)) as f32),
                Reduction::Sum => arr0(total_loss),
            }
        };
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct KLDivLossBackward<D>
where
    D: Dimension,
{
    input_gradient: Rc<SwitchableTensor<D>>,
    target_data: SharedTensor<D>,
    gradient: Rc<SwitchableTensor<Ix0>>,
    reduction: Reduction,
}

impl<D> KLDivLossBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input_gradient: Rc<SwitchableTensor<D>>,
        target_data: SharedTensor<D>,
        gradient: Rc<SwitchableTensor<Ix0>>,
        reduction: Reduction,
    ) -> Self {
        Self {
            input_gradient,
            target_data,
            gradient,
            reduction,
        }
    }
}

impl<D> Backward for KLDivLossBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut input_gradient = self.input_gradient.array_mut();
        let gradient = self.gradient.array();
        let target_data = self.target_data.borrow();
        let zip = Zip::from(&mut *input_gradient)
            .and_broadcast(&*gradient)
            .and(&*target_data);

        match self.reduction {
            Reduction::Mean => {
                let n = target_data.len_of(Axis(0)) as f32;
                zip.for_each(|op_grad, &grad, &target| *op_grad += -target * grad / n);
            }
            Reduction::Sum => {
                zip.for_each(|op_grad, &grad, &target| *op_grad += -target * grad);
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
