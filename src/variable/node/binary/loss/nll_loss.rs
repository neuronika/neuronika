use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Reduction,
    Tensor,
};
use ndarray::{Axis, Dimension, IntoDimension, Ix1, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NLLLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[allow(clippy::upper_case_acronyms)]
pub struct NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
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
            data: RefCell::new(Tensor::zeros(1)),
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
    type Dim = Ix1;

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
        loss_data[0] = {
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
                Reduction::Mean => -total_loss / input_data.len_of(Axis(0)) as f32,
                Reduction::Sum => -total_loss,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NLLLossBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::upper_case_acronyms)]
pub struct NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    diff_input: Rc<T>,
    target: Rc<U>,
    gradient: RefCell<Option<Tensor<Ix1>>>,
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
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            reduction,
            overwrite: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    type Dim = Ix1;

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
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn mean() {
        use crate::variable::node::LogSoftmax;

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input(3, vec![2., 0., 4.]);
        let input = Rc::new(LogSoftmax::new(
            new_input(
                (3, 5),
                vec![
                    0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0., 0.2, 0.5,
                ],
            ),
            1,
        ));
        input.forward();

        let loss = NLLLoss::new(input, target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![1.52222]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((3, 5), vec![0.; 15]);
        let loss_backward = NLLLossBackward::new(input_diff.clone(), target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 5),
                vec![
                    0.0000, 0.0000, -0.3333, 0.0000, 0.0000, -0.3333, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.3333,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 5),
                vec![
                    0.0000, 0.0000, -0.3333, 0.0000, 0.0000, -0.3333, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.3333,
                ],
            ) * 2.),
        );
    }

    #[test]
    fn sum() {
        use crate::variable::node::LogSoftmax;

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input(3, vec![2., 0., 4.]);
        let input = Rc::new(LogSoftmax::new(
            new_input(
                (3, 5),
                vec![
                    0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0., 0.2, 0.5,
                ],
            ),
            1,
        ));
        input.forward();

        let loss = NLLLoss::new(input, target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![4.56666]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((3, 5), vec![0.; 15]);
        let loss_backward = NLLLossBackward::new(input_diff.clone(), target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 5),
                vec![
                    0.0000, 0.0000, -1.0000, 0.0000, 0.0000, -1.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.0000,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 5),
                vec![
                    0.0000, 0.0000, -1.0000, 0.0000, 0.0000, -1.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.0000,
                ],
            ) * 2.),
        );
    }
}
