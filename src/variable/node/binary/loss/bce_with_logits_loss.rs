use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Reduction,
    Tensor,
};
use ndarray::{Ix1, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BCEWithLogitsLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::upper_case_acronyms)]
pub struct BCEWithLogitsLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    state: Cell<bool>,
}

impl<T, U> BCEWithLogitsLoss<T, U>
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
            state: Cell::new(false),
        }
    }
}

impl<T, U> Data for BCEWithLogitsLoss<T, U>
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

impl<T, U> Forward for BCEWithLogitsLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
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
                    .fold(0.0, |loss, input, target| {
                        let max = (-input).max(0.);
                        loss + (1. - target) * input
                            + max
                            + ((-max).exp() + (-input - max).exp()).ln()
                    });
            match self.reduction {
                Reduction::Mean => total_loss / input_data.len() as f32,
                Reduction::Sum => total_loss,
            }
        };
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        self.state.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BCEWithLogitsLossBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::upper_case_acronyms)]
pub struct BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    overwrite: Cell<bool>,
    diff_input: Rc<T>,
    input: Rc<U>,
    target: Rc<V>,
    reduction: Reduction,
}

impl<T, U, V> BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    pub(crate) fn new(
        diff_input: Rc<T>,
        input: Rc<U>,
        target: Rc<V>,
        reduction: Reduction,
    ) -> Self {
        Self {
            diff_input,
            input,
            target,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            reduction,
            overwrite: Cell::new(false),
        }
    }
}

impl<T, U, V> Gradient for BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U, V> Overwrite for BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U, V> Backward for BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let (mut operand_gradient, gradient, input_data, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient(),
                self.input.data(),
                self.target.data(),
            )
        };

        let zip = Zip::from(&mut *operand_gradient)
            .and_broadcast(&*gradient)
            .and(&*input_data)
            .and(&*target_data);

        match self.reduction {
            Reduction::Mean => {
                let n = input_data.len() as f32;
                if self.diff_input.can_overwrite() {
                    zip.for_each(|op_grad, grad, input, target| {
                        let input_sigmoid = 1.0 / (1.0 + (-input).exp());
                        *op_grad = (input_sigmoid - target) * grad / n
                    });
                    self.diff_input.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad, grad, input, target| {
                        let input_sigmoid = 1.0 / (1.0 + (-input).exp());
                        *op_grad += (input_sigmoid - target) * grad / n
                    });
                }
            }
            Reduction::Sum => {
                if self.diff_input.can_overwrite() {
                    zip.for_each(|op_grad, grad, input, target| {
                        let input_sigmoid = 1.0 / (1.0 + (-input).exp());
                        *op_grad = (input_sigmoid - target) * grad
                    });
                    self.diff_input.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad, grad, input, target| {
                        let input_sigmoid = 1.0 / (1.0 + (-input).exp());
                        *op_grad += (input_sigmoid - target) * grad
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
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![8.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            BCEWithLogitsLossBackward::new(input_diff.clone(), input, target, Reduction::Mean);
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.0000050465264,
                    -0.0000018543667,
                    0.11111042,
                    0.11111086,
                    0.111111015,
                    -0.00000003973643,
                    0.1111111,
                    0.11111111,
                    0.,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 3),
                vec![
                    -0.0000050465264,
                    -0.0000018543667,
                    0.11111042,
                    0.11111086,
                    0.111111015,
                    -0.00000003973643,
                    0.1111111,
                    0.11111111,
                    0.,
                ],
            ) * 2.),
        );
    }

    #[test]
    fn sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![72.0001]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            BCEWithLogitsLossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.00004541874,
                    -0.0000166893,
                    0.9999938,
                    0.99999774,
                    0.99999917,
                    -0.00000035762787,
                    0.9999999,
                    1.,
                    0.,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 3),
                vec![
                    -0.00004541874,
                    -0.0000166893,
                    0.9999938,
                    0.99999774,
                    0.99999917,
                    -0.00000035762787,
                    0.9999999,
                    1.,
                    0.,
                ],
            ) * 2.),
        );
    }

    #[test]
    fn no_grad() {
        // BCEWithLogitsLossBackward
        let node = BCEWithLogitsLossBackward::new(
            new_backward_input(3, vec![0.; 3]),
            new_input(3, vec![0.; 3]),
            new_input(3, vec![0.; 3]),
            Reduction::Mean,
        );

        node.no_grad();
        assert!(node.gradient.borrow().is_none());

        node.with_grad();
        assert_eq!(&*node.gradient(), Tensor::zeros(1));
    }
}
