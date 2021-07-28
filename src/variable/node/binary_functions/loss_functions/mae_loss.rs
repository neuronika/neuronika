use super::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAELoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::upper_case_acronyms)]
pub struct MAELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    computed: Cell<bool>,
}

impl<T, U> MAELoss<T, U>
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
            computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for MAELoss<T, U>
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

impl<T, U> Forward for MAELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
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
            let total_loss = Zip::from(&*input_data)
                .and(&*target_data)
                .fold(0.0, |loss, input, target| loss + (input - target).abs());
            match self.reduction {
                Reduction::Mean => total_loss / input_data.len() as f32,
                Reduction::Sum => total_loss,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MAELossBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::upper_case_acronyms)]
pub struct MAELossBackward<T, U, V>
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

impl<T, U, V> MAELossBackward<T, U, V>
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

impl<T, U, V> Gradient for MAELossBackward<T, U, V>
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

impl<T, U, V> Overwrite for MAELossBackward<T, U, V>
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

impl<T, U, V> Backward for MAELossBackward<T, U, V>
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
                        let diff = input - target;
                        *op_grad = if diff != 0. {
                            diff.signum() * grad / n
                        } else {
                            0.
                        }
                    });
                    self.diff_input.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad, grad, input, target| {
                        let diff = input - target;
                        *op_grad += if diff != 0. {
                            diff.signum() * grad / n
                        } else {
                            0.
                        }
                    });
                }
            }
            Reduction::Sum => {
                if self.diff_input.can_overwrite() {
                    zip.for_each(|op_grad, grad, input, target| {
                        let diff = input - target;
                        *op_grad = if diff != 0. { diff.signum() * grad } else { 0. }
                    });
                    self.diff_input.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad, grad, input, target| {
                        let diff = input - target;
                        *op_grad += if diff != 0. { diff.signum() * grad } else { 0. }
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn mean() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MAELoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![9.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            MAELossBackward::new(input_diff.clone(), input, target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor((3, 3), vec![0.1111; 9]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor((3, 3), vec![0.2222; 9]),
        );
    }

    #[test]
    fn sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MAELoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![81.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward = MAELossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![2.; 9]));
    }
}
