use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Tensor,
};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::Zip;

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sigmoid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Sigmoid<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    computed: Cell<bool>,
}

impl<T: Data> Sigmoid<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(operand.data().raw_dim()));

        Self {
            operand,
            data,
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Sigmoid<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand.data())
            .for_each(|v, o| *v = 1.0 / (1.0 + (-*o).exp()));
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Sigmoid<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SigmoidBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
}

impl<T, U> SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
        }
    }
}

impl<T, U> Gradient for SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for SigmoidBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, data_el| {
                *op_grad_el = *grad_el * *data_el * (1.0 - *data_el)
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, data_el| {
                *op_grad_el += *grad_el * *data_el * (1.0 - *data_el)
            });
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
mod test {

    use super::*;
    mod forward {
        use super::*;

        #[test]
        fn creation() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Sigmoid::new(input);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Sigmoid::new(input);

            node.forward();
            assert!(node.was_computed());

            node.forward();
            assert!(node.was_computed());

            node.reset_computation();
            assert!(!node.was_computed());

            node.reset_computation();
            assert!(!node.was_computed());
        }

        #[test]
        fn forward() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Sigmoid::new(input.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.01799, 0.04743, 0.1192, 0.26894, 0.5, 0.73106, 0.8808, 0.95257, 0.98201,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.01799, 0.04743, 0.1192, 0.26894, 0.5, 0.73106, 0.8808, 0.95257, 0.98201,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.04743, 0.1192, 0.26894, 0.5, 0.73106, 0.8808, 0.95257, 0.98201, 0.993307,
                    ],
                ),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = SigmoidBackward::new(
                new_backward_input(3, vec![0.; 3]),
                Rc::new(Sigmoid::new(new_input(3, vec![1., 2., 3.]))),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = SigmoidBackward::new(
                diff.clone(),
                Rc::new(Sigmoid::new(new_input(3, vec![1., 2., 3.]))),
            );

            node.backward();
            assert!(node.can_overwrite());
            assert!(!diff.can_overwrite());

            node.backward();
            assert!(node.can_overwrite());
            assert!(!diff.can_overwrite());

            diff.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(diff.can_overwrite());

            diff.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(diff.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(diff.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(diff.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!diff.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!diff.can_overwrite());
        }

        #[test]
        fn backward() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let not_diff = Rc::new(Sigmoid::new(new_input(3, vec![1., 2., 3.])));
            not_diff.forward();
            let node = SigmoidBackward::new(diff.clone(), not_diff);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.1966, 0.105, 0.0452]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.3932, 0.21, 0.0904]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![0.1966, 0.105, 0.0452]),
            );
        }

        #[test]
        fn no_grad() {
            // SigmoidBackward
            let node = SigmoidBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                new_input((3, 3), vec![0.; 9]),
            );

            node.no_grad();
            assert!(node.gradient.borrow().is_none());

            node.with_grad();
            assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
        }
    }
}
