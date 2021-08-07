use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Tensor,
};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Ix1, Zip};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Sum ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Sum<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<Ix1>>,
    computed: Cell<bool>,
}

impl<T: Data> Sum<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let data = RefCell::new(Tensor::zeros(1));

        Self {
            operand,
            data,
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Sum<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        self.data.borrow_mut()[0] = self.operand.data().sum();
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Sum<T> {
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SumBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SumBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<Ix1>>>,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T: Gradient + Overwrite> SumBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        Self {
            operand,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            overwrite: Cell::new(true),
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for SumBackward<T> {
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for SumBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for SumBackward<T> {
    fn backward(&self) {
        let mut op_grad = self.operand.gradient_mut();
        let grad = self.gradient();

        let zip = Zip::from(&mut *op_grad).and_broadcast(&*grad);
        if self.operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el| *op_grad_el = *grad_el);
            self.operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
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
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Sum::new(input);

            assert_eq!(*node.data(), Tensor::from_elem(1, 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Sum::new(input);

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
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Sum::new(input.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![45.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            {
                let mut data = input.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*input.data(),
                &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
            );

            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![45.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor(1, vec![54.]));
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = SumBackward::new(new_backward_input((10, 10), vec![0.; 100]));

            assert_eq!(*node.gradient(), Tensor::from_elem(1, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(1, 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input((10, 10), vec![0.; 100]);
            let node = SumBackward::new(diff.clone());

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
        #[allow(clippy::approx_constant)]
        fn backward() {
            let diff = new_backward_input((10, 10), vec![0.; 100]);
            let node = SumBackward::new(diff.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(1, vec![1.]);
            assert_almost_equals(&*node.gradient(), &new_tensor(1, vec![1.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![1.; 100]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![2.; 100]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((10, 10), vec![1.; 100]));
        }

        #[test]
        fn no_grad() {
            // SumBackward
            let node = SumBackward::new(new_backward_input((3, 3), vec![0.; 9]));

            node.no_grad();
            assert!(node.gradient.borrow().is_none());

            node.with_grad();
            assert_eq!(&*node.gradient(), Tensor::zeros(1));
        }
    }
}
