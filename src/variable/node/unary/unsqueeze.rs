use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient, Overwrite,
    Tensor,
};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Axis, Dimension, Zip};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Unsqueeze ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Unsqueeze<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<<<T as Data>::Dim as Dimension>::Larger>>,
    axis: usize,
    computed: Cell<bool>,
}

impl<T: Data> Unsqueeze<T> {
    pub fn new(operand: Rc<T>, axis: usize) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.insert_axis(Axis(axis))));

        Self {
            operand,
            data,
            axis,
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Unsqueeze<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let mut data = self.data.borrow_mut();
        let mut unsqueezed = data
            .axis_iter_mut(Axis(self.axis))
            .next()
            .unwrap()
            .into_dimensionality::<T::Dim>()
            .unwrap();
        let operand_data = self.operand.data();
        Zip::from(&mut unsqueezed)
            .and(&*operand_data)
            .for_each(|unsqueezed_el, operand_data_el| *unsqueezed_el = *operand_data_el);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Unsqueeze<T> {
    type Dim = <T::Dim as Dimension>::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ UnsqueezeBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct UnsqueezeBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<<T::Dim as Dimension>::Larger>>>,
    shape: <T::Dim as Dimension>::Larger,
    overwrite: Cell<bool>,
    operand: Rc<T>,
    axis: usize,
}

impl<T: Gradient + Overwrite> UnsqueezeBackward<T> {
    pub fn new(operand: Rc<T>, axis: usize) -> Self {
        let gradient = Tensor::zeros(operand.gradient().raw_dim().insert_axis(Axis(axis)));
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            operand,
            axis,
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for UnsqueezeBackward<T> {
    type Dim = <T::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for UnsqueezeBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for UnsqueezeBackward<T> {
    fn backward(&self) {
        push_gradient(
            &*self.operand,
            self.gradient()
                .axis_iter(Axis(self.axis))
                .next()
                .unwrap()
                .into_dimensionality::<T::Dim>()
                .unwrap(),
        );
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
            let node = Unsqueeze::new(input, 0);

            assert_eq!(*node.data(), Tensor::from_elem((1, 3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Unsqueeze::new(input, 0);

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
        #[should_panic]
        fn fail() {
            Unsqueeze::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                3,
            );
        }

        #[test]
        fn forward_rows() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Unsqueeze::new(input.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((1, 3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
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
                &new_tensor((1, 3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((1, 3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );
        }

        #[test]
        fn forward_columns() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Unsqueeze::new(input.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 1, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
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
                &new_tensor((3, 1, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 1, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );
        }

        #[test]
        fn forward_depths() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Unsqueeze::new(input.clone(), 2);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3, 1), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
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
                &new_tensor((3, 3, 1), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3, 1), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = UnsqueezeBackward::new(new_backward_input((4, 3), vec![0.; 12]), 0);

            assert_eq!(*node.gradient(), Tensor::from_elem((1, 4, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((1, 4, 3), 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = UnsqueezeBackward::new(diff.clone(), 0);

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
        fn backward_rows() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = UnsqueezeBackward::new(diff.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((1, 4, 3), vec![1.; 12]);
            assert_almost_equals(&*node.gradient(), &new_tensor((1, 4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_columns() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = UnsqueezeBackward::new(diff.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 1, 3), vec![1.; 12]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 1, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_depths() {
            let diff = new_backward_input((4, 3), vec![0.; 12]);
            let node = UnsqueezeBackward::new(diff.clone(), 2);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 3, 1), vec![1.; 12]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 3, 1), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }
    }
}
