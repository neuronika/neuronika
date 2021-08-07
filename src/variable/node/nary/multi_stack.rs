use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient,
    GradientOverwrite, Overwrite, Tensor,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

use ndarray::{Axis, Dimension, RemoveAxis, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiStack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiStack<D: Dimension + RemoveAxis + 'static> {
    operands: Vec<Rc<dyn Data<Dim = D>>>,
    axis: usize,
    data: RefCell<Tensor<D::Larger>>,
    computed: Cell<bool>,
}

impl<D: Dimension + RemoveAxis + 'static> MultiStack<D> {
    pub(crate) fn new(
        operands: Vec<Rc<dyn Data<Dim = D>>>,
        axis: usize,
        tensor: Tensor<D::Larger>,
    ) -> Self {
        let (data, computed) = (RefCell::new(tensor), Cell::new(false));

        Self {
            operands,
            axis,
            data,
            computed,
        }
    }
}

impl<D: Dimension + RemoveAxis> Data for MultiStack<D> {
    type Dim = D::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<D: Dimension + RemoveAxis> Forward for MultiStack<D> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (mut data, axis) = (self.data.borrow_mut(), self.axis);

        self.operands
            .iter()
            .zip(data.axis_iter_mut(Axis(axis)))
            .for_each(|(operand, axis_data)| {
                let operand_data = operand.data();
                Zip::from(&mut axis_data.into_dimensionality::<D>().unwrap())
                    .and(&*operand_data)
                    .for_each(|axis_data_el, operand_data_el| *axis_data_el = *operand_data_el)
            });
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiStackBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiStackBackward<D: Dimension + RemoveAxis> {
    gradient: RefCell<Option<Tensor<D::Larger>>>,
    shape: D::Larger,
    overwrite: Cell<bool>,
    operands: Vec<Rc<dyn GradientOverwrite<D>>>,
    axis: usize,
}

impl<D: Dimension + RemoveAxis> MultiStackBackward<D> {
    pub(crate) fn new(
        operands: Vec<Rc<dyn GradientOverwrite<D>>>,
        axis: usize,
        shape: D::Larger,
    ) -> Self {
        let gradient = RefCell::new(Some(Tensor::zeros(shape.clone())));
        let overwrite = Cell::new(true);

        Self {
            gradient,
            shape,
            overwrite,
            operands,
            axis,
        }
    }
}

impl<D: Dimension + RemoveAxis> Gradient for MultiStackBackward<D> {
    type Dim = D::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<D: Dimension + RemoveAxis> Overwrite for MultiStackBackward<D> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<D: Dimension + RemoveAxis> Backward for MultiStackBackward<D> {
    fn backward(&self) {
        let (axis, grad) = (self.axis, &self.gradient.borrow());

        self.operands
            .iter()
            .zip(grad.as_ref().unwrap().axis_iter(Axis(axis)))
            .for_each(|(operand, grad_view)| {
                push_gradient(operand.as_ref(), &grad_view);
            });
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
            let first = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let second = new_input((3, 3), vec![0.; 9]);
            let node = MultiStack::new(vec![first, second], 0, new_tensor((2, 3, 3), vec![0.; 18]));

            assert_eq!(*node.data(), Tensor::from_elem((2, 3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let first = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let second = new_input((3, 3), vec![0.; 9]);
            let node = MultiStack::new(vec![first, second], 0, new_tensor((2, 3, 3), vec![0.; 18]));

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
            let first = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let second = new_input((3, 3), vec![0.; 9]);
            let node = MultiStack::new(
                vec![first.clone(), second],
                0,
                new_tensor((2, 3, 3), vec![0.; 18]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 3, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            {
                let mut data = first.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*first.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 3, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (2, 3, 3),
                    vec![
                        -3., -2., -1., 0., 1., 2., 3., 4., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                    ],
                ),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = MultiStackBackward::new(
                vec![
                    new_backward_input((4, 3), vec![0.; 12]),
                    new_backward_input((4, 3), vec![0.; 12]),
                ],
                0,
                ndarray::Dim([2, 4, 3]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((2, 4, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((2, 4, 3), 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let first = new_backward_input((4, 3), vec![0.; 12]);
            let second = new_backward_input((4, 3), vec![0.; 12]);
            let node = MultiStackBackward::new(
                vec![first.clone(), second.clone()],
                0,
                ndarray::Dim([2, 4, 3]),
            );

            node.backward();
            assert!(node.can_overwrite());
            assert!(!first.can_overwrite());
            assert!(!second.can_overwrite());

            node.backward();
            assert!(node.can_overwrite());
            assert!(!first.can_overwrite());
            assert!(!second.can_overwrite());

            first.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(first.can_overwrite());
            assert!(!second.can_overwrite());

            first.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(first.can_overwrite());
            assert!(!second.can_overwrite());

            second.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(first.can_overwrite());
            assert!(second.can_overwrite());

            second.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(first.can_overwrite());
            assert!(second.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(first.can_overwrite());
            assert!(second.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(first.can_overwrite());
            assert!(second.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!first.can_overwrite());
            assert!(!second.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!first.can_overwrite());
            assert!(!second.can_overwrite());
        }

        #[test]
        fn backward() {
            let first = new_backward_input((4, 3), vec![0.; 12]);
            let second = new_backward_input((4, 3), vec![0.; 12]);
            let node = MultiStackBackward::new(
                vec![first.clone(), second.clone()],
                0,
                ndarray::Dim([2, 4, 3]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*first.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*second.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*first.gradient(), &new_tensor((4, 3), vec![2.; 12]));
            assert_almost_equals(&*second.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            first.set_overwrite(true);
            second.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*first.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*second.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn no_grad() {
            // MultiStackBackward
            let node = MultiStackBackward::new(
                vec![
                    new_backward_input((3, 3), vec![0.; 9]),
                    new_backward_input((3, 3), vec![0.; 9]),
                ],
                0,
                ndarray::Dim([2, 3, 3]),
            );

            node.no_grad();
            assert!(node.gradient.borrow().is_none());

            node.with_grad();
            assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
        }
    }
}
