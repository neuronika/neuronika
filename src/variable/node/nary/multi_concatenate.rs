use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient,
    GradientOverwrite, Overwrite, Tensor,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

use ndarray::{Axis, Dimension, Slice, Zip};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiConcatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiConcatenate<D: Dimension + 'static> {
    operands: Vec<Rc<dyn Data<Dim = D>>>,
    axis: usize,
    data: RefCell<Tensor<D>>,
    computed: Cell<bool>,
}

impl<D: Dimension + 'static> MultiConcatenate<D> {
    pub(crate) fn new(operands: Vec<Rc<dyn Data<Dim = D>>>, axis: usize, data: Tensor<D>) -> Self {
        let (data, computed) = (RefCell::new(data), Cell::new(false));

        Self {
            operands,
            axis,
            data,
            computed,
        }
    }
}

impl<D: Dimension> Data for MultiConcatenate<D> {
    type Dim = D;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<D: Dimension> Forward for MultiConcatenate<D> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (axis, mut offset, mut data) = (self.axis, 0, self.data.borrow_mut());

        self.operands.iter().for_each(|operand| {
            let operand_data = operand.data();
            let axis_len = operand_data.len_of(Axis(axis));
            let slice = Slice::from(offset..axis_len + offset);

            let view_mut = data.slice_axis_mut(Axis(axis), slice);
            Zip::from(view_mut)
                .and(&*operand_data)
                .for_each(|view_el, op_data_el| *view_el = *op_data_el);
            offset += axis_len;
        });
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiConcatenateBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MultiConcatenateBackward<D: Dimension> {
    gradient: RefCell<Option<Tensor<D>>>,
    shape: D,
    overwrite: Cell<bool>,
    operands: Vec<Rc<dyn GradientOverwrite<D>>>,
    axis: usize,
}

impl<D: Dimension> MultiConcatenateBackward<D> {
    pub(crate) fn new(operands: Vec<Rc<dyn GradientOverwrite<D>>>, axis: usize, shape: D) -> Self {
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

impl<D: Dimension> Gradient for MultiConcatenateBackward<D> {
    type Dim = D;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<D: Dimension> Overwrite for MultiConcatenateBackward<D> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<D: Dimension> Backward for MultiConcatenateBackward<D> {
    fn backward(&self) {
        let (axis, grad, mut offset) = (self.axis, &self.gradient.borrow(), 0);

        self.operands.iter().for_each(|operand| {
            let axis_len = operand.gradient().len_of(Axis(axis));

            let grad_view = grad
                .as_ref()
                .unwrap()
                .slice_axis(Axis(axis), Slice::from(offset..axis_len + offset));

            push_gradient(operand.as_ref(), &grad_view);
            offset += axis_len;
        });
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

#[cfg(test)]
mod test {
    use super::*;

    mod forward {
        use super::*;

        #[test]
        fn creation() {
            let first = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let second = new_input((3, 3), vec![0.; 9]);
            let node =
                MultiConcatenate::new(vec![first, second], 0, new_tensor((6, 3), vec![0.; 18]));

            assert_eq!(*node.data(), Tensor::from_elem((6, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let first = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let second = new_input((3, 3), vec![0.; 9]);
            let node =
                MultiConcatenate::new(vec![first, second], 0, new_tensor((6, 3), vec![0.; 18]));

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
            let second = new_input((2, 3), vec![1.; 6]);
            let node = MultiConcatenate::new(
                vec![first.clone(), second],
                0,
                new_tensor((5, 3), vec![0.; 15]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (5, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1.,
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
                    (5, 3),
                    vec![
                        -4., -3., -2., -1., 0., 1., 2., 3., 4., 1., 1., 1., 1., 1., 1.,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (5, 3),
                    vec![
                        -3., -2., -1., 0., 1., 2., 3., 4., 5., 1., 1., 1., 1., 1., 1.,
                    ],
                ),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = MultiConcatenateBackward::new(
                vec![
                    new_backward_input((4, 3), vec![0.; 12]),
                    new_backward_input((4, 2), vec![0.; 8]),
                ],
                1,
                ndarray::Dim([4, 5]),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((4, 5), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((4, 5), 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let first = new_backward_input((4, 3), vec![0.; 12]);
            let second = new_backward_input((4, 2), vec![0.; 8]);
            let node = MultiConcatenateBackward::new(
                vec![first.clone(), second.clone()],
                1,
                ndarray::Dim([4, 5]),
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
            let first = new_backward_input((3, 4), vec![0.; 12]);
            let second = new_backward_input((2, 4), vec![0.; 8]);
            let node = MultiConcatenateBackward::new(
                vec![first.clone(), second.clone()],
                0,
                ndarray::Dim([5, 4]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*first.gradient(), &new_tensor((3, 4), vec![1.; 12]));
            assert_almost_equals(&*second.gradient(), &new_tensor((2, 4), vec![1.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*first.gradient(), &new_tensor((3, 4), vec![2.; 12]));
            assert_almost_equals(&*second.gradient(), &new_tensor((2, 4), vec![2.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            first.set_overwrite(true);
            second.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*first.gradient(), &new_tensor((3, 4), vec![1.; 12]));
            assert_almost_equals(&*second.gradient(), &new_tensor((2, 4), vec![1.; 8]));
        }

        #[test]
        fn no_grad() {
            // MultiConcatenateBackward
            let node = MultiConcatenateBackward::new(
                vec![
                    new_backward_input((3, 3), vec![0.; 9]),
                    new_backward_input((3, 3), vec![0.; 9]),
                ],
                0,
                ndarray::Dim([6, 3]),
            );

            node.no_grad();
            assert!(node.gradient.borrow().is_none());

            node.with_grad();
            assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
        }
    }
}
