use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Tensor,
};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Axis, Zip};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Softmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Softmax<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    axis: usize,
    computed: Cell<bool>,
}

impl<T: Data> Softmax<T> {
    pub fn new(operand: Rc<T>, axis: usize) -> Self {
        let data = RefCell::new(Tensor::zeros(operand.data().raw_dim()));

        Self {
            operand,
            data,
            axis,
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Softmax<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let axis = self.axis;
        Zip::from(self.data.borrow_mut().lanes_mut(Axis(axis)))
            .and(self.operand.data().lanes(Axis(axis)))
            .for_each(|lane_v, lane_o| {
                let max = lane_o.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = &lane_o.map(|el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_v)
                    .and(num)
                    .for_each(|lane_v_el, num_el| *lane_v_el = *num_el / den);
            });
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Softmax<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SoftmaxBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct SoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
    axis: usize,
}

impl<T, U> SoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>, axis: usize) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            axis,
        }
    }
}

impl<T, U> Gradient for SoftmaxBackward<T, U>
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

impl<T, U> Overwrite for SoftmaxBackward<T, U>
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

impl<T, U> Backward for SoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();
        let axis = self.axis;
        let zip = Zip::from(op_grad.lanes_mut(Axis(axis)))
            .and(grad.lanes(Axis(axis)))
            .and(data.lanes(Axis(axis)));

        if self.diff_operand.can_overwrite() {
            zip.for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let sum = Zip::from(grad_lane)
                    .and(data_lane)
                    .fold(0., |acc, grad_el, data_el| acc + grad_el * data_el);
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, grad_el, data_el| {
                        *op_grad_el = data_el * (grad_el - sum)
                    })
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let sum = Zip::from(grad_lane)
                    .and(data_lane)
                    .fold(0., |acc, grad_el, data_el| acc + grad_el * data_el);
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, grad_el, data_el| {
                        *op_grad_el += data_el * (grad_el - sum)
                    })
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
            let node = Softmax::new(input, 0);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Softmax::new(input, 0);

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
        fn forward_rows() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Softmax::new(input.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330,
                        0.950330, 0.950330,
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
                        0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330,
                        0.950330, 0.950330,
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
                        0.002356, 0.002356, 0.002356, 0.047314, 0.047314, 0.047314, 0.950330,
                        0.950330, 0.950330,
                    ],
                ),
            );
        }

        #[test]
        fn forward_columns() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Softmax::new(input.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![
                        0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241, 0.090031,
                        0.244728, 0.665241,
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
                        0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241, 0.090031,
                        0.244728, 0.665241,
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
                        0.090031, 0.244728, 0.665241, 0.090031, 0.244728, 0.665241, 0.090031,
                        0.244728, 0.665241,
                    ],
                ),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let axis = 0;
            let node = SoftmaxBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                Rc::new(Softmax::new(
                    new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                    axis,
                )),
                axis,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let axis = 0;
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let node = SoftmaxBackward::new(
                diff.clone(),
                Rc::new(Softmax::new(
                    new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                    axis,
                )),
                axis,
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
        fn backward_rows() {
            let axis = 0;
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let not_diff = Rc::new(Softmax::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                axis,
            ));
            not_diff.forward();
            let node_b = SoftmaxBackward::new(diff.clone(), not_diff, axis);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node_b.gradient_mut() = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            assert_almost_equals(
                &*node_b.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node_b.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.01376, -0.01376, -0.01376, -0.13455, -0.13455, -0.13455, 0.148323,
                        0.148323, 0.148323,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node_b.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.02752, -0.02752, -0.02752, -0.2691, -0.2691, -0.2691, 0.296646,
                        0.296646, 0.296646,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node_b.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.01376, -0.01376, -0.01376, -0.13455, -0.13455, -0.13455, 0.148323,
                        0.148323, 0.148323,
                    ],
                ),
            );
        }

        #[test]
        fn backward_columns() {
            let axis = 1;
            let diff = new_backward_input((3, 3), vec![0.; 9]);
            let not_diff = Rc::new(Softmax::new(
                new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
                axis,
            ));
            not_diff.forward();
            let node = SoftmaxBackward::new(diff.clone(), not_diff, axis);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            assert_almost_equals(
                &*node.gradient(),
                &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.1418, -0.1408, 0.2826, -0.1418, -0.1408, 0.2826, -0.1418, -0.1408,
                        0.2826,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.2836, -0.2815, 0.5652, -0.2836, -0.2815, 0.5652, -0.2836, -0.2815,
                        0.5652,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(
                    (3, 3),
                    vec![
                        -0.1418, -0.1408, 0.2826, -0.1418, -0.1408, 0.2826, -0.1418, -0.1408,
                        0.2826,
                    ],
                ),
            );
        }

        #[test]
        fn no_grad() {
            // SoftmaxBackward
            let node = SoftmaxBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                new_input((3, 3), vec![0.; 9]),
                0,
            );

            node.no_grad();
            assert!(node.gradient.borrow().is_none());

            node.with_grad();
            assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
        }
    }
}
