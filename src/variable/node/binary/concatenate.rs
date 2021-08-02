use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient, Overwrite,
    Tensor,
};
use ndarray::{concatenate, Axis, RemoveAxis, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    data: RefCell<Tensor<Lhs::Dim>>,
    computed: Cell<bool>,
}

impl<Lhs, Rhs> Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let data = RefCell::new(
            concatenate(
                Axis(axis),
                &[
                    Tensor::zeros(left.data().raw_dim()).view(),
                    Tensor::zeros(right.data().raw_dim()).view(),
                ],
            )
            .unwrap(),
        );

        Self {
            left,
            right,
            data,
            axis,
            computed: Cell::new(false),
        }
    }
}

impl<Lhs, Rhs> Data for Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    type Dim = Lhs::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for Concatenate<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let lhs_data = self.left.data();
        let rhs_data = self.right.data();
        let mut data = self.data.borrow_mut();
        let axis = self.axis;
        let (mut lhs_portion, mut rhs_portion) = data
            .view_mut()
            .split_at(Axis(axis), lhs_data.len_of(Axis(axis)));
        Zip::from(&*lhs_data)
            .and(&mut lhs_portion)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
        Zip::from(&*rhs_data)
            .and(&mut rhs_portion)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConcatenateBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<Lhs::Dim>>>,
    shape: Lhs::Dim,
    overwrite: Cell<bool>,
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
}

impl<Lhs, Rhs> ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let gradient = concatenate(
            Axis(axis),
            &[left.gradient().view(), right.gradient().view()],
        )
        .unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            left,
            right,
            axis,
        }
    }
}

impl<Lhs, Rhs> Gradient for ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    type Dim = Lhs::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<Lhs, Rhs> Overwrite for ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<Lhs, Rhs> Backward for ConcatenateBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let (lhs_part, rhs_part) = gradient.view().split_at(
            Axis(self.axis),
            self.left.gradient_mut().len_of(Axis(self.axis)),
        );

        push_gradient(&*self.left, lhs_part);
        push_gradient(&*self.right, rhs_part);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConcatenateBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    left: Rc<T>,
    axis: usize,
}

impl<T> ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    pub fn new<U: Data<Dim = T::Dim>>(left: Rc<T>, right: Rc<U>, axis: usize) -> Self {
        let gradient =
            concatenate(Axis(axis), &[left.gradient().view(), right.data().view()]).unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            left,
            axis,
        }
    }
}

impl<T> Gradient for ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T> Overwrite for ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T> Backward for ConcatenateBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let (lhs_part, _) = gradient.view().split_at(
            Axis(self.axis),
            self.left.gradient_mut().len_of(Axis(self.axis)),
        );

        push_gradient(&*self.left, lhs_part);
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConcatenateBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    offset: usize,
    right: Rc<T>,
    axis: usize,
}

impl<T> ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    pub fn new<U: Data<Dim = T::Dim>>(left: Rc<U>, right: Rc<T>, axis: usize) -> Self {
        let gradient =
            concatenate(Axis(axis), &[left.data().view(), right.gradient().view()]).unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            right,
            offset: left.data().len_of(Axis(axis)),
            axis,
        }
    }
}

impl<T> Gradient for ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T> Overwrite for ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T> Backward for ConcatenateBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let (_, rhs_part) = gradient.view().split_at(Axis(self.axis), self.offset);
        push_gradient(&*self.right, rhs_part);
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
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((2, 3), vec![1.; 6]);
            let node = Concatenate::new(left, right, 0);

            assert_eq!(*node.data(), Tensor::from_elem((5, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((2, 3), vec![1.; 6]);
            let node = Concatenate::new(left, right, 0);

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
        fn fail_by_rows() {
            Concatenate::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                new_input((3, 2), vec![1.; 6]),
                0,
            );
        }

        #[test]
        #[should_panic]
        fn fail_by_columns() {
            Concatenate::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                new_input((2, 3), vec![1.; 6]),
                1,
            );
        }

        #[test]
        fn forward_rows() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((2, 3), vec![1.; 6]);
            let node = Concatenate::new(left.clone(), right, 0);

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
                let mut data = left.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*left.data(),
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

        #[test]
        fn forward_columns() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 2), vec![1.; 6]);
            let node = Concatenate::new(left.clone(), right, 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 5),
                    vec![
                        -4., -3., -2., 1., 1., -1., 0., 1., 1., 1., 2., 3., 4., 1., 1.,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ No Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            {
                let mut data = left.data_mut();
                *data = &*data + &Tensor::from_elem(1, 1.);
            }
            assert_almost_equals(
                &*left.data(),
                &new_tensor((3, 3), vec![-3., -2., -1., 0., 1., 2., 3., 4., 5.]),
            );

            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 5),
                    vec![
                        -4., -3., -2., 1., 1., -1., 0., 1., 1., 1., 2., 3., 4., 1., 1.,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 5),
                    vec![
                        -3., -2., -1., 1., 1., 0., 1., 2., 1., 1., 3., 4., 5., 1., 1.,
                    ],
                ),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = ConcatenateBackward::new(
                new_backward_input((4, 3), vec![0.; 12]),
                new_backward_input((4, 2), vec![0.; 8]),
                1,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((4, 5), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((4, 5), 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 2), vec![0.; 8]);
            let node = ConcatenateBackward::new(lhs.clone(), rhs.clone(), 1);

            node.backward();
            assert!(node.can_overwrite());
            assert!(!lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            node.backward();
            assert!(node.can_overwrite());
            assert!(!lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            lhs.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            lhs.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            rhs.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(rhs.can_overwrite());

            rhs.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(rhs.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(rhs.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(lhs.can_overwrite());
            assert!(rhs.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!lhs.can_overwrite());
            assert!(!rhs.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!lhs.can_overwrite());
            assert!(!rhs.can_overwrite());
        }

        #[test]
        fn backward_rows() {
            let lhs = new_backward_input((3, 4), vec![0.; 12]);
            let rhs = new_backward_input((2, 4), vec![0.; 8]);
            let node = ConcatenateBackward::new(lhs.clone(), rhs.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![2.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![2.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));
        }

        #[test]
        fn backward_columns() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 2), vec![0.; 8]);
            let node = ConcatenateBackward::new(lhs.clone(), rhs.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 5), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 5), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![2.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));
        }

        #[test]
        fn backward_left_rows() {
            let lhs = new_backward_input((3, 4), vec![0.; 12]);
            let node = ConcatenateBackwardLeft::new(lhs.clone(), new_input((2, 4), vec![0.; 8]), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((3, 4), vec![1.; 12]));
        }

        #[test]
        fn backward_left_columns() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = ConcatenateBackwardLeft::new(lhs.clone(), new_input((4, 2), vec![0.; 8]), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 5), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 5), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_right_rows() {
            let rhs = new_backward_input((2, 4), vec![0.; 8]);
            let node =
                ConcatenateBackwardRight::new(new_input((3, 4), vec![0.; 12]), rhs.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((5, 4), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((5, 4), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![2.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((2, 4), vec![1.; 8]));
        }

        #[test]
        fn backward_right_columns() {
            let rhs = new_backward_input((4, 2), vec![0.; 8]);
            let node =
                ConcatenateBackwardRight::new(new_input((4, 3), vec![0.; 12]), rhs.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 5), vec![1.; 20]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 5), vec![1.; 20]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![2.; 8]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 2), vec![1.; 8]));
        }
    }
}
