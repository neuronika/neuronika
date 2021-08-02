use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient, Overwrite,
    Tensor,
};
use ndarray::{stack, Axis, Dimension, RemoveAxis, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
    data: RefCell<Tensor<<Lhs::Dim as Dimension>::Larger>>,
    computed: Cell<bool>,
}

impl<Lhs, Rhs> Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let data = RefCell::new(
            stack(
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

impl<Lhs, Rhs> Data for Stack<Lhs, Rhs>
where
    Lhs: Data<Dim = Rhs::Dim>,
    Rhs: Data,
    Lhs::Dim: RemoveAxis,
{
    type Dim = <Lhs::Dim as Dimension>::Larger;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<Lhs, Rhs> Forward for Stack<Lhs, Rhs>
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
        let mut subview_iter = data.axis_iter_mut(Axis(axis));

        let mut subview_left = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<Lhs::Dim>()
            .unwrap();
        Zip::from(&*lhs_data)
            .and(&mut subview_left)
            .for_each(|single_el, fused_el| *fused_el = *single_el);

        let mut subview_right = subview_iter
            .next()
            .unwrap()
            .into_dimensionality::<Rhs::Dim>()
            .unwrap();
        Zip::from(&*rhs_data)
            .and(&mut subview_right)
            .for_each(|single_el, fused_el| *fused_el = *single_el);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StackBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct StackBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<<Lhs::Dim as Dimension>::Larger>>>,
    shape: <Lhs::Dim as Dimension>::Larger,
    overwrite: Cell<bool>,
    left: Rc<Lhs>,
    right: Rc<Rhs>,
    axis: usize,
}

impl<Lhs, Rhs> StackBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(left: Rc<Lhs>, right: Rc<Rhs>, axis: usize) -> Self {
        let gradient = stack(
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

impl<Lhs, Rhs> Gradient for StackBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    type Dim = <Lhs::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<Lhs, Rhs> Overwrite for StackBackward<Lhs, Rhs>
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

impl<Lhs, Rhs> Backward for StackBackward<Lhs, Rhs>
where
    Lhs: Gradient + Overwrite,
    Rhs: Gradient<Dim = Lhs::Dim> + Overwrite,
    Lhs::Dim: RemoveAxis,
{
    fn backward(&self) {
        let gradient = self.gradient();
        let mut subviews = gradient.axis_iter(Axis(self.axis));
        push_gradient(
            &*self.left,
            subviews
                .next()
                .unwrap()
                .into_dimensionality::<Lhs::Dim>()
                .unwrap(),
        );
        push_gradient(
            &*self.right,
            subviews
                .next()
                .unwrap()
                .into_dimensionality::<Rhs::Dim>()
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StackBackwardLeft ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct StackBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<<T::Dim as Dimension>::Larger>>>,
    shape: <T::Dim as Dimension>::Larger,
    overwrite: Cell<bool>,
    left: Rc<T>,
    axis: usize,
}

impl<T> StackBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    pub fn new<U: Data<Dim = T::Dim>>(left: Rc<T>, right: Rc<U>, axis: usize) -> Self {
        let gradient = stack(Axis(axis), &[left.gradient().view(), right.data().view()]).unwrap();
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

impl<T> Gradient for StackBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    type Dim = <T::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T> Overwrite for StackBackwardLeft<T>
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

impl<T> Backward for StackBackwardLeft<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn backward(&self) {
        push_gradient(
            &*self.left,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StackBackwardRight ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct StackBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    gradient: RefCell<Option<Tensor<<T::Dim as Dimension>::Larger>>>,
    shape: <T::Dim as Dimension>::Larger,
    overwrite: Cell<bool>,
    right: Rc<T>,
    axis: usize,
}

impl<T> StackBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    pub fn new<U: Data<Dim = T::Dim>>(left: Rc<U>, right: Rc<T>, axis: usize) -> Self {
        let gradient = stack(Axis(axis), &[left.data().view(), right.gradient().view()]).unwrap();
        let shape = gradient.raw_dim();

        Self {
            gradient: RefCell::new(Some(gradient)),
            shape,
            overwrite: Cell::new(true),
            right,
            axis,
        }
    }
}

impl<T> Gradient for StackBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    type Dim = <T::Dim as Dimension>::Larger;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T> Overwrite for StackBackwardRight<T>
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

impl<T> Backward for StackBackwardRight<T>
where
    T: Gradient + Overwrite,
    T::Dim: RemoveAxis,
{
    fn backward(&self) {
        push_gradient(
            &*self.right,
            self.gradient()
                .axis_iter(Axis(self.axis))
                .nth(1)
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
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![0.; 9]);
            let node = Stack::new(left, right, 0);

            assert_eq!(*node.data(), Tensor::from_elem((2, 3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![0.; 9]);
            let node = Stack::new(left, right, 0);

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
            Stack::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                new_input((3, 2), vec![0.; 6]),
                0,
            );
        }

        #[test]
        #[should_panic]
        fn fail_by_columns() {
            Stack::new(
                new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]),
                new_input((2, 3), vec![0.; 6]),
                1,
            );
        }

        #[test]
        fn forward_rows() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![0.; 9]);
            let node = Stack::new(left.clone(), right, 0);

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

        #[test]
        fn forward_columns() {
            let left = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let right = new_input((3, 3), vec![0.; 9]);
            let node = Stack::new(left.clone(), right, 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 2, 3),
                    vec![
                        -4., -3., -2., 0., 0., 0., -1., 0., 1., 0., 0., 0., 2., 3., 4., 0., 0., 0.,
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
                    (3, 2, 3),
                    vec![
                        -4., -3., -2., 0., 0., 0., -1., 0., 1., 0., 0., 0., 2., 3., 4., 0., 0., 0.,
                    ],
                ),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 2, 3),
                    vec![
                        -3., -2., -1., 0., 0., 0., 0., 1., 2., 0., 0., 0., 3., 4., 5., 0., 0., 0.,
                    ],
                ),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = StackBackward::new(
                new_backward_input((4, 3), vec![0.; 12]),
                new_backward_input((4, 3), vec![0.; 12]),
                0,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((2, 4, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((2, 4, 3), 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackward::new(lhs.clone(), rhs.clone(), 0);

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
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackward::new(lhs.clone(), rhs.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_columns() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackward::new(lhs.clone(), rhs.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 2, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 2, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            lhs.set_overwrite(true);
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*lhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_left_rows() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackwardLeft::new(lhs.clone(), new_input((4, 3), vec![0.; 12]), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

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
        fn backward_left_columns() {
            let lhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackwardLeft::new(lhs.clone(), new_input((4, 3), vec![0.; 12]), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 2, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 2, 3), vec![1.; 24]));

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
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackwardRight::new(new_input((4, 3), vec![0.; 12]), rhs.clone(), 0);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((2, 4, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((2, 4, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }

        #[test]
        fn backward_right_columns() {
            let rhs = new_backward_input((4, 3), vec![0.; 12]);
            let node = StackBackwardRight::new(new_input((4, 3), vec![0.; 12]), rhs.clone(), 1);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((4, 2, 3), vec![1.; 24]);
            assert_almost_equals(&*node.gradient(), &new_tensor((4, 2, 3), vec![1.; 24]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![2.; 12]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            rhs.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*rhs.gradient(), &new_tensor((4, 3), vec![1.; 12]));
        }
    }
}
