use super::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Negation<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    computed: Cell<bool>,
}

impl<T: Data> Negation<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let data = Tensor::zeros(operand.data().raw_dim());

        Self {
            operand,
            data: RefCell::new(data),
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Negation<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand.data())
            .for_each(|v, o| *v = -o);
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Negation<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NegationBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct NegationBackward<T: Gradient + Overwrite> {
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    operand: Rc<T>,
}

impl<T: Gradient + Overwrite> NegationBackward<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            operand,
        }
    }
}

impl<T: Gradient + Overwrite> Gradient for NegationBackward<T> {
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: Gradient + Overwrite> Overwrite for NegationBackward<T> {
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: Gradient + Overwrite> Backward for NegationBackward<T> {
    fn backward(&self) {
        push_gradient(&*self.operand, &-(&*self.gradient()));
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
            let node = Negation::new(input);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let input = new_input((3, 3), vec![-4., -3., -2., -1., 0., 1., 2., 3., 4.]);
            let node = Negation::new(input);

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
            let node = Negation::new(input.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
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
                &new_tensor((3, 3), vec![4., 3., 2., 1., 0., -1., -2., -3., -4.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![3., 2., 1., 0., -1., -2., -3., -4., -5.]),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = NegationBackward::new(new_backward_input((3, 3), vec![0.; 9]));

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = NegationBackward::new(input.clone());

            node.backward();
            assert!(node.can_overwrite());
            assert!(!input.can_overwrite());

            node.backward();
            assert!(node.can_overwrite());
            assert!(!input.can_overwrite());

            input.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(input.can_overwrite());

            input.set_overwrite(true);
            assert!(node.can_overwrite());
            assert!(input.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(input.can_overwrite());

            node.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(input.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!input.can_overwrite());

            node.backward();
            assert!(!node.can_overwrite());
            assert!(!input.can_overwrite());

            input.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(!input.can_overwrite());

            input.set_overwrite(false);
            assert!(!node.can_overwrite());
            assert!(!input.can_overwrite());
        }

        #[test]
        fn backward() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = NegationBackward::new(input.clone());

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![-1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![-2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            input.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![-1.; 9]));
        }
    }
}
