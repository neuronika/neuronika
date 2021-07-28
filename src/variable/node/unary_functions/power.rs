use super::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Power ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct Power<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    exp: i32,
    computed: Cell<bool>,
}

impl<T: Data> Power<T> {
    pub fn new(operand: Rc<T>, exp: i32) -> Self {
        let data = Tensor::zeros(operand.data().raw_dim());

        Self {
            operand,
            data: RefCell::new(data),
            exp,
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for Power<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let exp = self.exp;
        Zip::from(&mut *self.data.borrow_mut())
            .and(&*self.operand.data())
            .for_each(|v, o| *v = o.powi(exp));
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Power<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PowerBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct PowerBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
    exp: i32,
}

impl<T, U> PowerBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>, exp: i32) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            exp,
        }
    }
}

impl<T, U> Gradient for PowerBackward<T, U>
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

impl<T, U> Overwrite for PowerBackward<T, U>
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

impl<T, U> Backward for PowerBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let op_data = self.no_diff_operand.data();
        let grad = self.gradient();
        let exp = self.exp;

        let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*op_data);
        if self.diff_operand.can_overwrite() {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el = grad_el * op_data_el.powi(exp - 1) * exp as f32
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|op_grad_el, grad_el, op_data_el| {
                *op_grad_el += grad_el * op_data_el.powi(exp - 1) * exp as f32
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
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Power::new(input, 2);

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        fn computation_was_computed_transition() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Power::new(input, 2);

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
            let node = Power::new(input.clone(), 3);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![1., 8., 27., 64., 125., 216., 343., 512., 729.]),
            );

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
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![1., 8., 27., 64., 125., 216., 343., 512., 729.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor(
                    (3, 3),
                    vec![8., 27., 64., 125., 216., 343., 512., 729., 1_000.],
                ),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = PowerBackward::new(
                new_backward_input(3, vec![0.; 3]),
                new_input(3, vec![1., 2., 3.]),
                3,
            );

            assert_eq!(*node.gradient(), Tensor::from_elem(3, 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem(3, 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = PowerBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]), 3);

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
            let node = PowerBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]), 3);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3., 12., 27.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![6., 24., 54.]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*diff.gradient(), &new_tensor(3, vec![3., 12., 27.]));
        }

        #[test]
        fn backward_negative_exp() {
            let diff = new_backward_input(3, vec![0.; 3]);
            let node = PowerBackward::new(diff.clone(), new_input(3, vec![1., 2., 3.]), -3);

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor(3, vec![1.; 3]);
            assert_almost_equals(&*node.gradient(), &new_tensor(3, vec![1.; 3]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![-3., -0.1875, -0.037037]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![-6., -0.375, -0.074075]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Third Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            diff.set_overwrite(true);
            node.backward();
            assert_almost_equals(
                &*diff.gradient(),
                &new_tensor(3, vec![-3., -0.1875, -0.037037]),
            );
        }
    }
}
