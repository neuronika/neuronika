use rand::thread_rng;
use rand_distr::{Bernoulli, Distribution};

use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Eval, Forward, Gradient, Overwrite, Tensor,
};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::Zip;

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dropout ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Dropout<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    noise: RefCell<Tensor<T::Dim>>,
    distr: Bernoulli,
    p: f64,
    computed: Cell<bool>,
    train: Rc<Cell<bool>>,
}

impl<T: Data> Dropout<T> {
    pub fn new(operand: Rc<T>, p: f64, status: Rc<Cell<bool>>) -> Self {
        if !(0. ..=1.).contains(&p) {
            panic!(
                "error: dropout probability has to be between 0 and 1, but got {}.",
                p
            );
        }

        let (data, noise) = (
            RefCell::new(Tensor::zeros(operand.data().raw_dim())),
            RefCell::new(Tensor::zeros(operand.data().raw_dim())),
        );
        let distr = Bernoulli::new(1. - p).unwrap();

        Self {
            operand,
            data,
            noise,
            distr,
            p,
            computed: Cell::new(false),
            train: status,
        }
    }

    pub(crate) fn noise(&self) -> Ref<Tensor<T::Dim>> {
        self.noise.borrow()
    }

    pub(crate) fn status(&self) -> Rc<Cell<bool>> {
        self.train.clone()
    }
}

impl<T: Data> Forward for Dropout<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        if self.train.get() {
            let mut thread_rng = thread_rng();
            let (mut noise, distr, p) = (self.noise.borrow_mut(), &self.distr, &self.p);
            if (*p - 1.).abs() <= f64::EPSILON {
                Zip::from(&mut *self.data.borrow_mut()).for_each(|data_el| *data_el = 0.0);
            } else if *p <= f64::EPSILON {
                Zip::from(&mut *self.data.borrow_mut())
                    .and(&*self.operand.data())
                    .for_each(|data_el, operand_data_el| *data_el = *operand_data_el);
            } else {
                Zip::from(&mut *noise)
                    .for_each(|noise_el| *noise_el = distr.sample(&mut thread_rng) as i32 as f32);
                Zip::from(&mut *self.data.borrow_mut())
                    .and(&*self.operand.data())
                    .and(&*noise)
                    .for_each(|data_el, operand_data_el, noise_el| {
                        *data_el = (operand_data_el * noise_el) / (1. - *p as f32)
                    });
            }
        } else {
            self.data.borrow_mut().assign(&*self.operand.data());
        }
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<T: Data> Data for Dropout<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<T: Data> Eval for Dropout<T> {
    fn train(&self) {
        self.train.set(true);
    }

    fn eval(&self) {
        self.train.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DropoutBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<Dropout<U>>,
    p: f64,
    train: Rc<Cell<bool>>,
}

impl<T, U> DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(
        diff_operand: Rc<T>,
        no_diff_operand: Rc<Dropout<U>>,
        p: f64,
        forward_status: Rc<Cell<bool>>,
    ) -> DropoutBackward<T, U> {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            p,
            train: forward_status,
        }
    }
}

impl<T, U> Gradient for DropoutBackward<T, U>
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

impl<T, U> Overwrite for DropoutBackward<T, U>
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

impl<T, U> Backward for DropoutBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        if self.train.get() {
            let mut op_grad = self.diff_operand.gradient_mut();
            let grad = self.gradient();
            let p = &self.p;
            if (*p - 1.).abs() <= f64::EPSILON {
                if self.diff_operand.can_overwrite() {
                    Zip::from(&mut *op_grad).for_each(|op_grad_el| *op_grad_el = 0.);
                    self.diff_operand.set_overwrite(false);
                }
            } else if *p <= f64::EPSILON {
                let zip = Zip::from(&mut *op_grad).and(&*grad);
                if self.diff_operand.can_overwrite() {
                    zip.for_each(|op_grad_el, grad_el| *op_grad_el = *grad_el);
                    self.diff_operand.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad_el, grad_el| *op_grad_el += *grad_el);
                }
            } else {
                let noise = self.no_diff_operand.noise();
                let zip = Zip::from(&mut *op_grad).and(&*grad).and(&*noise);
                if self.diff_operand.can_overwrite() {
                    zip.for_each(|op_grad_el, grad_el, noise_el| *op_grad_el = *grad_el * noise_el);
                    self.diff_operand.set_overwrite(false);
                } else {
                    zip.for_each(|op_grad_el, grad_el, noise_el| {
                        *op_grad_el += *grad_el * noise_el
                    });
                }
            }
        } else {
            self.diff_operand.gradient_mut().assign(&*self.gradient());
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
            let node = Dropout::new(input, 0.5, Rc::new(Cell::new(true)));

            assert_eq!(*node.data(), Tensor::from_elem((3, 3), 0.));
            assert!(!node.was_computed());
        }

        #[test]
        #[should_panic(
            expected = "error: dropout probability has to be between 0 and 1, but got -0.5."
        )]
        fn creation_less_than_zero() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let _ = Dropout::new(input, -0.5, Rc::new(Cell::new(true)));
        }

        #[test]
        fn computation_was_computed_transition() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Dropout::new(input, 0.5, Rc::new(Cell::new(true)));

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
        fn forward_p_one() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Dropout::new(input.clone(), 1., Rc::new(Cell::new(true)));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor((3, 3), vec![0.; 9]));

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
            assert_almost_equals(&*node.data(), &new_tensor((3, 3), vec![0.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(&*node.data(), &new_tensor((3, 3), vec![0.; 9]));
        }

        #[test]
        fn forward_scaling() {
            let input = new_input((3, 3), vec![3.; 9]);
            let node = Dropout::new(input, 0.5, Rc::new(Cell::new(true)));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            node.data()
                .iter()
                .all(|el| *el <= f32::EPSILON || (el - 6.).abs() <= f32::EPSILON);
        }

        #[test]
        fn forward_p_zero() {
            let input = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
            let node = Dropout::new(input.clone(), 0., Rc::new(Cell::new(true)));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
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
                &new_tensor((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.reset_computation();
            node.forward();
            assert_almost_equals(
                &*node.data(),
                &new_tensor((3, 3), vec![2., 3., 4., 5., 6., 7., 8., 9., 10.]),
            );
        }
    }

    mod backward {
        use super::*;

        #[test]
        fn creation() {
            let node = DropoutBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                Rc::new(Dropout::new(
                    new_input((3, 3), vec![1.; 9]),
                    0.5,
                    Rc::new(Cell::new(true)),
                )),
                0.5,
                Rc::new(Cell::new(true)),
            );

            assert_eq!(*node.gradient(), Tensor::from_elem((3, 3), 0.));
            assert_eq!(*node.gradient_mut(), Tensor::from_elem((3, 3), 0.));
            assert!(node.can_overwrite());
        }

        #[test]
        fn computation_state_transition() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = DropoutBackward::new(
                input.clone(),
                Rc::new(Dropout::new(
                    new_input((3, 3), vec![1.; 9]),
                    0.5,
                    Rc::new(Cell::new(true)),
                )),
                0.5,
                Rc::new(Cell::new(true)),
            );

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
        fn backward_p_one() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = DropoutBackward::new(
                input.clone(),
                Rc::new(Dropout::new(
                    new_input((3, 3), vec![1.; 9]),
                    1.,
                    Rc::new(Cell::new(true)),
                )),
                1.,
                Rc::new(Cell::new(true)),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![0.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![0.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            input.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![0.; 9]));
        }

        #[test]
        fn backward_p_zero() {
            let input = new_backward_input((3, 3), vec![0.; 9]);
            let node = DropoutBackward::new(
                input.clone(),
                Rc::new(Dropout::new(
                    new_input((3, 3), vec![1.; 9]),
                    0.,
                    Rc::new(Cell::new(true)),
                )),
                0.,
                Rc::new(Cell::new(true)),
            );

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            *node.gradient_mut() = new_tensor((3, 3), vec![1.; 9]);
            assert_almost_equals(&*node.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![1.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Accumulation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![2.; 9]));

            // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Overwrite ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            input.set_overwrite(true);
            node.backward();
            assert_almost_equals(&*input.gradient(), &new_tensor((3, 3), vec![1.; 9]));
        }

        #[test]
        fn no_grad() {
            // DropoutBackward
            let node = DropoutBackward::new(
                new_backward_input((3, 3), vec![0.; 9]),
                Rc::new(Dropout::new(
                    new_input((3, 3), vec![0.; 9]),
                    0.5,
                    Rc::new(Cell::new(true)),
                )),
                0.5,
                Rc::new(Cell::new(true)),
            );

            node.no_grad();
            assert!(node.gradient.borrow().is_none());

            node.with_grad();
            assert_eq!(&*node.gradient(), Tensor::zeros(node.shape));
        }
    }
}
