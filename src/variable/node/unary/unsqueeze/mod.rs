#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, push_gradient, Backward, Data, Forward, Gradient, Overwrite,
    Tensor,
};
use ndarray::{Axis, Dimension, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ UNsqueeze ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

impl<T: Data> Debug for Unsqueeze<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Unsqueeze")
            .field("data", &self.data.borrow())
            .field("axis", &self.axis)
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<T: Data> Display for Unsqueeze<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ UnsqueezeBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

impl<T> Debug for UnsqueezeBackward<T>
where
    T: Gradient + Overwrite,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnsqueezeBackward")
            .field("gradient", &self.gradient.borrow())
            .field("axis", &self.axis)
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T> Display for UnsqueezeBackward<T>
where
    T: Gradient + Overwrite,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
