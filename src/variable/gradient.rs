use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Array, Dimension};

pub(crate) trait NoGrad {
    fn no_grad(&self);

    fn with_grad(&self);
}

pub(crate) struct Gradient<D>
where
    D: Dimension,
{
    array: RefCell<Option<Array<f32, D>>>,
    shape: D,
}

impl<D> Gradient<D>
where
    D: Dimension,
{
    pub(crate) fn zeros(shape: D) -> Self {
        Self {
            array: RefCell::new(Some(Array::zeros(shape.clone()))),
            shape,
        }
    }

    pub(crate) fn from_ndarray(array: Array<f32, D>) -> Self {
        let shape = array.raw_dim();
        let array = RefCell::new(Some(array));

        Self { shape, array }
    }

    pub(crate) fn borrow(&self) -> Ref<Array<f32, D>> {
        Ref::map(self.array.borrow(), |option| {
            option.as_ref().expect("Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`")
        })
    }

    pub(crate) fn borrow_mut(&self) -> RefMut<Array<f32, D>> {
        RefMut::map(self.array.borrow_mut(), |option| {
            option.as_mut().expect("Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`")
        })
    }

    pub(crate) fn shape(&self) -> D {
        self.shape.clone()
    }
}

impl<D> NoGrad for Gradient<D>
where
    D: Dimension,
{
    fn no_grad(&self) {
        *self.array.borrow_mut() = None;
    }

    fn with_grad(&self) {
        let mut option = self.array.borrow_mut();

        if option.is_none() {
            let shape = self.shape.clone();

            *option = Some(Array::zeros(shape))
        }
    }
}

pub(crate) struct BufferedGradient<D>
where
    D: Dimension,
{
    gradient: Rc<Gradient<D>>,
    buffer: RefCell<Option<Array<f32, D>>>,
}

impl<D> BufferedGradient<D>
where
    D: Dimension,
{
    pub(crate) fn from_gradient(gradient: Rc<Gradient<D>>) -> Self {
        let buffer = RefCell::new(Some(Array::zeros(gradient.shape())));

        Self { gradient, buffer }
    }

    pub(crate) fn borrow(&self) -> Ref<Array<f32, D>> {
        self.gradient.borrow()
    }

    pub(crate) fn buffer_mut(&self) -> RefMut<Array<f32, D>> {
        RefMut::map(self.buffer.borrow_mut(), |option| {
            option.as_mut().expect("Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`")
        })
    }

    pub(crate) fn shape(&self) -> D {
        self.gradient.shape()
    }
}

impl<D> NoGrad for BufferedGradient<D>
where
    D: Dimension,
{
    fn no_grad(&self) {
        self.gradient.no_grad();
        *self.buffer.borrow_mut() = None;
    }

    fn with_grad(&self) {
        self.gradient.with_grad();

        let mut option = self.buffer.borrow_mut();
        if option.is_none() {
            let shape = self.shape();
            *option = Some(Array::zeros(shape));
        }
    }
}
