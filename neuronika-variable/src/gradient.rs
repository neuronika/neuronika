use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

use ndarray::{Array, Dimension, ShapeBuilder};

pub(crate) trait NoGrad {
    fn no_grad(&self);

    fn with_grad(&self);
}

pub(crate) struct Gradient<T, D>
where
    D: Dimension,
{
    shape: D,
    array: RefCell<Option<T>>,
}

impl<T, D> Gradient<T, D>
where
    D: Dimension,
{
    pub(crate) fn borrow(&self) -> Ref<T> {
        Ref::map(self.array.borrow(), |option| {
            option.as_ref().expect("Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`")
        })
    }

    pub(crate) fn borrow_mut(&self) -> RefMut<T> {
        RefMut::map(self.array.borrow_mut(), |option| {
            option.as_mut().expect("Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`")
        })
    }

    pub(crate) fn shape(&self) -> D {
        self.shape.clone()
    }
}

impl<D> Gradient<Array<f32, D>, D>
where
    D: Dimension,
{
    pub(crate) fn ndarray_zeros<Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Self {
        let array = Array::zeros(shape);

        Self {
            shape: array.raw_dim(),
            array: RefCell::new(Some(array)),
        }
    }

    pub(crate) fn from_ndarray(array: Array<f32, D>) -> Self {
        let shape = array.raw_dim();
        let array = RefCell::new(Some(array));

        Self { shape, array }
    }
}

impl<D> NoGrad for Gradient<Array<f32, D>, D>
where
    D: Dimension,
{
    fn no_grad(&self) {
        *self.array.borrow_mut() = None;
    }

    fn with_grad(&self) {
        let mut option = self.array.borrow_mut();

        if option.is_none() {
            *option = Some(Array::zeros(self.shape.clone()))
        }
    }
}

pub(crate) struct BufferedGradient<T, D>
where
    D: Dimension,
{
    gradient: Rc<Gradient<T, D>>,
    buffer: RefCell<Option<T>>,
}

impl<T, D> BufferedGradient<T, D>
where
    D: Dimension,
{
    pub(crate) fn borrow(&self) -> Ref<T> {
        self.gradient.borrow()
    }

    #[cfg(test)]
    pub(crate) fn buffer(&self) -> Ref<T> {
        Ref::map(self.buffer.borrow(), |option| {
            option.as_ref().expect("Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`")
        })
    }

    pub(crate) fn buffer_mut(&self) -> RefMut<T> {
        RefMut::map(self.buffer.borrow_mut(), |option| {
            option.as_mut().expect("Trying to get a de-allocated gradient. Switch on the gradients first by using `.with_grad()`")
        })
    }

    pub(crate) fn shape(&self) -> D {
        self.gradient.shape()
    }
}

impl<D> BufferedGradient<Array<f32, D>, D>
where
    D: Dimension,
{
    pub(crate) fn from_ndarray(gradient: Rc<Gradient<Array<f32, D>, D>>) -> Self {
        let buffer = RefCell::new(Some(Array::zeros(gradient.shape())));

        Self { gradient, buffer }
    }
}

impl<D> NoGrad for BufferedGradient<Array<f32, D>, D>
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
            *option = Some(Array::zeros(self.shape()));
        }
    }
}
