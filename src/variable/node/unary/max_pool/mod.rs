use std::cell::{Cell, Ref, RefCell, RefMut};
use std::fmt::{Debug, Display};
use std::ops::Range;
use std::rc::Rc;

use ndarray::{Dimension, IntoDimension, Ix, RemoveAxis, Slice, Zip};
use crate::{Var, VarDiff};

#[cfg(test)]
use super::{new_backward_input, new_input, new_tensor};
use super::{
    Backward, Cache, Data, expect_tensor, expect_tensor_mut, Forward, Gradient, Overwrite,
    Tensor,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPooling Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub trait MaxPooling<T> {
    type Output;

    fn max_pool(operand: T,pool_shape: &[usize], stride: &[usize]) -> Self::Output;
}

impl<T: ?Sized> MaxPooling<Self> for Var<T>
where
    T: Data,
    T::Dim: RemoveAxis,
    <T::Dim as Dimension>::Smaller: RemoveAxis,
{
    type Output = Var<MaxPool<T>>;

    fn max_pool(operand: Self, pool_shape: &[usize], stride: &[usize]) -> Self::Output {
        Var::from(
            MaxPool::new(
                operand.node,
                pool_shape,
                stride,
            ),
            operand.past,
        )
    }
}

impl<T: ?Sized, U: ?Sized> MaxPooling<Self> for VarDiff<U, T>
where
    T: Gradient,
    U: Data<Dim=T::Dim>,
    <T as Gradient>::Dim: RemoveAxis,
    <<T as Gradient>::Dim as Dimension>::Smaller: RemoveAxis,
{
    type Output = VarDiff<MaxPool<U>, MaxPoolBackward<T, U>>;

    fn max_pool(operand: Self, pool_shape: &[usize], stride: &[usize]) -> Self::Output {
        let var = Var::max_pool(
            operand.var,
            pool_shape,
            stride,
        );
        let node = MaxPoolBackward::new(
            operand.node,
            var.node.clone(),
            pool_shape,
            stride
        );
        VarDiff::from(node, operand.past, var)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPool ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MaxPool<T: ?Sized>
    where
        T: Data,
{
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    pool_shape: Vec<usize>,
    stride: Vec<usize>,
    computed: Cell<bool>,
}

impl<T: ?Sized> MaxPool<T>
    where
        T: Data,
{
    pub fn new(operand: Rc<T>, pool_shape: &[usize], stride: &[usize]) -> Self {
        let operand_ndims = operand.data().ndim() - 2;
        assert_eq!(
            operand_ndims,
            stride.len(),
            "error: invalid stride {:?} for {}d input.",
            stride,
            operand_ndims
        );
        assert_eq!(
            operand_ndims,
            pool_shape.len(),
            "error: invalid pool shape {:?} for {}d input.",
            pool_shape,
            operand_ndims
        );
        operand.data().shape()
            .iter()
            .skip(2)
            .zip(pool_shape.iter())
            .for_each(|(operand_dim, pool_dim)| {
                assert!(
                    operand_dim >= pool_dim,
                    "Pool shape {:?} doesn't fit in input shape {:?}.",
                    pool_shape,
                    operand.data().shape()
                )
            });

        let shape =
            operand.data().shape().iter()
                .enumerate()
                .filter(|(i, _)| *i < 2)
                .map(|(_, e)| *e)
                .chain(
                    Zip::from(&operand.data().shape().iter().skip(2).cloned().collect::<Vec<usize>>())
                        .and(pool_shape)
                        .and(stride)
                        .map_collect(
                            |op_dim: &usize, pool_dim: &usize, stride_dim: &usize| {
                                1 + (op_dim - pool_dim) / stride_dim
                            }
                        )).collect::<Vec<usize>>();
        let data =
            RefCell::new(
                Tensor::zeros(shape).into_dimensionality::<T::Dim>().unwrap()
            );

        Self {
            operand,
            data,
            pool_shape: pool_shape.to_vec(),
            stride: stride.to_vec(),
            computed: Cell::new(false),
        }
    }
}

impl<T: ?Sized> Cache for MaxPool<T>
    where
        T: Data,
{
    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) { self.computed.set(false); }
}

impl<T: ?Sized> Forward for MaxPool<T>
    where
        T: Data,
        <T as Data>::Dim: RemoveAxis,
        <<T as Data>::Dim as Dimension>::Smaller: RemoveAxis,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (
            operand,
            mut data,
            pool_shape,
            stride,
        ) = (
            self.operand.data(),
            self.data.borrow_mut(),
            &self.pool_shape,
            &self.stride,
        );

        data.outer_iter_mut().zip(operand.outer_iter()).for_each(|(mut data_sample, op_sample)| {
            data_sample.outer_iter_mut().zip(op_sample.outer_iter()).for_each(|(mut data_channel, op_channel)| {
                data_channel.indexed_iter_mut()
                    .for_each(
                        |(i, y)| {
                            let i_pool: Vec<Range<usize>> = i.into_dimension()
                                .as_array_view()
                                .iter()
                                .enumerate()
                                .map(
                                    |(dim, index): (usize, &Ix)| {
                                        let first = index * stride[dim];
                                        first..(first + pool_shape[dim])
                                    }
                                ).collect();
                            *y = op_channel.slice_each_axis(
                                |ax| Slice::from(i_pool[ax.axis.index()].to_owned())
                            ).iter().fold(
                                f32::NEG_INFINITY, |acc, elem| acc.max(*elem),
                            )
                        }
                    )
            })
        })
    }
}

impl<T: ?Sized> Data for MaxPool<T>
    where
        T: Data,
{
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<T: ?Sized> Debug for MaxPool<T>
    where
        T: Data,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPool")
            .field("data", &self.data.borrow())
            .field("pool_shape", &self.pool_shape)
            .field("stride", &self.stride)
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<T: ?Sized> Display for MaxPool<T>
    where
        T: Data,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MaxPoolBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct MaxPoolBackward<T: ?Sized, U: ?Sized>
    where
        T: Gradient,
        U: Data<Dim=T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<MaxPool<U>>,
    pool_shape: Vec<usize>,
    stride: Vec<usize>,
}

impl<T: ?Sized, U: ?Sized> MaxPoolBackward<T, U>
    where
        T: Gradient,
        U: Data<Dim=T::Dim>,
{
    pub fn new(
        diff_operand: Rc<T>,
        no_diff_operand: Rc<MaxPool<U>>,
        pool_shape: &[usize],
        stride: &[usize],
    ) -> Self {
        let mut shape = T::Dim::zeros(diff_operand.gradient().ndim());
        shape[0] = diff_operand.gradient().shape()[0];
        shape[1] = diff_operand.gradient().shape()[1];
        itertools::izip!(
            shape.slice_mut().iter_mut().skip(2),
            diff_operand.gradient().shape().iter().skip(2),
            pool_shape,
            stride,
        ).for_each(
            |(dim, op_dim, pool_dim, stride)| {
                *dim = 1 + (op_dim - pool_dim) / stride
            },
        );

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            pool_shape: pool_shape.to_vec(),
            stride: stride.to_vec(),
        }
    }
}

impl<T: ?Sized, U: ?Sized> Gradient for MaxPoolBackward<T, U>
    where
        T: Gradient,
        U: Data<Dim=T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T: ?Sized, U: ?Sized> Overwrite for MaxPoolBackward<T, U>
    where
        T: Gradient,
        U: Data<Dim=T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T: ?Sized, U: ?Sized> Backward for MaxPoolBackward<T, U>
    where
        T: Gradient,
        U: Data<Dim=T::Dim>,
        <T as Gradient>::Dim: RemoveAxis,
        <<T as Gradient>::Dim as Dimension>::Smaller: RemoveAxis,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let op = self.no_diff_operand.operand.data();
        let op_data = self.no_diff_operand.data();
        let grad = self.gradient();
        let (
            pool_shape,
            stride,
        ) = (
            &self.pool_shape,
            &self.stride,
        );

        if self.diff_operand.can_overwrite() {
            Zip::from(grad.outer_iter())
                .and(op_grad.outer_iter_mut())
                .and(op_data.outer_iter())
                .and(op.outer_iter())
                .for_each(|grad_sample, mut op_grad_sample, op_data_sample, op_sample| {
                Zip::from(grad_sample.outer_iter())
                    .and(op_grad_sample.outer_iter_mut())
                    .and(op_data_sample.outer_iter())
                    .and(op_sample.outer_iter())
                    .for_each(|grad_channel, mut op_grad_channel, op_data_channel, op_channel| {
                    grad_channel.indexed_iter()
                        .zip(op_data_channel.iter())
                        .for_each(
                            |((i, grad_el), op_data_el)| {
                                let i_pool: Vec<Range<usize>> = i.into_dimension()
                                    .as_array_view()
                                    .iter()
                                    .enumerate()
                                    .map(
                                        |(dim, index): (usize, &Ix)| {
                                            let first = index * stride[dim];
                                            first..(first + pool_shape[dim])
                                        }
                                    ).collect();
                                let mut found_max = false;
                                op_grad_channel.slice_each_axis_mut(
                                    |ax| Slice::from(i_pool[ax.axis.index()].to_owned())
                                ).iter_mut()
                                    .zip(
                                        op_channel.slice_each_axis(
                                            |ax| Slice::from(i_pool[ax.axis.index()].to_owned())
                                        )
                                    ).for_each(
                                    |(op_grad_el, op_el)| {
                                        if !found_max {
                                            let is_max = op_data_el == op_el;
                                            *op_grad_el = is_max as usize as f32 * grad_el;
                                            found_max = is_max
                                        }
                                    }
                                )
                            }
                        )
                })
            });
            self.diff_operand.set_overwrite(false);
        } else {
            Zip::from(grad.outer_iter())
                .and(op_grad.outer_iter_mut())
                .and(op_data.outer_iter())
                .and(op.outer_iter())
                .for_each(|grad_sample, mut op_grad_sample, op_data_sample, op_sample| {
                    Zip::from(grad_sample.outer_iter())
                        .and(op_grad_sample.outer_iter_mut())
                        .and(op_data_sample.outer_iter())
                        .and(op_sample.outer_iter())
                        .for_each(|grad_channel, mut op_grad_channel, op_data_channel, op_channel| {
                            grad_channel.indexed_iter()
                                .zip(op_data_channel.iter())
                                .for_each(
                                    |((i, grad_el), op_data_el)| {
                                        let i_pool: Vec<Range<usize>> = i.into_dimension()
                                            .as_array_view()
                                            .iter()
                                            .enumerate()
                                            .map(
                                                |(dim, index): (usize, &Ix)| {
                                                    let first = index * stride[dim];
                                                    first..(first + pool_shape[dim])
                                                }
                                            ).collect();
                                        let mut found_max = false;
                                        op_grad_channel.slice_each_axis_mut(
                                            |ax| Slice::from(i_pool[ax.axis.index()].to_owned())
                                        ).iter_mut()
                                            .zip(
                                                op_channel.slice_each_axis(
                                                    |ax| Slice::from(i_pool[ax.axis.index()].to_owned())
                                                )
                                            ).for_each(
                                            |(op_grad_el, op_el)| {
                                                if !found_max {
                                                    let is_max = op_data_el == op_el;
                                                    *op_grad_el += is_max as usize as f32 * grad_el;
                                                    found_max = is_max
                                                }
                                            }
                                        )
                                    }
                                )
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

impl<T: ?Sized, U: ?Sized> Debug for MaxPoolBackward<T, U>
    where
        T: Gradient,
        U: Data<Dim=T::Dim>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPoolBackward")
            .field("gradient", &self.gradient.borrow())
            .field("pool_shape", &self.pool_shape)
            .field("stride", &self.stride)
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T: ?Sized, U: ?Sized> Display for MaxPoolBackward<T, U>
    where
        T: Gradient,
        U: Data<Dim=T::Dim>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
