mod constant;
mod padding_mode;
mod reflective;
mod replicative;
mod zero;

pub(crate) use padding_mode::SampleDim;

pub use constant::Constant;
pub use padding_mode::PaddingMode;
pub use reflective::Reflective;
pub use zero::Zero;

use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{Dimension, RemoveAxis, Slice};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub(crate) struct Pad<D, T>
where
    D: Dimension,
    D::Smaller: RemoveAxis,
    SampleDim<D>: Copy,
    T: PaddingMode<D>,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    computed: Cell<bool>,
    mode: T,
    padding: SampleDim<D>,
    batch_collapsed_dim: D::Smaller,
    batch_collapsed_padded_dim: D::Smaller,
}

impl<D, T> Pad<D, T>
where
    D: Dimension,
    D::Smaller: RemoveAxis,
    SampleDim<D>: Copy,
    T: PaddingMode<D>,
{
    pub(crate) fn new(
        operand_data: Rc<RefCell<Tensor<D>>>,
        data: Rc<RefCell<Tensor<D>>>,
        computed: Cell<bool>,
        mode: T,
        padding: SampleDim<D>,
    ) -> Self {
        let operand_data_dim = operand_data.borrow().raw_dim();
        let data_dim = data.borrow().raw_dim();

        let (mut batch_collapsed_dim, mut batch_collapsed_padded_dim) = (
            <D as Dimension>::Smaller::zeros(operand_data_dim.ndim() - 1),
            <D as Dimension>::Smaller::zeros(data_dim.ndim() - 1),
        );

        let outer_dimension: usize = operand_data_dim.slice().iter().take(2).product();

        batch_collapsed_dim[0] = outer_dimension;
        batch_collapsed_padded_dim[0] = outer_dimension;

        let (sample_dim, padded_sample_dim) = (
            operand_data_dim.slice().iter().skip(2),
            data_dim.slice().iter().skip(2),
        );

        batch_collapsed_dim
            .slice_mut()
            .iter_mut()
            .skip(1)
            .zip(sample_dim)
            .for_each(|(view_dim, inner_dim)| *view_dim = *inner_dim);
        batch_collapsed_padded_dim
            .slice_mut()
            .iter_mut()
            .skip(1)
            .zip(padded_sample_dim)
            .for_each(|(view_dim, inner_dim)| *view_dim = *inner_dim);

        Self {
            operand_data,
            data,
            computed,
            mode,
            padding,
            batch_collapsed_dim,
            batch_collapsed_padded_dim,
        }
    }
}

impl<D, T> Forward for Pad<D, T>
where
    D: Dimension,
    D::Smaller: RemoveAxis,
    SampleDim<D>: Copy,
    T: PaddingMode<D>,
{
    fn forward(&self) {
        let operand_data = self.operand_data.borrow();
        let mut data = self.data.borrow_mut();

        let batch_collapsed_padded_dim = self.batch_collapsed_padded_dim.clone();
        let batch_collapsed_dim = self.batch_collapsed_dim.clone();

        let (mut data_view_mut, operand_data_view) = (
            data.view_mut()
                .into_shape(batch_collapsed_padded_dim)
                .unwrap(),
            operand_data.view().into_shape(batch_collapsed_dim).unwrap(),
        );

        let mode = self.mode;
        let padding = self.padding;

        data_view_mut
            .outer_iter_mut()
            .into_par_iter()
            .zip(operand_data_view.outer_iter())
            .for_each(|(mut padded_sample, base_sample)| {
                mode.pad(&mut padded_sample, &base_sample, padding)
            });
    }
}

pub(crate) struct PadBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    gradient: Rc<RefCell<Option<Tensor<D>>>>,
    shape: D,
    padding: SampleDim<D>,
}

impl<D> PadBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        gradient: Rc<RefCell<Option<Tensor<D>>>>,
        shape: D,
        padding: SampleDim<D>,
    ) -> Self {
        Self {
            operand_gradient,
            gradient,
            shape,
            padding,
        }
    }
}

impl<D> Backward for PadBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);

        let padding = self.padding.slice();

        let gradient_slice = gradient.slice_each_axis(|ax| {
            let (index, len) = (ax.axis.index(), gradient.len_of(ax.axis));
            let range = {
                if index > 1 && padding[index - 2] != 0 {
                    padding[index - 2] as isize..-(padding[index - 2] as isize)
                } else {
                    0..len as isize
                }
            };

            Slice::from(range)
        });

        *operand_gradient += &gradient_slice;
    }
}
