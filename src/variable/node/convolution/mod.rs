use std::rc::Rc;

use ndarray::{
    iter::{AxisChunksIter, AxisChunksIterMut},
    linalg::general_mat_mul,
    Array, ArrayBase, Axis, Data, DataMut, Dimension, Ix2, Ix3, RemoveAxis, Zip,
};

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::variable::{
    gradient::Gradient,
    utils::{as_windows, as_windows_mut, columns_shape, Shared},
};

use super::{Backward, Forward};

/// Iterators needed for the **backward pass** of a grouped convolution.
type GroupedBackwardArgs<'a, D> = (
    AxisChunksIterMut<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
);

/// Partitions the **flattened input**, the **flattened kernel** and the **output map**
/// so that they can be used in a grouped convolution.
fn group_inputs<'a, D>(
    input: &'a Array<f32, D>,
    kernel: &'a Array<f32, D>,
    output: &'a mut Array<f32, D>,
    groups: usize,
) -> (
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIterMut<'a, f32, D>,
)
where
    D: Dimension,
{
    // Splits the input map along the channels.
    let input_groups = input.axis_chunks_iter(Axis(1), input.len_of(Axis(1)) / groups);
    // Splits the kernel along the output channels.
    let kernel_groups = kernel.axis_chunks_iter(Axis(0), kernel.len_of(Axis(0)) / groups);
    // Splits the output map along the channels.
    let output_map_groups = output.axis_chunks_iter_mut(Axis(1), output.len_of(Axis(1)) / groups);

    (input_groups, kernel_groups, output_map_groups)
}

/// Computes a shape from the array in input so that only the dimension of axis 0 is preserved.
fn flat_shape<D>(shape: D) -> Ix2
where
    D: Dimension,
{
    let mut flat_shape = Ix2::zeros(2);
    flat_shape[0] = shape[0];
    flat_shape[1] = shape.slice().iter().skip(1).product();
    flat_shape
}

/// Assigns to the **n**-dimensional feature map's gradient `dest` the **2**-dimensional
/// array `columns`. This method encapsulates the functionalities of **col2sig**, **col2im** and
/// **col2vol**.
fn assign_from_cols<D: Dimension, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
    dest: &mut ArrayBase<S, D>,
    columns: ArrayBase<T, Ix3>,
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) {
    let mut dest_windows_mut = as_windows_mut(dest, kernel_shape, stride, dilation);
    let from_cols = columns.into_shape(dest_windows_mut.raw_dim()).unwrap();

    // Safe because each sample is independent from one another.
    dest_windows_mut
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(from_cols.axis_iter(Axis(0)))
        .for_each(|(mut dest_view, src_view)| {
            Zip::from(&mut dest_view)
                .and(&src_view)
                .for_each(|dest_view_el, src_view_el| *dest_view_el += *src_view_el)
        });
}

fn convolution<
    D: Dimension + RemoveAxis,
    S: Data<Elem = f32>,
    U: Data<Elem = f32>,
    T: DataMut<Elem = f32>,
>(
    input: &ArrayBase<S, D>,
    kernel: &ArrayBase<U, D>,
    output: &mut ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
) {
    let (kernel_shape, flattened_kernel) = (
        kernel.shape(),
        kernel
            .view()
            .into_shape(flat_shape(kernel.raw_dim()))
            .unwrap(),
    );

    let input_windows = as_windows(input, kernel_shape, stride, dilation);
    let input_columns = input_windows
        .to_shape(columns_shape(input, kernel_shape, stride, dilation))
        .unwrap();

    Zip::from(input_columns.axis_iter(Axis(0)))
        .and(output.axis_iter_mut(Axis(0)))
        .par_for_each(|input_sample_columns, output_sample| {
            let flat_shape = flat_shape(output_sample.raw_dim());
            let mut flattened_sample_out_view_mut = output_sample.into_shape(flat_shape).unwrap();
            general_mat_mul(
                1.,
                &flattened_kernel,
                &input_sample_columns.t(),
                0.,
                &mut flattened_sample_out_view_mut,
            );
        });
}

fn grouped_convolution<D>(
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    output: &mut Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
) where
    D: Dimension + RemoveAxis,
{
    let (input_groups, kernel_groups, output_buffer_groups) =
        group_inputs(input, kernel, output, groups);
    kernel_groups
        .into_iter()
        .zip(input_groups.into_iter())
        .zip(output_buffer_groups.into_iter())
        .for_each(|((kernel, input), mut output)| {
            convolution(&input, &kernel, &mut output, stride, dilation);
        });
}

fn convolution_backward_input<
    D: Dimension + RemoveAxis,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
>(
    input_grad: &mut ArrayBase<S, D>,
    grad: &ArrayBase<T, D>,
    kernel: &ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
) {
    let (kernel_shape, flattened_kernel, grad_shape) = (
        kernel.shape(),
        kernel
            .view()
            .into_shape(flat_shape(kernel.raw_dim()))
            .unwrap(),
        grad.shape(),
    );

    let mut buffer_shape = Ix3::zeros(3);
    buffer_shape[0] = grad_shape[0];
    buffer_shape[1] = flattened_kernel.shape()[1];
    buffer_shape[2] = grad_shape.iter().skip(2).product();
    let mut buffer = Array::<f32, Ix3>::zeros(buffer_shape);

    Zip::from(grad.axis_iter(Axis(0)))
        .and(buffer.axis_iter_mut(Axis(0)))
        .par_for_each(|gradient_sample, mut buffer_sample| {
            let gradient_sample_flat_shape = flat_shape(gradient_sample.raw_dim());
            let flattened_sample_in = gradient_sample
                .into_shape(gradient_sample_flat_shape)
                .unwrap();
            general_mat_mul(
                1.,
                &flattened_kernel.t(),
                &flattened_sample_in,
                0.,
                &mut buffer_sample,
            );
        });

    assign_from_cols(input_grad, buffer, kernel_shape, stride, dilation);
}

fn convolution_backward_kernel<
    D: Dimension + RemoveAxis,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
>(
    kernel_grad: &mut ArrayBase<S, D>,
    grad: &ArrayBase<T, D>,
    input: &ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
) {
    let kernel_shape = kernel_grad.shape();
    let input_windows = as_windows(input, kernel_shape, stride, dilation);
    let columns_shape = columns_shape(input, kernel_shape, stride, dilation);
    let mut matrix_shape = Ix2::zeros(2);
    matrix_shape[0] = columns_shape[0] * columns_shape[1];
    matrix_shape[1] = columns_shape[2];
    let input_matrix = input_windows.to_shape(matrix_shape).unwrap();

    Zip::from(kernel_grad.axis_iter_mut(Axis(0)))
        .and(grad.axis_iter(Axis(1)))
        .par_for_each(|kernel_grad_view_mut, grad_view| {
            let kernel_grad_numel = kernel_grad_view_mut.shape().iter().product::<usize>();
            let grad_view_numel = grad_view.shape().iter().product::<usize>();

            general_mat_mul(
                1.0,
                &grad_view.to_shape((1, grad_view_numel)).unwrap(),
                &input_matrix,
                1.0,
                &mut kernel_grad_view_mut
                    .into_shape((1, kernel_grad_numel))
                    .unwrap(),
            );
        });
}

fn group_gradients_input<'a, D: Dimension, S: DataMut<Elem = f32>, U: Data<Elem = f32>>(
    input_grad: &'a mut ArrayBase<S, D>,
    grad: &'a ArrayBase<U, D>,
    kernel: &'a ArrayBase<U, D>,
    groups: usize,
) -> GroupedBackwardArgs<'a, D> {
    let input_grad_groups =
        input_grad.axis_chunks_iter_mut(Axis(1), input_grad.len_of(Axis(1)) / groups);
    let grad_groups = grad.axis_chunks_iter(Axis(1), grad.len_of(Axis(1)) / groups);
    let kernel_groups = kernel.axis_chunks_iter(Axis(0), kernel.len_of(Axis(0)) / groups);

    (input_grad_groups, grad_groups, kernel_groups)
}

fn group_gradients_kernel<'a, D: Dimension, S: DataMut<Elem = f32>, U: Data<Elem = f32>>(
    kernel_grad: &'a mut ArrayBase<S, D>,
    grad: &'a ArrayBase<U, D>,
    input: &'a ArrayBase<U, D>,
    groups: usize,
) -> GroupedBackwardArgs<'a, D> {
    let kernel_grad_groups =
        kernel_grad.axis_chunks_iter_mut(Axis(0), kernel_grad.len_of(Axis(0)) / groups);
    let grad_groups = grad.axis_chunks_iter(Axis(1), grad.len_of(Axis(1)) / groups);
    let input_groups = input.axis_chunks_iter(Axis(1), input.len_of(Axis(1)) / groups);

    (kernel_grad_groups, grad_groups, input_groups)
}

pub(super) fn grouped_convolution_backward_input<D: Dimension + RemoveAxis>(
    input_grad: &mut Array<f32, D>,
    grad: &Array<f32, D>,
    kernel: &Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
) {
    let (input_grad_groups, grad_groups, kernel_groups) =
        group_gradients_input(input_grad, grad, kernel, groups);

    grad_groups
        .into_iter()
        .zip(input_grad_groups.into_iter())
        .zip(kernel_groups.into_iter())
        .for_each(|((gradient, mut input_gradient), kernel)| {
            convolution_backward_input(&mut input_gradient, &gradient, &kernel, stride, dilation)
        });
}

fn grouped_convolution_backward_kernel<D: Dimension + RemoveAxis>(
    kernel_grad: &mut Array<f32, D>,
    grad: &Array<f32, D>,
    input: &Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
) {
    let (kernel_grad_groups, grad_groups, input_groups) =
        group_gradients_kernel(kernel_grad, grad, input, groups);

    grad_groups
        .into_iter()
        .zip(kernel_grad_groups.into_iter())
        .zip(input_groups.into_iter())
        .for_each(|((gradient, mut kernel_gradient), input)| {
            convolution_backward_kernel(&mut kernel_gradient, &gradient, &input, stride, dilation)
        });
}

pub(crate) struct Convolution<D>
where
    D: Dimension + RemoveAxis,
{
    input_data: Shared<Array<f32, D>>,
    kernel_data: Shared<Array<f32, D>>,
    stride: <D::Smaller as Dimension>::Smaller,
    dilation: <D::Smaller as Dimension>::Smaller,
    groups: usize,
    data: Shared<Array<f32, D>>,
}

impl<D> Convolution<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        input_data: Shared<Array<f32, D>>,
        kernel_data: Shared<Array<f32, D>>,
        stride: <D::Smaller as Dimension>::Smaller,
        dilation: <D::Smaller as Dimension>::Smaller,
        groups: usize,
        data: Shared<Array<f32, D>>,
    ) -> Self {
        Self {
            input_data,
            kernel_data,
            stride,
            dilation,
            groups,
            data,
        }
    }
}

impl<D> Forward for Convolution<D>
where
    D: Dimension + RemoveAxis,
{
    fn forward(&self) {
        if self.groups < 2 {
            convolution(
                &*self.input_data.borrow(),
                &*self.kernel_data.borrow(),
                &mut *self.data.borrow_mut(),
                self.stride.slice(),
                self.dilation.slice(),
            );
        } else {
            grouped_convolution(
                &*self.input_data.borrow(),
                &*self.kernel_data.borrow(),
                &mut *self.data.borrow_mut(),
                self.stride.slice(),
                self.dilation.slice(),
                self.groups,
            )
        }
    }
}

pub(crate) struct ConvolutionBackward<D>
where
    D: Dimension + RemoveAxis,
{
    backward_input: ConvolutionBackwardInput<D>,
    backward_kernel: ConvolutionBackwardKernel<D>,
}

impl<D> ConvolutionBackward<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        backward_input: ConvolutionBackwardInput<D>,
        backward_kernel: ConvolutionBackwardKernel<D>,
    ) -> Self {
        Self {
            backward_input,
            backward_kernel,
        }
    }
}

impl<D> Backward for ConvolutionBackward<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        self.backward_input.backward();
        self.backward_kernel.backward();
    }
}

pub(crate) struct ConvolutionBackwardInput<D>
where
    D: Dimension + RemoveAxis,
{
    kernel_data: Shared<Array<f32, D>>,
    input_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<D>>,
    stride: <D::Smaller as Dimension>::Smaller,
    dilation: <D::Smaller as Dimension>::Smaller,
    groups: usize,
}

impl<D> ConvolutionBackwardInput<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        kernel_data: Shared<Array<f32, D>>,
        input_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<D>>,
        stride: <D::Smaller as Dimension>::Smaller,
        dilation: <D::Smaller as Dimension>::Smaller,
        groups: usize,
    ) -> Self {
        Self {
            kernel_data,
            input_gradient,
            gradient,
            stride,
            dilation,
            groups,
        }
    }
}

impl<D> Backward for ConvolutionBackwardInput<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        if self.groups < 2 {
            convolution_backward_input(
                &mut *self.input_gradient.borrow_mut(),
                &*self.gradient.borrow(),
                &*self.kernel_data.borrow(),
                self.stride.slice(),
                self.dilation.slice(),
            );
        } else {
            grouped_convolution_backward_input(
                &mut *self.input_gradient.borrow_mut(),
                &*self.gradient.borrow(),
                &*self.kernel_data.borrow(),
                self.stride.slice(),
                self.dilation.slice(),
                self.groups,
            )
        }
    }
}

pub(crate) struct ConvolutionBackwardKernel<D>
where
    D: Dimension + RemoveAxis,
{
    input_data: Shared<Array<f32, D>>,
    kernel_gradient: Rc<Gradient<D>>,
    gradient: Rc<Gradient<D>>,
    stride: <D::Smaller as Dimension>::Smaller,
    dilation: <D::Smaller as Dimension>::Smaller,
    groups: usize,
}

impl<D> ConvolutionBackwardKernel<D>
where
    D: Dimension + RemoveAxis,
{
    pub(crate) fn new(
        input_data: Shared<Array<f32, D>>,
        kernel_gradient: Rc<Gradient<D>>,
        gradient: Rc<Gradient<D>>,
        stride: <D::Smaller as Dimension>::Smaller,
        dilation: <D::Smaller as Dimension>::Smaller,
        groups: usize,
    ) -> Self {
        Self {
            input_data,
            kernel_gradient,
            gradient,
            stride,
            dilation,
            groups,
        }
    }
}

impl<D> Backward for ConvolutionBackwardKernel<D>
where
    D: Dimension + RemoveAxis,
{
    fn backward(&self) {
        if self.groups < 2 {
            convolution_backward_kernel(
                &mut *self.kernel_gradient.borrow_mut(),
                &*self.gradient.borrow(),
                &*self.input_data.borrow(),
                self.stride.slice(),
                self.dilation.slice(),
            );
        } else {
            grouped_convolution_backward_kernel(
                &mut *self.kernel_gradient.borrow_mut(),
                &*self.gradient.borrow(),
                &*self.input_data.borrow(),
                self.stride.slice(),
                self.dilation.slice(),
                self.groups,
            )
        }
    }
}

#[cfg(test)]
mod test;
