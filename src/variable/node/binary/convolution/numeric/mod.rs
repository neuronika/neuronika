use super::{PaddingMode, ReflPad, ReplPad};
use ndarray::{
    iter::{AxisChunksIter, AxisChunksIterMut},
    Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, DataMut, Dimension, Ix2, IxDyn, RawData,
    RemoveAxis, ShapeBuilder, Slice, ViewRepr, Zip,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Checks that the arguments are correct for the given **convolution**. It verifies that the
/// `padding`, `stride` and `dilation` slices are of the right length; their length must match the
/// dimensionality of the convolution. It also check that `kernel` and `input` are of the same
/// dimension and that the kernel size, after dilation is applied, **is not bigger** that the actual
/// input size.
///
/// # Arguments
///
/// * `input_shape` - the shape of the input map of the convolution
/// * `kernel_shape` - the shape of the kernel
/// * `padding` - the padding to be applied to the input
/// * `stride` - the stride for the cross-correlation
/// * `dilation` - the spacing between the kernel points
pub(super) fn check_conv_args(
    input_shape: &[usize],
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) {
    // The type of convolution can be extrapolated by considering the number of input's dimension
    // skipping the first two, that are the batch size and input channels. The first two axes of
    // the input are always for the batch size and the number of input channels.
    let conv_dim = input_shape.len() - 2;
    assert_eq!(
        conv_dim,
        padding.len(),
        "error: invalid padding {:?} for {}d conv.",
        padding,
        conv_dim
    );

    assert_eq!(
        conv_dim,
        stride.len(),
        "error: invalid stride {:?} for {}d conv.",
        stride,
        conv_dim
    );

    assert_eq!(
        conv_dim,
        dilation.len(),
        "error: invalid dilation {:?} for {}d conv.",
        dilation,
        conv_dim
    );

    assert_eq!(
        kernel_shape.len(),
        input_shape.len(),
        "error: invalid kernel's shape {:?} for {}d conv",
        &kernel_shape,
        conv_dim
    );

    let dilated_kernel_size: Vec<usize> = kernel_shape
        .iter()
        .skip(2)
        .zip(dilation.iter())
        .map(|(kernel_dim, dilation_dim)| (kernel_dim - 1) * dilation_dim + 1)
        .collect();
    let padded_input_size: Vec<usize> = input_shape
        .iter()
        .skip(2)
        .zip(padding.iter())
        .map(|(input_dim, padding)| input_dim + padding * 2)
        .collect();
    padded_input_size
        .iter()
        .zip(dilated_kernel_size.iter())
        .for_each(|(padded_input_dim, dilated_kernel_dim)| {
            assert!(
                padded_input_dim >= dilated_kernel_dim,
                "Calculated padded input size per channel: {:?}. Kernel size: {:?}. The kernel size can't be greater than actual input size.",
                padded_input_size,
                dilated_kernel_size
            )
        });
}

/// Checks that the arguments are correct for the given **grouped convolution**. This function
/// should most of the time be used together with `check_conv_args`.
///
/// It enforces that both the number of **input channels** and **output channels** are divisible
/// by `groups`.
///
/// # Arguments
/// * `input_shape` - the shape of the input map of the convolution
/// * `kernel_shape` - the shape of the kernel
/// * `groups` -  the connections between inputs and outputs.
pub(super) fn check_groups_args(input_shape: &[usize], kernel_shape: &[usize], groups: usize) {
    assert_eq!(
        input_shape[1] % groups,
        0,
        "error: in_channels {} is not disible by groups {}",
        input_shape[1],
        groups
    );
    assert_eq!(
        kernel_shape[0] % groups,
        0,
        "error: out_channels {} is not disible by groups {}",
        kernel_shape[0],
        groups
    );
}

/// Computes the shape of the array resulting from the **n**-dimensional convolution
/// performed with the given parameters.
///
/// **n** can be either **1**, **2** or **3**
///
/// # Arguments
///
/// * `input_shape` - the shape of the input
/// * `kernel_shape` - the shape of the kernel
/// * `padding` - the padding around the input
/// * `stride` - the stride
/// * `dilation` - the dilation
///
/// ## 1-dimensional convolution
///
/// The **input** must be of shape **(N, Cin, L)**
/// * **N** is the batch size
/// * **Cin** is the number of input channels
/// * **L** is the **length** of the input
///
/// The **kernel** must be of shape **(Cout, Cin, Lk)**
/// * **Cout** is the number of output channels
/// * **Cin** is the number of input channels
/// * **Lk** is the **length** of the kernel
///
/// The resulting output shape will be **(N, Cout, Lout)**
///
/// ## 2-dimensional convolution
/// The **input** must be of shape **(N, Cin, H, W)**
/// * **N** is the batch size
/// * **Cin** is the number of input channels
/// * **H** is the **height** of the input
/// * **W** is the **width** of the input
///
/// The **kernel** must be of shape **(Cout, Cin, Hk, Wk)**
/// * **Cout** is the number of output channels
/// * **Cin** is the number of input channels
/// * **Hk** is the **height** of the kernel
/// * **Wk** is the **width** of the kernel
///
/// The resulting output shape will be **(N, Cout, Hout, Wout)**
///
/// ## 3-dimensional convolution
///
/// The **input** must be of shape **(N, Cin, D, H, W)**
/// * **N** is the batch size
/// * **Cin** is the number of input channels
/// * **D** is the **depth** of the input
/// * **H** is the **height** of the input
/// * **W** is the **width** of the input
///
/// The **kernel** must be of shape **(Cout, Cin, Dk,  Hk, Wk)**
/// * **Cout** is the number of output channels
/// * **Cin** is the number of input channels
/// * **Dk** is the **depth** of the kernel
/// * **Hk** is the **height** of the kernel
/// * **Wk** is the **width** of the kernel
///
/// The resulting output shape will be **(N, Cout, Dout, Hout, Wout)**
pub(super) fn conv_out_shape<D: Dimension>(
    input_shape: &[usize],
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> D {
    let in_shape_len = input_shape.len();
    // Initialize the dimension to be all 0s.
    let mut map_shape = D::zeros(in_shape_len);
    let map_shape_slice = map_shape.slice_mut();
    // Sets the batch size. The batch size doesn't change.
    map_shape_slice[0] = input_shape[0];
    // Sets the output channels.
    map_shape_slice[1] = kernel_shape[0];
    // First two components of the shape are always
    // the batch size and channels.
    itertools::izip!(
        map_shape_slice.iter_mut().skip(2), // Skips batch size and out channels.
        input_shape.iter().skip(2),         // Skips batch size and out channels.
        kernel_shape.iter().skip(2),        // Skips out channels and in channels.
        padding,
        stride,
        dilation
    )
    .for_each(|(map_s, in_s, k_s, pd, stri, dil)| {
        *map_s = (in_s + 2 * pd - dil * (k_s - 1) - 1) / stri + 1
    });
    map_shape
}

/// Computes the shape of the array resulting from the **n**-dimensional convolution
/// performed with the given parameters. `input_shape` is assumed to be the shape of an **already**
/// padded input.
///
/// # Arguments
/// * `input_shape` - the shape of the input
/// * `kernel_shape` - the shape of the kernel
/// * `stride` - the stride
/// * `dilation` - the dilation
fn conv_out_shape_padded<D: Dimension>(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> D {
    let in_shape_len = input_shape.len();
    // Initialize the dimension to be all 0s.
    let mut map_shape = D::zeros(in_shape_len);
    let map_shape_slice = map_shape.slice_mut();
    // Sets the batch size. The batch size doesn't change.
    map_shape_slice[0] = input_shape[0];
    // Sets the output channels.
    map_shape_slice[1] = kernel_shape[0];
    // First two components of the shape are always
    // the batch size and channels.
    itertools::izip!(
        map_shape_slice.iter_mut().skip(2), // Skips batch size and out channels.
        input_shape.iter().skip(2),         // Skips batch size and out channels.
        kernel_shape.iter().skip(2),        // Skips out channels and in channels.
        stride,
        dilation
    )
    .for_each(|(map_s, in_s, k_s, stri, dil)| *map_s = (in_s - dil * (k_s - 1) - 1) / stri + 1);
    map_shape
}

/// Computes the shape of the **input** after the padding is applied.
///
/// # Arguments
///
/// * `input_shape` - the shape of the input
/// * `padding` - the padding around the input
pub(super) fn padded_shape<D: Dimension>(input_shape: &[usize], padding: &[usize]) -> D {
    let in_shape_len = input_shape.len();
    let mut padded_input_shape = D::zeros(in_shape_len);
    padded_input_shape[0] = input_shape[0];
    padded_input_shape[1] = input_shape[1];
    padded_input_shape
        .slice_mut()
        .iter_mut()
        .skip(2)
        .zip(input_shape.iter().skip(2))
        .zip(padding.iter())
        .for_each(|((padded_s, original_s), pad)| *padded_s = original_s + 2 * pad);
    padded_input_shape
}

/// Pads the input array accordingly to padding and the supplied padding mode.
///
/// This function expects arrays of the following shapes:
///
/// * *(N, C, L)*, where L is the length of the sequences.
///
/// * *(N, C, H, W)*, where H and W are respectively the height and width of the images.
///
/// * *(N, C, D, H, W)*, where H, W, and D are respectively the depth, height and width of the
/// volumes.
///
/// In all three cases N is the batch size and C is the number of channels.
///
/// See also [`constant_pad`], [`reflection_pad`] and [`replication_pad`].
///
/// # Arguments
///
/// * `array` - array to be padded.
///
/// * `padding` - padding around to be applied to input.
///
/// * `padding_mode` - padding type. Can be either [`Zero`], [`Constant`], [`Reflective`] or
/// [`Replicative`].
pub(super) fn pad<D: Dimension, T: PaddingMode>(
    array: &Array<f32, D>,
    padding: &[usize],
    padding_mode: &T,
) -> Array<f32, D>
where
    <D as Dimension>::Smaller: RemoveAxis,
    <<D as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    let mut padded = {
        let padded_shape = padded_shape::<D>(array.shape(), padding);
        Array::<f32, D>::zeros(padded_shape)
    };
    let (padded_raw_dim, original_raw_dim) = (padded.raw_dim(), array.raw_dim());
    // The dimension of a single raw sample in the batch.
    let (padded_inner_dimensions, original_inner_dimensions) = (
        padded_raw_dim.slice().iter().skip(2),
        original_raw_dim.slice().iter().skip(2),
    );
    // The number of single raw samples in the batch.
    let outer_dimension: usize = original_raw_dim.slice().iter().take(2).product();

    // Reshapes by removing an axis, so that all samples can be iterated on and padded.
    let (mut input_view_dimension, mut original_view_dimension): (
        <D as Dimension>::Smaller,
        <D as Dimension>::Smaller,
    ) = (
        <D as Dimension>::Smaller::zeros(padded_raw_dim.ndim() - 1),
        <D as Dimension>::Smaller::zeros(original_raw_dim.ndim() - 1),
    );
    input_view_dimension[0] = outer_dimension;
    original_view_dimension[0] = outer_dimension;

    input_view_dimension
        .slice_mut()
        .iter_mut()
        .skip(1)
        .zip(padded_inner_dimensions)
        .for_each(|(view_dim, inner_dim)| *view_dim = *inner_dim);
    original_view_dimension
        .slice_mut()
        .iter_mut()
        .skip(1)
        .zip(original_inner_dimensions)
        .for_each(|(view_dim, inner_dim)| *view_dim = *inner_dim);

    let (mut input_view_mut, original_view) = (
        padded.view_mut().into_shape(input_view_dimension).unwrap(),
        array.view().into_shape(original_view_dimension).unwrap(),
    );

    input_view_mut
        .outer_iter_mut()
        .into_par_iter()
        .zip(original_view.outer_iter())
        .for_each(|(mut pad_sample, original_sample)| {
            padding_mode.pad_inplace(&mut pad_sample, &original_sample, padding)
        });
    padded
}

/// Returns an **unpadded** view of `array`. This method is supposed to be used only with arrays
/// of dimension greater or equal than *3*. The length of the padding slice must be *dim - 2* where
/// *dim* is the array's dimensionality.
fn unpad<'a, S: Data<Elem = f32> + 'a, D: Dimension + 'a>(
    array: &'a ArrayBase<S, D>,
    padding: &[usize],
) -> ArrayBase<ViewRepr<&'a f32>, D> {
    array.slice_each_axis(|ax| {
        let (ax_index, ax_len) = (ax.axis.index(), array.len_of(ax.axis));
        let range = {
            if ax_index > 1 && padding[ax_index - 2] != 0 {
                padding[ax_index - 2] as isize..-(padding[ax_index - 2] as isize)
            } else {
                0..ax_len as isize
            }
        };
        Slice::from(range)
    })
}

/// Computes the shape of a rolling window view.
///
/// # Arguments
/// * `input` - input array
/// * `window_shape` - the shape of each of the windows
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
pub(super) fn compute_rolling_window_shape<D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Vec<usize> {
    let mut win_indices_shape: D =
        conv_out_shape_padded(input.shape(), window_shape, stride, dilation);
    win_indices_shape[1] = 1;
    win_indices_shape
        .slice()
        .iter()
        .chain(window_shape.iter().skip(1))
        .cloned()
        .collect()
}

/// Computes the strides of a rolling window view.
///
/// # Arguments
/// * `input` - input array
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
pub(super) fn compute_rolling_window_strides<D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    stride: &[usize],
    dilation: &[usize],
) -> Vec<usize> {
    let indexing_strides: Vec<isize> = {
        let view = input.slice_each_axis(|ax| {
            let axis_index = ax.axis.index();
            if axis_index == 0 || axis_index == 1 {
                Slice::new(0, None, 1) // Batch stride and channel stride
            } else {
                Slice::new(0, None, stride[ax.axis.index() - 2] as isize)
            }
        });
        let view_strides: &[isize] = view.strides();
        view_strides.to_vec()
    };
    // Number of in channels doesn't count for the window's strides,
    // it must be left unchanged.
    let window_strides: Vec<isize> = input
        .strides()
        .iter()
        .skip(1) // Skip out channels
        .enumerate()
        .map(|(i, is)| {
            if i < 1 {
                *is
            } else {
                *is * (dilation[i - 1] as isize)
            }
        })
        .collect();
    indexing_strides
        .iter()
        .chain(window_strides.iter())
        .map(|s| *s as usize)
        .collect()
}

/// Returns a **rolling window view** of the input array.
///
/// # Arguments
/// * `input` - input array
/// * `window_shape` - the shape of each of the windows
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
pub(super) fn as_windows<'a, D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayView<'a, f32, IxDyn> {
    let rolling_window_shape: Vec<usize> =
        compute_rolling_window_shape(input, window_shape, stride, dilation);
    let rolling_window_strides: Vec<usize> =
        compute_rolling_window_strides(input, stride, dilation);

    unsafe {
        ArrayView::from_shape_ptr(
            rolling_window_shape.strides(rolling_window_strides),
            input.as_ptr(),
        )
    }
}

/// Returns a **mutable rolling window view** of the input array.
///
/// # Arguments
/// * `input` - input array
/// * `window_shape` - the shape of each of the windows
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
pub(super) fn as_windows_mut<'a, D: Dimension, S: DataMut<Elem = f32>>(
    input: &mut ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayViewMut<'a, f32, IxDyn> {
    let rolling_window_shape: Vec<usize> =
        compute_rolling_window_shape(input, window_shape, stride, dilation);
    let rolling_window_strides: Vec<usize> =
        compute_rolling_window_strides(input, stride, dilation);

    unsafe {
        ArrayViewMut::from_shape_ptr(
            rolling_window_shape.strides(rolling_window_strides),
            input.as_mut_ptr(),
        )
    }
}

/// Computes **sig2col**, **im2col** and **vol2col**.
///
/// # Arguments
/// * `input` - input array
/// * `kernel_shape` - the shape of the kernel
/// * `padding` - the padding to be applied to `input`
/// * `stride` - the stride
/// * `dilation` - the dilation
fn to_col<D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Array<f32, Ix2> {
    let mut o_shape = conv_out_shape_padded::<D>(input.shape(), kernel_shape, stride, dilation);
    o_shape[1] = 1;
    let (im2col_h, im2col_w): (usize, usize) = {
        (
            kernel_shape.iter().skip(1).product(),
            o_shape.slice().iter().product(),
        )
    };
    as_windows(&input, kernel_shape, stride, dilation)
        .into_owned()
        .into_shape((im2col_w, im2col_h))
        .unwrap()
        .reversed_axes()
}

/// Reshapes the array in input into a matrix so that only the dimension of
/// axis 0 is preserved.
///
/// **Panics** if `array` is not standard layout.
fn flatten<D: Dimension, S: RawData>(array: ArrayBase<S, D>) -> ArrayBase<S, Ix2> {
    let (kernel_shape, mut new_shape) = (array.raw_dim(), Ix2::zeros(2));
    new_shape[0] = kernel_shape[0];
    new_shape[1] = kernel_shape.slice().iter().skip(1).product();
    array.into_shape(new_shape).unwrap()
}

/// Puts the axis corresponding to the output channels of the feature map `array`
/// in the last position and returns the input with swapped axes.
pub(super) fn permute_channels<D: Dimension>(
    array: ArrayBase<ViewRepr<&f32>, D>,
) -> ArrayBase<ViewRepr<&f32>, D> {
    let mut dim = array.raw_dim();
    dim.set_last_elem(0);
    let dim_len = dim.ndim() - 1;
    dim.slice_mut()
        .iter_mut()
        .zip(1..=dim_len)
        .for_each(|(ax, el)| *ax = el);
    array.permuted_axes(dim)
}

/// Assigns the **2-dimensional** convolution result to the **n-dimensional** feature map.
pub(super) fn assign_to_output_map<D: Dimension, S: DataMut<Elem = f32>>(
    out_map: &mut ArrayBase<S, D>,
    flat_result: Array<f32, Ix2>,
) {
    let batch_size = out_map.shape()[0];
    let mut sample_size = out_map.raw_dim();
    sample_size[0] = 1;

    let convolved_samples =
        flat_result.axis_chunks_iter(Axis(1), flat_result.len_of(Axis(1)) / batch_size);
    let samples = out_map.axis_chunks_iter_mut(Axis(0), 1);

    samples
        .into_par_iter()
        .zip(convolved_samples)
        .for_each(|(mut sample, incoming_result)| {
            Zip::from(&mut sample)
                .and(
                    &incoming_result
                        .as_standard_layout()
                        .into_shape(sample_size.clone())
                        .unwrap(),
                )
                .for_each(|sample_el, incoming_el| *sample_el = *incoming_el)
        });
}

/// Assigns to the **n**-dimensional feature map's gradient `dest` the **2**-dimensional
/// array `columns`. This method encapsulates the functionalities of **col2sig**, **col2im** and
/// **col2vol**.
///
/// **n** can be either 3, 4 or 5.
pub(super) fn assign_from_cols<D: Dimension, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
    dest: &mut ArrayBase<S, D>,
    columns: ArrayBase<T, Ix2>,
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) {
    let mut dest_windows_mut = as_windows_mut(dest, kernel_shape, stride, dilation);
    let from_cols = columns.into_shape(dest_windows_mut.raw_dim()).unwrap();

    Zip::from(&mut dest_windows_mut)
        .and(&from_cols)
        .for_each(|dest_el, src_el| *dest_el += *src_el);
}

/// Partitions the **flattened input**, the **flattened kernel** and the **output map**
/// so that they can be used in a grouped convolution.
pub(super) fn group_inputs<'a, D: Dimension, S: Data<Elem = f32>, T: DataMut<Elem = f32>>(
    input: &'a ArrayBase<S, D>,
    kernel: &'a ArrayBase<S, D>,
    out_map: &'a mut ArrayBase<T, D>,
    groups: usize,
) -> (
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIterMut<'a, f32, D>,
) {
    // Splits the input map along the channels.
    let input_groups = input.axis_chunks_iter(Axis(1), input.len_of(Axis(1)) / groups);
    // Splits the kernel along the output channels.
    let kernel_groups = kernel.axis_chunks_iter(Axis(0), kernel.len_of(Axis(0)) / groups);
    // Splits the outputmap along the channels.
    let out_map_groups = out_map.axis_chunks_iter_mut(Axis(1), out_map.len_of(Axis(1)) / groups);

    (input_groups, kernel_groups, out_map_groups)
}

/// Iterators needed for the **backward pass** of a grouped convolution.
type GroupedBackwardArgs<'a, D> = (
    AxisChunksIterMut<'a, f32, D>,
    AxisChunksIterMut<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
);

/// Iterators needed for the **backward pass** of a grouped convolution where the kernel
/// is the only differentiable variable.
type GroupedBackwardArgsUnary<'a, D> = (
    AxisChunksIterMut<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
);

/// Partitions the **input gradient**, the **kernel gradient** and the **output map
/// gradient** so that they can be used in the backward pass of the grouped convolution.
pub(super) fn group_gradients<'a, D: Dimension, S: DataMut<Elem = f32>, U: Data<Elem = f32>>(
    input_grad: &'a mut ArrayBase<S, D>,
    kernel_grad: &'a mut ArrayBase<S, D>,
    out_map_grad: &'a ArrayBase<U, D>,
    input: &'a ArrayBase<U, D>,
    kernel: &'a ArrayBase<U, D>,
    groups: usize,
) -> GroupedBackwardArgs<'a, D> {
    let input_grad_groups =
        input_grad.axis_chunks_iter_mut(Axis(1), input_grad.len_of(Axis(1)) / groups);
    let kernel_grad_groups =
        kernel_grad.axis_chunks_iter_mut(Axis(0), kernel_grad.len_of(Axis(0)) / groups);
    let out_map_grad_groups =
        out_map_grad.axis_chunks_iter(Axis(1), out_map_grad.len_of(Axis(1)) / groups);
    let input_groups = input.axis_chunks_iter(Axis(1), input.len_of(Axis(1)) / groups);
    let kernel_groups = kernel.axis_chunks_iter(Axis(0), kernel.len_of(Axis(0)) / groups);

    (
        input_grad_groups,
        kernel_grad_groups,
        out_map_grad_groups,
        input_groups,
        kernel_groups,
    )
}

pub(super) fn group_gradients_unary<
    'a,
    D: Dimension,
    S: DataMut<Elem = f32>,
    U: Data<Elem = f32>,
>(
    kernel_grad: &'a mut ArrayBase<S, D>,
    out_map_grad: &'a ArrayBase<U, D>,
    input: &'a ArrayBase<U, D>,
    kernel: &'a ArrayBase<U, D>,
    groups: usize,
) -> GroupedBackwardArgsUnary<'a, D> {
    let kernel_grad_groups =
        kernel_grad.axis_chunks_iter_mut(Axis(0), kernel_grad.len_of(Axis(0)) / groups);
    let out_map_grad_groups =
        out_map_grad.axis_chunks_iter(Axis(1), out_map_grad.len_of(Axis(1)) / groups);
    let input_groups = input.axis_chunks_iter(Axis(1), input.len_of(Axis(1)) / groups);
    let kernel_groups = kernel.axis_chunks_iter(Axis(0), kernel.len_of(Axis(0)) / groups);

    (
        kernel_grad_groups,
        out_map_grad_groups,
        input_groups,
        kernel_groups,
    )
}

/// Performs an **n-dimensional** convolution where **n** can be either *1*, *2* or *3*.
/// Do note that this function doesn't take into account *groups* nor *padding*. The padding is
/// assumed to be already applied to the input map when this function is called.
///
/// The resulting output map is stored in `output`.
/// # Arguments
///
/// * `input` - the input map
/// * `kernel` - the kernel
/// * `output` - the output map where the convolution result will be stored
/// * `stride` - the stride controls the stride for the cross-correlation
/// * `dilation` - the dilation controls the spacing between the kernel points
pub(super) fn convolution<D: Dimension, S: Data<Elem = f32>, T: DataMut<Elem = f32>>(
    input: &ArrayBase<S, D>,
    kernel: &ArrayBase<S, D>,
    output: &mut ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
) {
    let (flat_kernel, flat_input) = (
        flatten(kernel.view()),
        to_col(&input, kernel.shape(), stride, dilation),
    );
    let convolution_result = flat_kernel.dot(&flat_input);
    assign_to_output_map(output, convolution_result);
}

/// Performs the **back-propagation** for an an **n-dimensional** convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// # Arguments
/// * `input_grad` - the gradient of the input map
/// * `kernel_grad` - the gradient of the kernel
/// * `grad` - the incoming gradient **d_out**.
/// * `input` - the input map
/// * `kernel` - the kernel
/// * `padding` - the padding that must be taken into account while accumulating the input's
/// gradient
/// * `stride` - the stride
/// * `dilation` - the dilation
/// * `overwrite_input_grad`  - specifies the kind of accumulation operation to be performed on
/// the input's gradient
/// * `overwrite_kernel_grad` - specifies the kind of accumulation operation to be performed on
/// the kernel's gradient
#[allow(clippy::too_many_arguments)]
pub(super) fn convolution_backward<
    D: Dimension,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
    U: Data<Elem = f32>,
>(
    input_grad: &mut ArrayBase<S, D>,
    kernel_grad: &mut ArrayBase<S, D>,
    grad: &ArrayBase<T, D>,
    input: &ArrayBase<U, D>,
    kernel: &ArrayBase<T, D>,
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
    overwrite_input_grad: bool,
    overwrite_kernel_grad: bool,
) {
    // Flattens the incoming gradient.
    let gradient = permute_channels(grad.view());
    let gradient_as_standard = gradient.as_standard_layout();
    let flat_gradient = flatten(gradient_as_standard);

    // Computes the kernel's gradient.
    let kernel_gradient = flat_gradient
        .dot(&to_col(input, kernel.shape(), stride, dilation).t())
        .into_shape(kernel_grad.raw_dim())
        .unwrap();

    // Assigns the kernel's gradient.
    let kernel_grad_zip = Zip::from(kernel_grad).and(&kernel_gradient);
    if overwrite_kernel_grad {
        kernel_grad_zip
            .for_each(|kernel_grad_el, incoming_grad_el| *kernel_grad_el = *incoming_grad_el);
    } else {
        kernel_grad_zip
            .for_each(|kernel_grad_el, incoming_grad_el| *kernel_grad_el += *incoming_grad_el);
    }

    // Computes the input's gradient.
    let flat_kernel = flatten(kernel.view());
    let input_gradient = flat_kernel.t().dot(&flat_gradient);

    // If padding is not present, just assigns the gradient to the input.
    if padding.iter().all(|pad| *pad == 0) {
        assign_from_cols(input_grad, input_gradient, kernel.shape(), stride, dilation);
    } else {
        // If padding is present a buffer is needed to accumulate the input's incoming gradient.
        let mut buffer: Array<f32, D> =
            Array::zeros(padded_shape::<D>(input_grad.shape(), padding));
        assign_from_cols(
            &mut buffer,
            input_gradient,
            kernel.shape(),
            stride,
            dilation,
        );

        // The actual input's incoming gradient is extracted from the buffer and assigned.
        let actual_gradient = unpad(&buffer, padding);
        let input_gradient_zip = Zip::from(input_grad).and(actual_gradient);
        if overwrite_input_grad {
            input_gradient_zip
                .for_each(|input_grad_el, incoming_grad_el| *input_grad_el = *incoming_grad_el);
        } else {
            input_gradient_zip
                .for_each(|input_grad_el, incoming_grad_el| *input_grad_el += *incoming_grad_el);
        }
    }
}

/// Performs the **back-propagation** for an an **n-dimensional** convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// This function should be used in those circumstances in which the kernel is the only
/// differentiable variable, such as the first layer of a CNN module.
pub(super) fn convolution_unary_backward<
    D: Dimension,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
    U: Data<Elem = f32>,
>(
    kernel_grad: &mut ArrayBase<S, D>,
    grad: &ArrayBase<T, D>,
    input: &ArrayBase<U, D>,
    kernel: &ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
    overwrite_kernel_grad: bool,
) {
    // Flattens the incoming gradient.
    let gradient = permute_channels(grad.view());
    let gradient_as_standard = gradient.as_standard_layout();
    let flat_gradient = flatten(gradient_as_standard);

    // Computes the kernel's gradient.
    let kernel_gradient = flat_gradient
        .dot(&to_col(input, kernel.shape(), stride, dilation).t())
        .into_shape(kernel_grad.raw_dim())
        .unwrap();

    // Assigns the kernel's gradient.
    let kernel_grad_zip = Zip::from(kernel_grad).and(&kernel_gradient);
    if overwrite_kernel_grad {
        kernel_grad_zip
            .for_each(|kernel_grad_el, incoming_grad_el| *kernel_grad_el = *incoming_grad_el);
    } else {
        kernel_grad_zip
            .for_each(|kernel_grad_el, incoming_grad_el| *kernel_grad_el += *incoming_grad_el);
    }
}

/// Performs an **n-dimensional grouped** convolution where **n** can be either *1*, *2* or *3*.
///
/// Do note that this function doesn't take into account *padding*. The padding is
/// assumed to be already applied to the input map when this function is called.
///
/// # Arguments
/// * `input` - the input map
/// * `kernel` - the kernel
/// * `output` - the output map where the convolution result will be stored
/// * `stride` - the stride
/// * `dilation` - the dilation
/// * `groups` - the number of groups
pub(super) fn convolution_with_groups<D: Dimension>(
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    output: &mut Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
) {
    let (input_groups, kernel_groups, output_buffer_groups) =
        group_inputs(input, kernel, output, groups);
    kernel_groups
        .into_par_iter()
        .zip(input_groups.into_iter())
        .zip(output_buffer_groups.into_iter())
        .for_each(|((kernel, input), mut output)| {
            convolution(&input, &kernel, &mut output, stride, dilation);
        });
}

/// Performs the **back-propagation** for an an **n-dimensional** grouped convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// # Arguments
/// * `input_grad` - the gradient of the input map
/// * `kernel_grad` - the gradient of the kernel
/// * `grad` - the incoming gradient **d_out**.
/// * `input` - the input map
/// * `kernel` - the kernel
/// * `padding` - the padding that must be taken into account while accumulating the input's
/// * `stride` - the stride
/// * `dilation` - the dilation
/// * `groups` - the number of groups
/// * `overwrite_input_grad`  - specifies the kind of accumulation operation to be performed on
/// the input gradient
/// * `overwrite_kernel_grad` - specifies the kind of accumulation operation to be performed on
/// the kernel gradient
#[allow(clippy::too_many_arguments)]
pub(super) fn convolution_with_groups_backward<D: Dimension>(
    input_grad: &mut Array<f32, D>,
    kernel_grad: &mut Array<f32, D>,
    grad: &Array<f32, D>,
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    overwrite_input_grad: bool,
    overwrite_kernel_grad: bool,
) {
    let (input_grad_groups, kernel_grad_groups, grad_groups, input_groups, kernel_groups) =
        group_gradients(input_grad, kernel_grad, &grad, &input, &kernel, groups);

    grad_groups
        .into_par_iter()
        .zip(kernel_grad_groups.into_iter())
        .zip(input_grad_groups.into_iter())
        .zip(kernel_groups.into_iter())
        .zip(input_groups.into_iter())
        .for_each(
            |((((gradient, mut kernel_gradient), mut input_gradient), kernel), input)| {
                convolution_backward(
                    &mut input_gradient,
                    &mut kernel_gradient,
                    &gradient,
                    &input,
                    &kernel,
                    padding,
                    stride,
                    dilation,
                    overwrite_input_grad,
                    overwrite_kernel_grad,
                )
            },
        );
}

/// Performs the **back-propagation** for an an **n-dimensional** grouped convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// This function should be used in those circumstances in which the kernel is the only
/// differentiable variable, such as the first layer of a CNN module.
#[allow(clippy::too_many_arguments)]
pub(super) fn convolution_with_groups_unary_backward<D: Dimension>(
    kernel_grad: &mut Array<f32, D>,
    grad: &Array<f32, D>,
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    overwrite_kernel_grad: bool,
) {
    let (kernel_grad_groups, grad_groups, input_groups, kernel_groups) =
        group_gradients_unary(kernel_grad, &grad, &input, &kernel, groups);

    grad_groups
        .into_par_iter()
        .zip(kernel_grad_groups.into_iter())
        .zip(kernel_groups.into_iter())
        .zip(input_groups.into_iter())
        .for_each(|(((gradient, mut kernel_gradient), kernel), input)| {
            convolution_unary_backward(
                &mut kernel_gradient,
                &gradient,
                &input,
                &kernel,
                stride,
                dilation,
                overwrite_kernel_grad,
            )
        });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
