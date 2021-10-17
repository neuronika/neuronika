use super::{PaddingMode, ReflPad, ReplPad};
use ndarray::{
    iter::{AxisChunksIter, AxisChunksIterMut},
    linalg::general_mat_mul,
    Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, DataMut, Dimension, Ix2, Ix3, IxDyn,
    RawData, RemoveAxis, ShapeBuilder, Slice, ViewRepr, Zip,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Type Definitions for Grouped Convolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arguments Checkers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Checks that the arguments are correct for the given **convolution**. It verifies that the
/// `padding`, `stride` and `dilation` slices are of the right length; their length must match the
/// dimensionality of the convolution. It also check that `kernel` and `input` are of the same
/// dimension and that the kernel size, after dilation is applied, **is not bigger** that the actual
/// input size.
///
/// # Arguments
///
/// * `input_shape` - shape of the input map of the convolution.
///
/// * `kernel_shape` - shape of the kernel.
///
/// * `padding` - padding to be applied to the input.
///
/// * `stride` - stride for the cross-correlation.
///
/// * `dilation` - spacing between the kernel points.
pub(super) fn check_conv_args(
    input_shape: &[usize],
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) {
    // The type of convolution can be derived by considering the number of input's dimension
    // skipping the first two, that are the batch size and input channels. The first two axes of
    // the input are always for the batch size and the number of input channels.
    let convolution_dimension = input_shape.len() - 2;
    assert_eq!(
        convolution_dimension,
        padding.len(),
        "error: invalid padding {:?} for {}d conv.",
        padding,
        convolution_dimension
    );

    assert_eq!(
        convolution_dimension,
        stride.len(),
        "error: invalid stride {:?} for {}d conv.",
        stride,
        convolution_dimension
    );

    assert_eq!(
        convolution_dimension,
        dilation.len(),
        "error: invalid dilation {:?} for {}d conv.",
        dilation,
        convolution_dimension
    );

    assert_eq!(
        kernel_shape.len(),
        input_shape.len(),
        "error: invalid kernel's shape {:?} for {}d conv",
        &kernel_shape,
        convolution_dimension
    );

    // Checks that the kernel size, taking into account dilation, is suitable for the padded input.
    let dilated_kernel_size: Vec<usize> = kernel_shape
        .iter()
        .skip(2)
        .zip(dilation.iter())
        .map(|(kernel_size, dilation_component)| (kernel_size - 1) * dilation_component + 1)
        .collect();
    let padded_input_size: Vec<usize> = input_shape
        .iter()
        .skip(2)
        .zip(padding.iter())
        .map(|(input_size, padding_component)| input_size + padding_component * 2)
        .collect();

    padded_input_size
        .iter()
        .zip(dilated_kernel_size.iter())
        .for_each(|(padded_input_dim, dilated_kernel_dim)| {
            assert!(
                padded_input_dim >= dilated_kernel_dim,
                "error: calculated padded input size per channel: {:?}. Kernel size: {:?}. 
                The kernel size can't be greater than actual input size.",
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
///
/// * `input_shape` - the shape of the input map of the convolution.
///
/// * `kernel_shape` - the shape of the kernel.
///
/// * `groups` -  the connections between inputs and outputs.
pub(super) fn check_groups_args(input_shape: &[usize], kernel_shape: &[usize], groups: usize) {
    assert_eq!(
        input_shape[1] % groups,
        0,
        "error: in channels {} is not divisible by groups {}",
        input_shape[1],
        groups
    );
    assert_eq!(
        kernel_shape[0] % groups,
        0,
        "error: out channels {} is not divisible by groups {}",
        kernel_shape[0],
        groups
    );
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Auxiliary Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Computes the shape of the array resulting from the **n**-dimensional convolution
/// performed with the given parameters.
///
/// **n** can be either **1**, **2** or **3**
///
/// # Arguments
///
/// * `input_shape` - the shape of the input
///
/// * `kernel_shape` - the shape of the kernel
///
/// * `padding` - the padding around the input
///
/// * `stride` - the stride
///
/// * `dilation` - the dilation
///
/// ## 1-dimensional convolution
///
/// The **input** must be of shape **(N, Cin, L)**.
/// * **N** is the batch size.
/// * **Cin** is the number of input channels.
/// * **L** is the **length** of the input.
///
/// The **kernel** must be of shape **(Cout, Cin, Lk)**.
/// * **Cout** is the number of output channels.
/// * **Cin** is the number of input channels.
/// * **Lk** is the **length** of the kernel.
/// The resulting output shape will be **(N, Cout, Lout)**.
///
/// ## 2-dimensional convolution
///
/// The **input** must be of shape **(N, Cin, H, W)**.
/// * **N** is the batch size.
/// * **Cin** is the number of input channels.
/// * **H** is the **height** of the input.
/// * **W** is the **width** of the input.
///
/// The **kernel** must be of shape **(Cout, Cin, Hk, Wk)**.
/// * **Cout** is the number of output channels.
/// * **Cin** is the number of input channels.
/// * **Hk** is the **height** of the kernel.
/// * **Wk** is the **width** of the kernel.
///
/// The resulting output shape will be **(N, Cout, Hout, Wout)**.
///
/// ## 3-dimensional convolution
///
/// The **input** must be of shape **(N, Cin, D, H, W)**.
/// * **N** is the batch size.
/// * **Cin** is the number of input channels.
/// * **D** is the **depth** of the input.
/// * **H** is the **height** of the input.
/// * **W** is the **width** of the input.
///
/// The **kernel** must be of shape **(Cout, Cin, Dk,  Hk, Wk)**.
/// * **Cout** is the number of output channels.
/// * **Cin** is the number of input channels.
/// * **Dk** is the **depth** of the kernel.
/// * **Hk** is the **height** of the kernel.
/// * **Wk** is the **width** of the kernel.
///
/// The resulting output shape will be **(N, Cout, Dout, Hout, Wout)**.
pub(super) fn conv_out_shape<D: Dimension>(
    input_shape: &[usize],
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> D {
    let mut output_map_shape: D =
        conv_out_shape_padded(input_shape, kernel_shape, stride, dilation);
    // Applies the padding.
    itertools::izip!(
        output_map_shape.slice_mut().iter_mut().skip(2), // Skips batch size and out channels.
        padding,
        stride,
    )
    .for_each(|(output_map_dim, padding, stride)| *output_map_dim += 2 * padding / stride);
    output_map_shape
}

/// Computes the shape of the array resulting from the **n**-dimensional convolution
/// performed with the given parameters. `input_shape` is assumed to be the shape of an **already**
/// padded input.
///
/// # Arguments
///
/// * `input_shape` - the shape of the input.
///
/// * `kernel_shape` - the shape of the kernel.
///
/// * `stride` - the stride.
///
/// * `dilation` - the dilation.
fn conv_out_shape_padded<D: Dimension>(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> D {
    // Initialize the dimension to be all 0s.
    let mut output_map_shape = D::zeros(input_shape.len());
    // Sets the batch size. The batch size doesn't change.
    output_map_shape[0] = input_shape[0];
    // Sets the output channels.
    output_map_shape[1] = kernel_shape[0];
    // First two components of the shape are always
    // the batch size and channels.
    itertools::izip!(
        output_map_shape.slice_mut().iter_mut().skip(2), // Skips batch size and out channels.
        input_shape.iter().skip(2),                      // Skips batch size and out channels.
        kernel_shape.iter().skip(2),                     // Skips out channels and in channels.
        stride,
        dilation
    )
    .for_each(
        |(output_map_dim, input_dim, kernel_dim, stride, dilation)| {
            *output_map_dim = (input_dim - dilation * (kernel_dim - 1) - 1) / stride + 1
        },
    );
    output_map_shape
}

/// Computes the shape of the **input** after the padding is applied.
///
/// # Arguments
///
/// * `input_shape` - the shape of the input.
///
/// * `padding` - the padding around the input.
fn padded_shape<D: Dimension>(input_shape: &[usize], padding: &[usize]) -> D {
    let mut padded_input_shape = D::zeros(input_shape.len());
    padded_input_shape[0] = input_shape[0]; // Copy batch size.
    padded_input_shape[1] = input_shape[1]; // Copy input channels.
    padded_input_shape
        .slice_mut()
        .iter_mut()
        .skip(2)
        .zip(input_shape.iter().skip(2))
        .zip(padding.iter())
        .for_each(|((padded_dim, original_dim), padding)| *padded_dim = original_dim + 2 * padding);
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
    // Computes the final shape.
    let mut padded = {
        let padded_shape = padded_shape::<D>(array.shape(), padding);
        Array::<f32, D>::zeros(padded_shape)
    };

    let (padded_raw_dim, original_raw_dim) = (padded.raw_dim(), array.raw_dim());

    // This is the dimension of a single sample in the batch.
    let (padded_inner_dimensions, original_inner_dimensions) = (
        padded_raw_dim.slice().iter().skip(2),
        original_raw_dim.slice().iter().skip(2),
    );

    // The number of single samples in the batch.
    let outer_dimension: usize = original_raw_dim.slice().iter().take(2).product();

    // Reshapes by removing an axis, so that all samples, from all channels, can be iterated on and
    // padded.
    let (mut padded_view_dimension, mut original_view_dimension): (
        <D as Dimension>::Smaller,
        <D as Dimension>::Smaller,
    ) = (
        <D as Dimension>::Smaller::zeros(padded_raw_dim.ndim() - 1),
        <D as Dimension>::Smaller::zeros(original_raw_dim.ndim() - 1),
    );
    padded_view_dimension[0] = outer_dimension;
    original_view_dimension[0] = outer_dimension;

    padded_view_dimension
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

    let (mut padded_view_mut, original_view) = (
        padded.view_mut().into_shape(padded_view_dimension).unwrap(),
        array.view().into_shape(original_view_dimension).unwrap(),
    );

    padded_view_mut
        .outer_iter_mut()
        .into_par_iter()
        .zip(original_view.outer_iter())
        .for_each(|(mut pad_sample, original_sample)| {
            padding_mode.pad_inplace(&mut pad_sample, &original_sample, padding)
        });
    padded
}

/// Returns an **un-padded** view of `array`. This method is supposed to be used only with arrays
/// of dimension greater or equal than *3*. The length of the padding slice must be *dim - 2* where
/// *dim* is the array's dimensionality. Returns an un-padded view.
///
/// # Arguments
///
/// * `array` - array to strip the padding off to.
///
/// * `padding` - padding.
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
///
/// * `input` - input array.
///
/// * `window_shape` - shape of each of the windows.
///
/// * `stride` - stride.
///
/// * `dilation` - spacing between each element of the windows.
fn compute_rolling_window_shape<D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Vec<usize> {
    let mut rolling_window_indices_shape: D =
        conv_out_shape_padded(input.shape(), window_shape, stride, dilation);
    rolling_window_indices_shape[1] = 1;
    rolling_window_indices_shape
        .slice()
        .iter()
        .chain(window_shape.iter().skip(1))
        .cloned()
        .collect()
}

/// Computes the strides of a rolling window view.
///
/// # Arguments
///
/// * `input` - input array.
///
/// * `stride` - stride.
///
/// * `dilation` - spacing between each element of the windows.
fn compute_rolling_window_strides<D: Dimension, S: Data<Elem = f32>>(
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
    let rolling_window_strides: Vec<isize> = input
        .strides()
        .iter()
        .skip(1) // Skip out channels
        .enumerate()
        .map(|(i, input_stride)| {
            if i < 1 {
                *input_stride
            } else {
                *input_stride * (dilation[i - 1] as isize)
            }
        })
        .collect();
    indexing_strides
        .iter()
        .chain(rolling_window_strides.iter())
        .map(|s| *s as usize)
        .collect()
}

/// Returns an immutable **rolling window view** of the input array.
///
/// # Arguments
///
/// * `input` - input array.
///
/// * `window_shape` - shape of each of the windows.
///
/// * `stride` - stride.
///
/// * `dilation` - spacing between each element of the windows.
fn as_windows<'a, D: Dimension, S: Data<Elem = f32>>(
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
///
/// * `input` - input array.
///
/// * `window_shape` - shape of each of the windows.
///
/// * `stride` - stride.
///
/// * `dilation` - spacing between each element of the windows.
fn as_windows_mut<'a, D: Dimension, S: DataMut<Elem = f32>>(
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

/// Puts the axis corresponding to the output channels of the feature map `array`
/// in the last position and returns the input with swapped axes.
///
/// # Arguments
///
/// `array` - array to permute the channels in.
fn permute_channels<D: Dimension>(
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

/// Computes the shapes of **sig2col**, **im2col** and **vol2col**.
///
/// # Arguments
///
/// * `input` - input array.
///
/// * `kernel_shape` - shape of the kernel.
///
/// * `padding` - padding to be applied to `input`.
///
/// * `stride` - stride.
///
/// * `dilation` - dilation.
fn columns_shape<D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Ix3 {
    let output_map_shape =
        conv_out_shape_padded::<D>(input.shape(), kernel_shape, stride, dilation);
    let mut columns_shape = Ix3::zeros(3);
    columns_shape[0] = output_map_shape[0];
    columns_shape[1] = output_map_shape.slice().iter().skip(2).product();
    columns_shape[2] = kernel_shape.iter().skip(1).product();

    columns_shape
}

/// Computes a shape from the array in input so that only the dimension of axis 0 is preserved.
///
/// # Arguments
///
/// ` array` - array to be flattened.
///
fn flat_shape<D: Dimension, S: RawData>(array: &ArrayBase<S, D>) -> Ix2 {
    let (original_shape, mut flat_shape) = (array.raw_dim(), Ix2::zeros(2));
    flat_shape[0] = original_shape[0];
    flat_shape[1] = original_shape.slice().iter().skip(1).product();
    flat_shape
}

/// Assigns to the **n**-dimensional feature map's gradient `dest` the **2**-dimensional
/// array `columns`. This method encapsulates the functionalities of **col2sig**, **col2im** and
/// **col2vol**.
///
/// **n** can be either 3, 4 or 5.
///
/// # Arguments
///
/// * `dest` - output map of destination.
///
/// * `columns` - gradient in columns format.
///
/// * `kernel_shape` - shape of the kernel.
///
/// * `stride` - strides.
///
/// * `dilation` - dilation.
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

/// Partitions the **flattened input**, the **flattened kernel** and the **output map**
/// so that they can be used in a grouped convolution.
///
/// # Arguments
///
/// * `input` - input of the convolution.
///
/// * `kernel` - kernel of the convolution.
///
/// * `output_map` - output map.
///
/// * `groups` - number of groups.
fn group_inputs<'a, D: Dimension, S: Data<Elem = f32>, T: DataMut<Elem = f32>>(
    input: &'a ArrayBase<S, D>,
    kernel: &'a ArrayBase<S, D>,
    output_map: &'a mut ArrayBase<T, D>,
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
    // Splits the output map along the channels.
    let output_map_groups =
        output_map.axis_chunks_iter_mut(Axis(1), output_map.len_of(Axis(1)) / groups);

    (input_groups, kernel_groups, output_map_groups)
}

/// Partitions the **input gradient**, the **kernel gradient** and the **output map
/// gradient** so that they can be used in the backward pass of the grouped convolution.
///
/// # Arguments
///
/// * `input_gradient` - gradient of the input.
///
/// * `kernel_grad` - gradient of the kernel.
///
/// * `output_map_grad` - gradient of the output map.
///
/// * `input` - input of the convolution.
///
/// * `kernel` - kernel of the convolution.
///
/// * `groups` - number of groups.
pub(super) fn group_gradients<'a, D: Dimension, S: DataMut<Elem = f32>, U: Data<Elem = f32>>(
    input_grad: &'a mut ArrayBase<S, D>,
    kernel_grad: &'a mut ArrayBase<S, D>,
    output_map_grad: &'a ArrayBase<U, D>,
    input: &'a ArrayBase<U, D>,
    kernel: &'a ArrayBase<U, D>,
    groups: usize,
) -> GroupedBackwardArgs<'a, D> {
    let input_grad_groups =
        input_grad.axis_chunks_iter_mut(Axis(1), input_grad.len_of(Axis(1)) / groups);
    let kernel_groups = kernel.axis_chunks_iter(Axis(0), kernel.len_of(Axis(0)) / groups);
    let (kernel_grad_groups, output_map_grad_groups, input_groups) =
        group_gradients_unary(kernel_grad, output_map_grad, input, groups);

    (
        input_grad_groups,
        kernel_grad_groups,
        output_map_grad_groups,
        input_groups,
        kernel_groups,
    )
}

/// Partitions the he **kernel gradient** and the **output map gradient** so that they can be used
/// in the backward pass of the grouped unary convolution.
///
/// # Arguments
///
/// * `kernel_grad` - gradient of the kernel.
///
/// * `output_map_grad` - gradient of the output map.
///
/// * `input` - input of the convolution.
///
/// * `groups` - number of groups.
pub(super) fn group_gradients_unary<
    'a,
    D: Dimension,
    S: DataMut<Elem = f32>,
    U: Data<Elem = f32>,
>(
    kernel_grad: &'a mut ArrayBase<S, D>,
    output_map_grad: &'a ArrayBase<U, D>,
    input: &'a ArrayBase<U, D>,
    groups: usize,
) -> GroupedBackwardArgsUnary<'a, D> {
    let kernel_grad_groups =
        kernel_grad.axis_chunks_iter_mut(Axis(0), kernel_grad.len_of(Axis(0)) / groups);
    let output_map_grad_groups =
        output_map_grad.axis_chunks_iter(Axis(1), output_map_grad.len_of(Axis(1)) / groups);
    let input_groups = input.axis_chunks_iter(Axis(1), input.len_of(Axis(1)) / groups);

    (kernel_grad_groups, output_map_grad_groups, input_groups)
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolutions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Performs an **n-dimensional** convolution where **n** can be either *1*, *2* or *3*.
/// Do note that this function doesn't take into account *groups* nor *padding*. The padding is
/// assumed to be already applied to the input map when this function is called.
///
/// The resulting output map is stored in `output`.
///
/// # Arguments
///
/// * `input` - input map.
///
/// * `kernel` - kernel.
///
/// * `output` - output map where the convolution result will be stored.
///
/// * `stride` - stride controls the stride for the cross-correlation.
///
/// * `dilation` - dilation controls the spacing between the kernel points.
pub(super) fn convolution<
    D: Dimension + RemoveAxis,
    S: Data<Elem = f32>,
    T: DataMut<Elem = f32>,
    U: Data<Elem = f32>,
>(
    input: &ArrayBase<S, D>,
    kernel: &ArrayBase<U, D>,
    output: &mut ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
) {
    let (kernel_shape, flattened_kernel) = (
        kernel.shape(),
        kernel.view().into_shape(flat_shape(kernel)).unwrap(),
    );

    let input_windows = as_windows(input, kernel_shape, stride, dilation);
    let input_columns = input_windows
        .to_shape(columns_shape(input, kernel_shape, stride, dilation))
        .unwrap();

    Zip::from(input_columns.axis_iter(Axis(0)))
        .and(output.axis_iter_mut(Axis(0)))
        .par_for_each(|input_sample_columns, output_sample| {
            let flat_shape = flat_shape(&output_sample);
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

/// Performs the **back-propagation** for an an **n-dimensional** convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// # Arguments
///
/// * `input_grad` - gradient of the input map.
///
/// * `grad` -  incoming gradient **d_out**.
///
/// * `kernel` -  kernel.
///
/// * `padding` - padding that must be taken into account while accumulating the input's
/// gradient.
///
/// * `stride` - stride.
///
/// * `dilation` - the dilation.
///
/// * `overwrite_input_grad`  - specifies the kind of accumulation operation to be performed on
/// the input's gradient.
pub(super) fn convolution_backward_input<
    D: Dimension + RemoveAxis,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
>(
    input_grad: &mut ArrayBase<S, D>,
    grad: &ArrayBase<T, D>,
    kernel: &ArrayBase<T, D>,
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
    overwrite_input_grad: bool,
) {
    let (kernel_shape, flattened_kernel, grad_shape) = (
        kernel.shape(),
        kernel.view().into_shape(flat_shape(kernel)).unwrap(),
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
            let gradient_sample_flat_shape = flat_shape(&gradient_sample);
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

    if padding.iter().all(|pad| *pad == 0) {
        assign_from_cols(input_grad, buffer, kernel_shape, stride, dilation);
    } else {
        let mut padded_buffer: Array<f32, D> =
            Array::zeros(padded_shape::<D>(input_grad.shape(), padding));
        assign_from_cols(&mut padded_buffer, buffer, kernel.shape(), stride, dilation);

        // The actual input's incoming gradient is extracted from the buffer and assigned.
        let actual_gradient = unpad(&padded_buffer, padding);
        let input_gradient_zip = Zip::from(input_grad).and(actual_gradient);
        if overwrite_input_grad {
            input_gradient_zip
                .par_for_each(|input_grad_el, incoming_grad_el| *input_grad_el = *incoming_grad_el);
        } else {
            input_gradient_zip.par_for_each(|input_grad_el, incoming_grad_el| {
                *input_grad_el += *incoming_grad_el
            });
        }
    }
}

/// Performs the **back-propagation** for an an **n-dimensional** convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// This function should be used in those circumstances in which the kernel is the only
/// differentiable variable, such as the first layer of a CNN module.
///
/// # Arguments
///
/// * `kernel_grad` - gradient of the kernel.
///
/// * `grad` -  incoming gradient **d_out**.
///
/// * `input` -  input map.
///
/// * `kernel` -  kernel.
///
/// * `padding` - padding that must be taken into account while accumulating the input's
/// gradient.
///
/// * `stride` - stride.
///
/// * `dilation` - the dilation.
///
/// * `overwrite_input_grad`  - specifies the kind of accumulation operation to be performed on
/// the input's gradient.
///
/// * `overwrite_kernel_grad` - specifies the kind of accumulation operation to be performed on
/// the kernel's gradient.
pub(super) fn convolution_backward_kernel<
    D: Dimension + RemoveAxis,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
>(
    kernel_grad: &mut ArrayBase<S, D>,
    grad: &ArrayBase<T, D>,
    input: &ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
    overwrite_kernel_grad: bool,
) {
    let gradient = permute_channels(grad.view());
    let flat_gradient = gradient.to_shape(flat_shape(&gradient)).unwrap();
    let beta = if overwrite_kernel_grad { 0. } else { 1. };
    let kernel_shape = kernel_grad.raw_dim();

    let input_windows = as_windows(input, kernel_shape.slice(), stride, dilation);
    let columns_shape = columns_shape(input, kernel_shape.slice(), stride, dilation);

    let mut matrix_shape = Ix2::zeros(2);
    matrix_shape[0] = columns_shape[0] * columns_shape[1];
    matrix_shape[1] = columns_shape[2];
    let input_matrix = input_windows.to_shape(matrix_shape).unwrap();

    general_mat_mul(
        1.,
        &flat_gradient,
        &input_matrix,
        beta,
        &mut kernel_grad
            .view_mut()
            .into_shape((flat_gradient.shape()[0], matrix_shape[1]))
            .unwrap(),
    );
}

/// Performs an **n-dimensional grouped** convolution where **n** can be either *1*, *2* or *3*.
///
/// Do note that this function doesn't take into account *padding*. The padding is
/// assumed to be already applied to the input map when this function is called.
///
/// # Arguments
///
/// * `input` -  input map.
///
/// * `kernel` -  kernel.
///
/// * `output` -  output map where the convolution result will be stored.
///
/// * `stride` -  stride.
///
/// * `dilation` -  dilation.
///
/// * `groups` -  number of groups.
pub(super) fn convolution_with_groups<D: Dimension + RemoveAxis>(
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
///
/// * `input_grad` - gradient of the input map.
///
/// * `kernel_grad` -  gradient of the kernel.
///
/// * `grad` -  incoming gradient **d_out**.
///
/// * `input` -  input map.
///
/// * `kernel` -  kernel.
///
/// * `padding` -  padding that must be taken into account while accumulating the input's gradient.
///
/// * `stride` -  stride.
///
/// * `dilation` -  dilation.
///
/// * `groups` -  number of groups.
///
/// * `overwrite_input_grad`  - specifies the kind of accumulation operation to be performed on
/// the input gradient.
///
/// * `overwrite_kernel_grad` - specifies the kind of accumulation operation to be performed on
/// the kernel gradient.
#[allow(clippy::too_many_arguments)]
pub(super) fn convolution_with_groups_backward<D: Dimension + RemoveAxis>(
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
        group_gradients(input_grad, kernel_grad, grad, input, kernel, groups);

    grad_groups
        .into_par_iter()
        .zip(kernel_grad_groups.into_iter())
        .zip(input_grad_groups.into_iter())
        .zip(kernel_groups.into_iter())
        .zip(input_groups.into_iter())
        .for_each(
            |((((gradient, mut kernel_gradient), mut input_gradient), kernel), input)| {
                convolution_backward_kernel(
                    &mut kernel_gradient,
                    &gradient,
                    &input,
                    stride,
                    dilation,
                    overwrite_kernel_grad,
                );
                convolution_backward_input(
                    &mut input_gradient,
                    &gradient,
                    &kernel,
                    padding,
                    stride,
                    dilation,
                    overwrite_input_grad,
                )
            },
        );
}

/// Performs the **back-propagation** for an an **n-dimensional** grouped convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// This function should be used in those circumstances in which the kernel is the only
/// differentiable variable, such as the first layer of a CNN module.
///
// # Arguments
///
/// * `kernel_grad` -  gradient of the kernel.
///
/// * `grad` -  incoming gradient **d_out**.
///
/// * `input` -  input map.
///
/// * `kernel` -  kernel.
///
/// * `stride` -  stride.
///
/// * `dilation` -  dilation.
///
/// * `groups` -  number of groups.
///
/// * `overwrite_input_grad`  - specifies the kind of accumulation operation to be performed on
/// the input gradient.
///
/// * `overwrite_kernel_grad` - specifies the kind of accumulation operation to be performed on
/// the kernel gradient.
#[allow(clippy::too_many_arguments)]
pub(super) fn convolution_with_groups_unary_backward<D: Dimension + RemoveAxis>(
    kernel_grad: &mut Array<f32, D>,
    grad: &Array<f32, D>,
    input: &Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    overwrite_kernel_grad: bool,
) {
    let (kernel_grad_groups, grad_groups, input_groups) =
        group_gradients_unary(kernel_grad, grad, input, groups);

    grad_groups
        .into_par_iter()
        .zip(kernel_grad_groups.into_iter())
        .zip(input_groups.into_iter())
        .for_each(|((gradient, mut kernel_gradient), input)| {
            convolution_backward_kernel(
                &mut kernel_gradient,
                &gradient,
                &input,
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
