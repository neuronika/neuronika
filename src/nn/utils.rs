use ndarray::{
    iter::{AxisChunksIter, AxisChunksIterMut},
    Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, DataMut, Dimension, Ix1, Ix2, Ix3,
    IxDyn, RawData, ShapeBuilder, Slice, ViewRepr, Zip,
};

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

/// Checks that the arguments are correct
/// for the given **convolution**.
fn check_conv_args(
    input_shape: &[usize],
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) {
    // The type of convolution can be derived by
    // considering the number of input's dimension
    // skipping the batch size and input channels.
    // First two axes of input are always for batch size
    // and input channels.
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

/// Checks that the arguments are correct
/// for the given **grouped convolution**.
fn check_groups_args(input_shape: &[usize], kernel_shape: &[usize], groups: usize) {
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
fn conv_out_shape<D: Dimension>(
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
        map_shape_slice.iter_mut().skip(2), // Skips bacth size and out channels.
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

/// Returns a **rolling window view** of the input array.
///
/// # Arguments
/// * `input` - input array
/// * `window_shape` - the shape of each of the windows
/// * `win_indices_shape` - the number of rolling windows
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
fn as_windows<'a, D: Dimension>(
    input: &Array<f32, D>,
    window_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayView<'a, f32, IxDyn> {
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
    let mut win_indices_shape: D =
        conv_out_shape(input.shape(), window_shape, padding, stride, dilation);
    win_indices_shape[1] = 1;
    let new_shape: Vec<usize> = win_indices_shape
        .slice()
        .iter()
        .chain(window_shape.iter().skip(1))
        .cloned()
        .collect();
    let strides: Vec<usize> = indexing_strides
        .iter()
        .chain(window_strides.iter())
        .map(|s| *s as usize)
        .collect();

    unsafe { ArrayView::from_shape_ptr(new_shape.strides(strides), input.as_ptr()) }
}

/// Returns a **mutable rolling window view** of the input array.
///
/// # Arguments
/// * `input` - input array
/// * `window_shape` - the shape of each of the windows
/// * `win_indices_shape` - the number of rolling windows.
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
fn as_windows_mut<'a, D: Dimension, S: DataMut<Elem = f32>>(
    input: &mut ArrayBase<S, D>,
    window_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayViewMut<'a, f32, IxDyn> {
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
    let mut win_indices_shape: D =
        conv_out_shape(input.shape(), window_shape, padding, stride, dilation);
    win_indices_shape[1] = 1;
    let new_shape: Vec<usize> = win_indices_shape
        .slice()
        .iter()
        .chain(window_shape.iter().skip(1))
        .cloned()
        .collect();
    let strides: Vec<usize> = indexing_strides
        .iter()
        .chain(window_strides.iter())
        .map(|s| *s as usize)
        .collect();

    unsafe { ArrayViewMut::from_shape_ptr(new_shape.strides(strides), input.as_mut_ptr()) }
}

/// Computes **sig2col**, **im2col** and **vol2col**.
///
/// # Arguments
///
/// * `input` - input array
/// * `window_shape` - the shape of the kernel
/// * `padding` - the padding to be applied to `input`
/// * `stride` - the stride.
/// * `dilation` - the dilation
pub fn to_col<D: Dimension>(
    input: &Array<f32, D>,
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Array<f32, Ix2> {
    let mut o_shape = conv_out_shape::<D>(input.shape(), kernel_shape, padding, stride, dilation);
    o_shape[1] = 1;
    let (im2col_h, im2col_w): (usize, usize) = {
        (
            kernel_shape.iter().skip(1).product(),
            o_shape.slice().iter().product(),
        )
    };
    as_windows(&input, kernel_shape, padding, stride, dilation)
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

/// Produces a zeroed vector of `usize` of the right lenght used to convolve
/// without any padding.
fn no_padding<D: Dimension>(input_raw_dim: D) -> Vec<usize> {
    input_raw_dim
        .slice()
        .iter()
        .take(input_raw_dim.ndim() - 2)
        .map(|_| 0)
        .collect()
}

/// Assigns the **2-dimensional** convolution result to the **n-dimensional** feature map.
fn assign_to_output_map<D: Dimension>(out_map: &mut Array<f32, D>, flat_result: Array<f32, Ix2>) {
    let batch_size = out_map.shape()[0];
    let mut sample_size = out_map.raw_dim();
    sample_size[0] = 1;

    let convolved_samples =
        flat_result.axis_chunks_iter(Axis(1), flat_result.len_of(Axis(1)) / batch_size);
    let samples = out_map.axis_chunks_iter_mut(Axis(0), 1);

    samples
        .into_iter()
        .zip(convolved_samples.into_iter())
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

/// Partitions the **flattened input**, the **flattened kernel** and the **output map**
/// so that they can be used in a grouped convolution.
///
/// Note that `flat_kernel` and `flat_input` can be obtained by using respectively the functions
/// `flatten` and `to_col`.
fn group_inputs<
    'a,
    D: Dimension,
    S: Data<Elem = f32>,
    T: Data<Elem = f32>,
    U: DataMut<Elem = f32>,
>(
    flat_input: &'a ArrayBase<S, Ix2>,
    flat_kernel: &'a ArrayBase<T, Ix2>,
    out_map: &'a mut ArrayBase<U, D>,
    groups: usize,
) -> (
    AxisChunksIter<'a, f32, Ix2>,
    AxisChunksIter<'a, f32, Ix2>,
    AxisChunksIterMut<'a, f32, D>,
) {
    let input_group_size = flat_input.len_of(Axis(0)) / groups;
    let kernel_group_size = flat_kernel.len_of(Axis(0)) / groups;
    let out_map_group_size = out_map.len_of(Axis(0)) / groups;

    let input_groups = flat_input.axis_chunks_iter(Axis(0), input_group_size);
    let kernel_groups = flat_kernel.axis_chunks_iter(Axis(0), kernel_group_size);
    let out_map_groups = out_map.axis_chunks_iter_mut(Axis(0), out_map_group_size);

    (input_groups, kernel_groups, out_map_groups)
}

/// Iterators needed for the **backward pass** of a grouped convolution.
type GroupedBackwardArgs<'a, D> = (
    AxisChunksIterMut<'a, f32, D>,
    AxisChunksIterMut<'a, f32, D>,
    AxisChunksIter<'a, f32, Ix2>,
    AxisChunksIter<'a, f32, Ix2>,
    AxisChunksIter<'a, f32, Ix2>,
);
/// Partitions the **input gradient**, the **kernel gradient** and the **flattened output map
/// gradient** so that they can be used in the backward pass of the grouped convolution.
///
/// Note that `flat_out_map_grad`  can be obtained by using the function `flatten` .
fn group_gradients<
    'a,
    D: Dimension,
    S: DataMut<Elem = f32>,
    T: DataMut<Elem = f32>,
    U: Data<Elem = f32>,
    V: Data<Elem = f32>,
    Z: Data<Elem = f32>,
>(
    input_grad: &'a mut ArrayBase<S, D>,
    kernel_grad: &'a mut ArrayBase<T, D>,
    flat_out_map_grad: &'a ArrayBase<U, Ix2>,
    flat_input: &'a ArrayBase<V, Ix2>,
    flat_kernel: &'a ArrayBase<Z, Ix2>,
    groups: usize,
) -> GroupedBackwardArgs<'a, D> {
    let input_grad_group_size = input_grad.len_of(Axis(1)) / groups;
    let kernel_grad_group_size = kernel_grad.len_of(Axis(0)) / groups;
    let flat_out_map_grad_group_size = flat_out_map_grad.len_of(Axis(0)) / groups;
    let input_group_size = flat_input.len_of(Axis(0)) / groups;
    let kernel_group_size = flat_kernel.len_of(Axis(0)) / groups;

    let input_grad_groups = input_grad.axis_chunks_iter_mut(Axis(1), input_grad_group_size);
    let kernel_grad_groups = kernel_grad.axis_chunks_iter_mut(Axis(0), kernel_grad_group_size);
    let flat_out_map_grad_groups =
        flat_out_map_grad.axis_chunks_iter(Axis(0), flat_out_map_grad_group_size);
    let input_groups = flat_input.axis_chunks_iter(Axis(0), input_group_size);
    let kernel_groups = flat_kernel.axis_chunks_iter(Axis(0), kernel_group_size);

    (
        input_grad_groups,
        kernel_grad_groups,
        flat_out_map_grad_groups,
        input_groups,
        kernel_groups,
    )
}

// Creates a buffer used to store the intermediate result of a **grouped** convolution.
fn create_buffer(
    flat_input: &Array<f32, Ix2>,
    flat_kernel: &ArrayView<f32, Ix2>,
) -> Array<f32, Ix2> {
    Array::<f32, Ix2>::zeros((flat_kernel.shape()[0], flat_input.shape()[1]))
}

/// Performs an **n-dimensional** convolution where **n** can be either *1*, *2* or *3*.
/// Do note that this function doesn't take into account *padding* nor *groups*.
///
/// The resulting output map is stored in `output`.
fn convolution<D: Dimension>(
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    output: &mut Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
) {
    let no_pad = no_padding(input.raw_dim());
    check_conv_args(input.shape(), kernel.shape(), &no_pad, stride, dilation);
    let (flat_kernel, flat_input) = (
        flatten(kernel.view()),
        to_col(&input, kernel.shape(), &no_pad, stride, dilation),
    );

    let convolution_result = flat_kernel.dot(&flat_input);
    assign_to_output_map(output, convolution_result);
}

/// Performs the **backpropagation** for an an **n-dimensional** convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// Do note that this function doesn't take into account *padding* nor *groups*.
fn convolution_backward<D: Dimension, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
    input_grad: &mut ArrayBase<S, D>,
    kernel_grad: &mut ArrayBase<S, D>,
    grad: &ArrayBase<T, D>,
    input: &Array<f32, D>,
    kernel: &ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
) {
    let no_pad = no_padding(input_grad.raw_dim());

    let gradient = permute_channels(grad.view());
    let gradient_as_standard = gradient.as_standard_layout();
    let flat_gradient = flatten(gradient_as_standard);

    let kernel_gradient = flat_gradient
        .dot(&to_col(input, kernel.shape(), &no_pad, stride, dilation).t())
        .into_shape(kernel_grad.raw_dim())
        .unwrap();
    Zip::from(kernel_grad)
        .and(&kernel_gradient)
        .for_each(|kernel_grad_el, incoming_grad_el| *kernel_grad_el = *incoming_grad_el);

    let flat_kernel = flatten(kernel.view());
    let input_gradient = flat_kernel.t().dot(&flat_gradient);
    let mut gradient_window_view =
        as_windows_mut(input_grad, kernel.shape(), &no_pad, stride, dilation);
    let input_gradient_window_view = input_gradient
        .into_shape(gradient_window_view.raw_dim())
        .unwrap();

    Zip::from(&mut gradient_window_view)
        .and(&input_gradient_window_view)
        .par_for_each(|input_gradient_el, incoming_gradient_el| {
            *input_gradient_el += *incoming_gradient_el
        });
}

/// Performs a **grouped** convolution.
fn convolution_with_groups<D: Dimension>(
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    output: &mut Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
) {
    let no_pad = no_padding(input.raw_dim());
    let (input_shape, kernel_shape) = (input.shape(), kernel.shape());
    check_conv_args(input_shape, kernel_shape, &no_pad, stride, dilation);
    check_groups_args(input_shape, kernel_shape, groups);

    let (flat_kernel, flat_input) = {
        let mut kernel_shape = kernel.shape().to_vec();
        kernel_shape[1] *= groups;
        (
            flatten(kernel.view()),
            to_col(input, &kernel_shape, &no_pad, stride, dilation),
        )
    };
    let mut output_buffer = create_buffer(&flat_input, &flat_kernel);

    let (input_groups, kernel_groups, output_buffer_groups) =
        group_inputs(&flat_input, &flat_kernel, &mut output_buffer, groups);
    kernel_groups
        .into_iter()
        .zip(input_groups.into_iter())
        .zip(output_buffer_groups.into_iter())
        .for_each(|((kernel, input), output)| {
            let conv_output = kernel.dot(&input);
            Zip::from(output)
                .and(&conv_output)
                .for_each(|output_el, incoming_result| *output_el = *incoming_result);
        });
    assign_to_output_map(output, output_buffer)
}

/// Performs the backward pass for a **grouped convolution**.
#[allow(clippy::too_many_arguments)]
pub fn convolution_with_groups_backward<D: Dimension>(
    input_grad: &mut Array<f32, D>,
    kernel_grad: &mut Array<f32, D>,
    grad: &Array<f32, D>,
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
) {
    let no_pad = no_padding(input_grad.raw_dim());

    let gradient = permute_channels(grad.view());
    let gradient_as_standard = gradient.as_standard_layout();
    let flat_gradient = flatten(gradient_as_standard);

    let flat_input = {
        let mut kernel_shape = kernel.shape().to_vec();
        kernel_shape[1] *= groups;
        to_col(&input, &kernel_shape, &no_pad, stride, dilation)
    };
    let flat_kernel = flatten(kernel.view());

    let (input_grad_groups, kernel_grad_groups, grad_groups, input_groups, kernel_groups) =
        group_gradients(
            input_grad,
            kernel_grad,
            &flat_gradient,
            &flat_input,
            &flat_kernel,
            groups,
        );

    grad_groups
        .into_par_iter()
        .zip(kernel_grad_groups.into_iter())
        .zip(input_grad_groups.into_iter())
        .zip(kernel_groups.into_iter())
        .zip(input_groups.into_iter())
        .for_each(
            |(
                (((flat_gradient, kernel_gradient), mut input_gradient), flat_kernel),
                flat_input,
            )| {
                let incoming_input_gradient = flat_kernel.t().dot(&flat_gradient);
                let mut input_rolling_window_mut = as_windows_mut(
                    &mut input_gradient,
                    kernel_gradient.shape(),
                    &no_pad,
                    stride,
                    dilation,
                );
                let reshaped_incoming_input_gradient = incoming_input_gradient
                    .into_shape(input_rolling_window_mut.raw_dim())
                    .unwrap();

                Zip::from(&mut input_rolling_window_mut)
                    .and(&reshaped_incoming_input_gradient)
                    .for_each(|input_gradient_el, incoming_gradient_el| {
                        *input_gradient_el += *incoming_gradient_el
                    });

                let incoming_kernel_gradient = flat_gradient
                    .dot(&flat_input.t())
                    .into_shape(kernel_gradient.raw_dim())
                    .unwrap();
                Zip::from(kernel_gradient)
                    .and(&incoming_kernel_gradient)
                    .for_each(|kernel_gradient_el, incoming_gradient_el| {
                        *kernel_gradient_el = *incoming_gradient_el
                    });
            },
        );
}

/// A `ndarray::Dimension` that supports **reflective padding**.
pub trait ReflPad: Dimension {
    fn reflection_pad(input: &Array<f32, Self>, padding: &[usize]) -> Array<f32, Self>;
}

/// A `ndarray::Dimension` that supports **replicative padding**.
pub trait ReplPad: Dimension {
    fn replication_pad(input: &Array<f32, Self>, padding: &[usize]) -> Array<f32, Self>;
}
/// Pads an **n**-dimensional Array with a constant value.
///
/// # Arguments
/// * `input` - the array to be padded
/// * `padding` - a slice specifying for each dimension the amount of padding
/// * `value` - the value for the padding
///
/// # Examples
///
/// ```
/// use neuronika::nn;
///
/// let arr = ndarray::array![
///    [1., 2., 3.],
///    [4., 5., 6.],
///    [7., 8., 9.]
/// ];
/// let padded = nn::utils::constant_pad(&arr, &[1, 1], 0.);
/// let result = ndarray::array![
///    [0., 0., 0., 0., 0.],
///    [0., 1., 2., 3., 0.],
///    [0., 4., 5., 6., 0.],
///    [0., 7., 8., 9., 0.],
///    [0., 0., 0., 0., 0.]
/// ];
///
/// assert_eq!(padded, result);
/// ```
pub fn constant_pad<D>(input: &Array<f32, D>, padding: &[usize], val: f32) -> Array<f32, D>
where
    D: Dimension,
{
    let padded_shape = {
        let mut padded_shape = input.raw_dim();
        padded_shape
            .slice_mut()
            .iter_mut()
            .zip(padding.iter())
            .for_each(|(ax_len, pad)| *ax_len += pad * 2);
        padded_shape
    };
    let mut padded = Array::from_elem(padded_shape, val);
    let mut orig_portion = padded.view_mut();
    orig_portion.slice_each_axis_inplace(|ax| {
        let (ax_index, ax_len) = (ax.axis.index(), input.len_of(ax.axis));
        let range = {
            if padding[ax_index] != 0 {
                padding[ax_index] as isize..-(padding[ax_index] as isize)
            } else {
                0..ax_len as isize
            }
        };
        Slice::from(range)
    });
    orig_portion.assign(input);
    padded
}

/// Pads the input array using the **reflection** of the input boundary.
///
/// Only **1**, **2** and **3** dimensional arrays support reflective padding.
///
/// # Arguments
/// * `input` - the array to be padded
/// * `padding` - a slice specifying for each dimension the amount of padding
///
/// # Examples
///
/// ```
/// use neuronika::nn;
///
/// let arr = ndarray::array![
///    [1., 2., 3.],
///    [4., 5., 6.],
///    [7., 8., 9.]
/// ];
/// let padded = nn::utils::reflection_pad(&arr, &[1, 1]);
/// let result = ndarray::array![
///    [5., 4., 5., 6., 5.],
///    [2., 1., 2., 3., 2.],
///    [5., 4., 5., 6., 5.],
///    [8., 7., 8., 9., 8.],
///    [5., 4., 5., 6., 5.]
/// ];
///
/// assert_eq!(padded, result);
/// ```
pub fn reflection_pad<D>(input: &Array<f32, D>, padding: &[usize]) -> Array<f32, D>
where
    D: ReflPad,
{
    D::reflection_pad(&input, padding)
}

/// Pads the input array using the **replication** of the input boundary.
///
/// Only **1**, **2** and **3** dimensional arrays support replicative padding.
///
/// # Arguments
/// * `input` - the array to be padded
/// * `padding` - a slice specifying for each dimension the amount of padding
///
/// # Examples
///
/// ```
/// use neuronika::nn;
///
/// let arr = ndarray::array![
///    [1., 2., 3.],
///    [4., 5., 6.],
///    [7., 8., 9.]
/// ];
/// let padded = nn::utils::replication_pad(&arr, &[1, 1]);
/// let result = ndarray::array![
///    [1., 1., 2., 3., 3.],
///    [1., 1., 2., 3., 3.],
///    [4., 4., 5., 6., 6.],
///    [7., 7., 8., 9., 9.],
///    [7., 7., 8., 9., 9.]
/// ];
///
/// assert_eq!(padded, result);
/// ```
pub fn replication_pad<D>(input: &Array<f32, D>, padding: &[usize]) -> Array<f32, D>
where
    D: ReplPad,
{
    D::replication_pad(&input, padding)
}

impl ReflPad for Ix1 {
    fn reflection_pad(input: &Array<f32, Ix1>, padding: &[usize]) -> Array<f32, Ix1> {
        let mut pos;
        let (in_len, out_len, pad) = {
            let len = input.len();
            let pad = padding[0];
            (len, len + pad * 2, pad)
        };
        let mut out = Array::<f32, _>::zeros(out_len);
        let (in_slice, out_slice) = (input.as_slice().unwrap(), out.as_slice_mut().unwrap());
        for (i, out_slice_el) in out_slice.iter_mut().enumerate().take(out_len) {
            if i < pad {
                pos = pad * 2 - i;
            } else if i >= pad && i < in_len + pad {
                pos = i;
            } else {
                pos = (in_len + pad - 1) * 2 - i;
            }
            pos -= pad;
            *out_slice_el = in_slice[pos];
        }
        out
    }
}

impl ReflPad for Ix2 {
    fn reflection_pad(input: &Array<f32, Ix2>, padding: &[usize]) -> Array<f32, Ix2> {
        let (mut pos_x, mut pos_y);
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let mut out = Array::<f32, _>::zeros((out_len_x, out_len_y));
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), out.as_slice_mut().unwrap()) };
        for i in 0..out_len_x {
            for j in 0..out_len_y {
                if j < pad_y {
                    pos_x = pad_y * 2 - j;
                } else if j >= pad_y && j < len_y + pad_y {
                    pos_x = j;
                } else {
                    pos_x = (len_y + pad_y - 1) * 2 - j;
                }
                pos_x -= pad_y;

                if i < pad_x {
                    pos_y = pad_x * 2 - i;
                } else if i >= pad_x && i < len_x + pad_x {
                    pos_y = i;
                } else {
                    pos_y = (len_x + pad_x - 1) * 2 - i;
                }
                pos_y -= pad_x;
                slice_out[i * out_len_y + j] = slice_in[pos_y * len_y + pos_x];
            }
        }
        out
    }
}

impl ReflPad for Ix3 {
    fn reflection_pad(input: &Array<f32, Ix3>, padding: &[usize]) -> Array<f32, Ix3> {
        let (mut pos_x, mut pos_y, mut pos_z);
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let mut out = Array::<f32, _>::zeros((out_len_z, out_len_x, out_len_y));
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), out.as_slice_mut().unwrap()) };

        for z in 0..out_len_z {
            for i in 0..out_len_x {
                for j in 0..out_len_y {
                    if j < pad_y {
                        pos_x = pad_y * 2 - j;
                    } else if j >= pad_y && j < len_y + pad_y {
                        pos_x = j;
                    } else {
                        pos_x = (len_y + pad_y - 1) * 2 - j;
                    }
                    pos_x -= pad_y;

                    if i < pad_x {
                        pos_y = pad_x * 2 - i;
                    } else if i >= pad_x && i < len_x + pad_x {
                        pos_y = i;
                    } else {
                        pos_y = (len_x + pad_x - 1) * 2 - i;
                    }
                    pos_y -= pad_x;

                    if z < pad_z {
                        pos_z = pad_z * 2 - z;
                    } else if z >= pad_z && z < len_z + pad_z {
                        pos_z = z;
                    } else {
                        pos_z = (len_z + pad_z - 1) * 2 - z;
                    }
                    pos_z -= pad_z;
                    slice_out[z * out_len_y * out_len_x + i * out_len_y + j] =
                        slice_in[pos_z * len_y * len_x + pos_y * len_y + pos_x];
                }
            }
        }
        out
    }
}

impl ReplPad for Ix1 {
    fn replication_pad(input: &Array<f32, Ix1>, padding: &[usize]) -> Array<f32, Ix1> {
        let mut pos;
        let (in_len, out_len, pad) = {
            let len = input.len();
            let pad = padding[0];
            (len, len + pad * 2, pad)
        };
        let mut out = Array::<f32, _>::zeros(out_len);
        let (in_slice, out_slice) = (input.as_slice().unwrap(), out.as_slice_mut().unwrap());
        for (j, out_slice_el) in out_slice.iter_mut().enumerate().take(out_len) {
            if j < pad {
                pos = pad;
            } else if j >= pad && j < in_len + pad {
                pos = j;
            } else {
                pos = in_len + pad - 1;
            }
            pos -= pad;
            *out_slice_el = in_slice[pos];
        }
        out
    }
}

impl ReplPad for Ix2 {
    fn replication_pad(input: &Array<f32, Ix2>, padding: &[usize]) -> Array<f32, Ix2> {
        let (mut pos_x, mut pos_y);
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let mut out = Array::<f32, _>::zeros((out_len_x, out_len_y));
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), out.as_slice_mut().unwrap()) };
        for i in 0..out_len_x {
            for j in 0..out_len_y {
                if j < pad_y {
                    pos_x = pad_y;
                } else if j >= pad_y && j < len_y + pad_y {
                    pos_x = j;
                } else {
                    pos_x = len_y + pad_y - 1;
                }
                pos_x -= pad_y;

                if i < pad_x {
                    pos_y = pad_x;
                } else if i >= pad_x && i < len_x + pad_x {
                    pos_y = i;
                } else {
                    pos_y = len_x + pad_x - 1;
                }
                pos_y -= pad_x;
                slice_out[i * out_len_y + j] = slice_in[pos_y * len_y + pos_x];
            }
        }
        out
    }
}

impl ReplPad for Ix3 {
    fn replication_pad(input: &Array<f32, Ix3>, padding: &[usize]) -> Array<f32, Ix3> {
        let (mut pos_x, mut pos_y, mut pos_z);
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let mut out = Array::<f32, _>::zeros((out_len_z, out_len_x, out_len_y));
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), out.as_slice_mut().unwrap()) };
        for z in 0..out_len_z {
            for i in 0..out_len_x {
                for j in 0..out_len_y {
                    if j < pad_y {
                        pos_x = pad_y;
                    } else if j >= pad_y && j < len_y + pad_y {
                        pos_x = j;
                    } else {
                        pos_x = len_y + pad_y - 1;
                    }
                    pos_x -= pad_y;

                    if i < pad_x {
                        pos_y = pad_x;
                    } else if i >= pad_x && i < len_x + pad_x {
                        pos_y = i;
                    } else {
                        pos_y = len_x + pad_x - 1;
                    }
                    pos_y -= pad_x;

                    if z < pad_z {
                        pos_z = pad_z;
                    } else if z >= pad_z && z < len_z + pad_z {
                        pos_z = z;
                    } else {
                        pos_z = len_z + pad_z - 1;
                    }
                    pos_z -= pad_z;

                    slice_out[z * out_len_y * out_len_x + i * out_len_y + j] =
                        slice_in[pos_z * len_y * len_x + pos_y * len_y + pos_x];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn im2col() {
        let input = ndarray::array![
            [
                [0., 1., 2., 3.],
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
                [12., 13., 14., 15.]
            ],
            [
                [0., 1., 2., 3.],
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
                [12., 13., 14., 15.]
            ],
            [
                [0., 1., 2., 3.],
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
                [12., 13., 14., 15.]
            ],
        ];
        // We must reshape the input, consider it as a bidimensional signal
        // with 3 channels each of 4 x 4.
        let d = input.clone().into_shape((1, 3, 4, 4)).unwrap();

        let im2col = ndarray::array![
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 2.0, 5.0, 6.0],
            [2.0, 3.0, 6.0, 7.0],
            [4.0, 5.0, 8.0, 9.0],
            [5.0, 6.0, 9.0, 10.0],
            [6.0, 7.0, 10.0, 11.0],
            [8.0, 9.0, 12.0, 13.0],
            [9.0, 10.0, 13.0, 14.0],
            [10.0, 11.0, 14.0, 15.0],
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 2.0, 5.0, 6.0],
            [2.0, 3.0, 6.0, 7.0],
            [4.0, 5.0, 8.0, 9.0],
            [5.0, 6.0, 9.0, 10.0],
            [6.0, 7.0, 10.0, 11.0],
            [8.0, 9.0, 12.0, 13.0],
            [9.0, 10.0, 13.0, 14.0],
            [10.0, 11.0, 14.0, 15.0],
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 2.0, 5.0, 6.0],
            [2.0, 3.0, 6.0, 7.0],
            [4.0, 5.0, 8.0, 9.0],
            [5.0, 6.0, 9.0, 10.0],
            [6.0, 7.0, 10.0, 11.0],
            [8.0, 9.0, 12.0, 13.0],
            [9.0, 10.0, 13.0, 14.0],
            [10.0, 11.0, 14.0, 15.0]
        ];
        assert_eq!(
            im2col,
            super::to_col(&d, &[1, 3, 3, 3], &[0, 0], &[1, 1], &[1, 1])
        );

        // Now let's increase the batch size by 1.
        let input_batch = ndarray::stack(ndarray::Axis(0), &[input.view(), input.view()]).unwrap();
        // We must reshape the input, consider it as 2 bidimensional signals
        // with 3 channels each of 4 x 4.
        let d = input_batch.into_shape((2, 3, 4, 4)).unwrap();

        // The im2col's result. Note that the im2col of signals
        // from the batch are concatenated along the columns.
        assert_eq!(
            ndarray::concatenate(ndarray::Axis(1), &[im2col.view(), im2col.view()]).unwrap(),
            super::to_col(&d, &[1, 3, 3, 3], &[0, 0], &[1, 1], &[1, 1])
        );
        // The nice thing about to_col is that it works for 1d, 2d, and 3d convolutions.
    }

    #[test]
    fn flatten() {
        use super::*;
        use ndarray::stack;
        // This is a kernel of 4 filters, reshaped output should be of shape (4,9).
        let kernel1 = (0..9)
            .map(|el| el as f32)
            .collect::<Array<f32, _>>()
            .into_shape((3, 3))
            .unwrap();
        let kernel2 = (9..18)
            .map(|el| el as f32)
            .collect::<Array<f32, _>>()
            .into_shape((3, 3))
            .unwrap();
        let kernel3 = (18..27)
            .map(|el| el as f32)
            .collect::<Array<f32, _>>()
            .into_shape((3, 3))
            .unwrap();
        let flattened = stack(Axis(0), &[kernel1.view(), kernel2.view(), kernel3.view()]).unwrap();
        assert_eq!(
            flatten(flattened),
            (0..27)
                .map(|el| el as f32)
                .collect::<Array<f32, _>>()
                .into_shape((3, 9))
                .unwrap()
        );
    }

    #[test]
    fn conv_args_ok() {
        // This is the input of a two dimensional convolution.
        // It is formed by 1 signal having 2 channels each of 4 x 4.
        let conv_input = ndarray::Array::<f32, _>::zeros((1, 2, 4, 4));
        super::check_conv_args(conv_input.shape(), &[1, 2, 2, 2], &[0, 0], &[1, 1], &[1, 1]);
    }

    #[test]
    #[should_panic(expected = "error: invalid kernel's shape [1, 2, 2] for 2d conv")]
    fn conv_args_invalid_kernel() {
        // This is the input of a two dimensional convolution.
        // It is formed by 1 signal having 2 channels each of 4 x 4.
        let conv_input = ndarray::Array::<f32, _>::zeros((1, 2, 4, 4));
        super::check_conv_args(conv_input.shape(), &[1, 2, 2], &[0, 0], &[1, 1], &[1, 1]);
    }

    #[test]
    fn constant_pad() {
        let arr = ndarray::Array::range(0., 25., 1.)
            .into_shape((5, 5))
            .unwrap();
        let padded = super::constant_pad(&arr, &[1, 2], 8.);
        assert_eq!(
            padded,
            ndarray::array![
                [8., 8., 8., 8., 8., 8., 8., 8., 8.],
                [8., 8., 0., 1., 2., 3., 4., 8., 8.],
                [8., 8., 5., 6., 7., 8., 9., 8., 8.],
                [8., 8., 10., 11., 12., 13., 14., 8., 8.],
                [8., 8., 15., 16., 17., 18., 19., 8., 8.],
                [8., 8., 20., 21., 22., 23., 24., 8., 8.],
                [8., 8., 8., 8., 8., 8., 8., 8., 8.],
            ]
        );
    }

    #[test]
    fn replication_pad_1d() {
        let arr = ndarray::Array::range(0., 5., 1.);
        let padded = super::replication_pad(&arr, &[2]);
        assert_eq!(padded, ndarray::array![0., 0., 0., 1., 2., 3., 4., 4., 4.],);
    }

    #[test]
    fn replication_pad_2d() {
        let arr = ndarray::Array::range(0., 25., 1.)
            .into_shape((5, 5))
            .unwrap();
        let padded = super::replication_pad(&arr, &[1, 2]);
        assert_eq!(
            padded,
            ndarray::array![
                [0., 0., 0., 1., 2., 3., 4., 4., 4.],
                [0., 0., 0., 1., 2., 3., 4., 4., 4.],
                [5., 5., 5., 6., 7., 8., 9., 9., 9.],
                [10., 10., 10., 11., 12., 13., 14., 14., 14.],
                [15., 15., 15., 16., 17., 18., 19., 19., 19.],
                [20., 20., 20., 21., 22., 23., 24., 24., 24.],
                [20., 20., 20., 21., 22., 23., 24., 24., 24.],
            ]
        );
    }

    #[test]
    fn replication_pad_3d() {
        let arr = ndarray::Array::range(0., 125., 1.)
            .into_shape((5, 5, 5))
            .unwrap();
        let padded = super::replication_pad(&arr, &[1, 2, 3]);
        assert_eq!(
            padded,
            ndarray::array![
                [
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [5., 5., 5., 5., 6., 7., 8., 9., 9., 9., 9.],
                    [10., 10., 10., 10., 11., 12., 13., 14., 14., 14., 14.],
                    [15., 15., 15., 15., 16., 17., 18., 19., 19., 19., 19.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.]
                ],
                [
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [5., 5., 5., 5., 6., 7., 8., 9., 9., 9., 9.],
                    [10., 10., 10., 10., 11., 12., 13., 14., 14., 14., 14.],
                    [15., 15., 15., 15., 16., 17., 18., 19., 19., 19., 19.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.]
                ],
                [
                    [25., 25., 25., 25., 26., 27., 28., 29., 29., 29., 29.],
                    [25., 25., 25., 25., 26., 27., 28., 29., 29., 29., 29.],
                    [25., 25., 25., 25., 26., 27., 28., 29., 29., 29., 29.],
                    [30., 30., 30., 30., 31., 32., 33., 34., 34., 34., 34.],
                    [35., 35., 35., 35., 36., 37., 38., 39., 39., 39., 39.],
                    [40., 40., 40., 40., 41., 42., 43., 44., 44., 44., 44.],
                    [45., 45., 45., 45., 46., 47., 48., 49., 49., 49., 49.],
                    [45., 45., 45., 45., 46., 47., 48., 49., 49., 49., 49.],
                    [45., 45., 45., 45., 46., 47., 48., 49., 49., 49., 49.]
                ],
                [
                    [50., 50., 50., 50., 51., 52., 53., 54., 54., 54., 54.],
                    [50., 50., 50., 50., 51., 52., 53., 54., 54., 54., 54.],
                    [50., 50., 50., 50., 51., 52., 53., 54., 54., 54., 54.],
                    [55., 55., 55., 55., 56., 57., 58., 59., 59., 59., 59.],
                    [60., 60., 60., 60., 61., 62., 63., 64., 64., 64., 64.],
                    [65., 65., 65., 65., 66., 67., 68., 69., 69., 69., 69.],
                    [70., 70., 70., 70., 71., 72., 73., 74., 74., 74., 74.],
                    [70., 70., 70., 70., 71., 72., 73., 74., 74., 74., 74.],
                    [70., 70., 70., 70., 71., 72., 73., 74., 74., 74., 74.]
                ],
                [
                    [75., 75., 75., 75., 76., 77., 78., 79., 79., 79., 79.],
                    [75., 75., 75., 75., 76., 77., 78., 79., 79., 79., 79.],
                    [75., 75., 75., 75., 76., 77., 78., 79., 79., 79., 79.],
                    [80., 80., 80., 80., 81., 82., 83., 84., 84., 84., 84.],
                    [85., 85., 85., 85., 86., 87., 88., 89., 89., 89., 89.],
                    [90., 90., 90., 90., 91., 92., 93., 94., 94., 94., 94.],
                    [95., 95., 95., 95., 96., 97., 98., 99., 99., 99., 99.],
                    [95., 95., 95., 95., 96., 97., 98., 99., 99., 99., 99.],
                    [95., 95., 95., 95., 96., 97., 98., 99., 99., 99., 99.]
                ],
                [
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [105., 105., 105., 105., 106., 107., 108., 109., 109., 109., 109.],
                    [110., 110., 110., 110., 111., 112., 113., 114., 114., 114., 114.],
                    [115., 115., 115., 115., 116., 117., 118., 119., 119., 119., 119.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.]
                ],
                [
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [105., 105., 105., 105., 106., 107., 108., 109., 109., 109., 109.],
                    [110., 110., 110., 110., 111., 112., 113., 114., 114., 114., 114.],
                    [115., 115., 115., 115., 116., 117., 118., 119., 119., 119., 119.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.]
                ]
            ]
        );
    }

    #[test]
    fn reflection_pad_1d() {
        let arr = ndarray::Array::range(0., 5., 1.);
        let padded = super::reflection_pad(&arr, &[2]);
        assert_eq!(padded, ndarray::array![2., 1., 0., 1., 2., 3., 4., 3., 2.],);
    }

    #[test]
    fn reflection_pad_2d() {
        let arr = ndarray::Array::range(0., 25., 1.)
            .into_shape((5, 5))
            .unwrap();
        let padded = super::reflection_pad(&arr, &[1, 2]);
        assert_eq!(
            padded,
            ndarray::array![
                [7., 6., 5., 6., 7., 8., 9., 8., 7.],
                [2., 1., 0., 1., 2., 3., 4., 3., 2.],
                [7., 6., 5., 6., 7., 8., 9., 8., 7.],
                [12., 11., 10., 11., 12., 13., 14., 13., 12.],
                [17., 16., 15., 16., 17., 18., 19., 18., 17.],
                [22., 21., 20., 21., 22., 23., 24., 23., 22.],
                [17., 16., 15., 16., 17., 18., 19., 18., 17.]
            ]
        );
    }

    #[test]
    fn reflection_pad_3d() {
        let arr = ndarray::Array::range(0., 125., 1.)
            .into_shape((5, 5, 5))
            .unwrap();
        let padded = super::reflection_pad(&arr, &[1, 2, 3]);
        assert_eq!(
            padded,
            ndarray::array![
                [
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                    [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                    [28., 27., 26., 25., 26., 27., 28., 29., 28., 27., 26.],
                    [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                    [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                    [48., 47., 46., 45., 46., 47., 48., 49., 48., 47., 46.],
                    [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.]
                ],
                [
                    [13., 12., 11., 10., 11., 12., 13., 14., 13., 12., 11.],
                    [8., 7., 6., 5., 6., 7., 8., 9., 8., 7., 6.],
                    [3., 2., 1., 0., 1., 2., 3., 4., 3., 2., 1.],
                    [8., 7., 6., 5., 6., 7., 8., 9., 8., 7., 6.],
                    [13., 12., 11., 10., 11., 12., 13., 14., 13., 12., 11.],
                    [18., 17., 16., 15., 16., 17., 18., 19., 18., 17., 16.],
                    [23., 22., 21., 20., 21., 22., 23., 24., 23., 22., 21.],
                    [18., 17., 16., 15., 16., 17., 18., 19., 18., 17., 16.],
                    [13., 12., 11., 10., 11., 12., 13., 14., 13., 12., 11.]
                ],
                [
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                    [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                    [28., 27., 26., 25., 26., 27., 28., 29., 28., 27., 26.],
                    [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                    [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                    [48., 47., 46., 45., 46., 47., 48., 49., 48., 47., 46.],
                    [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.]
                ],
                [
                    [63., 62., 61., 60., 61., 62., 63., 64., 63., 62., 61.],
                    [58., 57., 56., 55., 56., 57., 58., 59., 58., 57., 56.],
                    [53., 52., 51., 50., 51., 52., 53., 54., 53., 52., 51.],
                    [58., 57., 56., 55., 56., 57., 58., 59., 58., 57., 56.],
                    [63., 62., 61., 60., 61., 62., 63., 64., 63., 62., 61.],
                    [68., 67., 66., 65., 66., 67., 68., 69., 68., 67., 66.],
                    [73., 72., 71., 70., 71., 72., 73., 74., 73., 72., 71.],
                    [68., 67., 66., 65., 66., 67., 68., 69., 68., 67., 66.],
                    [63., 62., 61., 60., 61., 62., 63., 64., 63., 62., 61.]
                ],
                [
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                    [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                    [78., 77., 76., 75., 76., 77., 78., 79., 78., 77., 76.],
                    [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                    [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                    [98., 97., 96., 95., 96., 97., 98., 99., 98., 97., 96.],
                    [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.]
                ],
                [
                    [113., 112., 111., 110., 111., 112., 113., 114., 113., 112., 111.],
                    [108., 107., 106., 105., 106., 107., 108., 109., 108., 107., 106.],
                    [103., 102., 101., 100., 101., 102., 103., 104., 103., 102., 101.],
                    [108., 107., 106., 105., 106., 107., 108., 109., 108., 107., 106.],
                    [113., 112., 111., 110., 111., 112., 113., 114., 113., 112., 111.],
                    [118., 117., 116., 115., 116., 117., 118., 119., 118., 117., 116.],
                    [123., 122., 121., 120., 121., 122., 123., 124., 123., 122., 121.],
                    [118., 117., 116., 115., 116., 117., 118., 119., 118., 117., 116.],
                    [113., 112., 111., 110., 111., 112., 113., 114., 113., 112., 111.]
                ],
                [
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                    [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                    [78., 77., 76., 75., 76., 77., 78., 79., 78., 77., 76.],
                    [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                    [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                    [98., 97., 96., 95., 96., 97., 98., 99., 98., 97., 96.],
                    [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.]
                ]
            ]
        )
    }

    #[test]
    fn conv1d() {
        use ndarray::prelude::*;

        use super::*;
        use ndarray::Ix3;

        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((5, 3, 10)).unwrap();
        let kernel = Array::<f32, _>::ones((6, 3, 5));
        let stride = &[1];
        let padding = &[0];
        let dilation = &[1];

        let conv_out_shape =
            conv_out_shape::<Ix3>(input.shape(), kernel.shape(), padding, stride, dilation);

        let true_output_elems = vec![
            180., 195., 210., 225., 240., 255., 180., 195., 210., 225., 240., 255., 180., 195.,
            210., 225., 240., 255., 180., 195., 210., 225., 240., 255., 180., 195., 210., 225.,
            240., 255., 180., 195., 210., 225., 240., 255., 630., 645., 660., 675., 690., 705.,
            630., 645., 660., 675., 690., 705., 630., 645., 660., 675., 690., 705., 630., 645.,
            660., 675., 690., 705., 630., 645., 660., 675., 690., 705., 630., 645., 660., 675.,
            690., 705., 1080., 1095., 1110., 1125., 1140., 1155., 1080., 1095., 1110., 1125.,
            1140., 1155., 1080., 1095., 1110., 1125., 1140., 1155., 1080., 1095., 1110., 1125.,
            1140., 1155., 1080., 1095., 1110., 1125., 1140., 1155., 1080., 1095., 1110., 1125.,
            1140., 1155., 1530., 1545., 1560., 1575., 1590., 1605., 1530., 1545., 1560., 1575.,
            1590., 1605., 1530., 1545., 1560., 1575., 1590., 1605., 1530., 1545., 1560., 1575.,
            1590., 1605., 1530., 1545., 1560., 1575., 1590., 1605., 1530., 1545., 1560., 1575.,
            1590., 1605., 1980., 1995., 2010., 2025., 2040., 2055., 1980., 1995., 2010., 2025.,
            2040., 2055., 1980., 1995., 2010., 2025., 2040., 2055., 1980., 1995., 2010., 2025.,
            2040., 2055., 1980., 1995., 2010., 2025., 2040., 2055., 1980., 1995., 2010., 2025.,
            2040., 2055.,
        ];

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            stride,
            dilation,
        );

        let true_input_grad_elems = vec![
            6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18.,
            12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30.,
            24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24.,
            30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12.,
            18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6.,
            6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18.,
            12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30.,
            24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24.,
            30., 30., 24., 18., 12., 6.,
        ];

        let true_kernel_grad_elems = array![
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(kernel_grad, true_kernel_grad_elems);
    }

    #[test]
    fn conv2d() {
        use super::*;
        use ndarray::Ix4;

        // This is an input with a batch size of 3, 2 input channels each of 5 by 5.
        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((3, 2, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((3, 2, 2, 2));

        let stride = &[1, 1];
        let padding = &[0, 0];
        let dilation = &[1, 1];

        let conv_out_shape =
            conv_out_shape::<Ix4>(input.shape(), kernel.shape(), padding, stride, dilation);

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);
        let true_output_elems: Vec<f32> = vec![
            124., 132., 140., 148., 164., 172., 180., 188., 204., 212., 220., 228., 244., 252.,
            260., 268., 124., 132., 140., 148., 164., 172., 180., 188., 204., 212., 220., 228.,
            244., 252., 260., 268., 124., 132., 140., 148., 164., 172., 180., 188., 204., 212.,
            220., 228., 244., 252., 260., 268., 524., 532., 540., 548., 564., 572., 580., 588.,
            604., 612., 620., 628., 644., 652., 660., 668., 524., 532., 540., 548., 564., 572.,
            580., 588., 604., 612., 620., 628., 644., 652., 660., 668., 524., 532., 540., 548.,
            564., 572., 580., 588., 604., 612., 620., 628., 644., 652., 660., 668., 924., 932.,
            940., 948., 964., 972., 980., 988., 1004., 1012., 1020., 1028., 1044., 1052., 1060.,
            1068., 924., 932., 940., 948., 964., 972., 980., 988., 1004., 1012., 1020., 1028.,
            1044., 1052., 1060., 1068., 924., 932., 940., 948., 964., 972., 980., 988., 1004.,
            1012., 1020., 1028., 1044., 1052., 1060., 1068.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros((3, 2, 5, 5));
        let mut kernel_grad = Array::<f32, _>::zeros((3, 2, 2, 2));
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            stride,
            dilation,
        );

        let true_input_grad_elems: Vec<f32> = vec![
            3., 6., 6., 6., 3., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 6., 12., 12., 12.,
            6., 3., 6., 6., 6., 3., 3., 6., 6., 6., 3., 6., 12., 12., 12., 6., 6., 12., 12., 12.,
            6., 6., 12., 12., 12., 6., 3., 6., 6., 6., 3., 3., 6., 6., 6., 3., 6., 12., 12., 12.,
            6., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 3., 6., 6., 6., 3., 3., 6., 6., 6.,
            3., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 3., 6., 6.,
            6., 3., 3., 6., 6., 6., 3., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 6., 12., 12.,
            12., 6., 3., 6., 6., 6., 3., 3., 6., 6., 6., 3., 6., 12., 12., 12., 6., 6., 12., 12.,
            12., 6., 6., 12., 12., 12., 6., 3., 6., 6., 6., 3.,
        ];
        let true_kernel_grad_elems: Vec<f32> = vec![
            2832., 2880., 3072., 3120., 4032., 4080., 4272., 4320., 2832., 2880., 3072., 3120.,
            4032., 4080., 4272., 4320., 2832., 2880., 3072., 3120., 4032., 4080., 4272., 4320.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }

    #[test]
    fn conv2d_strided() {
        use super::*;
        use ndarray::Ix4;

        // This is an input with a batch size of 3, 2 input channels each of 5 by 5.
        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((3, 2, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((3, 2, 2, 2));

        let stride = &[2, 2];
        let padding = &[0, 0];
        let dilation = &[1, 1];

        let conv_out_shape =
            conv_out_shape::<Ix4>(input.shape(), kernel.shape(), padding, stride, dilation);

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        let true_output_elems: Vec<f32> = vec![
            124., 140., 204., 220., 124., 140., 204., 220., 124., 140., 204., 220., 524., 540.,
            604., 620., 524., 540., 604., 620., 524., 540., 604., 620., 924., 940., 1004., 1020.,
            924., 940., 1004., 1020., 924., 940., 1004., 1020.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros((3, 2, 5, 5));
        let mut kernel_grad = Array::<f32, _>::zeros((3, 2, 2, 2));
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            stride,
            dilation,
        );

        let true_input_grad_elems: Vec<f32> = vec![
            3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 0., 0.,
            0., 0., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3.,
            0., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3.,
            3., 3., 3., 0., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3.,
            3., 0., 3., 3., 3., 3., 0., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0.,
            3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., 3., 3.,
            3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 0., 0., 0., 0., 0.,
        ];

        let true_kernel_grad_elems: Vec<f32> = vec![
            672., 684., 732., 744., 972., 984., 1032., 1044., 672., 684., 732., 744., 972., 984.,
            1032., 1044., 672., 684., 732., 744., 972., 984., 1032., 1044.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }
    #[test]
    fn conv2d_dilated() {
        use super::*;
        use ndarray::Ix4;

        // This is an input with a batch size of 3, 2 input channels each of 5 by 5.
        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((3, 2, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((3, 2, 2, 2));

        let stride = &[2, 2];
        let padding = &[0, 0];
        let dilation = &[2, 2];

        let conv_out_shape =
            conv_out_shape::<Ix4>(input.shape(), kernel.shape(), padding, stride, dilation);

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        let true_output_elems: Vec<f32> = vec![
            148., 164., 228., 244., 148., 164., 228., 244., 148., 164., 228., 244., 548., 564.,
            628., 644., 548., 564., 628., 644., 548., 564., 628., 644., 948., 964., 1028., 1044.,
            948., 964., 1028., 1044., 948., 964., 1028., 1044.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros((3, 2, 5, 5));
        let mut kernel_grad = Array::<f32, _>::zeros((3, 2, 2, 2));
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            stride,
            dilation,
        );

        let true_input_grad_elems: Vec<f32> = vec![
            3., 0., 6., 0., 3., 0., 0., 0., 0., 0., 6., 0., 12., 0., 6., 0., 0., 0., 0., 0., 3.,
            0., 6., 0., 3., 3., 0., 6., 0., 3., 0., 0., 0., 0., 0., 6., 0., 12., 0., 6., 0., 0.,
            0., 0., 0., 3., 0., 6., 0., 3., 3., 0., 6., 0., 3., 0., 0., 0., 0., 0., 6., 0., 12.,
            0., 6., 0., 0., 0., 0., 0., 3., 0., 6., 0., 3., 3., 0., 6., 0., 3., 0., 0., 0., 0., 0.,
            6., 0., 12., 0., 6., 0., 0., 0., 0., 0., 3., 0., 6., 0., 3., 3., 0., 6., 0., 3., 0.,
            0., 0., 0., 0., 6., 0., 12., 0., 6., 0., 0., 0., 0., 0., 3., 0., 6., 0., 3., 3., 0.,
            6., 0., 3., 0., 0., 0., 0., 0., 6., 0., 12., 0., 6., 0., 0., 0., 0., 0., 3., 0., 6.,
            0., 3.,
        ];

        let true_kernel_grad_elems: Vec<f32> = vec![
            672., 696., 792., 816., 972., 996., 1092., 1116., 672., 696., 792., 816., 972., 996.,
            1092., 1116., 672., 696., 792., 816., 972., 996., 1092., 1116.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }

    #[test]
    fn grouped_conv2d() {
        use super::*;
        use ndarray::Ix4;
        // This is an input with a batch size of 4, 8 input channels each of 5 by 5.
        // Constructing an input.
        let input: Array<f32, Ix4> = (0..800)
            .map(|el| el as f32)
            .collect::<Array<f32, _>>()
            .into_shape((4, 8, 5, 5))
            .unwrap();

        // Both output and input channels need to be divisible by group.
        // Group is 2 so we must divide the input channels by 2.
        let kernel = Array::<f32, _>::ones((8, 4, 2, 2));
        let stride = &[1, 1];
        let padding = &[0, 0];
        let dilation = &[1, 1];
        let groups = 2;

        let conv_out_shape =
            conv_out_shape::<Ix4>(input.shape(), kernel.shape(), padding, stride, dilation);
        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution_with_groups(&input, &kernel, &mut conv_out, stride, dilation, groups);

        let true_output_elems = vec![
            648., 664., 680., 696., 728., 744., 760., 776., 808., 824., 840., 856., 888., 904.,
            920., 936., 648., 664., 680., 696., 728., 744., 760., 776., 808., 824., 840., 856.,
            888., 904., 920., 936., 648., 664., 680., 696., 728., 744., 760., 776., 808., 824.,
            840., 856., 888., 904., 920., 936., 648., 664., 680., 696., 728., 744., 760., 776.,
            808., 824., 840., 856., 888., 904., 920., 936., 2248., 2264., 2280., 2296., 2328.,
            2344., 2360., 2376., 2408., 2424., 2440., 2456., 2488., 2504., 2520., 2536., 2248.,
            2264., 2280., 2296., 2328., 2344., 2360., 2376., 2408., 2424., 2440., 2456., 2488.,
            2504., 2520., 2536., 2248., 2264., 2280., 2296., 2328., 2344., 2360., 2376., 2408.,
            2424., 2440., 2456., 2488., 2504., 2520., 2536., 2248., 2264., 2280., 2296., 2328.,
            2344., 2360., 2376., 2408., 2424., 2440., 2456., 2488., 2504., 2520., 2536., 3848.,
            3864., 3880., 3896., 3928., 3944., 3960., 3976., 4008., 4024., 4040., 4056., 4088.,
            4104., 4120., 4136., 3848., 3864., 3880., 3896., 3928., 3944., 3960., 3976., 4008.,
            4024., 4040., 4056., 4088., 4104., 4120., 4136., 3848., 3864., 3880., 3896., 3928.,
            3944., 3960., 3976., 4008., 4024., 4040., 4056., 4088., 4104., 4120., 4136., 3848.,
            3864., 3880., 3896., 3928., 3944., 3960., 3976., 4008., 4024., 4040., 4056., 4088.,
            4104., 4120., 4136., 5448., 5464., 5480., 5496., 5528., 5544., 5560., 5576., 5608.,
            5624., 5640., 5656., 5688., 5704., 5720., 5736., 5448., 5464., 5480., 5496., 5528.,
            5544., 5560., 5576., 5608., 5624., 5640., 5656., 5688., 5704., 5720., 5736., 5448.,
            5464., 5480., 5496., 5528., 5544., 5560., 5576., 5608., 5624., 5640., 5656., 5688.,
            5704., 5720., 5736., 5448., 5464., 5480., 5496., 5528., 5544., 5560., 5576., 5608.,
            5624., 5640., 5656., 5688., 5704., 5720., 5736., 7048., 7064., 7080., 7096., 7128.,
            7144., 7160., 7176., 7208., 7224., 7240., 7256., 7288., 7304., 7320., 7336., 7048.,
            7064., 7080., 7096., 7128., 7144., 7160., 7176., 7208., 7224., 7240., 7256., 7288.,
            7304., 7320., 7336., 7048., 7064., 7080., 7096., 7128., 7144., 7160., 7176., 7208.,
            7224., 7240., 7256., 7288., 7304., 7320., 7336., 7048., 7064., 7080., 7096., 7128.,
            7144., 7160., 7176., 7208., 7224., 7240., 7256., 7288., 7304., 7320., 7336., 8648.,
            8664., 8680., 8696., 8728., 8744., 8760., 8776., 8808., 8824., 8840., 8856., 8888.,
            8904., 8920., 8936., 8648., 8664., 8680., 8696., 8728., 8744., 8760., 8776., 8808.,
            8824., 8840., 8856., 8888., 8904., 8920., 8936., 8648., 8664., 8680., 8696., 8728.,
            8744., 8760., 8776., 8808., 8824., 8840., 8856., 8888., 8904., 8920., 8936., 8648.,
            8664., 8680., 8696., 8728., 8744., 8760., 8776., 8808., 8824., 8840., 8856., 8888.,
            8904., 8920., 8936., 10248., 10264., 10280., 10296., 10328., 10344., 10360., 10376.,
            10408., 10424., 10440., 10456., 10488., 10504., 10520., 10536., 10248., 10264., 10280.,
            10296., 10328., 10344., 10360., 10376., 10408., 10424., 10440., 10456., 10488., 10504.,
            10520., 10536., 10248., 10264., 10280., 10296., 10328., 10344., 10360., 10376., 10408.,
            10424., 10440., 10456., 10488., 10504., 10520., 10536., 10248., 10264., 10280., 10296.,
            10328., 10344., 10360., 10376., 10408., 10424., 10440., 10456., 10488., 10504., 10520.,
            10536., 11848., 11864., 11880., 11896., 11928., 11944., 11960., 11976., 12008., 12024.,
            12040., 12056., 12088., 12104., 12120., 12136., 11848., 11864., 11880., 11896., 11928.,
            11944., 11960., 11976., 12008., 12024., 12040., 12056., 12088., 12104., 12120., 12136.,
            11848., 11864., 11880., 11896., 11928., 11944., 11960., 11976., 12008., 12024., 12040.,
            12056., 12088., 12104., 12120., 12136., 11848., 11864., 11880., 11896., 11928., 11944.,
            11960., 11976., 12008., 12024., 12040., 12056., 12088., 12104., 12120., 12136.,
        ];

        assert_eq!(
            conv_out,
            Array::from_shape_vec(conv_out.raw_dim(), true_output_elems).unwrap()
        );

        // // Backward pass
        let mut input_grad = Array::<f32, _>::zeros((4, 8, 5, 5));
        let mut kernel_grad = Array::<f32, _>::zeros((8, 4, 2, 2));
        let d_out = Array::<f32, _>::ones(conv_out.raw_dim());

        convolution_with_groups_backward(
            &mut input_grad,
            &mut kernel_grad,
            &d_out,
            &input,
            &kernel,
            stride,
            dilation,
            groups,
        );

        let true_kernel_grad_elems: Vec<f32> = vec![
            19776., 19840., 20096., 20160., 21376., 21440., 21696., 21760., 22976., 23040., 23296.,
            23360., 24576., 24640., 24896., 24960., 19776., 19840., 20096., 20160., 21376., 21440.,
            21696., 21760., 22976., 23040., 23296., 23360., 24576., 24640., 24896., 24960., 19776.,
            19840., 20096., 20160., 21376., 21440., 21696., 21760., 22976., 23040., 23296., 23360.,
            24576., 24640., 24896., 24960., 19776., 19840., 20096., 20160., 21376., 21440., 21696.,
            21760., 22976., 23040., 23296., 23360., 24576., 24640., 24896., 24960., 26176., 26240.,
            26496., 26560., 27776., 27840., 28096., 28160., 29376., 29440., 29696., 29760., 30976.,
            31040., 31296., 31360., 26176., 26240., 26496., 26560., 27776., 27840., 28096., 28160.,
            29376., 29440., 29696., 29760., 30976., 31040., 31296., 31360., 26176., 26240., 26496.,
            26560., 27776., 27840., 28096., 28160., 29376., 29440., 29696., 29760., 30976., 31040.,
            31296., 31360., 26176., 26240., 26496., 26560., 27776., 27840., 28096., 28160., 29376.,
            29440., 29696., 29760., 30976., 31040., 31296., 31360.,
        ];
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel_grad.raw_dim(), true_kernel_grad_elems).unwrap()
        );

        let true_input_grad_elems: Vec<f32> = vec![
            4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8.,
            4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8.,
            8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8.,
            8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8.,
            8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16.,
            16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16.,
            16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16.,
            16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8.,
            8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4.,
            8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8.,
            16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8.,
            16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8.,
            16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4.,
            4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8.,
            4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8.,
            8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8.,
            8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8.,
            8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16.,
            16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16.,
            16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16.,
            16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8.,
            8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4.,
            8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8.,
            16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8.,
            16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8.,
            16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4.,
            4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input_grad.raw_dim(), true_input_grad_elems).unwrap()
        );
    }
}
