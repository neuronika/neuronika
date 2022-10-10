use std::{cell::RefCell, rc::Rc};

use ndarray::{
    Array, ArrayBase, ArrayViewD, ArrayViewMutD, Axis, Data, DataMut, DimMax, Dimension, Ix1, Ix2,
    Ix3, ShapeBuilder, Slice,
};

/// Shorthand for `Rc<RefCell<T>>`.
pub(crate) type Shared<T> = Rc<RefCell<T>>;
/// A broadcasted ndarray's dimension.
pub(crate) type Broadcast<D, E> = <D as DimMax<E>>::Output;

/// Utility trait to compute the dimensionality of algebraic operations' results.
pub(crate) trait DotDim<Rhs>
where
    Self: Dimension,
    Rhs: Dimension,
{
    /// Dimension of the resulting variable.
    type Output: Dimension;

    /// Does the actual computation of the shape.
    fn shape(lhs: Self, rhs: Rhs) -> <Self as DotDim<Rhs>>::Output;
}

impl DotDim<Ix2> for Ix1 {
    type Output = Ix1;

    fn shape(_: Self, rhs: Ix2) -> <Self as DotDim<Ix2>>::Output {
        let mut result = Ix1::zeros(1);
        result[0] = rhs.last_elem();
        result
    }
}

impl DotDim<Ix1> for Ix2 {
    type Output = Ix1;

    fn shape(lhs: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        let mut result = Ix1::zeros(1);
        result[0] = lhs[0];
        result
    }
}

impl DotDim<Ix2> for Ix2 {
    type Output = Ix2;

    fn shape(lhs: Self, rhs: Ix2) -> <Self as DotDim<Ix2>>::Output {
        let mut result = Ix2::zeros(2);
        result[0] = lhs[0];
        result[1] = rhs[1];
        result
    }
}

/// Computes the shape of the **input** after the padding is applied.
///
/// This function expects arrays having shape (batch size, channels, ...).
///
/// # Arguments
///
/// * `shape` - shape of the input.
///
/// * `padding` - padding around the input.
pub(crate) fn padded_shape<D>(shape: D, padding: <D::Smaller as Dimension>::Smaller) -> D
where
    D: Dimension,
{
    let shape = shape.slice();
    let padding = padding.slice();

    // Checks that the number of spatial dimension and input dimensions is the same.
    assert!(shape.len() - 2 == padding.len());

    let mut padded_input_shape = D::zeros(shape.len());
    padded_input_shape[0] = shape[0]; // Copy batch size.
    padded_input_shape[1] = shape[1]; // Copy input channels.
    padded_input_shape
        .slice_mut()
        .iter_mut()
        .skip(2)
        .zip(shape.iter().skip(2))
        .zip(padding.iter())
        .for_each(|((padded_dim, original_dim), padding)| *padded_dim = original_dim + 2 * padding);

    padded_input_shape
}

/// Computes the result of broadcasting between `left` and `right`.
///
/// # Arguments
///
/// * `left` - left dimensions.
///
/// * `right` - right dimensions.
pub(crate) fn cobroadcast<D, E>(left: D, right: E) -> Broadcast<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    let (bigger, smaller) = if left.ndim() >= right.ndim() {
        (left.slice(), right.slice())
    } else {
        (right.slice(), left.slice())
    };

    let mut out = <D as DimMax<E>>::Output::zeros(bigger.len());
    out.slice_mut()
        .iter_mut()
        .zip(bigger)
        .for_each(|(l, &r)| *l = r);

    out.slice_mut()
        .iter_mut()
        .skip(bigger.len() - smaller.len())
        .zip(smaller)
        .filter(|(l, r)| l != r)
        .for_each(|(l, &r)| match l {
            1 => *l = r,
            _ => assert_eq!(r, 1, "The two tensors have incompatible shape."),
        });

    out
}

/// Creates an empty tensor whose shape is the result of broadcasting between those of `left` and
/// `right`.
///
/// # Arguments
///
/// * `left` - left operand in the binary operations that admits broadcasting.
///
/// * `right` - right operand in the binary operations that admits broadcasting.
pub(crate) fn cobroadcasted_zeros<D, E>(
    left: &Array<f32, D>,
    right: &Array<f32, E>,
) -> Array<f32, Broadcast<D, E>>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    Array::zeros(cobroadcast(left.raw_dim(), right.raw_dim()))
}

/// Accumulates `source` into `target`, reverting the broadcasting.
///
/// ## Arguments
///
/// * `source` - Tensor to reduce.
/// * `target` - Tensor in which the accumulation must be pushed.
pub(crate) fn accumulate<D, E>(target: &mut Array<f32, D>, source: &Array<f32, E>)
where
    D: Dimension,
    E: Dimension,
{
    debug_assert!(target.ndim() <= source.ndim());

    if source.shape() == target.shape() {
        *target += source;
        return;
    }

    // Computes the difference between the number of dimensions
    let target_dims = target.ndim();
    let source_shape = source.shape();
    let k = source_shape.len() - target_dims;
    let mut reshape = D::Larger::zeros(target_dims + 1);
    reshape[0] = source_shape[..k].iter().product();
    reshape
        .slice_mut()
        .iter_mut()
        .skip(1)
        .zip(source_shape.iter().skip(k))
        .for_each(|(r, &s)| *r = s);

    // Reshapes such that the sub-views match the dimensionality of target
    let view = source.view().into_shape(reshape).unwrap();
    let axis = Axis(target_dims - 1);
    let mut lanes_v = view.lanes(axis).into_iter();
    while lanes_v.len() > 0 {
        for mut lane_t in target.lanes_mut(axis) {
            if let Some(lane_v) = lanes_v.next() {
                if lane_t.len() == 1 {
                    lane_t[0] += lane_v.sum();
                } else {
                    lane_t += &lane_v;
                }
            }
        }
    }
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
pub(crate) fn conv_out_shape<D>(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> D
where
    D: Dimension,
{
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
fn compute_rolling_window_shape<D, S>(
    input: &ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Vec<usize>
where
    D: Dimension,
    S: Data<Elem = f32>,
{
    let mut indices: D = conv_out_shape(input.shape(), window_shape, stride, dilation);
    indices[1] = 1;
    indices
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
fn compute_rolling_window_strides<D, S>(
    input: &ArrayBase<S, D>,
    stride: &[usize],
    dilation: &[usize],
) -> Vec<usize>
where
    D: Dimension,
    S: Data<Elem = f32>,
{
    let indexing_strides: Vec<isize> = {
        let view = input.slice_each_axis(|ax| {
            let axis_index = ax.axis.index();
            if axis_index == 0 || axis_index == 1 {
                Slice::new(0, None, 1) // Batch stride and channel stride
            } else {
                Slice::new(0, None, stride[ax.axis.index() - 2] as isize)
            }
        });
        let view_strides: &[isize] = ArrayBase::strides(&view);
        view_strides.to_vec()
    };
    // Number of in channels doesn't count for the window's strides,
    // it must be left unchanged.
    let window_strides: Vec<isize> = input
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
        .chain(window_strides.iter())
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
pub(crate) fn as_windows<'a, D, S>(
    input: &ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayViewD<'a, f32>
where
    D: Dimension,
    S: Data<Elem = f32>,
{
    let rolling_window_shape: Vec<usize> =
        compute_rolling_window_shape(input, window_shape, stride, dilation);
    let rolling_window_strides: Vec<usize> =
        compute_rolling_window_strides(input, stride, dilation);

    unsafe {
        ArrayViewD::from_shape_ptr(
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
pub(crate) fn as_windows_mut<'a, D, S>(
    input: &mut ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayViewMutD<'a, f32>
where
    D: Dimension,
    S: DataMut<Elem = f32>,
{
    let rolling_window_shape: Vec<usize> =
        compute_rolling_window_shape(input, window_shape, stride, dilation);
    let rolling_window_strides: Vec<usize> =
        compute_rolling_window_strides(input, stride, dilation);

    // Must ensure that the array is contiguous and that parallel code access the data per-batch.
    unsafe {
        ArrayViewMutD::from_shape_ptr(
            rolling_window_shape.strides(rolling_window_strides),
            input.as_mut_ptr(),
        )
    }
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
pub(crate) fn columns_shape<D, S>(
    input: &ArrayBase<S, D>,
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Ix3
where
    D: Dimension,
    S: Data<Elem = f32>,
{
    let output_map_shape = conv_out_shape::<D>(input.shape(), kernel_shape, stride, dilation);
    let mut columns_shape = Ix3::zeros(3);
    columns_shape[0] = output_map_shape[0];
    columns_shape[1] = output_map_shape.slice().iter().skip(2).product();
    columns_shape[2] = kernel_shape.iter().skip(1).product();

    columns_shape
}

/// Checks that the arguments are correct for the given **convolution**. It verifies that the
/// `stride` and `dilation` slices are of the right length; their length must match the
/// dimensionality of the convolution. It also check that `kernel` and `input` are of the same
/// dimension and that the kernel size, after dilation is applied, **is not bigger** that the actual
/// input size.
pub(crate) fn check_conv_args(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) {
    // The type of convolution can be derived by considering the number of input's dimension
    // skipping the first two, that are the batch size and input channels. The first two axes of
    // the input are always for the batch size and the number of input channels.
    let convolution_dimension = input_shape.len() - 2;

    assert_eq!(
        convolution_dimension,
        stride.len(),
        "Invalid stride {:?} for {}d conv.",
        stride,
        convolution_dimension
    );

    assert_eq!(
        convolution_dimension,
        dilation.len(),
        "Invalid dilation {:?} for {}d conv.",
        dilation,
        convolution_dimension
    );

    assert_eq!(
        kernel_shape.len(),
        input_shape.len(),
        "Invalid kernel shape {:?} for {}d conv",
        &kernel_shape,
        convolution_dimension
    );

    // Checks that the kernel size, taking into account dilation, is suitable for the padded input.
    input_shape
        .iter()
        .skip(2)
        .zip(kernel_shape.iter().skip(2))
        .zip(dilation.iter())
        .for_each(|((input_dim, kernel_dim), dilation_dim)| {
            let dilated_kernel_dim = (*kernel_dim - 1) * *dilation_dim + 1;
            assert!(
                *input_dim >= dilated_kernel_dim,
                "The kernel size can't be greater than actual input size.",
            )
        });
}

/// Checks that the arguments are correct for the given **grouped convolution**. This function
/// should most of the time be used together with `check_conv_args`.
///
/// It enforces that both the number of **input channels** and **output channels** are divisible
/// by `groups`.
pub(crate) fn check_groups_args(input_shape: &[usize], kernel_shape: &[usize], groups: usize) {
    assert_eq!(
        input_shape[1] % groups,
        0,
        "In channels {} is not divisible by groups {}",
        input_shape[1],
        groups
    );
    assert_eq!(
        kernel_shape[0] % groups,
        0,
        "Out channels {} is not divisible by groups {}",
        kernel_shape[0],
        groups
    );
}

#[cfg(test)]
pub(crate) const F16_EPSILON: f32 = 4.88e-04;

#[cfg(test)]
pub(crate) fn new_shared<T>(item: T) -> Rc<RefCell<T>> {
    Rc::new(RefCell::new(item))
}

#[cfg(test)]
pub(crate) fn are_similar<D: Dimension>(
    result: std::cell::Ref<Array<f32, D>>,
    expected: &Array<f32, D>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !result.abs_diff_eq(expected, F16_EPSILON) {
        return Err(format!("Result: {} | Expected: {}", result, expected).into());
    }

    Ok(())
}
