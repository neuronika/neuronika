use ndarray::{Array, ArrayView, Dimension, Ix1, Ix2, Ix3, IxDyn, ShapeBuilder, Slice};

/// Checks that the arguments are correct
/// for the given convolution.
pub fn check_conv_args(
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
    if conv_dim != padding.len() {
        panic!(
            "error: invalid padding {:?} for {}d conv.",
            padding, conv_dim
        );
    }
    if conv_dim != stride.len() {
        panic!("error: invalid stride {:?} for {}d conv.", stride, conv_dim);
    }
    if conv_dim != dilation.len() {
        panic!(
            "error: invalid dilation {:?} for {}d conv.",
            dilation, conv_dim
        );
    }
    if kernel_shape.len() != input_shape.len() {
        panic!(
            "error: invalid kernel's shape {:?} for {}d conv",
            &kernel_shape, conv_dim
        );
    }
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
pub fn conv_out_shape<D: Dimension>(
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
///
/// * `input` - input array
///
/// * `window_shape` - the shape of each of the windows
///
/// * `win_indices_shape` - the number of rolling windows.
///
/// * `stride` - the stride
///
/// * `dilation` - the spacing between each element of the windows
pub fn as_windows<'a, D: Dimension>(
    input: &Array<f32, D>,
    window_shape: &[usize],
    win_indices_shape: D,
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
        view_strides.iter().cloned().collect()
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

/// Computes **sig2col**, **im2col** and **vol2col**.
///
/// # Arguments
///
/// * `input` - input array.
/// * `window_shape` - the shape of the kernel
/// * `padding` - the padding to be applied to `input`
/// * `stride` - the stride.
/// * `dilation` - the dilation.
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
    as_windows(&input, kernel_shape, o_shape, stride, dilation)
        .into_owned()
        .into_shape((im2col_w, im2col_h))
        .unwrap()
        .reversed_axes()
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
        for i in 0..out_len {
            if i < pad {
                pos = pad * 2 - i;
            } else if i >= pad && i < in_len + pad {
                pos = i;
            } else {
                pos = (in_len + pad - 1) * 2 - i;
            }
            pos -= pad;
            out_slice[i] = in_slice[pos];
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
        for j in 0..out_len {
            if j < pad {
                pos = pad;
            } else if j >= pad && j < in_len + pad {
                pos = j;
            } else {
                pos = in_len + pad - 1;
            }
            pos -= pad;
            out_slice[j] = in_slice[pos];
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
        // The nice thing about im2col is that it works for 1d, 2d, and 3d convolutions.
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
}
