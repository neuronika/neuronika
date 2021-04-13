use ndarray::{Array, ArrayView, Dimension, Ix2, IxDyn, ShapeBuilder, Slice};

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

/// Computes **im2col**.
///
/// # Arguments
///
/// * `input` - input array.
/// * `window_shape` - the shape of the kernel
/// * `padding` - the padding to be applied to `input`
/// * `stride` - the stride.
/// * `dilation` - the dilation.
pub fn im2col<D: Dimension>(
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
            super::im2col(&d, &[1, 3, 3, 3], &[0, 0], &[1, 1], &[1, 1])
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
            super::im2col(&d, &[1, 3, 3, 3], &[0, 0], &[1, 1], &[1, 1])
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
}
