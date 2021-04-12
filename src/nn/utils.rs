use ndarray::{Array, ArrayView, Dimension, Ix2, IxDyn, ShapeBuilder, Slice};

/// Checks that the arguments are correct
/// for the given convolution.
fn check_conv_args(
    input_shape: &[usize],
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) {
    // First two axes of input are always for batch size
    // and input channels.
    let conv_dim = input_shape.len() - 2;
    assert_eq!(
        conv_dim,
        padding.len(),
        "error: invalid padding: {:?} for {}d conv.",
        padding,
        conv_dim
    );
    assert_eq!(
        conv_dim,
        stride.len(),
        "error: invalid stride: {:?} for {}d conv.",
        stride,
        conv_dim
    );
    assert_eq!(
        conv_dim,
        dilation.len(),
        "error: invalid dilation: {:?} for {}d conv.",
        dilation,
        conv_dim
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
/// `input` - input array
///
/// `window_shape` - the shape of each of the windows
///
/// `padding` - the padding around `input`
///
/// `stride` - the stride
///
/// `dilation` - the spacing between each element of the windows
pub fn as_windows<'a, D: Dimension>(
    input: &Array<f32, D>,
    window_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayView<'a, f32, IxDyn> {
    let ndim = input.ndim();
    let input_shape = input.shape();

    let mut indexing_strides = vec![0; ndim];
    {
        let view = input.slice_each_axis(|ax| {
            let axis_index = ax.axis.index();
            if axis_index == 0 || axis_index == 1 {
                Slice::new(0, None, 1) // Batch stride and channel stride
            } else {
                Slice::new(0, None, stride[ax.axis.index() - 2] as isize)
            }
        });
        let view_strides: &[isize] = view.strides();
        indexing_strides
            .iter_mut()
            .zip(view_strides)
            .for_each(|(is, vs)| *is = *vs);
    }

    let mut window_strides = vec![0; window_shape.len()];
    // Number of out channels and in channels
    // don't count for the window's strides, they
    // must be left unchanged.
    window_strides
        .iter_mut()
        .take(2)
        .zip(input.strides().iter().take(2))
        .for_each(|(ws, is)| *ws = *is);

    itertools::izip!(
        window_strides.iter_mut().skip(2), // Skip the out and in channels stride.
        input.strides().iter().skip(2),    // Again, we skip the batch size and in channels.
        dilation
    )
    .for_each(|(ws, is, dil)| *ws = *is * (*dil as isize));

    let win_indices_shape =
        conv_out_shape::<D>(input_shape, window_shape, padding, stride, dilation);

    let mut new_shape = IxDyn::zeros(win_indices_shape.ndim() + window_shape.len());
    let mut strides = IxDyn::zeros(win_indices_shape.ndim() + window_shape.len());

    new_shape
        .slice_mut()
        .iter_mut()
        .zip(win_indices_shape.slice().iter().chain(window_shape.iter()))
        .for_each(|(ns, _s)| *ns = *_s as usize);

    strides
        .slice_mut()
        .iter_mut()
        .zip(indexing_strides.iter().chain(window_strides.iter()))
        .for_each(|(s, _s)| *s = *_s as usize);

    unsafe { ArrayView::from_shape_ptr(new_shape.strides(strides), input.as_ptr()) }
}

/// Computes **im2col**.
pub fn im2col<D: Dimension>(
    input: &Array<f32, D>,
    window_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Array<f32, Ix2> {
    let i_shape = input.shape();
    // The type of convolution can be derived by
    // considering the number of input's dimension
    // skipping the batch size and input channels.
    let conv_dim = input.ndim() - 2;

    // Checks the argument correctness.
    check_conv_args(i_shape, window_shape, padding, stride, dilation);

    // Computes the shape of the convolution result.
    let o_shape = conv_out_shape::<D>(input.shape(), window_shape, padding, stride, dilation);
    // Computes the matrix height and width.
    let (im2col_h, im2col_w): (usize, usize) = {
        (
            // Multiply all the components of the kernel
            // but the number of the output channels.
            window_shape.iter().skip(1).product(),
            // Multiply all the components
            // of the output shape except for
            // the channels.
            o_shape.slice().iter().rev().take(conv_dim).product(),
        )
    };
    // Axes must be swapped.
    as_windows(&input, window_shape, padding, stride, dilation)
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
        let d = input.into_shape((1, 3, 4, 4)).unwrap();

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
    }
}
