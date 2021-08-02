use ndarray::{Array, ArrayBase, Data, DataMut, Dimension, IntoDimension, Ix1, Ix2, Ix3, Slice};

/// Padding modes logic.
pub trait PaddingMode: Send + Sync + Clone {
    fn pad_inplace<D: ReflPad + ReplPad, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        array: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    );

    fn pad<D: ReflPad + ReplPad, E: IntoDimension<Dim = D>>(
        &self,
        input: &Array<f32, D>,
        padding: E,
    ) -> Array<f32, D>;
}

/// Zero padding.
///
/// See [`.pad()`](Self::pad()) for more informations.
#[derive(Clone)]
pub struct Zero;
/// Constant padding.
///
/// See [`.pad()`](Self::pad()) for more informations.
#[derive(Clone)]
pub struct Constant {
    pub value: f32,
}

impl Constant {
    pub fn new(value: f32) -> Self {
        Self { value }
    }
}
/// Reflective padding.
///
/// See [`.pad()`](Self::pad()) for more informations.
#[derive(Clone)]
pub struct Reflective;
/// Replicative padding.
///
/// See [`.pad()`](Self::pad()) for more informations.
#[derive(Clone)]
pub struct Replicative;

impl PaddingMode for Zero {
    /// Pads the input array in place with zeros.
    ///
    /// See [`.pad()`](Self::pad()) for more informations.
    ///
    /// # Arguments
    ///
    /// * `input` - array to be padded.
    ///
    /// * `original` - the original unpadded array.
    ///
    /// * `padding` - slice specifying the amount of padding for each dimension.
    ///
    /// # Panics
    ///
    /// If `padding` length doesn't match `input`'s dimensions.
    fn pad_inplace<D: Dimension, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        input: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    ) {
        assert_eq!(
            padding.len(),
            input.ndim(),
            "error: padding length {} doesn't match array dimensions {}",
            padding.len(),
            input.ndim()
        );
        constant_pad_inplace(input, original, padding, 0.);
    }

    /// Pads the input array with zeros.
    ///
    ///
    /// # Arguments
    ///
    /// * `input` - the array to be padded.
    ///
    /// * `padding` - the amount of padding for each dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuronika::nn::{PaddingMode, Zero};
    ///
    /// let padding = Zero;
    /// let arr = ndarray::array![
    ///    [1., 2., 3.],
    ///    [4., 5., 6.],
    ///    [7., 8., 9.]
    /// ];
    /// let padded = padding.pad(&arr, (1, 1));
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
    fn pad<D: Dimension, E: IntoDimension<Dim = D>>(
        &self,
        input: &Array<f32, D>,
        padding: E,
    ) -> Array<f32, D> {
        constant_pad(input, padding, 0.)
    }
}

impl PaddingMode for Constant {
    /// Pads the input array in place using a constant value.
    ///
    /// See [`.pad()`](Self::pad()) for more informations.
    ///
    /// # Arguments
    ///
    /// * `input` - array to be padded.
    ///
    /// * `original` - the original unpadded array.
    ///
    /// * `padding` - slice specifying the amount of padding for each dimension.
    ///
    /// # Panics
    ///
    /// If `padding` length doesn't match `input`'s dimensions.
    fn pad_inplace<D: Dimension, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        input: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    ) {
        assert_eq!(
            padding.len(),
            input.ndim(),
            "error: padding length {} doesn't match array dimensions {}",
            padding.len(),
            input.ndim()
        );
        let value = self.value;
        constant_pad_inplace(input, original, padding, value);
    }

    /// Pads the input array with a constant value.
    ///
    ///
    /// # Arguments
    ///
    /// * `input` - the array to be padded.
    ///
    /// * `padding` - the amount of padding for each dimension.
    ///
    /// * `value` - the value for the padding.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuronika::nn::{PaddingMode, Constant};
    ///
    /// let padding = Constant::new(8.);
    /// let arr = ndarray::array![
    ///    [1., 2., 3.],
    ///    [4., 5., 6.],
    ///    [7., 8., 9.]
    /// ];
    /// let padded = padding.pad(&arr, (1, 1));
    /// let result = ndarray::array![
    ///    [8., 8., 8., 8., 8.],
    ///    [8., 1., 2., 3., 8.],
    ///    [8., 4., 5., 6., 8.],
    ///    [8., 7., 8., 9., 8.],
    ///    [8., 8., 8., 8., 8.]
    /// ];
    ///
    /// assert_eq!(padded, result);
    /// ```
    fn pad<D: Dimension, E: IntoDimension<Dim = D>>(
        &self,
        input: &Array<f32, D>,
        padding: E,
    ) -> Array<f32, D> {
        let value = self.value;
        constant_pad(input, padding, value)
    }
}

impl PaddingMode for Reflective {
    /// Pads the input array in place using the reflection of its boundary.
    ///
    /// See [`.pad()`](Self::pad()) for more informations.
    ///
    /// # Arguments
    ///
    /// * `input` - array to be padded.
    ///
    /// * `original` - the original unpadded array.
    ///
    /// * `padding` - slice specifying the amount of padding for each dimension.
    ///
    /// # Panics
    ///
    /// If `padding` length doesn't match `input`'s dimensions.
    fn pad_inplace<D: ReflPad, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        input: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    ) {
        assert_eq!(
            padding.len(),
            input.ndim(),
            "error: padding length {} doesn't match array dimensions {}",
            padding.len(),
            input.ndim()
        );
        D::reflection_pad_inplace(input, original, padding);
    }

    /// Pads the input array using the **reflection** of the input boundary.
    ///
    /// Only **1**, **2** and **3** dimensional arrays support reflective padding.
    ///
    /// # Arguments
    ///
    /// * `input` - the array to be padded.
    ///
    /// * `padding` - the amount of padding for each dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuronika::nn::{PaddingMode, Reflective};
    ///
    /// let padding = Reflective;
    /// let arr = ndarray::array![
    ///    [1., 2., 3.],
    ///    [4., 5., 6.],
    ///    [7., 8., 9.]
    /// ];
    ///
    /// let padded = padding.pad(&arr, (1, 1));
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
    fn pad<D: ReflPad, E: IntoDimension<Dim = D>>(
        &self,
        input: &Array<f32, D>,
        padding: E,
    ) -> Array<f32, D> {
        D::reflection_pad(input, padding.into_dimension().slice())
    }
}

impl PaddingMode for Replicative {
    /// Pads the input array in place using the replication of its boundary.
    ///
    /// See [`.pad()`](Self::pad()) for more informations.
    ///
    /// # Arguments
    ///
    /// * `input` - array to be padded.
    ///
    /// * `original` - the original unpadded array.
    ///
    /// * `padding` - slice specifying the amount of padding for each dimension.
    ///
    /// # Panics
    ///
    /// If `padding` length doesn't match `input`'s dimensions.
    fn pad_inplace<D: ReplPad, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        input: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    ) {
        assert_eq!(
            padding.len(),
            input.ndim(),
            "error: padding length {} doesn't match array dimensions {}",
            padding.len(),
            input.ndim()
        );
        D::replication_pad_inplace(input, original, padding);
    }

    /// Pads the input array using the **replication** of its boundary.
    ///
    /// Only **1**, **2** and **3** dimensional arrays support replicative padding.
    ///
    /// # Arguments
    ///
    /// * `input` - the array to be padded.
    ///
    /// * `padding` - the amount of padding for each dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuronika::nn::{Replicative, PaddingMode};
    ///
    /// let padding = Replicative;
    /// let arr = ndarray::array![
    ///    [1., 2., 3.],
    ///    [4., 5., 6.],
    ///    [7., 8., 9.]
    /// ];
    ///
    /// let padded = padding.pad(&arr, (1, 1));
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
    fn pad<D: ReplPad, E: IntoDimension<Dim = D>>(
        &self,
        input: &Array<f32, D>,
        padding: E,
    ) -> Array<f32, D> {
        D::replication_pad(input, padding.into_dimension().slice())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Paddings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A [`ndarray::Dimension`] that supports reflective padding.
pub trait ReflPad: Dimension {
    fn reflection_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Self>,
        padding: &[usize],
    ) -> Array<f32, Self>;

    fn reflection_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    );
}

/// A [`ndarray::Dimension`] that supports replicative padding.
pub trait ReplPad: Dimension {
    fn replication_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Self>,
        padding: &[usize],
    ) -> Array<f32, Self>;

    fn replication_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    );
}

fn constant_pad<S, D, E>(input: &ArrayBase<S, D>, padding: E, val: f32) -> Array<f32, D>
where
    D: Dimension,
    S: DataMut<Elem = f32>,
    E: IntoDimension<Dim = D>,
{
    let padding_into_dim = padding.into_dimension();
    let padded_shape = {
        let mut padded_shape = input.raw_dim();
        padded_shape
            .slice_mut()
            .iter_mut()
            .zip(padding_into_dim.slice().iter())
            .for_each(|(ax_len, pad)| *ax_len += pad * 2);
        padded_shape
    };
    let mut padded = Array::zeros(padded_shape);
    constant_pad_inplace(&mut padded, &input, padding_into_dim.slice(), val);
    padded
}

/// Pads the input array with a constant value. The operation is done inplace.
fn constant_pad_inplace<S, T, D>(
    input: &mut ArrayBase<S, D>,
    original: &ArrayBase<T, D>,
    padding: &[usize],
    val: f32,
) where
    D: Dimension,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
{
    input.map_inplace(|el| *el = val);
    let mut orig_portion = input.view_mut();
    orig_portion.slice_each_axis_inplace(|ax| {
        let (ax_index, ax_len) = (ax.axis.index(), original.len_of(ax.axis));
        let range = {
            if padding[ax_index] != 0 {
                padding[ax_index] as isize..-(padding[ax_index] as isize)
            } else {
                0..ax_len as isize
            }
        };
        Slice::from(range)
    });
    orig_portion.assign(original);
}

impl ReflPad for Ix1 {
    fn reflection_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix1>,
        padding: &[usize],
    ) -> Array<f32, Ix1> {
        let out_len = {
            let len = input.len();
            let pad = padding[0];
            len + pad * 2
        };
        let mut out = Array::<f32, _>::zeros(out_len);
        Self::reflection_pad_inplace(&mut out, input, padding);
        out
    }

    fn reflection_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Ix1>,
        input: &ArrayBase<T, Ix1>,
        padding: &[usize],
    ) {
        let mut pos;
        let (in_len, out_len, pad) = { (input.len(), to_pad.len(), padding[0]) };
        let (in_slice, out_slice) = (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap());
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
    }
}

impl ReflPad for Ix2 {
    fn reflection_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix2>,
        padding: &[usize],
    ) -> Array<f32, Ix2> {
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let mut out = Array::<f32, _>::zeros((out_len_x, out_len_y));
        Self::reflection_pad_inplace(&mut out, &input, &padding);
        out
    }

    fn reflection_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Ix2>,
        input: &ArrayBase<T, Ix2>,
        padding: &[usize],
    ) {
        let (mut pos_x, mut pos_y);
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap()) };
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
    }
}

impl ReflPad for Ix3 {
    fn reflection_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix3>,
        padding: &[usize],
    ) -> Array<f32, Ix3> {
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let mut out = Array::<f32, _>::zeros((out_len_z, out_len_x, out_len_y));
        Self::reflection_pad_inplace(&mut out, &input, padding);
        out
    }

    fn reflection_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    ) {
        let (mut pos_x, mut pos_y, mut pos_z);
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap()) };

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
    }
}

impl ReplPad for Ix1 {
    fn replication_pad<S: Data<Elem = f32>>(
        input: &ArrayBase<S, Ix1>,
        padding: &[usize],
    ) -> Array<f32, Ix1> {
        let out_len = {
            let len = input.len();
            let pad = padding[0];
            len + pad * 2
        };
        let mut out = Array::<f32, _>::zeros(out_len);
        Self::replication_pad_inplace(&mut out, &input, padding);
        out
    }

    fn replication_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    ) {
        let mut pos;
        let (in_len, out_len, pad) = (input.len(), to_pad.len(), padding[0]);
        let (in_slice, out_slice) = (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap());
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
    }
}

impl ReplPad for Ix2 {
    fn replication_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix2>,
        padding: &[usize],
    ) -> Array<f32, Ix2> {
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let mut out = Array::<f32, _>::zeros((out_len_x, out_len_y));
        Self::replication_pad_inplace(&mut out, &input, padding);
        out
    }

    fn replication_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    ) {
        let (mut pos_x, mut pos_y);
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap()) };
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
    }
}

impl ReplPad for Ix3 {
    fn replication_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix3>,
        padding: &[usize],
    ) -> Array<f32, Ix3> {
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let mut out = Array::<f32, _>::zeros((out_len_z, out_len_x, out_len_y));
        Self::replication_pad_inplace(&mut out, &input, padding);
        out
    }

    fn replication_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    ) {
        let (mut pos_x, mut pos_y, mut pos_z);
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap()) };
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
    }
}

#[cfg(test)]
mod test;
