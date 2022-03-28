use ndarray::{ArrayView, ArrayViewMut, Dimension, Ix3, Ix4, Ix5};

use super::{PaddingMode, SampleDim};

/// Reflective padding.
#[derive(Copy, Clone, Debug)]
pub struct Reflective;

impl PaddingMode<Ix3> for Reflective {
    fn pad(
        &self,
        padded: &mut ArrayViewMut<f32, SampleDim<Ix3>>,
        base: &ArrayView<f32, SampleDim<Ix3>>,
        padding: SampleDim<Ix3>,
    ) {
        let mut pos;

        let (base_len, pad) = (base.len(), padding.into_pattern());
        let (base_slice, padded_slice) = (base.as_slice().unwrap(), padded.as_slice_mut().unwrap());

        for (i, padded_slice_el) in padded_slice.iter_mut().enumerate() {
            if i < pad {
                pos = pad * 2 - i;
            } else if i >= pad && i < base_len + pad {
                pos = i;
            } else {
                pos = (base_len + pad - 1) * 2 - i;
            }

            pos -= pad;
            *padded_slice_el = base_slice[pos];
        }
    }
}

impl PaddingMode<Ix4> for Reflective {
    fn pad(
        &self,
        padded: &mut ArrayViewMut<f32, SampleDim<Ix4>>,
        base: &ArrayView<f32, SampleDim<Ix4>>,
        padding: SampleDim<Ix4>,
    ) {
        let (mut pos_x, mut pos_y);

        let (len_x, len_y) = {
            let base_shape = base.shape();
            (base_shape[0], base_shape[1])
        };

        let (pad_x, pad_y) = padding.into_pattern();
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let (slice_in, slice_out) = (base.as_slice().unwrap(), padded.as_slice_mut().unwrap());

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

impl PaddingMode<Ix5> for Reflective {
    fn pad(
        &self,
        padded: &mut ArrayViewMut<f32, SampleDim<Ix5>>,
        base: &ArrayView<f32, SampleDim<Ix5>>,
        padding: SampleDim<Ix5>,
    ) {
        let (mut pos_x, mut pos_y, mut pos_z);

        let (len_x, len_y, len_z) = {
            let base_shape = base.shape();
            (base_shape[1], base_shape[2], base_shape[0])
        };

        let (pad_z, pad_x, pad_y) = padding.into_pattern();
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let (slice_in, slice_out) = { (base.as_slice().unwrap(), padded.as_slice_mut().unwrap()) };

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

#[cfg(test)]
mod test;
