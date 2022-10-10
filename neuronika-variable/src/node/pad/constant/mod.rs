use ndarray::{ArrayView, ArrayViewMut, Dimension, RemoveAxis, Slice};

use super::{PaddingMode, SampleDim};

/// Constant padding.
#[derive(Copy, Clone, Debug)]
pub struct Constant(pub f32);

impl<D> PaddingMode<D> for Constant
where
    D: Dimension,
    D::Smaller: RemoveAxis,
{
    fn pad(
        &self,
        padded: &mut ArrayViewMut<f32, SampleDim<D>>,
        base: &ArrayView<f32, SampleDim<D>>,
        padding: SampleDim<D>,
    ) {
        padded.map_inplace(|el| *el = self.0);

        let padding_slice = padding.slice();

        let mut base_slice = padded.view_mut();

        base_slice.slice_each_axis_inplace(|ax| {
            let (ax_index, ax_len) = (ax.axis.index(), base.len_of(ax.axis));
            let range = {
                if padding_slice[ax_index] != 0 {
                    padding_slice[ax_index] as isize..-(padding_slice[ax_index] as isize)
                } else {
                    0..ax_len as isize
                }
            };
            Slice::from(range)
        });

        base_slice.assign(base);
    }
}

#[cfg(test)]
mod test;
