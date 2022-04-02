use ndarray::{ArrayView, ArrayViewMut, Dimension, RemoveAxis};

use super::SampleDim;
/// Padding mode.
pub trait PaddingMode<D>: Send + Sync + Copy
where
    D: Dimension,
    D::Smaller: RemoveAxis,
{
    fn pad(
        &self,
        padded: &mut ArrayViewMut<f32, SampleDim<D>>,
        base: &ArrayView<f32, SampleDim<D>>,
        padding: SampleDim<D>,
    );
}
