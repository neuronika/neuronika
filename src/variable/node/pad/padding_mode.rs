use super::SampleDim;
use ndarray::{ArrayView, ArrayViewMut, Dimension, RemoveAxis};

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
