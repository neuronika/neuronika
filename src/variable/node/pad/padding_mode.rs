use ndarray::{ArrayView, ArrayViewMut, Dimension, RemoveAxis};

pub(crate) type SampleDim<D> = <<D as Dimension>::Smaller as Dimension>::Smaller;

/// Computes the shape of the **input** after the padding is applied.
///
/// This function expects arrays having shape (batch size, channels, ...).
///
/// # Arguments
///
/// * `shape` - shape of the input.
///
/// * `padding` - padding around the input.
fn padded_shape<D>(shape: &[usize], padding: &[usize]) -> D
where
    D: Dimension,
{
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
