/// Specifies the reduction to apply to the *loss* output.
#[derive(Copy, Clone, Debug)]
pub enum Reduction {
    /// The output will be summed.
    Sum,
    /// The sum of the output will be divided by the batch size for the [`kldiv_loss`] and the
    /// [`nll_loss`]. For all other losses the output will be divided by the number of elements.
    Mean,
}
