//! # Loss functions.
//!
//! The purpose of a loss function is to compute the quantity that a model should seek to minimize
//! during training.
//!
//! All losses are provided via function handles.
//!
//! ## Regression losses
//!
//! * [`mse_loss`] - Measures the mean squared error between each element in the input and the
//! target.
//!
//! * [`mae_loss`] - Measures the mean absolute error between each element in the input and the
//! target.
//!
//! ## Probabilistic losses
//!
//! * [`bce_loss`] - Measures the binary cross entropy between the target and the input.
//!
//! * [`bce_with_logits_loss`] - Measures the binary cross entropy with logits between the target
//! and the input.
//!
//! * [`nll_loss`] -  Measures the negative log likelihood between the target and the input.
//!
//! * [`kldiv_loss`] -  Measures the Kullback-Leibler divergence between the target and the input.
use super::{
    variable::{
        BCELoss, BCELossBackward, BCEWithLogitsLoss, BCEWithLogitsLossBackward, KLDivLoss,
        KLDivLossBackward, MAELoss, MAELossBackward, MSELoss, MSELossBackward, NLLLoss,
        NLLLossBackward, Overwrite,
    },
    Data, Gradient, Var, VarDiff,
};
use ndarray::Dimension;

/// Specifies the reduction to apply to the *loss* output.
#[derive(Clone)]
pub enum Reduction {
    /// The output will be summed.
    Sum,
    /// The sum of the output will be divided by the batch size for the [`kldiv_loss`] and the
    /// [`nll_loss`]. For all other losses the output will be divided by the number of elements.
    Mean,
}

/// Computes the **mean squared error** *(squared L2 norm)* between each element in the input x
/// and target y.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑ (xᵢ- ʏᵢ)²
///        n  i=1
/// ```
pub fn mse_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<MSELoss<T, V>, MSELossBackward<U, T, V>>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data<Dim = T::Dim>,
{
    input.var.past.merge(target.past);
    let forward_node = MSELoss::new(
        input.var.node.clone(),
        target.node.clone(),
        reduction.clone(),
    );
    let var = Var::from(forward_node, input.var.past);

    let backward_node = MSELossBackward::new(input.node, input.var.node, target.node, reduction);
    VarDiff::from(backward_node, input.past, var)
}

/// Computes the **mean absolute error** *(MAE)* between each element in the input x and target y.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑ |xᵢ- ʏᵢ|
///        n  i=1
/// ```
pub fn mae_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<MAELoss<T, V>, MAELossBackward<U, T, V>>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data<Dim = T::Dim>,
{
    input.var.past.merge(target.past);
    let forward_node = MAELoss::new(
        input.var.node.clone(),
        target.node.clone(),
        reduction.clone(),
    );
    let var = Var::from(forward_node, input.var.past);

    let backward_node = MAELossBackward::new(input.node, input.var.node, target.node, reduction);
    VarDiff::from(backward_node, input.past, var)
}

/// Computes the **binary cross entropy** between the target y and input x.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑ - [ʏᵢ * ln(xᵢ) + (1 - ʏᵢ) * ln(1 - xᵢ)]
///        n  i=1
/// ```
///
/// Note that the target y should be numbers between 0 and 1.
/// Notice that if a component of the input x is either 0 or 1,
/// one of the log terms would be mathematically undefined in the above loss equation.
/// Rust sets *ln(0) = -inf*, however, an infinite term in the loss equation is not desirable.
/// Our solution is that BCELoss clamps its log function outputs to be greater than or equal
/// to -100. This way, we can always have a finite loss value.
pub fn bce_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<BCELoss<T, V>, BCELossBackward<U, T, V>>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data<Dim = T::Dim>,
{
    input.var.past.merge(target.past);
    let forward_node = BCELoss::new(
        input.var.node.clone(),
        target.node.clone(),
        reduction.clone(),
    );
    let var = Var::from(forward_node, input.var.past);

    let backward_node = BCELossBackward::new(input.node, input.var.node, target.node, reduction);
    VarDiff::from(backward_node, input.past, var)
}

/// Computes the **binary cross entropy with logits** between the target y and input x.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑  - [ʏᵢ * ln(σ(xᵢ)) + (1 - ʏᵢ) * ln(1 - σ(xᵢ))]
///        n  i=1
/// ```
/// This loss combines a sigmoid and a binary cross entropy.
/// This version is more numerically stable than using a plain sigmoid followed by a
/// binary cross entropy as, by combining the operations into one layer, we take
/// advantage of the log-sum-exp trick for numerical stability.
/// Note that the target y should be numbers between 0 and 1 and the
/// input x should be raw unnormalized scores.
pub fn bce_with_logits_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<BCEWithLogitsLoss<T, V>, BCEWithLogitsLossBackward<U, T, V>>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data<Dim = T::Dim>,
{
    input.var.past.merge(target.past);
    let forward_node = BCEWithLogitsLoss::new(
        input.var.node.clone(),
        target.node.clone(),
        reduction.clone(),
    );
    let var = Var::from(forward_node, input.var.past);

    let backward_node =
        BCEWithLogitsLossBackward::new(input.node, input.var.node, target.node, reduction);
    VarDiff::from(backward_node, input.past, var)
}

/// Computes the **negative log likelihood** between the target y and input x.
///
/// ```text
///         1   n
/// Lᴏss =  ―   ∑  - xₙ,ᵧₙ
///         n  i=1
/// ```
///
/// The input x given is expected to contain log-probabilities for each class,
/// this is typically achieved by using [`.log_softmax()`]. input has to be a of size either
/// (minibatch, C) or (minibatch, C, d1, d2, ..., dk) with k >= 1 for the K-dimensional
/// case. The target that this loss expects should be a class index in the range [0, C) where
/// C = number of classes. When the given reduction is equal to [`Reduction::Mean`] the total
/// loss is divided by the batch size.
///
/// As mentioned before, this loss can also be used for higher dimensional inputs, such as 2D
/// images, by providing an input of size (minibatch, C, d1, d2, ..., dk) with k >= 1 where
/// k is the number of dimensions. In the case of images, it computes NLL loss *per-pixel*.
///
/// In the K-dimensional case this loss expects a target of shape
/// (minibatch, d1, d2, ..., dk).
///
/// [`.log_softmax()`]: VarDiff::log_softmax()
pub fn nll_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<NLLLoss<T, V>, NLLLossBackward<U, V>>
where
    T: Data<Dim = <V::Dim as Dimension>::Larger>,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data,
    T::Dim: Copy,
{
    input.var.past.merge(target.past);
    let forward_node = NLLLoss::new(
        input.var.node.clone(),
        target.node.clone(),
        reduction.clone(),
    );
    let var = Var::from(forward_node, input.var.past);

    let backward_node = NLLLossBackward::new(input.node, target.node, reduction);
    VarDiff::from(backward_node, input.past, var)
}

/// Computes the **Kullback-Leibler** divergence between the target and the input.
///
/// ```text
///         n
/// Lᴏss =  ∑  ʏₙ * (ln(ʏₙ) - xₙ)
///        i=1
/// ```
///
/// The [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) is
/// a useful distance measure for continuous distributions and is often useful when performing
/// direct regression over the space of (discretely sampled) continuous output distributions.
///
/// The input given is expected to contain log-probabilities and is not restricted to a 2D Tensor,
/// while the targets are interpreted as probabilities. When the given reduction is equal
/// to [`Reduction::Mean`] the total loss is divided by the batch size.
///
/// This criterion expects a target variable of the same size as the input variable.
pub fn kldiv_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<KLDivLoss<T, V>, KLDivLossBackward<U, V>>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data<Dim = T::Dim>,
{
    input.var.past.merge(target.past);
    let forward_node = KLDivLoss::new(
        input.var.node.clone(),
        target.node.clone(),
        reduction.clone(),
    );
    let var = Var::from(forward_node, input.var.past);

    let backward_node = KLDivLossBackward::new(input.node, target.node, reduction);
    VarDiff::from(backward_node, input.past, var)
}
