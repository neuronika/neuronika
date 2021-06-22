// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ losses module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    variable::node::{
        backward::{
            BCELossBackward, BCEWithLogitsLossBackward, KLDivLossBackward, MAELossBackward,
            MSELossBackward, NLLLossBackward,
        },
        forward::{BCELoss, BCEWithLogitsLoss, KLDivLoss, MAELoss, MSELoss, NLLLoss},
        Overwrite,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        variable::{
            node::{Backward, Differentiable, Forward},
            Tensor,
        },
        Input, InputBackward,
    };
    use ndarray::{Dimension, StrideShape, Zip};
    use std::rc::Rc;

    const F16_EPSILON: f32 = 9.77e-04;

    fn assert_almost_equals<D: Dimension>(our: &Tensor<D>, their: &Tensor<D>) {
        assert!(
            Zip::from(our).and(their).all(|l, r| {
                (*l == 0. && *r == 0.)
                    || (!l.is_finite() && !r.is_finite())
                    || ((1. - r / l).abs() <= F16_EPSILON)
            }),
            "\nLeft:\n{}\nRight:\n{}",
            our,
            their
        );
    }

    fn new_input<D, Sh>(shape: Sh, elems: Vec<f32>) -> Rc<Input<D>>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Input::new(new_tensor(shape, elems)).node
    }

    fn new_backward_input<D, Sh>(shape: Sh, elems: Vec<f32>) -> Rc<InputBackward<D>>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Rc::new(Input::new(new_tensor(shape, elems)).node.differentiable())
    }

    fn new_tensor<D, Sh>(shape: Sh, elems: Vec<f32>) -> Tensor<D>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Tensor::from_shape_vec(shape, elems).unwrap()
    }

    #[test]
    fn mae_loss_mean() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MAELoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![9.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            MAELossBackward::new(input_diff.clone(), input, target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor((3, 3), vec![0.1111; 9]),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor((3, 3), vec![0.2222; 9]),
        );
    }

    #[test]
    fn mae_loss_sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MAELoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![81.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward = MAELossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![1.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![2.; 9]));
    }

    #[test]
    fn mse_loss_mean() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MSELoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![81.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            MSELossBackward::new(input_diff.clone(), input, target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![2.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![4.; 9]));
    }

    #[test]
    fn mse_loss_sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MSELoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![729.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward = MSELossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![18.; 9]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![36.; 9]));
    }

    #[test]
    fn bce_loss_mean() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
        let loss = BCELoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![22.9244]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            BCELossBackward::new(input_diff.clone(), input, target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -1.1111e+00,
                    -1.2346e-01,
                    1.1111e+00,
                    0.0000e+00,
                    0.0000e+00,
                    -9.32067e+05,
                    5.5556e-01,
                    0.0000e+00,
                    -9.32067e+05,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 3),
                vec![
                    -1.1111e+00,
                    -1.2346e-01,
                    1.1111e+00,
                    0.0000e+00,
                    0.0000e+00,
                    -9.32067e+05,
                    5.5556e-01,
                    0.0000e+00,
                    -9.32067e+05,
                ],
            ) * 2.),
        );
    }

    #[test]
    fn bce_loss_sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
        let loss = BCELoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![206.3199]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward = BCELossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -1.0000e+01,
                    -1.1111e+00,
                    1.0000e+01,
                    0.0000e+00,
                    0.0000e+00,
                    -8.3886e+6,
                    5.0000e+00,
                    0.0000e+00,
                    -8.3886e+6,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 3),
                vec![
                    -1.0000e+01,
                    -1.1111e+00,
                    1.0000e+01,
                    0.0000e+00,
                    0.0000e+00,
                    -8.3886e+6,
                    5.0000e+00,
                    0.0000e+00,
                    -8.3886e+6,
                ],
            ) * 2.),
        );
    }

    #[test]
    fn bce_with_logits_loss_mean() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![8.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            BCEWithLogitsLossBackward::new(input_diff.clone(), input, target, Reduction::Mean);
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.0000050465264,
                    -0.0000018543667,
                    0.11111042,
                    0.11111086,
                    0.111111015,
                    -0.00000003973643,
                    0.1111111,
                    0.11111111,
                    0.,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 3),
                vec![
                    -0.0000050465264,
                    -0.0000018543667,
                    0.11111042,
                    0.11111086,
                    0.111111015,
                    -0.00000003973643,
                    0.1111111,
                    0.11111111,
                    0.,
                ],
            ) * 2.),
        );
    }

    #[test]
    fn bce_with_logits_loss_sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![72.0001]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            BCEWithLogitsLossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -0.00004541874,
                    -0.0000166893,
                    0.9999938,
                    0.99999774,
                    0.99999917,
                    -0.00000035762787,
                    0.9999999,
                    1.,
                    0.,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 3),
                vec![
                    -0.00004541874,
                    -0.0000166893,
                    0.9999938,
                    0.99999774,
                    0.99999917,
                    -0.00000035762787,
                    0.9999999,
                    1.,
                    0.,
                ],
            ) * 2.),
        );
    }

    #[test]
    fn nll_loss_mean() {
        use crate::variable::node::LogSoftmax;

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input(3, vec![2., 0., 4.]);
        let input = Rc::new(LogSoftmax::new(
            new_input(
                (3, 5),
                vec![
                    0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0., 0.2, 0.5,
                ],
            ),
            1,
        ));
        input.forward();

        let loss = NLLLoss::new(input, target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![1.52222]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((3, 5), vec![0.; 15]);
        let loss_backward = NLLLossBackward::new(input_diff.clone(), target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 5),
                vec![
                    0.0000, 0.0000, -0.3333, 0.0000, 0.0000, -0.3333, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.3333,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 5),
                vec![
                    0.0000, 0.0000, -0.3333, 0.0000, 0.0000, -0.3333, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.3333,
                ],
            ) * 2.),
        );
    }

    #[test]
    fn nll_loss_sum() {
        use crate::variable::node::LogSoftmax;

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input(3, vec![2., 0., 4.]);
        let input = Rc::new(LogSoftmax::new(
            new_input(
                (3, 5),
                vec![
                    0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0., 0.2, 0.5,
                ],
            ),
            1,
        ));
        input.forward();

        let loss = NLLLoss::new(input, target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![4.56666]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((3, 5), vec![0.; 15]);
        let loss_backward = NLLLossBackward::new(input_diff.clone(), target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 5),
                vec![
                    0.0000, 0.0000, -1.0000, 0.0000, 0.0000, -1.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.0000,
                ],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (3, 5),
                vec![
                    0.0000, 0.0000, -1.0000, 0.0000, 0.0000, -1.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.0000,
                ],
            ) * 2.),
        );
    }

    #[test]
    fn kldiv_loss_mean() {
        use crate::variable::node::Logn;

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((2, 3), vec![0.2, 0.5, 0.3, 0.6, 0.0, 0.4]);
        let input = Rc::new(Logn::new(new_input(
            (2, 3),
            vec![0.4, 0.5, 0.1, 0.6, 0.1, 0.3],
        )));
        input.forward();

        let loss = KLDivLoss::new(input, target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![0.1530]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((2, 3), vec![0.; 6]);
        let loss_backward = KLDivLossBackward::new(input_diff.clone(), target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (2, 3),
                vec![-0.1000, -0.2500, -0.1500, -0.3000, 0.0000, -0.2000],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (2, 3),
                vec![-0.1000, -0.2500, -0.1500, -0.3000, 0.0000, -0.2000],
            ) * 2.),
        );
    }

    #[test]
    fn kldiv_loss_loss_sum() {
        use crate::variable::node::Logn;

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((2, 3), vec![0.2, 0.5, 0.3, 0.6, 0.0, 0.4]);
        let input = Rc::new(Logn::new(new_input(
            (2, 3),
            vec![0.4, 0.5, 0.1, 0.6, 0.1, 0.3],
        )));
        input.forward();

        let loss = KLDivLoss::new(input, target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![0.3060]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((2, 3), vec![0.; 6]);
        let loss_backward = KLDivLossBackward::new(input_diff.clone(), target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (2, 3),
                vec![-0.2000, -0.5000, -0.3000, -0.6000, 0.0000, -0.4000],
            ),
        );

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &(&new_tensor(
                (2, 3),
                vec![-0.2000, -0.5000, -0.3000, -0.6000, 0.0000, -0.4000],
            ) * 2.),
        );
    }
}
