mod bce_loss;
mod bce_with_logits_loss;
mod kldiv_loss;
mod mae_loss;
mod mse_loss;
mod nll_loss;

use super::{
    expect_tensor, expect_tensor_mut, Backward, Cache, Data, Forward, Gradient, Overwrite, Tensor,
};

use crate::nn::loss::Reduction;

#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};

pub(crate) use bce_loss::{BCELoss, BCELossBackward};
pub(crate) use bce_with_logits_loss::{BCEWithLogitsLoss, BCEWithLogitsLossBackward};
pub(crate) use kldiv_loss::{KLDivLoss, KLDivLossBackward};
pub(crate) use mae_loss::{MAELoss, MAELossBackward};
pub(crate) use mse_loss::{MSELoss, MSELossBackward};
pub(crate) use nll_loss::{NLLLoss, NLLLossBackward};
