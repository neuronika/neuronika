use ndarray::Dimension;
//#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

use crate::variable::{
    self, Backward, Convolve, ConvolveWithGroups, Data,
    Eval, Forward, Gradient, MatMatMulT, Overwrite,
    RawParam, Tensor, Var, VarDiff,
};
use crate::variable::node::{ Input, InputBackward };

/// Implement serialize and deserialize for Learnable parameters
/// Note, we only serialize the learnable parameters, not the gradients.

impl<D> serde::ser::Serialize for VarDiff<Input<D>, InputBackward<D>>
where
    D: Dimension + serde::ser::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::ser::Serializer,
    {
        self.data().serialize(serializer)
    }
}

impl<'d,D> serde::de::Deserialize<'d> for VarDiff<Input<D>, InputBackward<D>>
where
    D: Dimension + serde::de::Deserialize<'d>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
        where
            De: serde::de::Deserializer<'d>,
    {
        let data = ndarray::Array::<f32, D>::deserialize(deserializer).unwrap();
        Ok(Input::new(data).requires_grad())
    }
}
