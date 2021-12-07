use ndarray::Dimension;
use crate::variable::{ VarDiff, };
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

mod tests {

    #[test]
    fn test_model_io() {
        use crate::nn::{Linear};
        use serde::{Serialize, Deserialize};

        // Construct an example network
        #[derive(Serialize, Deserialize)]
        struct MLP {
            lin1: Linear,
            lin2: Linear,
            lin3: Linear,
        }

        let mlp = MLP {
            lin1: Linear::new(2, 3),
            lin2: Linear::new(3, 2),
            lin3: Linear::new(2, 1),
        };

        // Convert to json
        let mlp_as_json = serde_json::to_string(&mlp).unwrap();

        // Convert from json back.
        let mlp_from_json : MLP = serde_json::from_str(&mlp_as_json).unwrap();

        // Did we get the same thing?  Floating point values may not be exactly preserved,
        // but the tensors should have the same shape.
        assert_eq!(mlp.lin1.weight.data().dim(), mlp_from_json.lin1.weight.data().dim());
        assert_eq!(mlp.lin2.weight.data().dim(), mlp_from_json.lin2.weight.data().dim());
        assert_eq!(mlp.lin3.weight.data().dim(), mlp_from_json.lin3.weight.data().dim());
    }

}