use neuronika::nn::{ModelStatus, Linear, Learnable};
use neuronika::{ Param, VarDiff, Data, Gradient, Forward, Overwrite, Backward, MatMatMulT };
use ndarray::{Array, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Ix1, Ix2, Ix3};
use ndarray_rand::{RandomExt};
use serde::{Serialize, Deserialize};
use serde_json;

// Network model.

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
struct MLP {
    lin1: Linear,
    lin2: Linear,
    lin3: Linear,
    #[cfg_attr(feature = "serialize", serde(skip))]
    status: ModelStatus,
}

impl MLP {
    fn new() -> Self {
        let mut status = ModelStatus::default();

        Self {
            lin1: status.register(Linear::new(3, 5)),
            lin2: status.register(Linear::new(5, 5)),
            lin3: status.register(Linear::new(5, 1)),
            status,
        }
    }

    // This is needed after loading the model from a file.
    fn register_parameters(& mut self)
    {
        self.status = ModelStatus::default();
        self.status.register_ref(&mut self.lin1);
        self.status.register_ref(&mut self.lin2);
        self.status.register_ref(&mut self.lin3);
    }

    fn parameters(&self) -> Vec<Param> {
        self.status.parameters()
    }

    // MLP behavior. Notice the presence of the ReLU non-linearity.
     fn forward<I, T, U>(
         &self,
         input: I,
     ) -> VarDiff<
             impl Data<Dim = Ix2> + Forward,
             impl Gradient<Dim = Ix2> + Overwrite + Backward
         >
     where
         I: MatMatMulT<Learnable<Ix2>>,
         I::Output: Into<VarDiff<T, U>>,
         T: Data<Dim = Ix2> + Forward,
         U: Gradient<Dim = Ix2> + Backward + Overwrite,
     {
         let out1 = self.lin1.forward(input).relu();
         let out2 = self.lin2.forward(out1).relu();
         let out3 = self.lin3.forward(out2);
         out3
    }
}

/// Load model from a string
fn load_model() -> MLP {
    let mut net : MLP = serde_json::from_str(
        r#"{"lin1":{"weight":{"v":1,"dim":[5,3],"data":[0.31398147,-0.02374097,-0.045672387,-0.57606286,0.5287176,-0.038059983,-0.19196294,0.9338395,-0.34874597,-0.08579302,-0.21880743,0.26289353,0.12593554,-0.19557185,-0.6770759]},"bias":{"v":1,"dim":[5],"data":[0.036578782,-0.3663301,-0.23192844,0.1254652,0.5213851]}},"lin2":{"weight":{"v":1,"dim":[5,5],"data":[0.52091986,0.3500197,-0.06102618,-0.43995684,0.53706765,-0.09257236,-0.3584929,-0.43666622,0.43744308,-0.40631944,0.066774696,0.16129021,-0.25963476,0.26902968,0.1528883,0.12935583,-0.2496377,0.14702061,-0.012540738,-0.34052926,0.45684096,-0.12884608,0.21005273,-0.7786633,-0.08895902]},"bias":{"v":1,"dim":[5],"data":[0.6071196,-0.18910336,-0.2278286,0.044481196,0.10841279]}},"lin3":{"weight":{"v":1,"dim":[1,5],"data":[0.21673596,-0.021770507,-0.00067504647,0.5252394,0.06640336]},"bias":{"v":1,"dim":[1],"data":[0.7723236]}}}"#,
    ).unwrap();
    net.register_parameters();
    net
}

fn main()
{
    use neuronika::data::DataLoader;

    // Data set.
    let csv_content = "\
        Paw_size,Tail_length,Weight,Animal\n\
        0.2,5.0,15.0,Dog\n\
        0.08,12.0,4.0,Cat\n\
        0.07,13.0,5.0,Cat\n\
        0.05,3.0,0.8,Mouse";

    // Setup data loader
    let mut dataset = DataLoader::default().with_labels(&[3]).from_reader_fn(
        csv_content.as_bytes(),
        3,
        1,
        |(record, label): (Vec<f32>, String)| {
            let float_label = match label.as_str() {
                "Dog" => 1.,
                "Cat" => 2.,
                 _ => 3.,
            };
            (record, vec![float_label])
        },
    );

    // Create network
    let mut model : MLP = MLP::new();

    // Load model from a string.
    model = load_model();

    // Setup optimizer
    let mut optimizer = neuronika::optim::SGD::new(model.parameters(), 0.01, neuronika::optim::L2::new(0.0));

    // Train network
    for epoch in 0..5 {
        let batchedData = dataset.shuffle().batch(2).drop_last();
        let mut total_loss:f32  = 0.0;
        for (input_array, target_array) in batchedData {
            let input = neuronika::from_ndarray(input_array.to_owned());
            let target = neuronika::from_ndarray(target_array.to_owned());
            let result = model.forward(input);
            let loss = neuronika::nn::loss::mse_loss(result.clone(), target.clone(), neuronika::nn::loss::Reduction::Mean);
            loss.forward();
            total_loss += loss.data()[0];
            loss.backward(1.0);
            optimizer.step();
        }
        println!("Loss for epoch {} : {} ", epoch, total_loss);
    }

    // Save the model to json
    println!("Model parameters: {}", serde_json::to_string(&model).unwrap());
}