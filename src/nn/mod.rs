pub mod init {

    // Returns the recommended gain value for the given nonlinearity function.
    pub fn calculate_gain(non_linearity: &str, param: Option<f32>) -> f32 {
        match (non_linearity, param) {
            ("linear", None) | ("sigmoid", None) => 1.0,
            ("tanh", None) => 5.0 / 3.0,
            ("relu", None) => 2.0_f32.sqrt(),
            ("leaky_relu", Some(param_val)) => {
                let slope = param_val;
                (2.0 / (1.0 + slope.powi(2))).sqrt()
            }
            _ => {
                if let Some(p) = param {
                    panic!(
                        "error: unsupported nonlinearity: {} with param: {}.",
                        non_linearity, p
                    )
                } else {
                    panic!("error: this nonlinearity: {} needs a param.", non_linearity)
                }
            }
        }
    }
    // TODO: layers' init functions.
    // See https://pytorch.org/docs/stable/nn.init.html
}
