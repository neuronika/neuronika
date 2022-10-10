/// Penalty trait, defines the penalty regularization's logic.
pub trait Penalty: Copy + Send + Sync {
    /// Applies the penalty to an element of the gradient.
    fn penalize(&self, w: &f32) -> f32;
}

/// L2 penalty, also known as *weight decay* or *Tichonov regularization*.
#[derive(Copy, Clone)]
pub struct L2 {
    lambda: f32,
}

impl L2 {
    /// Creates a new L2 penalty regularization.
    ///
    /// # Arguments
    ///
    /// `lambda` - weight decay coefficient.
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
}

/// L1 penalty.
#[derive(Copy, Clone)]
pub struct L1 {
    lambda: f32,
}

impl L1 {
    /// Creates a new L1 penalty regularization.
    ///
    /// # Arguments
    ///
    /// `lambda` - L1 regularization coefficient.
    pub fn new(lambda: f32) -> Self {
        Self { lambda }
    }
}
/// ElasticNet regularization, linearly combines the *L1* and *L2* penalties.
#[derive(Copy, Clone)]
pub struct ElasticNet {
    lambda_l1: f32,
    lambda_l2: f32,
}

impl ElasticNet {
    /// Creates a new ElasticNet penalty regularization.
    ///
    /// # Arguments
    ///
    /// * `lambda_l2` - L2 regularization coefficient.
    ///
    /// * `lambda_l1` - L1 regularization coefficient.
    pub fn new(lambda_l1: f32, lambda_l2: f32) -> Self {
        Self {
            lambda_l1,
            lambda_l2,
        }
    }
}

impl Penalty for L2 {
    fn penalize(&self, w: &f32) -> f32 {
        2. * self.lambda * w
    }
}

impl Penalty for L1 {
    fn penalize(&self, w: &f32) -> f32 {
        self.lambda * w.signum()
    }
}

impl Penalty for ElasticNet {
    fn penalize(&self, w: &f32) -> f32 {
        self.lambda_l1 * w.signum() + 2. * self.lambda_l2 * w
    }
}
