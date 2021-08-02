use super::Optimizer;

/// Learning rate scheduler trait, defines the scheduler's logic.
pub trait LRScheduler {
    /// Updates the learning rate.
    fn step(&mut self);

    /// Returns an immutable reference to the last computed learning rate.
    fn get_last_lr(&self) -> &f32;

    /// Returns an immutable reference to the current learning rate.
    fn get_current_lr(&self) -> &f32;

    /// Returns an immutable reference to the current epoch.
    fn get_current_epoch(&self) -> &usize;

    /// Sets the current epoch.
    fn set_current_epoch(&mut self, epoch: usize);

    /// Prints the update of the learning rate. It should be called after `.step()`.
    fn print_lr(&self) {
        println!(
            "epoch {}: learning rate adjusted to {}",
            self.get_current_epoch() - 1,
            self.get_current_lr()
        );
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LambdaLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Sets the learning rate to the initial lr times a given function.
///
///```text
/// lrₜ = lr₀ * lr_fn(epoch)
///```
pub struct LambdaLR<'a, T: Optimizer, F: Fn(usize) -> f32> {
    optimizer: &'a T,
    lr_fn: F,
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
    initial_lr: f32,
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> LambdaLR<'a, T, F> {
    /// Creates a new LambdaLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `lr_fn` - function which computes a multiplicative factor given an `usize` parameter
    /// epoch.
    pub fn new(optimizer: &'a T, lr_fn: F) -> Self {
        let current_lr = optimizer.get_lr();
        Self {
            optimizer,
            lr_fn,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
            initial_lr: current_lr,
        }
    }
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> LRScheduler for LambdaLR<'a, T, F> {
    fn step(&mut self) {
        self.current_epoch += 1;

        self.last_lr = self.current_lr;
        self.current_lr = self.initial_lr * (self.lr_fn)(self.current_epoch);
        self.optimizer.set_lr(self.current_lr);
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiplicativeLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Multiplies the learning rate by the factor given in the specified function.
///
///```text
/// lrₜ = lrₜ₋₁ * lr_fn(epoch)
///```
pub struct MultiplicativeLR<'a, T: Optimizer, F: Fn(usize) -> f32> {
    optimizer: &'a T,
    lr_fn: F,
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> MultiplicativeLR<'a, T, F> {
    /// Creates a new MultiplicativeLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `lr_fn` - function which computes a multiplicative factor given an `usize` parameter
    /// epoch.
    pub fn new(optimizer: &'a T, lr_fn: F) -> Self {
        let current_lr = optimizer.get_lr();
        Self {
            optimizer,
            lr_fn,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
        }
    }
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> LRScheduler for MultiplicativeLR<'a, T, F> {
    fn step(&mut self) {
        self.current_epoch += 1;

        self.last_lr = self.current_lr;
        self.current_lr *= (self.lr_fn)(self.current_epoch);
        self.optimizer.set_lr(self.current_lr);
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ StepLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Decays the learning rate by `gamma` every `step_size` epochs.
///
///```text
/// lrₜ = lrₜ₋₁ * gamma if t mod step_size == 0 else lrₜ₋₁
///```
pub struct StepLR<'a, T: Optimizer> {
    optimizer: &'a T,
    gamma: f32,
    step_size: usize,
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
}

impl<'a, T: Optimizer> StepLR<'a, T> {
    /// Creates a new StepLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `step_size` - period of learning rate decay.
    ///
    /// * `gamma` - multiplicative factor for the learning rate decay.
    pub fn new(optimizer: &'a T, step_size: usize, gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma,
            step_size,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
        }
    }
}

impl<'a, T: Optimizer> LRScheduler for StepLR<'a, T> {
    fn step(&mut self) {
        self.current_epoch += 1;

        if self.current_epoch.rem_euclid(self.step_size) == 0 {
            self.last_lr = self.current_lr;
            self.current_lr = self.last_lr * self.gamma;
            self.optimizer.set_lr(self.current_lr);
        }
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiStepLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Decays the learning rate by gamma once the number of epoch reaches one of the specified
/// milestones.
///
///```text
/// lrₜ = lrₜ₋₁ * gamma if t is a milestone else lrₜ₋₁
///```
pub struct MultiStepLR<'a, T: Optimizer, const N: usize> {
    optimizer: &'a mut T,
    gamma: f32,
    milestones: [usize; N],
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
}

impl<'a, T: Optimizer, const N: usize> MultiStepLR<'a, T, N> {
    /// Creates a new MultiStepLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `milestones` - list of epoch indices. Must be increasing.
    ///
    /// * `gamma` - multiplicative factor for the learning rate decay.
    pub fn new(optimizer: &'a mut T, milestones: [usize; N], gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma,
            milestones,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
        }
    }
}

impl<'a, T: Optimizer, const N: usize> LRScheduler for MultiStepLR<'a, T, N> {
    fn step(&mut self) {
        self.current_epoch += 1;

        if self
            .milestones
            .iter()
            .any(|milestone| *milestone == self.current_epoch)
        {
            self.last_lr = self.current_lr;
            self.current_lr = self.last_lr * self.gamma;
            self.optimizer.set_lr(self.current_lr);
        }
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ExponentialLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Decays the learning rate by `gamma` every epoch.
///
///```text
/// lrₜ = lrₜ₋₁ * gamma
///```
pub struct ExponentialLR<'a, T: Optimizer> {
    optimizer: &'a T,
    gamma: f32,
    current_epoch: usize,
    current_lr: f32,
    last_lr: f32,
}

impl<'a, T: Optimizer> ExponentialLR<'a, T> {
    /// Creates a new ExponentialLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `gamma` - multiplicative factor for the learning rate decay.
    pub fn new(optimizer: &'a T, gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma,
            current_epoch: 0,
            current_lr,
            last_lr: 0.0,
        }
    }
}

impl<'a, T: Optimizer> LRScheduler for ExponentialLR<'a, T> {
    fn step(&mut self) {
        self.current_epoch += 1;

        self.last_lr = self.current_lr;
        self.current_lr = self.last_lr * self.gamma;
        self.optimizer.set_lr(self.current_lr);
    }

    fn get_last_lr(&self) -> &f32 {
        &self.last_lr
    }

    fn get_current_lr(&self) -> &f32 {
        &self.current_lr
    }

    fn set_current_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    fn get_current_epoch(&self) -> &usize {
        &self.current_epoch
    }
}
