//! Learning rate schedulers.
//!
//! Learning rate scheduling should be applied after optimizer’s update;
//! for instance, you should write your code this way:
//!
//! ```
//! # use neuronika::optim;
//! # use neuronika::optim::Optimizer;
//! # use neuronika::optim::lr_scheduler::LRScheduler;
//! # const EPOCHS: usize = 5;
//! # let optim = optim::SGD::new(vec![], 1., optim::L2::new(0.1));
//! # let scheduler = optim::lr_scheduler::LambdaLR::new(&optim, |epoch| epoch as f32);
//! # let mut loss = neuronika::ones(1).requires_grad() + 0.;
//! for epoch in 0..EPOCHS {
//!    loss.forward();
//!    loss.backward(1.0);
//!    optim.step();
//!    optim.zero_grad();
//!    scheduler.step();
//! }
//! ```
//!
//! Learning rate schedulers can be chained together. The result is that each scheduler is applied
//! one after the other on the learning rate obtained by the one preceding it.
//!
//! ```
//! # use neuronika::optim;
//! # use neuronika::optim::{SGD,Optimizer, L2};
//! # use neuronika::optim::lr_scheduler::{LRScheduler, LambdaLR, MultiplicativeLR};
//! # const EPOCHS: usize = 5;
//! let optim = SGD::new(vec![], 0.01, L2::new(0.1));
//! let scheduler1 = LambdaLR::new(&optim, |epoch| 1.0_f32 / epoch as f32);
//! let scheduler2 = MultiplicativeLR::new(&optim, |epoch| 0.1_f32 * epoch as f32);
//! # let mut loss = neuronika::ones(1).requires_grad() + 0.;
//!
//! for epoch in 0..EPOCHS {
//!    loss.forward();
//!    loss.backward(1.0);
//!    optim.step();
//!    optim.zero_grad();
//!    scheduler1.step();
//!    scheduler2.step();
//! }
//! ```
use super::Optimizer;
use std::cell::Cell;

/// Learning rate scheduler trait, defines the scheduler's logic.
pub trait LRScheduler {
    /// Updates the learning rate.
    fn step(&self);

    /// Returns an immutable reference to the last computed learning rate.
    fn get_last_lr(&self) -> f32;

    /// Returns an immutable reference to the current learning rate.
    fn get_current_lr(&self) -> f32;

    /// Returns an immutable reference to the current epoch.
    fn get_current_epoch(&self) -> usize;

    /// Sets the current epoch.
    fn set_current_epoch(&self, epoch: usize);

    /// Prints the update of the learning rate. It should be called after `.step()`.
    fn print_lr(&self) {
        println!(
            "epoch {}: learning rate adjusted to [{}]",
            self.get_current_epoch(),
            self.get_current_lr()
        );
    }
}

/// Prepares a learning rate scheduler to perform the next update step.
///
/// Sets `last_lr` as `current_lr` and increases `current_epoch`.
fn prepare_step(last_lr: &Cell<f32>, current_lr: &Cell<f32>, current_epoch: &Cell<usize>) {
    // Set current learning rate as last learning rate.
    last_lr.set(current_lr.get());
    // Set current epoch as last epoch.
    let last_epoch = current_epoch.get();
    // Increase current epoch.
    current_epoch.set(last_epoch + 1);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LambdaLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Sets the learning rate to the initial lr times a given function.
///
///```text
/// lrₜ = lr₀ * lr_fn(t)
///```
pub struct LambdaLR<'a, T: Optimizer, F: Fn(usize) -> f32> {
    optimizer: &'a T,
    lr_fn: F,
    current_epoch: Cell<usize>,
    current_lr: Cell<f32>,
    last_lr: Cell<f32>,
    initial_lr: Cell<f32>,
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
            current_epoch: Cell::new(0),
            current_lr: Cell::new(current_lr),
            last_lr: Cell::new(0.0),
            initial_lr: Cell::new(current_lr),
        }
    }
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> LRScheduler for LambdaLR<'a, T, F> {
    fn step(&self) {
        prepare_step(&self.last_lr, &self.current_lr, &self.current_epoch);
        self.current_lr
            .set(self.initial_lr.get() * (self.lr_fn)(self.current_epoch.get()));
        self.optimizer.set_lr(self.current_lr.get());
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr.get()
    }

    fn get_current_lr(&self) -> f32 {
        self.current_lr.get()
    }

    fn set_current_epoch(&self, epoch: usize) {
        self.current_epoch.replace(epoch);
    }

    fn get_current_epoch(&self) -> usize {
        self.current_epoch.get()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MultiplicativeLR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Multiplies the learning rate by the factor given in the specified function.
///
///```text
/// lrₜ = lrₜ₋₁ * lr_fn(t)
///```
pub struct MultiplicativeLR<'a, T: Optimizer, F: Fn(usize) -> f32> {
    optimizer: &'a T,
    lr_fn: F,
    current_epoch: Cell<usize>,
    current_lr: Cell<f32>,
    last_lr: Cell<f32>,
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
            current_epoch: Cell::new(0),
            current_lr: Cell::new(current_lr),
            last_lr: Cell::new(0.0),
        }
    }
}

impl<'a, T: Optimizer, F: Fn(usize) -> f32> LRScheduler for MultiplicativeLR<'a, T, F> {
    fn step(&self) {
        prepare_step(&self.last_lr, &self.current_lr, &self.current_epoch);
        self.current_lr
            .set(self.last_lr.get() * (self.lr_fn)(self.current_epoch.get()));
        self.optimizer.set_lr(self.current_lr.get());
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr.get()
    }

    fn get_current_lr(&self) -> f32 {
        self.current_lr.get()
    }

    fn set_current_epoch(&self, epoch: usize) {
        self.current_epoch.replace(epoch);
    }

    fn get_current_epoch(&self) -> usize {
        self.current_epoch.get()
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
    current_epoch: Cell<usize>,
    current_lr: Cell<f32>,
    last_lr: Cell<f32>,
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
            current_epoch: Cell::new(0),
            current_lr: Cell::new(current_lr),
            last_lr: Cell::new(0.0),
        }
    }
}

impl<'a, T: Optimizer> LRScheduler for StepLR<'a, T> {
    fn step(&self) {
        prepare_step(&self.last_lr, &self.current_lr, &self.current_epoch);
        if self.current_epoch.get().rem_euclid(self.step_size) == 0 {
            self.current_lr.set(self.last_lr.get() * self.gamma);
            self.optimizer.set_lr(self.current_lr.get());
        }
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr.get()
    }

    fn get_current_lr(&self) -> f32 {
        self.current_lr.get()
    }

    fn set_current_epoch(&self, epoch: usize) {
        self.current_epoch.replace(epoch);
    }

    fn get_current_epoch(&self) -> usize {
        self.current_epoch.get()
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
    optimizer: &'a T,
    gamma: f32,
    milestones: [usize; N],
    current_epoch: Cell<usize>,
    current_lr: Cell<f32>,
    last_lr: Cell<f32>,
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
    pub fn new(optimizer: &'a T, milestones: [usize; N], gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma,
            milestones,
            current_epoch: Cell::new(0),
            current_lr: Cell::new(current_lr),
            last_lr: Cell::new(0.0),
        }
    }
}

impl<'a, T: Optimizer, const N: usize> LRScheduler for MultiStepLR<'a, T, N> {
    fn step(&self) {
        prepare_step(&self.last_lr, &self.current_lr, &self.current_epoch);
        if self
            .milestones
            .iter()
            .any(|milestone| *milestone == self.current_epoch.get())
        {
            self.current_lr.set(self.last_lr.get() * self.gamma);
            self.optimizer.set_lr(self.current_lr.get());
        }
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr.get()
    }

    fn get_current_lr(&self) -> f32 {
        self.current_lr.get()
    }

    fn set_current_epoch(&self, epoch: usize) {
        self.current_epoch.replace(epoch);
    }

    fn get_current_epoch(&self) -> usize {
        self.current_epoch.get()
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
    current_epoch: Cell<usize>,
    current_lr: Cell<f32>,
    last_lr: Cell<f32>,
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
            current_epoch: Cell::new(0),
            current_lr: Cell::new(current_lr),
            last_lr: Cell::new(0.0),
        }
    }
}

impl<'a, T: Optimizer> LRScheduler for ExponentialLR<'a, T> {
    fn step(&self) {
        prepare_step(&self.last_lr, &self.current_lr, &self.current_epoch);
        self.current_lr.set(self.last_lr.get() * self.gamma);
        self.optimizer.set_lr(self.current_lr.get());
    }

    fn get_last_lr(&self) -> f32 {
        self.last_lr.get()
    }

    fn get_current_lr(&self) -> f32 {
        self.current_lr.get()
    }

    fn set_current_epoch(&self, epoch: usize) {
        self.current_epoch.replace(epoch);
    }

    fn get_current_epoch(&self) -> usize {
        self.current_epoch.get()
    }
}
#[cfg(test)]
mod test;
