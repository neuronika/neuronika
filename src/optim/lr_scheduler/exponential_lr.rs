use crate::optim::{Optimizer, OptimizerStatus};

use super::{prepare_step, LRScheduler};

use std::cell::Cell;

/// Decays the learning rate by `gamma` every epoch.
///
///```text
/// lrₜ = lrₜ₋₁ * gamma
///```
pub struct ExponentialLR<'a, T>
where
    T: OptimizerStatus,
{
    optimizer: &'a Optimizer<T>,
    gamma: Cell<f32>,
    current_epoch: Cell<usize>,
    current_lr: Cell<f32>,
    last_lr: Cell<f32>,
}

impl<'a, T> ExponentialLR<'a, T>
where
    T: OptimizerStatus,
{
    /// Creates a new ExponentialLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `gamma` - multiplicative factor for the learning rate decay.
    pub fn new(optimizer: &'a Optimizer<T>, gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma: Cell::new(gamma),
            current_epoch: Cell::new(0),
            current_lr: Cell::new(current_lr),
            last_lr: Cell::new(0.0),
        }
    }

    /// Sets a new gamma for the scheduler.
    pub fn set_gamma(&self, gamma: f32) {
        self.gamma.set(gamma)
    }

    /// Decays the learning rate by gamma every epoch.
    pub fn step(&self) {
        LRScheduler::step(self);
    }

    /// Returns the last learning rate value computed by this learning rate scheduler.
    pub fn get_last_lr(&self) -> f32 {
        LRScheduler::get_last_lr(self)
    }

    /// Returns the current learning rate value computed by this learning rate scheduler.
    pub fn get_current_lr(&self) -> f32 {
        LRScheduler::get_current_lr(self)
    }

    /// Sets the current epoch for this learning rate scheduler.
    pub fn set_current_epoch(&self, epoch: usize) {
        LRScheduler::set_current_epoch(self, epoch);
    }

    /// Returns the current epoch for this learning rate scheduler.
    pub fn get_current_epoch(&self) -> usize {
        LRScheduler::get_current_epoch(self)
    }

    /// Prints the learning rate update together with the epoch.
    pub fn print_lr(&self) {
        LRScheduler::print_lr(self);
    }
}

impl<'a, T> LRScheduler for ExponentialLR<'a, T>
where
    T: OptimizerStatus,
{
    fn step(&self) {
        prepare_step(&self.last_lr, &self.current_lr, &self.current_epoch);
        self.current_lr.set(self.last_lr.get() * self.gamma.get());
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
