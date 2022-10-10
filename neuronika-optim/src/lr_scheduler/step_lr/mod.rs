use std::cell::Cell;

use crate::{Optimizer, OptimizerStatus};

use super::{prepare_step, LRScheduler};

/// Decays the learning rate by `gamma` every `step_size` epochs.
///
///```text
/// lrₜ = lrₜ₋₁ * gamma if t mod step_size == 0 else lrₜ₋₁
///```
pub struct StepLR<'a, T>
where
    T: OptimizerStatus,
{
    optimizer: &'a Optimizer<T>,
    gamma: Cell<f32>,
    step_size: Cell<usize>,
    current_epoch: Cell<usize>,
    current_lr: Cell<f32>,
    last_lr: Cell<f32>,
}

impl<'a, T> StepLR<'a, T>
where
    T: OptimizerStatus,
{
    /// Creates a new StepLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `step_size` - period of learning rate decay.
    ///
    /// * `gamma` - multiplicative factor for the learning rate decay.
    pub fn new(optimizer: &'a Optimizer<T>, step_size: usize, gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma: Cell::new(gamma),
            step_size: Cell::new(step_size),
            current_epoch: Cell::new(0),
            current_lr: Cell::new(current_lr),
            last_lr: Cell::new(0.0),
        }
    }

    /// Sets a new gamma for the scheduler.
    pub fn set_gamma(&self, gamma: f32) {
        self.gamma.set(gamma)
    }

    /// Sets a new step size for the scheduler.
    pub fn set_step_size(&self, step_size: usize) {
        self.step_size.set(step_size)
    }

    /// Decays the learning rate by gamma every `step_size` epochs.
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

impl<'a, T> LRScheduler for StepLR<'a, T>
where
    T: OptimizerStatus,
{
    fn step(&self) {
        prepare_step(&self.last_lr, &self.current_lr, &self.current_epoch);
        if self.current_epoch.get().rem_euclid(self.step_size.get()) == 0 {
            self.current_lr.set(self.last_lr.get() * self.gamma.get());
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

#[cfg(test)]
mod test;
