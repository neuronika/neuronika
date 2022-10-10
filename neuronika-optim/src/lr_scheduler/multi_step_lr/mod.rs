use std::cell::{Cell, RefCell};

use crate::{Optimizer, OptimizerStatus};

use super::{prepare_step, LRScheduler};

/// Decays the learning rate by gamma once the number of epoch reaches one of the specified
/// milestones.
///
///```text
/// lrₜ = lrₜ₋₁ * gamma if t is a milestone else lrₜ₋₁
///```
pub struct MultiStepLR<'a, T>
where
    T: OptimizerStatus,
{
    optimizer: &'a Optimizer<T>,
    gamma: f32,
    milestones: RefCell<Vec<usize>>,
    current_epoch: Cell<usize>,
    current_lr: Cell<f32>,
    last_lr: Cell<f32>,
}

impl<'a, T> MultiStepLR<'a, T>
where
    T: OptimizerStatus,
{
    /// Creates a new MultiStepLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `milestones` - list of epoch indices. Must be increasing.
    ///
    /// * `gamma` - multiplicative factor for the learning rate decay.
    pub fn new(optimizer: &'a Optimizer<T>, milestones: Vec<usize>, gamma: f32) -> Self {
        let current_lr = optimizer.get_lr();

        Self {
            optimizer,
            gamma,
            milestones: RefCell::new(milestones),
            current_epoch: Cell::new(0),
            current_lr: Cell::new(current_lr),
            last_lr: Cell::new(0.0),
        }
    }

    /// Sets new milestones for the scheduler.
    pub fn set_milestones(&self, milestones: Vec<usize>) {
        *self.milestones.borrow_mut() = milestones;
    }

    /// Decays the learning rate by gamma once the number of epoch reaches one of the milestones.
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

impl<'a, T> LRScheduler for MultiStepLR<'a, T>
where
    T: OptimizerStatus,
{
    fn step(&self) {
        prepare_step(&self.last_lr, &self.current_lr, &self.current_epoch);
        if self
            .milestones
            .borrow()
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

#[cfg(test)]
mod test;
