use crate::{Optimizer, OptimizerStatus};

use super::{prepare_step, LRScheduler};

use std::cell::Cell;

/// Sets the learning rate to the initial lr times a given function.
///
///```text
/// lrₜ = lr₀ * lr_fn(t)
///```
pub struct LambdaLR<'a, T, F>
where
    T: OptimizerStatus,
    F: Fn(usize) -> f32,
{
    optimizer: &'a Optimizer<T>,
    lr_fn: F,
    current_epoch: Cell<usize>,
    current_lr: Cell<f32>,
    last_lr: Cell<f32>,
    initial_lr: Cell<f32>,
}

impl<'a, T, F> LambdaLR<'a, T, F>
where
    T: OptimizerStatus,
    F: Fn(usize) -> f32,
{
    /// Creates a new LambdaLR scheduler.
    ///
    /// # Arguments
    ///
    /// * `optimizer` - wrapped optimizer.
    ///
    /// * `lr_fn` - function which computes a multiplicative factor given an `usize` parameter
    /// epoch.
    pub fn new(optimizer: &'a Optimizer<T>, lr_fn: F) -> Self {
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

    /// Sets the learning rate to the initial learning times a given function.
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

impl<'a, T, F> LRScheduler for LambdaLR<'a, T, F>
where
    T: OptimizerStatus,
    F: Fn(usize) -> f32,
{
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

#[cfg(test)]
mod test;
