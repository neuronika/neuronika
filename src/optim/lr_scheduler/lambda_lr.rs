use crate::optim::{Optimizer, OptimizerStatus};

use super::LRScheduler;

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
