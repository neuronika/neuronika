use std::{cell::RefCell, rc::Rc};

/// Parameter optimization logic trait.
pub trait Optimize {
    /// Specifies the learning rule for the parameter.
    fn optimize(&mut self);

    /// Zeroes the gradient of this parameter.
    fn zero_grad(&mut self);
}

/// Parameter creation trait.
pub trait IntoParam<T>
where
    T: OptimizerStatus,
{
    type Param: 'static + Optimize;

    /// Specifies how an optimizer's parameter should be created given the optimizer status.
    fn into_param(self, status: Rc<T>) -> Self::Param;
}

/// Optimizer internal status trait.
pub trait OptimizerStatus {
    /// Gets the optimizer's learning rate.
    fn get_lr(&self) -> f32;

    /// Sets the optimizer's learning rate.
    fn set_lr(&self, lr: f32);
}

/// Generic optimization algorithm template.
pub struct Optimizer<T>
where
    T: OptimizerStatus,
{
    status: Rc<T>,
    params: RefCell<Vec<Box<dyn Optimize>>>,
}

impl<T> Optimizer<T>
where
    T: OptimizerStatus,
{
    /// Creates a new optimizer with the provided status.
    pub fn new(status: T) -> Self {
        let params = RefCell::default();
        let status = Rc::new(status);

        Self { status, params }
    }

    /// Returns the current learning rate.
    pub fn get_lr(&self) -> f32 {
        self.status.get_lr()
    }

    /// Sets a new value for the learning rate.
    pub fn set_lr(&self, lr: f32) {
        self.status.set_lr(lr)
    }

    /// Returns an immutable reference to the inner status.
    pub fn status(&self) -> &T {
        &self.status
    }

    /// Registers the variable to this optimizer. Following calls to `.step()` will apply the
    /// specified learning rule to the supplied variable.
    pub fn register<U>(&self, variable: U)
    where
        U: IntoParam<T>,
    {
        self.params
            .borrow_mut()
            .push(Box::new(variable.into_param(self.status.clone())))
    }

    /// Performs a single optimization step. It applies the provided learning rule to all the
    /// parameters registered in this optimizer.
    pub fn step(&self) {
        self.params
            .borrow_mut()
            .iter_mut()
            .for_each(|param| param.optimize());
    }

    /// Zeroes the gradients of all the parameters registered in this optimizer.
    pub fn zero_grad(&self) {
        self.params
            .borrow_mut()
            .iter_mut()
            .for_each(|param| param.zero_grad());
    }
}

impl<T> Default for Optimizer<T>
where
    T: OptimizerStatus + Default,
{
    fn default() -> Self {
        Self::new(T::default())
    }
}
