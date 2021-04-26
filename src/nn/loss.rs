// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ losses module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
use super::{graph::OPERATIONS_COUNTER, Backward, Data, Forward, Gradient, Tensor, Var, VarDiff};
use ndarray::{Ix1, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

/// Specifies the reduction to apply to the **Loss** output.
/// * `mean` - the sum of the output will be divided by the batch size.
/// * `sum` - the output will be summed.
#[derive(Clone)]
pub enum Reduction {
    Sum,
    Mean,
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mean Square Erorr Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MSELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    was_computed: Cell<bool>,
}

impl<T, U> MSELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    pub fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for MSELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<T, U> Forward for MSELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        let (mut loss_data, input_data, target_data) = {
            (
                self.data.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };
        loss_data[0] = {
            let total_loss = Zip::from(&*input_data)
                .and(&*target_data)
                .fold(0.0, |loss, input, target| loss + (input - target).powi(2));
            match self.reduction {
                Reduction::Mean => total_loss / input_data.len() as f32,
                Reduction::Sum => total_loss,
            }
        };
        true
    }
}

pub struct MSELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    diff_input: Rc<T>,
    input: Rc<U>,
    target: Rc<V>,
    gradient: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U, V> MSELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    pub fn new(diff_input: Rc<T>, input: Rc<U>, target: Rc<V>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            input,
            target,
            gradient: RefCell::new(Tensor::zeros(1)),
            reduction,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U, V> Gradient for MSELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }
    fn was_overwritten(&self) {
        self.can_overwrite.set(true)
    }
}

impl<T, U, V> Backward for MSELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data<Dim = T::Dim> + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut operand_gradient, gradient, input_data, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };

        let zip = Zip::from(&mut *operand_gradient)
            .and_broadcast(&*gradient)
            .and(&*input_data)
            .and(&*target_data);

        match self.reduction {
            Reduction::Mean => {
                let n = input_data.len() as f32;
                zip.for_each(|op_grad, grad, input, target| {
                    *op_grad = (2.0 * (input - target) * input) * grad / n
                });
            }
            Reduction::Sum => zip.for_each(|op_grad, grad, input, target| {
                *op_grad = (2.0 * (input - target) * input) * grad
            }),
        }
        true
    }
}

/// Computes a criterion that measures the **mean squared error** *(squared L2 norm)*
/// between each element in the `input` **x**  and `target`**y**.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑ (xᵢ- ʏᵢ)²
///        n  i=1
/// ```
pub fn mse_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    mut target: Var<V>,
    reduction: Reduction,
) -> VarDiff<impl Data<Dim = Ix1> + Forward, impl Gradient<Dim = Ix1> + Backward>
where
    T: Data + Forward,
    U: Gradient<Dim = T::Dim> + Backward,
    V: Data<Dim = T::Dim> + Forward,
{
    input.forward_path.append(&mut target.forward_path);

    let (input_forward, input_backward) = (input.forward, input.backward);
    let target_forward = target.forward;

    let (id, forward, backward) = (
        unsafe { OPERATIONS_COUNTER.next() },
        Rc::new(MSELoss::new(
            input_forward.clone(),
            target_forward.clone(),
            reduction.clone(),
        )),
        Rc::new(MSELossBackward::new(
            input_backward,
            input_forward,
            target_forward,
            reduction,
        )),
    );
    input
        .forward_path
        .insert(id, forward.clone() as Rc<dyn Forward>);
    input
        .backward_path
        .insert(id, backward.clone() as Rc<dyn Backward>);

    VarDiff {
        id,
        forward,
        backward,
        forward_path: input.forward_path,
        backward_path: input.backward_path,
        parameters: input.parameters,
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mean Absolute Error Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct MAELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    was_computed: Cell<bool>,
}

impl<T, U> MAELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    pub fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for MAELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<T, U> Forward for MAELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        let (mut loss_data, input_data, target_data) = {
            (
                self.data.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };
        loss_data[0] = {
            let total_loss = Zip::from(&*input_data)
                .and(&*target_data)
                .fold(0.0, |loss, input, target| loss + (input - target).abs());
            match self.reduction {
                Reduction::Mean => total_loss / input_data.len() as f32,
                Reduction::Sum => total_loss,
            }
        };
        true
    }
}

pub struct MAELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    diff_input: Rc<T>,
    input: Rc<U>,
    target: Rc<V>,
    gradient: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U, V> MAELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    pub fn new(diff_input: Rc<T>, input: Rc<U>, target: Rc<V>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            input,
            target,
            gradient: RefCell::new(Tensor::zeros(1)),
            reduction,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U, V> Gradient for MAELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }
    fn was_overwritten(&self) {
        self.can_overwrite.set(true)
    }
}

impl<T, U, V> Backward for MAELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data<Dim = T::Dim> + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut operand_gradient, gradient, input_data, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };

        let zip = Zip::from(&mut *operand_gradient)
            .and_broadcast(&*gradient)
            .and(&*input_data)
            .and(&*target_data);

        match self.reduction {
            Reduction::Mean => {
                let n = input_data.len() as f32;
                zip.for_each(|op_grad, grad, input, target| {
                    let diff = input - target;
                    *op_grad = if diff != 0. {
                        diff.signum() * grad / n
                    } else {
                        0.
                    }
                });
            }
            Reduction::Sum => zip.for_each(|op_grad, grad, input, target| {
                let diff = input - target;
                *op_grad = if diff != 0. { diff.signum() * grad } else { 0. }
            }),
        }
        true
    }
}

/// Computes a criterion that measures the **mean absolute error** *(MAE)* between
/// each element in the `input` **x** and `target` **y**.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑ |xᵢ- ʏᵢ|
///        n  i=1
/// ```
pub fn mae_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    mut target: Var<V>,
    reduction: Reduction,
) -> VarDiff<impl Data<Dim = Ix1> + Forward, impl Gradient<Dim = Ix1> + Backward>
where
    T: Data + Forward,
    U: Gradient<Dim = T::Dim> + Backward,
    V: Data<Dim = T::Dim> + Forward,
{
    input.forward_path.append(&mut target.forward_path);

    let (input_forward, input_backward) = (input.forward, input.backward);
    let target_forward = target.forward;

    let (id, forward, backward) = (
        unsafe { OPERATIONS_COUNTER.next() },
        Rc::new(MAELoss::new(
            input_forward.clone(),
            target_forward.clone(),
            reduction.clone(),
        )),
        Rc::new(MAELossBackward::new(
            input_backward,
            input_forward,
            target_forward,
            reduction,
        )),
    );
    input
        .forward_path
        .insert(id, forward.clone() as Rc<dyn Forward>);
    input
        .backward_path
        .insert(id, backward.clone() as Rc<dyn Backward>);

    VarDiff {
        id,
        forward,
        backward,
        forward_path: input.forward_path,
        backward_path: input.backward_path,
        parameters: input.parameters,
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Binary Cross Entropy Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct BCELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    was_computed: Cell<bool>,
}

impl<T, U> BCELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    pub fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for BCELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<T, U> Forward for BCELoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        let (mut loss_data, input_data, target_data) = {
            (
                self.data.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };
        const MIN_LOG: f32 = -100.;
        loss_data[0] = {
            let total_loss =
                Zip::from(&*input_data)
                    .and(&*target_data)
                    .fold(0.0, |loss, input, target| {
                        loss + (target * input.ln().clamp(MIN_LOG, std::f32::MAX)
                            + (1. - target) * (1. - input).ln().clamp(MIN_LOG, std::f32::MAX))
                    });
            match self.reduction {
                Reduction::Mean => total_loss / input_data.len() as f32,
                Reduction::Sum => total_loss,
            }
        };
        true
    }
}

pub struct BCELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    diff_input: Rc<T>,
    input: Rc<U>,
    target: Rc<V>,
    gradient: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U, V> BCELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    pub fn new(diff_input: Rc<T>, input: Rc<U>, target: Rc<V>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            input,
            target,
            gradient: RefCell::new(Tensor::zeros(1)),
            reduction,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U, V> Gradient for BCELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }
    fn was_overwritten(&self) {
        self.can_overwrite.set(true)
    }
}

impl<T, U, V> Backward for BCELossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data<Dim = T::Dim> + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut operand_gradient, gradient, input_data, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };

        let zip = Zip::from(&mut *operand_gradient)
            .and_broadcast(&*gradient)
            .and(&*input_data)
            .and(&*target_data);

        match self.reduction {
            Reduction::Mean => {
                let n = input_data.len() as f32;
                zip.for_each(|op_grad, grad, input, target| {
                    *op_grad = (1. - 2. * target) / ((1. - input) * input).max(std::f32::EPSILON)
                        * grad
                        / n
                });
            }
            Reduction::Sum => zip.for_each(|op_grad, grad, input, target| {
                *op_grad = (1. - 2. * target) / ((1. - input) * input).max(std::f32::EPSILON) * grad
            }),
        }
        true
    }
}

/// Computes a criterion that measures the **binary cross entropy** between
/// the `target` **y** and `input` **x**.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑ ʏᵢ * ln(xᵢ) + (1 - ʏᵢ) * ln(1 - xᵢ)
///        n  i=1
/// ```
///
/// Note that the target **y** should be numbers between 0 and 1.
/// Notice that if a component of the `input` **x** is either 0 or 1,
/// one of the log terms would be mathematically undefined in the above loss equation.
/// Rust sets *ln(0) = -inf*, however, an infinite term in the loss equation is not desirable.
/// Our solution is that BCELoss clamps its log function outputs to be greater than or equal
/// to -100. This way, we can always have a finite loss value.
pub fn bce_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    mut target: Var<V>,
    reduction: Reduction,
) -> VarDiff<impl Data<Dim = Ix1> + Forward, impl Gradient<Dim = Ix1> + Backward>
where
    T: Data + Forward,
    U: Gradient<Dim = T::Dim> + Backward,
    V: Data<Dim = T::Dim> + Forward,
{
    input.forward_path.append(&mut target.forward_path);

    let (input_forward, input_backward) = (input.forward, input.backward);
    let target_forward = target.forward;

    let (id, forward, backward) = (
        unsafe { OPERATIONS_COUNTER.next() },
        Rc::new(BCELoss::new(
            input_forward.clone(),
            target_forward.clone(),
            reduction.clone(),
        )),
        Rc::new(BCELossBackward::new(
            input_backward,
            input_forward,
            target_forward,
            reduction,
        )),
    );
    input
        .forward_path
        .insert(id, forward.clone() as Rc<dyn Forward>);
    input
        .backward_path
        .insert(id, backward.clone() as Rc<dyn Backward>);

    VarDiff {
        id,
        forward,
        backward,
        forward_path: input.forward_path,
        backward_path: input.backward_path,
        parameters: input.parameters,
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Binary Cross Entropy With Logits Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct BCEWithLogitsLoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    was_computed: Cell<bool>,
}

impl<T, U> BCEWithLogitsLoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    pub fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for BCEWithLogitsLoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<T, U> Forward for BCEWithLogitsLoss<T, U>
where
    T: Data + Forward,
    U: Data<Dim = T::Dim> + Forward,
{
    fn forward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }
        self.was_computed.set(true);
        let (mut loss_data, input_data, target_data) = {
            (
                self.data.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };
        loss_data[0] = {
            let total_loss =
                Zip::from(&*input_data)
                    .and(&*target_data)
                    .fold(0.0, |loss, input, target| {
                        let max = (-input).max(0.);
                        loss + (1. - target) * input
                            + max
                            + ((-max).exp() + (-input - max).exp()).ln()
                    });
            match self.reduction {
                Reduction::Mean => total_loss / input_data.len() as f32,
                Reduction::Sum => total_loss,
            }
        };
        true
    }
}

pub struct BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    diff_input: Rc<T>,
    input: Rc<U>,
    target: Rc<V>,
    gradient: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    can_overwrite: Cell<bool>,
    was_computed: Cell<bool>,
}

impl<T, U, V> BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    pub fn new(diff_input: Rc<T>, input: Rc<U>, target: Rc<V>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            input,
            target,
            gradient: RefCell::new(Tensor::zeros(1)),
            reduction,
            can_overwrite: Cell::new(true),
            was_computed: Cell::new(false),
        }
    }
}

impl<T, U, V> Gradient for BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        self.gradient.borrow()
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.gradient.borrow_mut()
    }

    fn can_overwrite(&self) -> bool {
        self.can_overwrite.get()
    }
    fn was_overwritten(&self) {
        self.can_overwrite.set(true)
    }
}

impl<T, U, V> Backward for BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Backward,
    U: Data<Dim = T::Dim> + Forward,
    V: Data<Dim = U::Dim> + Forward,
{
    fn backward(&self) -> bool {
        if self.was_computed.get() {
            return false;
        }

        self.was_computed.set(true);
        let (mut operand_gradient, gradient, input_data, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };

        let zip = Zip::from(&mut *operand_gradient)
            .and_broadcast(&*gradient)
            .and(&*input_data)
            .and(&*target_data);

        match self.reduction {
            Reduction::Mean => {
                let n = input_data.len() as f32;
                zip.for_each(|op_grad, grad, input, target| {
                    let input_sigmoid = if *input >= 15.0 {
                        1.0
                    } else if *input <= -15.0 {
                        0.0
                    } else {
                        1.0 / (1.0 + (-input).exp())
                    };
                    *op_grad = (input_sigmoid - target) * grad / n
                });
            }
            Reduction::Sum => zip.for_each(|op_grad, grad, input, target| {
                let input_sigmoid = if *input >= 15.0 {
                    1.0
                } else if *input <= -15.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-input).exp())
                };
                *op_grad = (input_sigmoid - target) * grad
            }),
        }
        true
    }
}

/// Computes a criterion that measures the **binary cross entropy** between
/// the `target` **y** and `input` **x**.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑ ʏᵢ * ln(σ(xᵢ)) + (1 - ʏᵢ) * ln(1 - σ(xᵢ))
///        n  i=1
/// ```
/// This loss combines a sigmoid and a **binary cross entropy**.
/// This version is more numerically stable than using a plain sigmoid followed by a
/// binary cross entropy as, by combining the operations into one layer, we take
/// advantage of the **log-sum-exp** trick for numerical stability.
/// Note that the target **y** should be numbers between 0 and 1 and the
/// input **x** should be raw unnormalized scores.
pub fn bce_with_logits_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    mut target: Var<V>,
    reduction: Reduction,
) -> VarDiff<impl Data<Dim = Ix1> + Forward, impl Gradient<Dim = Ix1> + Backward>
where
    T: Data + Forward,
    U: Gradient<Dim = T::Dim> + Backward,
    V: Data<Dim = T::Dim> + Forward,
{
    input.forward_path.append(&mut target.forward_path);

    let (input_forward, input_backward) = (input.forward, input.backward);
    let target_forward = target.forward;

    let (id, forward, backward) = (
        unsafe { OPERATIONS_COUNTER.next() },
        Rc::new(BCEWithLogitsLoss::new(
            input_forward.clone(),
            target_forward.clone(),
            reduction.clone(),
        )),
        Rc::new(BCEWithLogitsLossBackward::new(
            input_backward,
            input_forward,
            target_forward,
            reduction,
        )),
    );
    input
        .forward_path
        .insert(id, forward.clone() as Rc<dyn Forward>);
    input
        .backward_path
        .insert(id, backward.clone() as Rc<dyn Backward>);

    VarDiff {
        id,
        forward,
        backward,
        forward_path: input.forward_path,
        backward_path: input.backward_path,
        parameters: input.parameters,
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cross Entropy Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// TODO
pub struct CrossEntropyLoss {
    reduction: Reduction,
}
pub struct CrossEntropyLossBackward {
    reduction: Reduction,
}