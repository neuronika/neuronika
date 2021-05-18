// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ losses module ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
use super::{
    variable::{expect_tensor, expect_tensor_mut, node::Overwrite, OPERATIONS_COUNTER},
    Backward, Data, Forward, Gradient, Tensor, Var, VarDiff,
};
use ndarray::{Axis, Dimension, IntoDimension, Ix1, Zip};
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
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct MSELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    computed: Cell<bool>,
}

impl<T, U> MSELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    pub fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for MSELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<T, U> Forward for MSELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
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
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct MSELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = U::Dim>,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    overwrite: Cell<bool>,
    diff_input: Rc<T>,
    input: Rc<U>,
    target: Rc<V>,
    reduction: Reduction,
}

impl<T, U, V> MSELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = U::Dim>,
{
    pub fn new(diff_input: Rc<T>, input: Rc<U>, target: Rc<V>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            input,
            target,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            reduction,
            overwrite: Cell::new(false),
        }
    }
}

impl<T, U, V> Gradient for MSELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = U::Dim>,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U, V> Overwrite for MSELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U, V> Backward for MSELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let (mut operand_gradient, gradient, input_data, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient(),
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
                    *op_grad = (2.0 * (input - target)) * grad / n
                });
            }
            Reduction::Sum => zip.for_each(|op_grad, grad, input, target| {
                *op_grad = (2.0 * (input - target)) * grad
            }),
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
    }
}

/// Computes a criterion that measures the **mean squared error** *(squared L2 norm)*
/// between each element in the `input` **x**  and `target` **y**.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑ (xᵢ- ʏᵢ)²
///        n  i=1
/// ```
pub fn mse_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<impl Data<Dim = Ix1>, impl Gradient<Dim = Ix1> + Overwrite>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data<Dim = T::Dim>,
{
    input.var.past.merge(target.past);

    let (id, forward, backward) = (
        unsafe { OPERATIONS_COUNTER.next() },
        Rc::new(MSELoss::new(
            input.var.node.clone(),
            target.node.clone(),
            reduction.clone(),
        )),
        Rc::new(MSELossBackward::new(
            input.node,
            input.var.node,
            target.node,
            reduction,
        )),
    );
    input
        .var
        .past
        .append_forward(id, forward.clone() as Rc<dyn Forward>);
    input
        .past
        .append_backward(id, backward.clone() as Rc<dyn Backward>);

    VarDiff {
        var: Var {
            node: forward,
            past: input.var.past,
        },

        node: backward,
        past: input.past,
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mean Absolute Error Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct MAELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    computed: Cell<bool>,
}

impl<T, U> MAELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    pub fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for MAELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<T, U> Forward for MAELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
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
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct MAELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    overwrite: Cell<bool>,
    diff_input: Rc<T>,
    input: Rc<U>,
    target: Rc<V>,
    reduction: Reduction,
}

impl<T, U, V> MAELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    pub fn new(diff_input: Rc<T>, input: Rc<U>, target: Rc<V>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            input,
            target,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            reduction,
            overwrite: Cell::new(false),
        }
    }
}

impl<T, U, V> Gradient for MAELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U, V> Overwrite for MAELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U, V> Backward for MAELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let (mut operand_gradient, gradient, input_data, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient(),
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
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
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
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<impl Data<Dim = Ix1>, impl Gradient<Dim = Ix1> + Overwrite>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data<Dim = T::Dim>,
{
    input.var.past.merge(target.past);

    let (id, forward, backward) = (
        unsafe { OPERATIONS_COUNTER.next() },
        Rc::new(MAELoss::new(
            input.var.node.clone(),
            target.node.clone(),
            reduction.clone(),
        )),
        Rc::new(MAELossBackward::new(
            input.node,
            input.var.node,
            target.node,
            reduction,
        )),
    );
    input
        .var
        .past
        .append_forward(id, forward.clone() as Rc<dyn Forward>);
    input
        .past
        .append_backward(id, backward.clone() as Rc<dyn Backward>);

    VarDiff {
        var: Var {
            node: forward,
            past: input.var.past,
        },

        node: backward,
        past: input.past,
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Binary Cross Entropy Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct BCELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    computed: Cell<bool>,
}

impl<T, U> BCELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    pub fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for BCELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<T, U> Forward for BCELoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
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
                Reduction::Mean => -total_loss / input_data.len() as f32,
                Reduction::Sum => -total_loss,
            }
        };
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct BCELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    overwrite: Cell<bool>,
    diff_input: Rc<T>,
    input: Rc<U>,
    target: Rc<V>,
    reduction: Reduction,
}

impl<T, U, V> BCELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    pub fn new(diff_input: Rc<T>, input: Rc<U>, target: Rc<V>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            input,
            target,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            reduction,
            overwrite: Cell::new(false),
        }
    }
}

impl<T, U, V> Gradient for BCELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U, V> Overwrite for BCELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U, V> Backward for BCELossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = U::Dim>,
{
    fn backward(&self) {
        let (mut operand_gradient, gradient, input_data, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient(),
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
                    *op_grad =
                        (input - target) / ((1. - input) * input).max(std::f32::EPSILON) * grad / n
                });
            }
            Reduction::Sum => zip.for_each(|op_grad, grad, input, target| {
                *op_grad = (input - target) / ((1. - input) * input).max(std::f32::EPSILON) * grad
            }),
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
    }
}

/// Computes a criterion that measures the **binary cross entropy** between
/// the `target` **y** and `input` **x**.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑ - [ʏᵢ * ln(xᵢ) + (1 - ʏᵢ) * ln(1 - xᵢ)]
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
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<impl Data<Dim = Ix1>, impl Gradient<Dim = Ix1> + Overwrite>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data<Dim = T::Dim>,
{
    input.var.past.merge(target.past);

    let (id, forward, backward) = (
        unsafe { OPERATIONS_COUNTER.next() },
        Rc::new(BCELoss::new(
            input.var.node.clone(),
            target.node.clone(),
            reduction.clone(),
        )),
        Rc::new(BCELossBackward::new(
            input.node,
            input.var.node,
            target.node,
            reduction,
        )),
    );
    input
        .var
        .past
        .append_forward(id, forward.clone() as Rc<dyn Forward>);
    input
        .past
        .append_backward(id, backward.clone() as Rc<dyn Backward>);

    VarDiff {
        var: Var {
            node: forward,
            past: input.var.past,
        },

        node: backward,
        past: input.past,
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Binary Cross Entropy With Logits Loss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct BCEWithLogitsLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    state: Cell<bool>,
}

impl<T, U> BCEWithLogitsLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    pub fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            state: Cell::new(false),
        }
    }
}

impl<T, U> Data for BCEWithLogitsLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<T, U> Forward for BCEWithLogitsLoss<T, U>
where
    T: Data,
    U: Data<Dim = T::Dim>,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.state.set(true);
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
    }

    fn was_computed(&self) -> bool {
        self.state.get()
    }

    fn reset_computation(&self) {
        self.state.set(false);
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<Ix1>>>,
    overwrite: Cell<bool>,
    diff_input: Rc<T>,
    input: Rc<U>,
    target: Rc<V>,
    reduction: Reduction,
}

impl<T, U, V> BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    pub fn new(diff_input: Rc<T>, input: Rc<U>, target: Rc<V>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            input,
            target,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            reduction,
            overwrite: Cell::new(false),
        }
    }
}

impl<T, U, V> Gradient for BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U, V> Overwrite for BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U, V> Backward for BCEWithLogitsLossBackward<T, U, V>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
    V: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let (mut operand_gradient, gradient, input_data, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient(),
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
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
    }
}

/// Computes a criterion that measures the **binary cross entropy** between
/// the `target` **y** and `input` **x**.
///
/// ```text
///        1   n
/// Lᴏss = ―   ∑  - [ʏᵢ * ln(σ(xᵢ)) + (1 - ʏᵢ) * ln(1 - σ(xᵢ))]
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
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<impl Data<Dim = Ix1>, impl Gradient<Dim = Ix1> + Overwrite>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data<Dim = T::Dim>,
{
    input.var.past.merge(target.past);

    let (id, forward, backward) = (
        unsafe { OPERATIONS_COUNTER.next() },
        Rc::new(BCEWithLogitsLoss::new(
            input.var.node.clone(),
            target.node.clone(),
            reduction.clone(),
        )),
        Rc::new(BCEWithLogitsLossBackward::new(
            input.node,
            input.var.node,
            target.node,
            reduction,
        )),
    );
    input
        .var
        .past
        .append_forward(id, forward.clone() as Rc<dyn Forward>);
    input
        .past
        .append_backward(id, backward.clone() as Rc<dyn Backward>);

    VarDiff {
        var: Var {
            node: forward,
            past: input.var.past,
        },

        node: backward,
        past: input.past,
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ NLLLoss ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[allow(clippy::clippy::upper_case_acronyms)]
pub struct NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    input: Rc<T>,
    target: Rc<U>,
    data: RefCell<Tensor<Ix1>>,
    reduction: Reduction,
    computed: Cell<bool>,
}

impl<T, U> NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    pub fn new(input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            input,
            target,
            data: RefCell::new(Tensor::zeros(1)),
            reduction,
            computed: Cell::new(false),
        }
    }
}

impl<T, U> Data for NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    type Dim = Ix1;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<T, U> Forward for NLLLoss<T, U>
where
    T: Data<Dim = <U::Dim as Dimension>::Larger>,
    T::Dim: Copy,
    U: Data,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }
        self.computed.set(true);
        let (mut loss_data, input_data, target_data) = {
            (
                self.data.borrow_mut(),
                self.input.data(),
                self.target.data(),
            )
        };
        loss_data[0] = {
            let total_loss = Zip::indexed(&*input_data)
                .and_broadcast(&target_data.view().insert_axis(Axis(1)))
                .fold(0.0, |loss, idx, log, target| {
                    if idx.into_dimension().last_elem() == *target as usize {
                        loss + log
                    } else {
                        loss + 0.
                    }
                });
            match self.reduction {
                Reduction::Mean => -total_loss / input_data.len() as f32,
                Reduction::Sum => -total_loss,
            }
        };
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

#[allow(clippy::clippy::upper_case_acronyms)]
pub struct NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    diff_input: Rc<T>,
    target: Rc<U>,
    gradient: RefCell<Option<Tensor<Ix1>>>,
    reduction: Reduction,
    overwrite: Cell<bool>,
}

impl<T, U> NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    pub fn new(diff_input: Rc<T>, target: Rc<U>, reduction: Reduction) -> Self {
        Self {
            diff_input,
            target,
            gradient: RefCell::new(Some(Tensor::zeros(1))),
            reduction,
            overwrite: Cell::new(false),
        }
    }
}

impl<T, U> Gradient for NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    type Dim = Ix1;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for NLLLossBackward<T, U>
where
    T: Gradient<Dim = <U::Dim as Dimension>::Larger> + Overwrite,
    U: Data,
    T::Dim: Copy,
{
    fn backward(&self) {
        let (mut operand_gradient, gradient, target_data) = {
            (
                self.diff_input.gradient_mut(),
                self.gradient(),
                self.target.data(),
            )
        };
        let zip = Zip::indexed(&mut *operand_gradient)
            .and_broadcast(&*gradient)
            .and_broadcast(target_data.view().insert_axis(Axis(1)));

        match self.reduction {
            Reduction::Mean => {
                let n = target_data.len() as f32;
                zip.for_each(|idx, op_grad, grad, target| {
                    if idx.into_dimension().last_elem() == *target as usize {
                        *op_grad = grad * -1. / n
                    } else {
                        *op_grad = 0.;
                    }
                });
            }
            Reduction::Sum => zip.for_each(|idx, op_grad, grad, target| {
                if idx.into_dimension().last_elem() == *target as usize {
                    *op_grad = grad * -1.
                } else {
                    *op_grad = 0.
                }
            }),
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(1));
    }
}

/// Computes a criterion that measures the **negative log likelihood** between
/// the `target` **y** and `input` **x**.
///
/// ```text
///         1   n
/// Lᴏss =  ―   ∑  - xₙ,ᵧₙ
///         n  i=1
/// ```
///
/// The `input` **x** given is expected to contain **log-probabilities** for each class,
/// this is typically achieved by using `log_softmax`. `input` has to be a of size either
/// **(minibatch, C)** or **(minibatch, C, d1, d2, ..., dk)** with k >= 1 for the **K**-dimensional
/// case. The target that this loss expects should be a class index in the range **[0, C)** where
/// **C** = number of classes.
///
/// As mentioned before, this loss can also be used for higher dimensional inputs, such as 2D
/// images, by providing an input of size **(minibatch, C, d1, d2, ..., dk)** with k >= 1 where
/// **k** is the number of dimensions. In the case of images, it computes **NLL loss** *per-pixel*.
///
/// In the **K**-dimensional case this loss expects a target of shape
/// **(minibatch, d1, d2, ..., dk)**.
pub fn nll_loss<T, U, V>(
    mut input: VarDiff<T, U>,
    target: Var<V>,
    reduction: Reduction,
) -> VarDiff<impl Data<Dim = Ix1>, impl Gradient<Dim = Ix1> + Overwrite>
where
    T: Data<Dim = <V::Dim as Dimension>::Larger>,
    U: Gradient<Dim = T::Dim> + Overwrite,
    V: Data,
    T::Dim: Copy,
{
    input.var.past.merge(target.past);

    let (id, forward, backward) = (
        unsafe { OPERATIONS_COUNTER.next() },
        Rc::new(NLLLoss::new(
            input.var.node.clone(),
            target.node.clone(),
            reduction.clone(),
        )),
        Rc::new(NLLLossBackward::new(input.node, target.node, reduction)),
    );
    input
        .var
        .past
        .append_forward(id, forward.clone() as Rc<dyn Forward>);
    input
        .past
        .append_backward(id, backward.clone() as Rc<dyn Backward>);

    VarDiff {
        var: Var {
            node: forward,
            past: input.var.past,
        },

        node: backward,
        past: input.past,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{variable::node::Differentiable, Input, InputBackward};
    use ndarray::{Dimension, StrideShape};

    const F16_EPSILON: f32 = 9.77e-04;

    fn assert_almost_equals<D: Dimension>(our: &Tensor<D>, their: &Tensor<D>) {
        assert!(
            Zip::from(our).and(their).all(|l, r| {
                (*l == 0. && *r == 0.)
                    || (!l.is_finite() && !r.is_finite())
                    || ((1. - r / l).abs() <= F16_EPSILON)
            }),
            "\nLeft:\n{}\nRight:\n{}",
            our,
            their
        );
    }

    fn new_input<D, Sh>(shape: Sh, elems: Vec<f32>) -> Rc<Input<D>>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Input::new(new_tensor(shape, elems)).node
    }

    fn new_backward_input<D, Sh>(shape: Sh, elems: Vec<f32>) -> Rc<InputBackward<D>>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Rc::new(Input::new(new_tensor(shape, elems)).node.differentiable())
    }

    fn new_tensor<D, Sh>(shape: Sh, elems: Vec<f32>) -> Tensor<D>
    where
        D: Dimension + 'static,
        Sh: Into<StrideShape<D>>,
    {
        Tensor::from_shape_vec(shape, elems).unwrap()
    }

    #[test]
    fn mae_loss_mean() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MAELoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![9.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            MAELossBackward::new(input_diff.clone(), input, target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor((3, 3), vec![0.1111; 9]),
        );
    }

    #[test]
    fn mae_loss_sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MAELoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![81.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward = MAELossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![1.; 9]));
    }

    #[test]
    fn mse_loss_mean() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MSELoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![81.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            MSELossBackward::new(input_diff.clone(), input, target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![2.; 9]));
    }

    #[test]
    fn mse_loss_sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = MSELoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![729.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward = MSELossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![18.; 9]));
    }

    #[test]
    fn bce_loss_mean() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
        let loss = BCELoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![22.9244]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            BCELossBackward::new(input_diff.clone(), input, target, Reduction::Mean);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -1.1111e+00,
                    -1.2346e-01,
                    1.1111e+00,
                    0.0000e+00,
                    0.0000e+00,
                    -9.32067e+05,
                    5.5556e-01,
                    0.0000e+00,
                    -9.32067e+05,
                ],
            ),
        );
    }

    #[test]
    fn bce_loss_sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
        let loss = BCELoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![206.3199]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward = BCELossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();

        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -1.0000e+01,
                    -1.1111e+00,
                    1.0000e+01,
                    0.0000e+00,
                    0.0000e+00,
                    -8.3886e+6,
                    5.0000e+00,
                    0.0000e+00,
                    -8.3886e+6,
                ],
            ),
        );
    }

    #[test]
    fn bce_with_logits_loss_mean() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Mean);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![8.]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            BCEWithLogitsLossBackward::new(input_diff.clone(), input, target, Reduction::Mean);
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();
        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -5.0465e-06,
                    -1.8544e-06,
                    1.1111e-01,
                    1.1111e-01,
                    1.1111e-01,
                    0.0000e+00,
                    1.1111e-01,
                    1.1111e-01,
                    0.0000e+00,
                ],
            ),
        );
    }

    #[test]
    fn bce_with_logits_loss_sum() {
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
        let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
        let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Sum);

        loss.forward();
        assert_almost_equals(&*loss.data(), &new_tensor(1, vec![72.0001]));

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        let input_diff = new_backward_input((3, 3), vec![0.; 9]);
        let loss_backward =
            BCEWithLogitsLossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        loss_backward.backward();

        assert_almost_equals(
            &*input_diff.gradient(),
            &new_tensor(
                (3, 3),
                vec![
                    -4.5419e-05,
                    -1.6689e-05,
                    9.9999e-01,
                    1.0000e+00,
                    1.0000e+00,
                    0.0000e+00,
                    1.0000e+00,
                    1.0000e+00,
                    0.0000e+00,
                ],
            ),
        );
    }
}
