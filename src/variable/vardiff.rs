use super::{
    cobroadcasted_zeros, history::History, AdditionBackward, AdditionBackwardLeft,
    AdditionBackwardRight, Backward, Cat, ChunkBackward, ConcatenateBackward,
    ConcatenateBackwardLeft, ConcatenateBackwardRight, DivisionBackward, DivisionBackwardLeft,
    DivisionBackwardRight, DotDim, DropoutBackward, ExpBackward, LeakyReLUBackward,
    LogSoftmaxBackward, LognBackward, MatMatMul, MatMatMulT, MatVecMul, MatrixMatrixMulBackward,
    MatrixMatrixMulBackwardLeft, MatrixMatrixMulBackwardRight, MatrixMatrixMulTBackward,
    MatrixMatrixMulTBackwardLeft, MatrixMatrixMulTBackwardRight, MatrixVectorMulBackward,
    MatrixVectorMulBackwardLeft, MatrixVectorMulBackwardRight, MeanBackward,
    MultiConcatenateBackward, MultiStackBackward, MultiplicationBackward,
    MultiplicationBackwardLeft, MultiplicationBackwardRight, NegationBackward, OptionalTensor,
    PowerBackward, ReLUBackward, SigmoidBackward, SoftPlusBackward, SoftmaxBackward, SqrtBackward,
    Stack, StackBackward, StackBackwardLeft, StackBackwardRight, SubtractionBackward,
    SubtractionBackwardLeft, SubtractionBackwardRight, SumBackward, SwitchTensor, TanHBackward,
    Tensor, TransposeBackward, UnsqueezeBackward, Var, VecMatMul, VecVecMul,
    VectorMatrixMulBackward, VectorMatrixMulBackwardLeft, VectorMatrixMulBackwardRight,
    VectorVectorMulBackward, VectorVectorMulBackwardUnary,
};

use ndarray::{
    arr0, concatenate, stack, Axis, DimMax, Dimension, IntoDimension, Ix0, Ix1, Ix2, RemoveAxis,
};
#[cfg(feature = "serialize")]
use serde::{
    de::{Deserialize, Deserializer},
    ser::{Serialize, Serializer},
};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

/// A differentiable variable.
///
/// Differentiable variables can be created in the **two** following ways described hereafter:
///
/// 1. By calling [`.requires_grad()`](Var::requires_grad()) on a non-differentiable leaf.
///
/// 2. By performing any binary operation between a [`Var`] and a `VarDiff`. Differentiability
/// is thus a *contagious* property, that is, if during a computation a `VarDiff` is used, the
/// result of the computation itself, and that of any other subsequent computations performed on
/// it, will also be differentiable. As an obvious consequence, the results of operations
/// performed on two *VarDiff* will also be *VarDiff*.
#[derive(Clone)]
pub struct VarDiff<D>
where
    D: 'static + Dimension,
{
    pub(crate) var: Var<D>,
    pub(crate) gradient: Rc<OptionalTensor<D>>,
    pub(crate) history: History<Rc<dyn Backward>>,
    pub(crate) switchables: History<Rc<dyn SwitchTensor>>,
}

impl<D> VarDiff<D>
where
    D: 'static + Dimension,
{
    pub(crate) fn leaf(var: Var<D>, array: Tensor<D>) -> Self {
        Self {
            var,
            gradient: Rc::new(OptionalTensor::from_ndarray(array)),
            history: History::default(),
            switchables: History::default(),
        }
    }

    pub(crate) fn new<T>(
        var: Var<D>,
        gradient: Rc<OptionalTensor<D>>,
        op: T,
        mut history: History<Rc<dyn Backward>>,
        mut switchables: History<Rc<dyn SwitchTensor>>,
    ) -> VarDiff<D>
    where
        T: 'static + Backward,
    {
        let op = Rc::new(op);
        let id = Rc::as_ptr(&op) as *const () as usize;
        history.insert(id, op);

        let id = Rc::as_ptr(&gradient) as *const () as usize;
        switchables.insert(id, gradient.clone());

        Self {
            var,
            gradient,
            history,
            switchables,
        }
    }
}

impl<D> VarDiff<D>
where
    D: 'static + Dimension,
{
    /// Returns an immutable reference to the data inside `self`.
    ///
    /// At the differentiable variable's creation the data is filled with zeros. You can populate it
    /// with a call to [`.forward()`](VarDiff::forward()).
    pub fn data(&self) -> Ref<Tensor<D>> {
        self.var.data()
    }

    /// Returns a mutable reference to the data inside `self`.
    ///
    /// At the differentiable variable's creation the data is filled with zeros. You can populate it
    /// with a call to [`.forward()`](VarDiff::forward()).
    pub fn data_mut(&self) -> RefMut<Tensor<D>> {
        self.var.data_mut()
    }

    /// Returns an immutable reference to the gradient inside `self`.
    ///
    /// At the differentiable variable's creation the gradient is filled with zeros. You can
    /// populate it with a call to [`.backward()`](VarDiff::backward()).
    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.gradient.content()
    }

    /// Returns a mutable reference to the gradient inside `self`.
    ///
    /// At the differentiable variable's creation the gradient is filled with zeros. You can
    /// populate it with a call to [`.backward()`](VarDiff::backward()).
    pub fn grad_mut(&self) -> RefMut<Tensor<D>> {
        self.gradient.content_mut()
    }

    /// Propagates the computations forwards and populates all the variables and differentiable
    /// variables from the leaves of the graph to `self`.   
    pub fn forward(&self) {
        self.var.forward();

        // Prepares the buffer for the backward pass.
        let mut buffer = self.history.buffer_mut();

        // If the buffer is empty populate it.
        if buffer.is_empty() {
            *buffer = self.history.to_vec()
        }
    }

    /// Back-propagates through the computational graph and populates the gradients of the
    /// differentiable leaves that are ancestors of `self`. Before back-propagating the gradient
    /// of `self` is seeded with `seed`, thus, the leaves' gradients will be scaled accordingly.
    ///
    /// **Do note** that this method should be called after `.forward()`.
    ///
    /// The graph is differentiated through the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).
    pub fn backward(&self, seed: f32) {
        debug_assert_eq!(
            self.var.history.len(),
            self.var.history.buffer_len(),
            "Perhaps you forgot to call .forward()?"
        );

        // Seed the gradient.
        self.grad_mut().fill(seed);

        // Compute gradients.
        self.history
            .buffer()
            .iter()
            .rev()
            .for_each(|op| op.backward());

        // Reset forward path.
        let buffer = self.var.history.buffer();
        let res = buffer.binary_search_by(|op| {
            if op.was_computed() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        let pos = match res {
            Ok(index) => index,
            Err(index) => index,
        };

        buffer
            .iter()
            .skip(pos)
            .for_each(|op| op.reset_computation());
    }

    /// Disables gradient computation and de-allocates the gradient for `self` and all of its
    /// ancestors.
    pub fn no_grad(&self) {
        let mut buffer = self.switchables.buffer_mut();
        if buffer.is_empty() {
            *buffer = self.switchables.to_vec();
        }

        buffer.iter().for_each(|op| op.deallocate());
    }

    /// Re-enables gradient computation and re-allocates the gradient for `self` and all of its
    /// ancestors.
    pub fn with_grad(&self) {
        let mut buffer = self.switchables.buffer_mut();
        if buffer.is_empty() {
            *buffer = self.switchables.to_vec();
        }

        buffer.iter().for_each(|op| op.allocate());
    }
}

impl VarDiff<Ix1> {
    /// Performs a vector-matrix multiplication between the vector variable `self` and the matrix
    /// variable `rhs`.
    ///
    /// If `self` is *n* and `rhs` is *(n, m)* the output will be *m*.
    pub fn vm<Rhs>(self, rhs: Rhs) -> <Self as VecMatMul<Rhs>>::Output
    where
        Self: VecMatMul<Rhs>,
    {
        VecMatMul::vm(self, rhs)
    }

    /// Vector-vector product, a.k.a. *scalar product* or *inner product*.
    ///
    /// Performs the scalar product between the two vector variables `self` and `rhs`.
    pub fn vv<Rhs>(self, rhs: Rhs) -> <Self as VecVecMul<Rhs>>::Output
    where
        Self: VecVecMul<Rhs>,
    {
        VecVecMul::vv(self, rhs)
    }
}

impl VarDiff<Ix2> {
    /// Performs a matrix multiplication between the matrix variables `self` and `rhs`. If `self`
    /// is *(n, m)* and `rhs` is *(m, o)* the output will be *(n, o)*.
    pub fn mm<Rhs>(self, rhs: Rhs) -> <Self as MatMatMul<Rhs>>::Output
    where
        Self: MatMatMul<Rhs>,
    {
        MatMatMul::mm(self, rhs)
    }

    /// Performs a matrix multiplication between the matrix variables `self` and `rhs`.
    /// This is a **fused operation** as `rhs` is implicitly transposed. Fusing the two operations
    /// it's marginally faster than computing the matrix multiplication and the transposition
    /// separately.
    ///
    /// If `self` is  *(n, m)* and `rhs` is *(o, m)* the output will be *(n, o)*.
    pub fn mm_t<Rhs>(self, rhs: Rhs) -> <Self as MatMatMulT<Rhs>>::Output
    where
        Self: MatMatMulT<Rhs>,
    {
        MatMatMulT::mm_t(self, rhs)
    }

    /// Performs a matrix-vector multiplication between the matrix variable `self` and the vector
    /// variable `rhs`.
    ///
    /// If `self` is *(n, m)* and `rhs` is *m* the output will be *n*.
    pub fn mv<Rhs>(self, rhs: Rhs) -> <Self as MatVecMul<Rhs>>::Output
    where
        Self: MatVecMul<Rhs>,
    {
        MatVecMul::mv(self, rhs)
    }
}

impl<D> VarDiff<D>
where
    D: 'static + Dimension,
{
    /// Returns the sum of all elements in `self`.
    pub fn sum(self) -> VarDiff<Ix0> {
        let gradient = Rc::new(OptionalTensor::from_ndarray(arr0(0.)));
        let op = SumBackward::new(self.gradient, gradient.clone());

        VarDiff::new(self.var.sum(), gradient, op, self.history, self.switchables)
    }

    /// Returns the mean of all elements in `self`.
    pub fn mean(self) -> VarDiff<Ix0> {
        let gradient = Rc::new(OptionalTensor::from_ndarray(arr0(0.)));
        let op = MeanBackward::new(self.gradient, gradient.clone());

        VarDiff::new(
            self.var.mean(),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }

    /// Takes the power of each element in `self` with exponent `exp` and returns a differentiable
    /// variable with the result.
    ///
    /// # Arguments
    ///
    /// `exp` - exponent.
    pub fn pow(self, exp: i32) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let op = PowerBackward::new(self.gradient, self.var.data.clone(), gradient.clone(), exp);

        VarDiff::new(
            self.var.pow(exp),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }

    /// Takes the square root element-wise and returns a differentiable variable with the result.
    pub fn sqrt(self) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let var = self.var.sqrt();
        let op = SqrtBackward::new(self.gradient, var.data.clone(), gradient.clone());

        VarDiff::new(var, gradient, op, self.history, self.switchables)
    }

    /// Applies the *rectified linear unit* element-wise and and returns a differentiable
    /// variable with the result.
    ///
    /// *ReLU(x) = max(0, x)*
    pub fn relu(self) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let op = ReLUBackward::new(self.gradient, self.var.data.clone(), gradient.clone());

        VarDiff::new(
            self.var.relu(),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }

    /// Applies the *leaky rectified linear unit* element-wise and returns a differentiable
    /// variable with the result.
    ///
    /// *LeakyReLU(x) = max(0, x) + 0.01 * min(0, x)*
    pub fn leaky_relu(self) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let op = LeakyReLUBackward::new(self.gradient, self.var.data.clone(), gradient.clone());

        VarDiff::new(
            self.var.leaky_relu(),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }

    /// Applies the *softplus* element-wise and returns a differentiable variable with the result.
    ///
    /// *Softplus(x) = log(1 + exp(x))*
    pub fn softplus(self) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let op = SoftPlusBackward::new(self.gradient, self.var.data.clone(), gradient.clone());

        VarDiff::new(
            self.var.softplus(),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }

    /// Applies the *sigmoid* element-wise and returns a differentiable variable with the result.
    pub fn sigmoid(self) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let var = self.var.sigmoid();
        let op = SigmoidBackward::new(self.gradient, var.data.clone(), gradient.clone());

        VarDiff::new(var, gradient, op, self.history, self.switchables)
    }

    /// Applies the *tanh* element-wise and returns a differentiable variable with the result.
    pub fn tanh(self) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let var = self.var.tanh();
        let op = TanHBackward::new(self.gradient, var.data.clone(), gradient.clone());

        VarDiff::new(var, gradient, op, self.history, self.switchables)
    }

    /// Applies the *natural logarithm* element-wise and returns a differentiable variable with the
    /// result.
    pub fn ln(self) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let op = LognBackward::new(self.gradient, self.var.data.clone(), gradient.clone());

        VarDiff::new(self.var.ln(), gradient, op, self.history, self.switchables)
    }

    /// Applies the *exponential* element-wise and returns a differentiable variable with the
    /// result.
    pub fn exp(self) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let var = self.var.exp();
        let op = ExpBackward::new(self.gradient, var.data.clone(), gradient.clone());

        VarDiff::new(var, gradient, op, self.history, self.switchables)
    }

    /// Applies the *softmax* to `self` and returns a differentiable variable with the result.
    ///
    /// The *softmax* is applied to all slices along `axis`, and will re-scale them so
    /// that the elements lie in the range *[0, 1]* and sum to 1.0.
    ///
    /// # Arguments
    ///
    /// `axis` - axis along which softmax will be computed.
    pub fn softmax(self, axis: usize) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let var = self.var.softmax(axis);
        let op = SoftmaxBackward::new(self.gradient, var.data.clone(), gradient.clone(), axis);

        VarDiff::new(var, gradient, op, self.history, self.switchables)
    }

    /// Applies the *log-softmax* to `self` and returns a differentiable variable with the result.
    ///
    /// Applies a softmax followed by a logarithm. While mathematically equivalent to
    /// *log(softmax(x))*, doing these two operations separately is slower, and numerically
    /// unstable. This function uses an alternative formulation to compute the output and
    /// gradient correctly.
    ///
    /// See also [`.softmax()`].
    ///
    /// [`.softmax()`]: VarDiff::softmax()
    ///
    /// # Arguments
    ///
    /// `axis` - axis along which log-softmax will be computed.
    pub fn log_softmax(self, axis: usize) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let var = self.var.log_softmax(axis);
        let op = LogSoftmaxBackward::new(self.gradient, var.data.clone(), gradient.clone(), axis);

        VarDiff::new(var, gradient, op, self.history, self.switchables)
    }

    /// Returns a differentiable variable equivalent to `self` with its dimensions reversed.
    pub fn t(self) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let op = TransposeBackward::new(self.gradient, gradient.clone());

        VarDiff::new(self.var.t(), gradient, op, self.history, self.switchables)
    }

    /// Applies *dropout* to `self` and returns a differentiable variable with the result.
    ///
    /// During training, randomly zeroes some of the elements of `self` with probability *p* using
    /// samples from a Bernoulli distribution. Each channel will be zeroed out independently on
    /// every forward call.
    ///
    /// This has proven to be an effective technique for regularization and preventing the
    /// co-adaptation of neurons as described in the paper
    /// [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580).
    ///
    /// Furthermore, the outputs are scaled by a factor of 1/(1 - p) during training. This means
    /// that during evaluation the resulting variable simply computes an identity function.
    ///
    /// [`nn::Dropout`]: crate::nn::Dropout        
    ///
    /// # Arguments
    ///
    /// * `p` - dropout factor.
    ///
    /// * `status` - dropout status.
    pub fn dropout(self, p: f64, status: Rc<Cell<bool>>) -> VarDiff<D> {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let noise = Rc::new(RefCell::new(Tensor::zeros(self.gradient.shape())));
        let var =
            self.var
                .dropout_with_noise(self.gradient.shape(), p, noise.clone(), status.clone());
        let op = DropoutBackward::new(self.gradient, gradient.clone(), noise, p, status);

        VarDiff::new(var, gradient, op, self.history, self.switchables)
    }

    /// Splits `self` into a certain number of chunks of size `chunk_size` **skipping** the
    /// remainder along each dimension that doesn’t fit evenly.
    ///
    /// # Arguments
    ///
    /// `chunk_size` - shape of the chunks.
    pub fn chunks<E>(self, chunk_size: E) -> Vec<VarDiff<D>>
    where
        E: IntoDimension<Dim = D>,
    {
        let vars = self.var.chunks(chunk_size);
        vars.into_iter()
            .enumerate()
            .map(|(i, var)| {
                let gradient = Rc::new(OptionalTensor::zeros(var.data.borrow().raw_dim()));
                let op = ChunkBackward::new(self.gradient.clone(), gradient.clone(), i);
                VarDiff::new(
                    var,
                    gradient,
                    op,
                    self.history.clone(),
                    self.switchables.clone(),
                )
            })
            .collect()
    }

    /// Returns a new differentiable variable with a dimension of size one inserted at the position
    /// specified by `axis`.
    pub fn unsqueeze(self, axis: usize) -> VarDiff<D::Larger> {
        let gradient = Rc::new(OptionalTensor::zeros(
            self.gradient.shape().insert_axis(Axis(axis)),
        ));
        let op = UnsqueezeBackward::new(self.gradient, gradient.clone());

        VarDiff::new(
            self.var.unsqueeze(axis),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl<D> VarDiff<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    /// Concatenates the given sequence of differentiable variables `variables`, including
    /// `self`, along the given axis, and returns a differentiable variable with the results.
    ///
    /// # Arguments
    ///
    /// * `variables` - sequence of differentiable variables.
    ///
    /// * `axis` - axis to concatenate along to.
    ///
    /// # Panics
    ///
    /// If the variables have mismatching shapes, apart from along axis, if the variables are empty,
    /// if `axis` is out of bounds or if the result is larger than is possible to represent.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuronika;
    /// use ndarray;
    ///
    ///
    /// let a = neuronika::ones((3, 2)).requires_grad();
    /// let b = neuronika::full((3, 2), 4.).requires_grad();
    /// let c = neuronika::full((3, 2), 3.).requires_grad();
    ///
    /// let mut d = a.cat(&[b, c], 1);
    /// d.forward();
    ///
    /// assert_eq!(*d.data(), ndarray::array![[1., 1., 4., 4., 3., 3.],
    ///                                       [1., 1., 4., 4., 3., 3.],
    ///                                       [1., 1., 4., 4., 3., 3.]]);
    /// ```
    pub fn cat(mut self, variables: &[Self], axis: usize) -> VarDiff<D> {
        let var = {
            let vars: Vec<Var<D>> = variables.iter().cloned().map(|x| x.var).collect();
            self.var.cat(&vars, axis)
        };
        let gradient = Rc::new(OptionalTensor::zeros(var.data.borrow().raw_dim()));
        let mut operands_gradients = Vec::with_capacity(variables.len());

        operands_gradients.push(self.gradient);
        variables.iter().cloned().for_each(|variable| {
            self.history.merge(variable.history);
            operands_gradients.push(variable.gradient);
        });
        let op = MultiConcatenateBackward::new(operands_gradients, gradient.clone(), axis);

        VarDiff::new(var, gradient, op, self.history, self.switchables)
    }

    /// Stacks the given sequence of differentiable variables `variables`, including
    /// `self`, along the given axis, and returns a differentiable variable with the results.
    ///
    /// All variables must have the same shape.
    ///
    /// # Arguments
    ///
    /// * `variables` - sequence of differentiable variables.
    ///
    /// * `axis` - axis to stack along to.
    ///
    /// # Panics
    ///
    /// If the variables have mismatching shapes, apart from along axis, if the variables are empty,
    /// if `axis` is out of bounds or if the result is larger than is possible to represent.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuronika;
    /// use ndarray;
    ///
    ///
    /// let a = neuronika::ones((2, 2)).requires_grad();
    /// let b = neuronika::ones((2, 2)).requires_grad();
    /// let c = neuronika::ones((2, 2)).requires_grad();
    ///
    /// let d = a.stack(&[b, c], 0);
    /// d.forward();
    ///
    /// assert_eq!(*d.data(), ndarray::array![[[1., 1.],
    ///                                        [1., 1.]],
    ///                                       [[1., 1.],
    ///                                        [1., 1.]],
    ///                                       [[1., 1.],
    ///                                        [1., 1.]]]);
    /// ```
    pub fn stack(mut self, variables: &[Self], axis: usize) -> VarDiff<D::Larger> {
        let var = {
            let vars: Vec<Var<D>> = variables.iter().cloned().map(|x| x.var).collect();
            self.var.stack(&vars, axis)
        };
        let gradient = Rc::new(OptionalTensor::zeros(var.data.borrow().raw_dim()));
        let mut operands_gradients = Vec::with_capacity(variables.len());
        operands_gradients.push(self.gradient);
        variables.iter().cloned().for_each(|variable| {
            self.history.merge(variable.history);
            operands_gradients.push(variable.gradient);
        });

        let op = MultiStackBackward::new(operands_gradients, gradient.clone(), axis);

        VarDiff::new(var, gradient, op, self.history, self.switchables)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arithmetic Operations Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VarDiff - f32 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Add<f32> for VarDiff<D>
where
    D: 'static + Dimension,
    D: DimMax<Ix0>,
{
    type Output = VarDiff<<D as DimMax<Ix0>>::Output>;

    fn add(self, rhs: f32) -> Self::Output {
        self + crate::full((), rhs)
    }
}

impl<D> Sub<f32> for VarDiff<D>
where
    D: 'static + Dimension,
    D: DimMax<Ix0>,
{
    type Output = VarDiff<<D as DimMax<Ix0>>::Output>;

    fn sub(self, rhs: f32) -> Self::Output {
        self - crate::full((), rhs)
    }
}

impl<D> Mul<f32> for VarDiff<D>
where
    D: 'static + Dimension,
    D: DimMax<Ix0>,
{
    type Output = VarDiff<<D as DimMax<Ix0>>::Output>;

    fn mul(self, rhs: f32) -> Self::Output {
        self * crate::full((), rhs)
    }
}

impl<D> Div<f32> for VarDiff<D>
where
    D: 'static + Dimension,
    D: DimMax<Ix0>,
{
    type Output = VarDiff<<D as DimMax<Ix0>>::Output>;

    fn div(self, rhs: f32) -> Self::Output {
        self / crate::full((), rhs)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ f32 - VarDiff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Add<VarDiff<D>> for f32
where
    D: 'static + Dimension,
    Ix0: DimMax<D>,
{
    type Output = VarDiff<<Ix0 as DimMax<D>>::Output>;

    fn add(self, rhs: VarDiff<D>) -> Self::Output {
        crate::full((), self) + rhs
    }
}

impl<D> Sub<VarDiff<D>> for f32
where
    D: 'static + Dimension,
    Ix0: DimMax<D>,
{
    type Output = VarDiff<<Ix0 as DimMax<D>>::Output>;

    fn sub(self, rhs: VarDiff<D>) -> Self::Output {
        crate::full((), self) - rhs
    }
}

impl<D> Mul<VarDiff<D>> for f32
where
    D: 'static + Dimension,
    Ix0: DimMax<D>,
{
    type Output = VarDiff<<Ix0 as DimMax<D>>::Output>;

    fn mul(self, rhs: VarDiff<D>) -> Self::Output {
        crate::full((), self) * rhs
    }
}

impl<D> Div<VarDiff<D>> for f32
where
    D: 'static + Dimension,
    Ix0: DimMax<D>,
{
    type Output = VarDiff<<Ix0 as DimMax<D>>::Output>;

    fn div(self, rhs: VarDiff<D>) -> Self::Output {
        crate::full((), self) / rhs
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Neg for VarDiff<D>
where
    D: 'static + Dimension,
{
    type Output = VarDiff<D>;

    fn neg(self) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::zeros(self.gradient.shape()));
        let op = NegationBackward::new(self.gradient, gradient.clone());

        VarDiff::new(self.var.neg(), gradient, op, self.history, self.switchables)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Addition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D, E> Add<Var<E>> for VarDiff<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn add(self, rhs: Var<E>) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.data.borrow(),
        )));
        let op = AdditionBackwardLeft::<D, E>::new(self.gradient, gradient.clone());

        VarDiff::new(
            self.var.add(rhs),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl<D, E> Add<VarDiff<E>> for VarDiff<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn add(mut self, rhs: VarDiff<E>) -> Self::Output {
        self.history.merge(rhs.history);
        self.switchables.merge(rhs.switchables);

        let gradient = Rc::new(OptionalTensor::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let op = AdditionBackward::new(
            AdditionBackwardLeft::new(self.gradient, gradient.clone()),
            AdditionBackwardRight::new(rhs.gradient, gradient.clone()),
        );

        VarDiff::new(
            self.var.add(rhs.var),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Subtraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D, E> Sub<Var<E>> for VarDiff<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn sub(self, rhs: Var<E>) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.data.borrow(),
        )));
        let op = SubtractionBackwardLeft::<D, E>::new(self.gradient, gradient.clone());

        VarDiff::new(
            self.var.sub(rhs),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl<D, E> Sub<VarDiff<E>> for VarDiff<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn sub(mut self, rhs: VarDiff<E>) -> Self::Output {
        self.history.merge(rhs.history);

        let gradient = Rc::new(OptionalTensor::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let left = SubtractionBackwardLeft::new(self.gradient, gradient.clone());
        let right = SubtractionBackwardRight::new(rhs.gradient, gradient.clone());
        let op = SubtractionBackward::new(left, right);

        VarDiff::new(
            self.var.sub(rhs.var),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D, E> Mul<Var<E>> for VarDiff<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn mul(self, rhs: Var<E>) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.data.borrow(),
        )));
        let buffer = Rc::new(OptionalTensor::zeros(gradient.shape()));
        // Va inserito in `switchables`

        let op = MultiplicationBackwardLeft::new(
            self.gradient,
            rhs.data.clone(),
            gradient.clone(),
            buffer,
        );

        VarDiff::new(
            self.var.mul(rhs),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl<D, E> Mul<VarDiff<E>> for VarDiff<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn mul(mut self, rhs: VarDiff<E>) -> Self::Output {
        self.history.merge(rhs.history);

        let gradient = Rc::new(OptionalTensor::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let buffer = Rc::new(OptionalTensor::zeros(gradient.shape()));
        // Va inserito in `switchables`!
        let left = MultiplicationBackwardLeft::new(
            self.gradient,
            rhs.var.data.clone(),
            gradient.clone(),
            buffer.clone(),
        );
        let right = MultiplicationBackwardRight::new(
            rhs.gradient,
            self.var.data.clone(),
            gradient.clone(),
            buffer.clone(),
        );
        let op = MultiplicationBackward::new(left, right);

        VarDiff::new(
            self.var.mul(rhs.var),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Division ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D, E> Div<Var<E>> for VarDiff<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn div(self, rhs: Var<E>) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.data.borrow(),
        )));
        let buffer = Rc::new(OptionalTensor::zeros(gradient.shape()));
        let op =
            DivisionBackwardLeft::new(self.gradient, rhs.data.clone(), gradient.clone(), buffer);

        VarDiff::new(
            self.var.div(rhs),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl<D, E> Div<VarDiff<E>> for VarDiff<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn div(mut self, rhs: VarDiff<E>) -> Self::Output {
        self.history.merge(rhs.history);

        let gradient = Rc::new(OptionalTensor::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let buffer = Rc::new(OptionalTensor::zeros(gradient.shape()));
        let left = DivisionBackwardLeft::new(
            self.gradient,
            rhs.var.data.clone(),
            gradient.clone(),
            buffer.clone(),
        );
        let right = DivisionBackwardRight::new(
            self.var.data.clone(),
            rhs.var.data.clone(),
            rhs.gradient,
            gradient.clone(),
            buffer.clone(),
        );
        let op = DivisionBackward::new(left, right);

        VarDiff::new(
            self.var.div(rhs.var),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Operations Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MatMatMul<Var<Ix2>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm(self, rhs: Var<Ix2>) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.data().raw_dim(),
        )));
        let op =
            MatrixMatrixMulBackwardLeft::new(self.gradient, rhs.data.clone(), gradient.clone());

        VarDiff::new(
            self.var.mm(rhs),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl MatMatMul<VarDiff<Ix2>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm(mut self, rhs: VarDiff<Ix2>) -> Self::Output {
        self.history.merge(rhs.history);

        let gradient = Rc::new(OptionalTensor::zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let left =
            MatrixMatrixMulBackwardLeft::new(self.gradient, rhs.var.data.clone(), gradient.clone());
        let right = MatrixMatrixMulBackwardRight::new(
            self.var.data.clone(),
            rhs.gradient,
            gradient.clone(),
        );
        let op = MatrixMatrixMulBackward::new(left, right);

        VarDiff::new(
            self.var.mm(rhs.var),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition  ~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MatMatMulT<Var<Ix2>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm_t(self, rhs: Var<Ix2>) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.data.borrow().t().raw_dim(),
        )));
        let op =
            MatrixMatrixMulTBackwardLeft::new(self.gradient, rhs.data.clone(), gradient.clone());

        VarDiff::new(
            self.var.mm_t(rhs),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl MatMatMulT<VarDiff<Ix2>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm_t(mut self, rhs: VarDiff<Ix2>) -> Self::Output {
        self.history.merge(rhs.history);

        let gradient = Rc::new(OptionalTensor::zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let left = MatrixMatrixMulTBackwardLeft::new(
            self.gradient,
            rhs.var.data.clone(),
            gradient.clone(),
        );
        let right = MatrixMatrixMulTBackwardRight::new(
            self.var.data.clone(),
            rhs.gradient,
            gradient.clone(),
        );
        let op = MatrixMatrixMulTBackward::new(left, right);

        VarDiff::new(
            self.var.mm_t(rhs.var),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MatVecMul<Var<Ix1>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix1>;

    fn mv(self, rhs: Var<Ix1>) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.data().raw_dim(),
        )));
        let op =
            MatrixVectorMulBackwardLeft::new(self.gradient, rhs.data.clone(), gradient.clone());

        VarDiff::new(
            self.var.mv(rhs),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl MatVecMul<VarDiff<Ix1>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix1>;

    fn mv(mut self, rhs: VarDiff<Ix1>) -> Self::Output {
        self.history.merge(rhs.history);

        let gradient = Rc::new(OptionalTensor::zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let left =
            MatrixVectorMulBackwardLeft::new(self.gradient, rhs.var.data.clone(), gradient.clone());
        let right = MatrixVectorMulBackwardRight::new(
            self.var.data.clone(),
            rhs.gradient,
            gradient.clone(),
        );
        let op = MatrixVectorMulBackward::new(left, right);

        VarDiff::new(
            self.var.mv(rhs.var),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl VecMatMul<Var<Ix2>> for VarDiff<Ix1> {
    type Output = VarDiff<Ix1>;

    fn vm(self, rhs: Var<Ix2>) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.data().raw_dim(),
        )));
        let op =
            VectorMatrixMulBackwardLeft::new(self.gradient, rhs.data.clone(), gradient.clone());

        VarDiff::new(
            self.var.vm(rhs),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl VecMatMul<VarDiff<Ix2>> for VarDiff<Ix1> {
    type Output = VarDiff<Ix1>;

    fn vm(mut self, rhs: VarDiff<Ix2>) -> Self::Output {
        self.history.merge(rhs.history);

        let gradient = Rc::new(OptionalTensor::zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let left =
            VectorMatrixMulBackwardLeft::new(self.gradient, rhs.var.data.clone(), gradient.clone());
        let right = VectorMatrixMulBackwardRight::new(
            self.var.data.clone(),
            rhs.gradient,
            gradient.clone(),
        );
        let op = VectorMatrixMulBackward::new(left, right);

        VarDiff::new(
            self.var.vm(rhs.var),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl VecVecMul<Var<Ix1>> for VarDiff<Ix1> {
    type Output = VarDiff<Ix0>;

    fn vv(self, rhs: Var<Ix1>) -> Self::Output {
        let gradient = Rc::new(OptionalTensor::from_ndarray(Tensor::zeros(())));
        let op =
            VectorVectorMulBackwardUnary::new(rhs.data.clone(), self.gradient, gradient.clone());

        VarDiff::new(
            self.var.vv(rhs),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl VecVecMul<VarDiff<Ix1>> for VarDiff<Ix1> {
    type Output = VarDiff<Ix0>;

    fn vv(mut self, rhs: VarDiff<Ix1>) -> Self::Output {
        self.history.merge(rhs.history);

        let gradient = Rc::new(OptionalTensor::from_ndarray(Tensor::zeros(())));
        let left = VectorVectorMulBackwardUnary::new(
            self.var.data.clone(),
            self.gradient.clone(),
            gradient.clone(),
        );
        let right = VectorVectorMulBackwardUnary::new(
            rhs.var.data.clone(),
            rhs.gradient.clone(),
            gradient.clone(),
        );
        let op = VectorVectorMulBackward::new(left, right);

        VarDiff::new(
            self.var.vv(rhs.var),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cat and Stack traits implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Cat<Var<D>> for VarDiff<D>
where
    D: Dimension + RemoveAxis,
{
    type Output = VarDiff<D>;

    fn cat(self, rhs: Var<D>, axis: usize) -> Self::Output {
        let array = concatenate(
            Axis(axis),
            &[self.var.data.borrow().view(), rhs.data.borrow().view()],
        )
        .unwrap();
        let gradient = Rc::new(OptionalTensor::from_ndarray(array));
        let op = ConcatenateBackwardLeft::new(self.gradient, gradient.clone(), axis);

        VarDiff::new(
            Cat::cat(self.var, rhs, axis),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl<D> Cat<VarDiff<D>> for VarDiff<D>
where
    D: Dimension + RemoveAxis,
{
    type Output = VarDiff<D>;

    fn cat(mut self, rhs: VarDiff<D>, axis: usize) -> Self::Output {
        self.history.merge(rhs.history);
        let array = concatenate(
            Axis(axis),
            &[self.var.data.borrow().view(), rhs.var.data.borrow().view()],
        )
        .unwrap();
        let gradient = Rc::new(OptionalTensor::from_ndarray(array));
        let left = ConcatenateBackwardLeft::new(self.gradient, gradient.clone(), axis);
        let right = ConcatenateBackwardRight::new(
            rhs.gradient,
            gradient.clone(),
            axis,
            self.var.data.borrow().len_of(Axis(axis)),
        );
        let op = ConcatenateBackward::new(left, right);

        VarDiff::new(
            Cat::cat(self.var, rhs.var, axis),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Stack<Var<D>> for VarDiff<D>
where
    D: Dimension + RemoveAxis,
{
    type Output = VarDiff<D::Larger>;

    fn stack(self, rhs: Var<D>, axis: usize) -> Self::Output {
        let array = stack(
            Axis(axis),
            &[self.var.data.borrow().view(), rhs.data.borrow().view()],
        )
        .unwrap();
        let gradient = Rc::new(OptionalTensor::from_ndarray(array));
        let op = StackBackwardLeft::new(self.gradient, gradient.clone(), axis);

        VarDiff::new(
            Stack::stack(self.var, rhs, axis),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

impl<D> Stack<VarDiff<D>> for VarDiff<D>
where
    D: Dimension + RemoveAxis,
{
    type Output = VarDiff<D::Larger>;

    fn stack(mut self, rhs: VarDiff<D>, axis: usize) -> Self::Output {
        self.history.merge(rhs.history);
        let array = stack(
            Axis(axis),
            &[self.var.data.borrow().view(), rhs.var.data.borrow().view()],
        )
        .unwrap();
        let gradient = Rc::new(OptionalTensor::from_ndarray(array));
        let left = StackBackwardLeft::new(self.gradient, gradient.clone(), axis);
        let right = StackBackwardRight::new(rhs.gradient, gradient.clone(), axis);
        let op = StackBackward::new(left, right);

        VarDiff::new(
            Stack::stack(self.var, rhs.var, axis),
            gradient,
            op,
            self.history,
            self.switchables,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Debug ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Debug for VarDiff<D>
where
    D: Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.var, f)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Display ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Display for VarDiff<D>
where
    D: Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.var)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Serialize ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(feature = "serialize")]
impl<D> Serialize for VarDiff<D>
where
    D: Dimension + Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data().serialize(serializer)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Deserialize ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(feature = "serialize")]
impl<'d, D> Deserialize<'d> for VarDiff<D>
where
    D: Dimension + Deserialize<'d>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'d>,
    {
        let data = Tensor::deserialize(deserializer).unwrap();
        Ok(Var::leaf(data).requires_grad())
    }
}
