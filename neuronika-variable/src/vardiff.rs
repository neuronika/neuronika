use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

use ndarray::{
    arr0, concatenate, stack, Array, Axis, DimMax, Dimension, IntoDimension, Ix0, Ix1, Ix2,
    RemoveAxis, Zip,
};

use crate::{
    autograd::Backward,
    gradient::{BufferedGradient, Gradient, NoGrad},
    history::History,
    node::*,
    utils::{cobroadcasted_zeros, DotDim},
    var::Var,
    Cat, Convolution, MatMatMul, MatMatMulT, MatVecMul, Reduction, Stack, VecMatMul, VecVecMul,
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
    D: Dimension,
{
    pub(crate) var: Var<D>,
    pub(crate) grad: Rc<Gradient<Array<f32, D>, D>>,
    pub(crate) history: History<(Rc<dyn Backward>, Rc<dyn NoGrad>)>,
}

impl<D> VarDiff<D>
where
    D: Dimension,
{
    pub(crate) fn leaf(var: Var<D>, array: Array<f32, D>) -> Self {
        Self {
            var,
            grad: Rc::new(Gradient::from_ndarray(array)),
            history: History::default(),
        }
    }

    pub(crate) fn node(
        var: Var<D>,
        grad: Rc<Gradient<Array<f32, D>, D>>,
        op: (Rc<dyn Backward>, Rc<dyn NoGrad>),
        mut history: History<(Rc<dyn Backward>, Rc<dyn NoGrad>)>,
    ) -> VarDiff<D> {
        history.insert(Rc::as_ptr(&op.0) as *const () as usize, op);

        Self { var, grad, history }
    }

    /// Returns an immutable reference to the data inside `self`.
    ///
    /// At the differentiable variable's creation the data is filled with zeros. You can populate it
    /// with a call to [`.forward()`](VarDiff::forward()).
    pub fn data(&self) -> Ref<Array<f32, D>> {
        self.var.data()
    }

    /// Returns a mutable reference to the data inside `self`.
    ///
    /// At the differentiable variable's creation the data is filled with zeros. You can populate it
    /// with a call to [`.forward()`](VarDiff::forward()).
    pub fn data_mut(&self) -> RefMut<Array<f32, D>> {
        self.var.data_mut()
    }

    /// Returns an immutable reference to the gradient inside `self`.
    ///
    /// At the differentiable variable's creation the gradient is filled with zeros. You can
    /// populate it with a call to [`.backward()`](VarDiff::backward()).
    pub fn grad(&self) -> Ref<Array<f32, D>> {
        self.grad.borrow()
    }

    /// Returns a mutable reference to the gradient inside `self`.
    ///
    /// At the differentiable variable's creation the gradient is filled with zeros. You can
    /// populate it with a call to [`.backward()`](VarDiff::backward()).
    pub fn grad_mut(&self) -> RefMut<Array<f32, D>> {
        self.grad.borrow_mut()
    }

    /// Sets the variable's gradient to zero.
    pub fn zero_grad(&self) {
        Zip::from(&mut *self.grad_mut()).for_each(|grad_el| *grad_el = 0.0);
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
        assert_eq!(
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
            .for_each(|(op, _)| op.backward());
    }

    /// Disables gradient computation and de-allocates the gradient for `self` and all of its
    /// ancestors.
    pub fn no_grad(&self) {
        let mut buffer = self.history.buffer_mut();

        if buffer.is_empty() {
            *buffer = self.history.to_vec();
        }

        buffer.iter().for_each(|(_, grad)| grad.no_grad());
    }

    /// Re-enables gradient computation and re-allocates the gradient for `self` and all of its
    /// ancestors.
    pub fn with_grad(&self) {
        let mut buffer = self.history.buffer_mut();

        if buffer.is_empty() {
            *buffer = self.history.to_vec();
        }

        buffer.iter().for_each(|(_, grad)| grad.with_grad());
    }
}

impl VarDiff<Ix0> {
    /// Returns the scalar contained in the variable.
    pub fn item(&self) -> f32 {
        self.data()[()]
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
        let grad = Rc::new(Gradient::from_ndarray(arr0(0.)));
        let op = SumBackward::new(self.grad.clone(), grad.clone());
        let var = self.var.sum();

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Returns the mean of all elements in `self`.
    pub fn mean(self) -> VarDiff<Ix0> {
        let grad = Rc::new(Gradient::from_ndarray(arr0(0.)));
        let op = MeanBackward::new(self.grad, grad.clone());
        let var = self.var.mean();

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Takes the power of each element in `self` with exponent `exp` and returns a differentiable
    /// variable with the result.
    ///
    /// # Arguments
    ///
    /// `exp` - exponent.
    pub fn pow(self, exp: i32) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let op = PowerBackward::new(self.grad, self.var.data.clone(), grad.clone(), exp);
        let var = self.var.pow(exp);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Takes the square root element-wise and returns a differentiable variable with the result.
    pub fn sqrt(self) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let var = self.var.sqrt();
        let op = SqrtBackward::new(self.grad, var.data.clone(), grad.clone());

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Applies the *rectified linear unit* element-wise and and returns a differentiable
    /// variable with the result.
    ///
    /// *ReLU(x) = max(0, x)*
    pub fn relu(self) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let op = ReLUBackward::new(self.grad, self.var.data.clone(), grad.clone());
        let var = self.var.relu();

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Applies the *leaky rectified linear unit* element-wise and returns a differentiable
    /// variable with the result.
    ///
    /// *LeakyReLU(x) = max(0, x) + 0.01 * min(0, x)*
    pub fn leaky_relu(self) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let op = LeakyReLUBackward::new(self.grad, self.var.data.clone(), grad.clone());
        let var = self.var.leaky_relu();

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Applies the *softplus* element-wise and returns a differentiable variable with the result.
    ///
    /// *Softplus(x) = log(1 + exp(x))*
    pub fn softplus(self) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let op = SoftPlusBackward::new(self.grad, self.var.data.clone(), grad.clone());
        let var = self.var.softplus();

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Applies the *sigmoid* element-wise and returns a differentiable variable with the result.
    pub fn sigmoid(self) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let var = self.var.sigmoid();
        let op = SigmoidBackward::new(self.grad, var.data.clone(), grad.clone());

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Applies the *tanh* element-wise and returns a differentiable variable with the result.
    pub fn tanh(self) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let var = self.var.tanh();
        let op = TanHBackward::new(self.grad, var.data.clone(), grad.clone());

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Applies the *natural logarithm* element-wise and returns a differentiable variable with the
    /// result.
    pub fn ln(self) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let op = LognBackward::new(self.grad, self.var.data.clone(), grad.clone());
        let var = self.var.ln();

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Applies the *exponential* element-wise and returns a differentiable variable with the
    /// result.
    pub fn exp(self) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let var = self.var.exp();
        let op = ExpBackward::new(self.grad, var.data.clone(), grad.clone());

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let var = self.var.softmax(axis);
        let op = SoftmaxBackward::new(self.grad, var.data.clone(), grad.clone(), axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let var = self.var.log_softmax(axis);
        let op = LogSoftmaxBackward::new(self.grad, var.data.clone(), grad.clone(), axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Returns a differentiable variable equivalent to `self` with its dimensions reversed.
    pub fn t(self) -> VarDiff<D> {
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let op = TransposeBackward::new(self.grad, grad.clone());
        let var = self.var.t();

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let noise = Rc::new(RefCell::new(Array::zeros(self.grad.shape())));
        let var = self
            .var
            .dropout_with_noise(self.grad.shape(), p, noise.clone(), status.clone());
        let op = DropoutBackward::new(self.grad, grad.clone(), p, noise, status);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Splits `self` into a certain number of chunks of size `chunk_size` **skipping** the
    /// remainder along each dimension that doesn’t fit evenly.
    ///
    /// # Arguments
    ///
    /// `chunk_size` - shape for each chunk.
    pub fn chunks<E>(self, chunk_size: E) -> Vec<VarDiff<D>>
    where
        E: IntoDimension<Dim = D>,
    {
        let vars = self.var.chunks(chunk_size);
        vars.into_iter()
            .enumerate()
            .map(|(i, var)| {
                let grad = Rc::new(Gradient::ndarray_zeros(var.data.borrow().raw_dim()));
                let op = ChunkBackward::new(self.grad.clone(), grad.clone(), i);

                VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history.clone())
            })
            .collect()
    }

    /// Returns a new differentiable variable with a dimension of size one inserted at the position
    /// specified by `axis`.
    ///
    /// # Arguments
    ///
    /// `axis` - dimension to insert the new axis at.
    pub fn unsqueeze(self, axis: usize) -> VarDiff<D::Larger> {
        let grad = Rc::new(Gradient::ndarray_zeros(
            self.grad.shape().insert_axis(Axis(axis)),
        ));
        let op = UnsqueezeBackward::new(self.grad, grad.clone());
        let var = self.var.unsqueeze(axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Computes the mean absolute error between the two variables.
    ///
    /// # Arguments
    ///
    /// * `target` - target variable.
    ///
    /// * `reduction` - reduction to apply to the criterion's output.
    pub fn mae(self, target: Var<D>, reduction: Reduction) -> VarDiff<Ix0> {
        let grad = Rc::new(Gradient::ndarray_zeros(().into_dimension()));
        let op = AbsoluteErrorBackward::new(
            self.var.data.clone(),
            target.data.clone(),
            self.grad,
            grad.clone(),
            reduction,
        );
        let var = self.var.mae(target, reduction);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Computes the mean squared error *(squared L2 norm)* between the two variables.
    ///
    /// # Arguments
    ///
    /// * `target` - target variable.
    ///
    /// * `reduction` - reduction to apply to the criterion's output.
    pub fn mse(self, target: Var<D>, reduction: Reduction) -> VarDiff<Ix0> {
        let grad = Rc::new(Gradient::ndarray_zeros(().into_dimension()));
        let op = SquaredErrorBackward::new(
            self.var.data.clone(),
            target.data.clone(),
            self.grad,
            grad.clone(),
            reduction,
        );
        let var = self.var.mse(target, reduction);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Computes the binary cross entropy between the two variables. The elements should be numbers
    /// between 0 and 1.
    ///
    /// Notice that if a component of the input x is either 0 or 1, one of the log terms would be
    /// mathematically undefined.
    ///
    /// Rust sets *ln(0) = -inf*, however, an infinite term in the equation is not desirable.
    ///
    /// Our solution is that the binary cross entropy clamps its log function outputs to be greater
    /// than or equal to -100. This way, we can always have a finite value.
    ///
    /// # Arguments
    ///
    /// * `target` - target variable.
    ///
    /// * `reduction` - reduction to apply to the criterion's output.
    pub fn bce(self, target: Var<D>, reduction: Reduction) -> VarDiff<Ix0> {
        let grad = Rc::new(Gradient::ndarray_zeros(().into_dimension()));
        let op = BinaryCrossEntropyBackward::new(
            self.var.data.clone(),
            target.data.clone(),
            self.grad,
            grad.clone(),
            reduction,
        );
        let var = self.var.bce(target, reduction);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Computes the binary cross entropy with logits between the two variables. The elements
    /// should be numbers between 0 and 1.
    ///
    /// This function combines a sigmoid and a binary cross entropy and is more numerically stable
    /// than using the two operations separately as, by fusing them, we take advantage of the
    /// log-sum-exp trick for numerical stability.
    ///
    /// Note that the target should be numbers between 0 and 1 and `self` should contain raw
    /// un-normalized scores.
    ///
    /// # Arguments
    ///
    /// * `target` - target variable.
    ///
    /// * `reduction` - reduction to apply to the criterion's output.
    pub fn bce_with_logits(self, target: Var<D>, reduction: Reduction) -> VarDiff<Ix0> {
        let grad = Rc::new(Gradient::ndarray_zeros(().into_dimension()));
        let op = BCEWithLogitsBackward::new(
            self.var.data.clone(),
            self.grad,
            target.data.clone(),
            grad.clone(),
            reduction,
        );
        let var = self.var.bce_with_logits(target, reduction);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Computes the Kullback-Leibler divergence between the two variables.
    ///
    /// The [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) is
    /// a useful distance measure for continuous distributions and is often useful when performing
    /// direct regression over the space of (discretely sampled) continuous output distributions.
    ///
    /// The `self` is expected to contain log-probabilities while the target is interpreted
    /// as probabilities. When the given reduction is equal to [`Reduction::Mean`] the total
    /// divergence is divided by the batch size, i.e. the size of outermost dimension.
    ///
    /// # Arguments
    ///
    /// * `target` - target variable.
    ///
    /// * `reduction` - reduction to apply to the criterion's output.
    pub fn kldiv(self, target: Var<D>, reduction: Reduction) -> VarDiff<Ix0> {
        let grad = Rc::new(Gradient::ndarray_zeros(().into_dimension()));
        let op = KLDivBackward::new(self.grad, target.data.clone(), grad.clone(), reduction);
        let var = self.var.kldiv(target, reduction);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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
    /// # use neuronika_variable as neuronika;
    /// use ndarray;
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
    pub fn cat(mut self, vars: &[Self], axis: usize) -> VarDiff<D> {
        let var = {
            let vars: Vec<Var<D>> = vars.iter().cloned().map(|x| x.var).collect();
            self.var.cat(&vars, axis)
        };
        let mut op_grads = Vec::with_capacity(vars.len());
        op_grads.push(self.grad);
        vars.iter().cloned().for_each(|var| {
            self.history.merge(var.history);
            op_grads.push(var.grad);
        });

        let grad = Rc::new(Gradient::ndarray_zeros(var.data.borrow().raw_dim()));
        let op = MultiConcatenateBackward::new(op_grads, grad.clone(), axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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
    /// # use neuronika_variable as neuronika;
    /// use ndarray;
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
    pub fn stack(mut self, vars: &[Self], axis: usize) -> VarDiff<D::Larger> {
        let var = {
            let vars: Vec<Var<D>> = vars.iter().cloned().map(|x| x.var).collect();
            self.var.stack(&vars, axis)
        };
        let mut op_grands = Vec::with_capacity(vars.len());
        op_grands.push(self.grad);
        vars.iter().cloned().for_each(|var| {
            self.history.merge(var.history);
            op_grands.push(var.grad);
        });

        let grad = Rc::new(Gradient::ndarray_zeros(var.data.borrow().raw_dim()));
        let op = MultiStackBackward::new(op_grands, grad.clone(), axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }

    /// Computes the negative log likelihood between the variables.
    ///
    /// `self` is expected to contain log-probabilities for each class, this is typically achieved
    /// by using [`.log_softmax()`](Var::log_softmax) and has to be a of shape either (minibatch, C)
    /// or (minibatch, C, d1, d2, ..., dk) with k >= 1 for the K-dimensional
    /// case.
    ///
    /// The target variable should be a class index in the range [0, C) where C = number of classes.
    ///
    /// When the given reduction is equal to [`Reduction::Mean`] the total negative likelihood is
    /// divided by the batch size.
    ///
    /// As mentioned before, this criterion can also be used for higher dimensional inputs, such as 2D
    /// images, by providing an input of size (minibatch, C, d1, d2, ..., dk) with k >= 1 where
    /// k is the number of dimensions. In the case of images, it computes the negative
    /// log-likelihood *per-pixel*.
    ///
    /// In the K-dimensional case this function expects a target of shape (minibatch, d1, d2, ..., dk).
    ///
    /// [`.log_softmax()`]: VarDiff::log_softmax()
    ///
    /// # Arguments
    ///
    /// * `target` - target variable.
    ///
    /// * `reduction` - reduction to apply to the criterion's output.
    pub fn nll(self, target: Var<D::Smaller>, reduction: Reduction) -> VarDiff<Ix0> {
        let grad = Rc::new(Gradient::ndarray_zeros(().into_dimension()));
        let op = NegativeLogLikelihoodBackward::new(
            target.data.clone(),
            self.grad,
            grad.clone(),
            reduction,
        );
        let var = self.var.nll(target, reduction);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

impl<D> VarDiff<D>
where
    D: 'static + Dimension,
    D::Smaller: RemoveAxis,
    <D::Smaller as Dimension>::Smaller: Copy,
{
    /// Applies the specified padding over the spatial dimensions of the variable.
    pub fn pad<T, E>(self, padding: E, mode: T) -> VarDiff<D>
    where
        T: 'static + PaddingMode<D>,
        E: IntoDimension<Dim = <D::Smaller as Dimension>::Smaller>,
    {
        let padding = padding.into_dimension();
        let grad = Rc::new(Gradient::ndarray_zeros(self.var.data().raw_dim()));
        let var = self.var.pad(padding, mode);
        let op = PadBackward::new(self.grad, grad.clone(), padding);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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
        let grad = Rc::new(Gradient::ndarray_zeros(self.grad.shape()));
        let op = NegationBackward::new(self.grad, grad.clone());
        let var = self.var.neg();

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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
        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.data.borrow(),
        )));
        let op = AdditionBackwardLeft::<D, E>::new(self.grad, grad.clone());
        let var = self.var.add(rhs);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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

        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let op = AdditionBackward::new(
            AdditionBackwardLeft::new(self.grad, grad.clone()),
            AdditionBackwardRight::new(rhs.grad, grad.clone()),
        );
        let var = self.var.add(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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
        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.data.borrow(),
        )));
        let op = SubtractionBackwardLeft::<D, E>::new(self.grad, grad.clone());
        let var = self.var.sub(rhs);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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

        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let left = SubtractionBackwardLeft::new(self.grad, grad.clone());
        let right = SubtractionBackwardRight::new(rhs.grad, grad.clone());
        let op = SubtractionBackward::new(left, right);
        let var = self.var.sub(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
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
        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.data.borrow(),
        )));
        let buff = Rc::new(BufferedGradient::from_ndarray(grad.clone()));
        let op = MultiplicationBackwardLeft::new(rhs.data.clone(), self.grad, buff.clone());
        let var = self.var.mul(rhs);

        VarDiff::node(var, grad, (Rc::new(op), buff), self.history)
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

        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let buff = Rc::new(BufferedGradient::from_ndarray(grad.clone()));
        let left = MultiplicationBackwardLeft::new(rhs.var.data.clone(), self.grad, buff.clone());
        let right = MultiplicationBackwardRight::new(self.var.data.clone(), rhs.grad, buff.clone());
        let op = MultiplicationBackward::new(left, right);
        let var = self.var.mul(rhs.var);

        VarDiff::node(var, grad, (Rc::new(op), buff), self.history)
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
        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.data.borrow(),
        )));
        let buff = Rc::new(BufferedGradient::from_ndarray(grad.clone()));
        let op = DivisionBackwardLeft::new(rhs.data.clone(), self.grad, buff.clone());
        let var = self.var.div(rhs);

        VarDiff::node(var, grad, (Rc::new(op), buff), self.history)
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

        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.var.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let buff = Rc::new(BufferedGradient::from_ndarray(grad.clone()));
        let left = DivisionBackwardLeft::new(rhs.var.data.clone(), self.grad, buff.clone());
        let right = DivisionBackwardRight::new(
            self.var.data.clone(),
            rhs.var.data.clone(),
            rhs.grad,
            buff.clone(),
        );
        let op = DivisionBackward::new(left, right);
        let var = self.var.div(rhs.var);

        VarDiff::node(var, grad, (Rc::new(op), buff), self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Operations Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MatMatMul<Var<Ix2>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm(self, rhs: Var<Ix2>) -> Self::Output {
        let grad = Rc::new(Gradient::ndarray_zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.data().raw_dim(),
        )));
        let op = MatrixMatrixMulBackwardLeft::new(rhs.data.clone(), self.grad, grad.clone());
        let var = self.var.mm(rhs);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

impl MatMatMul<VarDiff<Ix2>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm(mut self, rhs: VarDiff<Ix2>) -> Self::Output {
        self.history.merge(rhs.history);

        let grad = Rc::new(Gradient::ndarray_zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let left = MatrixMatrixMulBackwardLeft::new(rhs.var.data.clone(), self.grad, grad.clone());
        let right =
            MatrixMatrixMulBackwardRight::new(self.var.data.clone(), rhs.grad, grad.clone());
        let op = MatrixMatrixMulBackward::new(left, right);
        let var = self.var.mm(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition  ~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MatMatMulT<Var<Ix2>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm_t(self, rhs: Var<Ix2>) -> Self::Output {
        let grad = Rc::new(Gradient::ndarray_zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.data.borrow().t().raw_dim(),
        )));
        let op = MatrixMatrixMulTBackwardLeft::new(self.grad, rhs.data.clone(), grad.clone());
        let var = self.var.mm_t(rhs);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

impl MatMatMulT<VarDiff<Ix2>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm_t(mut self, rhs: VarDiff<Ix2>) -> Self::Output {
        self.history.merge(rhs.history);

        let grad = Rc::new(Gradient::ndarray_zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let left = MatrixMatrixMulTBackwardLeft::new(self.grad, rhs.var.data.clone(), grad.clone());
        let right =
            MatrixMatrixMulTBackwardRight::new(self.var.data.clone(), rhs.grad, grad.clone());
        let op = MatrixMatrixMulTBackward::new(left, right);
        let var = self.var.mm_t(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MatVecMul<Var<Ix1>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix1>;

    fn mv(self, rhs: Var<Ix1>) -> Self::Output {
        let grad = Rc::new(Gradient::ndarray_zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.data().raw_dim(),
        )));
        let op = MatrixVectorMulBackwardLeft::new(self.grad, rhs.data.clone(), grad.clone());
        let var = self.var.mv(rhs);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

impl MatVecMul<VarDiff<Ix1>> for VarDiff<Ix2> {
    type Output = VarDiff<Ix1>;

    fn mv(mut self, rhs: VarDiff<Ix1>) -> Self::Output {
        self.history.merge(rhs.history);

        let grad = Rc::new(Gradient::ndarray_zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let left = MatrixVectorMulBackwardLeft::new(self.grad, rhs.var.data.clone(), grad.clone());
        let right =
            MatrixVectorMulBackwardRight::new(self.var.data.clone(), rhs.grad, grad.clone());
        let op = MatrixVectorMulBackward::new(left, right);
        let var = self.var.mv(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl VecMatMul<Var<Ix2>> for VarDiff<Ix1> {
    type Output = VarDiff<Ix1>;

    fn vm(self, rhs: Var<Ix2>) -> Self::Output {
        let grad = Rc::new(Gradient::ndarray_zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.data().raw_dim(),
        )));
        let op = VectorMatrixMulBackwardLeft::new(self.grad, rhs.data.clone(), grad.clone());
        let var = self.var.vm(rhs);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

impl VecMatMul<VarDiff<Ix2>> for VarDiff<Ix1> {
    type Output = VarDiff<Ix1>;

    fn vm(mut self, rhs: VarDiff<Ix2>) -> Self::Output {
        self.history.merge(rhs.history);

        let grad = Rc::new(Gradient::ndarray_zeros(DotDim::shape(
            self.var.data().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let left = VectorMatrixMulBackwardLeft::new(self.grad, rhs.var.data.clone(), grad.clone());
        let right =
            VectorMatrixMulBackwardRight::new(self.var.data.clone(), rhs.grad, grad.clone());
        let op = VectorMatrixMulBackward::new(left, right);
        let var = self.var.vm(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl VecVecMul<Var<Ix1>> for VarDiff<Ix1> {
    type Output = VarDiff<Ix0>;

    fn vv(self, rhs: Var<Ix1>) -> Self::Output {
        let grad = Rc::new(Gradient::from_ndarray(Array::zeros(())));
        let op = VectorVectorMulBackwardUnary::new(rhs.data.clone(), self.grad, grad.clone());
        let var = self.var.vv(rhs);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

impl VecVecMul<VarDiff<Ix1>> for VarDiff<Ix1> {
    type Output = VarDiff<Ix0>;

    fn vv(mut self, rhs: VarDiff<Ix1>) -> Self::Output {
        self.history.merge(rhs.history);

        let grad = Rc::new(Gradient::from_ndarray(Array::zeros(())));
        let left = VectorVectorMulBackwardUnary::new(
            self.var.data.clone(),
            self.grad.clone(),
            grad.clone(),
        );
        let right =
            VectorVectorMulBackwardUnary::new(rhs.var.data.clone(), rhs.grad.clone(), grad.clone());
        let op = VectorVectorMulBackward::new(left, right);
        let var = self.var.vv(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cat and Stack traits implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Cat<Var<D>> for VarDiff<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = VarDiff<D>;

    fn cat(self, rhs: Var<D>, axis: usize) -> Self::Output {
        let borrow = concatenate(
            Axis(axis),
            &[self.var.data.borrow().view(), rhs.data.borrow().view()],
        )
        .unwrap();
        let grad = Rc::new(Gradient::from_ndarray(borrow));
        let op = ConcatenateBackwardLeft::new(self.grad, grad.clone(), axis);
        let var = Cat::cat(self.var, rhs, axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

impl<D> Cat<VarDiff<D>> for VarDiff<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = VarDiff<D>;

    fn cat(mut self, rhs: VarDiff<D>, axis: usize) -> Self::Output {
        self.history.merge(rhs.history);

        let borrow = concatenate(
            Axis(axis),
            &[self.var.data.borrow().view(), rhs.var.data.borrow().view()],
        )
        .unwrap();
        let grad = Rc::new(Gradient::from_ndarray(borrow));
        let left = ConcatenateBackwardLeft::new(self.grad, grad.clone(), axis);
        let right = ConcatenateBackwardRight::new(
            rhs.grad,
            grad.clone(),
            axis,
            self.var.data.borrow().len_of(Axis(axis)),
        );
        let op = ConcatenateBackward::new(left, right);
        let var = Cat::cat(self.var, rhs.var, axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Stack<Var<D>> for VarDiff<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = VarDiff<D::Larger>;

    fn stack(self, rhs: Var<D>, axis: usize) -> Self::Output {
        let borrow = stack(
            Axis(axis),
            &[self.var.data.borrow().view(), rhs.data.borrow().view()],
        )
        .unwrap();
        let grad = Rc::new(Gradient::from_ndarray(borrow));
        let op = StackBackwardLeft::new(self.grad, grad.clone(), axis);
        let var = Stack::stack(self.var, rhs, axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

impl<D> Stack<VarDiff<D>> for VarDiff<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = VarDiff<D::Larger>;

    fn stack(mut self, rhs: VarDiff<D>, axis: usize) -> Self::Output {
        self.history.merge(rhs.history);

        let borrow = stack(
            Axis(axis),
            &[self.var.data.borrow().view(), rhs.var.data.borrow().view()],
        )
        .unwrap();
        let grad = Rc::new(Gradient::from_ndarray(borrow));
        let left = StackBackwardLeft::new(self.grad, grad.clone(), axis);
        let right = StackBackwardRight::new(rhs.grad, grad.clone(), axis);
        let op = StackBackward::new(left, right);
        let var = Stack::stack(self.var, rhs.var, axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Convolution<Var<D>, <D::Smaller as Dimension>::Smaller> for VarDiff<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = VarDiff<D>;

    fn convolution<T>(self, input: Var<D>, stride: T, dilation: T, groups: usize) -> Self::Output
    where
        T: IntoDimension<Dim = <D::Smaller as Dimension>::Smaller> + Copy,
    {
        let input_data = input.data.clone();
        let var = self.var.convolution(input, stride, dilation, groups);
        let shape = var.data().raw_dim();
        let stride = stride.into_dimension();
        let dilation = dilation.into_dimension();
        let grad = Rc::new(Gradient::ndarray_zeros(shape));
        let op = ConvolutionBackwardKernel::new(
            input_data,
            self.grad,
            grad.clone(),
            stride,
            dilation,
            groups,
        );

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

impl<D> Convolution<VarDiff<D>, <D::Smaller as Dimension>::Smaller> for VarDiff<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = VarDiff<D>;

    fn convolution<T>(
        mut self,
        input: VarDiff<D>,
        stride: T,
        dilation: T,
        groups: usize,
    ) -> Self::Output
    where
        T: IntoDimension<Dim = <D::Smaller as Dimension>::Smaller> + Copy,
    {
        self.history.merge(input.history);
        let kernel_data = self.var.data.clone();
        let kernel_grad = self.grad.clone();
        let input_data = input.var.data.clone();
        let input_grad = input.grad.clone();
        let var = self.var.convolution(input.var, stride, dilation, groups);
        let shape = var.data().raw_dim();
        let grad = Rc::new(Gradient::ndarray_zeros(shape));
        let backward_input = ConvolutionBackwardInput::new(
            kernel_data,
            input_grad,
            grad.clone(),
            stride.into_dimension(),
            dilation.into_dimension(),
            groups,
        );
        let backward_kernel = ConvolutionBackwardKernel::new(
            input_data,
            kernel_grad,
            grad.clone(),
            stride.into_dimension(),
            dilation.into_dimension(),
            groups,
        );

        let op = ConvolutionBackward::new(backward_input, backward_kernel);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Debug ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Debug for VarDiff<D>
where
    D: 'static + Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.var, f)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Display ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Display for VarDiff<D>
where
    D: 'static + Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.var)
    }
}
