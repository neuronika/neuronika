use super::{
    Addition, AdditionBackward, AdditionBackwardUnary, Backward, Cat, Chunk, ChunkBackward,
    Concatenate, ConcatenateBackward, ConcatenateBackwardLeft, Data, DifferentiableVariable,
    Division, DivisionBackward, DivisionBackwardLeft, DivisionBackwardRight, Dropout,
    DropoutBackward, Exp, ExpBackward, Forward, Gradient, GradientOverwrite, Input, LeakyReLU,
    LeakyReLUBackward, LogSoftmax, LogSoftmaxBackward, Logn, LognBackward, MatMatMul, MatMatMulT,
    MatVecMul, MatrixMatrixMul, MatrixMatrixMulBackward, MatrixMatrixMulBackwardLeft,
    MatrixMatrixMulT, MatrixMatrixMulTBackward, MatrixMatrixMulTBackwardLeft, MatrixVectorMul,
    MatrixVectorMulBackward, MatrixVectorMulBackwardLeft, Mean, MeanBackward, MultiConcatenate,
    MultiConcatenateBackward, MultiStack, MultiStackBackward, Multiplication,
    MultiplicationBackward, MultiplicationBackwardUnary, Negation, NegationBackward, Overwrite,
    Param, Power, PowerBackward, RawParam, ReLU, ReLUBackward, Sigmoid, SigmoidBackward, SoftPlus,
    SoftPlusBackward, Softmax, SoftmaxBackward, Sqrt, SqrtBackward, Stack, StackBackward,
    StackBackwardLeft, Subtraction, SubtractionBackward, SubtractionBackwardLeft,
    SubtractionBackwardRight, Sum, SumBackward, TanH, TanHBackward, Tensor, Transpose,
    TransposeBackward, Unsqueeze, UnsqueezeBackward, Var, VarDiffHistory, Variable, VecMatMul,
    VecVecMul, VectorMatrixMul, VectorMatrixMulBackward, VectorMatrixMulBackwardLeft,
    VectorVectorMul, VectorVectorMulBackward, VectorVectorMulBackwardUnary, OPERATIONS_COUNTER,
};
use crate::nn::Register;
use ndarray::{DimMax, Dimension, IntoDimension, Ix0, Ix1, Ix2, RemoveAxis};
#[cfg(feature = "serialize")]
use serde::{
    de::{Deserialize, Deserializer},
    ser::{Serialize, Serializer},
};
use std::{
    cell::{Cell, Ref, RefMut},
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
///
/// 2. By performing any binary operation between a [`Var`] and a `VarDiff`. Differentiability
/// is thus a *contagious* property, that is, if during a computation a `VarDiff` is used, the
/// result of the computation itself, and that of any other subsequent computations performed on
/// it, will also be differentiable. As an obvious consequence, the results of operations
/// performed on two *VarDiff* will also be *VarDiff*.
pub struct VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient + Overwrite + 'static,
{
    pub(crate) var: Var<T>,
    pub(crate) node: Rc<U>,
    pub(crate) past: VarDiffHistory,
}

impl<T, U> Clone for VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient + Overwrite + 'static,
{
    fn clone(&self) -> Self {
        Self {
            var: self.var.clone(),
            node: self.node.clone(),
            past: self.past.clone(),
        }
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient + Overwrite + Backward + 'static,
{
    pub(crate) fn from(node: U, mut past: VarDiffHistory, var: Var<T>) -> VarDiff<T, U> {
        let node = Rc::new(node);
        past.append_backward(unsafe { OPERATIONS_COUNTER.next() }, node.clone());

        VarDiff { var, node, past }
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient + Overwrite + 'static,
{
    /// Returns an immutable reference to the data inside `self`.
    ///
    /// At the differentiable variable's creation the data is filled with zeros. You can populate it
    /// with a call to [`.forward()`](VarDiff::forward()).
    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.var.node.data()
    }

    /// Returns a mutable reference to the data inside `self`.
    ///
    /// At the differentiable variable's creation the data is filled with zeros. You can populate it
    /// with a call to [`.forward()`](VarDiff::forward()).
    pub fn data_mut(&self) -> RefMut<Tensor<T::Dim>> {
        self.var.node.data_mut()
    }

    /// Returns an immutable reference to the gradient inside `self`.
    ///
    /// At the differentiable variable's creation the gradient is filled with zeros. You can
    /// populate it with a call to [`.backward()`](VarDiff::backward()).
    pub fn grad(&self) -> Ref<Tensor<U::Dim>> {
        self.node.gradient()
    }

    /// Returns a mutable reference to the gradient inside `self`.
    ///
    /// At the differentiable variable's creation the gradient is filled with zeros. You can
    /// populate it with a call to [`.backward()`](VarDiff::backward()).
    pub fn grad_mut(&self) -> RefMut<Tensor<U::Dim>> {
        self.node.gradient_mut()
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
{
    /// Propagates the computations forwards and populates all the variables and differentiable
    /// variables from the leaves of the graph to `self`.   
    pub fn forward(&self) {
        self.var.forward();

        debug_assert!(self.past.buffer().is_empty() || self.past.len() == self.past.buffer().len());

        // If the backward buffer isn't empty, then we're doing a `forward -> backward -> forward`
        // chain, thus we must reset the `overwrite` bit of every `backward` node of our past.
        self.past.prepare_buffer();
        let buffer = self.past.buffer();
        let mut res = buffer.binary_search_by(|n| {
            if n.can_overwrite() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        if let Err(i) = res {
            if buffer.get(i).is_some() {
                res = Ok(i);
            }
        };

        if let Ok(pos) = res {
            for node in &buffer[pos..] {
                node.set_overwrite(true);
            }
        }
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + Backward + 'static,
{
    /// Back-propagates through the computational graph and populates the gradients of the
    /// differentiable leaves that are ancestors of `self`. Before back-propagating the gradient
    /// of `self` is seeded with `seed`, thus, the leaves' gradients will be scaled accordingly.
    ///
    /// The graph is differentiated through the
    /// [chain rule](https://en.wikipedia.org/wiki/Chain_rule).
    ///
    /// The leaves whose gradients are populated by this method are also those referred by the
    /// vector of [`Param`] returned by [`.parameters()`].
    ///
    ///  [`.parameters()`]: VarDiff::parameters()
    pub fn backward(&self, seed: f32) {
        debug_assert!(!self.past.is_empty());

        self.node.gradient_mut().fill(seed);
        self.past.prepare_buffer();
        let buffer = self.past.buffer();
        for node in buffer.iter().rev() {
            node.backward();
        }

        debug_assert_eq!(
            self.var.past.len(),
            self.var.past.buffer().len(),
            "Perhaps you forgot to call forward()?"
        );

        self.var.past.prepare_buffer();
        let buffer = self.var.past.buffer();
        let mut res = buffer.binary_search_by(|n| {
            if n.was_computed() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        if let Err(i) = res {
            if buffer.get(i).is_some() {
                res = Ok(i);
            }
        };

        if let Ok(pos) = res {
            for node in &buffer[pos..] {
                node.reset_computation();
            }
        }
    }

    /// Disables gradient computation and de-allocates the gradient for `self` and all of its
    /// ancestors.
    pub fn no_grad(&self) {
        self.past.prepare_buffer();
        for node in self.past.buffer.borrow().iter() {
            node.no_grad();
        }
    }

    /// Re-enables gradient computation and re-allocates the gradient for `self` and all of its
    /// ancestors.
    pub fn with_grad(&self) {
        self.past.prepare_buffer();
        for node in self.past.buffer.borrow().iter() {
            node.with_grad();
        }
    }

    /// This has effect only on certain **ancestor** variables of `self`. It sets such variables
    /// and differentiable variables in training mode.
    ///    
    /// See also [`.dropout()`].
    ///
    /// [`.dropout()`]: VarDiff::dropout()
    pub fn train(&self) {
        // Status is shared.
        self.var.train();
    }

    /// This has effect only on certain **ancestor** variables of `self`. It sets such variables
    /// and differentiable variables in evaluation mode.
    ///    
    /// See also [`.dropout()`].
    ///
    /// [`.dropout()`]: VarDiff::dropout()
    pub fn eval(&self) {
        // Status is shared.
        self.var.eval();
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data<Dim = Ix1> + 'static,
    U: Gradient<Dim = Ix1> + Overwrite + 'static,
{
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

impl<T, U> VarDiff<T, U>
where
    T: Data<Dim = Ix2> + 'static,
    U: Gradient<Dim = Ix2> + Overwrite + 'static,
{
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

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
{
    /// Returns a vector of [`Param`] referencing all the differentiable leaves that are ancestors
    /// of the variable.
    ///
    /// If directly called on a differentiable leaf the resulting vector will include only a single
    /// `Param` referencing `self`.
    ///
    /// Ancestors that appear multiple times in the computation of the variable are listed only
    /// once. Thus, the parameters of a differentiable variable *z* resulting from a binary
    /// operation involving two other differentiable variables *x* and *y* will be the set union
    /// of the parameters of *x* and *y*. This can be extended to the general case.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuronika;
    ///
    /// // x has 2 parameters, as it is the result of an addition.
    /// let x = neuronika::rand((3,3)).requires_grad() + neuronika::rand((3,3)).requires_grad();
    /// // The same holds for y.
    /// let y = neuronika::rand(3).requires_grad() + neuronika::rand(1).requires_grad();
    ///
    /// assert!(x.parameters().len() == y.parameters().len() && y.parameters().len() == 2);
    ///
    /// // z is the result of an addition between x and y, so it will have 4 parameters.
    /// let z = x.clone() + y;
    /// assert_eq!(z.parameters().len(), 4);
    ///
    /// // If we add x to z there still will be 4 parameters, as x is already present among them.
    /// let w = z + x;
    /// assert_eq!(w.parameters().len(), 4);
    /// ```
    pub fn parameters(&self) -> Vec<Param<'_>> {
        self.past
            .parameters
            .iter()
            .cloned()
            .map(RawParam::into_param)
            .collect()
    }

    /// Returns the sum of all elements in `self`.
    pub fn sum(self) -> VarDiff<Sum<T>, SumBackward<U>> {
        let node = SumBackward::new(self.node);
        VarDiff::from(node, self.past, self.var.sum())
    }

    /// Returns the mean of all elements in `self`.
    pub fn mean(self) -> VarDiff<Mean<T>, MeanBackward<U>> {
        let node = MeanBackward::new(self.node);
        VarDiff::from(node, self.past, self.var.mean())
    }

    /// Takes the power of each element in `self` with exponent `exp` and returns a differentiable
    /// variable with the result.
    pub fn pow(self, exp: i32) -> VarDiff<Power<T>, PowerBackward<U, T>> {
        let node = PowerBackward::new(self.node, self.var.node.clone(), exp);
        VarDiff::from(node, self.past, self.var.pow(exp))
    }

    /// Takes the square root element-wise and returns a differentiable variable with the result.
    pub fn sqrt(self) -> VarDiff<Sqrt<T>, SqrtBackward<U, Sqrt<T>>> {
        let var = self.var.sqrt();
        let node = SqrtBackward::new(self.node, var.node.clone());
        VarDiff::from(node, self.past, var)
    }

    /// Applies the *rectified linear unit* element-wise and and returns a differentiable
    /// variable with the result.
    ///
    /// *ReLU(x) = max(0, x)*
    pub fn relu(self) -> VarDiff<ReLU<T>, ReLUBackward<U, T>> {
        let node = ReLUBackward::new(self.node, self.var.node.clone());
        VarDiff::from(node, self.past, self.var.relu())
    }

    /// Applies the *leaky rectified linear unit* element-wise and returns a differentiable
    /// variable with the result.
    ///
    /// *LeakyReLU(x) = max(0, x) + 0.01 * min(0, x)*
    pub fn leaky_relu(self) -> VarDiff<LeakyReLU<T>, LeakyReLUBackward<U, T>> {
        let node = LeakyReLUBackward::new(self.node, self.var.node.clone());
        VarDiff::from(node, self.past, self.var.leaky_relu())
    }

    /// Applies the *softplus* element-wise and returns a differentiable variable with the result.
    ///
    /// *Softplus(x) = log(1 + exp(x))*
    pub fn softplus(self) -> VarDiff<SoftPlus<T>, SoftPlusBackward<U, T>> {
        let node = SoftPlusBackward::new(self.node, self.var.node.clone());
        VarDiff::from(node, self.past, self.var.softplus())
    }

    /// Applies the *sigmoid* element-wise and returns a differentiable variable with the result.
    pub fn sigmoid(self) -> VarDiff<Sigmoid<T>, SigmoidBackward<U, Sigmoid<T>>> {
        let var = self.var.sigmoid();
        let node = SigmoidBackward::new(self.node, var.node.clone());
        VarDiff::from(node, self.past, var)
    }

    /// Applies the *tanh* element-wise and returns a differentiable variable with the result.
    pub fn tanh(self) -> VarDiff<TanH<T>, TanHBackward<U, TanH<T>>> {
        let var = self.var.tanh();
        let node = TanHBackward::new(self.node, var.node.clone());
        VarDiff::from(node, self.past, var)
    }

    /// Applies the *natural logarithm* element-wise and returns a differentiable variable with the
    /// result.
    pub fn ln(self) -> VarDiff<Logn<T>, LognBackward<U, T>> {
        let node = LognBackward::new(self.node, self.var.node.clone());
        VarDiff::from(node, self.past, self.var.ln())
    }

    /// Applies the *exponential* element-wise and returns a differentiable variable with the
    /// result.
    pub fn exp(self) -> VarDiff<Exp<T>, ExpBackward<U, Exp<T>>> {
        let var = self.var.exp();
        let node = ExpBackward::new(self.node, var.node.clone());
        VarDiff::from(node, self.past, var)
    }

    /// Applies the *softmax* to `self` and returns a differentiable variable with the result.
    ///
    /// The *softmax* is applied to all slices along `axis`, and will re-scale them so
    ///  that the elements lie in the range *[0, 1]* and sum to 1.0.
    pub fn softmax(self, axis: usize) -> VarDiff<Softmax<T>, SoftmaxBackward<U, Softmax<T>>> {
        let var = self.var.softmax(axis);
        let node = SoftmaxBackward::new(self.node, var.node.clone(), axis);
        VarDiff::from(node, self.past, var)
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
    pub fn log_softmax(
        self,
        axis: usize,
    ) -> VarDiff<LogSoftmax<T>, LogSoftmaxBackward<U, LogSoftmax<T>>> {
        let var = self.var.log_softmax(axis);
        let node = LogSoftmaxBackward::new(self.node, var.node.clone(), axis);
        VarDiff::from(node, self.past, var)
    }

    /// Returns a differentiable variable equivalent to `self` with its dimensions reversed.
    pub fn t(self) -> VarDiff<Transpose<T>, TransposeBackward<U>> {
        let node = TransposeBackward::new(self.node);
        VarDiff::from(node, self.past, self.var.t())
    }

    /// Applies *dropout* to `self` and returns a differentiable variable with the result.
    ///
    /// It is strongly suggested to use [`nn::Dropout`] instead of this method when working with
    /// neural networks.
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
    pub fn dropout(self, p: f64) -> VarDiff<Dropout<T>, DropoutBackward<U, T>> {
        self.dropout_with_status(p, Rc::new(Cell::new(true)))
    }

    /// Creates a new dropout differentiable variable sharing the status with its internal val.
    pub(crate) fn dropout_with_status(
        self,
        p: f64,
        status: Rc<Cell<bool>>,
    ) -> VarDiff<Dropout<T>, DropoutBackward<U, T>> {
        let var = self.var.dropout_with_status(p, status);
        let node = DropoutBackward::new(self.node, var.node.clone(), p, var.node.status());
        VarDiff::from(node, self.past, var)
    }

    /// Splits `self` into a certain number of chunks of size `chunk_size` **skipping** the
    /// remainder along each dimension that doesn’t fit evenly.
    pub fn chunks<E>(self, chunk_size: E) -> Vec<VarDiff<Chunk<T>, ChunkBackward<U>>>
    where
        E: IntoDimension<Dim = T::Dim>,
    {
        self.var
            .node
            .data()
            .exact_chunks(chunk_size)
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let var = Var::from(
                    Chunk::new(self.var.node.clone(), chunk.to_owned(), i),
                    self.var.past.clone(),
                );
                VarDiff::from(
                    ChunkBackward::new(self.node.clone(), chunk.map(|_| 0.), i),
                    self.past.clone(),
                    var,
                )
            })
            .collect()
    }

    /// Returns a new differentiable variable with a dimension of size one inserted at the position
    /// specified by `axis`.
    pub fn unsqueeze(self, axis: usize) -> VarDiff<Unsqueeze<T>, UnsqueezeBackward<U>> {
        VarDiff::from(
            UnsqueezeBackward::new(self.node, axis),
            self.past,
            self.var.unsqueeze(axis),
        )
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    T::Dim: RemoveAxis,
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
    /// use std::boxed::Box;
    /// use neuronika;
    /// use ndarray;
    ///
    ///
    /// let a = neuronika::ones((3, 2)).requires_grad();
    /// let b = neuronika::full((3, 2), 4.).requires_grad();
    /// let c = neuronika::full((3, 2), 3.).requires_grad();
    ///
    /// let mut d = a.cat(&[Box::new(b), Box::new(c)], 1);
    /// d.forward();
    ///
    /// assert_eq!(*d.data(), ndarray::array![[1., 1., 4., 4., 3., 3.],
    ///                                       [1., 1., 4., 4., 3., 3.],
    ///                                       [1., 1., 4., 4., 3., 3.]]);
    /// ```
    pub fn cat(
        mut self,
        variables: &[Box<dyn DifferentiableVariable<T::Dim>>],
        axis: usize,
    ) -> VarDiff<MultiConcatenate<T::Dim>, MultiConcatenateBackward<T::Dim>> {
        let vars: Vec<Box<dyn Variable<T::Dim>>> =
            variables.iter().map(|el| el.get_var()).collect();
        let var = self.var.cat(&vars, axis);
        let shape = var.data().raw_dim();

        let mut operands: Vec<Rc<dyn GradientOverwrite<T::Dim>>> =
            Vec::with_capacity(variables.len() + 1);
        operands.push(self.node);

        for variable in variables {
            self.past.merge(variable.get_past());
            operands.push(variable.get_node());
        }

        VarDiff::from(
            MultiConcatenateBackward::new(operands, axis, shape),
            self.past,
            var,
        )
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
    /// use std::boxed::Box;
    /// use neuronika;
    /// use ndarray;
    ///
    ///
    /// let a = neuronika::ones((2, 2)).requires_grad();
    /// let b = neuronika::ones((2, 2)).requires_grad();
    /// let c = neuronika::ones((2, 2)).requires_grad();
    ///
    /// let mut d = a.stack(&[Box::new(b), Box::new(c)], 0);
    /// d.forward();
    ///
    /// assert_eq!(*d.data(), ndarray::array![[[1., 1.],
    ///                                        [1., 1.]],
    ///                                       [[1., 1.],
    ///                                        [1., 1.]],
    ///                                       [[1., 1.],
    ///                                        [1., 1.]]]);
    /// ```
    pub fn stack(
        mut self,
        variables: &[Box<dyn DifferentiableVariable<T::Dim>>],
        axis: usize,
    ) -> VarDiff<MultiStack<T::Dim>, MultiStackBackward<T::Dim>> {
        let vars: Vec<Box<dyn Variable<T::Dim>>> =
            variables.iter().map(|el| el.get_var()).collect();
        let var = self.var.stack(&vars, axis);
        let shape = var.data().raw_dim();

        let mut operands: Vec<Rc<dyn GradientOverwrite<T::Dim>>> =
            Vec::with_capacity(variables.len() + 1);
        operands.push(self.node);

        for variable in variables {
            self.past.merge(variable.get_past());
            operands.push(variable.get_node());
        }

        VarDiff::from(
            MultiStackBackward::new(operands, axis, shape),
            self.past,
            var,
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arithmetic Operations Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VarDiff - f32 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T, U> Add<f32> for VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    T::Dim: DimMax<Ix0>,
{
    type Output = VarDiff<Addition<T, Input<Ix0>>, AdditionBackwardUnary<U, Input<Ix0>>>;

    fn add(self, rhs: f32) -> Self::Output {
        self + crate::full((), rhs)
    }
}

impl<T, U> Sub<f32> for VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    T::Dim: DimMax<Ix0>,
{
    type Output = VarDiff<Subtraction<T, Input<Ix0>>, SubtractionBackwardLeft<U, Input<Ix0>>>;

    fn sub(self, rhs: f32) -> Self::Output {
        self - crate::full((), rhs)
    }
}

impl<T, U> Mul<f32> for VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    T::Dim: DimMax<Ix0>,
{
    type Output =
        VarDiff<Multiplication<T, Input<Ix0>>, MultiplicationBackwardUnary<U, Input<Ix0>>>;

    fn mul(self, rhs: f32) -> Self::Output {
        self * crate::full((), rhs)
    }
}

impl<T, U> Div<f32> for VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    T::Dim: DimMax<Ix0>,
{
    type Output = VarDiff<Division<T, Input<Ix0>>, DivisionBackwardLeft<U, Input<Ix0>>>;

    fn div(self, rhs: f32) -> Self::Output {
        self / crate::full((), rhs)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ f32 - VarDiff ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T, U> Add<VarDiff<T, U>> for f32
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    Ix0: DimMax<T::Dim>,
{
    type Output = VarDiff<Addition<Input<Ix0>, T>, AdditionBackwardUnary<U, Input<Ix0>>>;

    fn add(self, rhs: VarDiff<T, U>) -> Self::Output {
        crate::full((), self) + rhs
    }
}

impl<T, U> Sub<VarDiff<T, U>> for f32
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    Ix0: DimMax<T::Dim>,
{
    type Output = VarDiff<Subtraction<Input<Ix0>, T>, SubtractionBackwardRight<U, Input<Ix0>>>;

    fn sub(self, rhs: VarDiff<T, U>) -> Self::Output {
        crate::full((), self) - rhs
    }
}

impl<T, U> Mul<VarDiff<T, U>> for f32
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    Ix0: DimMax<T::Dim>,
{
    type Output =
        VarDiff<Multiplication<Input<Ix0>, T>, MultiplicationBackwardUnary<U, Input<Ix0>>>;

    fn mul(self, rhs: VarDiff<T, U>) -> Self::Output {
        crate::full((), self) * rhs
    }
}

impl<T, U> Div<VarDiff<T, U>> for f32
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    Ix0: DimMax<T::Dim>,
{
    type Output = VarDiff<Division<Input<Ix0>, T>, DivisionBackwardRight<Input<Ix0>, T, U>>;

    fn div(self, rhs: VarDiff<T, U>) -> Self::Output {
        crate::full((), self) / rhs
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T, U> Neg for VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
{
    type Output = VarDiff<Negation<T>, NegationBackward<U>>;

    fn neg(self) -> Self::Output {
        VarDiff::from(NegationBackward::new(self.node), self.past, self.var.neg())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Addition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> Add<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B1::Dim: Dimension + DimMax<F2::Dim>,
{
    type Output = VarDiff<Addition<F1, F2>, AdditionBackwardUnary<B1, F2>>;

    fn add(self, rhs: Var<F2>) -> Self::Output {
        let node = AdditionBackwardUnary::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.add(rhs))
    }
}

impl<F1, B1, F2, B2> Add<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    B2: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B1::Dim: Dimension + DimMax<B2::Dim>,
{
    type Output = VarDiff<Addition<F1, F2>, AdditionBackward<B1, B2>>;

    fn add(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = AdditionBackward::new(self.node, rhs.node);
        VarDiff::from(node, self.past, self.var.add(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Subtraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> Sub<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B1::Dim: Dimension + DimMax<F2::Dim>,
{
    type Output = VarDiff<Subtraction<F1, F2>, SubtractionBackwardLeft<B1, F2>>;

    fn sub(self, rhs: Var<F2>) -> Self::Output {
        let node = SubtractionBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.sub(rhs))
    }
}

impl<F1, B1, F2, B2> Sub<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    B2: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B1::Dim: Dimension + DimMax<B2::Dim>,
{
    type Output = VarDiff<Subtraction<F1, F2>, SubtractionBackward<B1, B2>>;

    fn sub(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = SubtractionBackward::new(self.node, rhs.node);
        VarDiff::from(node, self.past, self.var.sub(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> Mul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B1::Dim: Dimension + DimMax<F2::Dim>,
{
    type Output = VarDiff<Multiplication<F1, F2>, MultiplicationBackwardUnary<B1, F2>>;

    fn mul(self, rhs: Var<F2>) -> Self::Output {
        let node = MultiplicationBackwardUnary::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.mul(rhs))
    }
}

impl<F1, B1, F2, B2> Mul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    B2: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B1::Dim: Dimension + DimMax<B2::Dim>,
{
    type Output = VarDiff<Multiplication<F1, F2>, MultiplicationBackward<F1, B1, F2, B2>>;

    fn mul(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = MultiplicationBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.mul(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Division ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> Div<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B1::Dim: Dimension + DimMax<F2::Dim>,
{
    type Output = VarDiff<Division<F1, F2>, DivisionBackwardLeft<B1, F2>>;

    fn div(self, rhs: Var<F2>) -> Self::Output {
        let node = DivisionBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.div(rhs))
    }
}

impl<F1, B1, F2, B2> Div<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    B2: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B1::Dim: Dimension + DimMax<B2::Dim>,
{
    type Output = VarDiff<Division<F1, F2>, DivisionBackward<F1, B1, F2, B2>>;

    fn div(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = DivisionBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.div(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Operations Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> MatMatMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = VarDiff<MatrixMatrixMul<F1, F2>, MatrixMatrixMulBackwardLeft<B1, F2>>;

    fn mm(self, rhs: Var<F2>) -> Self::Output {
        let node = MatrixMatrixMulBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.mm(rhs))
    }
}

impl<F1, B1, F2, B2> MatMatMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixMatrixMul<F1, F2>, MatrixMatrixMulBackward<F1, B1, F2, B2>>;

    fn mm(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = MatrixMatrixMulBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.mm(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition  ~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> MatMatMulT<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = VarDiff<MatrixMatrixMulT<F1, F2>, MatrixMatrixMulTBackwardLeft<B1, F2>>;

    fn mm_t(self, rhs: Var<F2>) -> Self::Output {
        let node = MatrixMatrixMulTBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.mm_t(rhs))
    }
}

impl<F1, B1, F2, B2> MatMatMulT<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixMatrixMulT<F1, F2>, MatrixMatrixMulTBackward<F1, B1, F2, B2>>;

    fn mm_t(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = MatrixMatrixMulTBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.mm_t(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> MatVecMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = VarDiff<MatrixVectorMul<F1, F2>, MatrixVectorMulBackwardLeft<B1, F2>>;

    fn mv(self, rhs: Var<F2>) -> Self::Output {
        let node = MatrixVectorMulBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.mv(rhs))
    }
}

impl<F1, B1, F2, B2> MatVecMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
    B2: Gradient<Dim = Ix1> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixVectorMul<F1, F2>, MatrixVectorMulBackward<F1, B1, F2, B2>>;

    fn mv(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = MatrixVectorMulBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.mv(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> VecMatMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = VarDiff<VectorMatrixMul<F1, F2>, VectorMatrixMulBackwardLeft<B1, F2>>;

    fn vm(self, rhs: Var<F2>) -> Self::Output {
        let node = VectorMatrixMulBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.vm(rhs))
    }
}

impl<F1, B1, F2, B2> VecMatMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<VectorMatrixMul<F1, F2>, VectorMatrixMulBackward<F1, B1, F2, B2>>;

    fn vm(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = VectorMatrixMulBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.vm(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> VecVecMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = VarDiff<VectorVectorMul<F1, F2>, VectorVectorMulBackwardUnary<B1, F2>>;

    fn vv(self, rhs: Var<F2>) -> Self::Output {
        let node = VectorVectorMulBackwardUnary::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.vv(rhs))
    }
}

impl<F1, B1, F2, B2> VecVecMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
    B2: Gradient<Dim = Ix1> + Overwrite + 'static,
{
    type Output = VarDiff<VectorVectorMul<F1, F2>, VectorVectorMulBackward<F1, B1, F2, B2>>;

    fn vv(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = VectorVectorMulBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.vv(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cat and Stack traits implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> Cat<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = B1::Dim> + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    B1: Gradient + Overwrite + 'static,
    F1::Dim: RemoveAxis,
    B1::Dim: RemoveAxis,
{
    type Output = VarDiff<Concatenate<F1, F2>, ConcatenateBackwardLeft<B1>>;

    fn cat(self, rhs: Var<F2>, axis: usize) -> Self::Output {
        let node = ConcatenateBackwardLeft::new(self.node, rhs.node.clone(), axis);
        VarDiff::from(node, self.past, Cat::cat(self.var, rhs, axis))
    }
}

impl<F1, B1, F2, B2> Cat<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = B1::Dim> + Overwrite + 'static,
    F1::Dim: RemoveAxis,
    B1::Dim: RemoveAxis,
{
    type Output = VarDiff<Concatenate<F1, F2>, ConcatenateBackward<B1, B2>>;

    fn cat(mut self, rhs: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        self.past.merge(rhs.past);
        let node = ConcatenateBackward::new(self.node, rhs.node, axis);
        VarDiff::from(node, self.past, Cat::cat(self.var, rhs.var, axis))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2> Stack<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = B1::Dim> + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    B1: Gradient + Overwrite + 'static,
    F1::Dim: RemoveAxis,
    B1::Dim: RemoveAxis,
{
    type Output = VarDiff<super::node::Stack<F1, F2>, StackBackwardLeft<B1>>;

    fn stack(self, rhs: Var<F2>, axis: usize) -> Self::Output {
        let node = StackBackwardLeft::new(self.node, rhs.node.clone(), axis);
        VarDiff::from(node, self.past, Stack::stack(self.var, rhs, axis))
    }
}

impl<F1, B1, F2, B2> Stack<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + 'static,
    B1: Gradient + Overwrite + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = B1::Dim> + Overwrite + 'static,
    F1::Dim: RemoveAxis,
    B1::Dim: RemoveAxis,
{
    type Output = VarDiff<super::node::Stack<F1, F2>, StackBackward<B1, B2>>;

    fn stack(mut self, rhs: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        self.past.merge(rhs.past);
        let node = StackBackward::new(self.node, rhs.node, axis);
        VarDiff::from(node, self.past, Stack::stack(self.var, rhs.var, axis))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Register ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T, U> Register for VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient + Overwrite + 'static,
{
    fn register_params(&self, params: &mut Vec<RawParam>) {
        params.extend(self.past.parameters.iter().cloned())
    }

    fn register_status(&mut self, _: Rc<Cell<bool>>) {}
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Debug ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T, U> Debug for VarDiff<T, U>
where
    T: Data + Debug,
    U: Gradient<Dim = T::Dim> + Overwrite + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarDiff")
            .field("var", &self.var)
            .field("node", &self.node)
            .field("past", &self.past.len())
            .field("parameters", &self.parameters().len())
            .finish()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Display ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T: Data + Display, U: Gradient + Overwrite + Display> Display for VarDiff<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.var)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Serialize ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(feature = "serialize")]
impl<D> Serialize for VarDiff<Input<D>, super::InputBackward<D>>
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
impl<'d, D> Deserialize<'d> for VarDiff<Input<D>, super::InputBackward<D>>
where
    D: Dimension + Deserialize<'d>,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: Deserializer<'d>,
    {
        let data = ndarray::Array::<f32, D>::deserialize(deserializer).unwrap();
        Ok(Input::new(data).requires_grad())
    }
}
