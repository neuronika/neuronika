use super::{
    Addition, AdditionBackwardUnary, Cat, Changeable, Chunk, Concatenate, ConcatenateBackwardRight,
    Data, Division, DivisionBackwardRight, Dropout, Eval, Exp, Forward, Gradient, Input,
    InputBackward, LeakyReLU, LogSoftmax, Logn, MatMatMul, MatMatMulT, MatVecMul, MatrixMatrixMul,
    MatrixMatrixMulBackwardRight, MatrixMatrixMulT, MatrixMatrixMulTBackwardRight, MatrixVectorMul,
    MatrixVectorMulBackwardRight, Mean, MultiConcatenate, MultiStack, Multiplication,
    MultiplicationBackwardUnary, Negation, Overwrite, Param, Power, ReLU, Sigmoid, SoftPlus,
    Softmax, Sqrt, Stack, StackBackwardRight, Subtraction, SubtractionBackwardRight, Sum, TanH,
    Tensor, Transpose, Unsqueeze, VarDiff, VarDiffHistory, VarHistory, Variable, VecMatMul,
    VecVecMul, VectorMatrixMul, VectorMatrixMulBackwardRight, VectorVectorMul,
    VectorVectorMulBackwardUnary, OPERATIONS_COUNTER,
};
use ndarray::{concatenate, stack, Axis, DimMax, Dimension, IntoDimension, Ix1, Ix2, RemoveAxis};
use std::{
    cell::{Cell, Ref, RefMut},
    collections::HashSet,
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

/// A non-differentiable variable.
///
/// This, together with its differentiable counterpart [`VarDiff`], is the main building block of
/// every computation.
///
/// Conceptually, it can be thought of as a [`ndarray::Array`] for which the computations are
/// automatically kept track of.
pub struct Var<T: Data + 'static> {
    pub(crate) node: Rc<T>,
    pub(crate) past: VarHistory,
}

impl<T: Data + 'static> Clone for Var<T> {
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            past: self.past.clone(),
        }
    }
}

impl<D: Dimension> Var<Input<D>> {
    /// Promotes `self` to a differentiable variable. A subsequent call to [`.backward()`]
    /// will compute its gradient.
    ///
    /// [`.backward()`]: VarDiff::backward()
    ///
    /// # Examples
    ///
    /// This is the preferred usage.
    ///
    ///```
    /// use neuronika;
    ///
    /// let x = neuronika::ones(5).requires_grad();
    ///```
    ///
    /// This is also permitted, however, one should be aware of the difference between `x_diff` and
    /// `x`.
    ///
    ///```
    /// use neuronika;
    ///
    /// let x = neuronika::ones(5);
    /// let y = x.clone() + neuronika::ones(1);
    ///
    /// let x_diff = x.requires_grad();
    ///```
    pub fn requires_grad(self) -> VarDiff<Input<D>, InputBackward<D>> {
        debug_assert!(self.past.is_empty(), "error: the variable is not a leaf.");
        let node = Rc::new(self.node.differentiable());
        let mut gradient = node.gradient_mut();
        let mut parameters = HashSet::new();
        parameters.insert(Param::new(
            self.node.data_mut().as_mut_ptr(),
            gradient.as_mut_ptr(),
            gradient.shape().to_vec(),
        ));

        VarDiff {
            var: self,
            node: node.clone(),
            past: VarDiffHistory::new(parameters),
        }
    }

    /// Assigns `array` to the variable's data.
    ///
    /// # Arguments
    ///
    /// `array` - new content.
    pub fn assign<S: ndarray::Data<Elem = f32>>(&self, array: &ndarray::ArrayBase<S, D>) {
        self.node.data_mut().assign(array)
    }
}

impl<T: Data + Forward + 'static> Var<T> {
    /// Propagates the computations forwards and populates all the variables from the leaves of the
    /// graph to `self`.
    pub fn forward(&mut self) {
        if self.node.was_computed() {
            // If the user has already called `.forward()` on this var,
            // then he wants to recompute it.
            assert_eq!(self.past.len(), self.past.buffer().len());
            for node in self.past.buffer() {
                node.reset_computation();
            }
        }

        self.past.prepare_buffer();

        let buffer = self.past.buffer();
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
                node.forward();
            }
        }
    }

    pub(crate) fn from(node: T, mut past: VarHistory) -> Self {
        let node = Rc::new(node);
        past.append_forward(unsafe { OPERATIONS_COUNTER.next() }, node.clone());

        Var { node, past }
    }

    /// This has effect only on certain **ancestor** variables of `self`. It sets such variables
    /// in training mode.
    ///    
    /// See also [`.dropout()`].
    ///
    /// [`.dropout()`]: Var::dropout()
    ///
    /// # Examples
    ///
    /// The following snippet pictures the effect of several calls placed at different locations
    /// inside the program. The last call switches all the dropout variables in training mode.
    pub fn train(&self) {
        for changeable in &self.past.changeables {
            let Changeable { id: _, node } = changeable;
            node.train();
        }
    }

    /// This has effect only on certain **ancestor** variables of `self`. It sets such variables
    /// in evaluation mode.
    ///    
    /// See also [`.dropout()`].
    ///
    /// [`.dropout()`]: Var::dropout()
    pub fn eval(&self) {
        for changeable in &self.past.changeables {
            let Changeable { id: _, node } = changeable;
            node.train();
        }
    }
}

impl<T: Data + Forward + Eval + 'static> Var<T> {
    pub(crate) fn from_changeable(node: T, mut past: VarHistory) -> Self {
        let node = Rc::new(node);
        let id = unsafe { OPERATIONS_COUNTER.next() };
        past.append_forward(id, node.clone());
        past.append_changeable(Changeable {
            id,
            node: node.clone(),
        });

        Var { node, past }
    }
}

impl<T: Data<Dim = Ix1> + 'static> Var<T> {
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

impl<T: Data<Dim = Ix2> + 'static> Var<T> {
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

impl<T: Data + 'static> Var<T> {
    pub(crate) fn new(node: T) -> Self {
        Self {
            node: Rc::new(node),
            past: VarHistory::new(),
        }
    }

    /// Returns an immutable reference to the data inside `self`.
    ///
    /// At the variable's creation the data is filled with zeros. You can populate it with a
    /// call to [`.forward()`](Var::forward()).
    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.node.data()
    }

    /// Returns a mutable reference to the data inside `self`.
    ///
    /// At the variable's creation the data is filled with zeros. You can populate it with a
    /// call to [`.forward()`](Var::forward()).
    pub fn data_mut(&self) -> RefMut<Tensor<T::Dim>> {
        self.node.data_mut()
    }

    /// Returns the sum of all elements in `self`.
    pub fn sum(self) -> Var<Sum<T>> {
        Var::from(Sum::new(self.node), self.past)
    }

    /// Returns the mean of all elements in `self`.
    pub fn mean(self) -> Var<Mean<T>> {
        Var::from(Mean::new(self.node), self.past)
    }

    /// Takes the power of each element in `self` with exponent `exp` and returns a variable with the
    /// result.
    pub fn pow(self, exp: i32) -> Var<Power<T>> {
        Var::from(Power::new(self.node, exp), self.past)
    }

    /// Takes the square root element-wise and returns a variable with the result.
    pub fn sqrt(self) -> Var<Sqrt<T>> {
        Var::from(Sqrt::new(self.node), self.past)
    }

    /// Applies the *rectified linear unit* element-wise and returns a variable with the
    /// result.
    ///
    /// *ReLU(x) = max(0, x)*
    pub fn relu(self) -> Var<ReLU<T>> {
        Var::from(ReLU::new(self.node), self.past)
    }

    /// Applies the *leaky rectified linear unit* element-wise and returns a variable with
    /// the result.
    ///
    /// *LeakyReLU(x) = max(0, x) + 0.01 * min(0, x)*
    pub fn leaky_relu(self) -> Var<LeakyReLU<T>> {
        Var::from(LeakyReLU::new(self.node), self.past)
    }

    /// Applies the *softplus* element-wise and returns a variable with the result.
    ///
    /// *Softplus(x) = log(1 + exp(x))*
    pub fn softplus(self) -> Var<SoftPlus<T>> {
        Var::from(SoftPlus::new(self.node), self.past)
    }

    /// Applies the *sigmoid* element-wise and returns a variable with the result.
    pub fn sigmoid(self) -> Var<Sigmoid<T>> {
        Var::from(Sigmoid::new(self.node), self.past)
    }

    /// Applies the *tanh* element-wise and returns a variable with the result.
    pub fn tanh(self) -> Var<TanH<T>> {
        Var::from(TanH::new(self.node), self.past)
    }

    /// Applies the *natural logarithm* element-wise and returns a variable with the result.
    pub fn ln(self) -> Var<Logn<T>> {
        Var::from(Logn::new(self.node), self.past)
    }

    /// Applies the *exponential* element-wise and returns a variable with the result.
    pub fn exp(self) -> Var<Exp<T>> {
        Var::from(Exp::new(self.node), self.past)
    }

    /// Applies the *softmax* to `self` and returns a variable with the result.
    ///
    /// The *softmax* is applied to all slices along `axis`, and will re-scale them so
    ///  that the elements lie in the range *[0, 1]* and sum to 1.0.
    pub fn softmax(self, axis: usize) -> Var<Softmax<T>> {
        Var::from(Softmax::new(self.node, axis), self.past)
    }

    /// Applies the *log-softmax* to `self` and returns a variable with the result.
    ///
    /// Applies a softmax followed by a logarithm. While mathematically equivalent to
    /// *log(softmax(x))*, doing these two operations separately is slower, and numerically
    /// unstable. This function uses an alternative formulation to compute the output and
    /// gradient correctly.
    ///
    /// See also [`.softmax()`].
    ///
    /// [`.softmax()`]: Var::softmax()
    pub fn log_softmax(self, axis: usize) -> Var<LogSoftmax<T>> {
        Var::from(LogSoftmax::new(self.node, axis), self.past)
    }

    /// Returns a variable equivalent to `self` with its dimensions reversed.
    pub fn t(self) -> Var<Transpose<T>> {
        Var::from(Transpose::new(self.node), self.past)
    }

    /// Applies *dropout* to `self` and returns a variable with the result.
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
    pub fn dropout(self, p: f64) -> Var<Dropout<T>> {
        self.dropout_with_status(p, Rc::new(Cell::new(true)))
    }

    /// Creates a new dropout variable with a status. This method is used in the `Dropout` component
    ///  of the `nn` module.
    pub(crate) fn dropout_with_status(self, p: f64, status: Rc<Cell<bool>>) -> Var<Dropout<T>> {
        Var::from_changeable(Dropout::new(self.node, p, status), self.past)
    }

    /// Splits `self` into a certain number of chunks of size `chunk_size` **skipping** the
    /// remainder along each dimension that doesn’t fit evenly.
    pub fn chunks<E: IntoDimension<Dim = T::Dim>>(self, chunk_size: E) -> Vec<Var<Chunk<T>>> {
        self.node
            .data()
            .exact_chunks(chunk_size)
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                Var::from(
                    Chunk::new(self.node.clone(), chunk.to_owned(), i),
                    self.past.clone(),
                )
            })
            .collect()
    }
}

impl<T> Var<T>
where
    T: Data + 'static,
    T::Dim: RemoveAxis,
{
    /// Returns a new variable with a dimension of size one inserted at the position specified by
    /// `axis`.
    pub fn unsqueeze(self, axis: usize) -> Var<Unsqueeze<T>> {
        Var::from(Unsqueeze::new(self.node, axis), self.past)
    }

    /// Concatenates the given sequence of non-differentiable variables `variables`, including
    /// `self`, along the given axis, and returns a non-differentiable variable with the results.
    ///
    /// # Arguments
    ///
    /// * `variables` - sequence of non-differentiable variables.
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
    /// let a = neuronika::ones((3, 2));
    /// let b = neuronika::full((3, 2), 4.);
    /// let c = neuronika::full((3, 2), 3.);
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
        variables: &[Box<dyn Variable<T::Dim>>],
        axis: usize,
    ) -> Var<MultiConcatenate<T::Dim>> {
        let mut operands: Vec<Rc<dyn Data<Dim = T::Dim>>> = Vec::with_capacity(variables.len() + 1);
        operands.push(self.node.clone());

        variables.iter().for_each(|variable| {
            self.past.merge(variable.get_past());
            operands.push(variable.get_node());
        });

        let data = {
            let tensors: Vec<Ref<Tensor<T::Dim>>> =
                operands.iter().map(|operand| operand.data()).collect();
            let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();
            concatenate(Axis(axis), &views).unwrap()
        };

        Var::from(MultiConcatenate::new(operands, axis, data), self.past)
    }

    /// Stacks the given sequence of non-differentiable variables `variables`, including
    /// `self`, along the given axis, and returns a non-differentiable variable with the results.
    ///
    /// All variables must have the same shape.
    ///
    /// # Arguments
    ///
    /// * `variables` - sequence of non-differentiable variables.
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
    /// let a = neuronika::ones((2, 2));
    /// let b = neuronika::ones((2, 2));
    /// let c = neuronika::ones((2, 2));
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
        variables: &[Box<dyn Variable<T::Dim>>],
        axis: usize,
    ) -> Var<MultiStack<T::Dim>> {
        let mut operands: Vec<Rc<dyn Data<Dim = T::Dim>>> = Vec::with_capacity(variables.len() + 1);
        operands.push(self.node.clone());

        variables.iter().for_each(|variable| {
            self.past.merge(variable.get_past());
            operands.push(variable.get_node());
        });

        let data = {
            let tensors: Vec<Ref<Tensor<T::Dim>>> =
                operands.iter().map(|operand| operand.data()).collect();
            let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();
            stack(Axis(axis), &views).unwrap()
        };
        Var::from(MultiStack::new(operands, axis, data), self.past)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arithmetic Operations Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Var - f32 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T> Add<f32> for Var<T>
where
    T: Data + Forward + 'static,
    T::Dim: DimMax<Ix1>,
{
    type Output = Var<Addition<T, Input<Ix1>>>;

    fn add(self, rhs: f32) -> Self::Output {
        self + crate::full(1, rhs)
    }
}

impl<T> Sub<f32> for Var<T>
where
    T: Data + Forward + 'static,
    T::Dim: DimMax<Ix1>,
{
    type Output = Var<Subtraction<T, Input<Ix1>>>;

    fn sub(self, rhs: f32) -> Self::Output {
        self - crate::full(1, rhs)
    }
}

impl<T> Mul<f32> for Var<T>
where
    T: Data + Forward + 'static,
    T::Dim: DimMax<Ix1>,
{
    type Output = Var<Multiplication<T, Input<Ix1>>>;

    fn mul(self, rhs: f32) -> Self::Output {
        self * crate::full(1, rhs)
    }
}

impl<T> Div<f32> for Var<T>
where
    T: Data + Forward + 'static,
    T::Dim: DimMax<Ix1>,
{
    type Output = Var<Division<T, Input<Ix1>>>;

    fn div(self, rhs: f32) -> Self::Output {
        self / crate::full(1, rhs)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ f32 - Var ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T> Add<Var<T>> for f32
where
    T: Data + Forward + 'static,
    Ix1: DimMax<T::Dim>,
{
    type Output = Var<Addition<Input<Ix1>, T>>;

    fn add(self, rhs: Var<T>) -> Self::Output {
        crate::full(1, self) + rhs
    }
}

impl<T> Sub<Var<T>> for f32
where
    T: Data + Forward + 'static,
    Ix1: DimMax<T::Dim>,
{
    type Output = Var<Subtraction<Input<Ix1>, T>>;

    fn sub(self, rhs: Var<T>) -> Self::Output {
        crate::full(1, self) - rhs
    }
}

impl<T> Mul<Var<T>> for f32
where
    T: Data + Forward + 'static,
    Ix1: DimMax<T::Dim>,
{
    type Output = Var<Multiplication<Input<Ix1>, T>>;

    fn mul(self, rhs: Var<T>) -> Self::Output {
        crate::full(1, self) * rhs
    }
}

impl<T> Div<Var<T>> for f32
where
    T: Data + Forward + 'static,
    Ix1: DimMax<T::Dim>,
{
    type Output = Var<Division<Input<Ix1>, T>>;

    fn div(self, rhs: Var<T>) -> Self::Output {
        crate::full(1, self) / rhs
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T: Data + 'static> Neg for Var<T> {
    type Output = Var<Negation<T>>;

    fn neg(self) -> Self::Output {
        Var::from(Negation::new(self.node), self.past)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Addition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<Lhs, Rhs> Add<Var<Rhs>> for Var<Lhs>
where
    Lhs: Data + 'static,
    Rhs: Data + 'static,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Output = Var<Addition<Lhs, Rhs>>;

    fn add(mut self, rhs: Var<Rhs>) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(Addition::new(self.node.clone(), rhs.node), self.past)
    }
}

impl<F1, F2, B2> Add<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B2: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B2::Dim: Dimension + DimMax<F1::Dim>,
{
    type Output = VarDiff<Addition<F1, F2>, AdditionBackwardUnary<B2, F1>>;

    fn add(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = AdditionBackwardUnary::new(rhs.node, self.node.clone());
        VarDiff::from(node, rhs.past, self.add(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Subtraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<Lhs, Rhs> Sub<Var<Rhs>> for Var<Lhs>
where
    Lhs: Data + 'static,
    Rhs: Data + 'static,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Output = Var<Subtraction<Lhs, Rhs>>;

    fn sub(mut self, rhs: Var<Rhs>) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(Subtraction::new(self.node, rhs.node), self.past)
    }
}

impl<F1, F2, B2> Sub<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B2: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B2::Dim: Dimension + DimMax<F1::Dim>,
{
    type Output = VarDiff<Subtraction<F1, F2>, SubtractionBackwardRight<B2, F1>>;

    fn sub(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = SubtractionBackwardRight::new(rhs.node, self.node.clone());
        VarDiff::from(node, rhs.past, self.sub(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<Lhs, Rhs> Mul<Var<Rhs>> for Var<Lhs>
where
    Lhs: Data + 'static,
    Rhs: Data + 'static,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Output = Var<Multiplication<Lhs, Rhs>>;

    fn mul(mut self, rhs: Var<Rhs>) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(Multiplication::new(self.node, rhs.node), self.past)
    }
}

impl<F1, F2, B2> Mul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B2: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    B2::Dim: Dimension + DimMax<F1::Dim>,
{
    type Output = VarDiff<Multiplication<F1, F2>, MultiplicationBackwardUnary<B2, F1>>;

    fn mul(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = MultiplicationBackwardUnary::new(rhs.node, self.node.clone());
        VarDiff::from(node, rhs.past, self.mul(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Division ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<Lhs, Rhs> Div<Var<Rhs>> for Var<Lhs>
where
    Lhs: Data + 'static,
    Rhs: Data + 'static,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Output = Var<Division<Lhs, Rhs>>;

    fn div(mut self, rhs: Var<Rhs>) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(Division::new(self.node, rhs.node), self.past)
    }
}

impl<F1, F2, B2> Div<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data + 'static,
    B2: Gradient + Overwrite + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    F1::Dim: Dimension + DimMax<B2::Dim>,
{
    type Output = VarDiff<Division<F1, F2>, DivisionBackwardRight<F1, F2, B2>>;

    fn div(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = DivisionBackwardRight::new(self.node.clone(), rhs.var.node.clone(), rhs.node);
        VarDiff::from(node, rhs.past, self.div(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Operations Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> MatMatMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = Var<MatrixMatrixMul<F1, F2>>;

    fn mm(mut self, rhs: Var<F2>) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(MatrixMatrixMul::new(self.node, rhs.node), self.past)
    }
}

impl<F1, F2, B2> MatMatMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixMatrixMul<F1, F2>, MatrixMatrixMulBackwardRight<F1, B2>>;

    fn mm(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = MatrixMatrixMulBackwardRight::new(self.node.clone(), rhs.node);
        VarDiff::from(node, rhs.past, self.mm(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition  ~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> MatMatMulT<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = Var<MatrixMatrixMulT<F1, F2>>;

    fn mm_t(mut self, rhs: Var<F2>) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(MatrixMatrixMulT::new(self.node, rhs.node), self.past)
    }
}

impl<F1, F2, B2> MatMatMulT<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixMatrixMulT<F1, F2>, MatrixMatrixMulTBackwardRight<F1, B2>>;

    fn mm_t(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = MatrixMatrixMulTBackwardRight::new(self.node.clone(), rhs.node);
        VarDiff::from(node, rhs.past, self.mm_t(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> MatVecMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = Var<MatrixVectorMul<F1, F2>>;

    fn mv(mut self, rhs: Var<F2>) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(MatrixVectorMul::new(self.node, rhs.node), self.past)
    }
}

impl<F1, F2, B2> MatVecMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix1> + 'static,
    B2: Gradient<Dim = Ix1> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixVectorMul<F1, F2>, MatrixVectorMulBackwardRight<F1, B2>>;

    fn mv(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = MatrixVectorMulBackwardRight::new(self.node.clone(), rhs.node);
        VarDiff::from(node, rhs.past, self.mv(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> VecMatMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = Var<VectorMatrixMul<F1, F2>>;

    fn vm(mut self, rhs: Var<F2>) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(VectorMatrixMul::new(self.node, rhs.node), self.past)
    }
}

impl<F1, F2, B2> VecMatMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<VectorMatrixMul<F1, F2>, VectorMatrixMulBackwardRight<F1, B2>>;

    fn vm(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = VectorMatrixMulBackwardRight::new(self.node.clone(), rhs.node);
        VarDiff::from(node, rhs.past, self.vm(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> VecVecMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = Var<VectorVectorMul<F1, F2>>;

    fn vv(mut self, rhs: Var<F2>) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(VectorVectorMul::new(self.node, rhs.node), self.past)
    }
}

impl<F1, F2, B2> VecVecMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix1> + 'static,
    B2: Gradient<Dim = Ix1> + Overwrite + 'static,
{
    type Output = VarDiff<VectorVectorMul<F1, F2>, VectorVectorMulBackwardUnary<B2, F1>>;

    fn vv(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = VectorVectorMulBackwardUnary::new(rhs.node, self.node.clone());
        VarDiff::from(node, rhs.past, self.vv(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cat and Stack traits implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> Cat<Var<F2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    F1::Dim: RemoveAxis,
{
    type Output = Var<Concatenate<F1, F2>>;

    fn cat(mut self, rhs: Var<F2>, axis: usize) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(Concatenate::new(self.node, rhs.node, axis), self.past)
    }
}

impl<F1, F2, B2> Cat<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = B2::Dim> + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    B2: Gradient + Overwrite + 'static,
    F1::Dim: RemoveAxis,
    B2::Dim: RemoveAxis,
{
    type Output = VarDiff<Concatenate<F1, F2>, ConcatenateBackwardRight<B2>>;

    fn cat(self, rhs: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        let node = ConcatenateBackwardRight::new(self.node.clone(), rhs.node, axis);
        VarDiff::from(node, rhs.past, Cat::cat(self, rhs.var, axis))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> Stack<Var<F2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    F1::Dim: RemoveAxis,
{
    type Output = Var<super::node::Stack<F1, F2>>;

    fn stack(mut self, rhs: Var<F2>, axis: usize) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(
            super::node::Stack::new(self.node, rhs.node, axis),
            self.past,
        )
    }
}

impl<F1, F2, B2> Stack<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = F1::Dim> + Overwrite + 'static,
    B2::Dim: RemoveAxis,
    F1::Dim: RemoveAxis,
{
    type Output = VarDiff<super::node::Stack<F1, F2>, StackBackwardRight<B2>>;

    fn stack(self, rhs: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        let node = StackBackwardRight::new(self.node.clone(), rhs.node, axis);
        VarDiff::from(node, rhs.past, Stack::stack(self, rhs.var, axis))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Debug ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T: Data + Debug> Debug for Var<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Var")
            .field("node", &self.node)
            .field("past", &self.past.len())
            .finish()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Display ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T: Data + Display> Display for Var<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.node)
    }
}
