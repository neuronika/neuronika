use super::{
    cobroadcasted_zeros, node, Addition, AdditionBackwardRight, BufferedGradient, Cat, Chunk,
    Concatenate, ConcatenateBackwardRight, Division, DivisionBackwardRight, DotDim, Dropout, Exp,
    Forward, Gradient, History, LeakyReLU, LogSoftmax, Logn, MatMatMul, MatMatMulT, MatVecMul,
    MatrixMatrixMul, MatrixMatrixMulBackwardRight, MatrixMatrixMulT, MatrixMatrixMulTBackwardRight,
    MatrixVectorMul, MatrixVectorMulBackwardRight, Mean, MultiConcatenate, MultiStack,
    Multiplication, MultiplicationBackwardRight, Negation, Power, ReLU, Sigmoid, SoftPlus, Softmax,
    Sqrt, Stack, StackBackwardRight, Subtraction, SubtractionBackwardRight, Sum, TanH, Transpose,
    Unsqueeze, VarDiff, VecMatMul, VecVecMul, VectorMatrixMul, VectorMatrixMulBackwardRight,
    VectorVectorMul, VectorVectorMulBackwardUnary,
};
use ndarray::{
    arr0, concatenate, stack, Array, Axis, DimMax, Dimension, IntoDimension, Ix0, Ix1, Ix2,
    RemoveAxis,
};

use std::{
    cell::{Cell, Ref, RefCell, RefMut},
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
#[derive(Clone)]
pub struct Var<D>
where
    D: Dimension,
{
    pub(crate) data: Rc<RefCell<Array<f32, D>>>,
    pub(crate) history: History<(Rc<dyn Forward>, Cell<bool>)>,
}

impl<D> Var<D>
where
    D: Dimension,
{
    pub(crate) fn leaf(array: Array<f32, D>) -> Self {
        Self {
            data: Rc::new(RefCell::new(array)),
            history: History::default(),
        }
    }

    pub(crate) fn node(
        data: Rc<RefCell<Array<f32, D>>>,
        op: Rc<dyn Forward>,
        mut history: History<(Rc<dyn Forward>, Cell<bool>)>,
    ) -> Self {
        history.insert(Rc::as_ptr(&op) as *const () as usize, (op, Cell::default()));

        Self { data, history }
    }

    /// Returns an immutable reference to the data inside `self`.
    ///
    /// At the variable's creation the data is filled with zeros. You can populate it with a
    /// call to [`.forward()`](Var::forward()).
    pub fn data(&self) -> Ref<Array<f32, D>> {
        self.data.borrow()
    }

    /// Returns a mutable reference to the data inside `self`.
    ///
    /// At the variable's creation the data is filled with zeros. You can populate it with a
    /// call to [`.forward()`](Var::forward()).
    pub fn data_mut(&self) -> RefMut<Array<f32, D>> {
        self.data.borrow_mut()
    }

    /// Promotes `self` to a differentiable variable. A subsequent call to [`.backward()`]
    /// will compute its grad.
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
    pub fn requires_grad(self) -> VarDiff<D> {
        let grad = Array::zeros(self.data.borrow().raw_dim());
        VarDiff::leaf(self, grad)
    }

    /// Propagates the computations forwards and populates all the variables from the leaves of the
    /// graph to `self`.
    pub fn forward(&self) {
        let mut buffer = self.history.buffer_mut(); // Borrows for the scope

        // If the length of the buffer is greater than 0 it means that forward has already been
        // called and the path must be recomputed, else the buffer is empty and must be populated.
        if buffer.is_empty() {
            *buffer = self.history.to_vec()
        } else {
            buffer.iter().for_each(|(_, done)| done.set(false));
        }

        buffer
            .iter()
            .filter(|(_, done)| !done.get())
            .for_each(|(op, _)| op.forward());
    }
}

impl Var<Ix1> {
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

impl Var<Ix2> {
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

impl<D> Var<D>
where
    D: 'static + Dimension,
{
    /// Returns the sum of all elements in `self`.
    pub fn sum(self) -> Var<Ix0> {
        let data = Rc::new(RefCell::new(arr0(0.)));
        let op = Sum::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Returns the mean of all elements in `self`.
    pub fn mean(self) -> Var<Ix0> {
        let data = Rc::new(RefCell::new(arr0(0.)));
        let op = Mean::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Takes the power of each element in `self` with exponent `exp` and returns a variable with the
    /// result.
    ///
    /// # Arguments
    ///
    /// `exp` - exponent.
    pub fn pow(self, exp: i32) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = Power::new(self.data, data.clone(), exp);

        Var::node(data, Rc::new(op), self.history)
    }

    /// Takes the square root element-wise and returns a variable with the result.
    pub fn sqrt(self) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = Sqrt::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies the *rectified linear unit* element-wise and returns a variable with the
    /// result.
    ///
    /// *ReLU(x) = max(0, x)*
    pub fn relu(self) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = ReLU::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies the *leaky rectified linear unit* element-wise and returns a variable with
    /// the result.
    ///
    /// *LeakyReLU(x) = max(0, x) + 0.01 * min(0, x)*
    pub fn leaky_relu(self) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = LeakyReLU::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies the *softplus* element-wise and returns a variable with the result.
    ///
    /// *Softplus(x) = log(1 + exp(x))*
    pub fn softplus(self) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = SoftPlus::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies the *sigmoid* element-wise and returns a variable with the result.
    pub fn sigmoid(self) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = Sigmoid::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies the *tanh* element-wise and returns a variable with the result.
    pub fn tanh(self) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = TanH::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies the *natural logarithm* element-wise and returns a variable with the result.
    pub fn ln(self) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = Logn::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies the *exponential* element-wise and returns a variable with the result.
    pub fn exp(self) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = Exp::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies the *softmax* to `self` and returns a variable with the result.
    ///
    /// The *softmax* is applied to all slices along `axis`, and will re-scale them so
    ///  that the elements lie in the range *[0, 1]* and sum to 1.0.
    ///
    /// # Arguments    
    ///  
    /// `axis` - axis along which softmax will be computed.
    pub fn softmax(self, axis: usize) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = Softmax::new(self.data, data.clone(), axis);

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies the *log-softmax* to `self` and returns a variable with the result.
    ///
    /// Applies a softmax followed by a logarithm. While mathematically equivalent to
    /// *log(softmax(x))*, doing these two operations separately is slower, and numerically
    /// unstable. This function uses an alternative formulation to compute the output and
    /// grad correctly.
    ///
    /// See also [`.softmax()`].
    ///
    /// [`.softmax()`]: Var::softmax()
    ///
    /// # Arguments
    ///
    /// `axis` - axis along which log-softmax will be computed.
    pub fn log_softmax(self, axis: usize) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = LogSoftmax::new(self.data, data.clone(), axis);

        Var::node(data, Rc::new(op), self.history)
    }

    /// Returns a variable equivalent to `self` with its dimensions reversed.
    pub fn t(self) -> Var<D> {
        let shape = self.data.borrow().t().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = Transpose::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }

    /// Applies *dropout* to `self` and returns a variable with the result.
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
    pub fn dropout(self, p: f64, status: Rc<Cell<bool>>) -> Var<D> {
        let shape = self.data.borrow().raw_dim();
        let noise = Rc::new(RefCell::new(Array::zeros(shape.clone())));

        self.dropout_with_noise(shape, p, noise, status)
    }

    pub(crate) fn dropout_with_noise(
        self,
        shape: D,
        p: f64,
        noise: Rc<RefCell<Array<f32, D>>>,
        status: Rc<Cell<bool>>,
    ) -> Var<D> {
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = Dropout::new(self.data, data.clone(), p, noise, status);

        Var::node(data, Rc::new(op), self.history)
    }

    /// Splits `self` into a certain number of chunks of size `chunk_size` **skipping** the
    /// remainder along each dimension that doesn’t fit evenly.
    pub fn chunks<E>(self, chunk_size: E) -> Vec<Var<D>>
    where
        E: IntoDimension<Dim = D>,
    {
        self.data
            .borrow()
            .exact_chunks(chunk_size)
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let data = Rc::new(RefCell::new(chunk.to_owned()));
                let op = Chunk::new(self.data.clone(), data.clone(), i);

                Var::node(data, Rc::new(op), self.history.clone())
            })
            .collect()
    }

    /// Returns a new variable with a dimension of size one inserted at the position specified by
    /// `axis`.
    pub fn unsqueeze(self, axis: usize) -> Var<D::Larger> {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape.insert_axis(Axis(axis)))));
        let op = Unsqueeze::new(self.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }
}

impl<D> Var<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    /// Concatenates the given sequence of non-differentiable variables, including self`, along
    /// the given axis, and returns a non-differentiable variable with the results.
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
    /// use neuronika;
    /// use ndarray;
    ///
    ///
    /// let a = neuronika::ones((3, 2));
    /// let b = neuronika::full((3, 2), 4.);
    /// let c = neuronika::full((3, 2), 3.);
    ///
    /// let mut d = a.cat(&[b, c], 1);
    /// d.forward();
    ///
    /// assert_eq!(*d.data(), ndarray::array![[1., 1., 4., 4., 3., 3.],
    ///                                       [1., 1., 4., 4., 3., 3.],
    ///                                       [1., 1., 4., 4., 3., 3.]]);
    /// ```
    pub fn cat(mut self, variables: &[Self], axis: usize) -> Var<D> {
        let mut operands_data = Vec::with_capacity(variables.len());
        operands_data.push(self.data);

        variables.iter().cloned().for_each(|variable| {
            self.history.merge(variable.history);
            operands_data.push(variable.data);
        });

        let data = {
            let tensors: Vec<Ref<Array<f32, D>>> = operands_data
                .iter()
                .map(|operand| operand.borrow())
                .collect();
            let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();
            Rc::new(RefCell::new(concatenate(Axis(axis), &views).unwrap()))
        };
        let op = MultiConcatenate::new(operands_data, data.clone(), axis);

        Var::node(data, Rc::new(op), self.history)
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
    /// use neuronika;
    /// use ndarray;
    ///
    ///
    /// let a = neuronika::ones((2, 2));
    /// let b = neuronika::ones((2, 2));
    /// let c = neuronika::ones((2, 2));
    ///
    /// let mut d = a.stack(&[b, c], 0);
    /// d.forward();
    ///
    /// assert_eq!(*d.data(), ndarray::array![[[1., 1.],
    ///                                        [1., 1.]],
    ///                                       [[1., 1.],
    ///                                        [1., 1.]],
    ///                                       [[1., 1.],
    ///                                        [1., 1.]]]);
    /// ```
    pub fn stack(mut self, variables: &[Self], axis: usize) -> Var<D::Larger> {
        let mut operands_data = Vec::with_capacity(variables.len());
        operands_data.push(self.data);

        variables.iter().cloned().for_each(|variable| {
            self.history.merge(variable.history);
            operands_data.push(variable.data);
        });

        let data = {
            let tensors: Vec<Ref<Array<f32, D>>> = operands_data
                .iter()
                .map(|operand| operand.borrow())
                .collect();
            let views: Vec<_> = tensors.iter().map(|tensor| tensor.view()).collect();
            Rc::new(RefCell::new(stack(Axis(axis), &views).unwrap()))
        };
        let op = Rc::new(MultiStack::new(operands_data, data.clone(), axis));

        Var::node(data, op, self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arithmetic Operations Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Var - f32 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Add<f32> for Var<D>
where
    D: 'static + Dimension + DimMax<Ix0>,
{
    type Output = Var<<D as DimMax<Ix0>>::Output>;

    fn add(self, rhs: f32) -> Self::Output {
        self + crate::full((), rhs)
    }
}

impl<D> Sub<f32> for Var<D>
where
    D: 'static + Dimension + DimMax<Ix0>,
{
    type Output = Var<<D as DimMax<Ix0>>::Output>;

    fn sub(self, rhs: f32) -> Self::Output {
        self - crate::full((), rhs)
    }
}

impl<D> Mul<f32> for Var<D>
where
    D: 'static + Dimension + DimMax<Ix0>,
{
    type Output = Var<<D as DimMax<Ix0>>::Output>;

    fn mul(self, rhs: f32) -> Self::Output {
        self * crate::full((), rhs)
    }
}

impl<D> Div<f32> for Var<D>
where
    D: 'static + Dimension + DimMax<Ix0>,
{
    type Output = Var<<D as DimMax<Ix0>>::Output>;

    fn div(self, rhs: f32) -> Self::Output {
        self / crate::full((), rhs)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ f32 - Var ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Add<Var<D>> for f32
where
    D: 'static + Dimension,
    Ix0: DimMax<D>,
{
    type Output = Var<<Ix0 as DimMax<D>>::Output>;

    fn add(self, rhs: Var<D>) -> Self::Output {
        crate::full((), self) + rhs
    }
}

impl<D> Sub<Var<D>> for f32
where
    D: 'static + Dimension,
    Ix0: DimMax<D>,
{
    type Output = Var<<Ix0 as DimMax<D>>::Output>;

    fn sub(self, rhs: Var<D>) -> Self::Output {
        crate::full((), self) - rhs
    }
}

impl<D> Mul<Var<D>> for f32
where
    D: 'static + Dimension,
    Ix0: DimMax<D>,
{
    type Output = Var<<Ix0 as DimMax<D>>::Output>;

    fn mul(self, rhs: Var<D>) -> Self::Output {
        crate::full((), self) * rhs
    }
}

impl<D> Div<Var<D>> for f32
where
    D: 'static + Dimension,
    Ix0: DimMax<D>,
{
    type Output = Var<<Ix0 as DimMax<D>>::Output>;

    fn div(self, rhs: Var<D>) -> Self::Output {
        crate::full((), self) / rhs
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Neg for Var<D>
where
    D: 'static + Dimension,
{
    type Output = Var<D>;

    fn neg(self) -> Self::Output {
        let shape = self.data.borrow().raw_dim();
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = Rc::new(Negation::new(self.data, data.clone()));

        Var::node(data, op, self.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Addition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D, E> Add<Var<E>> for Var<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = Var<<D as DimMax<E>>::Output>;

    fn add(mut self, rhs: Var<E>) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(cobroadcasted_zeros(
            &self.data.borrow(),
            &rhs.data.borrow(),
        )));
        let op = Rc::new(Addition::new(self.data, rhs.data, data.clone()));

        Var::node(data, op, self.history)
    }
}

impl<D, E> Add<VarDiff<E>> for Var<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn add(self, rhs: VarDiff<E>) -> Self::Output {
        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let op = AdditionBackwardRight::<D, E>::new(rhs.grad, grad.clone());
        let var = self.add(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Subtraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D, E> Sub<Var<E>> for Var<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = Var<<D as DimMax<E>>::Output>;

    fn sub(mut self, rhs: Var<E>) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(cobroadcasted_zeros(
            &self.data.borrow(),
            &rhs.data.borrow(),
        )));
        let op = Subtraction::new(self.data, rhs.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }
}

impl<D, E> Sub<VarDiff<E>> for Var<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn sub(self, rhs: VarDiff<E>) -> Self::Output {
        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let op = SubtractionBackwardRight::<D, E>::new(rhs.grad, grad.clone());
        let var = self.sub(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D, E> Mul<Var<E>> for Var<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = Var<<D as DimMax<E>>::Output>;

    fn mul(mut self, rhs: Var<E>) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(cobroadcasted_zeros(
            &self.data.borrow(),
            &rhs.data.borrow(),
        )));
        let op = Multiplication::new(self.data, rhs.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }
}

impl<D, E> Mul<VarDiff<E>> for Var<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn mul(self, rhs: VarDiff<E>) -> Self::Output {
        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let buff = Rc::new(BufferedGradient::from_gradient(grad.clone()));
        let op = MultiplicationBackwardRight::new(rhs.grad, self.data.clone(), buff.clone());
        let var = self.mul(rhs.var);

        VarDiff::node(var, grad, (Rc::new(op), buff), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Division ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D, E> Div<Var<E>> for Var<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = Var<<D as DimMax<E>>::Output>;

    fn div(mut self, rhs: Var<E>) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(cobroadcasted_zeros(
            &self.data.borrow(),
            &rhs.data.borrow(),
        )));
        let op = Division::new(self.data, rhs.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }
}

impl<D, E> Div<VarDiff<E>> for Var<D>
where
    D: 'static + Dimension + DimMax<E>,
    E: 'static + Dimension,
{
    type Output = VarDiff<<D as DimMax<E>>::Output>;

    fn div(self, rhs: VarDiff<E>) -> Self::Output {
        let grad = Rc::new(Gradient::from_ndarray(cobroadcasted_zeros(
            &self.data.borrow(),
            &rhs.var.data.borrow(),
        )));
        let buff = Rc::new(BufferedGradient::from_gradient(grad.clone()));
        let op = DivisionBackwardRight::new(
            self.data.clone(),
            rhs.var.data.clone(),
            rhs.grad,
            buff.clone(),
        );
        let var = self.div(rhs.var);

        VarDiff::node(var, grad, (Rc::new(op), buff), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Operations Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MatMatMul<Var<Ix2>> for Var<Ix2> {
    type Output = Var<Ix2>;

    fn mm(mut self, rhs: Var<Ix2>) -> Self::Output {
        self.history.merge(rhs.history);

        let shape = DotDim::shape(self.data.borrow().raw_dim(), rhs.data.borrow().raw_dim());
        let data = Rc::new(RefCell::new(Array::zeros(shape)));
        let op = MatrixMatrixMul::new(self.data, rhs.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }
}

impl MatMatMul<VarDiff<Ix2>> for Var<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm(self, rhs: VarDiff<Ix2>) -> Self::Output {
        let grad = Rc::new(Gradient::zeros(DotDim::shape(
            self.data.borrow().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let op = MatrixMatrixMulBackwardRight::new(self.data.clone(), rhs.grad, grad.clone());
        let var = self.mm(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition  ~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MatMatMulT<Var<Ix2>> for Var<Ix2> {
    type Output = Var<Ix2>;

    fn mm_t(mut self, rhs: Var<Ix2>) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(Array::zeros(DotDim::shape(
            self.data.borrow().raw_dim(),
            rhs.data.borrow().t().raw_dim(),
        ))));
        let op = MatrixMatrixMulT::new(self.data, rhs.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }
}

impl MatMatMulT<VarDiff<Ix2>> for Var<Ix2> {
    type Output = VarDiff<Ix2>;

    fn mm_t(self, rhs: VarDiff<Ix2>) -> Self::Output {
        let grad = Rc::new(Gradient::zeros(DotDim::shape(
            self.data.borrow().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let op = MatrixMatrixMulTBackwardRight::new(self.data.clone(), rhs.grad, grad.clone());
        let var = self.mm_t(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl MatVecMul<Var<Ix1>> for Var<Ix2> {
    type Output = Var<Ix1>;

    fn mv(mut self, rhs: Var<Ix1>) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(Array::zeros(DotDim::shape(
            self.data.borrow().raw_dim(),
            rhs.data.borrow().raw_dim(),
        ))));
        let op = MatrixVectorMul::new(self.data, rhs.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }
}

impl MatVecMul<VarDiff<Ix1>> for Var<Ix2> {
    type Output = VarDiff<Ix1>;

    fn mv(self, rhs: VarDiff<Ix1>) -> Self::Output {
        let grad = Rc::new(Gradient::zeros(DotDim::shape(
            self.data.borrow().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let op = MatrixVectorMulBackwardRight::new(self.data.clone(), rhs.grad, grad.clone());
        let var = self.mv(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl VecMatMul<Var<Ix2>> for Var<Ix1> {
    type Output = Var<Ix1>;

    fn vm(mut self, rhs: Var<Ix2>) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(Array::zeros(DotDim::shape(
            self.data.borrow().raw_dim(),
            rhs.data.borrow().raw_dim(),
        ))));
        let op = VectorMatrixMul::new(self.data, rhs.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }
}

impl VecMatMul<VarDiff<Ix2>> for Var<Ix1> {
    type Output = VarDiff<Ix1>;

    fn vm(self, rhs: VarDiff<Ix2>) -> Self::Output {
        let grad = Rc::new(Gradient::zeros(DotDim::shape(
            self.data.borrow().raw_dim(),
            rhs.var.data().raw_dim(),
        )));
        let op = VectorMatrixMulBackwardRight::new(self.data.clone(), rhs.grad, grad.clone());
        let var = self.vm(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl VecVecMul<Var<Ix1>> for Var<Ix1> {
    type Output = Var<Ix0>;

    fn vv(mut self, rhs: Var<Ix1>) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(arr0(0.)));
        let op = VectorVectorMul::new(self.data, rhs.data, data.clone());

        Var::node(data, Rc::new(op), self.history)
    }
}

impl VecVecMul<VarDiff<Ix1>> for Var<Ix1> {
    type Output = VarDiff<Ix0>;

    fn vv(self, rhs: VarDiff<Ix1>) -> Self::Output {
        let grad = Rc::new(Gradient::from_ndarray(Array::zeros(())));
        let op = VectorVectorMulBackwardUnary::new(self.data.clone(), rhs.grad, grad.clone());
        let var = self.vv(rhs.var);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cat and Stack traits implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Cat<Var<D>> for Var<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = Var<D>;

    fn cat(mut self, rhs: Var<D>, axis: usize) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(
            concatenate(
                Axis(axis),
                &[
                    Array::zeros(self.data.borrow().raw_dim()).view(),
                    Array::zeros(rhs.data.borrow().raw_dim()).view(),
                ],
            )
            .unwrap(),
        ));
        let op = Concatenate::new(self.data, rhs.data, data.clone(), axis);

        Var::node(data, Rc::new(op), self.history)
    }
}

impl<D> Cat<VarDiff<D>> for Var<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = VarDiff<D>;

    fn cat(self, rhs: VarDiff<D>, axis: usize) -> Self::Output {
        let array = concatenate(
            Axis(axis),
            &[
                Array::zeros(self.data.borrow().raw_dim()).view(),
                Array::zeros(rhs.var.data.borrow().raw_dim()).view(),
            ],
        )
        .unwrap();
        let grad = Rc::new(Gradient::from_ndarray(array));
        let offset = self.data.borrow().len_of(Axis(axis));
        let op = ConcatenateBackwardRight::new(rhs.grad, grad.clone(), axis, offset);
        let var = Cat::cat(self, rhs.var, axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Stack<Var<D>> for Var<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = Var<D::Larger>;

    fn stack(mut self, rhs: Var<D>, axis: usize) -> Self::Output {
        self.history.merge(rhs.history);

        let data = Rc::new(RefCell::new(
            stack(
                Axis(axis),
                &[
                    Array::zeros(self.data.borrow().raw_dim()).view(),
                    Array::zeros(rhs.data.borrow().raw_dim()).view(),
                ],
            )
            .unwrap(),
        ));
        let op = node::Stack::new(self.data, rhs.data, data.clone(), axis);

        Var::node(data, Rc::new(op), self.history)
    }
}

impl<D> Stack<VarDiff<D>> for Var<D>
where
    D: 'static + Dimension + RemoveAxis,
{
    type Output = VarDiff<D::Larger>;

    fn stack(self, rhs: VarDiff<D>, axis: usize) -> Self::Output {
        let array = stack(
            Axis(axis),
            &[
                Array::zeros(self.data.borrow().raw_dim()).view(),
                Array::zeros(rhs.var.data.borrow().raw_dim()).view(),
            ],
        )
        .unwrap();
        let grad = Rc::new(Gradient::from_ndarray(array));
        let op = StackBackwardRight::new(rhs.grad, grad.clone(), axis);
        let var = Stack::stack(self, rhs.var, axis);

        VarDiff::node(var, grad.clone(), (Rc::new(op), grad), rhs.history)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Debug ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Debug for Var<D>
where
    D: 'static + Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.data.borrow(), f)
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Display ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<D> Display for Var<D>
where
    D: 'static + Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.data.borrow())
    }
}
