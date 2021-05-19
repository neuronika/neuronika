pub mod node;

use ndarray::{
    Array, ArrayD, ArrayView, ArrayViewMutD, DimMax, Dimension, IntoDimension, Ix, Ix1, Ix2,
    RawArrayViewMut, RemoveAxis,
};
use node::{
    Addition, AdditionBackward, AdditionBackwardUnary, Backward, ChangeBehaviour, Chunk,
    ChunkBackward, Concatenate, ConcatenateBackward, ConcatenateBackwardLeft,
    ConcatenateBackwardRight, Data, Differentiable, Division, DivisionBackward,
    DivisionBackwardLeft, DivisionBackwardRight, Dropout, DropoutBackward, Exp, ExpBackward,
    Forward, Gradient, LeakyReLU, LeakyReLUBackward, LogSoftmax, LogSoftmaxBackward, Logn,
    LognBackward, MatrixMatrixMul, MatrixMatrixMulBackward, MatrixMatrixMulBackwardLeft,
    MatrixMatrixMulBackwardRight, MatrixMatrixMulT, MatrixMatrixMulTBackward,
    MatrixMatrixMulTBackwardLeft, MatrixMatrixMulTBackwardRight, MatrixVectorMul,
    MatrixVectorMulBackward, MatrixVectorMulBackwardLeft, MatrixVectorMulBackwardRight,
    Multiplication, MultiplicationBackward, MultiplicationBackwardUnary, Negation,
    NegationBackward, Overwrite, Power, PowerBackward, ReLU, ReLUBackward, Sigmoid,
    SigmoidBackward, SoftPlus, SoftPlusBackward, Softmax, SoftmaxBackward, Stack as StackF,
    StackBackward, StackBackwardLeft, StackBackwardRight, Subtraction, SubtractionBackward,
    SubtractionBackwardLeft, SubtractionBackwardRight, Sum, SumBackward, TanH, TanHBackward,
    Transpose, TransposeBackward, Unsqueeze, UnsqueezeBackward, VectorMatrixMul,
    VectorMatrixMulBackward, VectorMatrixMulBackwardLeft, VectorMatrixMulBackwardRight,
    VectorVectorMul, VectorVectorMulBackward, VectorVectorMulBackwardUnary,
};
use std::{
    cell::{Ref, RefCell, RefMut},
    collections::BTreeMap,
    collections::HashSet,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

pub use node::{Input, InputBackward};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Type Aliases ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;
pub(crate) type BroadTensor<Lhs, Rhs> = Tensor<Broadcasted<Lhs, Rhs>>;
pub(crate) type DynTensor = ArrayD<f32>;
pub(crate) type Tensor<D> = Array<f32, D>;
pub(crate) type TensorView<'a, D> = ArrayView<'a, f32, D>;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Global Var Identifier ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub(crate) struct OperationsCounter {
    count: usize,
}

impl OperationsCounter {
    pub fn next(&mut self) -> usize {
        self.count += 1;
        self.count
    }
}

#[derive(Clone)]
pub(crate) struct VarHistory {
    path: BTreeMap<usize, Rc<dyn Forward>>,
    buffer: Vec<Rc<dyn Forward>>,
    changeables: HashSet<*const dyn ChangeBehaviour>,
}

impl VarHistory {
    pub fn new() -> Self {
        Self {
            path: BTreeMap::new(),
            buffer: Vec::new(),
            changeables: HashSet::new(),
        }
    }

    pub fn merge(&mut self, mut other: VarHistory) {
        self.path.append(&mut other.path);
    }

    pub fn append_forward(&mut self, id: usize, next: Rc<dyn Forward>) {
        self.path.insert(id, next);
        self.buffer.truncate(0);
    }

    pub fn append_changeable(&mut self, next: *const dyn ChangeBehaviour) {
        self.changeables.insert(next);
    }

    pub fn len(&self) -> usize {
        self.path.len()
    }

    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }

    pub fn prepare_buffer(&mut self) {
        if self.buffer.is_empty() {
            self.buffer = self.path.values().cloned().collect();
        }
    }

    pub fn buffer(&self) -> &Vec<Rc<dyn Forward>> {
        &self.buffer
    }
}

#[derive(Clone)]
pub(crate) struct DiffVarHistory {
    path: BTreeMap<usize, Rc<dyn Backward>>,
    buffer: Vec<Rc<dyn Backward>>,
    parameters: HashSet<Param>,
    changeables: HashSet<*const dyn ChangeBehaviour>,
}

impl DiffVarHistory {
    pub fn new(parameters: HashSet<Param>) -> Self {
        Self {
            path: BTreeMap::new(),
            buffer: Vec::new(),
            parameters,
            changeables: HashSet::new(),
        }
    }

    pub fn merge(&mut self, mut other: DiffVarHistory) {
        self.path.append(&mut other.path);
        self.parameters.extend(other.parameters);
        self.changeables.extend(other.changeables);
    }

    pub fn append_backward(&mut self, id: usize, next: Rc<dyn Backward>) {
        self.path.insert(id, next);
        self.buffer.truncate(0);
    }

    pub fn append_changeable(&mut self, next: *const dyn ChangeBehaviour) {
        self.changeables.insert(next);
    }

    pub fn len(&self) -> usize {
        self.path.len()
    }

    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }

    pub fn prepare_buffer(&mut self) {
        if self.buffer.is_empty() {
            self.buffer = self.path.values().cloned().collect();
        }
    }

    pub fn buffer(&self) -> &Vec<Rc<dyn Backward>> {
        &self.buffer
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Param Struct ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// A builder of mutable views over a differentiable variable's data and gradient.
///
/// See also [`parameters()`] for more informations.
///
/// [`parameters()`]: VarDiff::parameters()
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Param {
    data: *mut f32,
    grad: *mut f32,
    shape: Vec<Ix>,
}

impl Param {
    pub(crate) fn new(data: *mut f32, grad: *mut f32, shape: Vec<Ix>) -> Self {
        Self { data, grad, shape }
    }

    /// Consumes the Param, yelding mutable views over the data and the gradient of the
    /// differentiable variable that it refers to. The lifetime `'a` is for the
    /// scope of the borrow.
    ///
    /// The views are [`ArrayViewMutD`].
    ///
    /// [`ArrayViewMutD`]: ndarray::ArrayViewMutD
    pub fn get<'a>(self) -> (ArrayViewMutD<'a, f32>, ArrayViewMutD<'a, f32>) {
        unsafe {
            (
                RawArrayViewMut::from_shape_ptr(self.shape.clone(), self.data)
                    .deref_into_view_mut(),
                RawArrayViewMut::from_shape_ptr(self.shape, self.grad).deref_into_view_mut(),
            )
        }
    }
}

pub(crate) static mut OPERATIONS_COUNTER: OperationsCounter = OperationsCounter { count: 0 };

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Matrix-matrix multiplication.
pub trait MatMatMul<Rhs> {
    type Output;

    /// Matrix-matrix multiplication.
    ///
    /// Performs a matrix multiplication between the matrix variables `self` and `other`. If `self`
    /// is *(n, m)* and `other` is *(m, o)* the output will be *(n, o)*.
    fn mm_mul(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition ~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Matrix-matrix multiplication with transpoed right hand side operand.
pub trait MatMatMulT<Rhs> {
    type Output;

    /// Matrix-matrix multiplication with transposition.
    ///
    /// Performs a matrix multiplication between the matrix variables `self` and `other`.
    /// This is a **fused operation** as `other` is implicitly transposed. Fusing the two operations
    /// it's marginally faster than computing the matrix multiplication and the transposition
    /// separately.
    ///
    /// If `self` is  *(n, m)* and `other` is *(o, m)* the output will be *(n, o)*.
    fn mm_mul_t(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Vector Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Matrix-vector multiplication.
pub trait MatVecMul<Rhs> {
    type Output;

    /// Matrix-vector multiplication.
    ///
    /// Performs a matrix-vector multiplication between the matrix variable `self` and the vector
    /// variable `other`.
    ///
    /// If `self` is *(n, m)* and `other` is *m* the output will be *n*.
    fn mv_mul(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Vector Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Vector-matrix multiplication.
pub trait VecMatMul<Rhs> {
    type Output;

    /// Vector-matrix multiplication.
    ///
    /// Performs a vector-matrix multiplication between the vector variable `self` and the matrix
    /// variable `other`.
    ///
    /// If `self` is **n** and `other` is *(n, m)* the output will be *m*.
    fn vm_mul(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Vector Vector Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Vector-vector multiplication.
pub trait VecVecMul<Rhs> {
    type Output;

    /// Vector-vector product, a.k.a. *scalar product* or *inner product*.
    ///
    /// Performs the scalar product between the two vector variables `self` and `other`.
    fn vv_mul(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concat and Stack traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Concatenation.
pub trait Cat<Rhs> {
    type Output;

    /// Concatenates the variables `self` and `other` along `axis`.
    ///
    /// All variables must have the same shape, except in the concatenating dimension.
    fn cat(self, other: Rhs, axis: usize) -> Self::Output;
}

/// Stacking.
pub trait Stack<Rhs> {
    type Output;

    /// Stacks the variables `self` and `other` along a new dimension specified by `axis`.
    ///
    /// All variables must have the same shape.
    fn stack(self, other: Rhs, axis: usize) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Utils ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates an empty tensor whose shape is the result of broadcasting between `left` and `right`.
pub(crate) fn broadcasted_zeros<Lhs, Rhs>(
    left: &Tensor<Lhs>,
    right: &Tensor<Rhs>,
) -> BroadTensor<Lhs, Rhs>
where
    Lhs: Dimension + DimMax<Rhs>,
    Rhs: Dimension,
{
    let (bigger, smaller) = if left.ndim() >= right.ndim() {
        (left.shape(), right.shape())
    } else {
        (right.shape(), left.shape())
    };
    let mut broad_dim = <Lhs as DimMax<Rhs>>::Output::zeros(bigger.len());
    broad_dim
        .slice_mut()
        .iter_mut()
        .zip(bigger.iter())
        .for_each(|(l, r)| *l = *r);
    broad_dim
        .slice_mut()
        .iter_mut()
        .rev()
        .zip(smaller.iter().rev())
        .for_each(|(l, r)| *l = std::cmp::max(*l, *r));
    Tensor::zeros(broad_dim)
}

/// Requests the gradient of `tensor` as a reference.
///
/// **panics** if the gradient has been deallocated.
pub(crate) fn expect_tensor<D: Dimension>(tensor: &RefCell<Option<Tensor<D>>>) -> Ref<Tensor<D>> {
    Ref::map(tensor.borrow(), |b| {
        b.as_ref().expect(
            "error: trying to get a deallocated gradient. 
        Switch on the gradients first by using with_grad().",
        )
    })
}

/// Requests the gradient of `tensor` as a mutable reference.
///
/// **panics** if the gradient has been deallocated.
pub(crate) fn expect_tensor_mut<D: Dimension>(
    tensor: &RefCell<Option<Tensor<D>>>,
) -> RefMut<Tensor<D>> {
    RefMut::map(tensor.borrow_mut(), |b| {
        b.as_mut().expect(
            "error: trying to get a deallocated gradient. 
        Switch on the gradients first by using with_grad().",
        )
    })
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Non differentiable Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// A non-differentiable variable.
///
/// This, together with its differentiable counterpart [`VarDiff`], is the main building block of
/// every computation.
///
/// Conceptually, it can be thought of as a [`ndarray::Array`] for which the computations are
/// automatically kept track of.
///
/// It is important to note that cloning is extremely memory efficient as only a shallow copy is
/// returned. Cloning a variable is thus the way to go if it must be used multiple times.
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
    /// Promotes `self` to a differentiable variable. A subsequent call to [`backward()`]
    /// will compute its gradient.
    ///
    /// [`backward()`]: VarDiff::backward()
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
        debug_assert_eq!(
            self.past.is_empty(),
            true,
            "error: the variable is not a leaf."
        );
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
            past: DiffVarHistory::new(parameters),
        }
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
    /// See also [`dropout()`].
    ///
    ///  [`dropout()`]: Var::dropout()    
    pub fn train(&self) {
        for changeable in &self.past.changeables {
            unsafe {
                (&**changeable).train();
            }
        }
    }

    /// This has effect only on certain **ancestor** variables of `self`. It sets such variables
    /// in evaluation mode.
    ///    
    /// See also [`dropout()`].
    ///
    ///  [`dropout()`]: Var::dropout()   
    pub fn eval(&self) {
        for changeable in &self.past.changeables {
            unsafe {
                (&**changeable).eval();
            }
        }
    }
}

impl<T: Data + Forward + ChangeBehaviour + 'static> Var<T> {
    pub(crate) fn from_changable(node: T, mut past: VarHistory) -> Self {
        let node = Rc::new(node);
        past.append_forward(unsafe { OPERATIONS_COUNTER.next() }, node.clone());
        past.append_changeable(node.as_ref() as *const dyn ChangeBehaviour);

        Var { node, past }
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
    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.node.data()
    }

    /// Returns the sum of all elements in `self`.
    pub fn sum(self) -> Var<Sum<T>> {
        Var::from(Sum::new(self.node), self.past)
    }

    /// Takes the power of each element in `self` with exponent `exp` and returns a variable with the
    /// result.
    pub fn pow(self, exp: i32) -> Var<Power<T>> {
        Var::from(Power::new(self.node, exp), self.past)
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
    pub fn softmax(self, axis: usize) -> Var<Softmax<T>> {
        Var::from(Softmax::new(self.node, axis), self.past)
    }

    /// Applies the *log-softmax* to `self` and returns a variable with the result.
    pub fn log_softmax(self, axis: usize) -> Var<LogSoftmax<T>> {
        Var::from(LogSoftmax::new(self.node, axis), self.past)
    }

    /// Returns a variable equivalent to `self` with its dimensions reversed.
    pub fn t(self) -> Var<Transpose<T>> {
        Var::from(Transpose::new(self.node), self.past)
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
    pub fn dropout(self, p: f64) -> Var<Dropout<T>> {
        Var::from_changable(Dropout::new(self.node, p), self.past)
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
}

impl<T: Data + Forward + ChangeBehaviour> Var<T> {
    /// Sets `self` in training mode.
    pub fn set_train(&self) {
        self.node.train();
    }

    /// Sets `self` in evaluation mode.
    pub fn set_eval(&self) {
        self.node.eval();
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Differentiable Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A differentiable variable.
///
/// Differentiable variables can be created in the **two** following ways described hereafter:
///
/// 1. By calling [`requires_grad()`] on a non-differentiable leaf.
///
/// [`requires_grad()`]: Var::requires_grad()
///
/// 2. By performing any binary operation between a [`Var`] and a `VarDiff`. Differentiability
/// is thus a *contagious* property, that is, if during a computation a `VarDiff` is used, the
/// result of the computation itself, and that of any other subsequent computations performed on
/// it, will also be differentiable. As an obvious consequence, the results of operations
/// performed on two *VarDiff* will also be *VarDiff*.
///
/// It is important to note that cloning is extremely memory efficient as only a shallow copy is
/// returned. Cloning a differentiable variable is thus the way to go if it must be used multiple
/// times.
pub struct VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient + Overwrite + 'static,
{
    pub(crate) var: Var<T>,
    pub(crate) node: Rc<U>,
    pub(crate) past: DiffVarHistory,
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

impl<D> VarDiff<Input<D>, InputBackward<D>>
where
    D: Dimension,
{
    /// Returns an immutable reference to the gradient inside `self`.
    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.node.gradient()
    }

    /// Returns a mutable reference to the data inside `self`.
    pub(crate) fn data_mut(&mut self) -> RefMut<Tensor<D>> {
        self.var.node.data_mut()
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient + Overwrite + Backward + 'static,
{
    pub(crate) fn from(node: U, mut past: DiffVarHistory, var: Var<T>) -> VarDiff<T, U> {
        let node = Rc::new(node);
        past.append_backward(unsafe { OPERATIONS_COUNTER.next() }, node.clone());

        VarDiff { var, node, past }
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + Forward + ChangeBehaviour + 'static,
    U: Gradient + Overwrite + Backward + ChangeBehaviour + 'static,
{
    pub(crate) fn from_changable(node: U, mut past: DiffVarHistory, var: Var<T>) -> VarDiff<T, U> {
        let node = Rc::new(node);
        past.append_backward(unsafe { OPERATIONS_COUNTER.next() }, node.clone());
        past.append_changeable(node.as_ref() as *const dyn ChangeBehaviour);

        VarDiff { var, node, past }
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient + Overwrite + 'static,
{
    /// Returns an immutable reference to the data inside `self`.
    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.var.node.data()
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + Backward + 'static,
{
    /// Propagates the computations forwards and populates all the variables and differetiable
    /// variables from the leaves of the graph to `self`.   
    pub fn forward(&mut self) {
        self.var.forward();

        debug_assert!(self.past.buffer().is_empty() || self.past.len() == self.past.buffer().len());
        // ! If the backward buffer isn't empty, then we're doing a `forward -> backward -> forward` chain,
        // ! thus we must reset the `overwrite` bit of every `backward` node of our past
        for node in self.past.buffer() {
            // Todo: This can be done more efficently by looking for the first node
            // Todo: that must be reset, in the same way for `forward` e `backward`

            node.set_overwrite(true);
        }
    }

    /// Back-propagates through the computational graph and populates the gradients of the
    /// differentiable leaves that are ancestors of `self`. Before back-propagating the gradient
    /// of `self` is seeded with `seed`, thus, the leaves' gradients will be scaled accordingly.
    ///
    /// The graph is differentiated through the
    /// [chain rule](https://en.wikipedia.org/wiki/Chain_rule).
    ///
    /// The leaves whose gradients are populated by this method are also those referred by the
    /// vector of [`Param`] returned by [`parameters()`].
    ///
    ///  [`parameters()`]: VarDiff::parameters()
    pub fn backward(&mut self, seed: f32) {
        debug_assert_eq!(self.past.is_empty(), false);

        self.node.gradient_mut().fill(seed);
        self.past.prepare_buffer();
        let buffer = self.past.buffer();
        for node in buffer.iter().rev() {
            node.backward();
        }

        // TODO: remove this comments
        // ! We are sure that the forward computation must have be already done
        debug_assert_eq!(self.var.past.len(), self.var.past.buffer().len());
        for node in self.var.past.buffer() {
            // Todo: This can be done more efficently by looking for the first node
            // Todo: that must be reset, in the same way for `forward`

            node.reset_computation();
        }
    }

    /// Disables gradient computation and deallocates the gradient for `self` and all of its
    /// ancestors.
    pub fn no_grad(&mut self) {
        self.past.prepare_buffer();
        for node in &self.past.buffer {
            node.no_grad();
        }
    }

    /// Re-enables gradient computation and re-allocates the gradient for `self` and all of its
    /// ancestors.
    pub fn with_grad(&mut self) {
        self.past.prepare_buffer();
        for node in &self.past.buffer {
            node.with_grad();
        }
    }

    /// This has effect only on certain **ancestor** variables of `self`. It sets such variables
    /// and differentiable variables in training mode.
    ///    
    /// See also [`dropout()`].
    ///
    ///  [`dropout()`]: VarDiff::dropout()
    pub fn train(&self) {
        for changeable in &self.past.changeables {
            unsafe {
                (&**changeable).train();
            }
        }
    }

    /// This has effect only on certain **ancestor** variables of `self`. It sets such variables
    /// and differentiable variables in evaluation mode.
    ///    
    /// See also [`dropout()`].
    ///
    ///  [`dropout()`]: VarDiff::dropout()
    pub fn eval(&self) {
        for changeable in &self.past.changeables {
            unsafe {
                (&**changeable).eval();
            }
        }
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
    /// let x = neuronika::rand((3,3)).requires_grad() + neuronika::rand((3,3)).requires_grad();
    /// let y = neuronika::rand(3).requires_grad() + neuronika::rand(1).requires_grad();
    ///
    /// assert!(x.parameters().len() == y.parameters().len() && y.parameters().len() == 2);
    ///
    /// let z = x.clone() + y;
    /// assert_eq!(z.parameters().len(), 4);
    ///
    /// let w = z + x;
    /// assert_eq!(w.parameters().len(), 4);
    /// ```
    pub fn parameters(&self) -> Vec<Param> {
        self.past.parameters.iter().cloned().collect()
    }

    /// Returns the sum of all elements in `self`.
    pub fn sum(self) -> VarDiff<Sum<T>, SumBackward<U>> {
        let node = SumBackward::new(self.node);
        VarDiff::from(node, self.past, self.var.sum())
    }

    /// Takes the power of each element in `self` with exponent `exp` and returns a differentiable
    /// variable with the result.
    pub fn pow(self, exp: i32) -> VarDiff<Power<T>, PowerBackward<U, T>> {
        let node = PowerBackward::new(self.node, self.var.node.clone(), exp);
        VarDiff::from(node, self.past, self.var.pow(exp))
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
    pub fn softplus(self) -> VarDiff<SoftPlus<T>, SoftPlusBackward<U, T>> {
        let node = SoftPlusBackward::new(self.node, self.var.node.clone());
        VarDiff::from(node, self.past, self.var.softplus())
    }

    /// Applies the *sigmoid* element-wise and returns a differentiiable variable with the result.
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
    pub fn softmax(self, axis: usize) -> VarDiff<Softmax<T>, SoftmaxBackward<U, Softmax<T>>> {
        let var = self.var.softmax(axis);
        let node = SoftmaxBackward::new(self.node, var.node.clone(), axis);
        VarDiff::from(node, self.past, var)
    }

    /// Applies the *log-softmax* to `self` and returns a differentiable variable with the result.
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
    pub fn dropout(self, p: f64) -> VarDiff<Dropout<T>, DropoutBackward<U, T>> {
        let var = self.var.dropout(p);
        let node = DropoutBackward::new(self.node, var.node.clone(), p);
        VarDiff::from_changable(node, self.past, var)
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
}

impl<T, U> VarDiff<T, U>
where
    T: Data + Forward + ChangeBehaviour + 'static,
    U: Gradient + Overwrite + Backward + ChangeBehaviour + 'static,
{
    /// Sets `self` in training mode.
    pub fn set_train(&self) {
        self.var.set_eval();
        self.node.train();
    }

    /// Sets `self` in evaluation mode.
    pub fn set_eval(&self) {
        self.var.set_eval();
        self.node.eval();
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    T::Dim: RemoveAxis,
{
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arithmetic Operations Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T: Data + 'static> Neg for Var<T> {
    type Output = Var<Negation<T>>;

    fn neg(self) -> Self::Output {
        Var::from(Negation::new(self.node), self.past)
    }
}

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

impl<F1, F2> MatMatMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = Var<MatrixMatrixMul<F1, F2>>;

    fn mm_mul(mut self, rhs: Var<F2>) -> Self::Output {
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

    fn mm_mul(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = MatrixMatrixMulBackwardRight::new(self.node.clone(), rhs.node);
        VarDiff::from(node, rhs.past, self.mm_mul(rhs.var))
    }
}

impl<F1, B1, F2> MatMatMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = VarDiff<MatrixMatrixMul<F1, F2>, MatrixMatrixMulBackwardLeft<B1, F2>>;

    fn mm_mul(self, rhs: Var<F2>) -> Self::Output {
        let node = MatrixMatrixMulBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.mm_mul(rhs))
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

    fn mm_mul(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = MatrixMatrixMulBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.mm_mul(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition  ~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> MatMatMulT<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = Var<MatrixMatrixMulT<F1, F2>>;

    fn mm_mul_t(mut self, rhs: Var<F2>) -> Self::Output {
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

    fn mm_mul_t(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = MatrixMatrixMulTBackwardRight::new(self.node.clone(), rhs.node);
        VarDiff::from(node, rhs.past, self.mm_mul_t(rhs.var))
    }
}

impl<F1, B1, F2> MatMatMulT<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = VarDiff<MatrixMatrixMulT<F1, F2>, MatrixMatrixMulTBackwardLeft<B1, F2>>;

    fn mm_mul_t(self, rhs: Var<F2>) -> Self::Output {
        let node = MatrixMatrixMulTBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.mm_mul_t(rhs))
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

    fn mm_mul_t(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = MatrixMatrixMulTBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.mm_mul_t(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> MatVecMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = Var<MatrixVectorMul<F1, F2>>;

    fn mv_mul(mut self, rhs: Var<F2>) -> Self::Output {
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

    fn mv_mul(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = MatrixVectorMulBackwardRight::new(self.node.clone(), rhs.node);
        VarDiff::from(node, rhs.past, self.mv_mul(rhs.var))
    }
}

impl<F1, B1, F2> MatVecMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = VarDiff<MatrixVectorMul<F1, F2>, MatrixVectorMulBackwardLeft<B1, F2>>;

    fn mv_mul(self, rhs: Var<F2>) -> Self::Output {
        let node = MatrixVectorMulBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.mv_mul(rhs))
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

    fn mv_mul(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = MatrixVectorMulBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.mv_mul(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> VecMatMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = Var<VectorMatrixMul<F1, F2>>;

    fn vm_mul(mut self, rhs: Var<F2>) -> Self::Output {
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

    fn vm_mul(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = VectorMatrixMulBackwardRight::new(self.node.clone(), rhs.node);
        VarDiff::from(node, rhs.past, self.vm_mul(rhs.var))
    }
}

impl<F1, B1, F2> VecMatMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = VarDiff<VectorMatrixMul<F1, F2>, VectorMatrixMulBackwardLeft<B1, F2>>;

    fn vm_mul(self, rhs: Var<F2>) -> Self::Output {
        let node = VectorMatrixMulBackwardLeft::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.vm_mul(rhs))
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

    fn vm_mul(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = VectorMatrixMulBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.vm_mul(rhs.var))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> VecVecMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = Var<VectorVectorMul<F1, F2>>;

    fn vv_mul(mut self, rhs: Var<F2>) -> Self::Output {
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

    fn vv_mul(self, rhs: VarDiff<F2, B2>) -> Self::Output {
        let node = VectorVectorMulBackwardUnary::new(rhs.node, self.node.clone());
        VarDiff::from(node, rhs.past, self.vv_mul(rhs.var))
    }
}

impl<F1, B1, F2> VecVecMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = VarDiff<VectorVectorMul<F1, F2>, VectorVectorMulBackwardUnary<B1, F2>>;

    fn vv_mul(self, rhs: Var<F2>) -> Self::Output {
        let node = VectorVectorMulBackwardUnary::new(self.node, rhs.node.clone());
        VarDiff::from(node, self.past, self.var.vv_mul(rhs))
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

    fn vv_mul(mut self, rhs: VarDiff<F2, B2>) -> Self::Output {
        self.past.merge(rhs.past);
        let node = VectorVectorMulBackward::new(
            self.var.node.clone(),
            self.node,
            rhs.var.node.clone(),
            rhs.node,
        );
        VarDiff::from(node, self.past, self.var.vv_mul(rhs.var))
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
        VarDiff::from(node, rhs.past, self.cat(rhs.var, axis))
    }
}

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
        VarDiff::from(node, self.past, self.var.cat(rhs, axis))
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
        VarDiff::from(node, self.past, self.var.cat(rhs.var, axis))
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, F2> Stack<Var<F2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    F1::Dim: RemoveAxis,
{
    type Output = Var<StackF<F1, F2>>;

    fn stack(mut self, rhs: Var<F2>, axis: usize) -> Self::Output {
        self.past.merge(rhs.past);
        Var::from(StackF::new(self.node, rhs.node, axis), self.past)
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
    type Output = VarDiff<StackF<F1, F2>, StackBackwardRight<B2>>;

    fn stack(self, rhs: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        let node = StackBackwardRight::new(self.node.clone(), rhs.node, axis);
        VarDiff::from(node, rhs.past, self.stack(rhs.var, axis))
    }
}

impl<F1, B1, F2> Stack<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = B1::Dim> + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    B1: Gradient + Overwrite + 'static,
    F1::Dim: RemoveAxis,
    B1::Dim: RemoveAxis,
{
    type Output = VarDiff<StackF<F1, F2>, StackBackwardLeft<B1>>;

    fn stack(self, rhs: Var<F2>, axis: usize) -> Self::Output {
        let node = StackBackwardLeft::new(self.node, rhs.node.clone(), axis);
        VarDiff::from(node, self.past, self.var.stack(rhs, axis))
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
    type Output = VarDiff<StackF<F1, F2>, StackBackward<B1, B2>>;

    fn stack(mut self, rhs: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        self.past.merge(rhs.past);
        let node = StackBackward::new(self.node, rhs.node, axis);
        VarDiff::from(node, self.past, self.var.stack(rhs.var, axis))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod forward {
        use super::*;

        #[test]
        fn creation() {
            let a = Input::new(
                Tensor::from_shape_vec((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]).unwrap(),
            );
            let b = Input::new(Tensor::from_shape_vec((1, 3), vec![1., 1., 1.]).unwrap());

            let mut c = a + b;
            let mut d = c.clone().relu().ln();
            c.forward();
            d.forward();
        }
    }
}
