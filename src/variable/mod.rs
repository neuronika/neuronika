use ndarray::{
    Array, ArrayD, ArrayViewMutD, DimMax, Dimension, IntoDimension, Ix, Ix1, Ix2, RawArrayViewMut,
    RemoveAxis,
};
use node::{
    backward::{Backward, Differentiable, Gradient, Overwrite},
    forward::{Data, Forward},
    Addition, AdditionBackward, AdditionBackwardUnary, Chunk, ChunkBackward, Concatenate,
    ConcatenateBackward, ConcatenateBackwardLeft, ConcatenateBackwardRight, Division,
    DivisionBackward, DivisionBackwardLeft, DivisionBackwardRight, Dropout, DropoutBackward, Exp,
    ExpBackward, LeakyReLU, LeakyReLUBackward, LogSoftmax, LogSoftmaxBackward, Logn, LognBackward,
    MatrixMatrixMul, MatrixMatrixMulBackward, MatrixMatrixMulBackwardLeft,
    MatrixMatrixMulBackwardRight, MatrixMatrixMulT, MatrixMatrixMulTBackward,
    MatrixMatrixMulTBackwardLeft, MatrixMatrixMulTBackwardRight, MatrixVectorMul,
    MatrixVectorMulBackward, MatrixVectorMulBackwardLeft, MatrixVectorMulBackwardRight,
    Multiplication, MultiplicationBackward, MultiplicationBackwardUnary, Negation,
    NegationBackward, Power, PowerBackward, ReLU, ReLUBackward, Sigmoid, SigmoidBackward, SoftPlus,
    SoftPlusBackward, Softmax, SoftmaxBackward, Stack as StackF, StackBackward, StackBackwardLeft,
    StackBackwardRight, Subtraction, SubtractionBackward, SubtractionBackwardLeft,
    SubtractionBackwardRight, Sum, SumBackward, TanH, TanHBackward, Transpose, TransposeBackward,
    Unsqueeze, UnsqueezeBackward, VectorMatrixMul, VectorMatrixMulBackward,
    VectorMatrixMulBackwardLeft, VectorMatrixMulBackwardRight, VectorVectorMul,
    VectorVectorMulBackward, VectorVectorMulBackwardUnary,
};
use std::{
    cell::{Ref, RefMut},
    collections::BTreeMap,
    collections::HashSet,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

pub mod node;

pub use node::{Input, InputBackward};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Type Aliases ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;
pub(crate) type BroadTensor<Lhs, Rhs> = Tensor<Broadcasted<Lhs, Rhs>>;
pub(crate) type DynTensor = ArrayD<f32>;
pub(crate) type Tensor<D> = Array<f32, D>;

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
}

impl VarHistory {
    pub fn new() -> Self {
        Self {
            path: BTreeMap::new(),
            buffer: Vec::new(),
        }
    }

    pub fn merge(&mut self, mut other: VarHistory) {
        self.path.append(&mut other.path);
    }

    pub fn extend(&mut self, id: usize, next: Rc<dyn Forward>) {
        self.path.insert(id, next);
        self.buffer.truncate(0);
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
}

impl DiffVarHistory {
    pub fn new(parameters: HashSet<Param>) -> Self {
        Self {
            path: BTreeMap::new(),
            buffer: Vec::new(),
            parameters,
        }
    }

    pub fn merge(&mut self, mut other: DiffVarHistory) {
        self.path.append(&mut other.path);
        self.parameters.extend(other.parameters);
    }

    pub fn extend(&mut self, id: usize, next: Rc<dyn Backward>) {
        self.path.insert(id, next);
        self.buffer.truncate(0);
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

pub trait MatMatMul<Rhs> {
    type Output;

    fn mm_mul(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition ~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub trait MatMatMulT<Rhs> {
    type Output;
    fn mm_mul_t(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Vector Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub trait MatVecMul<Rhs> {
    type Output;

    fn mv_mul(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Vector Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub trait VecMatMul<Rhs> {
    type Output;

    fn vm_mul(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Vector Vector Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait VecVecMul<Rhs> {
    type Output;

    fn vv_mul(self, other: Rhs) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concat and Stack traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait Cat<Rhs> {
    type Output;

    fn cat(self, other: Rhs, axis: usize) -> Self::Output;
}

pub trait Stack<Rhs> {
    type Output;

    fn stack(self, other: Rhs, axis: usize) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Non differentiable Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Var<T: Data + 'static> {
    pub(crate) node: Rc<T>,
    pub(crate) past: VarHistory,
}

impl<T> Clone for Var<T>
where
    T: Data + 'static,
{
    fn clone(&self) -> Self {
        Self {
            node: self.node.clone(),
            past: self.past.clone(),
        }
    }
}

impl<D> Var<Input<D>>
where
    D: Dimension,
{
    pub fn requires_grad(self) -> VarDiff<Input<D>, InputBackward<D>> {
        debug_assert_eq!(self.past.is_empty(), true);

        if Rc::strong_count(&self.node) > 2 {
            panic!("error: cannot make the Input differentiable.")
        }
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
    pub fn forward(&mut self) {
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
        past.extend(unsafe { OPERATIONS_COUNTER.next() }, node.clone());

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

    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.node.data()
    }

    pub fn sum(self) -> Var<Sum<T>> {
        Var::from(Sum::new(self.node), self.past)
    }

    pub fn pow(self, exp: i32) -> Var<Power<T>> {
        Var::from(Power::new(self.node, exp), self.past)
    }

    pub fn relu(self) -> Var<ReLU<T>> {
        Var::from(ReLU::new(self.node), self.past)
    }

    pub fn leaky_relu(self) -> Var<LeakyReLU<T>> {
        Var::from(LeakyReLU::new(self.node), self.past)
    }

    pub fn softplus(self) -> Var<SoftPlus<T>> {
        Var::from(SoftPlus::new(self.node), self.past)
    }

    pub fn sigmoid(self) -> Var<Sigmoid<T>> {
        Var::from(Sigmoid::new(self.node), self.past)
    }

    pub fn tanh(self) -> Var<TanH<T>> {
        Var::from(TanH::new(self.node), self.past)
    }

    pub fn ln(self) -> Var<Logn<T>> {
        Var::from(Logn::new(self.node), self.past)
    }

    pub fn exp(self) -> Var<Exp<T>> {
        Var::from(Exp::new(self.node), self.past)
    }

    pub fn softmax(self, axis: usize) -> Var<Softmax<T>> {
        Var::from(Softmax::new(self.node, axis), self.past)
    }

    pub fn log_softmax(self, axis: usize) -> Var<LogSoftmax<T>> {
        Var::from(LogSoftmax::new(self.node, axis), self.past)
    }

    pub fn t(self) -> Var<Transpose<T>> {
        Var::from(Transpose::new(self.node), self.past)
    }

    pub fn dropout(self, p: f64) -> Var<Dropout<T>> {
        Var::from(Dropout::new(self.node, p), self.past)
    }

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
    pub fn unsqueeze(self, axis: usize) -> Var<Unsqueeze<T>> {
        Var::from(Unsqueeze::new(self.node, axis), self.past)
    }
}

impl<T> Var<Dropout<T>>
where
    T: Data,
{
    pub fn train(&self, status: bool) {
        self.node.set_train(status);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Differentiable Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.node.gradient()
    }

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
        past.extend(unsafe { OPERATIONS_COUNTER.next() }, node.clone());

        VarDiff { var, node, past }
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient + Overwrite + 'static,
{
    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.var.node.data()
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + Backward + 'static,
{
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

    pub fn backward(&mut self, seed: f32) {
        debug_assert_eq!(self.past.is_empty(), false);

        self.node.gradient_mut().fill(seed);
        self.past.prepare_buffer();
        let buffer = self.past.buffer();
        for node in buffer.iter().rev() {
            node.backward();
        }

        // ! We are sure that the forward computation must have be already done
        debug_assert_eq!(self.var.past.len(), self.var.past.buffer().len());
        for node in self.var.past.buffer() {
            // Todo: This can be done more efficently by looking for the first node
            // Todo: that must be reset, in the same way for `forward`

            node.reset_computation();
        }
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
{
    pub fn parameters(&self) -> Vec<Param> {
        self.past.parameters.iter().cloned().collect()
    }

    pub fn sum(self) -> VarDiff<Sum<T>, SumBackward<U>> {
        let node = SumBackward::new(self.node);
        VarDiff::from(node, self.past, self.var.sum())
    }

    pub fn pow(self, exp: i32) -> VarDiff<Power<T>, PowerBackward<U, T>> {
        let node = PowerBackward::new(self.node, self.var.node.clone(), exp);
        VarDiff::from(node, self.past, self.var.pow(exp))
    }

    pub fn relu(self) -> VarDiff<ReLU<T>, ReLUBackward<U, T>> {
        let node = ReLUBackward::new(self.node, self.var.node.clone());
        VarDiff::from(node, self.past, self.var.relu())
    }

    pub fn leaky_relu(self) -> VarDiff<LeakyReLU<T>, LeakyReLUBackward<U, T>> {
        let node = LeakyReLUBackward::new(self.node, self.var.node.clone());
        VarDiff::from(node, self.past, self.var.leaky_relu())
    }

    pub fn softplus(self) -> VarDiff<SoftPlus<T>, SoftPlusBackward<U, T>> {
        let node = SoftPlusBackward::new(self.node, self.var.node.clone());
        VarDiff::from(node, self.past, self.var.softplus())
    }

    pub fn sigmoid(self) -> VarDiff<Sigmoid<T>, SigmoidBackward<U, Sigmoid<T>>> {
        let var = self.var.sigmoid();
        let node = SigmoidBackward::new(self.node, var.node.clone());
        VarDiff::from(node, self.past, var)
    }

    pub fn tanh(self) -> VarDiff<TanH<T>, TanHBackward<U, TanH<T>>> {
        let var = self.var.tanh();
        let node = TanHBackward::new(self.node, var.node.clone());
        VarDiff::from(node, self.past, var)
    }

    pub fn ln(self) -> VarDiff<Logn<T>, LognBackward<U, T>> {
        let node = LognBackward::new(self.node, self.var.node.clone());
        VarDiff::from(node, self.past, self.var.ln())
    }

    pub fn exp(self) -> VarDiff<Exp<T>, ExpBackward<U, Exp<T>>> {
        let var = self.var.exp();
        let node = ExpBackward::new(self.node, var.node.clone());
        VarDiff::from(node, self.past, var)
    }

    pub fn softmax(self, axis: usize) -> VarDiff<Softmax<T>, SoftmaxBackward<U, Softmax<T>>> {
        let var = self.var.softmax(axis);
        let node = SoftmaxBackward::new(self.node, var.node.clone(), axis);
        VarDiff::from(node, self.past, var)
    }

    pub fn log_softmax(
        self,
        axis: usize,
    ) -> VarDiff<LogSoftmax<T>, LogSoftmaxBackward<U, LogSoftmax<T>>> {
        let var = self.var.log_softmax(axis);
        let node = LogSoftmaxBackward::new(self.node, var.node.clone(), axis);
        VarDiff::from(node, self.past, var)
    }

    pub fn t(self) -> VarDiff<Transpose<T>, TransposeBackward<U>> {
        let node = TransposeBackward::new(self.node);
        VarDiff::from(node, self.past, self.var.t())
    }

    pub fn dropout(self, p: f64) -> VarDiff<Dropout<T>, DropoutBackward<U, T>> {
        let var = self.var.dropout(p);
        let node = DropoutBackward::new(self.node, var.node.clone(), p);
        VarDiff::from(node, self.past, var)
    }

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

impl<T, U> VarDiff<Dropout<T>, DropoutBackward<U, T>>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
{
    pub fn train(&self, status: bool) {
        self.var.node.set_train(status);
        self.node.set_train(status);
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    T::Dim: RemoveAxis,
{
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
