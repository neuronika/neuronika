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
pub struct Var<T>
where
    T: Data + 'static,
{
    pub(crate) id: usize,
    pub(crate) node: Rc<T>,
    pub(crate) path: BTreeMap<usize, Rc<dyn Forward>>,
    pub(crate) buffer: Vec<Rc<dyn Forward>>,
}

impl<T> Clone for Var<T>
where
    T: Data + 'static,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            node: self.node.clone(),
            path: self.path.clone(),
            buffer: self.buffer.clone(),
        }
    }
}

impl<D> Var<Input<D>>
where
    D: Dimension,
{
    pub fn requires_grad(self) -> VarDiff<Input<D>, InputBackward<D>> {
        debug_assert_eq!(self.path.is_empty(), true);
        debug_assert_eq!(self.buffer.is_empty(), true);

        if Rc::strong_count(&self.node) > 2 {
            panic!("error: cannot make the Input differentiable.")
        }
        let backward = Rc::new(self.node.differentiable());
        let mut parameters = HashSet::new();
        {
            let (mut data, mut gradient) = (self.node.data_mut(), backward.gradient_mut());
            parameters.insert(Param::new(
                data.as_mut_ptr(),
                gradient.as_mut_ptr(),
                gradient.shape().to_vec(),
            ));
        }

        VarDiff {
            id: usize::MAX,
            forward: self.node,
            backward,
            forward_path: self.path,
            backward_path: BTreeMap::new(),
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters,
        }
    }
}

impl<T: Data + Forward + 'static> Var<T> {
    pub fn forward(&mut self) {
        if self.buffer.is_empty() {
            self.buffer = self.path.values().cloned().collect()
        }

        let mut res = self.buffer.binary_search_by(|node| {
            if node.was_computed() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        if let Err(i) = res {
            if self.buffer.get(i).is_some() {
                res = Ok(i);
            }
        };

        if let Ok(pos) = res {
            for node in &self.buffer[pos..] {
                node.forward();
            }
        }
    }
}

impl<T: Data + 'static> Var<T> {
    pub(crate) fn new(node: T) -> Self {
        Self {
            id: usize::MAX,
            node: Rc::new(node),
            path: BTreeMap::new(),
            buffer: Vec::new(),
        }
    }

    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.node.data()
    }

    pub fn sum(mut self) -> Var<Sum<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Sum::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn pow(mut self, exp: i32) -> Var<Power<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Power::new(self.node, exp));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn relu(mut self) -> Var<ReLU<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(ReLU::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn leaky_relu(mut self) -> Var<LeakyReLU<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(LeakyReLU::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn softplus(mut self) -> Var<SoftPlus<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(SoftPlus::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn sigmoid(mut self) -> Var<Sigmoid<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Sigmoid::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn tanh(mut self) -> Var<TanH<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(TanH::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn ln(mut self) -> Var<Logn<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Logn::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn exp(mut self) -> Var<Exp<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Exp::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn softmax(mut self, axis: usize) -> Var<Softmax<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Softmax::new(self.node, axis));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn log_softmax(mut self, axis: usize) -> Var<LogSoftmax<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(LogSoftmax::new(self.node, axis));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn t(mut self) -> Var<Transpose<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Transpose::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn dropout(mut self, p: f64) -> Var<Dropout<T>> {
        let node = self.node;
        let (id, node) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Dropout::new(node, p)),
        );
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }

    pub fn chunks<E: IntoDimension<Dim = T::Dim>>(self, chunk_size: E) -> Vec<Var<Chunk<T>>> {
        let node = self.node;
        let path = self.path;
        let data = node.data();
        let chunks = data.exact_chunks(chunk_size).into_iter().enumerate();
        chunks
            .map(|(i, chunk)| {
                let (id, node) = (
                    unsafe { OPERATIONS_COUNTER.next() },
                    Rc::new(Chunk::new(node.clone(), chunk.to_owned(), i)),
                );

                let mut new_forward_path = path.clone();
                new_forward_path.insert(id, node.clone() as Rc<dyn Forward>);

                Var {
                    id,
                    node,
                    path: new_forward_path,
                    buffer: Vec::new(),
                }
            })
            .collect()
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

impl<T> Var<T>
where
    T: Data + 'static,
    T::Dim: RemoveAxis,
{
    pub fn unsqueeze(mut self, axis: usize) -> Var<Unsqueeze<T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Unsqueeze::new(self.node, axis));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Differentiable Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient + Overwrite + 'static,
{
    pub(crate) id: usize,
    pub(crate) forward: Rc<T>,
    pub(crate) backward: Rc<U>,
    pub(crate) forward_path: BTreeMap<usize, Rc<dyn Forward>>,
    pub(crate) backward_path: BTreeMap<usize, Rc<dyn Backward>>,
    pub(crate) forward_buffer: Vec<Rc<dyn Forward>>,
    pub(crate) backward_buffer: Vec<Rc<dyn Backward>>,
    pub(crate) parameters: HashSet<Param>,
}

impl<T, U> Clone for VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient + Overwrite + 'static,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            forward: self.forward.clone(),
            backward: self.backward.clone(),
            forward_path: self.forward_path.clone(),
            backward_path: self.backward_path.clone(),
            forward_buffer: self.forward_buffer.clone(),
            backward_buffer: self.backward_buffer.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

impl<D> VarDiff<Input<D>, InputBackward<D>>
where
    D: Dimension,
{
    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.backward.gradient()
    }

    pub(crate) fn data_mut(&mut self) -> RefMut<Tensor<D>> {
        self.forward.data_mut()
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + Overwrite + 'static,
{
    pub fn forward(&mut self) {
        if self.forward_buffer.is_empty() {
            self.forward_buffer = self.forward_path.values().cloned().collect()
        }

        let mut res = self.forward_buffer.binary_search_by(|node| {
            if node.was_computed() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        if let Err(pos) = res {
            if pos != self.forward_buffer.len() {
                debug_assert_eq!(pos, 0);

                res = Ok(pos);
            }
        };

        if let Ok(pos) = res {
            for node in &self.forward_buffer[pos..] {
                node.forward();
            }

            // ! Differently from below, we don't know if this `forward` comes after a `backward`
            // ! so we can't ensure anything
            if !self.backward_buffer.is_empty() {
                debug_assert_eq!(self.backward_path.len(), self.backward_buffer.len());

                for node in &self.backward_buffer {
                    // Todo: This can be done more efficently by looking for the first node
                    // Todo: that must be reset, in the same way for `forward` e `backward`

                    node.set_overwrite(true);
                }
            }
        }
    }

    pub fn backward(&mut self, seed: f32) {
        debug_assert_eq!(self.backward_path.is_empty(), false);

        if self.backward_buffer.is_empty() {
            self.backward_buffer = self.backward_path.values().cloned().collect()
        }

        self.backward.gradient_mut().map_inplace(|el| *el = seed);
        for node in self.backward_buffer.iter().rev() {
            node.backward();
        }

        // ! We are sure that the forward computation must have be already done
        debug_assert_eq!(self.forward_path.len(), self.forward_buffer.len());
        for node in &self.forward_buffer {
            // Todo: This can be done more efficently by looking for the first node
            // Todo: that must be reset, in the same way for `forward`

            node.reset_computation();
        }
    }

    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.forward.data()
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
{
    pub fn parameters(&self) -> Vec<Param> {
        self.parameters.iter().cloned().collect()
    }

    pub fn sum(mut self) -> VarDiff<Sum<T>, SumBackward<U>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Sum::new(self.forward));
        let backward = Rc::new(SumBackward::new(self.backward));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn pow(mut self, exp: i32) -> VarDiff<Power<T>, PowerBackward<U, T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Power::new(self.forward.clone(), exp));
        let backward = Rc::new(PowerBackward::new(self.backward, self.forward, exp));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn relu(mut self) -> VarDiff<ReLU<T>, ReLUBackward<U, T>> {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(ReLU::new(self.forward.clone()));
        let backward = Rc::new(ReLUBackward::new(self.backward, self.forward));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn leaky_relu(mut self) -> VarDiff<LeakyReLU<T>, LeakyReLUBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(LeakyReLU::new(forward.clone())),
            Rc::new(LeakyReLUBackward::new(backward, forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn softplus(mut self) -> VarDiff<SoftPlus<T>, SoftPlusBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(SoftPlus::new(forward.clone())),
            Rc::new(SoftPlusBackward::new(backward, forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn sigmoid(mut self) -> VarDiff<Sigmoid<T>, SigmoidBackward<U, Sigmoid<T>>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Sigmoid::new(forward)),
        );
        let backward = Rc::new(SigmoidBackward::new(backward, forward.clone()));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn tanh(mut self) -> VarDiff<TanH<T>, TanHBackward<U, TanH<T>>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(TanH::new(forward)),
        );
        let backward = Rc::new(TanHBackward::new(backward, forward.clone()));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn ln(mut self) -> VarDiff<Logn<T>, LognBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Logn::new(forward.clone())),
            Rc::new(LognBackward::new(backward, forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }
    pub fn exp(mut self) -> VarDiff<Exp<T>, ExpBackward<U, Exp<T>>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Exp::new(forward)),
        );
        let backward = Rc::new(ExpBackward::new(backward, forward.clone()));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn softmax(mut self, axis: usize) -> VarDiff<Softmax<T>, SoftmaxBackward<U, Softmax<T>>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Softmax::new(forward, axis)),
        );
        let backward = Rc::new(SoftmaxBackward::new(backward, forward.clone(), axis));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn log_softmax(
        mut self,
        axis: usize,
    ) -> VarDiff<LogSoftmax<T>, LogSoftmaxBackward<U, LogSoftmax<T>>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(LogSoftmax::new(forward, axis)),
        );
        let backward = Rc::new(LogSoftmaxBackward::new(backward, forward.clone(), axis));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn t(mut self) -> VarDiff<Transpose<T>, TransposeBackward<U>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Transpose::new(forward)),
            Rc::new(TransposeBackward::new(backward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }

    pub fn chunks<E: IntoDimension<Dim = T::Dim>>(
        self,
        chunk_size: E,
    ) -> Vec<VarDiff<Chunk<T>, ChunkBackward<U>>> {
        let (forward, backward) = (self.forward, self.backward);
        let (forward_path, backward_path) = (self.forward_path, self.backward_path);
        let parameters = self.parameters;
        let data = forward.data();
        let chunks = data.exact_chunks(chunk_size).into_iter().enumerate();
        chunks
            .map(|(i, chunk)| {
                let (id, forward, backward) = (
                    unsafe { OPERATIONS_COUNTER.next() },
                    Rc::new(Chunk::new(forward.clone(), chunk.to_owned(), i)),
                    Rc::new(ChunkBackward::new(backward.clone(), chunk.map(|_| 0.0), i)),
                );

                let mut new_forward_path = forward_path.clone();
                new_forward_path.insert(id, forward.clone() as Rc<dyn Forward>);
                let mut new_backward_path = backward_path.clone();
                new_backward_path.insert(id, backward.clone() as Rc<dyn Backward>);

                VarDiff {
                    id,
                    forward,
                    backward,
                    forward_path: new_forward_path,
                    backward_path: new_backward_path,
                    forward_buffer: Vec::new(),
                    backward_buffer: Vec::new(),
                    parameters: parameters.clone(),
                }
            })
            .collect()
    }

    pub fn dropout(mut self, p: f64) -> VarDiff<Dropout<T>, DropoutBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Dropout::new(forward, p)),
        );
        let backward = Rc::new(DropoutBackward::new(backward, forward.clone(), p));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }
}

impl<T, U> VarDiff<Dropout<T>, DropoutBackward<U, T>>
where
    T: Data,
    U: Gradient<Dim = T::Dim> + Overwrite,
{
    pub fn train(&self, status: bool) {
        self.forward.set_train(status);
        self.backward.set_train(status);
    }
}

impl<T, U> VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
    T::Dim: RemoveAxis,
{
    pub fn unsqueeze(mut self, axis: usize) -> VarDiff<Unsqueeze<T>, UnsqueezeBackward<U>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Unsqueeze::new(forward, axis)),
            Rc::new(UnsqueezeBackward::new(backward, axis)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arithmetic Operations Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T: Data + 'static> Neg for Var<T> {
    type Output = Var<Negation<T>>;

    fn neg(mut self) -> Self::Output {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Negation::new(self.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }
}

impl<T, U> Neg for VarDiff<T, U>
where
    T: Data + 'static,
    U: Gradient<Dim = T::Dim> + Overwrite + 'static,
{
    type Output = VarDiff<Negation<T>, NegationBackward<U>>;

    fn neg(mut self) -> Self::Output {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Negation::new(self.forward));
        let backward = Rc::new(NegationBackward::new(self.backward));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
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

    fn add(mut self, mut rhs: Var<Rhs>) -> Self::Output {
        self.path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Addition::new(self.node, rhs.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Self::Output {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
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

    fn add(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        rhs.forward_path.append(&mut self.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Addition::new(self.node.clone(), rhs.forward));
        let backward = Rc::new(AdditionBackwardUnary::new(rhs.backward, self.node));
        rhs.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: rhs.forward_path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
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

    fn add(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = rhs.node;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Addition::new(lhs_forward, rhs_forward.clone())),
            Rc::new(AdditionBackwardUnary::new(lhs_backward, rhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
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

    fn add(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);
        self.backward_path.append(&mut rhs.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (rhs.forward, rhs.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Addition::new(lhs_forward, rhs_forward)),
            Rc::new(AdditionBackward::new(lhs_backward, rhs_backward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&rhs.parameters).cloned().collect(),
        }
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

    fn sub(mut self, mut rhs: Var<Rhs>) -> Self::Output {
        self.path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Subtraction::new(self.node, rhs.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Self::Output {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
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

    fn sub(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        rhs.forward_path.append(&mut self.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Subtraction::new(self.node.clone(), rhs.forward));
        let backward = Rc::new(SubtractionBackwardRight::new(rhs.backward, self.node));
        rhs.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: rhs.forward_path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
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

    fn sub(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Subtraction::new(self.forward, rhs.node.clone()));
        let backward = Rc::new(SubtractionBackwardLeft::new(self.backward, rhs.node));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
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

    fn sub(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);
        self.backward_path.append(&mut rhs.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (rhs.forward, rhs.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Subtraction::new(lhs_forward, rhs_forward)),
            Rc::new(SubtractionBackward::new(lhs_backward, rhs_backward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&rhs.parameters).cloned().collect(),
        }
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

    fn mul(mut self, mut rhs: Var<Rhs>) -> Self::Output {
        self.path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Multiplication::new(self.node, rhs.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Self::Output {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
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

    fn mul(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        rhs.forward_path.append(&mut self.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Multiplication::new(self.node.clone(), rhs.forward));
        let backward = Rc::new(MultiplicationBackwardUnary::new(rhs.backward, self.node));
        rhs.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: rhs.forward_path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
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

    fn mul(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Multiplication::new(self.forward, rhs.node.clone()));
        let backward = Rc::new(MultiplicationBackwardUnary::new(self.backward, rhs.node));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
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

    fn mul(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);
        self.backward_path.append(&mut rhs.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (rhs.forward, rhs.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Multiplication::new(
                lhs_forward.clone(),
                rhs_forward.clone(),
            )),
            Rc::new(MultiplicationBackward::new(
                lhs_forward,
                lhs_backward,
                rhs_forward,
                rhs_backward,
            )),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&rhs.parameters).cloned().collect(),
        }
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

    fn div(mut self, mut rhs: Var<Rhs>) -> Self::Output {
        self.path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Division::new(self.node, rhs.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Self::Output {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
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

    fn div(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        rhs.forward_path.append(&mut self.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Division::new(self.node.clone(), rhs.forward.clone()));
        let backward = Rc::new(DivisionBackwardRight::new(
            self.node,
            rhs.forward,
            rhs.backward,
        ));
        rhs.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: rhs.forward_path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
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

    fn div(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Division::new(self.forward, rhs.node.clone()));
        let backward = Rc::new(DivisionBackwardLeft::new(self.backward, rhs.node));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
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

    fn div(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);
        self.backward_path.append(&mut rhs.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (rhs.forward, rhs.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Division::new(lhs_forward.clone(), rhs_forward.clone())),
            Rc::new(DivisionBackward::new(
                lhs_forward,
                lhs_backward,
                rhs_forward,
                rhs_backward,
            )),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&rhs.parameters).cloned().collect(),
        }
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Operations Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2, B2> MatMatMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixMatrixMul<F1, F2>, MatrixMatrixMulBackward<F1, B1, F2, B2>>;

    fn mm_mul(mut self, mut other: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);
        self.backward_path.append(&mut other.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (other.forward, other.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(MatrixMatrixMul::new(
                lhs_forward.clone(),
                rhs_forward.clone(),
            )),
            Rc::new(MatrixMatrixMulBackward::new(
                lhs_forward,
                lhs_backward,
                rhs_forward,
                rhs_backward,
            )),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&other.parameters).cloned().collect(),
        }
    }
}

impl<F1, B1, F2> MatMatMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = VarDiff<MatrixMatrixMul<F1, F2>, MatrixMatrixMulBackwardLeft<B1, F2>>;

    fn mm_mul(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(MatrixMatrixMul::new(self.forward, rhs.node.clone()));
        let backward = Rc::new(MatrixMatrixMulBackwardLeft::new(self.backward, rhs.node));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> MatMatMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixMatrixMul<F1, F2>, MatrixMatrixMulBackwardRight<F1, B2>>;

    fn mm_mul(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        self.path.append(&mut rhs.forward_path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(MatrixMatrixMul::new(self.node.clone(), rhs.forward));
        let backward = Rc::new(MatrixMatrixMulBackwardRight::new(self.node, rhs.backward));
        self.path.insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);
        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
    }
}

impl<F1, F2> MatMatMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = Var<MatrixMatrixMul<F1, F2>>;

    fn mm_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.path.append(&mut other.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(MatrixMatrixMul::new(self.node, other.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication with Transposition  ~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2, B2> MatMatMulT<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixMatrixMulT<F1, F2>, MatrixMatrixMulTBackward<F1, B1, F2, B2>>;

    fn mm_mul_t(mut self, mut other: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);
        self.backward_path.append(&mut other.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (other.forward, other.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(MatrixMatrixMulT::new(
                lhs_forward.clone(),
                rhs_forward.clone(),
            )),
            Rc::new(MatrixMatrixMulTBackward::new(
                lhs_forward,
                lhs_backward,
                rhs_forward,
                rhs_backward,
            )),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&other.parameters).cloned().collect(),
        }
    }
}

impl<F1, B1, F2> MatMatMulT<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = VarDiff<MatrixMatrixMulT<F1, F2>, MatrixMatrixMulTBackwardLeft<B1, F2>>;

    fn mm_mul_t(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(MatrixMatrixMulT::new(self.forward, rhs.node.clone()));
        let backward = Rc::new(MatrixMatrixMulTBackwardLeft::new(self.backward, rhs.node));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> MatMatMulT<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixMatrixMulT<F1, F2>, MatrixMatrixMulTBackwardRight<F1, B2>>;

    fn mm_mul_t(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        self.path.append(&mut rhs.forward_path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(MatrixMatrixMulT::new(self.node.clone(), rhs.forward));
        let backward = Rc::new(MatrixMatrixMulTBackwardRight::new(self.node, rhs.backward));
        self.path.insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);
        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
    }
}

impl<F1, F2> MatMatMulT<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = Var<MatrixMatrixMulT<F1, F2>>;

    fn mm_mul_t(mut self, mut other: Var<F2>) -> Self::Output {
        self.path.append(&mut other.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(MatrixMatrixMulT::new(self.node, other.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2, B2> MatVecMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
    B2: Gradient<Dim = Ix1> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixVectorMul<F1, F2>, MatrixVectorMulBackward<F1, B1, F2, B2>>;

    fn mv_mul(mut self, mut other: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);
        self.backward_path.append(&mut other.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (other.forward, other.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(MatrixVectorMul::new(
                lhs_forward.clone(),
                rhs_forward.clone(),
            )),
            Rc::new(MatrixVectorMulBackward::new(
                lhs_forward,
                lhs_backward,
                rhs_forward,
                rhs_backward,
            )),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&other.parameters).cloned().collect(),
        }
    }
}

impl<F1, B1, F2> MatVecMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + 'static,
    B1: Gradient<Dim = Ix2> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = VarDiff<MatrixVectorMul<F1, F2>, MatrixVectorMulBackwardLeft<B1, F2>>;
    fn mv_mul(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(MatrixVectorMul::new(self.forward, rhs.node.clone()));
        let backward = Rc::new(MatrixVectorMulBackwardLeft::new(self.backward, rhs.node));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> MatVecMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix1> + 'static,
    B2: Gradient<Dim = Ix1> + Overwrite + 'static,
{
    type Output = VarDiff<MatrixVectorMul<F1, F2>, MatrixVectorMulBackwardRight<F1, B2>>;

    fn mv_mul(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        self.path.append(&mut rhs.forward_path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(MatrixVectorMul::new(self.node.clone(), rhs.forward));
        let backward = Rc::new(MatrixVectorMulBackwardRight::new(self.node, rhs.backward));
        self.path.insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);
        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
    }
}

impl<F1, F2> MatVecMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = Var<MatrixVectorMul<F1, F2>>;

    fn mv_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.path.append(&mut other.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(MatrixVectorMul::new(self.node, other.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl<F1, B1, F2, B2> VecMatMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<VectorMatrixMul<F1, F2>, VectorMatrixMulBackward<F1, B1, F2, B2>>;

    fn vm_mul(mut self, mut other: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);
        self.backward_path.append(&mut other.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (other.forward, other.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(VectorMatrixMul::new(
                lhs_forward.clone(),
                rhs_forward.clone(),
            )),
            Rc::new(VectorMatrixMulBackward::new(
                lhs_forward,
                lhs_backward,
                rhs_forward,
                rhs_backward,
            )),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&other.parameters).cloned().collect(),
        }
    }
}

impl<F1, B1, F2> VecMatMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = VarDiff<VectorMatrixMul<F1, F2>, VectorMatrixMulBackwardLeft<B1, F2>>;
    fn vm_mul(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(VectorMatrixMul::new(self.forward, rhs.node.clone()));
        let backward = Rc::new(VectorMatrixMulBackwardLeft::new(self.backward, rhs.node));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> VecMatMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix2> + 'static,
    B2: Gradient<Dim = Ix2> + Overwrite + 'static,
{
    type Output = VarDiff<VectorMatrixMul<F1, F2>, VectorMatrixMulBackwardRight<F1, B2>>;

    fn vm_mul(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        self.path.append(&mut rhs.forward_path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(VectorMatrixMul::new(self.node.clone(), rhs.forward));
        let backward = Rc::new(VectorMatrixMulBackwardRight::new(self.node, rhs.backward));
        self.path.insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);
        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
    }
}

impl<F1, F2> VecMatMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix2> + 'static,
{
    type Output = Var<VectorMatrixMul<F1, F2>>;

    fn vm_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.path.append(&mut other.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(VectorMatrixMul::new(self.node, other.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2, B2> VecVecMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
    B2: Gradient<Dim = Ix1> + Overwrite + 'static,
{
    type Output = VarDiff<VectorVectorMul<F1, F2>, VectorVectorMulBackward<F1, B1, F2, B2>>;
    fn vv_mul(mut self, mut other: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);
        self.backward_path.append(&mut other.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (other.forward, other.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(VectorVectorMul::new(
                lhs_forward.clone(),
                rhs_forward.clone(),
            )),
            Rc::new(VectorVectorMulBackward::new(
                lhs_forward,
                lhs_backward,
                rhs_forward,
                rhs_backward,
            )),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&other.parameters).cloned().collect(),
        }
    }
}

impl<F1, B1, F2> VecVecMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + 'static,
    B1: Gradient<Dim = Ix1> + Overwrite + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = VarDiff<VectorVectorMul<F1, F2>, VectorVectorMulBackwardUnary<B1, F2>>;
    fn vv_mul(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(VectorVectorMul::new(self.forward, rhs.node.clone()));
        let backward = Rc::new(VectorVectorMulBackwardUnary::new(self.backward, rhs.node));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> VecVecMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix1> + 'static,
    B2: Gradient<Dim = Ix1> + Overwrite + 'static,
{
    type Output = VarDiff<VectorVectorMul<F1, F2>, VectorVectorMulBackwardUnary<B2, F1>>;

    fn vv_mul(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        self.path.append(&mut rhs.forward_path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(VectorVectorMul::new(self.node.clone(), rhs.forward));
        let backward = Rc::new(VectorVectorMulBackwardUnary::new(rhs.backward, self.node));
        self.path.insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
    }
}

impl<F1, F2> VecVecMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + 'static,
    F2: Data<Dim = Ix1> + 'static,
{
    type Output = Var<VectorVectorMul<F1, F2>>;

    fn vv_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.path.append(&mut other.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(VectorVectorMul::new(self.node, other.node));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cat and Stack traits implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    fn cat(mut self, mut other: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);
        self.backward_path.append(&mut other.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (other.forward, other.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Concatenate::new(lhs_forward, rhs_forward, axis)),
            Rc::new(ConcatenateBackward::new(lhs_backward, rhs_backward, axis)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&other.parameters).cloned().collect(),
        }
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
    fn cat(mut self, mut rhs: Var<F2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Concatenate::new(self.forward, rhs.node.clone(), axis));
        let backward = Rc::new(ConcatenateBackwardLeft::new(self.backward, rhs.node, axis));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
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
    fn cat(mut self, mut rhs: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        self.path.append(&mut rhs.forward_path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(Concatenate::new(self.node.clone(), rhs.forward, axis));
        let backward = Rc::new(ConcatenateBackwardRight::new(self.node, rhs.backward, axis));
        self.path.insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
    }
}

impl<F1, F2> Cat<Var<F2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    F1::Dim: RemoveAxis,
{
    type Output = Var<Concatenate<F1, F2>>;
    fn cat(mut self, mut other: Var<F2>, axis: usize) -> Self::Output {
        self.path.append(&mut other.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(Concatenate::new(self.node, other.node, axis));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    fn stack(mut self, mut other: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);
        self.backward_path.append(&mut other.backward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let (rhs_forward, rhs_backward) = (other.forward, other.backward);

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(StackF::new(lhs_forward, rhs_forward, axis)),
            Rc::new(StackBackward::new(lhs_backward, rhs_backward, axis)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters.union(&other.parameters).cloned().collect(),
        }
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
    fn stack(mut self, mut rhs: Var<F2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut rhs.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(StackF::new(self.forward, rhs.node.clone(), axis));
        let backward = Rc::new(StackBackwardLeft::new(self.backward, rhs.node, axis));
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: self.parameters,
        }
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
    fn stack(mut self, mut rhs: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        self.path.append(&mut rhs.forward_path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(StackF::new(self.node.clone(), rhs.forward, axis));
        let backward = Rc::new(StackBackwardRight::new(self.node, rhs.backward, axis));
        self.path.insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.path,
            backward_path: rhs.backward_path,
            forward_buffer: Vec::new(),
            backward_buffer: Vec::new(),
            parameters: rhs.parameters,
        }
    }
}

impl<F1, F2> Stack<Var<F2>> for Var<F1>
where
    F1: Data + 'static,
    F2: Data<Dim = F1::Dim> + 'static,
    F1::Dim: RemoveAxis,
{
    type Output = Var<StackF<F1, F2>>;

    fn stack(mut self, mut other: Var<F2>, axis: usize) -> Self::Output {
        self.path.append(&mut other.path);

        let id = unsafe { OPERATIONS_COUNTER.next() };
        let node = Rc::new(StackF::new(self.node, other.node, axis));
        self.path.insert(id, node.clone() as Rc<dyn Forward>);

        Var {
            id,
            node,
            path: self.path,
            buffer: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod var {
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
