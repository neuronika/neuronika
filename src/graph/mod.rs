pub mod node;
use itertools::Itertools;
use ndarray::{
    Array, ArrayD, DimMax, Dimension, IntoDimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn,
    RemoveAxis,
};
use node::{
    backward::{Backward, Differentiable, Gradient},
    forward::{Data, Forward},
    Addition, AdditionBackward, AdditionBackwardUnary, Chunk, ChunkBackward, Concatenate,
    ConcatenateBackward, ConcatenateBackwardLeft, ConcatenateBackwardRight, Division,
    DivisionBackward, DivisionBackwardLeft, DivisionBackwardRight, Exp, ExpBackward, LeakyReLU,
    LeakyReLUBackward, LogSoftmax, LogSoftmaxBackward, Logn, LognBackward, MatrixMatrixMul,
    MatrixMatrixMulBackward, MatrixMatrixMulBackwardLeft, MatrixMatrixMulBackwardRight,
    MatrixVectorMul, MatrixVectorMulBackward, MatrixVectorMulBackwardLeft,
    MatrixVectorMulBackwardRight, Multiplication, MultiplicationBackward,
    MultiplicationBackwardUnary, Negation, NegationBackward, Power, PowerBackward, ReLU,
    ReLUBackward, Sigmoid, SigmoidBackward, SoftPlus, SoftPlusBackward, Softmax, SoftmaxBackward,
    Stack as StackF, StackBackward, StackBackwardLeft, StackBackwardRight, Subtraction,
    SubtractionBackward, SubtractionBackwardLeft, SubtractionBackwardRight, Sum, SumBackward, TanH,
    TanHBackward, Transpose, TransposeBackward, Unsqueeze, UnsqueezeBackward, VectorMatrixMul,
    VectorMatrixMulBackward, VectorMatrixMulBackwardLeft, VectorMatrixMulBackwardRight,
    VectorVectorMul, VectorVectorMulBackward, VectorVectorMulBackwardUnary,
};
pub use node::{Input, InputBackward};
use std::{
    cell::{Ref, RefMut},
    collections::BTreeMap,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Type Aliases ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;
pub(crate) type BroadTensor<Lhs, Rhs> = Tensor<Broadcasted<Lhs, Rhs>>;
pub(crate) type Tensor<D> = Array<f32, D>;
pub(crate) type DynTensor = ArrayD<f32>;

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

pub(crate) static mut OPERATIONS_COUNTER: OperationsCounter = OperationsCounter { count: 0 };

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ParamDim Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub trait ParamDim: Dimension + 'static {
    fn insert(item: Param<Self>, dest: &mut Parameters);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ParamDim Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl ParamDim for Ix1 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.oned.push(item);
    }
}

impl ParamDim for Ix2 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.twod.push(item);
    }
}

impl ParamDim for Ix3 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.threed.push(item);
    }
}

impl ParamDim for Ix4 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.fourd.push(item);
    }
}

impl ParamDim for Ix5 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.fived.push(item);
    }
}

impl ParamDim for Ix6 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.sixd.push(item);
    }
}

impl ParamDim for IxDyn {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.dynd.push(item);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Param Struct ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Param<D>
where
    D: Dimension,
{
    id: usize,
    input: Rc<Input<D>>,
    input_diff: Rc<InputBackward<D>>,
}

impl<D> Param<D>
where
    D: Dimension,
{
    fn get_id(&self) -> usize {
        self.id
    }

    fn zero_grad(&self) {
        self.input_diff.zero_grad();
    }

    pub(crate) fn data_mut(&self) -> RefMut<Tensor<D>> {
        self.input.data_mut()
    }

    pub(crate) fn grad(&self) -> Ref<Tensor<D>> {
        self.input_diff.gradient()
    }
}

impl<D> Clone for Param<D>
where
    D: Dimension,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            input: self.input.clone(),
            input_diff: self.input_diff.clone(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Parameters struct ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Clone)]
/// Contains the learnable ancestors of the node.
pub struct Parameters {
    // Contains the one dimensional learnable ancestors
    oned: Vec<Param<Ix1>>,
    // Contains the two dimensional learnable ancestors
    twod: Vec<Param<Ix2>>,
    // Contains the three dimensional learnable ancestors
    threed: Vec<Param<Ix3>>,
    // Contains the four dimensional learnable ancestors
    fourd: Vec<Param<Ix4>>,
    // Contains the five dimensional learnable ancestors
    fived: Vec<Param<Ix5>>,
    // Contains the six dimensional learnable ancestors
    sixd: Vec<Param<Ix6>>,
    // Contains the dynamic dimensional learnable ancestors
    dynd: Vec<Param<IxDyn>>,
}

impl Parameters {
    fn new() -> Parameters {
        Parameters {
            oned: Vec::new(),
            twod: Vec::new(),
            threed: Vec::new(),
            fourd: Vec::new(),
            fived: Vec::new(),
            sixd: Vec::new(),
            dynd: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.oned.len()
            + self.twod.len()
            + self.threed.len()
            + self.fourd.len()
            + self.fived.len()
            + self.sixd.len()
            + self.dynd.len()
    }

    pub(crate) fn get_oned(&self) -> &[Param<Ix1>] {
        &self.oned
    }
    pub(crate) fn get_twod(&self) -> &[Param<Ix2>] {
        &self.twod
    }
    pub(crate) fn get_threed(&self) -> &[Param<Ix3>] {
        &self.threed
    }
    pub(crate) fn get_fourd(&self) -> &[Param<Ix4>] {
        &self.fourd
    }
    pub(crate) fn get_fived(&self) -> &[Param<Ix5>] {
        &self.fived
    }
    pub(crate) fn get_sixd(&self) -> &[Param<Ix6>] {
        &self.sixd
    }
    pub(crate) fn get_dynd(&self) -> &[Param<IxDyn>] {
        &self.dynd
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~ Functions to keep track of differentiable history ~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn merge_parameters(lhs_params: Parameters, rhs_params: Parameters) -> Parameters {
    Parameters {
        oned: merge(&lhs_params.oned[..], &rhs_params.oned[..]),
        twod: merge(&lhs_params.twod[..], &rhs_params.twod[..]),
        threed: merge(&lhs_params.threed[..], &rhs_params.threed[..]),
        fourd: merge(&lhs_params.fourd[..], &rhs_params.fourd[..]),
        fived: merge(&lhs_params.fived[..], &rhs_params.fived[..]),
        sixd: merge(&lhs_params.sixd[..], &rhs_params.sixd[..]),
        dynd: merge(&lhs_params.dynd[..], &rhs_params.dynd[..]),
    }
}

fn merge<D: ParamDim>(lhs_up: &[Param<D>], rhs_up: &[Param<D>]) -> Vec<Param<D>> {
    lhs_up
        .iter()
        .merge_join_by(rhs_up.iter(), |lhs_par, rhs_par| {
            lhs_par.get_id().cmp(&rhs_par.get_id())
        })
        .map(|choice| match choice {
            itertools::EitherOrBoth::Left(lhs_par) => lhs_par,
            itertools::EitherOrBoth::Right(rhs_par) => rhs_par,
            itertools::EitherOrBoth::Both(lhs_par, _) => lhs_par,
        })
        .cloned()
        .collect()
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub trait MatMatMul<Rhs> {
    type Output;
    fn mm_mul(self, other: Rhs) -> Self::Output;
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
    T: Data + Forward + 'static,
{
    pub(crate) id: usize,
    pub(crate) forward: Rc<T>,
    pub(crate) forward_path: BTreeMap<usize, Rc<dyn Forward>>,
    pub(crate) forward_buffer: Vec<Rc<dyn Forward>>,
}

impl<T> Var<T>
where
    T: Data + Forward + 'static,
{
    pub fn new(node: T) -> Self {
        let id = unsafe { OPERATIONS_COUNTER.next() };
        let forward = Rc::new(node);
        let mut forward_path = BTreeMap::new();
        forward_path.insert(id, forward.clone() as Rc<dyn Forward>);

        Self {
            id,
            forward,
            forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

impl<T> Clone for Var<T>
where
    T: Data + Forward + 'static,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            forward: self.forward.clone(),
            forward_path: self.forward_path.clone(),
            forward_buffer: self.forward_buffer.clone(),
        }
    }
}

impl<D> Var<Input<D>>
where
    D: ParamDim,
{
    pub fn requires_grad(self) -> VarDiff<Input<D>, InputBackward<D>> {
        let (id, forward) = (self.id, self.forward);
        if Rc::strong_count(&forward) > 2 {
            panic!("error: cannot make the Input differentiable.")
        }
        let backward = Rc::new(forward.differentiable());
        let forward_path = self.forward_path;
        let mut backward_path = BTreeMap::new();
        backward_path.insert(id, backward.clone() as Rc<dyn Backward>);
        let mut parameters = Parameters::new();
        D::insert(
            Param {
                id,
                input: forward.clone(),
                input_diff: backward.clone(),
            },
            &mut parameters,
        );
        debug_assert_eq!(forward_path.contains_key(&id), true);
        VarDiff {
            id,
            forward,
            backward,
            forward_path,
            backward_path,
            parameters,
        }
    }
}

impl<T> Var<T>
where
    T: Data + Forward + 'static,
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

        if let Err(i) = res {
            if self.forward_buffer.get(i).is_some() {
                res = Ok(i);
            }
        };

        if let Ok(pos) = res {
            for node in &self.forward_buffer[pos..] {
                node.forward();
            }
        }
    }

    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.forward.data()
    }

    pub fn sum(mut self) -> Var<Sum<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Sum::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn pow(mut self, exp: i32) -> Var<Power<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Power::new(forward, exp)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn relu(mut self) -> Var<ReLU<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(ReLU::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn leaky_relu(mut self) -> Var<LeakyReLU<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(LeakyReLU::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn softplus(mut self) -> Var<SoftPlus<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(SoftPlus::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn sigmoid(mut self) -> Var<Sigmoid<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Sigmoid::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn tanh(mut self) -> Var<TanH<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(TanH::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn ln(mut self) -> Var<Logn<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Logn::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn exp(mut self) -> Var<Exp<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Exp::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn softmax(mut self, axis: usize) -> Var<Softmax<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Softmax::new(forward, axis)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn log_softmax(mut self, axis: usize) -> Var<LogSoftmax<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(LogSoftmax::new(forward, axis)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn t(mut self) -> Var<Transpose<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Transpose::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }

    pub fn chunks<E: IntoDimension<Dim = T::Dim>>(self, chunk_size: E) -> Vec<Var<Chunk<T>>> {
        let forward = self.forward;
        let forward_path = self.forward_path;
        let data = forward.data();
        let chunks = data.exact_chunks(chunk_size).into_iter().enumerate();
        chunks
            .map(|(i, chunk)| {
                let (id, forward) = (
                    unsafe { OPERATIONS_COUNTER.next() },
                    Rc::new(Chunk::new(forward.clone(), chunk.to_owned(), i)),
                );

                let mut new_forward_path = forward_path.clone();
                new_forward_path.insert(id, forward.clone() as Rc<dyn Forward>);

                Var {
                    id,
                    forward,
                    forward_path: new_forward_path,
                    forward_buffer: Vec::new(),
                }
            })
            .collect()
    }
}

impl<T, D> Var<T>
where
    T: Data<Dim = D> + Forward + 'static,
    D: Dimension + RemoveAxis,
{
    pub fn unsqueeze(mut self, axis: usize) -> Var<Unsqueeze<T>> {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Unsqueeze::new(forward, axis)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Differentiable Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient + Backward + 'static,
{
    pub(crate) id: usize,
    pub(crate) forward: Rc<T>,
    pub(crate) backward: Rc<U>,
    pub(crate) forward_path: BTreeMap<usize, Rc<dyn Forward>>,
    pub(crate) backward_path: BTreeMap<usize, Rc<dyn Backward>>,
    pub(crate) parameters: Parameters,
}

impl<T, U> Clone for VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient + Backward + 'static,
{
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            forward: self.forward.clone(),
            backward: self.backward.clone(),
            forward_path: self.forward_path.clone(),
            backward_path: self.backward_path.clone(),
            parameters: self.parameters.clone(),
        }
    }
}

impl<D> VarDiff<Input<D>, InputBackward<D>>
where
    D: ParamDim,
{
    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.backward.gradient()
    }

    pub(crate) fn data_mut(&mut self) -> RefMut<Tensor<D>> {
        self.forward.data_mut()
    }
}

impl<T, U, D> VarDiff<T, U>
where
    T: Data<Dim = D> + Forward + 'static,
    U: Gradient<Dim = D> + Backward + 'static,
    D: Dimension,
{
    pub fn forward(&self) {
        for f in &self.forward_path {
            f.1.forward();
        }
    }
    pub fn backward(&self, seed: f32) {
        self.backward.gradient_mut().map_inplace(|el| *el = seed);
        for b in &self.backward_path {
            b.1.backward();
        }
    }
    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.forward.data()
    }
    pub fn zero_gradient(&mut self) {
        for param in &mut self.parameters.oned[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.twod[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.threed[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.fourd[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.fived[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.sixd[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.dynd[..] {
            param.zero_grad();
        }
    }
    pub fn parameters(&self) -> &Parameters {
        &self.parameters
    }

    pub fn sum(mut self) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Sum::new(forward)),
            Rc::new(SumBackward::new(backward)),
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
            parameters: self.parameters,
        }
    }

    pub fn pow(mut self, exp: i32) -> VarDiff<Power<T>, PowerBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Power::new(forward.clone(), exp)),
            Rc::new(PowerBackward::new(backward, forward, exp)),
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
            parameters: self.parameters,
        }
    }

    pub fn relu(mut self) -> VarDiff<ReLU<T>, ReLUBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(ReLU::new(forward.clone())),
            Rc::new(ReLUBackward::new(backward, forward)),
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
            parameters: self.parameters,
        }
    }

    pub fn sigmoid(mut self) -> VarDiff<Sigmoid<T>, SigmoidBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Sigmoid::new(forward.clone())),
            Rc::new(SigmoidBackward::new(backward, forward)),
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
            parameters: self.parameters,
        }
    }

    pub fn tanh(mut self) -> VarDiff<TanH<T>, TanHBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(TanH::new(forward.clone())),
            Rc::new(TanHBackward::new(backward, forward)),
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
            parameters: self.parameters,
        }
    }
    pub fn exp(mut self) -> VarDiff<Exp<T>, ExpBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Exp::new(forward.clone())),
            Rc::new(ExpBackward::new(backward, forward)),
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
            parameters: self.parameters,
        }
    }

    pub fn softmax(mut self, axis: usize) -> VarDiff<Softmax<T>, SoftmaxBackward<U, T>> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Softmax::new(forward.clone(), axis)),
            Rc::new(SoftmaxBackward::new(backward, forward, axis)),
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
                    parameters: parameters.clone(),
                }
            })
            .collect()
    }
}

impl<T, U, D> VarDiff<T, U>
where
    T: Data<Dim = D> + Forward + 'static,
    U: Gradient<Dim = D> + Backward + 'static,
    D: Dimension + RemoveAxis,
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
            parameters: self.parameters,
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arithmetic Operations Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<T> Neg for Var<T>
where
    T: Data + Forward + 'static,
{
    type Output = Var<Negation<T>>;

    fn neg(mut self) -> Self::Output {
        let forward = self.forward;
        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Negation::new(forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

impl<T, U> Neg for VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient + Backward + 'static,
{
    type Output = VarDiff<Negation<T>, NegationBackward<U>>;

    fn neg(mut self) -> Self::Output {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Negation::new(forward)),
            Rc::new(NegationBackward::new(backward)),
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
            parameters: self.parameters,
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Addition ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<Lhs, Rhs> Add<Var<Rhs>> for Var<Lhs>
where
    Lhs: Data + Forward + 'static,
    Rhs: Data + Forward + 'static,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Output = Var<Addition<Lhs, Rhs>>;

    fn add(mut self, mut rhs: Var<Rhs>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);

        let lhs_forward = self.forward;
        let rhs_forward = rhs.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Addition::new(lhs_forward, rhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Self::Output {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

impl<F1, F2, B2> Add<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B2: Gradient + Backward + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    <B2 as Gradient>::Dim: DimMax<<F1 as Data>::Dim>,
{
    type Output = VarDiff<Addition<F1, F2>, AdditionBackwardUnary<B2, F1>>;

    fn add(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        rhs.forward_path.append(&mut self.forward_path);

        let lhs_forward = self.forward;
        let (rhs_forward, rhs_backward) = (rhs.forward, rhs.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Addition::new(lhs_forward.clone(), rhs_forward)),
            Rc::new(AdditionBackwardUnary::new(rhs_backward, lhs_forward)),
        );
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
            parameters: rhs.parameters,
        }
    }
}

impl<F1, B1, F2> Add<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    <B1 as Gradient>::Dim: DimMax<<F2 as Data>::Dim>,
{
    type Output = VarDiff<Addition<F1, F2>, AdditionBackwardUnary<B1, F2>>;

    fn add(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = rhs.forward;

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
            parameters: self.parameters,
        }
    }
}

impl<F1, B1, F2, B2> Add<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    B2: Gradient + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, rhs.parameters),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Subtraction ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<Lhs, Rhs> Sub<Var<Rhs>> for Var<Lhs>
where
    Lhs: Data + Forward + 'static,
    Rhs: Data + Forward + 'static,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Output = Var<Subtraction<Lhs, Rhs>>;

    fn sub(mut self, mut rhs: Var<Rhs>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);

        let lhs_forward = self.forward;
        let rhs_forward = rhs.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Subtraction::new(lhs_forward, rhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Self::Output {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

impl<F1, F2, B2> Sub<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B2: Gradient + Backward + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    <B2 as Gradient>::Dim: DimMax<<F1 as Data>::Dim>,
{
    type Output = VarDiff<Subtraction<F1, F2>, SubtractionBackwardRight<B2, F1>>;

    fn sub(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        rhs.forward_path.append(&mut self.forward_path);

        let lhs_forward = self.forward;
        let (rhs_forward, rhs_backward) = (rhs.forward, rhs.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Subtraction::new(lhs_forward.clone(), rhs_forward)),
            Rc::new(SubtractionBackwardRight::new(rhs_backward, lhs_forward)),
        );
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
            parameters: rhs.parameters,
        }
    }
}

impl<F1, B1, F2> Sub<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    <B1 as Gradient>::Dim: DimMax<<F2 as Data>::Dim>,
{
    type Output = VarDiff<Subtraction<F1, F2>, SubtractionBackwardLeft<B1, F2>>;

    fn sub(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = rhs.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Subtraction::new(lhs_forward, rhs_forward.clone())),
            Rc::new(SubtractionBackwardLeft::new(lhs_backward, rhs_forward)),
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
            parameters: self.parameters,
        }
    }
}

impl<F1, B1, F2, B2> Sub<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    B2: Gradient + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, rhs.parameters),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<Lhs, Rhs> Mul<Var<Rhs>> for Var<Lhs>
where
    Lhs: Data + Forward + 'static,
    Rhs: Data + Forward + 'static,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Output = Var<Multiplication<Lhs, Rhs>>;

    fn mul(mut self, mut rhs: Var<Rhs>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);

        let lhs_forward = self.forward;
        let rhs_forward = rhs.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Multiplication::new(lhs_forward, rhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Self::Output {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

impl<F1, F2, B2> Mul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B2: Gradient + Backward + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    <B2 as Gradient>::Dim: DimMax<<F1 as Data>::Dim>,
{
    type Output = VarDiff<Multiplication<F1, F2>, MultiplicationBackwardUnary<B2, F1>>;

    fn mul(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        rhs.forward_path.append(&mut self.forward_path);

        let lhs_forward = self.forward;
        let (rhs_forward, rhs_backward) = (rhs.forward, rhs.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Multiplication::new(lhs_forward.clone(), rhs_forward)),
            Rc::new(MultiplicationBackwardUnary::new(rhs_backward, lhs_forward)),
        );
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
            parameters: rhs.parameters,
        }
    }
}

impl<F1, B1, F2> Mul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    <B1 as Gradient>::Dim: DimMax<<F2 as Data>::Dim>,
{
    type Output = VarDiff<Multiplication<F1, F2>, MultiplicationBackwardUnary<B1, F2>>;

    fn mul(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = rhs.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Multiplication::new(lhs_forward, rhs_forward.clone())),
            Rc::new(MultiplicationBackwardUnary::new(lhs_backward, rhs_forward)),
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
            parameters: self.parameters,
        }
    }
}

impl<F1, B1, F2, B2> Mul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    B2: Gradient + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, rhs.parameters),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Division ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<Lhs, Rhs> Div<Var<Rhs>> for Var<Lhs>
where
    Lhs: Data + Forward + 'static,
    Rhs: Data + Forward + 'static,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Output = Var<Division<Lhs, Rhs>>;

    fn div(mut self, mut rhs: Var<Rhs>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);

        let lhs_forward = self.forward;
        let rhs_forward = rhs.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Division::new(lhs_forward, rhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Self::Output {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

impl<F1, F2, B2> Div<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B2: Gradient + Backward + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    <F1 as Data>::Dim: DimMax<<B2 as Gradient>::Dim>,
{
    type Output = VarDiff<Division<F1, F2>, DivisionBackwardRight<F1, F2, B2>>;

    fn div(mut self, mut rhs: VarDiff<F2, B2>) -> Self::Output {
        rhs.forward_path.append(&mut self.forward_path);

        let lhs_forward = self.forward;
        let (rhs_forward, rhs_backward) = (rhs.forward, rhs.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Division::new(lhs_forward.clone(), rhs_forward.clone())),
            Rc::new(DivisionBackwardRight::new(
                lhs_forward,
                rhs_forward,
                rhs_backward,
            )),
        );
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
            parameters: rhs.parameters,
        }
    }
}

impl<F1, B1, F2> Div<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    F1::Dim: Dimension + DimMax<F2::Dim>,
    <B1 as Gradient>::Dim: DimMax<<F2 as Data>::Dim>,
{
    type Output = VarDiff<Division<F1, F2>, DivisionBackwardLeft<B1, F2>>;

    fn div(mut self, mut rhs: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = rhs.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Division::new(lhs_forward, rhs_forward.clone())),
            Rc::new(DivisionBackwardLeft::new(lhs_backward, rhs_forward)),
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
            parameters: self.parameters,
        }
    }
}

impl<F1, B1, F2, B2> Div<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    F2: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    B2: Gradient + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, rhs.parameters),
        }
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Algebraic Operations Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Matrix Multiplication ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2, B2> MatMatMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + Forward + 'static,
    B1: Gradient<Dim = Ix2> + Backward + 'static,
    F2: Data<Dim = Ix2> + Forward + 'static,
    B2: Gradient<Dim = Ix2> + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, other.parameters),
        }
    }
}

impl<F1, B1, F2> MatMatMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + Forward + 'static,
    B1: Gradient<Dim = Ix2> + Backward + 'static,
    F2: Data<Dim = Ix2> + Forward + 'static,
{
    type Output = VarDiff<MatrixMatrixMul<F1, F2>, MatrixMatrixMulBackwardLeft<B1, F2>>;

    fn mm_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = other.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(MatrixMatrixMul::new(lhs_forward, rhs_forward.clone())),
            Rc::new(MatrixMatrixMulBackwardLeft::new(lhs_backward, rhs_forward)),
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
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> MatMatMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + Forward + 'static,
    F2: Data<Dim = Ix2> + Forward + 'static,
    B2: Gradient<Dim = Ix2> + Backward + 'static,
{
    type Output = VarDiff<MatrixMatrixMul<F1, F2>, MatrixMatrixMulBackwardRight<F1, B2>>;

    fn mm_mul(mut self, mut other: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (rhs_forward, rhs_backward) = (other.forward, other.backward);
        let lhs_forward = self.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(MatrixMatrixMul::new(lhs_forward.clone(), rhs_forward)),
            Rc::new(MatrixMatrixMulBackwardRight::new(lhs_forward, rhs_backward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: other.backward_path,
            parameters: other.parameters,
        }
    }
}

impl<F1, F2> MatMatMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + Forward + 'static,
    F2: Data<Dim = Ix2> + Forward + 'static,
{
    type Output = Var<MatrixMatrixMul<F1, F2>>;

    fn mm_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let rhs_forward = other.forward;
        let lhs_forward = self.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(MatrixMatrixMul::new(lhs_forward, rhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MatrixVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2, B2> MatVecMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + Forward + 'static,
    B1: Gradient<Dim = Ix2> + Backward + 'static,
    F2: Data<Dim = Ix1> + Forward + 'static,
    B2: Gradient<Dim = Ix1> + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, other.parameters),
        }
    }
}

impl<F1, B1, F2> MatVecMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix2> + Forward + 'static,
    B1: Gradient<Dim = Ix2> + Backward + 'static,
    F2: Data<Dim = Ix1> + Forward + 'static,
{
    type Output = VarDiff<MatrixVectorMul<F1, F2>, MatrixVectorMulBackwardLeft<B1, F2>>;
    fn mv_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = other.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(MatrixVectorMul::new(lhs_forward, rhs_forward.clone())),
            Rc::new(MatrixVectorMulBackwardLeft::new(lhs_backward, rhs_forward)),
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
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> MatVecMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + Forward + 'static,
    F2: Data<Dim = Ix1> + Forward + 'static,
    B2: Gradient<Dim = Ix1> + Backward + 'static,
{
    type Output = VarDiff<MatrixVectorMul<F1, F2>, MatrixVectorMulBackwardRight<F1, B2>>;

    fn mv_mul(mut self, mut other: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (rhs_forward, rhs_backward) = (other.forward, other.backward);
        let lhs_forward = self.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(MatrixVectorMul::new(lhs_forward.clone(), rhs_forward)),
            Rc::new(MatrixVectorMulBackwardRight::new(lhs_forward, rhs_backward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: other.backward_path,
            parameters: other.parameters,
        }
    }
}

impl<F1, F2> MatVecMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix2> + Forward + 'static,
    F2: Data<Dim = Ix1> + Forward + 'static,
{
    type Output = Var<MatrixVectorMul<F1, F2>>;

    fn mv_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let rhs_forward = other.forward;
        let lhs_forward = self.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(MatrixVectorMul::new(lhs_forward, rhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorMatrixMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl<F1, B1, F2, B2> VecMatMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + Forward + 'static,
    B1: Gradient<Dim = Ix1> + Backward + 'static,
    F2: Data<Dim = Ix2> + Forward + 'static,
    B2: Gradient<Dim = Ix2> + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, other.parameters),
        }
    }
}

impl<F1, B1, F2> VecMatMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + Forward + 'static,
    B1: Gradient<Dim = Ix1> + Backward + 'static,
    F2: Data<Dim = Ix2> + Forward + 'static,
{
    type Output = VarDiff<VectorMatrixMul<F1, F2>, VectorMatrixMulBackwardLeft<B1, F2>>;
    fn vm_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = other.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(VectorMatrixMul::new(lhs_forward, rhs_forward.clone())),
            Rc::new(VectorMatrixMulBackwardLeft::new(lhs_backward, rhs_forward)),
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
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> VecMatMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + Forward + 'static,
    F2: Data<Dim = Ix2> + Forward + 'static,
    B2: Gradient<Dim = Ix2> + Backward + 'static,
{
    type Output = VarDiff<VectorMatrixMul<F1, F2>, VectorMatrixMulBackwardRight<F1, B2>>;

    fn vm_mul(mut self, mut other: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (rhs_forward, rhs_backward) = (other.forward, other.backward);
        let lhs_forward = self.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(VectorMatrixMul::new(lhs_forward.clone(), rhs_forward)),
            Rc::new(VectorMatrixMulBackwardRight::new(lhs_forward, rhs_backward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: other.backward_path,
            parameters: other.parameters,
        }
    }
}

impl<F1, F2> VecMatMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + Forward + 'static,
    F2: Data<Dim = Ix2> + Forward + 'static,
{
    type Output = Var<VectorMatrixMul<F1, F2>>;

    fn vm_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let rhs_forward = other.forward;
        let lhs_forward = self.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(VectorMatrixMul::new(lhs_forward, rhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VectorVectorMul ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2, B2> VecVecMul<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + Forward + 'static,
    B1: Gradient<Dim = Ix1> + Backward + 'static,
    F2: Data<Dim = Ix1> + Forward + 'static,
    B2: Gradient<Dim = Ix1> + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, other.parameters),
        }
    }
}

impl<F1, B1, F2> VecVecMul<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = Ix1> + Forward + 'static,
    B1: Gradient<Dim = Ix1> + Backward + 'static,
    F2: Data<Dim = Ix1> + Forward + 'static,
{
    type Output = VarDiff<VectorVectorMul<F1, F2>, VectorVectorMulBackwardUnary<B1, F2>>;
    fn vv_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = other.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(VectorVectorMul::new(lhs_forward, rhs_forward.clone())),
            Rc::new(VectorVectorMulBackwardUnary::new(lhs_backward, rhs_forward)),
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
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> VecVecMul<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + Forward + 'static,
    F2: Data<Dim = Ix1> + Forward + 'static,
    B2: Gradient<Dim = Ix1> + Backward + 'static,
{
    type Output = VarDiff<VectorVectorMul<F1, F2>, VectorVectorMulBackwardUnary<B2, F1>>;

    fn vv_mul(mut self, mut other: VarDiff<F2, B2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (rhs_forward, rhs_backward) = (other.forward, other.backward);
        let lhs_forward = self.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(VectorVectorMul::new(lhs_forward.clone(), rhs_forward)),
            Rc::new(VectorVectorMulBackwardUnary::new(rhs_backward, lhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: other.backward_path,
            parameters: other.parameters,
        }
    }
}

impl<F1, F2> VecVecMul<Var<F2>> for Var<F1>
where
    F1: Data<Dim = Ix1> + Forward + 'static,
    F2: Data<Dim = Ix1> + Forward + 'static,
{
    type Output = Var<VectorVectorMul<F1, F2>>;

    fn vv_mul(mut self, mut other: Var<F2>) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let rhs_forward = other.forward;
        let lhs_forward = self.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(VectorVectorMul::new(lhs_forward, rhs_forward)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Cat and Stack traits implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Concatenate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2, B2> Cat<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    F2: Data<Dim = F1::Dim> + Forward + 'static,
    B2: Gradient<Dim = B1::Dim> + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, other.parameters),
        }
    }
}

impl<F1, B1, F2> Cat<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = B1::Dim> + Forward + 'static,
    F2: Data<Dim = F1::Dim> + Forward + 'static,
    B1: Gradient + Backward + 'static,
    F1::Dim: RemoveAxis,
    B1::Dim: RemoveAxis,
{
    type Output = VarDiff<Concatenate<F1, F2>, ConcatenateBackwardLeft<B1>>;
    fn cat(mut self, mut other: Var<F2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = other.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Concatenate::new(lhs_forward, rhs_forward.clone(), axis)),
            Rc::new(ConcatenateBackwardLeft::new(
                lhs_backward,
                rhs_forward,
                axis,
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
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> Cat<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data<Dim = B2::Dim> + Forward + 'static,
    F2: Data<Dim = F1::Dim> + Forward + 'static,
    B2: Gradient + Backward + 'static,
    F1::Dim: RemoveAxis,
    B2::Dim: RemoveAxis,
{
    type Output = VarDiff<Concatenate<F1, F2>, ConcatenateBackwardRight<B2>>;
    fn cat(mut self, mut other: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (rhs_forward, rhs_backward) = (other.forward, other.backward);
        let lhs_forward = self.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Concatenate::new(lhs_forward.clone(), rhs_forward, axis)),
            Rc::new(ConcatenateBackwardRight::new(
                lhs_forward,
                rhs_backward,
                axis,
            )),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: other.backward_path,
            parameters: other.parameters,
        }
    }
}

impl<F1, F2> Cat<Var<F2>> for Var<F1>
where
    F1: Data + Forward + 'static,
    F2: Data<Dim = F1::Dim> + Forward + 'static,
    F1::Dim: RemoveAxis,
{
    type Output = Var<Concatenate<F1, F2>>;
    fn cat(mut self, mut other: Var<F2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let lhs_forward = self.forward;
        let rhs_forward = other.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Concatenate::new(lhs_forward, rhs_forward, axis)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Stack ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl<F1, B1, F2, B2> Stack<VarDiff<F2, B2>> for VarDiff<F1, B1>
where
    F1: Data + Forward + 'static,
    B1: Gradient + Backward + 'static,
    F2: Data<Dim = F1::Dim> + Forward + 'static,
    B2: Gradient<Dim = B1::Dim> + Backward + 'static,
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
            parameters: merge_parameters(self.parameters, other.parameters),
        }
    }
}

impl<F1, B1, F2> Stack<Var<F2>> for VarDiff<F1, B1>
where
    F1: Data<Dim = B1::Dim> + Forward + 'static,
    F2: Data<Dim = F1::Dim> + Forward + 'static,
    B1: Gradient + Backward + 'static,
    F1::Dim: RemoveAxis,
    B1::Dim: RemoveAxis,
{
    type Output = VarDiff<StackF<F1, F2>, StackBackwardLeft<B1>>;
    fn stack(mut self, mut other: Var<F2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (lhs_forward, lhs_backward) = (self.forward, self.backward);
        let rhs_forward = other.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(StackF::new(lhs_forward, rhs_forward.clone(), axis)),
            Rc::new(StackBackwardLeft::new(lhs_backward, rhs_forward, axis)),
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
            parameters: self.parameters,
        }
    }
}

impl<F1, F2, B2> Stack<VarDiff<F2, B2>> for Var<F1>
where
    F1: Data + Forward + 'static,
    F2: Data<Dim = F1::Dim> + Forward + 'static,
    B2: Gradient<Dim = F1::Dim> + Backward + 'static,
    B2::Dim: RemoveAxis,
    F1::Dim: RemoveAxis,
{
    type Output = VarDiff<StackF<F1, F2>, StackBackwardRight<B2>>;
    fn stack(mut self, mut other: VarDiff<F2, B2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let (rhs_forward, rhs_backward) = (other.forward, other.backward);
        let lhs_forward = self.forward;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(StackF::new(lhs_forward.clone(), rhs_forward, axis)),
            Rc::new(StackBackwardRight::new(lhs_forward, rhs_backward, axis)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        VarDiff {
            id,
            forward,
            backward,
            forward_path: self.forward_path,
            backward_path: other.backward_path,
            parameters: other.parameters,
        }
    }
}

impl<F1, F2> Stack<Var<F2>> for Var<F1>
where
    F1: Data + Forward + 'static,
    F2: Data<Dim = F1::Dim> + Forward + 'static,
    F1::Dim: RemoveAxis,
{
    type Output = Var<StackF<F1, F2>>;
    fn stack(mut self, mut other: Var<F2>, axis: usize) -> Self::Output {
        self.forward_path.append(&mut other.forward_path);

        let lhs_forward = self.forward;
        let rhs_forward = other.forward;

        let (id, forward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(StackF::new(lhs_forward, rhs_forward, axis)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);

        Var {
            id,
            forward,
            forward_path: self.forward_path,
            forward_buffer: Vec::new(),
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
