pub mod node;
use itertools::Itertools;
use ndarray::{Array, DimMax, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, RemoveAxis};
use node::{
    backward::{Backward, Differentiable, Gradient},
    forward::{Data, Forward},
    Addition, AdditionBackward, AdditionBackwardUnary, Division, DivisionBackward,
    DivisionBackwardLeft, DivisionBackwardRight, Exp, ExpBackward, Input, InputBackward, LeakyReLU,
    LeakyReLUBackward, Logn, LognBackward, Multiplication, MultiplicationBackward,
    MultiplicationBackwardUnary, Negation, NegationBackward, Power, PowerBackward, ReLU,
    ReLUBackward, Sigmoid, SigmoidBackward, SoftPlus, SoftPlusBackward, Softmax, SoftmaxBackward,
    Subtraction, SubtractionBackward, SubtractionBackwardLeft, SubtractionBackwardRight, Sum,
    SumBackward, TanH, TanHBackward, Transpose, TransposeBackward, Unsqueeze, UnsqueezeBackward,
};
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Global Var Identifier ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
struct OperationsCounter {
    count: usize,
}

impl OperationsCounter {
    pub fn next(&mut self) -> usize {
        self.count += 1;
        self.count
    }
}

static mut OPERATIONS_COUNTER: OperationsCounter = OperationsCounter { count: 0 };

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ParamDim Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait ParamDim: Dimension + 'static {
    fn insert(item: Param<Self>, dest: &mut Parameters);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ParamDim Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl ParamDim for Ix1 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.oned_params.push(item);
    }
}

impl ParamDim for Ix2 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.twod_params.push(item);
    }
}

impl ParamDim for Ix3 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.threed_params.push(item);
    }
}

impl ParamDim for Ix4 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.fourd_params.push(item);
    }
}

impl ParamDim for Ix5 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.fived_params.push(item);
    }
}

impl ParamDim for Ix6 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.sixd_params.push(item);
    }
}

impl ParamDim for IxDyn {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.dynd_params.push(item);
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
    oned_params: Vec<Param<Ix1>>,
    // Contains the two dimensional learnable ancestors
    twod_params: Vec<Param<Ix2>>,
    // Contains the three dimensional learnable ancestors
    threed_params: Vec<Param<Ix3>>,
    // Contains the four dimensional learnable ancestors
    fourd_params: Vec<Param<Ix4>>,
    // Contains the five dimensional learnable ancestors
    fived_params: Vec<Param<Ix5>>,
    // Contains the six dimensional learnable ancestors
    sixd_params: Vec<Param<Ix6>>,
    // Contains the dynamic dimensional learnable ancestors
    dynd_params: Vec<Param<IxDyn>>,
}

impl Parameters {
    fn new() -> Parameters {
        Parameters {
            oned_params: Vec::new(),
            twod_params: Vec::new(),
            threed_params: Vec::new(),
            fourd_params: Vec::new(),
            fived_params: Vec::new(),
            sixd_params: Vec::new(),
            dynd_params: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.oned_params.len()
            + self.twod_params.len()
            + self.threed_params.len()
            + self.fourd_params.len()
            + self.fived_params.len()
            + self.sixd_params.len()
            + self.dynd_params.len()
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~ Functions to keep track of differentiable history ~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn merge_parameters(lhs_params: Parameters, rhs_params: Parameters) -> Parameters {
    Parameters {
        oned_params: merge(&lhs_params.oned_params[..], &rhs_params.oned_params[..]),
        twod_params: merge(&lhs_params.twod_params[..], &rhs_params.twod_params[..]),
        threed_params: merge(&lhs_params.threed_params[..], &rhs_params.threed_params[..]),
        fourd_params: merge(&lhs_params.fourd_params[..], &rhs_params.fourd_params[..]),
        fived_params: merge(&lhs_params.fived_params[..], &rhs_params.fived_params[..]),
        sixd_params: merge(&lhs_params.sixd_params[..], &rhs_params.sixd_params[..]),
        dynd_params: merge(&lhs_params.dynd_params[..], &rhs_params.dynd_params[..]),
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
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Non differentiable Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Var<T>
where
    T: Data + Forward + 'static,
{
    id: usize,
    forward: Rc<T>,
    forward_path: BTreeMap<usize, Rc<dyn Forward>>,
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
        }
    }
}

impl<D> Var<Input<D>>
where
    D: ParamDim,
{
    pub fn requires_grad(self) -> VarDiff<Input<D>, InputBackward<D>> {
        let (id, forward) = (self.id, self.forward);
        if Rc::strong_count(&forward) > 1 {
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
    pub fn forward(&self) {
        for f in &self.forward_path {
            f.1.forward();
        }
    }
    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.forward.data()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Differentiable Variable ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient + Backward + 'static,
{
    id: usize,
    forward: Rc<T>,
    backward: Rc<U>,
    forward_path: BTreeMap<usize, Rc<dyn Forward>>,
    backward_path: BTreeMap<usize, Rc<dyn Backward>>,
    parameters: Parameters,
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

// !!!TODO!!!
// impl<T> GraphBuilder<T>
// where
//     T: Node<Dim = Ix1>,
// {
//     pub fn vv_mul<U>(&self, other: &GraphBuilder<U>) -> GraphBuilder<impl Node<Dim = Ix1>>
//     where
//         U: Node<Dim = Ix1>,
//     {
//         GraphBuilder::new(
//             Rc::new(VectorVectorMul::new(
//                 Rc::clone(&self.repr),
//                 Rc::clone(&other.repr),
//             )),
//             track_upstream(&self.upstream, &other.upstream),
//         )
//     }

//     pub fn vm_mul<U>(&self, other: &GraphBuilder<U>) -> GraphBuilder<impl Node<Dim = Ix1>>
//     where
//         U: Node<Dim = Ix2>,
//     {
//         GraphBuilder::new(
//             Rc::new(VectorMatrixMul::new(
//                 Rc::clone(&self.repr),
//                 Rc::clone(&other.repr),
//             )),
//             track_upstream(&self.upstream, &other.upstream),
//         )
//     }
// }

// impl<T> GraphBuilder<T>
// where
//     T: Node<Dim = Ix2>,
// {
//     pub fn mm_mul<U>(&self, other: &GraphBuilder<U>) -> GraphBuilder<impl Node<Dim = Ix2>>
//     where
//         U: Node<Dim = Ix2>,
//     {
//         GraphBuilder::new(
//             Rc::new(MatrixMatrixMul::new(
//                 Rc::clone(&self.repr),
//                 Rc::clone(&other.repr),
//             )),
//             track_upstream(&self.upstream, &other.upstream),
//         )
//     }

//     pub fn mv_mul<U>(&self, other: &GraphBuilder<U>) -> GraphBuilder<impl Node<Dim = Ix1>>
//     where
//         U: Node<Dim = Ix1>,
//     {
//         GraphBuilder::new(
//             Rc::new(MatrixVectorMul::new(
//                 Rc::clone(&self.repr),
//                 Rc::clone(&other.repr),
//             )),
//             track_upstream(&self.upstream, &other.upstream),
//         )
//     }
// }

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
        for param in &mut self.parameters.oned_params[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.twod_params[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.threed_params[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.fourd_params[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.fived_params[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.sixd_params[..] {
            param.zero_grad();
        }
        for param in &mut self.parameters.dynd_params[..] {
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
            Rc::new(Sum::new(forward.clone())),
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

    pub fn pow(mut self, exp: i32) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
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

    pub fn relu(mut self) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
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

    pub fn leaky_relu(mut self) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
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

    pub fn softplus(mut self) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
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

    pub fn sigmoid(mut self) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
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

    pub fn tanh(mut self) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
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

    pub fn ln(mut self) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
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
    pub fn exp(mut self) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
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

    pub fn softmax(
        mut self,
        axis: usize,
    ) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
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

    pub fn t(mut self) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Transpose::new(forward.clone())),
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
}

impl<T, U, D> VarDiff<T, U>
where
    T: Data<Dim = D> + Forward + 'static,
    U: Gradient<Dim = D> + Backward + 'static,
    D: Dimension + RemoveAxis,
{
    pub fn unsqueeze(
        mut self,
        axis: usize,
    ) -> VarDiff<impl Forward + Data, impl Backward + Gradient> {
        let (forward, backward) = (self.forward, self.backward);
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Unsqueeze::new(forward.clone(), axis)),
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

    // !!!TODO!!!
    //     pub fn cat(
    //         self,
    //         other: GraphBuilder<impl Node<Dim = T::Dim>>,
    //         axis: usize,
    //     ) -> GraphBuilder<impl Node<Dim = T::Dim>> {
    //         GraphBuilder::new(
    //             Rc::new(Concatenate::new(self.repr, other.repr, axis)),
    //             track_upstream(&self.upstream, &other.upstream),
    //         )
    //     }

    //     pub fn stack(
    //         self,
    //         other: GraphBuilder<impl Node<Dim = T::Dim>>,
    //         axis: usize,
    //     ) -> GraphBuilder<impl Node<Dim = <T::Dim as Dimension>::Larger>> {
    //         GraphBuilder::new(
    //             Rc::new(Stack::new(self.repr, other.repr, axis)),
    //             track_upstream(&self.upstream, &other.upstream),
    //         )
    //     }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Arithmetic Operations Implementation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Negation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
            Rc::new(Negation::new(forward.clone())),
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
