pub mod node;

use itertools::Itertools;
use ndarray::{Array, DimMax, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, RemoveAxis};
// use node::{
//     Addition, Concatenate, Division, Exp, LeakyRelu, Logn, MatrixMatrixMul, MatrixVectorMul,
//     Multiplication, Negation, Node, Parameter, Power, Relu, Sigmoid, Softmax, Softplus, Stack,
//     Subtraction, Sum, Tanh, Transpose, Unsqueeze, VectorMatrixMul, VectorVectorMul,
// };
use node::{
    backward::{Backward, Differentiable, Gradient},
    broadcasted_zeros,
    forward::{Data, Forward},
    Addition, AdditionBackward, AdditionBackwardUnary, Input, InputBackward,
};
use std::{
    cell::{Ref, RefCell, RefMut},
    collections::BTreeMap,
    fmt::Debug,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
    rc::Rc,
};

pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;
pub(crate) type BroadTensor<Lhs, Rhs> = Tensor<Broadcasted<Lhs, Rhs>>;
pub(crate) type Tensor<D> = Array<f32, D>;

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

pub trait ParamDim: Dimension + 'static {
    fn insert(item: Param<Self>, dest: &mut Parameters);
}

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

pub struct Var<T>
where
    T: Data + Forward + 'static,
{
    last: (usize, Rc<T>),
    forward_path: BTreeMap<usize, Rc<dyn Forward>>,
}

impl<T> Var<T>
where
    T: Data + Forward + 'static,
{
    pub fn new(node: T) -> Self {
        let (id, forward) = (unsafe { OPERATIONS_COUNTER.next() }, Rc::new(node));
        let mut forward_path = BTreeMap::new();
        forward_path.insert(id, forward.clone() as Rc<dyn Forward>);

        Self {
            last: (id, forward),
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
            last: (self.last.0, self.last.1.clone()),
            forward_path: self.forward_path.clone(),
        }
    }
}

impl<D> Var<Input<D>>
where
    D: ParamDim,
{
    pub fn requires_grad(self) -> VarDiff<Input<D>, InputBackward<D>> {
        let (id, forward) = self.last;
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
            last: (id, forward, backward),
            forward_path,
            backward_path,
            parameters,
        }
    }
}

pub struct VarDiff<T, U>
where
    T: Data + Forward + 'static,
    U: Gradient + Backward + 'static,
{
    last: (usize, Rc<T>, Rc<U>),
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
            last: (self.last.0, self.last.1.clone(), self.last.2.clone()),
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
        self.last.2.gradient()
    }

    pub(crate) fn data_mut(&mut self) -> RefMut<Tensor<D>> {
        self.last.1.data_mut()
    }
    fn zero_grad(&mut self) {
        self.last.2.zero_grad()
    }
}

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

// impl<T: Node> GraphBuilder<T> {
//     pub(super) fn new(repr: Rc<T>, upstream: Parameters) -> Self {
//         GraphBuilder {
//             repr,
//             grad: None,
//             upstream,
//         }
//     }
// }

//     pub fn backward(&mut self, seed: f32) {
//         let data_ref: &Tensor<T::Dim> = &self.repr.data();
//         self.grad
//             .get_or_insert_with(|| RefCell::new(data_ref.map(|_| seed)))
//             .borrow_mut()
//             .map_inplace(|el| *el = seed);

//         if let Some(ref grad) = self.grad {
//             self.repr.backward(&grad.borrow());
//         }
//     }

//     pub fn data(&self) -> Ref<Tensor<T::Dim>> {
//         self.repr.data()
//     }
//     pub fn forward(&self) {
//         self.repr.forward()
//     }

//     pub fn clear(&self) {
//         self.repr.clear();
//     }

//     pub fn requires_grad(&self) -> bool {
//         self.repr.requires_grad()
//     }

//     pub fn zero_gradient(&mut self) {
//         for param in &mut self.upstream.oned_params[..] {
//             param.zero_grad();
//         }
//         for param in &mut self.upstream.twod_params[..] {
//             param.zero_grad();
//         }
//         for param in &mut self.upstream.threed_params[..] {
//             param.zero_grad();
//         }
//         for param in &mut self.upstream.fourd_params[..] {
//             param.zero_grad();
//         }
//     }

//     pub fn upstream(&self) -> &Parameters {
//         &self.upstream
//     }

//     pub fn sum(&self) -> GraphBuilder<impl Node<Dim = Ix1>> {
//         GraphBuilder::new(
//             Rc::new(Sum::new(Rc::clone(&self.repr))),
//             self.upstream.clone(),
//         )
//     }

//     pub fn pow(&self, exp: i32) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder {
//             repr: Rc::new(Power::new(Rc::clone(&self.repr), exp)),
//             grad: None,
//             upstream: self.upstream.clone(),
//         }
//     }

//     pub fn relu(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder::new(
//             Rc::new(Relu::new(Rc::clone(&self.repr))),
//             self.upstream.clone(),
//         )
//     }

//     pub fn leaky_relu(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder::new(
//             Rc::new(LeakyRelu::new(Rc::clone(&self.repr))),
//             self.upstream.clone(),
//         )
//     }

//     pub fn softplus(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder::new(
//             Rc::new(Softplus::new(Rc::clone(&self.repr))),
//             self.upstream.clone(),
//         )
//     }

//     pub fn sigmoid(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder::new(
//             Rc::new(Sigmoid::new(Rc::clone(&self.repr))),
//             self.upstream.clone(),
//         )
//     }

//     pub fn tanh(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder::new(
//             Rc::new(Tanh::new(Rc::clone(&self.repr))),
//             self.upstream.clone(),
//         )
//     }

//     pub fn ln(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder::new(
//             Rc::new(Logn::new(Rc::clone(&self.repr))),
//             self.upstream.clone(),
//         )
//     }

//     pub fn exp(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder::new(
//             Rc::new(Exp::new(Rc::clone(&self.repr))),
//             self.upstream.clone(),
//         )
//     }

//     pub fn softmax(&self, axis: usize) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder::new(
//             Rc::new(Softmax::new(Rc::clone(&self.repr), axis)),
//             self.upstream.clone(),
//         )
//     }

//     pub fn t(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
//         GraphBuilder::new(
//             Rc::new(Transpose::new(Rc::clone(&self.repr))),
//             self.upstream.clone(),
//         )
//     }
// }

// impl<T> GraphBuilder<T>
// where
//     T: Node,
//     T::Dim: RemoveAxis + 'static,
// {
//     pub fn unsqueeze(
//         &self,
//         axis: usize,
//     ) -> GraphBuilder<impl Node<Dim = <T::Dim as Dimension>::Larger>> {
//         GraphBuilder::new(
//             Rc::new(Unsqueeze::new(Rc::clone(&self.repr), axis)),
//             self.upstream.clone(),
//         )
//     }

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
// }

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

// macro_rules! impl_node_arithmetic_ops {
//     ($trait:ident, $fun:ident, $repr:ident) => {
//         impl<Lhs, Rhs> $trait<GraphBuilder<Rhs>> for GraphBuilder<Lhs>
//         where
//             Lhs: Node,
//             Rhs: Node,
//             Lhs::Dim: DimMax<Rhs::Dim> + 'static,
//         {
//             type Output = GraphBuilder<$repr<Lhs, Rhs>>;

//             fn $fun(self, other: GraphBuilder<Rhs>) -> Self::Output {
//                 GraphBuilder::new(
//                     Rc::new($repr::new(self.repr, other.repr)),
//                     track_upstream(&self.upstream, &other.upstream),
//                 )
//             }
//         }
//     };
// }

// impl_node_arithmetic_ops!(Add, add, Addition);
// impl_node_arithmetic_ops!(Sub, sub, Subtraction);
// impl_node_arithmetic_ops!(Mul, mul, Multiplication);
// impl_node_arithmetic_ops!(Div, div, Division);

// impl<T: Node> Neg for GraphBuilder<T> {
//     type Output = GraphBuilder<Negation<T>>;

//     fn neg(self) -> Self::Output {
//         GraphBuilder::new(Rc::new(Negation::new(self.repr)), self.upstream)
//     }
// }

impl<Lhs, Rhs> Add<Var<Rhs>> for Var<Lhs>
where
    Lhs: Data + Forward + 'static,
    Rhs: Data + Forward + 'static,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Output = Var<Addition<Lhs, Rhs>>;

    fn add(mut self, mut rhs: Var<Rhs>) -> Self::Output {
        self.forward_path.append(&mut rhs.forward_path);

        let (_, lhs_node) = self.last;
        let (_, rhs_node) = rhs.last;

        let (id, node) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Addition::new(lhs_node, rhs_node)),
        );
        self.forward_path
            .insert(id, node.clone() as Rc<dyn Forward>);

        Self::Output {
            last: (id, node),
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

        let (_, lhs_node) = self.last;
        let (_, rhs_forward, rhs_backward) = rhs.last;
        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Addition::new(lhs_node.clone(), rhs_forward)),
            Rc::new(AdditionBackwardUnary::new(rhs_backward, lhs_node)),
        );
        rhs.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        rhs.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            last: (id, forward, backward),
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

        let (_, lhs_forward, lhs_backward) = self.last;
        let (_, rhs_node) = rhs.last;

        let (id, forward, backward) = (
            unsafe { OPERATIONS_COUNTER.next() },
            Rc::new(Addition::new(lhs_forward, rhs_node.clone())),
            Rc::new(AdditionBackwardUnary::new(lhs_backward, rhs_node)),
        );
        self.forward_path
            .insert(id, forward.clone() as Rc<dyn Forward>);
        self.backward_path
            .insert(id, backward.clone() as Rc<dyn Backward>);

        Self::Output {
            last: (id, forward, backward),
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

        let (_, lhs_forward, lhs_backward) = self.last;
        let (_, rhs_forward, rhs_backward) = rhs.last;

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
            last: (id, forward, backward),
            forward_path: self.forward_path,
            backward_path: self.backward_path,
            parameters: merge_parameters(self.parameters, rhs.parameters),
        }
    }
}
