pub mod node;

use itertools::Itertools;
use ndarray::{Array, DimMax, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, RemoveAxis};
use node::{
    Addition, Concatenate, Division, Exp, LeakyRelu, Logn, MatrixMatrixMul, MatrixVectorMul,
    Multiplication, Negation, Node, Parameter, Power, Relu, Sigmoid, Softmax, Softplus, Stack,
    Subtraction, Sum, Tanh, Transpose, Unsqueeze, VectorMatrixMul, VectorVectorMul,
};
use std::{
    cell::{Ref, RefCell, RefMut},
    fmt::Debug,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
    rc::Rc,
};

pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;
pub(crate) type BroadTensor<Lhs, Rhs> = Tensor<Broadcasted<Lhs, Rhs>>;
pub(crate) type Tensor<D> = Array<f32, D>;

pub trait ParamDim: Dimension + 'static {
    fn insert(item: GraphBuilder<Parameter<Self>>, dest: &mut Parameters);
}

impl ParamDim for Ix1 {
    fn insert(item: GraphBuilder<Parameter<Self>>, dest: &mut Parameters) {
        dest.oned_params.push(item);
    }
}

impl ParamDim for Ix2 {
    fn insert(item: GraphBuilder<Parameter<Self>>, dest: &mut Parameters) {
        dest.twod_params.push(item);
    }
}

impl ParamDim for Ix3 {
    fn insert(item: GraphBuilder<Parameter<Self>>, dest: &mut Parameters) {
        dest.threed_params.push(item);
    }
}

impl ParamDim for Ix4 {
    fn insert(item: GraphBuilder<Parameter<Self>>, dest: &mut Parameters) {
        dest.fourd_params.push(item);
    }
}

impl ParamDim for Ix5 {
    fn insert(item: GraphBuilder<Parameter<Self>>, dest: &mut Parameters) {
        dest.fived_params.push(item);
    }
}

impl ParamDim for Ix6 {
    fn insert(item: GraphBuilder<Parameter<Self>>, dest: &mut Parameters) {
        dest.sixd_params.push(item);
    }
}

impl ParamDim for IxDyn {
    fn insert(item: GraphBuilder<Parameter<Self>>, dest: &mut Parameters) {
        dest.dynd_params.push(item);
    }
}

#[derive(Debug, Clone)]
/// Contains the learnable ancestors of the node.
pub struct Parameters {
    // Contains the one dimensional learnable ancestors
    oned_params: Vec<GraphBuilder<Parameter<Ix1>>>,
    // Contains the two dimensional learnable ancestors
    twod_params: Vec<GraphBuilder<Parameter<Ix2>>>,
    // Contains the three dimensional learnable ancestors
    threed_params: Vec<GraphBuilder<Parameter<Ix3>>>,
    // Contains the four dimensional learnable ancestors
    fourd_params: Vec<GraphBuilder<Parameter<Ix4>>>,
    // Contains the five dimensional learnable ancestors
    fived_params: Vec<GraphBuilder<Parameter<Ix5>>>,
    // Contains the six dimensional learnable ancestors
    sixd_params: Vec<GraphBuilder<Parameter<Ix6>>>,
    // Contains the dynamic dimensional learnable ancestors
    dynd_params: Vec<GraphBuilder<Parameter<IxDyn>>>,
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

// A pointer to a node in the computational graph.
#[derive(Debug)]
pub struct GraphBuilder<T: Node> {
    repr: Rc<T>,
    grad: Option<RefCell<Tensor<T::Dim>>>,
    upstream: Parameters,
}

impl<T: Node> Clone for GraphBuilder<T> {
    fn clone(&self) -> Self {
        GraphBuilder {
            repr: Rc::clone(&self.repr),
            grad: None,
            upstream: self.upstream.clone(),
        }
    }
}

impl<D> GraphBuilder<Parameter<D>>
where
    D: ParamDim,
{
    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.repr.deref().grad.borrow()
    }

    pub(crate) fn data_mut(&mut self) -> RefMut<Tensor<D>> {
        self.repr.data.borrow_mut()
    }

    fn as_ptr(&self) -> *const Parameter<D> {
        self.repr.deref() as *const Parameter<D>
    }
    fn zero_grad(&mut self) {
        self.repr.zero_grad()
    }
}

impl<T> GraphBuilder<T>
where
    T: Node<Dim = Ix1>,
{
    pub fn vv_mul<U>(&self, other: &GraphBuilder<U>) -> GraphBuilder<impl Node<Dim = Ix1>>
    where
        U: Node<Dim = Ix1>,
    {
        GraphBuilder::new(
            Rc::new(VectorVectorMul::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn vm_mul<U>(&self, other: &GraphBuilder<U>) -> GraphBuilder<impl Node<Dim = Ix1>>
    where
        U: Node<Dim = Ix2>,
    {
        GraphBuilder::new(
            Rc::new(VectorMatrixMul::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

impl<T> GraphBuilder<T>
where
    T: Node<Dim = Ix2>,
{
    pub fn mm_mul<U>(&self, other: &GraphBuilder<U>) -> GraphBuilder<impl Node<Dim = Ix2>>
    where
        U: Node<Dim = Ix2>,
    {
        GraphBuilder::new(
            Rc::new(MatrixMatrixMul::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn mv_mul<U>(&self, other: &GraphBuilder<U>) -> GraphBuilder<impl Node<Dim = Ix1>>
    where
        U: Node<Dim = Ix1>,
    {
        GraphBuilder::new(
            Rc::new(MatrixVectorMul::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

impl<T: Node> GraphBuilder<T> {
    pub(super) fn new(repr: Rc<T>, upstream: Parameters) -> Self {
        GraphBuilder {
            repr,
            grad: None,
            upstream,
        }
    }

    pub fn backward(&mut self, seed: f32) {
        let data_ref: &Tensor<T::Dim> = &self.repr.data();
        self.grad
            .get_or_insert_with(|| RefCell::new(data_ref.map(|_| seed)))
            .borrow_mut()
            .map_inplace(|el| *el = seed);

        if let Some(ref grad) = self.grad {
            self.repr.backward(&grad.borrow());
        }
    }

    pub fn data(&self) -> Ref<Tensor<T::Dim>> {
        self.repr.data()
    }
    pub fn forward(&self) {
        self.repr.forward()
    }

    pub fn clear(&self) {
        self.repr.clear();
    }

    pub fn requires_grad(&self) -> bool {
        self.repr.requires_grad()
    }

    pub fn zero_gradient(&mut self) {
        for param in &mut self.upstream.oned_params[..] {
            param.zero_grad();
        }
        for param in &mut self.upstream.twod_params[..] {
            param.zero_grad();
        }
        for param in &mut self.upstream.threed_params[..] {
            param.zero_grad();
        }
        for param in &mut self.upstream.fourd_params[..] {
            param.zero_grad();
        }
    }

    pub fn upstream(&self) -> &Parameters {
        &self.upstream
    }

    pub fn sum(&self) -> GraphBuilder<impl Node<Dim = Ix1>> {
        GraphBuilder::new(
            Rc::new(Sum::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn pow(&self, exp: i32) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder {
            repr: Rc::new(Power::new(Rc::clone(&self.repr), exp)),
            grad: None,
            upstream: self.upstream.clone(),
        }
    }

    pub fn relu(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(Relu::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn leaky_relu(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(LeakyRelu::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn softplus(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(Softplus::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn sigmoid(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(Sigmoid::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn tanh(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(Tanh::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn ln(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(Logn::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn exp(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(Exp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn softmax(&self, axis: usize) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(Softmax::new(Rc::clone(&self.repr), axis)),
            self.upstream.clone(),
        )
    }

    pub fn t(&self) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(Transpose::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }
}

impl<T> GraphBuilder<T>
where
    T: Node,
    T::Dim: RemoveAxis + 'static,
{
    pub fn unsqueeze(
        &self,
        axis: usize,
    ) -> GraphBuilder<impl Node<Dim = <T::Dim as Dimension>::Larger>> {
        GraphBuilder::new(
            Rc::new(Unsqueeze::new(Rc::clone(&self.repr), axis)),
            self.upstream.clone(),
        )
    }

    pub fn cat(
        self,
        other: GraphBuilder<impl Node<Dim = T::Dim>>,
        axis: usize,
    ) -> GraphBuilder<impl Node<Dim = T::Dim>> {
        GraphBuilder::new(
            Rc::new(Concatenate::new(self.repr, other.repr, axis)),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn stack(
        self,
        other: GraphBuilder<impl Node<Dim = T::Dim>>,
        axis: usize,
    ) -> GraphBuilder<impl Node<Dim = <T::Dim as Dimension>::Larger>> {
        GraphBuilder::new(
            Rc::new(Stack::new(self.repr, other.repr, axis)),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

fn track_upstream(lhs_params: &Parameters, rhs_params: &Parameters) -> Parameters {
    Parameters {
        oned_params: track_ancestors(&lhs_params.oned_params[..], &rhs_params.oned_params[..]),
        twod_params: track_ancestors(&lhs_params.twod_params[..], &rhs_params.twod_params[..]),
        threed_params: track_ancestors(
            &lhs_params.threed_params[..],
            &rhs_params.threed_params[..],
        ),
        fourd_params: track_ancestors(&lhs_params.fourd_params[..], &rhs_params.fourd_params[..]),
        fived_params: track_ancestors(&lhs_params.fived_params[..], &rhs_params.fived_params[..]),
        sixd_params: track_ancestors(&lhs_params.sixd_params[..], &rhs_params.sixd_params[..]),
        dynd_params: track_ancestors(&lhs_params.dynd_params[..], &rhs_params.dynd_params[..]),
    }
}

fn track_ancestors<D: ParamDim>(
    lhs_up: &[GraphBuilder<Parameter<D>>],
    rhs_up: &[GraphBuilder<Parameter<D>>],
) -> Vec<GraphBuilder<Parameter<D>>> {
    lhs_up
        .iter()
        .merge_join_by(rhs_up.iter(), |lhs_par, rhs_par| {
            lhs_par.as_ptr().cmp(&rhs_par.as_ptr())
        })
        .map(|choice| match choice {
            itertools::EitherOrBoth::Left(lhs_par) => lhs_par,
            itertools::EitherOrBoth::Right(rhs_par) => rhs_par,
            itertools::EitherOrBoth::Both(lhs_par, _) => lhs_par,
        })
        .cloned()
        .collect()
}

macro_rules! impl_node_arithmetic_ops {
    ($trait:ident, $fun:ident, $repr:ident) => {
        impl<Lhs, Rhs> $trait<GraphBuilder<Rhs>> for GraphBuilder<Lhs>
        where
            Lhs: Node,
            Rhs: Node,
            Lhs::Dim: DimMax<Rhs::Dim> + 'static,
        {
            type Output = GraphBuilder<$repr<Lhs, Rhs>>;

            fn $fun(self, other: GraphBuilder<Rhs>) -> Self::Output {
                GraphBuilder::new(
                    Rc::new($repr::new(self.repr, other.repr)),
                    track_upstream(&self.upstream, &other.upstream),
                )
            }
        }
    };
}

impl_node_arithmetic_ops!(Add, add, Addition);
impl_node_arithmetic_ops!(Sub, sub, Subtraction);
impl_node_arithmetic_ops!(Mul, mul, Multiplication);
impl_node_arithmetic_ops!(Div, div, Division);

impl<T: Node> Neg for GraphBuilder<T> {
    type Output = GraphBuilder<Negation<T>>;

    fn neg(self) -> Self::Output {
        GraphBuilder::new(Rc::new(Negation::new(self.repr)), self.upstream)
    }
}
