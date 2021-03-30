pub(crate) mod node;

use itertools::Itertools;
use ndarray::{Array, DimMax, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn, RemoveAxis};
use node::{
    Addition, Concatenate, Division, Dot, Exp, LeakyRelu, Logn, Multiplication, Negation, Node,
    Parameter, Power, Relu, ScalarProduct, Sigmoid, Softmax, Softplus, Stack, Subtraction, Sum,
    Tanh, Transpose, Unsqueeze, VectorDot,
};
use std::{
    cell::{Ref, RefCell},
    fmt::Debug,
    ops::{Add, Deref, Div, Mul, Neg, Sub},
    rc::Rc,
};

pub(crate) type Tensor<D> = Array<f32, D>;
pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;

#[derive(Debug, Clone)]
/// Containse the learnable ancestors of the node.
pub struct Parameters {
    // Contains the one dimensional learnable ancestors
    oned_params: Vec<GraphBuilder<Parameter<Ix1>, Ix1>>,
    // Contains the two dimensional learnable ancestors
    twod_params: Vec<GraphBuilder<Parameter<Ix2>, Ix2>>,
    // Contains the three dimensional learnable ancestors
    threed_params: Vec<GraphBuilder<Parameter<Ix3>, Ix3>>,
    // Contains the four dimensional learnable ancestors
    fourd_params: Vec<GraphBuilder<Parameter<Ix4>, Ix4>>,
    // Contains the five dimensional learnable ancestors
    fived_params: Vec<GraphBuilder<Parameter<Ix5>, Ix5>>,
    // Contains the six dimensional learnable ancestors
    sixd_params: Vec<GraphBuilder<Parameter<Ix6>, Ix6>>,
    // Contains the dynamic dimensional learnable ancestors
    dynd_params: Vec<GraphBuilder<Parameter<IxDyn>, IxDyn>>,
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
pub struct GraphBuilder<T, D>
where
    T: Node,
    D: Dimension,
{
    repr: Rc<T>,
    grad: Option<RefCell<Tensor<D>>>,
    upstream: Parameters,
}

impl<T, D> Clone for GraphBuilder<T, D>
where
    T: Node,
    D: Dimension,
{
    fn clone(&self) -> Self {
        GraphBuilder {
            repr: Rc::clone(&self.repr),
            grad: None,
            upstream: self.upstream.clone(),
        }
    }
}

impl<D> GraphBuilder<Parameter<D>, D>
where
    D: Dimension + 'static,
{
    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.repr.deref().grad.borrow()
    }

    fn as_ptr(&self) -> *const Parameter<D> {
        self.repr.deref() as *const Parameter<D>
    }
    fn zero_grad(&mut self) {
        self.repr.zero_grad()
    }
}

impl<T> GraphBuilder<T, Ix1>
where
    T: Node<Data = Tensor<Ix1>, Gradient = Tensor<Ix1>>,
{
    pub fn dot<U>(&self, other: &GraphBuilder<U, Ix1>) -> GraphBuilder<ScalarProduct<T, U>, Ix1>
    where
        U: Node<Data = Tensor<Ix1>, Gradient = Tensor<Ix1>>,
    {
        GraphBuilder::new(
            Rc::new(ScalarProduct::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

impl<T> GraphBuilder<T, Ix2>
where
    T: Node<Data = Tensor<Ix2>, Gradient = Tensor<Ix2>>,
{
    pub fn mm<U>(&self, other: &GraphBuilder<U, Ix2>) -> GraphBuilder<Dot<T, U>, Ix2>
    where
        U: Node<Data = Tensor<Ix2>, Gradient = Tensor<Ix2>>,
    {
        GraphBuilder::new(
            Rc::new(Dot::new(Rc::clone(&self.repr), Rc::clone(&other.repr))),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn mv_mul<U>(&self, other: &GraphBuilder<U, Ix1>) -> GraphBuilder<VectorDot<T, U>, Ix1>
    where
        U: Node<Data = Tensor<Ix1>, Gradient = Tensor<Ix1>>,
    {
        GraphBuilder::new(
            Rc::new(VectorDot::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

impl<T, D, E> GraphBuilder<T, E>
where
    T: Node<Data = Tensor<D>, Gradient = Tensor<E>>,
    D: Dimension + 'static,
    E: Dimension + 'static,
{
    pub(super) fn new(repr: Rc<T>, upstream: Parameters) -> Self {
        GraphBuilder {
            repr,
            grad: None,
            upstream,
        }
    }
}

impl<T, D> GraphBuilder<T, D>
where
    T: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    pub fn data(&self) -> Ref<T::Data> {
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

    pub fn backward(&mut self, seed: f32) {
        let data_ref: &Tensor<D> = &self.repr.data();
        self.grad
            .get_or_insert_with(|| RefCell::new(data_ref.map(|_| seed)))
            .borrow_mut()
            .map_inplace(|el| *el = seed);

        if let Some(ref grad) = self.grad {
            self.repr.backward(&grad.borrow());
        }
    }

    pub fn sum(&self) -> GraphBuilder<Sum<T, D>, D> {
        GraphBuilder::new(
            Rc::new(Sum::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn pow(&self, exp: i32) -> GraphBuilder<Power<T, D>, D> {
        GraphBuilder {
            repr: Rc::new(Power::new(Rc::clone(&self.repr), exp)),
            grad: None,
            upstream: self.upstream.clone(),
        }
    }

    pub fn relu(&self) -> GraphBuilder<Relu<T, D>, D> {
        GraphBuilder::new(
            Rc::new(Relu::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn leaky_relu(&self) -> GraphBuilder<LeakyRelu<T, D>, D> {
        GraphBuilder::new(
            Rc::new(LeakyRelu::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn softplus(&self) -> GraphBuilder<Softplus<T, D>, D> {
        GraphBuilder::new(
            Rc::new(Softplus::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn sigmoid(&self) -> GraphBuilder<Sigmoid<T, D>, D> {
        GraphBuilder::new(
            Rc::new(Sigmoid::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn tanh(&self) -> GraphBuilder<Tanh<T, D>, D> {
        GraphBuilder::new(
            Rc::new(Tanh::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn ln(&self) -> GraphBuilder<Logn<T, D>, D> {
        GraphBuilder::new(
            Rc::new(Logn::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn exp(&self) -> GraphBuilder<Exp<T, D>, D> {
        GraphBuilder::new(
            Rc::new(Exp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn softmax(&self, axis: usize) -> GraphBuilder<Softmax<T, D>, D> {
        GraphBuilder::new(
            Rc::new(Softmax::new(Rc::clone(&self.repr), axis)),
            self.upstream.clone(),
        )
    }

    pub fn t(&self) -> GraphBuilder<Transpose<T, D>, D> {
        GraphBuilder::new(
            Rc::new(Transpose::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }
}

impl<D, T> GraphBuilder<T, D>
where
    D: RemoveAxis + 'static,
    T: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
{
    pub fn unsqueeze(&self, axis: usize) -> GraphBuilder<Unsqueeze<T, D>, D::Larger> {
        GraphBuilder::new(
            Rc::new(Unsqueeze::new(Rc::clone(&self.repr), axis)),
            self.upstream.clone(),
        )
    }

    pub fn cat<U>(
        self,
        other: GraphBuilder<U, D>,
        axis: usize,
    ) -> GraphBuilder<Concatenate<T, U, D>, D>
    where
        U: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    {
        GraphBuilder::new(
            Rc::new(Concatenate::new(self.repr, other.repr, axis)),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn stack<U>(
        self,
        other: GraphBuilder<U, D>,
        axis: usize,
    ) -> GraphBuilder<Stack<T, U, D>, D::Larger>
    where
        U: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    {
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

fn track_ancestors<D: Dimension + 'static>(
    lhs_up: &[GraphBuilder<Parameter<D>, D>],
    rhs_up: &[GraphBuilder<Parameter<D>, D>],
) -> Vec<GraphBuilder<Parameter<D>, D>> {
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
        impl<LHS, RHS, D, E> $trait<GraphBuilder<RHS, E>> for GraphBuilder<LHS, D>
        where
            LHS: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
            RHS: Node<Data = Tensor<E>, Gradient = Tensor<E>>,
            D: Dimension + DimMax<E> + 'static,
            E: Dimension + 'static,
        {
            type Output = GraphBuilder<$repr<LHS, RHS, D, E>, Broadcasted<D, E>>;

            fn $fun(self, other: GraphBuilder<RHS, E>) -> Self::Output {
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

impl<T, D> Neg for GraphBuilder<T, D>
where
    T: Node<Data = Tensor<D>, Gradient = Tensor<D>>,
    D: Dimension + 'static,
{
    type Output = GraphBuilder<Negation<T, D>, D>;
    fn neg(self) -> Self::Output {
        GraphBuilder::new(Rc::new(Negation::new(self.repr)), self.upstream.clone())
    }
}
