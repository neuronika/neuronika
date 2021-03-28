pub(crate) mod node;

use itertools::Itertools;
use ndarray::{Array, DimMax, Dimension, Ix1, Ix2, RemoveAxis};
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

pub trait TrackableClone {
    fn clone_box(&self) -> Box<dyn Trackable>;
}

pub trait Trackable: Debug + TrackableClone {
    fn zero_grad(&mut self);
    fn into_trackable(self) -> Box<dyn Trackable>;
    fn get_id(&self) -> usize;
}

impl<D> TrackableClone for GraphBuilder<Parameter<D>, D>
where
    D: Dimension + 'static,
{
    fn clone_box(&self) -> Box<dyn Trackable> {
        Box::new(self.clone())
    }
}

impl<D> Trackable for GraphBuilder<Parameter<D>, D>
where
    D: Dimension + 'static,
{
    fn zero_grad(&mut self) {
        self.repr.zero_grad()
    }

    fn into_trackable(self) -> Box<dyn Trackable> {
        Box::new(self)
    }

    fn get_id(&self) -> usize {
        self.repr.id
    }
}

impl Clone for Box<dyn Trackable> {
    fn clone(&self) -> Box<dyn Trackable> {
        self.clone_box()
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
    upstream: Vec<Box<dyn Trackable>>,
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
    pub(super) fn new(repr: Rc<T>, upstream: Vec<Box<dyn Trackable>>) -> Self {
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

    pub fn upstream(&self) -> &[Box<dyn Trackable>] {
        &self.upstream[..]
    }

    pub fn zero_gradient(&mut self) {
        for param in self.upstream_mut() {
            param.zero_grad();
        }
    }

    pub fn upstream_mut(&mut self) -> &mut [Box<dyn Trackable>] {
        &mut self.upstream[..]
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

pub fn track_upstream(
    lhs_up: &[Box<dyn Trackable>],
    rhs_up: &[Box<dyn Trackable>],
) -> Vec<Box<dyn Trackable>> {
    lhs_up
        .iter()
        .merge_join_by(rhs_up.iter(), |lhs_par, rhs_par| {
            (lhs_par.get_id()).cmp(&rhs_par.get_id())
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
