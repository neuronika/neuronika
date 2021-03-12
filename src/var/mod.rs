pub(super) mod numeric;
pub(crate) mod ops;

use itertools::Itertools;
use ndarray::{Dimension, Ix1, Ix2, RemoveAxis};
use numeric::{DStack, DStacked, HStack, HStacked, Max, Maximum, Tensor, VStack, VStacked};
use ops::{
    AddOp, Borrow, DStackOp, DivOp, DotOp, DotVecOp, ExpOp, HStackOp, LeakyReLUOp, LnOp, MulOp,
    NegOp, Op, Param, PowOp, ReLUOp, ScalProdOp, SigmoidOp, SoftmaxOp, SoftplusOp, SubOp, SumOp,
    TOp, TanhOp, VStackOp,
};
use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::rc::Rc;

pub trait TrackableClone {
    fn clone_box(&self) -> Box<dyn Trackable>;
}

pub trait Trackable: Debug + TrackableClone {
    fn zero_grad(&mut self);
    fn as_trackable(self) -> Box<dyn Trackable>;
    fn get_id(&self) -> usize;
}

impl<D> TrackableClone for Var<Param<D>, D>
where
    D: Dimension + RemoveAxis + 'static,
{
    fn clone_box(&self) -> Box<dyn Trackable> {
        Box::new(self.clone())
    }
}

impl<D> Trackable for Var<Param<D>, D>
where
    D: Dimension + RemoveAxis + 'static,
{
    fn zero_grad(&mut self) {
        self.repr.zero_grad()
    }

    fn as_trackable(self) -> Box<dyn Trackable> {
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
pub struct Var<T, D>
where
    T: Op,
    D: Dimension + RemoveAxis,
{
    repr: Rc<T>,
    grad: Option<RefCell<Tensor<D>>>,
    upstream: Vec<Box<dyn Trackable>>,
}

impl<T, D> Clone for Var<T, D>
where
    T: Op,
    D: Dimension + RemoveAxis,
{
    fn clone(&self) -> Self {
        Var {
            repr: Rc::clone(&self.repr),
            grad: None,
            upstream: self.upstream.clone(),
        }
    }
}

impl<D> Var<Param<D>, D>
where
    D: Dimension + RemoveAxis + 'static,
{
    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.repr.deref().grad.borrow()
    }
}

impl<T> Var<T, Ix1>
where
    T: Op<Data = Tensor<Ix1>, Grad = Tensor<Ix1>>,
{
    pub fn dot<U>(&self, other: &Var<U, Ix1>) -> Var<ScalProdOp<T, U>, Ix1>
    where
        U: Op<Data = Tensor<Ix1>, Grad = Tensor<Ix1>>,
    {
        Var::new(
            Rc::new(ScalProdOp::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

impl<T> Var<T, Ix2>
where
    T: Op<Data = Tensor<Ix2>, Grad = Tensor<Ix2>>,
{
    pub fn mm<U>(&self, other: &Var<U, Ix2>) -> Var<DotOp<T, U>, Ix2>
    where
        U: Op<Data = Tensor<Ix2>, Grad = Tensor<Ix2>>,
    {
        Var::new(
            Rc::new(DotOp::new(Rc::clone(&self.repr), Rc::clone(&other.repr))),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn mv_mul<U>(&self, other: &Var<U, Ix1>) -> Var<DotVecOp<T, U>, Ix1>
    where
        U: Op<Data = Tensor<Ix1>, Grad = Tensor<Ix1>>,
    {
        Var::new(
            Rc::new(DotVecOp::new(Rc::clone(&self.repr), Rc::clone(&other.repr))),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

impl<T, D, E> Var<T, E>
where
    T: Op<Data = Tensor<D>, Grad = Tensor<E>>,
    D: Dimension + RemoveAxis + 'static,
    E: Dimension + RemoveAxis + 'static,
{
    pub(super) fn new(repr: Rc<T>, upstream: Vec<Box<dyn Trackable>>) -> Self {
        Var {
            repr: repr,
            grad: None,
            upstream: upstream,
        }
    }
}

impl<T, D> Var<T, D>
where
    T: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    pub fn data(&self) -> Borrow<T::Data> {
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
            .get_or_insert_with(|| {
                RefCell::new(Tensor {
                    data: data_ref.data.map(|_| seed),
                })
            })
            .borrow_mut()
            .data
            .map_inplace(|el| *el = seed);

        if let Some(ref grad) = self.grad {
            self.repr.backward(&grad.borrow());
        }
    }

    pub fn sum(&self) -> Var<SumOp<T, D>, D> {
        Var::new(
            Rc::new(SumOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn pow(&self, exp: i32) -> Var<PowOp<T, D>, D> {
        Var {
            repr: Rc::new(PowOp::new(Rc::clone(&self.repr), exp)),
            grad: None,
            upstream: self.upstream.clone(),
        }
    }

    pub fn relu(&self) -> Var<ReLUOp<T, D>, D> {
        Var::new(
            Rc::new(ReLUOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn leaky_relu(&self) -> Var<LeakyReLUOp<T, D>, D> {
        Var::new(
            Rc::new(LeakyReLUOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn softplus(&self) -> Var<SoftplusOp<T, D>, D> {
        Var::new(
            Rc::new(SoftplusOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn sigmoid(&self) -> Var<SigmoidOp<T, D>, D> {
        Var::new(
            Rc::new(SigmoidOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn tanh(&self) -> Var<TanhOp<T, D>, D> {
        Var::new(
            Rc::new(TanhOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn ln(&self) -> Var<LnOp<T, D>, D>
    where
        D: Max<D>,
    {
        Var::new(
            Rc::new(LnOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn exp(&self) -> Var<ExpOp<T, D>, D> {
        Var::new(
            Rc::new(ExpOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn softmax(&self, axis: usize) -> Var<SoftmaxOp<T, D>, D> {
        Var::new(
            Rc::new(SoftmaxOp::new(Rc::clone(&self.repr), axis)),
            self.upstream.clone(),
        )
    }

    pub fn t(&self) -> Var<TOp<T, D>, D> {
        Var::new(
            Rc::new(TOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn hstack<U, E>(&self, other: &Var<U, E>) -> Var<HStackOp<T, U, D, E>, HStacked<D, E>>
    where
        U: Op<Data = Tensor<E>, Grad = Tensor<E>>,
        D: HStack<E>,
        E: Dimension + RemoveAxis + 'static,
    {
        Var::new(
            Rc::new(HStackOp::new(Rc::clone(&self.repr), Rc::clone(&other.repr))),
            self.upstream.clone(),
        )
    }

    pub fn vstack<U, E>(&self, other: &Var<U, E>) -> Var<VStackOp<T, U, D, E>, VStacked<D, E>>
    where
        U: Op<Data = Tensor<E>, Grad = Tensor<E>>,
        D: VStack<E>,
        E: Dimension + RemoveAxis + 'static,
    {
        Var::new(
            Rc::new(VStackOp::new(Rc::clone(&self.repr), Rc::clone(&other.repr))),
            self.upstream.clone(),
        )
    }

    pub fn dstack<U, E>(&self, other: &Var<U, D>) -> Var<DStackOp<T, U, D>, DStacked<D, D>>
    where
        U: Op<Data = Tensor<D>, Grad = Tensor<D>>,
        D: DStack<D>,
    {
        Var::new(
            Rc::new(DStackOp::new(Rc::clone(&self.repr), Rc::clone(&other.repr))),
            self.upstream.clone(),
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
        impl<LHS, RHS, D, E> $trait<Var<RHS, E>> for Var<LHS, D>
        where
            LHS: Op<Data = Tensor<D>, Grad = Tensor<D>>,
            RHS: Op<Data = Tensor<E>, Grad = Tensor<E>>,
            D: Dimension + RemoveAxis + Max<E> + 'static,
            E: Dimension + RemoveAxis + 'static,
        {
            type Output = Var<$repr<LHS, RHS, D, E>, Maximum<D, E>>;

            fn $fun(self, other: Var<RHS, E>) -> Self::Output {
                Var::new(
                    Rc::new($repr::new(self.repr, other.repr)),
                    track_upstream(&self.upstream, &other.upstream),
                )
            }
        }
    };
}

impl_node_arithmetic_ops!(Add, add, AddOp);
impl_node_arithmetic_ops!(Sub, sub, SubOp);
impl_node_arithmetic_ops!(Mul, mul, MulOp);
impl_node_arithmetic_ops!(Div, div, DivOp);

impl<T, D> Neg for Var<T, D>
where
    T: Op<Data = Tensor<D>, Grad = Tensor<D>>,
    D: Dimension + RemoveAxis + 'static,
{
    type Output = Var<NegOp<T, D>, D>;
    fn neg(self) -> Self::Output {
        Var::new(Rc::new(NegOp::new(self.repr)), self.upstream.clone())
    }
}
