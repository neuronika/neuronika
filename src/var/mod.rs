pub(super) mod numeric;
pub(super) mod reprs;

use itertools::Itertools;
use numeric::DataRepr;
use reprs::{
    Borrow, InternalAdd, InternalBinConcat, InternalDiv, InternalDot, InternalExp, InternalLn,
    InternalMul, InternalMultiConcat, InternalNeg, InternalPow, InternalReLU, InternalRepr,
    InternalSigmoid, InternalSoftmax, InternalSub, InternalSum, InternalT, InternalVecDot,
    Parameter,
};
use std::cell::{Ref, RefCell};
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::rc::Rc;

// A pointer to a node in the computational graph.
pub struct Var<T>
where
    T: InternalRepr,
{
    repr: Rc<T>,
    grad: Option<RefCell<DataRepr>>,
    upstream: Vec<Var<Parameter>>,
}

impl<T> Clone for Var<T>
where
    T: InternalRepr,
{
    fn clone(&self) -> Self {
        Var {
            repr: Rc::clone(&self.repr),
            grad: None,
            upstream: self.upstream.clone(),
        }
    }
}

impl Var<Parameter> {
    fn as_ptr(&self) -> *const Parameter {
        self.repr.deref() as *const Parameter
    }

    pub fn grad(&self) -> Ref<DataRepr> {
        self.repr.deref().grad.borrow()
    }
}

impl<T> Var<T>
where
    T: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    pub(super) fn new(repr: Rc<T>, upstream: Vec<Var<Parameter>>) -> Self {
        Var {
            repr: repr,
            grad: None,
            upstream: upstream,
        }
    }

    pub fn data(&self) -> Borrow<T::Data> {
        self.repr.data()
    }

    pub fn forward(&self) {
        self.repr.forward()
    }

    pub fn clear(&self) {
        self.repr.clear();
    }

    pub fn upstream(&self) -> &[Var<Parameter>] {
        &self.upstream[..]
    }

    pub fn zero_gradient(&self) {
        for param in self.upstream() {
            param.repr.zero_grad();
        }
    }

    pub fn upstream_mut(&mut self) -> &mut [Var<Parameter>] {
        &mut self.upstream[..]
    }

    pub fn backward(&mut self, seed: f32) {
        let data_ref: &DataRepr = &self.repr.data();
        self.grad
            .get_or_insert_with(|| RefCell::new(data_ref.map(|_| seed)))
            .borrow_mut()
            .map_inplace(|_| seed);

        if let Some(ref grad) = self.grad {
            self.repr.backward(&grad.borrow());
        }
    }

    pub fn dot<U>(&self, other: &Var<U>) -> Var<InternalVecDot<T, U>>
    where
        U: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    {
        Var::new(
            Rc::new(InternalVecDot::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn mm<U>(&self, other: &Var<U>) -> Var<InternalDot<T, U>>
    where
        U: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    {
        Var::new(
            Rc::new(InternalDot::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn pow(&self, exp: u16) -> Var<InternalPow<T>> {
        Var::new(
            Rc::new(InternalPow::new(Rc::clone(&self.repr), exp)),
            self.upstream.clone(),
        )
    }

    pub fn sum(&self) -> Var<InternalSum<T>> {
        Var::new(
            Rc::new(InternalSum::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn relu(&self) -> Var<InternalReLU<T>> {
        Var::new(
            Rc::new(InternalReLU::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn sigmoid(&self) -> Var<InternalSigmoid<T>> {
        Var::new(
            Rc::new(InternalSigmoid::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn ln(&self) -> Var<InternalLn<T>> {
        Var::new(
            Rc::new(InternalLn::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn exp(&self) -> Var<InternalExp<T>> {
        Var::new(
            Rc::new(InternalExp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn softmax(&self, axis: usize) -> Var<InternalSoftmax<T>> {
        Var::new(
            Rc::new(InternalSoftmax::new(Rc::clone(&self.repr), axis)),
            self.upstream.clone(),
        )
    }

    pub fn t(&self) -> Var<InternalT<T>> {
        Var::new(
            Rc::new(InternalT::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn cat<U>(&self, other: &Var<U>, axis: usize) -> Var<InternalBinConcat<T, U>>
    where
        U: InternalRepr<Data = DataRepr, Grad = DataRepr>,
    {
        Var::new(
            Rc::new(InternalBinConcat::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
                axis,
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

pub fn multi_cat<T>(vars: &[&Var<T>], axis: usize) -> Var<InternalMultiConcat<T>>
where
    T: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    let clones: Vec<Rc<T>> = vars.iter().map(|v| Rc::clone(&v.repr)).collect();
    let upstreams: Vec<&[Var<Parameter>]> = vars.iter().map(|var| var.upstream()).collect();

    let upstream = track_multi_upstream(upstreams);

    Var::new(
        Rc::new(InternalMultiConcat::new(&clones[..], axis)),
        upstream,
    )
}

fn track_upstream(lhs_up: &[Var<Parameter>], rhs_up: &[Var<Parameter>]) -> Vec<Var<Parameter>> {
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

fn track_multi_upstream(upstreams: Vec<&[Var<Parameter>]>) -> Vec<Var<Parameter>> {
    upstreams
        .iter()
        .fold(Vec::<Var<Parameter>>::new(), |mut acc, other| {
            let mut addings = Vec::<Var<Parameter>>::new();
            acc.iter()
                .merge_join_by(other.iter(), |acc_el, other_el| {
                    acc_el.as_ptr().cmp(&other_el.as_ptr())
                })
                .for_each(|choice| match choice {
                    itertools::EitherOrBoth::Left(_) => (),
                    itertools::EitherOrBoth::Right(rhs) => addings.push(rhs.clone()),
                    itertools::EitherOrBoth::Both(_, _) => (),
                });
            acc.extend(addings);
            acc
        })
}

macro_rules! impl_node_arithmetic_ops {
    ($trait:ident, $fun:ident, $repr:ident) => {
        impl<LHS, RHS> $trait<Var<RHS>> for Var<LHS>
        where
            RHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
            LHS: InternalRepr<Data = DataRepr, Grad = DataRepr>,
        {
            type Output = Var<$repr<LHS, RHS>>;
            fn $fun(self, other: Var<RHS>) -> Self::Output {
                Var::new(
                    Rc::new($repr::new(self.repr, other.repr)),
                    track_upstream(&self.upstream, &other.upstream),
                )
            }
        }
    };
}

impl_node_arithmetic_ops!(Add, add, InternalAdd);
impl_node_arithmetic_ops!(Sub, sub, InternalSub);
impl_node_arithmetic_ops!(Mul, mul, InternalMul);
impl_node_arithmetic_ops!(Div, div, InternalDiv);

impl<T> Neg for Var<T>
where
    T: InternalRepr<Data = DataRepr, Grad = DataRepr>,
{
    type Output = Var<InternalNeg<T>>;
    fn neg(self) -> Self::Output {
        Var::new(Rc::new(InternalNeg::new(self.repr)), self.upstream.clone())
    }
}
