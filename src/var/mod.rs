pub(super) mod numeric;
pub(crate) mod ops;

use itertools::Itertools;
use numeric::DataRepr;
use ops::{
    AddOp, BinCatOp, Borrow, DivOp, DotOp, ExpOp, LeakyReLUOp, LnOp, MulOp, MultiCatOp, NegOp, Op,
    Param, PowOp, ReLUOp, ScalProdOp, SigmoidOp, SoftmaxOp, SubOp, SumOp, TOp, TanhOp,
};
use std::cell::{Ref, RefCell};
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::rc::Rc;

// A pointer to a node in the computational graph.
pub struct Var<T>
where
    T: Op,
{
    repr: Rc<T>,
    grad: Option<RefCell<DataRepr>>,
    upstream: Vec<Var<Param>>,
}

impl<T> Clone for Var<T>
where
    T: Op,
{
    fn clone(&self) -> Self {
        Var {
            repr: Rc::clone(&self.repr),
            grad: None,
            upstream: self.upstream.clone(),
        }
    }
}

impl Var<Param> {
    fn as_ptr(&self) -> *const Param {
        self.repr.deref() as *const Param
    }

    pub fn grad(&self) -> Ref<DataRepr> {
        self.repr.deref().grad.borrow()
    }
}

impl<T> Var<T>
where
    T: Op<Data = DataRepr, Grad = DataRepr>,
{
    pub(super) fn new(repr: Rc<T>, upstream: Vec<Var<Param>>) -> Self {
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

    pub fn upstream(&self) -> &[Var<Param>] {
        &self.upstream[..]
    }

    pub fn zero_gradient(&self) {
        for param in self.upstream() {
            param.repr.zero_grad();
        }
    }

    pub fn upstream_mut(&mut self) -> &mut [Var<Param>] {
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

    pub fn dot<U>(&self, other: &Var<U>) -> Var<ScalProdOp<T, U>>
    where
        U: Op<Data = DataRepr, Grad = DataRepr>,
    {
        Var::new(
            Rc::new(ScalProdOp::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn mm<U>(&self, other: &Var<U>) -> Var<DotOp<T, U>>
    where
        U: Op<Data = DataRepr, Grad = DataRepr>,
    {
        Var::new(
            Rc::new(DotOp::new(Rc::clone(&self.repr), Rc::clone(&other.repr))),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn pow(&self, exp: u16) -> Var<PowOp<T>> {
        Var::new(
            Rc::new(PowOp::new(Rc::clone(&self.repr), exp)),
            self.upstream.clone(),
        )
    }

    pub fn sum(&self) -> Var<SumOp<T>> {
        Var::new(
            Rc::new(SumOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn relu(&self) -> Var<ReLUOp<T>> {
        Var::new(
            Rc::new(ReLUOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn leaky_relu(&self, slope: f32) -> Var<LeakyReLUOp<T>> {
        Var::new(
            Rc::new(LeakyReLUOp::new(Rc::clone(&self.repr), slope)),
            self.upstream.clone(),
        )
    }

    pub fn sigmoid(&self) -> Var<SigmoidOp<T>> {
        Var::new(
            Rc::new(SigmoidOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn tanh(&self) -> Var<TanhOp<T>> {
        Var::new(
            Rc::new(TanhOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn ln(&self) -> Var<LnOp<T>> {
        Var::new(
            Rc::new(LnOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn exp(&self) -> Var<ExpOp<T>> {
        Var::new(
            Rc::new(ExpOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn softmax(&self, axis: usize) -> Var<SoftmaxOp<T>> {
        Var::new(
            Rc::new(SoftmaxOp::new(Rc::clone(&self.repr), axis)),
            self.upstream.clone(),
        )
    }

    pub fn t(&self) -> Var<TOp<T>> {
        Var::new(
            Rc::new(TOp::new(Rc::clone(&self.repr))),
            self.upstream.clone(),
        )
    }

    pub fn cat<U>(&self, other: &Var<U>, axis: usize) -> Var<BinCatOp<T, U>>
    where
        U: Op<Data = DataRepr, Grad = DataRepr>,
    {
        Var::new(
            Rc::new(BinCatOp::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
                axis,
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

pub fn multi_cat<T>(vars: &[&Var<T>], axis: usize) -> Var<MultiCatOp<T>>
where
    T: Op<Data = DataRepr, Grad = DataRepr>,
{
    let clones: Vec<Rc<T>> = vars.iter().map(|v| Rc::clone(&v.repr)).collect();
    let upstreams: Vec<&[Var<Param>]> = vars.iter().map(|var| var.upstream()).collect();

    let upstream = track_multi_upstream(upstreams);

    Var::new(Rc::new(MultiCatOp::new(&clones[..], axis)), upstream)
}

fn track_upstream(lhs_up: &[Var<Param>], rhs_up: &[Var<Param>]) -> Vec<Var<Param>> {
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

fn track_multi_upstream(upstreams: Vec<&[Var<Param>]>) -> Vec<Var<Param>> {
    upstreams
        .iter()
        .fold(Vec::<Var<Param>>::new(), |mut acc, other| {
            let mut addings = Vec::<Var<Param>>::new();
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
            RHS: Op<Data = DataRepr, Grad = DataRepr>,
            LHS: Op<Data = DataRepr, Grad = DataRepr>,
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

impl_node_arithmetic_ops!(Add, add, AddOp);
impl_node_arithmetic_ops!(Sub, sub, SubOp);
impl_node_arithmetic_ops!(Mul, mul, MulOp);
impl_node_arithmetic_ops!(Div, div, DivOp);

impl<T> Neg for Var<T>
where
    T: Op<Data = DataRepr, Grad = DataRepr>,
{
    type Output = Var<NegOp<T>>;
    fn neg(self) -> Self::Output {
        Var::new(Rc::new(NegOp::new(self.repr)), self.upstream.clone())
    }
}
