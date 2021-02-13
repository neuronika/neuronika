pub(super) mod numeric;
pub(super) mod reprs;

use itertools::Itertools;
use num_traits::{One, Zero};
use numeric::DataRepr;
use reprs::*;
use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::rc::Rc;

// A pointer to a node in the computational graph.
pub struct Var<T, A>
where
    T: InternalRepr,
    A: 'static
        + Copy
        + Debug
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Zero
        + One
        + Send
        + Sync,
{
    repr: Rc<T>,
    grad: Option<RefCell<DataRepr<A>>>,
    upstream: Vec<Var<Parameter<A>, A>>,
}

impl<T, A> Clone for Var<T, A>
where
    T: InternalRepr,
    A: Copy
        + Debug
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Zero
        + One
        + Send
        + Sync,
{
    fn clone(&self) -> Self {
        Var {
            repr: Rc::clone(&self.repr),
            grad: None,
            upstream: self.upstream.clone(),
        }
    }
}

impl<A> Var<Parameter<A>, A>
where
    A: Copy
        + Debug
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Zero
        + One
        + Send
        + Sync,
{
    fn as_ptr(&self) -> *const Parameter<A> {
        self.repr.deref() as *const Parameter<A>
    }

    pub fn grad(&self) -> Ref<DataRepr<A>> {
        self.repr.deref().grad.borrow()
    }
}

impl<T, A> Var<T, A>
where
    T: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
    A: Copy
        + Debug
        + Neg<Output = A>
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Zero
        + One
        + Send
        + Sync,
{
    pub(super) fn new(repr: Rc<T>, upstream: Vec<Var<Parameter<A>, A>>) -> Self {
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

    pub fn upstream(&self) -> &[Var<Parameter<A>, A>] {
        &self.upstream[..]
    }

    pub fn zero_gradient(&self) {
        for param in self.upstream() {
            param.repr.zero_grad();
        }
    }

    pub fn upstream_mut(&mut self) -> &mut [Var<Parameter<A>, A>] {
        &mut self.upstream[..]
    }

    pub fn backward(&mut self, seed: A) {
        let data_ref: &DataRepr<A> = &self.repr.data();
        self.grad
            .get_or_insert_with(|| RefCell::new(data_ref.map(|_| seed)))
            .borrow_mut()
            .map_inplace(|_| seed);

        if let Some(ref grad) = self.grad {
            self.repr.backward(&grad.borrow());
        }
    }

    pub fn dot<U>(&self, other: &Var<U, A>) -> Var<InternalVecDot<A, T, U>, A>
    where
        U: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
    {
        Var::new(
            Rc::new(InternalVecDot::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }

    pub fn mm<U>(&self, other: &Var<U, A>) -> Var<InternalDot<A, T, U>, A>
    where
        U: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
    {
        Var::new(
            Rc::new(InternalDot::new(
                Rc::clone(&self.repr),
                Rc::clone(&other.repr),
            )),
            track_upstream(&self.upstream, &other.upstream),
        )
    }
}

fn track_upstream<
    A: Debug
        + Copy
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Div<Output = A>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + Zero
        + One
        + Send
        + Sync,
>(
    lhs_up: &[Var<Parameter<A>, A>],
    rhs_up: &[Var<Parameter<A>, A>],
) -> Vec<Var<Parameter<A>, A>> {
    lhs_up
        .iter()
        .merge_join_by(rhs_up.iter(), |lhs_par, rhs_par| {
            lhs_par.as_ptr().cmp(&rhs_par.as_ptr())
        })
        .map(|choice| match choice {
            itertools::EitherOrBoth::Left(lhs_par) => lhs_par,
            itertools::EitherOrBoth::Right(lhs_par) => lhs_par,
            itertools::EitherOrBoth::Both(lhs_par, _) => lhs_par,
        })
        .cloned()
        .collect()
}

macro_rules! impl_node_arithmetic_ops {
    ($trait:ident, $fun:ident, $repr:ident) => {
        impl<A, LHS, RHS> $trait<Var<RHS, A>> for Var<LHS, A>
        where
            A: Copy
                + Debug
                + Add<Output = A>
                + Sub<Output = A>
                + Mul<Output = A>
                + Div<Output = A>
                + AddAssign
                + SubAssign
                + MulAssign
                + DivAssign
                + Zero
                + One
                + Send
                + Sync
                + Neg<Output = A>,
            RHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
            LHS: InternalRepr<Data = DataRepr<A>, Grad = DataRepr<A>>,
        {
            type Output = Var<$repr<A, LHS, RHS>, A>;
            fn $fun(self, other: Var<RHS, A>) -> Self::Output {
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
