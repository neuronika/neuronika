use ndarray::linalg::{general_mat_mul, general_mat_vec_mul};
use ndarray::{arr1, Array, Array2, ArrayView1, Axis, Dimension, Ix1, Ix2, Ix3, RemoveAxis, Zip};
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt::{Display, Formatter, Result};
use std::ops::{Add, Div, Mul, Neg, Sub};

// ===================================== Computational Graph Aux. Components =====================================

// Forward action tracker. Ensures
// that the actual computation only happens
// when the node is fully accumulated.
#[derive(Debug, PartialEq)]
pub enum ForwardAction {
    Evaluate,
    Cached,
}

// Backward action tracker. Keeps track
// of the gradient accumulation.
#[derive(Debug, PartialEq)]
pub enum BackwardAction {
    // Set the gradient.
    Set,
    // Accumulates the gradient.
    Increment,
}

// Keeps track of the number of times
// that a node in the computational graph
// has been evaluated during either the forward
// or the backward pass.
#[derive(Debug, Default)]
pub struct PassCounter {
    forward_count: Cell<usize>,
    backward_count: Cell<usize>,
}

impl PassCounter {
    pub fn clear(&self) {
        self.forward_count.set(0);
        self.backward_count.set(0);
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.forward_count.get() == 0
    }

    pub fn recurse_backward(&self) -> bool {
        let backward_count = self.backward_count.get();
        let forward_count = self.forward_count.get();

        if backward_count == forward_count {
            self.clear();
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn forward_action(&self) -> ForwardAction {
        let count = self.forward_count.get();
        self.forward_count.set(count + 1);

        match count {
            0 => ForwardAction::Evaluate,
            _ => ForwardAction::Cached,
        }
    }

    #[inline(always)]
    pub fn backward_action(&self) -> BackwardAction {
        let backward_count = self.backward_count.get();

        let action = match backward_count {
            0 => BackwardAction::Set,
            _ => BackwardAction::Increment,
        };

        self.backward_count.set(backward_count + 1);
        action
    }
}

// ============================================= Types Relations =============================================

/// Relation among two [Dimension] items.
///
/// This trait is needed in order to use the *broadcasting* semantic of the standard algebraic operations
/// among [tensor]s.
///
/// [tensor]: prova::Tensor
pub trait Max<R>
where
    Self: Dimension,
    R: Dimension,
{
    type Output: Dimension + RemoveAxis + Max<R> + Max<Self>;

    fn max(self, rhs: R) -> Maximum<Self, R>;
}

/// The [Max] of two [Dimension] items.
///
/// [Max]: crate::Max
/// [Dimension]: ndarray::Dimension
pub type Maximum<L, R> = <L as Max<R>>::Output;

impl Max<Ix1> for Ix1 {
    type Output = Ix1;

    fn max(self, rhs: Ix1) -> Ix1 {
        let self_els = self.into_pattern();
        let rhs_els = rhs.into_pattern();
        if self_els > rhs_els {
            self
        } else {
            rhs
        }
    }
}

impl Max<Ix2> for Ix2 {
    type Output = Ix2;

    fn max(self, rhs: Ix2) -> Ix2 {
        let dim0 = if self.into_pattern().0 == 1 {
            rhs.into_pattern().1
        } else {
            self.into_pattern().0
        };
        let dim1 = if self.into_pattern().1 == 1 {
            rhs.into_pattern().1
        } else {
            self.into_pattern().1
        };
        Ix2(dim0, dim1)
    }
}
impl Max<Ix3> for Ix3 {
    type Output = Ix3;

    fn max(self, rhs: Ix3) -> Ix3 {
        let dim0 = if self.into_pattern().0 == 1 {
            rhs.into_pattern().1
        } else {
            self.into_pattern().0
        };
        let dim1 = if self.into_pattern().1 == 1 {
            rhs.into_pattern().1
        } else {
            self.into_pattern().1
        };
        let dim2 = if self.into_pattern().2 == 1 {
            rhs.into_pattern().2
        } else {
            self.into_pattern().2
        };
        Ix3(dim0, dim1, dim2)
    }
}

impl Max<Ix1> for Ix2 {
    type Output = Ix2;

    fn max(self, rhs: Ix1) -> Ix2 {
        let self_pattern = self.into_pattern();
        let right_pattern = rhs.into_pattern();

        let broadcasted = if self_pattern.1 == 1 {
            right_pattern
        } else {
            self_pattern.1
        };

        Ix2(self_pattern.0, broadcasted)
    }
}

impl Max<Ix2> for Ix1 {
    type Output = Ix2;

    fn max(self, rhs: Ix2) -> Ix2 {
        let self_pattern = self.into_pattern();
        let right_pattern = rhs.into_pattern();

        let broadcasted = if right_pattern.1 == 1 {
            self_pattern
        } else {
            right_pattern.1
        };

        Ix2(right_pattern.0, broadcasted)
    }
}

impl Max<Ix1> for Ix3 {
    type Output = Ix3;

    fn max(self, rhs: Ix1) -> Ix3 {
        let self_pattern = self.into_pattern();
        let right_pattern = rhs.into_pattern();

        let broadcasted = if self_pattern.2 == 1 {
            right_pattern
        } else {
            self_pattern.2
        };

        Ix3(self_pattern.0, self_pattern.1, broadcasted)
    }
}

impl Max<Ix3> for Ix1 {
    type Output = Ix3;

    fn max(self, rhs: Ix3) -> Ix3 {
        let self_pattern = self.into_pattern();
        let right_pattern = rhs.into_pattern();

        let broadcasted = if right_pattern.2 == 1 {
            self_pattern
        } else {
            right_pattern.2
        };

        Ix3(right_pattern.0, right_pattern.1, broadcasted)
    }
}

impl Max<Ix2> for Ix3 {
    type Output = Ix3;

    fn max(self, rhs: Ix2) -> Ix3 {
        let self_pattern = self.into_pattern();
        let right_pattern = rhs.into_pattern();

        let broadcasted = (
            if self_pattern.1 == 1 {
                right_pattern.0
            } else {
                self_pattern.1
            },
            if self_pattern.2 == 1 {
                right_pattern.1
            } else {
                self_pattern.2
            },
        );

        Ix3(self_pattern.0, broadcasted.0, broadcasted.1)
    }
}

impl Max<Ix3> for Ix2 {
    type Output = Ix3;

    fn max(self, rhs: Ix3) -> Ix3 {
        let self_pattern = self.into_pattern();
        let right_pattern = rhs.into_pattern();

        let broadcasted = (
            if right_pattern.1 == 1 {
                self_pattern.0
            } else {
                right_pattern.1
            },
            if right_pattern.2 == 1 {
                self_pattern.1
            } else {
                right_pattern.2
            },
        );

        Ix3(self_pattern.0, broadcasted.0, broadcasted.1)
    }
}

pub trait VStack<R>
where
    Self: Dimension,
    R: Dimension,
{
    type Output: Dimension + RemoveAxis;

    fn vstack(self, rhs: R) -> VStacked<Self, R>;
}

pub type VStacked<L, R> = <L as VStack<R>>::Output;

impl VStack<Ix1> for Ix1 {
    type Output = Ix2;
    fn vstack(self, rhs: Ix1) -> Ix2 {
        Ix2(2, rhs.size())
    }
}

impl VStack<Ix2> for Ix2 {
    type Output = Ix2;
    fn vstack(self, rhs: Ix2) -> Ix2 {
        Ix2(
            self.into_pattern().0 + rhs.into_pattern().0,
            rhs.into_pattern().1,
        )
    }
}

impl VStack<Ix3> for Ix3 {
    type Output = Ix3;
    fn vstack(self, rhs: Ix3) -> Ix3 {
        Ix3(
            self.into_pattern().0 + rhs.into_pattern().0,
            rhs.into_pattern().1,
            rhs.into_pattern().2,
        )
    }
}

pub trait HStack<R>
where
    Self: Dimension,
    R: Dimension,
{
    type Output: Dimension + RemoveAxis;

    fn hstack(self, rhs: R) -> HStacked<Self, R>;
}

pub type HStacked<L, R> = <L as HStack<R>>::Output;

impl HStack<Ix1> for Ix1 {
    type Output = Ix1;
    fn hstack(self, rhs: Ix1) -> Ix1 {
        Ix1(self.size() + rhs.size())
    }
}

impl HStack<Ix2> for Ix2 {
    type Output = Ix2;
    fn hstack(self, rhs: Ix2) -> Ix2 {
        Ix2(
            self.into_pattern().0,
            self.into_pattern().1 + rhs.into_pattern().1,
        )
    }
}

impl HStack<Ix3> for Ix3 {
    type Output = Ix3;
    fn hstack(self, rhs: Ix3) -> Ix3 {
        Ix3(
            self.into_pattern().0,
            self.into_pattern().1 + rhs.into_pattern().1,
            rhs.into_pattern().2,
        )
    }
}

pub trait DStack<R>
where
    Self: Dimension,
    R: Dimension,
{
    type Output: Dimension + RemoveAxis;

    fn dstack(self, rhs: R) -> DStacked<Self, R>;
}

pub type DStacked<L, R> = <L as DStack<R>>::Output;

impl DStack<Ix2> for Ix2 {
    type Output = Ix3;
    fn dstack(self, _: Ix2) -> Ix3 {
        Ix3(self.into_pattern().0, self.into_pattern().1, 2)
    }
}

// =========================================== Operators Overload ===========================================

/// Automatically implements the overload of the `+`, `-`, `*` and `/` binary algebraic operators for
/// [tensor]s.
///
/// [tensor]: crate::Tensor
macro_rules! impl_arithmetic_ops {
    ($(($trait: ident, $fun: ident, $op: tt)),+ $(,)?) => {
        $(
            impl<L, R> $trait<&Tensor<R>> for &Tensor<L>
            where
                L: Dimension + Max<R>,
                R: Dimension,
            {
                type Output = Tensor<Maximum<L, R>>;

                fn $fun(self, rhs: &Tensor<R>) -> Self::Output {
                    let shape = self.data.raw_dim().max(rhs.data.raw_dim());
                    let mut data = Array::<f32, Maximum<L, R>>::zeros(shape);
                    Zip::from(&mut data)
                        .and_broadcast(&self.data)
                        .and_broadcast(&rhs.data)
                        .par_apply(|res, l, r| *res = l $op r);

                    Self::Output { data }
                }
            }

        )*
    };
}

impl_arithmetic_ops!((Add, add, +), (Sub, sub, -), (Mul, mul, *), (Div, div, /));

// Implements the Neg trait.
impl<D> Neg for &Tensor<D>
where
    D: Dimension,
{
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        Self::Output { data: -&self.data }
    }
}

// ============================================ Forward Pass Functions ============================================

// Implements forward pass funcs for binary ops.
macro_rules! impl_fwd_bin_ops {
    ($fun:ident, $closure:expr) => {
        pub(super) fn $fun<E, F>(&mut self, lhs: &Tensor<E>, rhs: &Tensor<F>)
        where
            E: Dimension,
            F: Dimension,
        {
            Zip::from(&mut self.data)
                .and_broadcast(&lhs.data)
                .and_broadcast(&rhs.data)
                .apply($closure);
        }
    };
}

// Implements forward pass funcs for unary ops.
macro_rules! impl_fwd_un_ops {
    ($fun:ident, $closure:expr) => {
        pub(super) fn $fun(&mut self, other: &Self) {
            Zip::from(&mut self.data).and(&other.data).apply($closure);
        }
    };
}

// Implements forward pass funcs for parametrized unary ops.
macro_rules! impl_fwd_un_ops_param {
    ($fun:ident, $closure:expr, $param:ident, $param_t:ty) => {
        pub(super) fn $fun(&mut self, other: &Self, $param: $param_t) {
            Zip::from(&mut self.data).and(&other.data).apply($closure);
        }
    };
}

// ============================================ Backward Pass Functions ============================================

// Implements backward pass funcs for unary non-parametrized ops.
macro_rules! impl_bkwrd_un_ops {
    ($fun:ident, $closure_set:expr, $closure_incr:expr) => {
        pub(super) fn $fun(&mut self, down_grad: &Self, t: &Self, action: &BackwardAction) {
            match action {
                BackwardAction::Set => {
                    Zip::from(&mut self.data)
                        .and(&down_grad.data)
                        .and(&t.data)
                        .apply($closure_set);
                }
                BackwardAction::Increment => {
                    Zip::from(&mut self.data)
                        .and(&down_grad.data)
                        .and(&t.data)
                        .apply($closure_incr);
                }
            }
        }
    };
}

// Implements backward pass funcs for unary parametrized ops.
macro_rules! impl_bkwrd_un_ops_param {
    ($fun:ident, $closure_set:expr, $closure_incr:expr, $param:ident, $param_t:ty) => {
        pub(super) fn $fun(
            &mut self,
            down_grad: &Self,
            t: &Self,
            action: &BackwardAction,
            $param: $param_t,
        ) {
            match action {
                BackwardAction::Set => {
                    Zip::from(&mut self.data)
                        .and(&down_grad.data)
                        .and(&t.data)
                        .apply($closure_set);
                }
                BackwardAction::Increment => {
                    Zip::from(&mut self.data)
                        .and(&down_grad.data)
                        .and(&t.data)
                        .apply($closure_incr);
                }
            }
        }
    };
}

// =============================================== Tensor Type ===============================================

/// A *n*-dimensional [tensor] of *real* values that support efficient [broadcasting].
///
/// All the standard mathematic binary operators like `+`, `-`, `*` and `/`, exploit **SIMD** computation
/// and are also executed in multiple threads whenever possible.
///
/// [tensor]: https://en.wikipedia.org/wiki/Tensor
/// [broadcasting]: https://numpy.org/devdocs/user/theory.broadcasting.html
#[derive(Debug, PartialEq)]
pub struct Tensor<D>
where
    D: Dimension,
{
    /// Content of the tensor
    pub data: Array<f32, D>,
}

impl<D> Display for Tensor<D>
where
    D: Dimension,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.data)
    }
}

// ============================================ Impl for Tensor Type ============================================

// Methods specific to the two dimensional Tensor.
impl Tensor<Ix2> {
    pub(super) fn mat_mul(
        &self,
        rhs: &Self,
        target: &mut Self,
        alpha: f32,
        beta: f32,
        t_lhs: bool,
        t_rhs: bool,
    ) {
        match (t_lhs, t_rhs) {
            (true, true) => {
                general_mat_mul(alpha, &self.data.t(), &rhs.data.t(), beta, &mut target.data)
            }
            (true, false) => {
                general_mat_mul(alpha, &self.data.t(), &rhs.data, beta, &mut target.data)
            }
            (false, true) => {
                general_mat_mul(alpha, &self.data, &rhs.data.t(), beta, &mut target.data)
            }
            (false, false) => general_mat_mul(alpha, &self.data, &rhs.data, beta, &mut target.data),
        }
    }

    pub(super) fn mat_vec_mul(
        &self,
        rhs: &Tensor<Ix1>,
        target: &mut Tensor<Ix1>,
        alpha: f32,
        beta: f32,
        t: bool,
    ) {
        match t {
            true => general_mat_vec_mul(alpha, &self.data.t(), &rhs.data, beta, &mut target.data),
            false => general_mat_vec_mul(alpha, &self.data, &rhs.data, beta, &mut target.data),
        }
    }
}

// Methods for all dimensional Tensors.
impl<D> Tensor<D>
where
    D: Dimension + RemoveAxis,
{
    // Gets the number of elements stored in this
    // Tensor.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    // Gets the shape of the data stored in this
    // Tensor.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    // Initializes another Tensor with the same
    // dimensionsality as this with all its
    // values set to zero.
    pub fn zeros(&self) -> Self {
        Self {
            data: <Array<f32, D>>::zeros(self.data.raw_dim()),
        }
    }

    pub fn set_zero(&mut self) {
        self.data.map_inplace(|el| *el = 0.0);
    }

    // Creates another Tensor with transposed data.
    pub fn t(&self) -> Self {
        Self {
            data: self.data.t().to_owned(),
        }
    }

    // Creates another Tensor whose data is the sum of
    // all of self's elements.
    pub fn sum(&self) -> Tensor<Ix1> {
        Tensor {
            data: arr1(&[self.data.sum()]),
        }
    }

    pub(super) fn accumulate<E>(&mut self, other: &Tensor<E>, scale: f32, action: &BackwardAction)
    where
        E: Dimension + RemoveAxis,
    {
        let (trgt_data, other_data) = { (&mut self.data, &other.data) };

        if trgt_data.len() == 1 {
            match action {
                BackwardAction::Set => {
                    Zip::from(trgt_data).apply(|el| *el = other_data.sum() * scale)
                }
                BackwardAction::Increment => {
                    Zip::from(trgt_data).apply(|el| *el += other_data.sum() * scale)
                }
            }
        } else {
            match trgt_data.ndim().cmp(&other_data.ndim()) {
                Ordering::Less => {
                    let mut dyn_other = other_data.sum_axis(Axis(0)).into_dyn();
                    while trgt_data.ndim() < dyn_other.ndim() {
                        dyn_other = dyn_other.sum_axis(Axis(0));
                    }
                    let static_other = dyn_other.into_dimensionality::<D>().unwrap();
                    let mut axis_of_len_one = false;
                    for i in 0..trgt_data.ndim() {
                        let size = trgt_data.len_of(Axis(i));
                        if size == 1_usize {
                            axis_of_len_one = true;
                            match action {
                                BackwardAction::Set => {
                                    Zip::from(trgt_data.lanes_mut(Axis(i)))
                                        .and(static_other.lanes(Axis(i)))
                                        .apply(|dest_lane, src_lane| {
                                            Zip::from(dest_lane).apply(|dest_view_el| {
                                                *dest_view_el = src_lane.sum() * scale
                                            });
                                        });
                                }
                                BackwardAction::Increment => {
                                    Zip::from(trgt_data.lanes_mut(Axis(i)))
                                        .and(static_other.lanes(Axis(i)))
                                        .apply(|dest_lane, src_lane| {
                                            Zip::from(dest_lane).apply(|dest_view_el| {
                                                *dest_view_el += src_lane.sum() * scale
                                            });
                                        });
                                }
                            }
                        }
                    }
                    if !axis_of_len_one {
                        match action {
                            BackwardAction::Set => {
                                Zip::from(trgt_data)
                                    .and(&static_other)
                                    .apply(|el_trgt, el_other| *el_trgt = *el_other * scale);
                            }
                            BackwardAction::Increment => {
                                Zip::from(trgt_data)
                                    .and(&static_other)
                                    .apply(|el_trgt, el_other| *el_trgt += *el_other * scale);
                            }
                        }
                    }
                }
                Ordering::Equal => {
                    let other_same_dim = other_data.view().into_dimensionality::<D>().unwrap();
                    let mut axis_of_len_one = false;
                    for i in 0..trgt_data.ndim() {
                        let size = trgt_data.len_of(Axis(i));
                        if size == 1_usize {
                            axis_of_len_one = true;
                            match action {
                                BackwardAction::Set => {
                                    Zip::from(trgt_data.lanes_mut(Axis(i)))
                                        .and(other_same_dim.lanes(Axis(i)))
                                        .apply(|dest_lane, src_lane| {
                                            Zip::from(dest_lane).apply(|dest_view_el| {
                                                *dest_view_el = src_lane.sum() * scale
                                            });
                                        });
                                }
                                BackwardAction::Increment => {
                                    Zip::from(trgt_data.lanes_mut(Axis(i)))
                                        .and(other_same_dim.lanes(Axis(i)))
                                        .apply(|dest_lane, src_lane| {
                                            Zip::from(dest_lane).apply(|dest_view_el| {
                                                *dest_view_el += src_lane.sum() * scale
                                            });
                                        });
                                }
                            }
                        }
                    }
                    if !axis_of_len_one {
                        match action {
                            BackwardAction::Set => {
                                Zip::from(trgt_data)
                                    .and_broadcast(&other_same_dim)
                                    .apply(|el_trgt, el_other| *el_trgt = *el_other * scale);
                            }
                            BackwardAction::Increment => {
                                Zip::from(trgt_data)
                                    .and_broadcast(&other_same_dim)
                                    .apply(|el_trgt, el_other| *el_trgt += *el_other * scale);
                            }
                        }
                    }
                }
                Ordering::Greater => match action {
                    BackwardAction::Set => {
                        Zip::from(trgt_data)
                            .and_broadcast(other_data)
                            .apply(|el_trgt, el_other| *el_trgt = *el_other * scale);
                    }
                    BackwardAction::Increment => {
                        Zip::from(trgt_data)
                            .and_broadcast(other_data)
                            .apply(|el_trgt, el_other| *el_trgt += *el_other * scale);
                    }
                },
            }
        }
    }

    impl_fwd_bin_ops!(add_fwd, |self_el, lhs_el, rhs_el| *self_el =
        *lhs_el + *rhs_el);
    impl_fwd_bin_ops!(sub_fwd, |self_el, lhs_el, rhs_el| *self_el =
        *lhs_el - *rhs_el);
    impl_fwd_bin_ops!(mul_fwd, |self_el, lhs_el, rhs_el| *self_el =
        *lhs_el * *rhs_el);
    impl_fwd_bin_ops!(div_fwd, |self_el, lhs_el, rhs_el| *self_el =
        *lhs_el / *rhs_el);

    impl_fwd_un_ops!(relu_fwd, |data_el, src_el| *data_el =
        if *src_el < 0.0 { 0.0 } else { *src_el });
    impl_bkwrd_un_ops!(
        relu_bkwrd,
        |grad_val, down_grad_val, data_val| {
            *grad_val = if *data_val > 0.0 { *down_grad_val } else { 0.0 }
        },
        |grad_val, down_grad_val, data_val| {
            *grad_val += if *data_val > 0.0 { *down_grad_val } else { 0.0 }
        }
    );

    impl_fwd_un_ops!(leaky_relu_fwd, |data_el, src_el| {
        *data_el = if *src_el < 0.0 { 0.01 } else { *src_el }
    });
    impl_bkwrd_un_ops!(
        leaky_relu_bkwrd,
        |grad_val, down_grad_val, data_val| {
            *grad_val = if *data_val > 0.0 {
                *down_grad_val
            } else {
                0.01
            }
        },
        |grad_val, down_grad_val, data_val| {
            *grad_val += if *data_val > 0.0 {
                *down_grad_val
            } else {
                0.01
            }
        }
    );

    impl_fwd_un_ops!(softplus_fwd, |data_el, src_el| {
        *data_el = if *src_el < -15.0 {
            0.0
        } else if *src_el > 15.0 {
            *src_el
        } else {
            (1.0 + src_el.exp()).ln()
        }
    });
    impl_bkwrd_un_ops!(
        softplus_bkwrd,
        |grad_val, down_grad_val, data_val| {
            *grad_val = if *data_val >= 15.0 {
                *down_grad_val
            } else if *data_val <= -15.0 {
                0.0
            } else {
                down_grad_val / (1.0 + (-*data_val).exp())
            }
        },
        |grad_val, down_grad_val, data_val| {
            *grad_val += if *data_val >= 15.0 {
                *down_grad_val
            } else if *data_val <= -15.0 {
                0.0
            } else {
                down_grad_val / (1.0 + (-*data_val).exp())
            }
        }
    );

    impl_fwd_un_ops!(sigmoid_fwd, |data_el, src_el| {
        *data_el = if *src_el >= 15.0 {
            1.0
        } else if *src_el <= -15.0 {
            0.0
        } else {
            1.0 / (1.0 + (-*src_el).exp())
        }
    });
    impl_bkwrd_un_ops!(
        sigmoid_bkwrd,
        |grad_val, down_grad_val, data_val| {
            *grad_val = *down_grad_val * *data_val * (1.0 - *data_val)
        },
        |grad_val, down_grad_val, data_val| {
            *grad_val += *down_grad_val * *data_val * (1.0 - *data_val)
        }
    );

    impl_fwd_un_ops!(exp_fwd, |data_el, src_el| *data_el = src_el.exp());
    impl_bkwrd_un_ops!(
        exp_bkwrd,
        |grad_val, down_grad_val, data_val| { *grad_val = *down_grad_val * *data_val },
        |grad_val, down_grad_val, data_val| { *grad_val += *down_grad_val * *data_val }
    );

    impl_fwd_un_ops!(tanh_fwd, |data_el, src_el| *data_el = src_el.tanh());
    impl_bkwrd_un_ops!(
        tanh_bkwrd,
        |grad_val, down_grad_val, data_val| {
            *grad_val = *down_grad_val * (1.0 - data_val.powi(2))
        },
        |grad_val, down_grad_val, data_val| {
            *grad_val += *down_grad_val * (1.0 - data_val.powi(2))
        }
    );

    impl_fwd_un_ops_param!(
        pow_fwd,
        |data_el, src_el| *data_el = src_el.powi(exp),
        exp,
        i32
    );
    impl_bkwrd_un_ops_param!(
        pow_bkwrd,
        |grad_val, down_grad_val, data_val| {
            *grad_val = *down_grad_val * data_val.powi(exp - 1) * exp as f32
        },
        |grad_val, down_grad_val, data_val| {
            *grad_val += *down_grad_val * data_val.powi(exp - 1) * exp as f32
        },
        exp,
        i32
    );

    pub fn softmax(&self, axis: usize) -> Self {
        let mut new = self.zeros();
        Zip::from(self.data.lanes(Axis(axis)))
            .and(new.data.lanes_mut(Axis(axis)))
            .apply(|lane_self, lane_new| {
                let max = lane_self.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = &lane_self.map(|el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_new)
                    .and(num)
                    .apply(|lane_new_el, num_el| *lane_new_el = *num_el / den);
            });
        new
    }

    pub fn softmax_fwd(&mut self, t: &Self, axis: usize) {
        Zip::from(t.data.lanes(Axis(axis)))
            .and(self.data.lanes_mut(Axis(axis)))
            .apply(|lane_self, lane_new| {
                let max = lane_self.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = &lane_self.map(|el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_new)
                    .and(num)
                    .apply(|lane_new_el, num_el| *lane_new_el = *num_el / den);
            });
    }

    pub fn softmax_bkwrd(
        &mut self,
        input_grad: &Self,
        data: &Self,
        jacobian: &mut Array2<f32>,
        action: &BackwardAction,
        axis: usize,
    ) {
        fn fill_jacobian(jacobian: &mut Array2<f32>, data: &ArrayView1<f32>) {
            for (row_idx, (mut row, row_val)) in jacobian
                .genrows_mut()
                .into_iter()
                .zip(data.iter())
                .enumerate()
            {
                for (col_idx, (grad, col_val)) in row
                    .as_slice_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(data.iter())
                    .enumerate()
                {
                    if row_idx == col_idx {
                        *grad = row_val * (1.0 - col_val);
                    } else {
                        *grad = -row_val * col_val;
                    }
                }
            }
        };
        let beta = match action {
            BackwardAction::Set => 0.0,
            BackwardAction::Increment => 1.0,
        };
        Zip::from(self.data.lanes_mut(Axis(axis)))
            .and(data.data.lanes(Axis(axis)))
            .and(input_grad.data.lanes(Axis(axis)))
            .apply(|mut d_g_col, data_col, grad_col| {
                fill_jacobian(jacobian, &data_col);
                general_mat_vec_mul(1.0, &jacobian, &grad_col, beta, &mut d_g_col);
            });
    }

    pub(super) fn vstack<E: Dimension>(&self, other: &Tensor<E>) -> Tensor<VStacked<D, E>>
    where
        D: VStack<E>,
    {
        let (self_data, other_data) = { (&self.data, &other.data) };

        let mut new = Tensor {
            data: Array::<f32, VStacked<D, E>>::zeros(
                self_data.raw_dim().vstack(other_data.raw_dim()),
            ),
        };

        new.vstack_fwd(self, other);
        new
    }

    pub(super) fn hstack<E: Dimension>(&self, other: &Tensor<E>) -> Tensor<HStacked<D, E>>
    where
        D: HStack<E>,
    {
        let (self_data, other_data) = { (&self.data, &other.data) };

        let mut new = Tensor {
            data: Array::<f32, HStacked<D, E>>::zeros(
                self_data.raw_dim().hstack(other_data.raw_dim()),
            ),
        };

        new.hstack_fwd(self, other);
        new
    }

    pub(super) fn vstack_fwd<E: Dimension, F: Dimension>(
        &mut self,
        lhs: &Tensor<E>,
        rhs: &Tensor<F>,
    ) {
        let (lhs_data, rhs_data) = { (&lhs.data, &rhs.data) };
        let split = if lhs_data.ndim() == 1 && rhs_data.ndim() == 1 {
            lhs_data.len()
        } else {
            lhs_data.len_of(Axis(1))
        };

        let (self_dest, other_dest) = self.data.view_mut().split_at(Axis(1), split);
        Zip::from(self_dest)
            .and_broadcast(lhs_data)
            .apply(|self_dest_el, self_data_el| *self_dest_el = *self_data_el);
        Zip::from(other_dest)
            .and_broadcast(rhs_data)
            .apply(|self_dest_el, self_data_el| *self_dest_el = *self_data_el);
    }

    pub(super) fn hstack_fwd<E: Dimension, F: Dimension>(
        &mut self,
        lhs: &Tensor<E>,
        rhs: &Tensor<F>,
    ) {
        let (lhs_data, rhs_data) = { (&lhs.data, &rhs.data) };
        let split = if lhs_data.ndim() == 1 && rhs_data.ndim() == 1 {
            lhs_data.len()
        } else {
            lhs_data.len_of(Axis(0))
        };

        let (self_dest, other_dest) = self.data.view_mut().split_at(Axis(0), split);
        Zip::from(self_dest)
            .and_broadcast(lhs_data)
            .apply(|self_dest_el, self_data_el| *self_dest_el = *self_data_el);
        Zip::from(other_dest)
            .and_broadcast(rhs_data)
            .apply(|self_dest_el, self_data_el| *self_dest_el = *self_data_el);
    }

    pub(super) fn vstack_bkwrd<E: Dimension, F: Dimension>(
        &self,
        lhs_grad: &mut Tensor<E>,
        rhs_grad: &mut Tensor<F>,
        action: &BackwardAction,
    ) {
        let (lhs_grad_data, rhs_grad_data) = { (&mut lhs_grad.data, &mut rhs_grad.data) };

        let split = if lhs_grad_data.ndim() == 1 && rhs_grad_data.ndim() == 1 {
            lhs_grad_data.len()
        } else {
            lhs_grad_data.len_of(Axis(1))
        };

        let (lhs_dest, rhs_dest) = self.data.view().split_at(Axis(1), split);
        match action {
            BackwardAction::Set => {
                Zip::from(lhs_grad_data)
                    .and_broadcast(lhs_dest)
                    .apply(|dest_el, src_el| *dest_el = *src_el);
                Zip::from(rhs_grad_data)
                    .and_broadcast(rhs_dest)
                    .apply(|dest_el, src_el| *dest_el = *src_el);
            }
            BackwardAction::Increment => {
                Zip::from(lhs_grad_data)
                    .and_broadcast(lhs_dest)
                    .apply(|dest_el, src_el| *dest_el += *src_el);
                Zip::from(rhs_grad_data)
                    .and_broadcast(rhs_dest)
                    .apply(|dest_el, src_el| *dest_el += *src_el);
            }
        }
    }

    pub(super) fn hstack_bkwrd<E: Dimension, F: Dimension>(
        &self,
        lhs_grad: &mut Tensor<E>,
        rhs_grad: &mut Tensor<F>,
        action: &BackwardAction,
    ) {
        let (lhs_grad_data, rhs_grad_data) = { (&mut lhs_grad.data, &mut rhs_grad.data) };

        let split = if lhs_grad_data.ndim() == 1 && rhs_grad_data.ndim() == 1 {
            lhs_grad_data.len()
        } else {
            lhs_grad_data.len_of(Axis(0))
        };

        let (lhs_dest, rhs_dest) = self.data.view().split_at(Axis(0), split);
        match action {
            BackwardAction::Set => {
                Zip::from(lhs_grad_data)
                    .and_broadcast(lhs_dest)
                    .apply(|dest_el, src_el| *dest_el = *src_el);
                Zip::from(rhs_grad_data)
                    .and_broadcast(rhs_dest)
                    .apply(|dest_el, src_el| *dest_el = *src_el);
            }
            BackwardAction::Increment => {
                Zip::from(lhs_grad_data)
                    .and_broadcast(lhs_dest)
                    .apply(|dest_el, src_el| *dest_el += *src_el);
                Zip::from(rhs_grad_data)
                    .and_broadcast(rhs_dest)
                    .apply(|dest_el, src_el| *dest_el += *src_el);
            }
        }
    }

    pub(super) fn dstack(&self, other: &Self) -> Tensor<DStacked<D, D>>
    where
        D: DStack<D>,
    {
        let (self_data, other_data) = { (&self.data, &other.data) };

        let mut new = Tensor {
            data: Array::<f32, DStacked<D, D>>::zeros(
                self_data.raw_dim().dstack(other_data.raw_dim()),
            ),
        };

        new.dstack_fwd(self, other);
        new
    }

    pub(super) fn dstack_fwd<E: Dimension>(&mut self, lhs: &Tensor<E>, rhs: &Tensor<E>) {
        let (lhs_data, rhs_data) = { (&lhs.data, &rhs.data) };
        let (lhs_dest, rhs_dest) = {
            let (one, the_other) = self.data.view_mut().split_at(Axis(2), 1);
            (one.remove_axis(Axis(2)), the_other.remove_axis(Axis(2)))
        };
        Zip::from(lhs_dest)
            .and_broadcast(lhs_data)
            .apply(|self_dest_el, self_data_el| *self_dest_el = *self_data_el);
        Zip::from(rhs_dest)
            .and_broadcast(rhs_data)
            .apply(|self_dest_el, self_data_el| *self_dest_el = *self_data_el);
    }

    pub(super) fn dstack_bkwrd<E: Dimension>(
        &self,
        lhs_grad: &mut Tensor<E>,
        rhs_grad: &mut Tensor<E>,
        action: &BackwardAction,
    ) {
        let (lhs_grad_data, rhs_grad_data) = { (&mut lhs_grad.data, &mut rhs_grad.data) };
        let (lhs_grad_src, rhs_grad_src) = {
            let (one, the_other) = self.data.view().split_at(Axis(2), 1);
            (one.remove_axis(Axis(2)), the_other.remove_axis(Axis(2)))
        };
        match action {
            BackwardAction::Set => {
                Zip::from(lhs_grad_data)
                    .and_broadcast(lhs_grad_src)
                    .apply(|lhs_dest_el, src_data_el| *lhs_dest_el = *src_data_el);
                Zip::from(rhs_grad_data)
                    .and_broadcast(rhs_grad_src)
                    .apply(|rhs_dest_el, src_data_el| *rhs_dest_el = *src_data_el);
            }
            BackwardAction::Increment => {
                Zip::from(lhs_grad_data)
                    .and_broadcast(lhs_grad_src)
                    .apply(|lhs_dest_el, src_data_el| *lhs_dest_el += *src_data_el);
                Zip::from(rhs_grad_data)
                    .and_broadcast(rhs_grad_src)
                    .apply(|rhs_dest_el, src_data_el| *rhs_dest_el += *src_data_el);
            }
        }
    }
}

#[cfg(test)]
mod tests;
