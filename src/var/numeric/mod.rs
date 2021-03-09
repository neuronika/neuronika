use ndarray::linalg::{general_mat_mul, general_mat_vec_mul};
use ndarray::{
    arr1, concatenate, Array, Array2, ArrayView, ArrayView1, Axis, Dimension, Ix0, Ix1, Ix2, Ix3,
    Ix4, Ix5, RemoveAxis, Zip,
};
use std::cell::{Cell, RefMut};
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
    type Output: Dimension;

    fn max(self, rhs: R) -> Maximum<Self, R>;
}

/// The [Max] of two [Dimension] items.
///
/// [Max]: crate::Max
/// [Dimension]: ndarray::Dimension
pub type Maximum<L, R> = <L as Max<R>>::Output;

/// Automatically implements all the trivial cases for the [Max] relation.
///
/// [Max]: Prova::Max
macro_rules! impl_unary_max {
    ($($dim: ty),+ $(,)?) => {
        $(
            impl Max<$dim> for $dim {
                type Output = $dim;

                fn max(self, _: $dim) -> $dim {
                    self
                }
            }
        )*
    };
}

/// Automatically implements all the cases for the [Max] relation accordingly.
///
/// [Max]: Prova::Max
macro_rules! impl_binary_max {
    ($small: ty, $big: ty) => {
        impl Max<$small> for $big {
            type Output = $big;

            fn max(self, _: $small) -> $big { self }
        }

        impl Max<$big> for $small {
            type Output = $big;

            fn max(self, rhs: $big) -> $big { rhs }
        }
    };

    ($(($small: ty, $big: ty)),+ $(,)?) => {
        $(impl_binary_max!{$small, $big })*
    };
}

impl_unary_max!(Ix0, Ix1, Ix2, Ix3, Ix4, Ix5);

impl_binary_max!(
    (Ix0, Ix3),
    (Ix0, Ix2),
    (Ix0, Ix1),
    (Ix1, Ix3),
    (Ix1, Ix2),
    (Ix2, Ix3)
);

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
        Self::Output { data: -self.data }
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
        pub(super) fn $fun(&mut self, down_grad: &Self, t: &Self, action: BackwardAction) {
            match action {
                Set => {
                    Zip::from(&mut self.data)
                        .and(&down_grad.data)
                        .and(&t.data)
                        .apply($closure_set);
                }
                Increment => {
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
            action: BackwardAction,
            $param: $param_t,
        ) {
            match action {
                Set => {
                    Zip::from(&mut self.data)
                        .and(&down_grad.data)
                        .and(&t.data)
                        .apply($closure_set);
                }
                Increment => {
                    Zip::from(&mut self.data)
                        .and(&down_grad.data)
                        .and(&t.data)
                        .apply($closure_incr);
                }
            }
        }
    };
}

// ============================================ Accumulation Utilities ============================================

macro_rules! impl_acc_utils {
    ($singleton:ident, $geq:ident, $op:tt) => {
        // Questa funzione si occupa del caso in cui dest
        // sia un singoletto, ovvero : [el], [[el]], [[[el]]] ecc...
        //
        // Restituisce true nel caso in cui lo sia in modo tale da segnalare
        // che l'operazione si è conlcusa con successo.
        pub fn $singleton<T: Dimension, D: Dimension>(
            dest: &mut Array<f32, T>,
            src: &Array<f32, D>,
            a: f32
        ) -> bool {
            // Se dentro dest c'è solo un elemento...
            if dest.len() == 1 {
                let reduced_src = src.sum();
                Zip::from(dest).apply(|dest_el| *dest_el $op reduced_src * a);
                true
            } else {
                false
            }
        }

        // Il risultato ha sempre la forma di `dest`.
        // Se hanno la dimensione diversa, comanda sempre `dest`.
        // Se la dimensione e` uguale, ma la shape cambia,
        // ed  la lunghezza di una di queste sia `

        // Questa funzione di occupa del caso in cui la dimensione di dest
        // sia maggiore o uguale della dimensione di src, anche lei restituisce
        // un booleano per la stessa ragione di prima.
        pub fn $geq<T: Dimension, D: Dimension>(
            dest: &mut Array<f32, T>,
            src: ArrayView<f32, D>,
            a: f32
        ) -> bool {
            // Nel caso in cui la dimensione di dest e di src coincidano...
            if dest.ndim() == src.ndim() {
                let mut axis_of_len_one = false;
                // Questo codice gestisce i casi in cui una o più delle assi di
                // dest abbia lunghezza 1, in tal caso bisogna sommare gli elementi
                // della corrispondente asse di src dentro l'unico presente
                // nell'asse di dest
                for i in 0..dest.ndim() {
                    let size = dest.len_of(Axis(i));
                    if size == 1_usize {
                        axis_of_len_one = true;
                        dest.lanes_mut(Axis(i))
                            .into_iter()
                            .zip(src.lanes(Axis(i)))
                            .for_each(|(dest_lane, src_lane)| {
                                Zip::from(dest_lane).apply(|dest_view_el| *dest_view_el $op src_lane.sum() * a);
                            });
                    }
                }
                // Se nessuna delle assi aveva lunghezza uno...
                if !axis_of_len_one {
                    Zip::from(dest)
                        .and_broadcast(src)
                        .apply(|dest_el, src_el| *dest_el $op *src_el * a);
                }
                true
            } else if dest.ndim() > src.ndim() {
                // Se la dimensione di dest è maggiore di quella di src ndarray se la cava da solo.
                Zip::from(dest)
                    .and_broadcast(src)
                    .apply(|dest_el, src_el| *dest_el $op *src_el * a);
                true
            } else {
                false
            }
        }
    };
}

pub fn singleton_div_assign_square<T: Dimension, D: Dimension>(
    dest: &mut Array<f32, T>,
    src: &Array<f32, D>,
) -> bool {
    // Se dentro dest c'è solo un elemento...
    if dest.len() == 1 {
        let reduced_src = src.map(|el| el.powi(2)).sum();
        Zip::from(dest).apply(|dest_el| *dest_el /= reduced_src);
        true
    } else {
        false
    }
}

pub fn geq_div_assign_square<T: Dimension, D: Dimension>(
    dest: &mut Array<f32, T>,
    src: ArrayView<f32, D>,
) -> bool {
    // Nel caso in cui la dimensione di dest e di src coincidano...
    if dest.ndim() == src.ndim() {
        let mut axis_of_len_one = false;
        // Questo codice gestisce i casi in cui una o più delle assi di
        // dest abbia lunghezza 1, in tal caso bisogna sommare gli elementi
        // della corrispondente asse di src dentro l'unico presente
        // nell'asse di dest
        for i in 0..dest.ndim() {
            let size = dest.len_of(Axis(i));
            if size == 1_usize {
                axis_of_len_one = true;
                dest.lanes_mut(Axis(i))
                    .into_iter()
                    .zip(src.lanes(Axis(i)))
                    .for_each(|(dest_lane, src_lane)| {
                        Zip::from(dest_lane).apply(|dest_view_el| {
                            *dest_view_el /= src_lane.map(|el| el.powi(2)).sum()
                        });
                    });
            }
        }
        // Se nessuna delle assi aveva lunghezza uno...
        if !axis_of_len_one {
            Zip::from(dest)
                .and_broadcast(src)
                .apply(|dest_el, src_el| *dest_el /= src_el.powi(2));
        }
        true
    } else if dest.ndim() > src.ndim() {
        // Se la dimensione di dest è maggiore di quella di src ndarray se la cava da solo.
        Zip::from(dest)
            .and_broadcast(src)
            .apply(|dest_el, src_el| *dest_el /= src_el.powi(2));
        true
    } else {
        false
    }
}

impl_acc_utils!(singleton_assign, geq_assign, =);
impl_acc_utils!(singleton_add_assign, geq_add_assign, +=);
impl_acc_utils!(singleton_sub_assign, geq_sub_assign, -=);

macro_rules! impl_tensor_one_assig {
    ($fun:ident, $singleton:ident, $geq:ident) => {
        fn $fun<D: Dimension + RemoveAxis>(&mut self, src: &Tensor<D>, a: f32) {
            // Se dest non è un singoletto controllo le dimensioni di dest e src
            if !$singleton(&mut self.data, &src.data, a) {
                // Se la dimensione di dest non è maggiore o uguale...
                if !$geq(&mut self.data, src.data.view(), a) {
                    //... effettuo un'opportuna riduzione
                    for lane in src.data.lanes(Axis(0)) {
                        $geq(&mut self.data, lane, a);
                    }
                }
            }
        }
    };
}

macro_rules! impl_tensor_two_assign {
    ($fun:ident, $singleton:ident, $geq:ident) => {
        fn $fun<D: Dimension + RemoveAxis>(&mut self, src: &Tensor<D>, a: f32) {
            if !$singleton(&mut self.data, &src.data, a) {
                if !$geq(&mut self.data, src.data.view(), a) {
                    for view in src.data.axis_iter(Axis(0)) {
                        // Questa è l'opportuna riduzione
                        $geq(&mut self.data, view, a);
                    }
                }
            }
        }
    };
}

macro_rules! impl_tensor_three_assign {
    ($fun:ident, $singleton:ident, $geq:ident) => {
        fn $fun<D: Dimension + RemoveAxis>(&mut self, src: &Tensor<D>, a: f32) {
            if !$singleton(&mut self.data, &src.data, a) {
                $geq(&mut self.data, src.data.view(), a);
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
struct Tensor<D>
where
    D: Dimension,
{
    /// Content of the tensor
    data: Array<f32, D>,
}

impl<D> Display for Tensor<D>
where
    D: Dimension,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.data)
    }
}

// Methods specific to the one dimensional Tensor.
impl Tensor<Ix1> {
    impl_tensor_one_assig!(assign, singleton_assign, geq_assign);
    impl_tensor_one_assig!(add_assign, singleton_add_assign, geq_add_assign);
    impl_tensor_one_assig!(sub_assign, singleton_sub_assign, geq_sub_assign);
    fn div_assign_square<D: Dimension + RemoveAxis>(&mut self, src: &Tensor<D>) {
        // Se dest non è un singoletto controllo le dimensioni di dest e src
        if !singleton_div_assign_square(&mut self.data, &src.data) {
            // Se la dimensione di dest non è maggiore o uguale...
            if !geq_div_assign_square(&mut self.data, src.data.view()) {
                //... effettuo un'opportuna riduzione
                for lane in src.data.lanes(Axis(0)) {
                    geq_div_assign_square(&mut self.data, lane);
                }
            }
        }
    }
}

// Methods specific to the two dimensional Tensor.
impl Tensor<Ix2> {
    impl_tensor_two_assign!(assign, singleton_assign, geq_assign);
    impl_tensor_two_assign!(add_assign, singleton_add_assign, geq_add_assign);
    impl_tensor_two_assign!(sub_assign, singleton_sub_assign, geq_sub_assign);
    fn div_assign_square<D: Dimension + RemoveAxis>(&mut self, src: &Tensor<D>) {
        // Se dest non è un singoletto controllo le dimensioni di dest e src
        if !singleton_div_assign_square(&mut self.data, &src.data) {
            // Se la dimensione di dest non è maggiore o uguale...
            if !geq_div_assign_square(&mut self.data, src.data.view()) {
                //... effettuo un'opportuna riduzione
                for view in src.data.axis_iter(Axis(0)) {
                    geq_div_assign_square(&mut self.data, view);
                }
            }
        }
    }

    fn mat_mul(
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

    fn mat_vec_mul(
        &self,
        rhs: Tensor<Ix1>,
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

// Methods specific to the three dimensional Tensor.
impl Tensor<Ix3> {
    // Per Tensor 3d abbiamo solo il caso in cui dest dim >= src dim
    // perchè di più di tre dimensioni non ce ne facciamo di nulla.
    impl_tensor_three_assign!(assign, singleton_assign, geq_assign);
    impl_tensor_three_assign!(add_assign, singleton_add_assign, geq_add_assign);
    impl_tensor_three_assign!(sub_assign, singleton_sub_assign, geq_sub_assign);
    fn div_assign_square<D: Dimension + RemoveAxis>(&mut self, src: &Tensor<D>) {
        // Se dest non è un singoletto controllo le dimensioni di dest e src
        if !singleton_div_assign_square(&mut self.data, &src.data) {
            geq_div_assign_square(&mut self.data, src.data.view());
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

    // Concatenates Tensors of the same dimensionality.
    pub fn cat(tensors: &[&Self], axis: usize) -> Self {
        let data: Vec<ArrayView<f32, D>> = tensors.iter().map(|t| t.data.view()).collect();
        Self {
            data: concatenate(Axis(axis), &data).ok().unwrap(),
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
        let new = self.zeros();
        Zip::from(self.data.lanes(Axis(axis)))
            .and(new.data.lanes_mut(Axis(axis)))
            .apply(|lane_self, mut lane_new| {
                let max = lane_self.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = &lane_self.map(|el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_new)
                    .and(num)
                    .apply(|lane_new_el, num_el| *lane_new_el = *num_el / den);
            });
        new
    }

    pub fn softmax_bkwrd(
        &mut self,
        input_grad: &Self,
        data: &Self,
        jacobian: &mut Array2<f32>,
        action: BackwardAction,
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
            Set => 0.0,
            Increment => 1.0,
        };
        Zip::from(self.data.lanes_mut(Axis(axis)))
            .and(data.data.lanes(Axis(axis)))
            .and(input_grad.data.lanes(Axis(axis)))
            .apply(|mut d_g_col, data_col, grad_col| {
                fill_jacobian(jacobian, &data_col);
                general_mat_vec_mul(1.0, &jacobian, &grad_col, beta, &mut d_g_col);
            });
    }
}
