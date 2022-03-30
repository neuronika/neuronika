use ndarray::{Array, ArrayD, Axis, DimMax, Dimension, IntoDimension, Ix0, Ix1, Ix2, Zip};
use std::{
    cell::{Ref, RefCell, RefMut},
    rc::Rc,
};

pub(crate) use binary::*;
// pub use binary::{
//     Constant, Convolve, ConvolveWithGroups, PaddingMode, Reflective, Replicative, Zero,
// };
pub(crate) use nary::*;
pub(crate) use unary::*;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Nodes' Modules ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mod binary;
mod nary;
mod unary;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Type Aliases ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub(crate) type Broadcasted<Lhs, Rhs> = <Lhs as DimMax<Rhs>>::Output;
pub(crate) type BroadTensor<Lhs, Rhs> = Tensor<Broadcasted<Lhs, Rhs>>;
pub(crate) type DynTensor = ArrayD<f32>;
pub(crate) type Tensor<D> = Array<f32, D>;
pub(crate) type Shared<T> = Rc<RefCell<T>>;

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Computational Nodes` Traits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Forward-propagation behavior.
///
/// This trait is implemented by all the internal forward components of `Var` and `VarDiff`.
///
/// The main method it provides is the `.forward()` method that is used to propagate computations
/// from the leaf variables to the graph's root.
///
/// The other two methods this trait provides, namely `.was_computed()` and `.reset_computation()`,
/// are used to perform caching during the forward pass.
///
/// Caching is critical to avoid recomputing paths and to achieve good performance when a
/// computational graph has more than one root, like the one, for instance, of a recurrent neural
/// network.
pub trait Forward {
    /// Propagates the computations forwards.
    ///
    /// It also defines the logic for the computation of the node.
    fn forward(&self);

    /// Returns `true` if the node was computed, `false` otherwise.
    fn was_computed(&self) -> bool;

    /// Reset the node's flag, making it computable again.
    fn reset_computation(&self);
}

/// Back-propagation behavior.
///
/// This trait is implemented by all the internal backward components of `VarDiff`.
///
/// The main method it provides is the `.backward()` method that is used to back-propagate gradients
/// from the root variables to the graph's leaves.
///
/// The other two methods, namely `.no_grad()` and `.with_grad()` are used to shut down
/// gradients' computation.
pub trait Backward {
    /// Propagates the computations backwards.
    ///
    /// It also defines the logic for the back-propagation of the node.
    fn backward(&self);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DotDim ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Utility trait useful to compute the dimensionality of algebraic operations' results.
pub(crate) trait DotDim<Rhs>
where
    Self: Dimension,
    Rhs: Dimension,
{
    /// Dimension of the resulting variable.
    type Output: Dimension;

    /// Does the actual computation of the shape.
    fn shape(lhs: Self, rhs: Rhs) -> <Self as DotDim<Rhs>>::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DotDim implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

impl DotDim<Ix1> for Ix1 {
    type Output = Ix0;

    fn shape(_: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        ().into_dimension()
    }
}

impl DotDim<Ix2> for Ix1 {
    type Output = Ix1;

    fn shape(_: Self, rhs: Ix2) -> <Self as DotDim<Ix2>>::Output {
        let mut result = Ix1::zeros(1);
        result[0] = rhs.last_elem();
        result
    }
}

impl DotDim<Ix1> for Ix2 {
    type Output = Ix1;

    fn shape(lhs: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        let mut result = Ix1::zeros(1);
        result[0] = lhs[0];
        result
    }
}

impl DotDim<Ix2> for Ix2 {
    type Output = Ix2;

    fn shape(lhs: Self, rhs: Ix2) -> <Self as DotDim<Ix2>>::Output {
        let mut result = Ix2::zeros(2);
        result[0] = lhs[0];
        result[1] = rhs[1];
        result
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gradient Accumulation Utilities  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Shrinks `array` summing in-place along `axis`.
///
/// # Arguments
///
/// * `array` - array to reduce.
///
/// * `axis` - axis to sum along to.
fn sum_axis_inplace(array: &mut DynTensor, axis: Axis) {
    let (first, rest) = array.view_mut().split_at(axis, 1);
    Zip::from(first.remove_axis(axis))
        .and(rest.lanes(axis))
        .for_each(|dst, src| *dst += src.sum());
    array.index_axis_inplace(axis, 0);
}

/// Reduces `src` to the desired `dim` dimension, reverting the broadcasting.
///
/// # Arguments
///
/// * `dim` - desired dimension for the source tensor.
///
/// * `src` - tensor to reduce.
pub fn reduce<D: Dimension, E: Dimension>(dim: D, src: &Tensor<E>) -> Tensor<D> {
    let mut src = src.clone().into_dyn();

    while src.ndim() > dim.ndim() {
        sum_axis_inplace(&mut src, Axis(0));
    }

    for (axis, size) in dim.slice().iter().enumerate() {
        if *size == 1 {
            sum_axis_inplace(&mut src, Axis(axis));
            src.insert_axis_inplace(Axis(axis));
        }
    }

    debug_assert_eq!(
        src.raw_dim(),
        dim.into_dyn(),
        "Dimension mismatch in gradient reduction."
    );

    if src.is_standard_layout() {
        src.into_dimensionality::<D>().unwrap()
    } else {
        src.clone().into_dimensionality::<D>().unwrap()
    }
}

pub trait SwitchTensor {
    fn deallocate(&self);

    fn allocate(&self);
}

pub struct OptionalTensor<D>
where
    D: Dimension,
{
    tensor: RefCell<Option<Tensor<D>>>,
    shape: D,
}

impl<D> OptionalTensor<D>
where
    D: Dimension,
{
    pub fn zeros(shape: D) -> Self {
        Self {
            tensor: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
        }
    }

    pub fn from_ndarray(tensor: Tensor<D>) -> Self {
        let shape = tensor.raw_dim();

        Self {
            tensor: RefCell::new(Some(tensor)),
            shape,
        }
    }

    // Queste due funzioni potrebbero essere senza il controllo se venisse implementato
    // un cambio di tipo a seconda che contengano un `Some(_)` o un `None`.
    // Sarebbe molto piu` pulito, ma e` da valutare se ne valga la pena
    pub fn content(&self) -> Ref<Tensor<D>> {
        debug_assert!(self.tensor.borrow().is_some(), "Trying to get a de-allocated gradient. Switch on the gradients first by using with_grad().");

        Ref::map(self.tensor.borrow(), |b| b.as_ref().unwrap())
    }

    // Queste due funzioni potrebbero essere senza il controllo se venisse implementato
    // un cambio di tipo a seconda che contengano un `Some(_)` o un `None`.
    // Sarebbe molto piu` pulito, ma e` da valutare se ne valga la pena
    pub fn content_mut(&self) -> RefMut<Tensor<D>> {
        debug_assert!(self.tensor.borrow().is_some(), "Trying to get a de-allocated gradient. Switch on the gradients first by using with_grad().");

        RefMut::map(self.tensor.borrow_mut(), |b| b.as_mut().unwrap())
    }

    pub fn shape(&self) -> D {
        self.shape.clone()
    }
}

impl<D> SwitchTensor for OptionalTensor<D>
where
    D: Dimension,
{
    fn deallocate(&self) {
        *self.tensor.borrow_mut() = None;
    }

    fn allocate(&self) {
        let mut tensor = self.tensor.borrow_mut();
        if tensor.is_some() {
            return;
        }

        *tensor = Some(Tensor::zeros(self.shape()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tensor Utilities ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Creates an empty tensor whose shape is the result of broadcasting between those of `left` and
/// `right`.
///
/// # Arguments
///
/// * `left` - left operand in the binary operations that admits broadcasting.
///
/// * `right` - right operand in the binary operations that admits broadcasting.
pub(crate) fn cobroadcasted_zeros<Lhs, Rhs>(
    left: &Tensor<Lhs>,
    right: &Tensor<Rhs>,
) -> BroadTensor<Lhs, Rhs>
where
    Lhs: Dimension + DimMax<Rhs>,
    Rhs: Dimension,
{
    let (bigger, smaller) = if left.ndim() >= right.ndim() {
        (left.shape(), right.shape())
    } else {
        (right.shape(), left.shape())
    };
    let mut out = <Lhs as DimMax<Rhs>>::Output::zeros(bigger.len());
    out.slice_mut()
        .iter_mut()
        .zip(bigger.iter())
        .for_each(|(l, r)| *l = *r);
    let k = bigger.len() - smaller.len();
    out.slice_mut()
        .iter_mut()
        .skip(k)
        .zip(smaller.iter())
        .for_each(|(l, r)| {
            if *l != *r {
                if *l == 1 {
                    *l = *r
                } else if *r != 1 {
                    panic!("The two tensors have incompatible shape.")
                }
            }
        });
    Tensor::zeros(out)
}

/// Returns a `Ref` to `tensor`. This function is used to access gradients.
///
/// # Arguments
///
/// `tensor` - gradient.
///
/// # Panics
///
/// If the gradient has been de-allocated.
pub(crate) fn expect_tensor<D>(tensor: &Rc<RefCell<Option<Tensor<D>>>>) -> Ref<Tensor<D>>
where
    D: Dimension,
{
    Ref::map(tensor.borrow(), |b| {
        b.as_ref().expect(
            "Trying to get a de-allocated gradient. Switch on the gradients first by using with_grad().",
        )
    })
}

/// Returns a `RefMut` to `tensor`. This function is used to access gradients.
///
/// # Arguments
///
/// `tensor` - gradient.
///
/// # Panics
///
/// If the gradient has been de-allocated.
pub(crate) fn expect_tensor_mut<D>(tensor: &Rc<RefCell<Option<Tensor<D>>>>) -> RefMut<Tensor<D>>
where
    D: Dimension,
{
    RefMut::map(tensor.borrow_mut(), |b| {
        b.as_mut().expect(
            "Trying to get a de-allocated gradient. Switch on the gradients first by using with_grad().",
        )
    })
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Testing Utilities ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#[cfg(test)]
const F16_EPSILON: f32 = 9.77e-04;

#[cfg(test)]
/// Checks element-wise whether `array` is within `F16_EPSILON` of `target`.
///
/// # Arguments
///
/// * `array` - array to check.
///
/// * `target` - target to check against.
///
/// # Panics
///
/// If `array` is not within `F16_EPSILON` of `target`.
fn assert_almost_equals<D: Dimension>(array: &Tensor<D>, target: &Tensor<D>) {
    assert!(
        Zip::from(array).and(target).all(|l, r| {
            (*l == 0. && *r == 0.)
                || (!l.is_finite() && !r.is_finite())
                || ((1. - r / l).abs() <= F16_EPSILON)
        }),
        "\nLeft:\n{}\nRight:\n{}",
        array,
        target
    );
}

#[cfg(test)]
/// Creates a new tensor with shape `shape` and elements `elements`.
///
/// # Arguments
///
/// * `shape` - shape.
///
/// * `elements` - elements.
fn new_tensor<D, Sh>(shape: Sh, elements: Vec<f32>) -> Rc<RefCell<Tensor<D>>>
where
    D: Dimension + 'static,
    Sh: Into<ndarray::StrideShape<D>>,
{
    Rc::new(RefCell::new(
        Tensor::from_shape_vec(shape, elements).unwrap(),
    ))
}

fn new_opt_tensor<D, Sh>(shape: Sh, elements: Vec<f32>) -> Rc<RefCell<Option<Tensor<D>>>>
where
    D: Dimension + 'static,
    Sh: Into<ndarray::StrideShape<D>>,
{
    Rc::new(RefCell::new(Some(
        Tensor::from_shape_vec(shape, elements).unwrap(),
    )))
}
