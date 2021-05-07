pub mod model_selection;
pub mod nn;
pub mod optim;
mod variable;
pub use ndarray::{self, Array, Array2, Dimension, Ix1, Ix2, ShapeBuilder};
pub use ndarray_rand::rand_distr::Uniform;
pub use ndarray_rand::RandomExt;
pub use variable::{
    node::{Input, InputBackward},
    Cat, MatMatMul, MatVecMul, Stack, Var, VarDiff, VecMatMul, VecVecMul,
};

/// Creates an Input node with one, two or three dimensions.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::tensor!([1., 2., 3., 4.]);
///
/// let t2 = neuronika::tensor!([[1., 2.],
///                             [3., 4.]]);
///
/// let t3 = neuronika::tensor!([[[1., 2.], [3., 4.]],
///                             [[5., 6.], [7., 8.]]]);
///
/// assert_eq!(t1.data().shape(), &[4]);
/// assert_eq!(t2.data().shape(), &[2, 2]);
/// assert_eq!(t3.data().shape(), &[2, 2, 2]);
/// ```
#[macro_export]
macro_rules! tensor {
    ([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]) => {{
        $crate::Input::new($crate::ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*]))
    }};
    ([$([$($x:expr),* $(,)*]),+ $(,)*]) => {{
        $crate::Input::new($crate::ndarray::Array2::from(vec![$([$($x,)*],)*]))
    }};
    ([$($x:expr),* $(,)*]) => {{
        $crate::Input::new($crate::ndarray::Array1::from(vec![$($x,)*]))
    }};
}

/// Creates an Input node node with zeroed data.
///
/// The shape is of type `ndarray::ShapeBuilder`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::zeros(1);
///
/// let t2 = neuronika::zeros((1, 5));
///
/// let t3 = neuronika::zeros([1, 2, 3]);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
pub fn zeros<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Var<Input<D>> {
    Input::new(Array::from_elem(shape, 0.0))
}

/// Creates an Input node with data filled with ones.
///
/// The shape is of type `ndarray::ShapeBuilder`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::ones(1);
///
/// let t2 = neuronika::ones((1, 5));
///
/// let t3 = neuronika::ones([1, 2, 3]);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
pub fn ones<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Var<Input<D>> {
    Input::new(Array::from_elem(shape, 1.0))
}

/// Creates an Input node with data filled with the `f32` value `el`.
///
/// The shape is of type `ndarray::ShapeBuilder`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::full(1, 5.); // Filled with 5.0
///
/// let t2 = neuronika::full((1, 5), 6.); // Filled with 6.0
///
/// let t3 = neuronika::full([1, 2, 3], 8.); // Filled with 8.0
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
pub fn full<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh, elem: f32) -> Var<Input<D>> {
    Input::new(Array::from_elem(shape, elem))
}

/// Returns an Input node with values sampled from a uniform distribution
/// on the interval **[0,1)**.
///
/// The shape is of type `ndarray::ShapeBuilder`.
pub fn rand<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Var<Input<D>> {
    Input::new(Array::random(shape, Uniform::new(0., 1.)))
}

/// Creates an identity matrix of size `n` (a square 2D matrix).
///
/// # Panics
///
/// If `n * n` would overflow `isize`.
pub fn eye(n: usize) -> Var<Input<Ix2>> {
    Input::new(Array2::eye(n))
}

/// Create a one-dimensional array with `n` evenly spaced elements from `start` to `end`
/// (exclusive). The elements must be `f32`.
///
///
/// # Panics
///
/// If the length is greater than `isize::MAX`.
///
/// # Examples
///
/// ```rust
/// use neuronika;
/// use ndarray::arr1;
///
/// let tensor = neuronika::linspace(0., 1., 5);
/// assert!(*tensor.data() == arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
/// ```
pub fn linspace(start: f32, end: f32, n: usize) -> Var<Input<Ix1>> {
    Input::new(Array::linspace(start, end, n))
}

/// Create a one-dimensional array with `n` logarithmically spaced
/// elements, with the starting value being `base.powf(start)` and the
/// final one being `base.powf(end)`. Elements must be `f32`.
///
/// If `base` is negative, all values will be negative.
///
/// # Panics
//
/// If `n` is greater than `isize::MAX` or if converting `n - 1`
/// to type `f32` fails.
pub fn logspace(base: f32, start: f32, end: f32, n: usize) -> Var<Input<Ix1>> {
    Input::new(Array::logspace(base, start, end, n))
}

/// Create a one-dimensional array with `n` geometrically spaced elements
/// from `start` to `end` (inclusive). Elements must be `f32`.
///
/// Returns `None` if `start` and `end` have different signs or if either
/// one is zero. Conceptually, this means that in order to obtain a `Some`
/// result, `end / start` must be positive.
///
/// # Panics
/// If `n` is greater than `isize::MAX` or if converting `n - 1`
/// to type `f32` fails.
pub fn geomspace(start: f32, end: f32, n: usize) -> Option<Var<Input<Ix1>>> {
    Array::geomspace(start, end, n).map(Input::new)
}

/// Create a one-dimensional array with elements from `start` to `end`
/// (exclusive), incrementing by `step`. Elemetns must be a `f32`.
///
/// # Panics
///
/// If the length is greater than `isize::MAX`.
///
/// ```rust
/// use neuronika;
/// use ndarray::arr1;
///
/// let tensor = neuronika::range(0., 5., 1.);
/// assert!(*tensor.data() == arr1(&[0., 1., 2., 3., 4.]))
/// ```
pub fn range(start: f32, end: f32, step: f32) -> Var<Input<Ix1>> {
    Input::new(Array::range(start, end, step))
}
