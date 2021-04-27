mod graph;
pub mod nn;
pub use graph::{
    node::{Input, InputBackward},
    Cat, MatMatMul, MatVecMul, Stack, VecMatMul, VecVecMul,
};
pub use ndarray;
pub use ndarray_rand::rand_distr::Uniform;
pub use ndarray_rand::RandomExt;

/// Creates an Input node with one, two or three dimensions.
///
/// The return type and its differentiability depend on the `bool` passed after the data.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::tensor!([1., 2., 3., 4.], false);
///
/// let t2 = neuronika::tensor!([[1., 2.],
///                             [3., 4.]], true);
///
/// let t3 = neuronika::tensor!([[[1., 2.], [3., 4.]],
///                             [[5., 6.], [7., 8.]]], true);
///
/// assert_eq!(t1.data().shape(), &[4]);
/// assert_eq!(t2.data().shape(), &[2, 2]);
/// assert_eq!(t3.data().shape(), &[2, 2, 2]);
/// ```
#[macro_export]
macro_rules! tensor {
    ([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*], true) => {{
        $crate::Input::new($crate::ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*])).requires_grad()
    }};
    ([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*], false) => {{
        $crate::Input::new($crate::ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*]))
    }};
    ([$([$($x:expr),* $(,)*]),+ $(,)*], true) => {{
        $crate::Input::new($crate::ndarray::Array2::from(vec![$([$($x,)*],)*])).requires_grad()
    }};
    ([$([$($x:expr),* $(,)*]),+ $(,)*], false) => {{
        $crate::Input::new($crate::ndarray::Array2::from(vec![$([$($x,)*],)*]))
    }};
    ([$($x:expr),* $(,)*], true) => {{
        $crate::Input::new($crate::ndarray::Array1::from(vec![$($x,)*])).requires_grad()
    }};

    ([$($x:expr),* $(,)*], false) => {{
        $crate::Input::new($crate::ndarray::Array1::from(vec![$($x,)*]))
    }};
}

/// Creates an Input node node with zeroed data.
///
/// The return type and its differentiability depend on the `bool` passed after the shape.
///
/// The shape is of type `ndarray::ShapeBuilder`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::zeros!(1, true);
///
/// let t2 = neuronika::zeros!((1, 5), true);
///
/// let t3 = neuronika::zeros!([1, 2, 3], false);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
#[macro_export]
macro_rules! zeros {
    ($sh:expr, true) => {{
        $crate::Input::new($crate::ndarray::Array::from_elem($sh, 0.0)).requires_grad()
    }};
    ($sh:expr, false) => {{
        $crate::Input::new($crate::ndarray::Array::from_elem($sh, 0.0))
    }};
}

/// Creates an Input node with data filled with ones.
///
/// The return type and its differentiability depend on the `bool` passed after the shape.
///
/// The shape is of type `ndarray::ShapeBuilder`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::ones!(1, true);
///
/// let t2 = neuronika::ones!((1, 5), true);
///
/// let t3 = neuronika::ones!([1, 2, 3], false);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
#[macro_export]
macro_rules! ones {
    ($sh:expr, true) => {{
        $crate::Input::new($crate::ndarray::Array::from_elem($sh, 1.0)).requires_grad()
    }};
    ($sh:expr, false) => {{
        $crate::Input::new($crate::ndarray::Array::from_elem($sh, 1.0))
    }};
}

/// Creates an Input node with data filled with the `f32` value `el`.
///
/// The return type and its differentiability depend on the `bool` passed after the shape.
///
/// The shape is of type `ndarray::ShapeBuilder`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::full!(1, 5., true); // Filled with 5.0
///
/// let t2 = neuronika::full!((1, 5), 6., true); // Filled with 6.0
///
/// let t3 = neuronika::full!([1, 2, 3], 8., false); // Filled with 8.0
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
#[macro_export]
macro_rules! full {
    ($sh:expr, $el:expr, true) => {{
        $crate::Input::new($crate::ndarray::Array::from_elem($sh, $el)).requires_grad()
    }};
    ($sh:expr, $el:expr, false) => {{
        $crate::Input::new($crate::ndarray::Array::from_elem($sh, $el))
    }};
}

/// Returns either an Input node with values
/// sampled from a uniform distribution on the interval **[0,1)**.
///
/// The shape is of type `ndarray::ShapeBuilder`.
#[macro_export]
macro_rules! rand {
    ($sh:expr, true) => {{
        use $crate::RandomExt;
        $crate::Input::new($crate::ndarray::Array::random(
            $sh,
            $crate::Uniform::new(0., 1.),
        ))
        .requires_grad()
    }};
    ($sh:expr, false) => {{
        use $crate::RandomExt;
        $crate::Input::new($crate::ndarray::Array::random(
            $sh,
            $crate::Uniform::new(0., 1.),
        ))
    }};
}

/// Creates an identity matrix of size `n` (a square 2D matrix).
///
/// If `true` is passed after the size then the node will be differentiable.
///
/// # Panics
///
/// If `n * n` would overflow `isize`.
#[macro_export]
macro_rules! eye {
    ($sh:expr, true) => {{
        $crate::Input::new($crate::ndarray::Array2::eye($sh)).requires_grad()
    }};
    ($sh:expr, false) => {{
        $crate::Input::new($crate::ndarray::Array2::eye($sh))
    }};
}

/// Create a one-dimensional array with `n` evenly spaced elements from `start` to `end`
/// (exclusive). The elements must be `f32`.
///
/// If `true` is passed the node will be differentiable.
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
/// let tensor = neuronika::linspace!(0., 1., 5, true);
/// assert!(*tensor.data() == arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
/// ```
#[macro_export]
macro_rules! linspace {
    ($start:expr, $end:expr, $n:expr, true) => {{
        $crate::Input::new($crate::ndarray::Array::linspace($start, $end, $n)).requires_grad()
    }};
    ($start:expr, $end:expr, $n:expr, false) => {{
        $crate::Input::new($crate::ndarray::Array::linspace($start, $end, $n))
    }};
}

/// Create a one-dimensional array with `n` logarithmically spaced
/// elements, with the starting value being `base.powf(start)` and the
/// final one being `base.powf(end)`. Elements must be `f32`.
///
/// If `true` is passed the `grad` field the node will be differentiable.
///
/// If `base` is negative, all values will be negative.
///
/// # Panics
//
/// If `n` is greater than `isize::MAX` or if converting `n - 1`
/// to type `f32` fails.
#[macro_export]
macro_rules! logspace {
    ($base:expr, $start:expr, $end:expr, $n:expr, true) => {{
        $crate::Input::new($crate::ndarray::Array::logspace($base, $start, $end, $n))
            .requires_grad()
    }};
    ($base:expr, $start:expr, $end:expr, $n:expr, false) => {{
        $crate::Input::new($crate::ndarray::Array::logspace($base, $start, $end, $n))
    }};
}

/// Create a one-dimensional array with `n` geometrically spaced elements
/// from `start` to `end` (inclusive). Elements must be `f32`.
///
/// If `true` is passed after the size the node will be differentiable.
///
/// Returns `None` if `start` and `end` have different signs or if either
/// one is zero. Conceptually, this means that in order to obtain a `Some`
/// result, `end / start` must be positive.
///
/// # Panics
/// If `n` is greater than `isize::MAX` or if converting `n - 1`
/// to type `f32` fails.
#[macro_export]
macro_rules! geomspace {
    ($start:expr, $end:expr, $n:expr, true) => {{
        match $crate::ndarray::Array::geomspace($start, $end, $n) {
            None => None,
            Some(array) => Some($crate::Input::new(array).requires_grad()),
        }
    }};
    ($start:expr, $end:expr, $n:expr, false) => {{
        match $crate::ndarray::Array::geomspace($start, $end, $n) {
            None => None,
            Some(array) => Some($crate::Input::new(array)),
        }
    }};
}

/// Create a one-dimensional array with elements from `start` to `end`
/// (exclusive), incrementing by `step`. Elemetns must be a `f32`.
///
/// If `true` is passed after the size the node will be differentiable.
///
/// # Panics
///
/// If the length is greater than `isize::MAX`.
///
/// ```rust
/// use neuronika;
/// use ndarray::arr1;
///
/// let tensor = neuronika::range!(0., 5., 1., true);
/// assert!(*tensor.data() == arr1(&[0., 1., 2., 3., 4.]))
/// ```
#[macro_export]
macro_rules! range {
    ($start:expr, $end:expr, $step:expr, true) => {{
        $crate::Input::new($crate::ndarray::Array::range($start, $end, $step)).requires_grad()
    }};
    ($start:expr, $end:expr, $step:expr, false) => {{
        $crate::Input::new($crate::ndarray::Array::range($start, $end, $step))
    }};
}
