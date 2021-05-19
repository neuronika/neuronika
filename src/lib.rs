pub mod data;
pub mod nn;
pub mod optim;
mod variable;
use ndarray::{Array, Array2, Dimension, Ix1, Ix2, ShapeBuilder};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use variable::node::{Input, InputBackward};
pub use variable::{
    Cat, MatMatMul, MatMatMulT, MatVecMul, Param, Stack, Var, VarDiff, VecMatMul, VecVecMul,
};

/// Creates a variable from a **[ndarray]** array.
///
/// # Examples
///
/// ```
/// use ndarray;
/// use neuronika;
///
/// let a = ndarray::array![[1., 2.], [3.,4.]];
/// let t = neuronika::from_ndarray(a.clone());
///
/// assert_eq!(*t.data(), a);
/// ```
pub fn from_ndarray<D: Dimension>(array: Array<f32, D>) -> Var<Input<D>> {
    Input::new(array)
}

/// Creates a variable with zeroed data.
///
/// The shape is of type [`ndarray::ShapeBuilder`].
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::zeros(1);
/// let t2 = neuronika::zeros((1, 5));
/// let t3 = neuronika::zeros([1, 2, 3]);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
pub fn zeros<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Var<Input<D>> {
    Input::new(Array::from_elem(shape, 0.0))
}

/// Creates a variable with data filled with ones.
///
/// The shape is of type [`ndarray::ShapeBuilder`].
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::ones(1);
/// let t2 = neuronika::ones((1, 5));
/// let t3 = neuronika::ones([1, 2, 3]);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
pub fn ones<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Var<Input<D>> {
    Input::new(Array::from_elem(shape, 1.0))
}

/// Creates a variable with data filled with the value `el`.
///
/// `el` must be `f32` and the shape of type [`ndarray::ShapeBuilder`].
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::full(1, 5.); // Filled with 5.0
/// let t2 = neuronika::full((1, 5), 6.); // Filled with 6.0
/// let t3 = neuronika::full([1, 2, 3], 8.); // Filled with 8.0
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
pub fn full<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh, elem: f32) -> Var<Input<D>> {
    Input::new(Array::from_elem(shape, elem))
}

/// Creates a variable with values sampled from a uniform distribution on the interval *[0,1)*.
///
/// The shape is of type `[ndarray::ShapeBuilder]`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t = neuronika::rand([4, 5, 6]);
///
/// assert_eq!(t.data().shape(), &[4, 5, 6]);
/// ```
pub fn rand<D: Dimension, Sh: ShapeBuilder<Dim = D>>(shape: Sh) -> Var<Input<D>> {
    Input::new(Array::random(shape, Uniform::new(0., 1.)))
}

/// Creates a variable with an identity matrix of size `n` (a square 2D matrix).
///
/// # Panics
///
/// If `n * n` would overflow `isize`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// use ndarray::Array2;
///
/// let tensor = neuronika::eye(3);
/// assert_eq!(*tensor.data(), Array2::eye(3));
/// ```
pub fn eye(n: usize) -> Var<Input<Ix2>> {
    Input::new(Array2::eye(n))
}

/// Creates a one-dimensional variable with `n` evenly spaced elements from `start` to `end`
/// (exclusive). The elements must be `f32`'s.
///
/// # Panics
///
/// If the length is greater than `isize::MAX`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// use ndarray::arr1;
///
/// let tensor = neuronika::linspace(0., 1., 5);
/// assert!(*tensor.data() == arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
/// ```
pub fn linspace(start: f32, end: f32, n: usize) -> Var<Input<Ix1>> {
    Input::new(Array::linspace(start, end, n))
}

/// Creates a one-dimensional variable with `n` logarithmically spaced elements, with the starting
/// value being `base.powf(start)` and the final one being `base.powf(end)`. The elements must be
/// `f32`'s.
///
/// If `base` is negative, all values will be negative.
///
/// # Panics
///
/// If `n` is greater than `isize::MAX` or if converting `n - 1` to type `f32` fails.
pub fn logspace(base: f32, start: f32, end: f32, n: usize) -> Var<Input<Ix1>> {
    Input::new(Array::logspace(base, start, end, n))
}

/// Creates a one-dimensional variable with `n` geometrically spaced elements from `start` to `end`
/// (inclusive). The elements must be `f32`'s.
///
/// Returns `None` if `start` and `end` have different signs or if either one is zero. Conceptually,
/// this means that in order to obtain a `Some` result, `end / start` must be positive.
///
/// # Panics
///
/// If `n` is greater than `isize::MAX` or if converting `n - 1` to type `f32` fails.
pub fn geomspace(start: f32, end: f32, n: usize) -> Option<Var<Input<Ix1>>> {
    Array::geomspace(start, end, n).map(Input::new)
}

/// Creates a one-dimensional variable with elements from `start` to `end` (exclusive),
/// incrementing by `step`. Elements must be `f32`'s.
///
/// # Panics
///
/// If the length is greater than `isize::MAX`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// use ndarray::arr1;
///
/// let tensor = neuronika::range(0., 5., 1.);
/// assert!(*tensor.data() == arr1(&[0., 1., 2., 3., 4.]))
/// ```
pub fn range(start: f32, end: f32, step: f32) -> Var<Input<Ix1>> {
    Input::new(Array::range(start, end, step))
}

#[cfg(test)]
mod tests {
    #[test]
    fn from_ndarray_test() {
        use super::*;
        let a = ndarray::array![[1., 2.], [3., 4.]];
        let t = from_ndarray(a.clone());

        assert_eq!(*t.data(), a);
    }

    #[test]
    fn zeros() {
        use super::*;

        let t1 = zeros(1);
        let t2 = zeros((1, 5));
        let t3 = zeros([1, 2, 3]);

        assert_eq!(t1.data().shape(), &[1]);
        assert_eq!(t2.data().shape(), &[1, 5]);
        assert_eq!(t3.data().shape(), &[1, 2, 3]);

        assert!(
            t1.data().iter().all(|el| *el <= f32::EPSILON)
                && t2.data().iter().all(|el| *el <= f32::EPSILON)
                && t3.data().iter().all(|el| *el <= f32::EPSILON)
        )
    }
    #[test]
    fn ones() {
        use super::*;

        let t1 = ones(1);
        let t2 = ones((1, 5));
        let t3 = ones([1, 2, 3]);

        assert_eq!(t1.data().shape(), &[1]);
        assert_eq!(t2.data().shape(), &[1, 5]);
        assert_eq!(t3.data().shape(), &[1, 2, 3]);

        assert!(
            t1.data().iter().all(|el| (*el - 1.).abs() <= f32::EPSILON)
                && t2.data().iter().all(|el| (*el - 1.).abs() <= f32::EPSILON)
                && t3.data().iter().all(|el| (*el - 1.).abs() <= f32::EPSILON)
        )
    }
    #[test]
    fn full() {
        use super::*;

        let t1 = full(1, 5.);
        let t2 = full((1, 5), 6.);
        let t3 = full([1, 2, 3], 8.);

        assert!(
            t1.data().iter().all(|el| (*el - 5.).abs() <= f32::EPSILON)
                && t2.data().iter().all(|el| (*el - 6.).abs() <= f32::EPSILON)
                && t3.data().iter().all(|el| (*el - 8.).abs() <= f32::EPSILON)
        )
    }

    #[test]
    fn rand_test() {
        use super::*;
        let t = rand([4, 5, 6]);

        assert_eq!(t.data().shape(), &[4, 5, 6]);
    }

    #[test]
    fn eye_test() {
        use super::*;
        let tensor = eye(3);

        assert_eq!(*tensor.data(), Array2::eye(3));
    }

    #[test]
    fn linspace() {
        use super::*;
        let tensor = linspace(0., 1., 5);
        assert!(*tensor.data() == ndarray::arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
    }

    #[test]
    fn range_test() {
        use super::*;
        let tensor = range(0., 5., 1.);
        assert!(*tensor.data() == ndarray::arr1(&[0., 1., 2., 3., 4.]))
    }
}
