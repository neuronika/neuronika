mod autograd;
mod gradient;
mod history;
mod node;
mod utils;
mod var;
mod vardiff;

#[cfg(feature = "serialize")]
mod serde;

use ndarray::{Array, Array2, Dimension, Ix1, Ix2, ShapeBuilder};

use ndarray_rand::{rand_distr::Uniform, RandomExt};

use neuronika_core::*;

pub use crate::{
    node::{Constant, PaddingMode, Reflective, Replicative, Zero},
    var::Var,
    vardiff::VarDiff,
};

#[cfg(feature = "cuda")]
pub mod cuda;

/// Specifies the reduction to apply to the criterion output.
#[derive(Copy, Clone, Debug)]
pub enum Reduction {
    /// The output will be summed.
    Sum,
    /// The sum of the output will be divided by the batch size for the Kullback-Leibler divergence
    /// and the negative log-likelihood. For all other criterions the output will be divided by
    /// the number of elements.
    Mean,
}

/// Creates a variable from a **[ndarray]** array that owns its data.
///
/// # Examples
///
/// ```
/// # use neuronika_variable as neuronika;
/// use ndarray;
///
/// let a = ndarray::array![[1., 2.], [3.,4.]];
/// let t = neuronika::from_ndarray(a.clone());
///
/// assert_eq!(*t.data(), a);
/// ```
pub fn from_ndarray<D>(array: Array<f32, D>) -> Var<D>
where
    D: Dimension,
{
    Var::leaf(array)
}

/// Creates a variable with zeroed data.
///
/// # Examples
///
/// ```
/// # use neuronika_variable as neuronika;
///
/// let t1 = neuronika::zeros(1);
/// let t2 = neuronika::zeros((1, 5));
/// let t3 = neuronika::zeros([1, 2, 3]);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
pub fn zeros<D, Sh>(shape: Sh) -> Var<D>
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    Var::leaf(Array::from_elem(shape, 0.0))
}

/// Creates a variable with data filled with ones.
///
/// # Examples
///
/// ```
/// # use neuronika_variable as neuronika;
///
/// let t1 = neuronika::ones(1);
/// let t2 = neuronika::ones((1, 5));
/// let t3 = neuronika::ones([1, 2, 3]);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
pub fn ones<D, Sh>(shape: Sh) -> Var<D>
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    Var::leaf(Array::from_elem(shape, 1.0))
}

/// Creates a variable with data filled with a constant value.
///
/// # Examples
///
/// ```
/// # use neuronika_variable as neuronika;
///
/// let t1 = neuronika::full(1, 5.); // Filled with 5.0
/// let t2 = neuronika::full((1, 5), 6.); // Filled with 6.0
/// let t3 = neuronika::full([1, 2, 3], 8.); // Filled with 8.0
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
/// ```
pub fn full<D, Sh>(shape: Sh, elem: f32) -> Var<D>
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    Var::leaf(Array::from_elem(shape, elem))
}

/// Creates a variable with values sampled from a uniform distribution on the interval *[0,1)*.
///
/// # Examples
///
/// ```
/// # use neuronika_variable as neuronika;
///
/// let x = neuronika::rand([4, 5, 6]);
/// assert_eq!(x.data().shape(), &[4, 5, 6]);
/// ```
pub fn rand<D, Sh>(shape: Sh) -> Var<D>
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    Var::leaf(Array::random(shape, Uniform::new(0., 1.)))
}

/// Creates a variable with an identity matrix of size *n*.
///
/// # Panics
///
/// If `n * n` would overflow `isize`.
///
/// # Examples
///
/// ```
/// # use neuronika_variable as neuronika;
/// use ndarray::Array2;
///
/// let tensor = neuronika::eye(3);
/// assert_eq!(*tensor.data(), Array2::eye(3));
/// ```
pub fn eye(n: usize) -> Var<Ix2> {
    Var::leaf(Array2::eye(n))
}

/// Creates a one-dimensional variable with *n* evenly spaced elements.
///
/// The elements range from `start` to `end` (exclusive).
///
/// # Panics
///
/// If the length is greater than [`isize::MAX`].
///
/// [`isize::MAX`]: https://doc.rust-lang.org/std/primitive.isize.html#associatedconstant.MAX
///
/// # Examples
///
/// ```
/// # use neuronika_variable as neuronika;
/// use ndarray::arr1;
///
/// let tensor = neuronika::linspace(0., 1., 5);
/// assert!(*tensor.data() == arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
/// ```
pub fn linspace(start: f32, end: f32, n: usize) -> Var<Ix1> {
    Var::leaf(Array::linspace(start, end, n))
}

/// Creates a one-dimensional variable with *n* logarithmically spaced elements.
///
/// The starting value is `base.powf(start)` and the final one is `base.powf(end)`.
///
/// If `base` is negative, all values will be negative.
///
/// # Panics
///
/// If `n` is greater than [`isize::MAX`] or if converting `n - 1` to type `f32` fails.
///
/// [`isize::MAX`]: https://doc.rust-lang.org/std/primitive.isize.html#associatedconstant.MAX
pub fn logspace(base: f32, start: f32, end: f32, n: usize) -> Var<Ix1> {
    Var::leaf(Array::logspace(base, start, end, n))
}

/// Creates a one-dimensional variable with *n* geometrically spaced elements.
///
/// The elements range from `start` to `end` (inclusive).
///
/// Returns `None` if `start` and `end` have different signs or if either one is zero. Conceptually,
/// this means that in order to obtain a `Some` result, `end / start` must be positive.
///
/// # Panics
///
/// If `n` is greater than [`isize::MAX`] or if converting `n - 1` to type `f32` fails.
///
/// [`isize::MAX`]: https://doc.rust-lang.org/std/primitive.isize.html#associatedconstant.MAX
pub fn geomspace(start: f32, end: f32, n: usize) -> Option<Var<Ix1>> {
    Array::geomspace(start, end, n).map(Var::leaf)
}

/// Creates a one-dimensional variable with elements from *start* to *end* spaced by *step*.
///
/// The elements range from `start` to `end` (exclusive).
///
/// # Panics
///
/// If the length is greater than
/// [`isize::MAX`].
///
/// [`isize::Max`]: https://doc.rust-lang.org/std/primitive.isize.html#associatedconstant.MAX
///
/// # Examples
///
/// ```
/// # use neuronika_variable as neuronika;
/// use ndarray::arr1;
///
/// let tensor = neuronika::range(0., 5., 1.);
/// assert!(*tensor.data() == arr1(&[0., 1., 2., 3., 4.]))
/// ```
pub fn range(start: f32, end: f32, step: f32) -> Var<Ix1> {
    Var::leaf(Array::range(start, end, step))
}

/// Concatenates the variables `lhs` and `rhs` along `axis`.
///
/// All variables must have the same shape, except in the concatenating dimension.
///
/// # Arguments
///
/// * `lhs` - variable.
///
/// * `rhs` - other variable.
///
/// * `axis` - axis to concatenate along to.
///
/// # Panics
///
/// If the variables have mismatching shapes, apart from along axis, if the variables are empty,
/// if `axis` is out of bounds or if the result is larger than is possible to represent.
pub fn cat<Lhs, Rhs>(lhs: Lhs, rhs: Rhs, axis: usize) -> <Lhs as Cat<Rhs>>::Output
where
    Lhs: Cat<Rhs>,
{
    Cat::cat(lhs, rhs, axis)
}

/// Stacks the variables `lhs` and `rhs` along `axis`.
///
/// All variables must have the same shape.
///
/// # Arguments
///
/// * `lhs` - variable.
///
/// * `rhs` - other variable.
///
/// * `axis` - axis to stack along to.
///
/// # Panics
///
/// If the variables have mismatching shapes, apart from along axis, if the variables are empty,
/// if `axis` is out of bounds or if the result is larger than is possible to represent.
pub fn stack<Lhs, Rhs>(lhs: Lhs, rhs: Rhs, axis: usize) -> <Lhs as Stack<Rhs>>::Output
where
    Lhs: Stack<Rhs>,
{
    Stack::stack(lhs, rhs, axis)
}

#[cfg(test)]
mod tests {
    #[test]
    fn from_ndarray_test() {
        use super::from_ndarray;
        let a = ndarray::array![[1., 2.], [3., 4.]];
        let t = from_ndarray(a.clone());

        assert_eq!(*t.data(), a);
    }

    #[test]
    fn zeros() {
        use super::zeros;

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
        use super::ones;

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
        use super::full;

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
        use super::rand;
        let t = rand([4, 5, 6]);

        assert_eq!(t.data().shape(), &[4, 5, 6]);
    }

    #[test]
    fn eye_test() {
        use super::{eye, Array2};
        let tensor = eye(3);

        assert_eq!(*tensor.data(), Array2::<f32>::eye(3));
    }

    #[test]
    fn linspace() {
        use super::linspace;
        let tensor = linspace(0., 1., 5);
        assert!(*tensor.data() == ndarray::arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
    }

    #[test]
    fn logspace() {
        use super::logspace;
        let tensor = logspace(2., 1., 5., 5);
        assert!(*tensor.data() == ndarray::arr1(&[2., 4., 8., 16., 32.]))
    }

    #[test]
    fn geomspace() {
        use super::geomspace;
        let tensor = geomspace(1., 1000., 4);
        assert!(tensor
            .unwrap()
            .data()
            .iter()
            .zip(ndarray::arr1(&[1.0_f32, 10.0_f32, 100.0_f32, 1000.0_f32]).iter())
            .all(|(&t, &a)| (t.round() - a.round()).abs() <= f32::EPSILON));
    }

    #[test]
    fn range_test() {
        use super::*;
        let tensor = range(0., 5., 1.);
        assert!(*tensor.data() == ndarray::arr1(&[0., 1., 2., 3., 4.]))
    }
}

#[cfg(test)]
mod test;
