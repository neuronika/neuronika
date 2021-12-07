//! The `neuronika` crate provides auto-differentiation and dynamic neural networks.
//!
//! Neuronika is a machine learning framework written in pure Rust, built with a focus on ease of
//! use, fast experimentation and performance.
//!
//! # Highlights
//!
//! * Define by run computational graphs.
//! * Reverse-mode automatic differentiation.
//! * Dynamic neural networks.
//!
//! # Variables
//!
//! The main building blocks of neuronika are *variables* and *differentiable variables*.
//! This means that when using this crate you will be handling and manipulating instances of [`Var`]
//! and [`VarDiff`].
//!
//! Variables are lean and powerful abstractions over the computational graph's nodes. Neuronika
//! empowers you with the ability of imperatively building and differentiating such graphs with
//! minimal amount of code and effort.
//!
//! Both differentiable and non-differentiable variables can be understood as *tensors*. You
//! can perform all the basic arithmetic operations on them, such as: `+`, `-`, `*` and `/`.
//! Refer to [`Var`] and [`VarDiff`] for a complete list of the available operations.
//!
//! It is important to note that cloning variables is extremely memory efficient as only a shallow
//! copy is returned. Cloning a variable is thus the way to go if you need to use it several times.
//!
//! The provided API is linear in thought and minimal as it is carefully tailored around you, the
//! user.
//!
//! ### Quickstart
//!
//! If you’re familiar with Pytorch or Numpy, you will easily follow these example. If not, brace
//! yourself and follow along.
//!
//! First thing first, you should import neuronika.
//!
//! ```
//! use neuronika;
//! ```
//!
//! Neuronika's variables can be initialized in many ways. In the following, we will show some of
//! the possible alternatives:
//!
//! **With random or constant values**:
//!
//! Here `shape` determines the dimensionality of the output variable.
//! ```
//! let shape = [3, 4];
//!
//! let rand_variable = neuronika::rand(shape);
//! let ones_variable = neuronika::ones(shape);
//! let constant_variable = neuronika::full(shape, 7.);
//! ```
//!
//! **From a ndarray array**
//!
//! ```
//! use ndarray::array;
//!
//! let array = array![1., 2.];
//! let x_ndarray = neuronika::from_ndarray(array);
//! ```
//!
//! ## Leaf Variables
//!
//! You can create leaf variables by using one of the many provided functions, such as [`zeros()`],
//! [`ones()`], [`full()`] and [`rand()`]. Refer to the [complete list](#functions) for additional
//! information.
//!
//! Leaf variables are so called because they form the *leaves* of the computational graph, as are
//! not the result of any computation.
//!
//! Every leaf variable is by default created as non-differentiable, to promote it to a
//! *differentiable* leaf, i. e. a variable for which you can compute the gradient, you can use
//! [`.requires_grad()`](Var::requires_grad()).
//!
//! Differentiable leaf variables are leaves that have been promoted. You will encounter them
//! very often in your journey through neuronika as they are the the main components of the
//! neural networks' building blocks. To learn more in detail about those check the
//! [`nn`](module@nn) module.
//!
//! Differentiable leaves hold a gradient, you can access it with [`.grad()`](VarDiff::grad()).
//!
//! ## Differentiability Arithmetic
//!
//! As stated before, you can manipulate variables by performing operations on them; the results of
//! those computations will also be variables, although not leaf ones.
//!
//! The result of an operation between two differentiable variables will also be a differentiable
//! variable and the converse holds for non-differentiable variables. However, things behave
//! slightly differently when an operation is performed between a non-differentiable variable and a
//! differentiable one, as the resulting variable will be differentiable.
//!
//! You can think of differentiability as a *sticky* property. The table that follows is a summary
//! of how differentiability is broadcasted through variables.
//!
//!  **Operands** | Var     | VarDiff
//! --------------|---------|---------
//!  **Var**      | Var     | VarDiff
//!  **VarDiff**  | VarDiff | VarDiff
//!
//!
//! ## Differentiable Ancestors
//!
//! The differentiable ancestors of a variable are the differentiable leaves of the graph involved
//! in its computation. Obviously, only [`VarDiff`] can have a set of ancestors.
//!
//! You can gain access, via mutable views, to all the ancestors of a variable by iterating through
//! the vector of [`Param`] returned by [`.parameters()`](VarDiff::parameters()).
//! To gain more insights about the role that such components fulfil in neuronika feel free to check
//! the [`optim`] module.
//!
//! # Computational Graph
//!
//! A computational graph is implicitly created as you write your program. You can differentiate it
//! with respect to some of the differentiable leaves, thus populating their gradients, by using
//! [`.backward()`](VarDiff::backward()).
//!
//! It is important to note that the computational graph is *lazily* evaluated, this means that
//! neuronika decouples the construction of the graph from the actual computation of the nodes'
//! values. You must use `.forward()` in order to obtain the actual result of the computation.
//!
//!```
//! # #[cfg(feature = "blas")]
//! # extern crate blas_src;
//!use neuronika;
//!
//!let x = neuronika::rand(5);      //----+
//!let q = neuronika::rand((5, 5)); //    |- Those lines build the graph.
//!                                 //    |
//!let y = x.clone().vm(q).vv(x);   //----+
//!                                 //
//!y.forward();                     // After .forward() is called y contains the result.
//!```
//!
//! ## Freeing and keeping the graph
//!
//! By default, computational graphs will persist in the program's memory. If you want or need to be
//! more conservative about that you can wrap any arbitrary subset of the computations in an inner
//! scope. This allows for the corresponding portion of the graph to be freed when the end of
//! the scope is reached by the execution of your program.
//!
//!```
//! # #[cfg(feature = "blas")]
//! # extern crate blas_src;
//!use neuronika;
//!
//!let w = neuronika::rand((3, 3)).requires_grad(); // -----------------+
//!let b = neuronika::rand(3).requires_grad();      //                  |
//!let x = neuronika::rand((10, 3));                // -----------------+- Leaves are created
//!                                                 //                  
//!{                                                // ---+             
//!     let h = x.mm(w.t()) + b;                    //    | w's and b's
//!     h.forward();                                //    | grads are   
//!     h.backward(1.0);                            //    | accumulated
//!}                                                // ---+             |- Graph is freed and
//!                                                 // -----------------+  only leaves remain
//!```
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/neuronika/neuronika/main/misc/neuronika_brain.svg"
)]
#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/neuronika/neuronika/main/misc/neuronika_brain.ico"
)]

pub mod data;
pub mod nn;
pub mod optim;
mod variable;
use ndarray::{Array, Array2, Dimension, Ix1, Ix2, ShapeBuilder};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
pub use variable::{
    Backward, Cat, Convolve, ConvolveWithGroups, Data, Eval, Forward, Gradient, MatMatMul,
    MatMatMulT, Overwrite, Param, Stack, Var, VarDiff, VecMatMul, VecVecMul,
};
use variable::{Input, InputBackward};

/// Creates a variable from a **[ndarray]** array that owns its data.
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

/// Creates a variable with data filled with a constant value.
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
/// The shape is of type [`ndarray::ShapeBuilder`].
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

/// Creates a variable with an identity matrix of size *n*.
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
/// use neuronika;
/// use ndarray::arr1;
///
/// let tensor = neuronika::linspace(0., 1., 5);
/// assert!(*tensor.data() == arr1(&[0.0, 0.25, 0.5, 0.75, 1.0]))
/// ```
pub fn linspace(start: f32, end: f32, n: usize) -> Var<Input<Ix1>> {
    Input::new(Array::linspace(start, end, n))
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
pub fn logspace(base: f32, start: f32, end: f32, n: usize) -> Var<Input<Ix1>> {
    Input::new(Array::logspace(base, start, end, n))
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
pub fn geomspace(start: f32, end: f32, n: usize) -> Option<Var<Input<Ix1>>> {
    Array::geomspace(start, end, n).map(Input::new)
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
/// use neuronika;
/// use ndarray::arr1;
///
/// let tensor = neuronika::range(0., 5., 1.);
/// assert!(*tensor.data() == arr1(&[0., 1., 2., 3., 4.]))
/// ```
pub fn range(start: f32, end: f32, step: f32) -> Var<Input<Ix1>> {
    Input::new(Array::range(start, end, step))
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
