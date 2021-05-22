//! The `neuronika` crate provides autodifferentiation and dynamic neural networks.
//!
//! Neuronika is a machine learning framework written in pure Rust, built with a focus on ease of
//! use, fast experimentation and performance.
//!
//! # Highlights
//!
//! * Define by run computational graphs
//! * Reverse-mode automatic differentiation
//! * Dynamic neural networks
//!
//! # Variables
//!
//! The main building blocks of neuronika are *variables* and *differentiable variables*.
//! This means that when you use this crate you are handling and manipulating instances of [`Var`]
//! and [`VarDiff`].
//!
//! Variables are lean and powerful abstractions over the computational graph's nodes. Neuronika
//! empowers you with the ability of imperatively building and differentiating such graphs with
//! minimal amount of code and effort.
//!
//! Both differentiable and non-differentiable variables can be understood as *tensors*, you
//! can perform all the basic arithmetic operations on them, such as: `+`, `-`, `*` and `/`.
//! Refer to [`Var`] and [`VarDiff`] for a complete list of the avaiable operations.
//!
//! The provided API is linear in thought and minimal as it is carefully tailored around you, the
//! user.
//!
//! ## Leaf Variables
//!
//! You can create leaf variables by using one of the many provided functions, such as [`zeros()`],
//! [`ones()`], [`full()`] and [`rand()`]. Feel free to refer to the [complete list](#functions).
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
//! It is important to note that the computational grap is *lazily* evalutated, this means that
//! neuronika decouples the construction of the graph from the actual computation of the nodes'
//! values. You must use `.forward()` in order to obtain the actual result of the computation.
//!
//!```
//! use neuronika;
//!
//! let x = neuronika::rand(5);                //----+
//! let q = neuronika::rand((5,5));            //    | Those lines just build
//!                                            //    | the graph.
//! let mut y = x.clone().vm_mul(q).vv_mul(x); //----+
//!                                            //
//! y.forward();                               // After .forward() is called y
//!                                            // contains the result.
//!```
//!
//! ## Freeing and keeping the graph
//!
//! By default computational graphs will persist in the program's memory. If you want to be more
//! conservative about this aspect you can place any arbitrary subset the computations in an inner
//! scope. This allows for the corresponding portion of the graph to be freed when the end of the
//! scope is reached by your program.
//!
//!```
//! use neuronika;
//!
//! let w = neuronika::rand((3,3)).requires_grad(); //------------------+    
//! let b = neuronika::rand(3).requires_grad();     //                  |
//! let x = neuronika::rand((10,3));                //                  |-- Leaves created   
//!                                                 //                  |
//! {                                               // ---+             |
//!     let mut h = x.mm_mul(w.t()) + b;            //    | w's and b's |
//!     h.forward();                                //    | grads are   |
//!     h.backward(1.0);                            //    | accumulated |
//! }                                               // ---+             |-- Graph freed and
//!                                                 // -----------------+   only leaves remain
//!```
pub mod data;
pub mod nn;
pub mod optim;
mod variable;
use ndarray::{Array, Array2, Dimension, Ix1, Ix2, ShapeBuilder};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use variable::node::{Input, InputBackward};
pub use variable::{Param, Var, VarDiff};

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

/// Creates a one-dimensional variable with *n* logarithmically spaced elements.
///
/// The starting value is `base.powf(start)` and the final one is `base.powf(end)`.
///
/// If `base` is negative, all values will be negative.
///
/// # Panics
///
/// If `n` is greater than `isize::MAX` or if converting `n - 1` to type `f32` fails.
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
/// If `n` is greater than `isize::MAX` or if converting `n - 1` to type `f32` fails.
pub fn geomspace(start: f32, end: f32, n: usize) -> Option<Var<Input<Ix1>>> {
    Array::geomspace(start, end, n).map(Input::new)
}

/// Creates a one-dimensional variable with elements from *start* to *end* spaced by *step*.
///
/// The elements range from `start` to `end` (exclusive).
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
