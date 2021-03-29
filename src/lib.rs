mod graph;
pub mod nn;
pub use graph::node::{Input, Parameter};
pub use ndarray;

/// Creates either a [**`Parameter`**](type.Paramater.html) or an [**`Input`**](type.Paramater.html)
/// node with one, two or three dimensions.
///
/// The return type and its differentiability depend on the `bool` passed after the data.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::tensor!([1., 2., 3., 4.]; false);
///
/// let t2 = neuronika::tensor!([[1., 2.],
///                             [3., 4.]]; true);
///
/// let t3 = neuronika::tensor!([[[1., 2.], [3., 4.]],
///                             [[5., 6.], [7., 8.]]]; true);
///
/// assert_eq!(t1.data().shape(), &[4]);
/// assert_eq!(t2.data().shape(), &[2, 2]);
/// assert_eq!(t3.data().shape(), &[2, 2, 2]);
///
/// assert!(!t1.requires_grad());
/// assert!(t2.requires_grad());
/// assert!(t3.requires_grad());
/// ```
#[macro_export]
macro_rules! tensor {
    ([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]; true) => {{
        let new = $crate::ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*]);
        $crate::Parameter::new(new)
    }};
    ([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]; false) => {{
        let new = $crate::ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*]);
        $crate::Input::new(new)
    }};
    ([$([$($x:expr),* $(,)*]),+ $(,)*]; true) => {{
        let new = $crate::ndarray::Array2::from(vec![$([$($x,)*],)*]);
        $crate::Parameter::new(new)
    }};
    ([$([$($x:expr),* $(,)*]),+ $(,)*]; false) => {{
        let new = $crate::ndarray::Array2::from(vec![$([$($x,)*],)*]);
        $crate::Input::new(new)
    }};
    ([$($x:expr),* $(,)*]; true) => {{
        let new = $crate::ndarray::Array1::from(vec![$($x,)*]);
        $crate::Parameter::new(new)
    }};

    ([$($x:expr),* $(,)*]; false) => {{
        let new = $crate::ndarray::Array1::from(vec![$($x,)*]);
        $crate::Input::new(new)
    }};
}

/// Creates either a [**`Parameter`**](type.Paramater.html) or an [**`Input`**](type.Paramater.html)
/// node with zeroed data.
///
/// The return type and its differentiability depend on the `bool` passed after the shape.
///
/// The shape is of type `ndarray::ShapeBuilder`. 
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::zeros!(1; true);
///
/// let t2 = neuronika::zeros!((1, 5); true);
///
/// let t3 = neuronika::zeros!([1, 2, 3]; false);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
///
/// assert!(t1.requires_grad());
/// assert!(t2.requires_grad());
/// assert!(!t3.requires_grad());
/// ```
#[macro_export]
macro_rules! zeros {
    ($sh:expr; true) => {{
        let new = $crate::ndarray::Array::from_elem($sh, 0.0);
        $crate::Parameter::new(new)
    }};
    ($sh:expr; false) => {{
        let new = $crate::ndarray::Array::from_elem($sh, 0.0);
        $crate::Input::new(new)
    }};
}

/// Creates either a [**`Parameter`**](type.Paramater.html) or an [**`Input`**](type.Paramater.html)
/// node with data filled with ones.
///
/// The return type and its differentiability depend on the `bool` passed after the shape.
///
/// The shape is of type `ndarray::ShapeBuilder`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::ones!(1; true);
///
/// let t2 = neuronika::ones!((1, 5); true);
///
/// let t3 = neuronika::ones!([1, 2, 3]; false);
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
///
/// assert!(t1.requires_grad());
/// assert!(t2.requires_grad());
/// assert!(!t3.requires_grad());
/// ```
#[macro_export]
macro_rules! ones {
    ($sh:expr; true) => {{
        let new = $crate::ndarray::Array::from_elem($sh, 1.0);
        $crate::Parameter::new(new)
    }};
    ($sh:expr; false) => {{
        let new = $crate::ndarray::Array::from_elem($sh, 1.0);
        $crate::Input::new(new)
    }};
}

/// Creates either a [**`Parameter`**](type.Paramater.html) or an [**`Input`**](type.Paramater.html)
/// node with data filled with the `f32` value `el`.
///
/// The return type and its differentiability depend on the `bool` passed after the shape.
///
/// The shape is of type `ndarray::ShapeBuilder`.
///
/// # Examples
///
/// ```
/// use neuronika;
/// let t1 = neuronika::full!(1, 5.; true); // Filled with 5.0
///
/// let t2 = neuronika::full!((1, 5), 6.; true); // Filled with 6.0
///
/// let t3 = neuronika::full!([1, 2, 3], 8.; false); // Filled with 8.0
///
/// assert_eq!(t1.data().shape(), &[1]);
/// assert_eq!(t2.data().shape(), &[1, 5]);
/// assert_eq!(t3.data().shape(), &[1, 2, 3]);
///
/// assert!(t1.requires_grad());
/// assert!(t2.requires_grad());
/// assert!(!t3.requires_grad());
/// ```
#[macro_export]
macro_rules! full {
    ($sh:expr, $el:expr; true) => {{
        let new = $crate::ndarray::Array::from_elem($sh, $el);
        $crate::Parameter::new(new)
    }};
    ($sh:expr, $el:expr; false) => {{
        let new = $crate::ndarray::Array::from_elem($sh, $el);
        $crate::Input::new(new)
    }};
}

#[macro_export]
macro_rules! cat {
    ($axis:expr, [$a:ident, $b:ident])=>{
        {
            $a.cat($b, $axis)
        }
    };
    ($axis:expr, [$a:ident, $($b:ident),*])=>{
       {
           $a.cat($crate::cat!($axis, [$($b),*]), $axis)
       }
    }
}

#[macro_export]
macro_rules! stack {
    ($axis:expr, [$a:ident, $b:ident])=>{
        {
            $a.unsqueeze($axis).cat($b.unsqueeze($axis), $axis)
        }
    };
    ($axis:expr, [$a:ident, $($b:ident),*])=>{
       {
           $a.unsqueeze($axis).cat($crate::stack!($axis, [$($b),*]), $axis)
       }
    }
}
