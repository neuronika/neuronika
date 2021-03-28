mod graph;
pub mod nn;
pub use graph::node::{Input, Parameter};
pub use ndarray;

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
        let new = Tensor::new($crate::ndarray::Array1::from(vec![$($x,)*]));
        Input::new(new)
    }};
}

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
