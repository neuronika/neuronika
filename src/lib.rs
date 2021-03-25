mod graph;
pub mod nn;
pub use graph::numeric::Tensor; // Pub for now
pub use graph::ops::{Input, Param}; // Same

pub use ndarray; // Used in macro export

#[macro_export]
macro_rules! tensor {
    ([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]; true) => {{
        let t = Tensor::new($crate::ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*]));
        Param::new(t)
    }};
    ([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]; false) => {{
        let t = Tensor::new($crate::ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*]));
        Input::new(t)
    }};
    ([$([$($x:expr),* $(,)*]),+ $(,)*]; true) => {{
        let t = Tensor::new($crate::ndarray::Array2::from(vec![$([$($x,)*],)*]));
        Param::new(t)
    }};
    ([$([$($x:expr),* $(,)*]),+ $(,)*]; false) => {{
        let t = Tensor::new($crate::ndarray::Array2::from(vec![$([$($x,)*],)*]));
        Input::new(t)
    }};
    ([$($x:expr),* $(,)*]; true) => {{
        let t = Tensor::new($crate::ndarray::Array1::from(vec![$($x,)*]));
        Param::new(t)
    }};

    ([$($x:expr),* $(,)*]; false) => {{
        let t = Tensor::new($crate::ndarray::Array1::from(vec![$($x,)*]));
        Input::new(t)
    }};
}

#[macro_export]
macro_rules! zeros {
    ($sh:expr; true) => {{
        let t = Tensor::new($crate::ndarray::Array::from_elem($sh, 0.0));
        Param::new(t)
    }};
    ($sh:expr; false) => {{
        let t = Tensor::new($crate::ndarray::Array::from_elem($sh, 0.0));
        Input::new(t)
    }};
}

#[macro_export]
macro_rules! ones {
    ($sh:expr; true) => {{
        let t = Tensor::new($crate::ndarray::Array::from_elem($sh, 1.0));
        Param::new(t)
    }};
    ($sh:expr; false) => {{
        let t = Tensor::new($crate::ndarray::Array::from_elem($sh, 1.0));
        Input::new(t)
    }};
}

#[macro_export]
macro_rules! full {
    ($sh:expr, $el:expr; true) => {{
        let t = Tensor::new($crate::ndarray::Array::from_elem($sh, $el));
        Param::new(t)
    }};
    ($sh:expr, $el:expr; false) => {{
        let t = Tensor::new($crate::ndarray::Array::from_elem($sh, $el));
        Input::new(t)
    }};
}