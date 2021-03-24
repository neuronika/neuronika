pub mod nn;
mod graph;
pub use graph::numeric::Tensor;
pub use graph::ops::{Input, Param};
pub use graph::{track_upstream, Trackable};

#[macro_export]
macro_rules! tensor {
    ([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]; true) => {{
        let t = Tensor {data:ndarray::Array3::from(vec![$([$([$($x,)*],)*],)*])};
        Param::new(t)
    }};
    ([$([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*]; false) => {{
        let t = Tensor {data:Array::from(vec![$([$([$($x,)*],)*],)*])};
        Input::new(t)
    }};
    ([$([$($x:expr),* $(,)*]),+ $(,)*]; true) => {{
        let t = Tensor {data: Array::from(vec![$([$($x,)*],)*])};
        Param::new(t)
    }};
    ([$([$($x:expr),* $(,)*]),+ $(,)*]; false) => {{
        let t = Tensor {data: Array::from(vec![$([$($x,)*],)*])};
        Input::new(t)
    }};
    ([$($x:expr),* $(,)*]; true) => {{
        let t = Tensor {data: Array::from(vec![$($x,)*])};
        Param::new(t)
    }};

    ([$($x:expr),* $(,)*]; false) => {{
        let t = Tensor {data: Array::from(vec![$($x,)*])};
        Input::new(t)
    }};
}
