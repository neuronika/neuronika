pub mod nn;
mod var;
pub use var::multi_cat;
pub use var::numeric::{
    constant_mat, constant_vec, eye, DataRepr::Matrix, DataRepr::Scalar, DataRepr::Vector,
};
pub use var::ops::{Input, Param};
