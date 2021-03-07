pub mod nn;
mod var;
pub use var::multi_cat;
pub use var::numeric::{
    constant_mat, constant_vec, eye, normal_mat, normal_vec, uniform_mat, uniform_vec,
    DataRepr::Matrix, DataRepr::Scalar, DataRepr::Vector,
};
pub use var::ops::{Input, Param};
