mod var;
use var::numeric::{DataRepr::Matrix, DataRepr::Scalar, DataRepr::Vector};
use var::reprs::{Input, Parameter};

#[cfg(test)]
mod tests {
    use super::{Input, Matrix, Parameter, Vector};
    use ndarray::prelude::array;
    #[test]
    fn it_works() {
        let w = Parameter::new(Matrix(array![[1, 1, 1], [1, 1, 1], [1, 1, 1]]));
        let x = Input::new(Matrix(array![[3, 3, 3], [3, 3, 3], [3, 3, 3]]));
        let b = Parameter::new(Vector(array![1, 1, 1]));
        let mut h = w.mm(&x) + b;
        h.forward();
        &mut h.backward(1);
    }
}
