mod var;
use var::numeric::{DataRepr::Matrix, DataRepr::Scalar, DataRepr::Vector};
use var::reprs::{Input, Parameter};

#[cfg(test)]
mod tests {
    use super::{Input, Matrix, Parameter, Vector};
    use ndarray::prelude::array;
    #[test]
    fn it_works() {
        let w = Parameter::new(Matrix(array![
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ]));
        let x = Input::new(Matrix(array![
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0]
        ]));
        let b = Parameter::new(Vector(array![1.0, 1.0, 1.0]));
        let mut h = (w.mm(&x) + b).pow(2).sum();
        h.forward();
        &mut h.backward(1.0);
    }
}
