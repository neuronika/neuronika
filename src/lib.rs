mod nn;
mod var;
use var::multi_cat;
use var::numeric::{DataRepr::Matrix, DataRepr::Scalar, DataRepr::Vector};
use var::ops::{Input, Param};

#[cfg(test)]
mod tests {
    use super::{Input, Matrix, Param, Scalar, Vector};
    use ndarray::prelude::array;
    #[test]
    fn it_works() {
        let w = Param::new(Matrix(array![
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ]));
        let x = Input::new(Matrix(array![
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
            [3.0, 3.0, 3.0]
        ]));
        let b = Param::new(Vector(array![1.0, 1.0, 1.0]));
        let mut h = (w.mm(&x) + b).softmax(1).sum();
        h.forward();
        &mut h.backward(1.0);
    }
}
