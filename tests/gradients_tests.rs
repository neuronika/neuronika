use ndarray::{arr1, Array, Array1};

#[test]
fn scalar_add() {
    let x = neuronika::tensor!([1.0]; true);
    let y = neuronika::tensor!([1.0]; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[1.0]));
    assert_eq!(*y_grad, arr1(&[1.0]));

    let x = neuronika::tensor!([1.0]; true);
    let y = neuronika::full!(10, 1.0; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[10.0]));

    assert_eq!(*y_grad, Array1::from(vec![1.0; 10]));

    let x = neuronika::tensor!([1.0]; true);
    let y = neuronika::full!((10,10), 1.0; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[100.0]));
    assert_eq!(*y_grad, Array::from_elem([10, 10], 1.0));
}

#[test]
fn scalar_sub() {
    let x = neuronika::tensor!([5.0]; true);
    let y = neuronika::tensor!([3.0]; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[1.0]));
    assert_eq!(*y_grad, arr1(&[-1.0]));

    let x = neuronika::tensor!([5.0]; true);
    let y = neuronika::full!(10, 3.0; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[10.0]));
    assert_eq!(*y_grad, Array1::from(vec![-1.0; 10]));

    let x = neuronika::tensor!([1.0]; true);
    let y = neuronika::full!((10, 10), 1.0; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[100.0]));
    assert_eq!(*y_grad, Array::from_elem([10, 10], -1.0));
}

#[test]
fn scalar_mul() {
    let x = neuronika::tensor!([5.0]; true);
    let y = neuronika::tensor!([3.0]; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[3.0]));
    assert_eq!(*y_grad, arr1(&[5.0]));

    let x = neuronika::tensor!([5.0]; true);
    let y = neuronika::full!(10, 3.0; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[30.0]));
    assert_eq!(*y_grad, Array1::from(vec![5.0; 10]));

    let x = neuronika::tensor!([5.0]; true);
    let y = neuronika::full!((10, 10), 3.0; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[300.0]));
    assert_eq!(*y_grad, Array::from_elem([10, 10], 5.0));
}

#[test]
fn scalar_div() {
    let x = neuronika::tensor!([5.0]; true);
    let y = neuronika::tensor!([5.0]; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[0.2]));
    assert_eq!(*y_grad, arr1(&[-0.2]));

    let x = neuronika::tensor!([5.0]; true);
    let y = neuronika::full!(10, 5.0; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[2.0]));
    assert_eq!(*y_grad, Array1::from(vec![-0.2; 10]));

    let x = neuronika::tensor!([5.0]; true);
    let y = neuronika::full!((5, 5), 5.0; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[5.0]));
    assert_eq!(*y_grad, Array::from_elem([5, 5], -0.2));
}

#[test]
fn vector_add() {
    let x = neuronika::full!(10, 1.0; true);
    let y = neuronika::tensor!([1.0]; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![1.0; 10]));
    assert_eq!(*y_grad, arr1(&[10.0]));

    let x = neuronika::full!(10, 1.0; true);
    let y = neuronika::full!(10, 1.0; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![1.0; 10]));
    assert_eq!(*y_grad, Array1::from(vec![1.0; 10]));

    let x = neuronika::full!(10, 10.0; true);
    let y = neuronika::full!((10, 10), 1.0; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![10.0; 10]));
    assert_eq!(*y_grad, Array::from_elem([10, 10], 1.0));

    let x = neuronika::full!(10, 1.0; true);
    let y = neuronika::full!((10, 1), 1.0; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![10.0; 10]));
    assert_eq!(*y_grad, Array::from_elem([10, 1], 10.0));
}

#[test]
fn vector_sub() {
    let x = neuronika::full!(10, 1.0; true);
    let y = neuronika::tensor!([1.0]; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![1.0; 10]));
    assert_eq!(*y_grad, arr1(&[-10.0]));

    let x = neuronika::full!(10, 1.0; true);
    let y = neuronika::full!(10, 1.0; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![1.0; 10]));
    assert_eq!(*y_grad, Array1::from(vec![-1.0; 10]));

    let x = neuronika::full!(10, 1.0; true);
    let y = neuronika::full!((10, 10), -1.0; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![10.0; 10]));
    assert_eq!(*y_grad, Array::from_elem([10, 10], -1.0));

    let x = neuronika::full!(10, 1.0; true);
    let y = neuronika::full!((10, 1), 1.0; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![10.0; 10]));
    assert_eq!(*y_grad, Array::from_elem([10, 1], -10.0));
}

#[test]
fn vector_mul() {
    let x = neuronika::full!(10, 5.0; true);
    let y = neuronika::tensor!([3.0]; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![3.0; 10]));
    assert_eq!(*y_grad, arr1(&[50.0]));

    let x = neuronika::full!(10, 5.0; true);
    let y = neuronika::full!(10, 3.0; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![3.0; 10]));
    assert_eq!(*y_grad, Array1::from(vec![5.0; 10]));

    let x = neuronika::full!(10, 5.0; true);
    let y = neuronika::full!((10, 10), 3.0; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![30.0; 10]));
    assert_eq!(*y_grad, Array::from_elem([10, 10], 5.0));

    let x = neuronika::full!(10, 5.0; true);
    let y = neuronika::full!((10, 1), 3.0; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![30.0; 10]));
    assert_eq!(*y_grad, Array::from_elem([10, 1], 50.0));
}

#[test]
fn vector_div() {
    let x = neuronika::full!(10, 5.0; true);
    let y = neuronika::tensor!([5.0]; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![0.2; 10]));
    assert_eq!(*y_grad, arr1(&[-2.0]));

    let x = neuronika::full!(10, 5.0; true);
    let y = neuronika::full!(10, 5.0; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![0.2; 10]));
    assert_eq!(*y_grad, Array1::from(vec![-0.2; 10]));

    let x = neuronika::full!(5, 5.0; true);

    let y = neuronika::full!((5, 5), 5.0; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![1.0; 5]));
    assert_eq!(*y_grad, Array::from_elem([5, 5], -0.2));

    let x = neuronika::full!(5, 5.0; true);
    let y = neuronika::full!((5, 1), 5.0; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array1::from(vec![1.0; 5]));
    assert_eq!(*y_grad, Array::from_elem([5, 1], -1.0));
}

#[test]
fn matrix_add() {
    let x = neuronika::full!((5, 5), 1.0; true);
    let y = neuronika::tensor!([1.0]; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 1.0));
    assert_eq!(*y_grad, arr1(&[25.0]));

    let x = neuronika::full!((5, 5), 1.0; true);
    let y = neuronika::full!(5, 1.0; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 1.0));
    assert_eq!(*y_grad, Array1::from(vec![5.0; 5]));

    let x = neuronika::full!((5, 5), 1.0; true);
    let y = neuronika::full!((5, 5), 1.0; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 1.0));
    assert_eq!(*y_grad, Array::from_elem([5, 5], 1.0));

    let x = neuronika::full!((5, 1), 1.0; true);
    let y = neuronika::full!((1, 5), 1.0; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 1], 5.0));
    assert_eq!(*y_grad, Array::from_elem([1, 5], 5.0));

    let x = neuronika::full!((1, 1), 1.0; true);
    let y = neuronika::full!((5, 5), 1.0; true);

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([1, 1], 25.0));
    assert_eq!(*y_grad, Array::from_elem([5, 5], 1.0));
}

#[test]
fn matrix_sub() {
    let x = neuronika::full!((5, 5), 1.0; true);
    let y = neuronika::tensor!([1.0]; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 1.0));
    assert_eq!(*y_grad, arr1(&[-25.0]));

    let x = neuronika::full!((5, 5), 1.0; true);
    let y = neuronika::full!(5, -1.0; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 1.0));
    assert_eq!(*y_grad, Array1::from(vec![-5.0; 5]));

    let x = neuronika::full!((5, 5), 1.0; true);
    let y = neuronika::full!((5, 5), -1.0; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 1.0));
    assert_eq!(*y_grad, Array::from_elem([5, 5], -1.0));

    let x = neuronika::full!((5, 5), 1.0; true);
    let y = neuronika::full!((1, 5), -1.0; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 1.0));
    assert_eq!(*y_grad, Array::from_elem([1, 5], -5.0));

    let x = neuronika::full!((1, 1), 1.0; true);
    let y = neuronika::full!((5, 5), 1.0; true);

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([1, 1], 25.0));
    assert_eq!(*y_grad, Array::from_elem([5, 5], -1.0));
}

#[test]
fn matrix_mul() {
    let x = neuronika::full!((5, 5), 5.0; true);

    let y = neuronika::tensor!([3.0]; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 3.0));
    assert_eq!(*y_grad, arr1(&[125.0]));

    let x = neuronika::full!((5, 5), 5.0; true);
    let y = neuronika::full!(5, 3.0; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 3.0));
    assert_eq!(*y_grad, Array1::from(vec![25.0; 5]));

    let x = neuronika::full!((5, 5), 5.0; true);
    let y = neuronika::full!((5, 5), 3.0; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 5], 3.0));
    assert_eq!(*y_grad, Array::from_elem([5, 5], 5.0));

    let x = neuronika::full!((5, 1), 5.0; true);
    let y = neuronika::full!((1, 5), 3.0; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([5, 1], 15.0));
    assert_eq!(*y_grad, Array::from_elem([1, 5], 25.0));

    let x = neuronika::full!((1, 1), 5.0; true);
    let y = neuronika::full!((5, 5), 3.0; true);

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([1, 1], 75.0));
    assert_eq!(*y_grad, Array::from_elem([5, 5], 5.0));
}

#[test]
fn matrix_div() {
    let x = neuronika::full!((2, 2), 5.0; true);
    let y = neuronika::tensor!([5.0]; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([2, 2], 0.2));
    assert_eq!(*y_grad, arr1(&[-0.8]));

    let x = neuronika::full!((2, 2), 5.0; true);
    let y = neuronika::full!(2, 5.0; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([2, 2], 0.2));
    assert_eq!(*y_grad, Array1::from(vec![-0.4; 2]));

    let x = neuronika::full!((2, 2), 5.0; true);
    let y = neuronika::full!((2, 2), 5.0; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([2, 2], 0.2));
    assert_eq!(*y_grad, Array::from_elem([2, 2], -0.2));

    let x = neuronika::full!((2, 1), 5.0; true);
    let y = neuronika::full!((1, 2), 5.0; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([2, 1], 0.4));
    assert_eq!(*y_grad, Array::from_elem([1, 2], -0.4));

    let x = neuronika::full!((1, 1), 5.0; true);
    let y = neuronika::full!((2, 2), 5.0; true);

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Array::from_elem([1, 1], 0.8));
    assert_eq!(*y_grad, Array::from_elem([2, 2], -0.2));
}

#[test]
fn upstream_test() {
    let x = neuronika::full!((2, 2), 5.0; true);
    let y = neuronika::full!((2, 2), 5.0; true);
    let z = neuronika::full!((1, 1), 1.0; true);

    let w = x.clone() + y + z + x;

    assert_eq!(w.upstream().len(), 3);
}

#[test]
fn stack() {
    let x = neuronika::full!(2, 1.0; true);
    let y = neuronika::full!(2, 1.0; true);
    let z = neuronika::full!(2, 1.0; true);

    let matrix_3x2 = neuronika::stack!(0, [x, y, z]);
    let matrix_2x3 = neuronika::stack!(1, [x, y, z]);

    assert_eq!(
        *matrix_3x2.data(),
        ndarray::array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    );
    assert_eq!(
        *matrix_2x3.data(),
        ndarray::array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );
}

#[test]
fn concatenate() {
    let x = neuronika::full!(2, 1.0; true);
    let y = neuronika::full!(2, 1.0; true);
    let z = neuronika::full!(2, 1.0; true);

    let long_vector = neuronika::cat!(0, [x, y, z]);

    assert_eq!(
        *long_vector.data(),
        ndarray::array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    );
}
