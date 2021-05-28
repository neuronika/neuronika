use ndarray::{arr1, Array, Array1};

#[test]
fn scalar_add() {
    // ------------------------------ Begin ------------------------------
    let x = neuronika::from_ndarray(ndarray::array![1.]).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![1.]).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem(1, 0.));

    z.forward();
    z.forward();
    assert_eq!(*z.data(), Array::from_elem(1, 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(1, 1.));
    assert_eq!(*y.grad(), Array::from_elem(1, 1.));

    let x = neuronika::from_ndarray(ndarray::array![1.]).requires_grad();
    let y = neuronika::full(10, 1.).requires_grad();

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[10.]));

    assert_eq!(*y_grad, Array1::from(vec![1.; 10]));

    let x = neuronika::from_ndarray(ndarray::array![1.]).requires_grad();
    let y = neuronika::full((10, 10), 1.).requires_grad();

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[100.]));
    assert_eq!(*y_grad, Array::from_elem([10, 10], 1.));
}

#[test]
fn scalar_sub() {
    let x = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![3.]).requires_grad();

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[1.]));
    assert_eq!(*y_grad, arr1(&[-1.]));

    let x = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();
    let y = neuronika::full(10, 3.).requires_grad();

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[10.]));
    assert_eq!(*y_grad, Array1::from(vec![-1.; 10]));

    let x = neuronika::from_ndarray(ndarray::array![1.]).requires_grad();
    let y = neuronika::full((10, 10), 1.).requires_grad();

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[100.]));
    assert_eq!(*y_grad, Array::from_elem([10, 10], -1.));
}

#[test]
fn scalar_mul() {
    let x = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![3.]).requires_grad();

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[3.]));
    assert_eq!(*y_grad, arr1(&[5.]));

    let x = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();
    let y = neuronika::full(10, 3.).requires_grad();

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[30.]));
    assert_eq!(*y_grad, Array1::from(vec![5.; 10]));

    let x = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();
    let y = neuronika::full((10, 10), 3.).requires_grad();

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[300.]));
    assert_eq!(*y_grad, Array::from_elem([10, 10], 5.));
}

#[test]
fn scalar_div() {
    let x = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[0.2]));
    assert_eq!(*y_grad, arr1(&[-0.2]));

    let x = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();
    let y = neuronika::full(10, 5.).requires_grad();

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[2.]));
    assert_eq!(*y_grad, Array1::from(vec![-0.2; 10]));

    let x = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();
    let y = neuronika::full((5, 5), 5.).requires_grad();

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, arr1(&[5.]));
    assert_eq!(*y_grad, Array::from_elem([5, 5], -0.2));
}

#[test]
fn vector_add() {
    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 1.).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![1.]).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem(10, 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 1.));
    assert_eq!(*y.grad(), Array::from_elem(1, 10.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 1.).requires_grad();
    let y = neuronika::full(10, 1.).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem(10, 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 1.));
    assert_eq!(*y.grad(), Array::from_elem(10, 1.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 10.).requires_grad();
    let y = neuronika::full((10, 10), 1.).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 11.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 10.));
    assert_eq!(*y.grad(), Array::from_elem((10, 10), 1.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 1.).requires_grad();
    let y = neuronika::full((10, 1), 1.).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 10.));
    assert_eq!(*y.grad(), Array::from_elem((10, 1), 10.));
}

#[test]
fn vector_sub() {
    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 1.).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![1.]).requires_grad();

    let mut z = x.clone() - y.clone();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 1.));
    assert_eq!(*y.grad(), Array::from_elem(1, -10.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 1.).requires_grad();
    let y = neuronika::full(10, 1.).requires_grad();

    let mut z = x.clone() - y.clone();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 1.));
    assert_eq!(*y.grad(), Array::from_elem(10, -1.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 1.).requires_grad();
    let y = neuronika::full((10, 10), -1.).requires_grad();

    let mut z = x.clone() - y.clone();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 10.));
    assert_eq!(*y.grad(), Array::from_elem((10, 10), -1.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 1.).requires_grad();
    let y = neuronika::full((10, 1), 1.).requires_grad();

    let mut z = x.clone() - y.clone();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 0.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 10.));
    assert_eq!(*y.grad(), Array::from_elem((10, 1), -10.));
}

#[test]
fn vector_mul() {
    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 5.).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![3.]).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem(10, 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 3.));
    assert_eq!(*y.grad(), Array::from_elem(1, 50.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 5.).requires_grad();
    let y = neuronika::full(10, 3.).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem(10, 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 3.));
    assert_eq!(*y.grad(), Array::from_elem(10, 5.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 5.).requires_grad();
    let y = neuronika::full((10, 10), 3.).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 30.));
    assert_eq!(*y.grad(), Array::from_elem((10, 10), 5.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 5.).requires_grad();
    let y = neuronika::full((10, 1), 3.).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((10, 10), 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 30.));
    assert_eq!(*y.grad(), Array::from_elem((10, 1), 50.));
}

#[test]
fn vector_div() {
    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 5.).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();

    let mut z = x.clone() / y.clone();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem(10, 1.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 0.2));
    assert_eq!(*y.grad(), Array::from_elem(1, -2.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(10, 5.).requires_grad();
    let y = neuronika::full(10, 5.).requires_grad();

    let mut z = x.clone() / y.clone();
    assert_eq!(*z.data(), Array::from_elem(10, 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem(10, 1.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(10, 0.2));
    assert_eq!(*y.grad(), Array::from_elem(10, -0.2));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(5, 5.).requires_grad();
    let y = neuronika::full((5, 5), 5.).requires_grad();

    let mut z = x.clone() / y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 1.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(5, 1.));
    assert_eq!(*y.grad(), Array::from_elem((5, 5), -0.2));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full(5, 5.).requires_grad();
    let y = neuronika::full((5, 1), 5.).requires_grad();

    let mut z = x.clone() / y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 1.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem(5, 1.));
    assert_eq!(*y.grad(), Array::from_elem((5, 1), -1.));
}

#[test]
fn matrix_add() {
    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 1.).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![1.]).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 1.));
    assert_eq!(*y.grad(), Array::from_elem(1, 25.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 1.).requires_grad();
    let y = neuronika::full(5, 1.).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 1.));
    assert_eq!(*y.grad(), Array::from_elem(5, 5.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 1.).requires_grad();
    let y = neuronika::full((5, 5), 1.).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 1.));
    assert_eq!(*y.grad(), Array::from_elem((5, 5), 1.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 1), 1.).requires_grad();
    let y = neuronika::full((1, 5), 1.).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 1), 5.));
    assert_eq!(*y.grad(), Array::from_elem((1, 5), 5.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((1, 1), 1.).requires_grad();
    let y = neuronika::full((5, 5), 1.).requires_grad();

    let mut z = x.clone() + y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((1, 1), 25.));
    assert_eq!(*y.grad(), Array::from_elem((5, 5), 1.));
}

#[test]
fn matrix_sub() {
    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 1.).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![1.]).requires_grad();

    let mut z = x.clone() - y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 1.));
    assert_eq!(*y.grad(), Array::from_elem(1, -25.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 1.).requires_grad();
    let y = neuronika::full(5, -1.).requires_grad();

    let mut z = x.clone() - y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 1.));
    assert_eq!(*y.grad(), Array::from_elem(5, -5.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 1.).requires_grad();
    let y = neuronika::full((5, 5), -1.).requires_grad();

    let mut z = x.clone() - y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 1.));
    assert_eq!(*y.grad(), Array::from_elem((5, 5), -1.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 1.).requires_grad();
    let y = neuronika::full((1, 5), -1.).requires_grad();

    let mut z = x.clone() - y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 2.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 1.));
    assert_eq!(*y.grad(), Array::from_elem((1, 5), -5.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((1, 1), 1.).requires_grad();
    let y = neuronika::full((5, 5), 1.).requires_grad();

    let mut z = x.clone() - y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((1, 1), 25.));
    assert_eq!(*y.grad(), Array::from_elem((5, 5), -1.));
}

#[test]
fn matrix_mul() {
    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 5.).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![3.]).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 3.));
    assert_eq!(*y.grad(), Array::from_elem(1, 125.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 5.).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![3.]).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 3.));
    assert_eq!(*y.grad(), Array::from_elem(1, 125.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 5.).requires_grad();
    let y = neuronika::full(5, 3.).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 3.));
    assert_eq!(*y.grad(), Array::from_elem(5, 25.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 5), 5.).requires_grad();
    let y = neuronika::full((5, 5), 3.).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 5), 3.));
    assert_eq!(*y.grad(), Array::from_elem((5, 5), 5.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((5, 1), 5.).requires_grad();
    let y = neuronika::full((1, 5), 3.).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((5, 1), 15.));
    assert_eq!(*y.grad(), Array::from_elem((1, 5), 25.));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((1, 1), 5.).requires_grad();
    let y = neuronika::full((5, 5), 3.).requires_grad();

    let mut z = x.clone() * y.clone();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((5, 5), 15.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((1, 1), 75.));
    assert_eq!(*y.grad(), Array::from_elem((5, 5), 5.));
}

#[test]
fn matrix_div() {
    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((2, 2), 5.).requires_grad();
    let y = neuronika::from_ndarray(ndarray::array![5.]).requires_grad();

    let mut z = x.clone() / y.clone();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 1.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((2, 2), 0.2));
    assert_eq!(*y.grad(), Array::from_elem(1, -0.8));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((2, 2), 5.).requires_grad();
    let y = neuronika::full(2, 5.).requires_grad();

    let mut z = x.clone() / y.clone();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 1.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((2, 2), 0.2));
    assert_eq!(*y.grad(), Array::from_elem(2, -0.4));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((2, 2), 5.).requires_grad();
    let y = neuronika::full((2, 2), 5.).requires_grad();

    let mut z = x.clone() / y.clone();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 1.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((2, 2), 0.2));
    assert_eq!(*y.grad(), Array::from_elem((2, 2), -0.2));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((2, 1), 5.).requires_grad();
    let y = neuronika::full((1, 2), 5.).requires_grad();

    let mut z = x.clone() / y.clone();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 1.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((2, 1), 0.4));
    assert_eq!(*y.grad(), Array::from_elem((1, 2), -0.4));

    // ------------------------------ Begin ------------------------------
    let x = neuronika::full((1, 1), 5.).requires_grad();
    let y = neuronika::full((2, 2), 5.).requires_grad();

    let mut z = x.clone() / y.clone();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 0.));

    z.forward();
    assert_eq!(*z.data(), Array::from_elem((2, 2), 1.));

    z.backward(1.);
    assert_eq!(*x.grad(), Array::from_elem((1, 1), 0.8));
    assert_eq!(*y.grad(), Array::from_elem((2, 2), -0.2));
}

#[test]
fn parameters_test() {
    let x = neuronika::full((2, 2), 5.).requires_grad();
    let y = neuronika::full((2, 2), 5.).requires_grad();
    let z = neuronika::full((1, 1), 1.).requires_grad();

    let w = x.clone() + y + z + x;

    assert_eq!(w.parameters().len(), 3);
}

#[test]
fn stack() {
    let x = neuronika::full(2, 1.).requires_grad();
    let y = neuronika::full(2, 1.).requires_grad();

    let mut res = x.clone().stack(y.clone(), 0);
    assert_eq!(*res.data(), ndarray::array![[0., 0.], [0., 0.]]);

    res.forward();
    assert_eq!(*res.data(), ndarray::array![[1., 1.], [1., 1.]]);

    let mut res = x.stack(y, 1);
    assert_eq!(*res.data(), ndarray::array![[0., 0.], [0., 0.]]);
    res.forward();
    assert_eq!(*res.data(), ndarray::array![[1., 1.], [1., 1.]]);
}

#[test]
fn concatenate() {
    let x = neuronika::full(2, 1.).requires_grad();
    let y = neuronika::full(2, 1.);

    let mut res = x.cat(y, 0);
    assert_eq!(*res.data(), ndarray::array![0., 0., 0., 0.]);

    res.forward();
    assert_eq!(*res.data(), ndarray::array![1., 1., 1., 1.]);
}
