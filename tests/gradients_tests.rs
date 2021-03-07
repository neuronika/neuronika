use neuronika::*;

#[test]
fn scalar_add() {
    let x = Param::new(Scalar(1.0));
    let y = Param::new(Scalar(1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(1.0));
    assert_eq!(*y_grad, Scalar(1.0));

    let x = Param::new(Scalar(1.0));
    let y = Param::new(constant_vec([10], 1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(10.0));
    assert_eq!(*y_grad, constant_vec([10], 1.0));

    let x = Param::new(Scalar(1.0));
    let y = Param::new(constant_mat([10, 10], 1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(100.0));
    assert_eq!(*y_grad, constant_mat([10, 10], 1.0));
}

#[test]
fn scalar_sub() {
    let x = Param::new(Scalar(5.0));
    let y = Param::new(Scalar(3.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(1.0));
    assert_eq!(*y_grad, Scalar(-1.0));

    let x = Param::new(Scalar(5.0));
    let y = Param::new(constant_vec([10], 3.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(10.0));
    assert_eq!(*y_grad, constant_vec([10], -1.0));

    let x = Param::new(Scalar(1.0));
    let y = Param::new(constant_mat([10, 10], 1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(100.0));
    assert_eq!(*y_grad, constant_mat([10, 10], -1.0));
}

#[test]
fn scalar_mul() {
    let x = Param::new(Scalar(5.0));
    let y = Param::new(Scalar(3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(3.0));
    assert_eq!(*y_grad, Scalar(5.0));

    let x = Param::new(Scalar(5.0));
    let y = Param::new(constant_vec([10], 3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(30.0));
    assert_eq!(*y_grad, constant_vec([10], 5.0));

    let x = Param::new(Scalar(5.0));
    let y = Param::new(constant_mat([10, 10], 3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(300.0));
    assert_eq!(*y_grad, constant_mat([10, 10], 5.0));
}

#[test]
fn scalar_div() {
    let x = Param::new(Scalar(5.0));
    let y = Param::new(Scalar(5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(0.2));
    assert_eq!(*y_grad, Scalar(-0.2));

    let x = Param::new(Scalar(5.0));
    let y = Param::new(constant_vec([10], 5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(2.0));
    assert_eq!(*y_grad, constant_vec([10], -0.2));

    let x = Param::new(Scalar(5.0));
    let y = Param::new(constant_mat([5, 5], 5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, Scalar(5.0));
    assert_eq!(*y_grad, constant_mat([5, 5], -0.2));
}

#[test]
fn vector_add() {
    let x = Param::new(constant_vec([10], 1.0));
    let y = Param::new(Scalar(1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 1.0));
    assert_eq!(*y_grad, Scalar(10.0));

    let x = Param::new(constant_vec([10], 1.0));
    let y = Param::new(constant_vec([10], 1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 1.0));
    assert_eq!(*y_grad, constant_vec([10], 1.0));

    let x = Param::new(constant_vec([10], 1.0));
    let y = Param::new(constant_mat([10, 10], 1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 10.0));
    assert_eq!(*y_grad, constant_mat([10, 10], 1.0));

    let x = Param::new(constant_vec([10], 1.0));
    let y = Param::new(constant_mat([10, 1], 1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 10.0));
    assert_eq!(*y_grad, constant_mat([10, 1], 10.0));
}

#[test]
fn vector_sub() {
    let x = Param::new(constant_vec([10], 1.0));
    let y = Param::new(Scalar(1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 1.0));
    assert_eq!(*y_grad, Scalar(-10.0));

    let x = Param::new(constant_vec([10], 1.0));
    let y = Param::new(constant_vec([10], 1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 1.0));
    assert_eq!(*y_grad, constant_vec([10], -1.0));

    let x = Param::new(constant_vec([10], 1.0));
    let y = Param::new(constant_mat([10, 10], -1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 10.0));
    assert_eq!(*y_grad, constant_mat([10, 10], -1.0));

    let x = Param::new(constant_vec([10], 1.0));
    let y = Param::new(constant_mat([10, 1], 1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 10.0));
    assert_eq!(*y_grad, constant_mat([10, 1], -10.0));
}

#[test]
fn vector_mul() {
    let x = Param::new(constant_vec([10], 5.0));
    let y = Param::new(Scalar(3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 3.0));
    assert_eq!(*y_grad, Scalar(50.0));

    let x = Param::new(constant_vec([10], 5.0));
    let y = Param::new(constant_vec([10], 3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 3.0));
    assert_eq!(*y_grad, constant_vec([10], 5.0));

    let x = Param::new(constant_vec([10], 5.0));
    let y = Param::new(constant_mat([10, 10], 3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 30.0));
    assert_eq!(*y_grad, constant_mat([10, 10], 5.0));

    let x = Param::new(constant_vec([10], 5.0));
    let y = Param::new(constant_mat([10, 1], 3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 30.0));
    assert_eq!(*y_grad, constant_mat([10, 1], 50.0));
}

#[test]
fn vector_div() {
    let x = Param::new(constant_vec([10], 5.0));
    let y = Param::new(Scalar(5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 0.2));
    assert_eq!(*y_grad, Scalar(-2.0));

    let x = Param::new(constant_vec([10], 5.0));
    let y = Param::new(constant_vec([10], 5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([10], 0.2));
    assert_eq!(*y_grad, constant_vec([10], -0.2));

    let x = Param::new(constant_vec([5], 5.0));
    let y = Param::new(constant_mat([5, 5], 5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([5], 1.0));
    assert_eq!(*y_grad, constant_mat([5, 5], -0.2));

    let x = Param::new(constant_vec([5], 5.0));
    let y = Param::new(constant_mat([5, 1], 5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_vec([5], 1.0));
    assert_eq!(*y_grad, constant_mat([5, 1], -1.0));
}

#[test]
fn matrix_add() {
    let x = Param::new(constant_mat([5, 5], 1.0));
    let y = Param::new(Scalar(1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 5], 1.0));
    assert_eq!(*y_grad, Scalar(25.0));

    let x = Param::new(constant_mat([5, 5], 1.0));
    let y = Param::new(constant_vec([5], 1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 5], 1.0));
    assert_eq!(*y_grad, constant_vec([5], 5.0));

    let x = Param::new(constant_mat([5, 5], 1.0));
    let y = Param::new(constant_mat([5, 5], 1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 5], 1.0));
    assert_eq!(*y_grad, constant_mat([5, 5], 1.0));

    let x = Param::new(constant_mat([5, 1], 1.0));
    let y = Param::new(constant_mat([1, 5], 1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 1], 5.0));
    assert_eq!(*y_grad, constant_mat([1, 5], 5.0));

    let x = Param::new(constant_mat([1, 1], 1.0));
    let y = Param::new(constant_mat([5, 5], 1.0));

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([1, 1], 25.0));
    assert_eq!(*y_grad, constant_mat([5, 5], 1.0));
}

#[test]
fn matrix_sub() {
    let x = Param::new(constant_mat([5, 5], 1.0));
    let y = Param::new(Scalar(1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 5], 1.0));
    assert_eq!(*y_grad, Scalar(-25.0));

    let x = Param::new(constant_mat([5, 5], 1.0));
    let y = Param::new(constant_vec([5], -1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 5], 1.0));
    assert_eq!(*y_grad, constant_vec([5], -5.0));

    let x = Param::new(constant_mat([5, 5], 1.0));
    let y = Param::new(constant_mat([5, 5], -1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 5], 1.0));
    assert_eq!(*y_grad, constant_mat([5, 5], -1.0));

    let x = Param::new(constant_mat([5, 1], 1.0));
    let y = Param::new(constant_mat([1, 5], -1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 1], 5.0));
    assert_eq!(*y_grad, constant_mat([1, 5], -5.0));

    let x = Param::new(constant_mat([1, 1], 1.0));
    let y = Param::new(constant_mat([5, 5], 1.0));

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([1, 1], 25.0));
    assert_eq!(*y_grad, constant_mat([5, 5], -1.0));
}

#[test]
fn matrix_mul() {
    let x = Param::new(constant_mat([5, 5], 5.0));
    let y = Param::new(Scalar(3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 5], 3.0));
    assert_eq!(*y_grad, Scalar(125.0));

    let x = Param::new(constant_mat([5, 5], 5.0));
    let y = Param::new(constant_vec([5], 3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 5], 3.0));
    assert_eq!(*y_grad, constant_vec([5], 25.0));

    let x = Param::new(constant_mat([5, 5], 5.0));
    let y = Param::new(constant_mat([5, 5], 3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 5], 3.0));
    assert_eq!(*y_grad, constant_mat([5, 5], 5.0));

    let x = Param::new(constant_mat([5, 1], 5.0));
    let y = Param::new(constant_mat([1, 5], 3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([5, 1], 15.0));
    assert_eq!(*y_grad, constant_mat([1, 5], 25.0));

    let x = Param::new(constant_mat([1, 1], 5.0));
    let y = Param::new(constant_mat([5, 5], 3.0));

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([1, 1], 75.0));
    assert_eq!(*y_grad, constant_mat([5, 5], 5.0));
}

#[test]
fn matrix_div() {
    let x = Param::new(constant_mat([2, 2], 5.0));
    let y = Param::new(Scalar(5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([2, 2], 0.2));
    assert_eq!(*y_grad, Scalar(-0.8));

    let x = Param::new(constant_mat([2, 2], 5.0));
    let y = Param::new(constant_vec([2], 5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([2, 2], 0.2));
    assert_eq!(*y_grad, constant_vec([2], -0.4));

    let x = Param::new(constant_mat([2, 2], 5.0));
    let y = Param::new(constant_mat([2, 2], 5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([2, 2], 0.2));
    assert_eq!(*y_grad, constant_mat([2, 2], -0.2));

    let x = Param::new(constant_mat([2, 1], 5.0));
    let y = Param::new(constant_mat([1, 2], 5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([2, 1], 0.4));
    assert_eq!(*y_grad, constant_mat([1, 2], -0.4));

    let x = Param::new(constant_mat([1, 1], 5.0));
    let y = Param::new(constant_mat([2, 2], 5.0));

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(*x_grad, constant_mat([1, 1], 0.8));
    assert_eq!(*y_grad, constant_mat([2, 2], -0.2));
}
