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
