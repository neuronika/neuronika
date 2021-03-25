use ndarray::{arr1, Array, Array1};
use neuronika::*;

// TODO: rewrite tests using new API.

#[test]
fn scalar_add() {
    let x = Param::new(Tensor {
        array: arr1(&[1.0]),
    });
    let y = Param::new(Tensor {
        array: arr1(&[1.0]),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[1.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[1.0])
        }
    );

    let x = Param::new(Tensor {
        array: arr1(&[1.0]),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[10.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![1.0; 10])
        }
    );

    let x = Param::new(Tensor {
        array: arr1(&[1.0]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([10, 10], 1.0),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[100.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([10, 10], 1.0)
        }
    );
}

#[test]
fn scalar_sub() {
    let x = Param::new(Tensor {
        array: arr1(&[5.0]),
    });
    let y = Param::new(Tensor {
        array: arr1(&[3.0]),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[1.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[-1.0])
        }
    );

    let x = Param::new(Tensor {
        array: arr1(&[5.0]),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![3.0; 10]),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[10.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![-1.0; 10])
        }
    );

    let x = Param::new(Tensor {
        array: arr1(&[1.0]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([10, 10], 1.0),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[100.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([10, 10], -1.0)
        }
    );
}

#[test]
fn scalar_mul() {
    let x = Param::new(Tensor {
        array: arr1(&[5.0]),
    });
    let y = Param::new(Tensor {
        array: arr1(&[3.0]),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[3.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[5.0])
        }
    );

    let x = Param::new(Tensor {
        array: arr1(&[5.0]),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![3.0; 10]),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[30.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![5.0; 10])
        }
    );

    let x = Param::new(Tensor {
        array: arr1(&[5.0]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([10, 10], 3.0),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[300.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([10, 10], 5.0)
        }
    );
}

#[test]
fn scalar_div() {
    let x = Param::new(Tensor {
        array: arr1(&[5.0]),
    });
    let y = Param::new(Tensor {
        array: arr1(&[5.0]),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[0.2])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[-0.2])
        }
    );

    let x = Param::new(Tensor {
        array: arr1(&[5.0]),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![5.0; 10]),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[2.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![-0.2; 10])
        }
    );

    let x = Param::new(Tensor {
        array: arr1(&[5.0]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([5, 5], 5.0),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: arr1(&[5.0])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([5, 5], -0.2)
        }
    );
}

#[test]
fn vector_add() {
    let x = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });
    let y = Param::new(Tensor {
        array: arr1(&[1.0]),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![1.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[10.0])
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![1.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![1.0; 10])
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![10.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([10, 10], 1.0),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![10.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([10, 10], 1.0)
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([10, 1], 1.0),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![10.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([10, 1], 10.0)
        }
    );
}

#[test]
fn vector_sub() {
    let x = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });
    let y = Param::new(Tensor {
        array: arr1(&[1.0]),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![1.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[-10.0])
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![1.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![-1.0; 10])
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([10, 10], -1.0),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![10.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([10, 10], -1.0)
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![1.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([10, 1], 1.0),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![10.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([10, 1], -10.0)
        }
    );
}

#[test]
fn vector_mul() {
    let x = Param::new(Tensor {
        array: Array1::from(vec![5.0; 10]),
    });
    let y = Param::new(Tensor {
        array: arr1(&[3.0]),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![3.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[50.0])
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![5.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![3.0; 10]),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![3.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![5.0; 10])
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![5.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([10, 10], 3.0),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![30.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([10, 10], 5.0)
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![5.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([10, 1], 3.0),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![30.0; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([10, 1], 50.0)
        }
    );
}

#[test]
fn vector_div() {
    let x = Param::new(Tensor {
        array: Array1::from(vec![5.0; 10]),
    });
    let y = Param::new(Tensor {
        array: arr1(&[5.0]),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![0.2; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[-2.0])
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![5.0; 10]),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![5.0; 10]),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![0.2; 10])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![-0.2; 10])
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![5.0; 5]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([5, 5], 5.0),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![1.0; 5])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([5, 5], -0.2)
        }
    );

    let x = Param::new(Tensor {
        array: Array1::from(vec![5.0; 5]),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([5, 1], 5.0),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array1::from(vec![1.0; 5])
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([5, 1], -1.0)
        }
    );
}

#[test]
fn matrix_add() {
    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });
    let y = Param::new(Tensor {
        array: arr1(&[1.0]),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[25.0])
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![1.0; 5]),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![5.0; 5])
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 1], 1.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([1, 5], 1.0),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 1], 5.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([1, 5], 5.0)
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([1, 1], 1.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });

    let mut z = x.clone() + y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([1, 1], 25.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );
}

#[test]
fn matrix_sub() {
    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });
    let y = Param::new(Tensor {
        array: arr1(&[1.0]),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[-25.0])
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![-1.0; 5]),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![-5.0; 5])
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([5, 5], -1.0),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([5, 5], -1.0)
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([1, 5], -1.0),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([1, 5], -5.0)
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([5, 5], 1.0),
    });

    let mut z = x.clone() - y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 1.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([5, 5], -1.0)
        }
    );
}

#[test]
fn matrix_mul() {
    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 5.0),
    });
    let y = Param::new(Tensor {
        array: arr1(&[3.0]),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 3.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[125.0])
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 5.0),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![3.0; 5]),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 3.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![25.0; 5])
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 5], 5.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([5, 5], 3.0),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 5], 3.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([5, 5], 5.0)
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([5, 1], 5.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([1, 5], 3.0),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([5, 1], 15.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([1, 5], 25.0)
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([1, 1], 5.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([5, 5], 3.0),
    });

    let mut z = x.clone() * y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([1, 1], 75.0)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([5, 5], 5.0)
        }
    );
}

#[test]
fn matrix_div() {
    let x = Param::new(Tensor {
        array: Array::from_elem([2, 2], 5.0),
    });
    let y = Param::new(Tensor {
        array: arr1(&[5.0]),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([2, 2], 0.2)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: arr1(&[-0.8])
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([2, 2], 5.0),
    });
    let y = Param::new(Tensor {
        array: Array1::from(vec![5.0; 2]),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([2, 2], 0.2)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array1::from(vec![-0.4; 2])
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([2, 2], 5.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([2, 2], 5.0),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([2, 2], 0.2)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([2, 2], -0.2)
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([2, 1], 5.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([1, 2], 5.0),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([2, 1], 0.4)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([1, 2], -0.4)
        }
    );

    let x = Param::new(Tensor {
        array: Array::from_elem([1, 1], 5.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([2, 2], 5.0),
    });

    let mut z = x.clone() / y.clone();

    z.forward();
    z.backward(1.0);

    let x_grad = x.grad();
    let y_grad = y.grad();

    assert_eq!(
        *x_grad,
        Tensor {
            array: Array::from_elem([1, 1], 0.8)
        }
    );
    assert_eq!(
        *y_grad,
        Tensor {
            array: Array::from_elem([2, 2], -0.2)
        }
    );
}

#[test]
fn upstream_test() {
    let x = Param::new(Tensor {
        array: Array::from_elem([2, 2], 5.0),
    });
    let y = Param::new(Tensor {
        array: Array::from_elem([2, 2], 5.0),
    });
    let z = Param::new(Tensor {
        array: Array::from_elem([1, 1], 1.0),
    });

    let w = x.clone() + y.clone() + z.clone() + x.clone();

    assert_eq!(w.upstream().len(), 3);
}