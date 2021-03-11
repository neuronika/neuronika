use super::{BackwardAction, Tensor};
use ndarray::prelude::array;
use ndarray::{Array1, Array2};

#[test]
fn add_test() {
    let scalar = Tensor { data: array![1.0] };
    let vector = Tensor {
        data: array![1.0, 2.0, 3.0],
    };
    let matrix = Tensor {
        data: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    };
    let singleton_matrix = Tensor {
        data: array![[1.0]],
    };

    // Scalar + Scalar.
    let scalar_res: &Array1<f32> = &(&scalar + &Tensor { data: array![9.0] }).data;
    assert_eq!(scalar_res[0], 10.0);

    // Scalar + Scalar forward.
    let mut forward_scalar_res = Tensor { data: array![0.0] };
    forward_scalar_res.add_fwd(&scalar, &Tensor { data: array![9.0] });
    assert_eq!(forward_scalar_res.data, *scalar_res);

    // Scalar + Vector.
    let res = &scalar + &vector;
    let scalar_vector_res: &Array1<f32> = &res.data;
    assert_eq!(*scalar_vector_res, array![2.0, 3.0, 4.0]);

    // Scalar + Vector forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.add_fwd(&scalar, &vector);
    assert_eq!(forward_vec_res.data, *scalar_vector_res);

    // Scalar + Matrix.
    let res = &scalar + &matrix;
    let scalar_matrix_res: &Array2<f32> = &&res.data;
    assert_eq!(
        *scalar_matrix_res,
        array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    );

    // Scalar + Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.add_fwd(&scalar, &matrix);
    assert_eq!(forward_mat_res.data, *scalar_matrix_res);

    // Vector + Scalar.
    let res = &vector + &scalar;
    let vector_scalar_res: &Array1<f32> = &res.data;
    assert_eq!(*vector_scalar_res, *scalar_vector_res);

    // Vector + Scalar forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.add_fwd(&vector, &scalar);
    assert_eq!(forward_vec_res.data, *vector_scalar_res);

    // Vector + Vector.
    let res = &vector
        + &Tensor {
            data: array![3.0, 2.0, 1.0],
        };
    let vector_vector_res: &Array1<f32> = &res.data;
    //println!("{:?}", vector_vector_res);
    assert_eq!(*vector_vector_res, array![4.0, 4.0, 4.0]);

    // Vector + Vector forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.add_fwd(
        &vector,
        &Tensor {
            data: array![3.0, 2.0, 1.0],
        },
    );
    assert_eq!(forward_vec_res.data, *vector_vector_res);

    // Vector + Matrix
    let res = &vector + &matrix;
    let vector_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *vector_matrix_res,
        array![[2.0, 4.0, 6.0], [5.0, 7.0, 9.0], [8.0, 10.0, 12.0]]
    );

    // Vector + Matrix forward.
    let mut forward_vec_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_vec_res.add_fwd(&vector, &matrix);
    assert_eq!(forward_vec_res.data, *vector_matrix_res);

    // Vector + singleton Matrix.
    let res = &vector + &singleton_matrix;
    let vector_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(*vector_s_matrix_res, array![[2.0, 3.0, 4.0]]);

    // Vector + singleton Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0]],
    };
    forward_mat_res.add_fwd(&vector, &singleton_matrix);
    assert_eq!(forward_mat_res.data, *vector_s_matrix_res);

    // Singleton Matrix + Vector.
    let res = &singleton_matrix + &vector;
    let vector_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(*vector_s_matrix_res, array![[2.0, 3.0, 4.0]]);

    // Singleton Matrix + Vector forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0]],
    };
    forward_mat_res.add_fwd(&singleton_matrix, &vector);
    assert_eq!(forward_mat_res.data, *vector_s_matrix_res);

    // Matrix + Scalar.
    let res = &matrix + &scalar;
    let matrix_scalar_res: &Array2<f32> = &res.data;
    assert_eq!(*matrix_scalar_res, *scalar_matrix_res);

    // Matrix + Scalar forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.add_fwd(&matrix, &scalar);
    assert_eq!(forward_mat_res.data, *matrix_scalar_res);

    // Matrix + Vector.
    let res = &matrix + &vector;
    let matrix_vector_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_vector_res,
        array![[2.0, 4.0, 6.0], [5.0, 7.0, 9.0], [8.0, 10.0, 12.0]]
    );

    // Matrix + Vector forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.add_fwd(&matrix, &vector);
    assert_eq!(forward_mat_res.data, *matrix_vector_res);

    // Matrix + Matrix.
    let res = &matrix
        + &Tensor {
            data: array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        };
    let matrix_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_matrix_res,
        array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    );

    // Matrix + Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.add_fwd(
        &matrix,
        &Tensor {
            data: array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        },
    );
    assert_eq!(forward_mat_res.data, *matrix_matrix_res);

    // Matrix + Matrix broadcast.
    let res = &matrix
        + &Tensor {
            data: array![[1.0, 2.0, 3.0]],
        };
    let matrix_matrix_b_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_matrix_b_res,
        array![[2.0, 4.0, 6.0], [5.0, 7.0, 9.0], [8.0, 10.0, 12.0]]
    );

    // Matrix + Matrix forward broadcast.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.add_fwd(
        &matrix,
        &Tensor {
            data: array![[1.0, 2.0, 3.0]],
        },
    );
    assert_eq!(forward_mat_res.data, *matrix_matrix_b_res);

    // Singleton Matrix + Matrix.
    let res = &singleton_matrix + &matrix;
    let matrix_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_s_matrix_res,
        array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    );

    // Singleton Matrix + Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.add_fwd(&singleton_matrix, &matrix);
    assert_eq!(forward_mat_res.data, *matrix_s_matrix_res);
}

#[test]
fn sub_test() {
    let scalar = Tensor { data: array![1.0] };
    let vector = Tensor {
        data: array![1.0, 2.0, 3.0],
    };
    let matrix = Tensor {
        data: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    };
    let singleton_matrix = Tensor {
        data: array![[1.0]],
    };

    // Scalar - Scalar.
    let scalar_res: Array1<f32> = (&scalar - &Tensor { data: array![1.0] }).data;
    assert_eq!(scalar_res[0], 0.0);

    // Scalar - Scalar forward.
    let mut forward_scalar_res = Tensor { data: array![0.0] };
    forward_scalar_res.sub_fwd(&scalar, &Tensor { data: array![1.0] });
    assert_eq!(forward_scalar_res.data, scalar_res);

    // Scalar - Vector.
    let res = &scalar - &vector;
    let scalar_vector_res: &Array1<f32> = &res.data;
    assert_eq!(*scalar_vector_res, array![0.0, -1.0, -2.0]);

    // Scalar - Vector forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.sub_fwd(&scalar, &vector);
    assert_eq!(forward_vec_res.data, *scalar_vector_res);

    // Scalar - Matrix.
    let res = &scalar - &matrix;
    let scalar_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *scalar_matrix_res,
        array![[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0], [-6.0, -7.0, -8.0]]
    );

    // Scalar - Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.sub_fwd(&scalar, &matrix);
    assert_eq!(forward_mat_res.data, *scalar_matrix_res);

    // Vector - Scalar.
    let res = &vector - &scalar;
    let vector_scalar_res: &Array1<f32> = &res.data;
    assert_eq!(*vector_scalar_res, -scalar_vector_res);

    // Vector - Scalar forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.sub_fwd(&vector, &scalar);
    assert_eq!(forward_vec_res.data, *vector_scalar_res);

    // Vector - Vector.
    let res = &vector
        - &Tensor {
            data: array![3.0, 2.0, 1.0],
        };
    let vector_vector_res: &Array1<f32> = &res.data;
    assert_eq!(*vector_vector_res, array![-2.0, 0.0, 2.0]);

    // Vector - Vector forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.sub_fwd(
        &vector,
        &Tensor {
            data: array![3.0, 2.0, 1.0],
        },
    );
    assert_eq!(forward_vec_res.data, *vector_vector_res);

    // Vector - Matrix
    let res = &vector - &matrix;
    let vector_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *vector_matrix_res,
        array![[0.0, 0.0, 0.0], [-3.0, -3.0, -3.0], [-6.0, -6.0, -6.0]]
    );

    // Vector - Matrix forward.
    let mut forward_vec_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_vec_res.sub_fwd(&vector, &matrix);
    assert_eq!(forward_vec_res.data, *vector_matrix_res);

    // Vector - singleton Matrix.
    let res = &vector - &singleton_matrix;
    let vector_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(*vector_s_matrix_res, array![[0.0, 1.0, 2.0]]);

    // Vector - singleton Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0]],
    };
    forward_mat_res.sub_fwd(&vector, &singleton_matrix);
    assert_eq!(forward_mat_res.data, *vector_s_matrix_res);

    // Singleton Matrix - Vector.
    let res = &singleton_matrix - &vector;
    let vector_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(*vector_s_matrix_res, array![[0.0, -1.0, -2.0]]);

    // Singleton Matrix - Vector forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0]],
    };
    forward_mat_res.sub_fwd(&singleton_matrix, &vector);
    assert_eq!(forward_mat_res.data, *vector_s_matrix_res);

    // Matrix - Scalar.
    let res = &matrix - &scalar;
    let matrix_scalar_res: &Array2<f32> = &res.data;
    assert_eq!(*matrix_scalar_res, -scalar_matrix_res);

    // Matrix - Scalar forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.sub_fwd(&matrix, &scalar);
    assert_eq!(forward_mat_res.data, *matrix_scalar_res);

    // Matrix - Vector.
    let res = &matrix - &vector;
    let matrix_vector_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_vector_res,
        array![[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]]
    );

    // Matrix - Vector forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.sub_fwd(&matrix, &vector);
    assert_eq!(forward_mat_res.data, *matrix_vector_res);

    // Matrix - Matrix.
    let res = &matrix
        - &Tensor {
            data: array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        };
    let matrix_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_matrix_res,
        array![[-8.0, -6.0, -4.0], [-2.0, 0.0, 2.0], [4.0, 6.0, 8.0]]
    );

    // Matrix - Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.sub_fwd(
        &matrix,
        &Tensor {
            data: array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        },
    );
    assert_eq!(forward_mat_res.data, *matrix_matrix_res);

    // Matrix - Matrix broadcast.
    let res = &matrix
        - &Tensor {
            data: array![[1.0, 2.0, 3.0]],
        };
    let matrix_matrix_b_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_matrix_b_res,
        array![[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]]
    );

    // Matrix - Matrix forward broadcast.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.sub_fwd(
        &matrix,
        &Tensor {
            data: array![[1.0, 2.0, 3.0]],
        },
    );
    assert_eq!(forward_mat_res.data, *matrix_matrix_b_res);

    // Singleton Matrix - Matrix.
    let res = &singleton_matrix - &matrix;
    let matrix_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_s_matrix_res,
        -array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    );

    // Singleton Matrix - Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.sub_fwd(&singleton_matrix, &matrix);
    assert_eq!(forward_mat_res.data, *matrix_s_matrix_res);
}

#[test]
fn mul_test() {
    let scalar = Tensor { data: array![3.0] };
    let vector = Tensor {
        data: array![1.0, 2.0, 3.0],
    };
    let matrix = Tensor {
        data: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    };
    let singleton_matrix = Tensor {
        data: array![[1.0]],
    };

    // Scalar * Scalar.
    let scalar_res: Array1<f32> = (&scalar * &Tensor { data: array![-3.] }).data;
    assert_eq!(scalar_res[0], -9.0);

    // Scalar * Scalar forward.
    let mut forward_scalar_res = Tensor { data: array![0.0] };
    forward_scalar_res.mul_fwd(&scalar, &Tensor { data: array![-3.] });
    assert_eq!(forward_scalar_res.data, scalar_res);

    // Scalar * Vector.
    let res = &scalar * &vector;
    let scalar_vector_res: &Array1<f32> = &res.data;
    assert_eq!(*scalar_vector_res, array![3.0, 6.0, 9.0]);

    // Scalar * Vector forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.mul_fwd(&scalar, &vector);
    assert_eq!(forward_vec_res.data, *scalar_vector_res);

    // Scalar * Matrix.
    let res = &scalar * &matrix;
    let scalar_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *scalar_matrix_res,
        array![[3.0, 6.0, 9.0], [12.0, 15.0, 18.0], [21.0, 24.0, 27.0]]
    );

    // Scalar * Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.mul_fwd(&scalar, &matrix);
    assert_eq!(forward_mat_res.data, *scalar_matrix_res);

    // Vector * Scalar.
    let res = &vector * &scalar;
    let vector_scalar_res: &Array1<f32> = &res.data;
    assert_eq!(*vector_scalar_res, *scalar_vector_res);

    // Vector * Scalar forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.mul_fwd(&vector, &scalar);
    assert_eq!(forward_vec_res.data, *vector_scalar_res);

    // Vector * Vector.
    let res = &vector
        * &Tensor {
            data: array![3.0, 2.0, 1.0],
        };
    let vector_vector_res: &Array1<f32> = &res.data;
    assert_eq!(*vector_vector_res, array![3.0, 4.0, 3.0]);

    // Vector * Vector forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.mul_fwd(
        &vector,
        &Tensor {
            data: array![3.0, 2.0, 1.0],
        },
    );
    assert_eq!(forward_vec_res.data, *vector_vector_res);

    // Vector * Matrix
    let res = &vector * &matrix;
    let vector_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *vector_matrix_res,
        array![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0], [7.0, 16.0, 27.0]]
    );

    // Vector * Matrix forward.
    let mut forward_vec_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_vec_res.mul_fwd(&vector, &matrix);
    assert_eq!(forward_vec_res.data, *vector_matrix_res);

    // Vector * singleton Matrix.
    let res = &vector * &singleton_matrix;
    let vector_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(*vector_s_matrix_res, array![[1.0, 2.0, 3.0]]);

    // Vector * singleton Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0]],
    };
    forward_mat_res.mul_fwd(&vector, &singleton_matrix);
    assert_eq!(forward_mat_res.data, *vector_s_matrix_res);

    // Singleton Matrix * Vector.
    let res = &singleton_matrix * &vector;
    let vector_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(*vector_s_matrix_res, array![[1.0, 2.0, 3.0]]);

    // Singleton Matrix * Vector forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0]],
    };
    forward_mat_res.mul_fwd(&singleton_matrix, &vector);
    assert_eq!(forward_mat_res.data, *vector_s_matrix_res);

    // Matrix * Scalar.
    let res = &matrix * &scalar;
    let matrix_scalar_res: &Array2<f32> = &res.data;
    assert_eq!(*matrix_scalar_res, *scalar_matrix_res);

    // Matrix * Scalar forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.mul_fwd(&matrix, &scalar);
    assert_eq!(forward_mat_res.data, *matrix_scalar_res);

    // Matrix * Vector.
    let res = &matrix * &vector;
    let matrix_vector_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_vector_res,
        array![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0], [7.0, 16.0, 27.0]]
    );

    // Matrix * Vector forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.mul_fwd(&matrix, &vector);
    assert_eq!(forward_mat_res.data, *matrix_vector_res);

    // Matrix * Matrix.
    let res = &matrix
        * &Tensor {
            data: array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        };
    let matrix_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_matrix_res,
        array![[9.0, 16.0, 21.0], [24.0, 25.0, 24.0], [21.0, 16.0, 9.0]]
    );

    // Matrix * Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.mul_fwd(
        &matrix,
        &Tensor {
            data: array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        },
    );
    assert_eq!(forward_mat_res.data, *matrix_matrix_res);

    // Matrix * Matrix broadcast.
    let res = &matrix
        * &Tensor {
            data: array![[1.0, 2.0, 3.0]],
        };
    let matrix_matrix_b_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_matrix_b_res,
        array![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0], [7.0, 16.0, 27.0]]
    );

    // Matrix * Matrix forward broadcast.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.mul_fwd(
        &matrix,
        &Tensor {
            data: array![[1.0, 2.0, 3.0]],
        },
    );
    assert_eq!(forward_mat_res.data, *matrix_matrix_b_res);

    // Singleton Matrix * Matrix.
    let res = &singleton_matrix * &matrix;
    let matrix_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_s_matrix_res,
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    );

    // Singleton Matrix * Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.mul_fwd(&singleton_matrix, &matrix);
    assert_eq!(forward_mat_res.data, *matrix_s_matrix_res);
}

#[test]
fn div_test() {
    let scalar = Tensor { data: array![3.0] };
    let vector = Tensor {
        data: array![3.0, 3.0, 3.0],
    };
    let matrix = Tensor {
        data: array![[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
    };
    let singleton_matrix = Tensor {
        data: array![[1.0]],
    };

    // Scalar / Scalar.
    let scalar_res: Array1<f32> = (&scalar / &Tensor { data: array![-3.] }).data;
    assert_eq!(scalar_res[0], -1.0);

    // Scalar / Scalar forward.
    let mut forward_scalar_res = Tensor { data: array![0.0] };
    forward_scalar_res.div_fwd(&scalar, &Tensor { data: array![-3.] });
    assert_eq!(forward_scalar_res.data, scalar_res);

    // Scalar / Vector.
    let res = &scalar / &vector;
    let scalar_vector_res: &Array1<f32> = &res.data;
    assert_eq!(*scalar_vector_res, array![1.0, 1.0, 1.0]);

    // Scalar / Vector forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.div_fwd(&scalar, &vector);
    assert_eq!(forward_vec_res.data, *scalar_vector_res);

    // Scalar / Matrix.
    let res = &scalar / &matrix;
    let scalar_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *scalar_matrix_res,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Scalar / Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.div_fwd(&scalar, &matrix);
    assert_eq!(forward_mat_res.data, *scalar_matrix_res);

    // Vector / Scalar.
    let res = &vector / &scalar;
    let vector_scalar_res: &Array1<f32> = &res.data;
    assert_eq!(*vector_scalar_res, *scalar_vector_res);

    // Vector / Scalar forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.div_fwd(&vector, &scalar);
    assert_eq!(forward_vec_res.data, *vector_scalar_res);

    // Vector / Vector.
    let res = &vector
        / &Tensor {
            data: array![1.0, 3.0, 6.0],
        };
    let vector_vector_res: &Array1<f32> = &res.data;
    assert_eq!(*vector_vector_res, array![3.0, 1.0, 0.5]);

    // Vector / Vector forward.
    let mut forward_vec_res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    forward_vec_res.div_fwd(
        &vector,
        &Tensor {
            data: array![1.0, 3.0, 6.0],
        },
    );
    assert_eq!(forward_vec_res.data, *vector_vector_res);

    // Vector / Matrix
    let res = &vector / &matrix;
    let vector_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *vector_matrix_res,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Vector / Matrix forward.
    let mut forward_vec_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_vec_res.div_fwd(&vector, &matrix);
    assert_eq!(forward_vec_res.data, *vector_matrix_res);

    // Vector / singleton Matrix.
    let res = &vector / &singleton_matrix;
    let vector_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(*vector_s_matrix_res, array![[3.0, 3.0, 3.0]]);

    // Vector / singleton Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0]],
    };
    forward_mat_res.div_fwd(&vector, &singleton_matrix);
    assert_eq!(forward_mat_res.data, *vector_s_matrix_res);

    // Singleton Matrix / Vector.
    let res = &singleton_matrix / &vector;
    let vector_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *vector_s_matrix_res,
        array![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]
    );

    // Singleton Matrix / Vector forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0]],
    };
    forward_mat_res.div_fwd(&singleton_matrix, &vector);
    assert_eq!(forward_mat_res.data, *vector_s_matrix_res);

    // Matrix / Scalar.
    let res = &matrix / &scalar;
    let matrix_scalar_res: &Array2<f32> = &res.data;
    assert_eq!(*matrix_scalar_res, *scalar_matrix_res);

    // Matrix / Scalar forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.div_fwd(&matrix, &scalar);
    assert_eq!(forward_mat_res.data, *matrix_scalar_res);

    // Matrix / Vector.
    let res = &matrix / &vector;
    let matrix_vector_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_vector_res,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix / Vector forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.div_fwd(&matrix, &vector);
    assert_eq!(forward_mat_res.data, *matrix_vector_res);

    // Matrix / Matrix.
    let res = &matrix
        / &Tensor {
            data: array![[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]],
        };
    let matrix_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_matrix_res,
        array![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    );

    // Matrix / Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.div_fwd(
        &matrix,
        &Tensor {
            data: array![[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]],
        },
    );
    assert_eq!(forward_mat_res.data, *matrix_matrix_res);

    // Matrix / Matrix broadcast.
    let res = &matrix
        / &Tensor {
            data: array![[3.0, 6.0, 3.0]],
        };
    let matrix_matrix_b_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_matrix_b_res,
        array![[1.0, 0.5, 1.0], [1.0, 0.5, 1.0], [1.0, 0.5, 1.0]]
    );

    // Matrix / Matrix forward broadcast.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.div_fwd(
        &matrix,
        &Tensor {
            data: array![[3.0, 6.0, 3.0]],
        },
    );
    assert_eq!(forward_mat_res.data, *matrix_matrix_b_res);

    // Singleton Matrix / Matrix.
    let res = &singleton_matrix / &matrix;
    let matrix_s_matrix_res: &Array2<f32> = &res.data;
    assert_eq!(
        *matrix_s_matrix_res,
        array![
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        ]
    );

    // Singleton Matrix / Matrix forward.
    let mut forward_mat_res = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };
    forward_mat_res.div_fwd(&singleton_matrix, &matrix);
    assert_eq!(forward_mat_res.data, *matrix_s_matrix_res);
}

#[test]
fn assign_test() {
    let mut scalar_trgt = Tensor { data: array![0.0] };
    let mut vector_trgt = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    let mut matrix_trgt = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };

    let scalar = Tensor { data: array![1.0] };
    let vector = Tensor {
        data: array![1.0, 1.0, 1.0],
    };
    let matrix = Tensor {
        data: array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    };

    // Scalar scalar assignment.
    scalar_trgt.accumulate(&scalar, 1.0, &BackwardAction::Set);
    assert_eq!(scalar_trgt.data[0], scalar.data[0]);

    // Scalar scalar vector.
    scalar_trgt.accumulate(&vector, 1.0, &BackwardAction::Set);
    assert_eq!(scalar_trgt.data[0], 3.0);

    // Scalar scalar matrix.
    scalar_trgt.accumulate(&matrix, 1.0, &BackwardAction::Set);
    assert_eq!(scalar_trgt.data[0], 9.0);

    // Vector scalar assignment.
    vector_trgt.accumulate(&scalar, 1.0, &BackwardAction::Set);
    assert_eq!(vector_trgt.data, array![1.0, 1.0, 1.0]);

    // Vector vector assignment.
    vector_trgt.accumulate(&vector, 1.0, &BackwardAction::Set);
    assert_eq!(vector_trgt.data, array![1.0, 1.0, 1.0]);

    // Vector matrix assignment.
    vector_trgt.accumulate(&matrix, 1.0, &BackwardAction::Set);
    assert_eq!(vector_trgt.data, array![3.0, 3.0, 3.0]);

    // Matrix scalar assignment.
    matrix_trgt.accumulate(&scalar, 1.0, &BackwardAction::Set);
    assert_eq!(
        matrix_trgt.data,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix vector assignment.
    matrix_trgt.accumulate(&vector, 1.0, &BackwardAction::Set);
    assert_eq!(
        matrix_trgt.data,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix matrix assignment.
    matrix_trgt.accumulate(&matrix, 1.0, &BackwardAction::Set);
    assert_eq!(
        matrix_trgt.data,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );
}

#[test]
fn scaled_assign_test() {
    let mut scalar_trgt = Tensor { data: array![0.0] };
    let mut vector_trgt = Tensor {
        data: array![0.0, 0.0, 0.0],
    };
    let mut matrix_trgt = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };

    let scalar = Tensor { data: array![1.0] };
    let vector = Tensor {
        data: array![1.0, 1.0, 1.0],
    };
    let matrix = Tensor {
        data: array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    };

    // Scalar scalar assignment.
    scalar_trgt.accumulate(&scalar, -1.0, &BackwardAction::Set);
    assert_eq!(scalar_trgt.data[0], -scalar.data[0]);

    // Scalar scalar vector.
    scalar_trgt.accumulate(&vector, -1.0, &BackwardAction::Set);
    assert_eq!(scalar_trgt.data[0], -3.0);

    // Scalar scalar matrix.
    scalar_trgt.accumulate(&matrix, -1.0, &BackwardAction::Set);
    assert_eq!(scalar_trgt.data[0], -9.0);

    // Vector scalar assignment.
    vector_trgt.accumulate(&scalar, -1.0, &BackwardAction::Set);
    assert_eq!(vector_trgt.data, -array![1.0, 1.0, 1.0]);

    // Vector vector assignment.
    vector_trgt.accumulate(&vector, -1.0, &BackwardAction::Set);
    assert_eq!(vector_trgt.data, -array![1.0, 1.0, 1.0]);

    // Vector matrix assignment.
    vector_trgt.accumulate(&matrix, -1.0, &BackwardAction::Set);
    assert_eq!(vector_trgt.data, -array![3.0, 3.0, 3.0]);

    // Matrix scalar assignment.
    matrix_trgt.accumulate(&scalar, -1.0, &BackwardAction::Set);
    assert_eq!(
        matrix_trgt.data,
        -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix vector assignment.
    matrix_trgt.accumulate(&vector, -1.0, &BackwardAction::Set);
    assert_eq!(
        matrix_trgt.data,
        -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix matrix assignment.
    matrix_trgt.accumulate(&matrix, -1.0, &BackwardAction::Set);
    assert_eq!(
        matrix_trgt.data,
        -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );
}

#[test]
fn add_assign_test() {
    let mut scalar_trgt = Tensor { data: array![5.0] };
    let mut vector_trgt = Tensor {
        data: array![5.0, 5.0, 5.0],
    };
    let mut matrix_trgt = Tensor {
        data: array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
    };

    let scalar = Tensor { data: array![5.0] };
    let vector = Tensor {
        data: array![5.0, 5.0, 5.0],
    };
    let matrix = Tensor {
        data: array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
    };

    // Scalar scalar assignment.
    scalar_trgt.accumulate(&scalar, 1.0, &BackwardAction::Increment);
    assert_eq!(scalar_trgt.data[0], 10.0);

    // Scalar scalar vector.
    scalar_trgt.accumulate(&vector, 1.0, &BackwardAction::Increment);
    assert_eq!(scalar_trgt.data[0], 25.0);

    // Scalar scalar matrix.
    scalar_trgt.accumulate(&matrix, 1.0, &BackwardAction::Increment);
    assert_eq!(scalar_trgt.data[0], 70.0);

    // Vector scalar assignment.
    vector_trgt.accumulate(&scalar, 1.0, &BackwardAction::Increment);
    assert_eq!(vector_trgt.data, array![10.0, 10.0, 10.0]);

    // Vector vector assignment.
    vector_trgt.accumulate(&vector, 1.0, &BackwardAction::Increment);
    assert_eq!(vector_trgt.data, array![15.0, 15.0, 15.0]);

    // Vector matrix assignment.
    vector_trgt.accumulate(&matrix, 1.0, &BackwardAction::Increment);
    assert_eq!(vector_trgt.data, array![30.0, 30.0, 30.0]);

    // Matrix scalar assignment.
    matrix_trgt.accumulate(&scalar, 1.0, &BackwardAction::Increment);
    assert_eq!(
        matrix_trgt.data,
        array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    );

    // Matrix vector assignment.
    matrix_trgt.accumulate(&vector, 1.0, &BackwardAction::Increment);
    assert_eq!(
        matrix_trgt.data,
        array![[15.0, 15.0, 15.0], [15.0, 15.0, 15.0], [15.0, 15.0, 15.0]]
    );

    // Matrix matrix assignment.
    matrix_trgt.accumulate(&matrix, 1.0, &BackwardAction::Increment);
    assert_eq!(
        matrix_trgt.data,
        array![[20.0, 20.0, 20.0], [20.0, 20.0, 20.0], [20.0, 20.0, 20.0]]
    );
}

#[test]
fn scaled_add_assign_test() {
    let mut scalar_trgt = Tensor { data: array![5.0] };
    let mut vector_trgt = Tensor {
        data: array![5.0, 5.0, 5.0],
    };
    let mut matrix_trgt = Tensor {
        data: array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
    };

    let scalar = Tensor { data: array![5.0] };
    let vector = Tensor {
        data: array![5.0, 5.0, 5.0],
    };
    let matrix = Tensor {
        data: array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
    };

    // Scalar scalar assignment.
    &mut scalar_trgt.accumulate(&scalar, -1.0, &BackwardAction::Increment);
    assert_eq!(scalar_trgt.data[0], 0.0);
    scalar_trgt.set_zero();

    // Scalar scalar vector.
    &mut scalar_trgt.accumulate(&vector, -1.0, &BackwardAction::Increment);
    assert_eq!(scalar_trgt.data[0], -15.0);
    scalar_trgt.set_zero();

    // Scalar scalar matrix.
    &mut scalar_trgt.accumulate(&matrix, -1.0, &BackwardAction::Increment);
    assert_eq!(scalar_trgt.data[0], -45.0);
    scalar_trgt.set_zero();

    // Vector scalar assignment.
    &mut vector_trgt.accumulate(&scalar, -1.0, &BackwardAction::Increment);
    assert_eq!(vector_trgt.data, array![0.0, 0.0, 0.0]);
    vector_trgt.set_zero();

    // Vector vector assignment.
    &mut vector_trgt.accumulate(&vector, -1.0, &BackwardAction::Increment);
    assert_eq!(vector_trgt.data, array![-5.0, -5.0, -5.0]);
    vector_trgt.set_zero();

    // Vector matrix assignment.
    &mut vector_trgt.accumulate(&matrix, -1.0, &BackwardAction::Increment);
    assert_eq!(vector_trgt.data, array![-15.0, -15.0, -15.0]);
    vector_trgt.set_zero();

    // Matrix scalar assignment.
    &mut matrix_trgt.accumulate(&scalar, -1.0, &BackwardAction::Increment);
    assert_eq!(
        matrix_trgt.data,
        array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    );
    matrix_trgt.set_zero();

    // Matrix vector assignment.
    &mut matrix_trgt.accumulate(&vector, -1.0, &BackwardAction::Increment);
    assert_eq!(
        matrix_trgt.data,
        array![[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]
    );
    matrix_trgt.set_zero();

    // Matrix matrix assignment.
    &mut matrix_trgt.accumulate(&matrix, -1.0, &BackwardAction::Increment);
    assert_eq!(
        matrix_trgt.data,
        array![[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]
    );
    matrix_trgt.set_zero();
}

#[test]
fn mat_mat_mul_test() {
    // 2 rows 3 cols floating point matrix.
    let matrix_2x3 = Tensor {
        data: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    };
    // 3 rows 2 cols floating point matrix.
    let matrix_3x2 = Tensor {
        data: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    };
    // 2 rows 2 cols floating point matrix.
    let mut matrix_2x2 = Tensor {
        data: array![[0.0, 0.0], [0.0, 0.0]],
    };
    // 3 rows 3 cols floating point matrix.
    let mut matrix_3x3 = Tensor {
        data: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };

    // Output should be: 2x3 mul_fwd 3x2 -> 2x2.
    matrix_2x3.mat_mul(&matrix_3x2, &mut matrix_2x2, 1.0, 0.0, false, false);
    assert_eq!(matrix_2x2.data.dim(), (2, 2));

    // Output should be: 2x3 mult (2x3)^t -> 2x2.
    matrix_2x3.mat_mul(&matrix_2x3, &mut matrix_2x2, 1.0, 0.0, false, true);
    assert_eq!(matrix_2x2.data.dim(), (2, 2));

    // Output should be: (2x3)^t mult 3x2 -> 3x3.
    matrix_2x3.mat_mul(&matrix_2x3, &mut matrix_3x3, 1.0, 0.0, true, false);
    assert_eq!(matrix_3x3.data.dim(), (3, 3));
}

#[test]
fn mat_vec_mul_test() {
    // 3 rows 3 cols floating point matrix.
    let matrix_3x3 = Tensor {
        data: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    };
    // 3-dim vector.
    let vector = Tensor {
        data: array![1.0, 1.0, 1.0],
    };
    // 3-dim vector.
    let mut res = Tensor {
        data: array![0.0, 0.0, 0.0],
    };

    matrix_3x3.mat_vec_mul(&vector, &mut res, 1.0, 0.0, false);
    assert_eq!(res.data, array![6.0, 15.0, 24.0]);
}
