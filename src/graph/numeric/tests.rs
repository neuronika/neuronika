use super::Tensor;
use ndarray::prelude::array;
use ndarray::{Array1, Array2, Axis};

//TODO: This tests really should be integrated into the tests folder.

#[test]
fn add_test() {
    let scalar = Tensor { array: array![1.0] };
    let vector = Tensor {
        array: array![1.0, 2.0, 3.0],
    };
    let matrix = Tensor {
        array: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    };
    let singleton_matrix = Tensor {
        array: array![[1.0]],
    };

    // Scalar + Scalar.
    let scalar_res: &Array1<f32> = &(&scalar + &Tensor { array: array![9.0] }).array;
    assert_eq!(scalar_res[0], 10.0);

    // Scalar + Vector.
    let res = &scalar + &vector;
    let scalar_vector_res: &Array1<f32> = &res.array;
    assert_eq!(*scalar_vector_res, array![2.0, 3.0, 4.0]);

    // Scalar + Matrix.
    let res = &scalar + &matrix;
    let scalar_matrix_res: &Array2<f32> = &&res.array;
    assert_eq!(
        *scalar_matrix_res,
        array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    );

    // Vector + Scalar.
    let res = &vector + &scalar;
    let vector_scalar_res: &Array1<f32> = &res.array;
    assert_eq!(*vector_scalar_res, *scalar_vector_res);

    // Vector + Vector.
    let res = &vector
        + &Tensor {
            array: array![3.0, 2.0, 1.0],
        };
    let vector_vector_res: &Array1<f32> = &res.array;
    //println!("{:?}", vector_vector_res);
    assert_eq!(*vector_vector_res, array![4.0, 4.0, 4.0]);

    // Vector + Matrix
    let res = &vector + &matrix;
    let vector_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *vector_matrix_res,
        array![[2.0, 4.0, 6.0], [5.0, 7.0, 9.0], [8.0, 10.0, 12.0]]
    );

    // Vector + singleton Matrix.
    let res = &vector + &singleton_matrix;
    let vector_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(*vector_s_matrix_res, array![[2.0, 3.0, 4.0]]);

    // Singleton Matrix + Vector.
    let res = &singleton_matrix + &vector;
    let vector_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(*vector_s_matrix_res, array![[2.0, 3.0, 4.0]]);

    // Matrix + Scalar.
    let res = &matrix + &scalar;
    let matrix_scalar_res: &Array2<f32> = &res.array;
    assert_eq!(*matrix_scalar_res, *scalar_matrix_res);

    // Matrix + Vector.
    let res = &matrix + &vector;
    let matrix_vector_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_vector_res,
        array![[2.0, 4.0, 6.0], [5.0, 7.0, 9.0], [8.0, 10.0, 12.0]]
    );

    // Matrix + Matrix.
    let res = &matrix
        + &Tensor {
            array: array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        };
    let matrix_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_matrix_res,
        array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    );

    // Matrix + Matrix broadcast.
    let res = &matrix
        + &Tensor {
            array: array![[1.0, 2.0, 3.0]],
        };
    let matrix_matrix_b_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_matrix_b_res,
        array![[2.0, 4.0, 6.0], [5.0, 7.0, 9.0], [8.0, 10.0, 12.0]]
    );

    // Singleton Matrix + Matrix.
    let res = &singleton_matrix + &matrix;
    let matrix_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_s_matrix_res,
        array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    );
}

#[test]
fn sub_test() {
    let scalar = Tensor { array: array![1.0] };
    let vector = Tensor {
        array: array![1.0, 2.0, 3.0],
    };
    let matrix = Tensor {
        array: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    };
    let singleton_matrix = Tensor {
        array: array![[1.0]],
    };

    // Scalar - Scalar.
    let scalar_res: Array1<f32> = (&scalar - &Tensor { array: array![1.0] }).array;
    assert_eq!(scalar_res[0], 0.0);

    // Scalar - Vector.
    let res = &scalar - &vector;
    let scalar_vector_res: &Array1<f32> = &res.array;
    assert_eq!(*scalar_vector_res, array![0.0, -1.0, -2.0]);

    // Scalar - Matrix.
    let res = &scalar - &matrix;
    let scalar_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *scalar_matrix_res,
        array![[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0], [-6.0, -7.0, -8.0]]
    );

    // Vector - Scalar.
    let res = &vector - &scalar;
    let vector_scalar_res: &Array1<f32> = &res.array;
    assert_eq!(*vector_scalar_res, -scalar_vector_res);

    // Vector - Vector.
    let res = &vector
        - &Tensor {
            array: array![3.0, 2.0, 1.0],
        };
    let vector_vector_res: &Array1<f32> = &res.array;
    assert_eq!(*vector_vector_res, array![-2.0, 0.0, 2.0]);

    // Vector - Matrix
    let res = &vector - &matrix;
    let vector_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *vector_matrix_res,
        array![[0.0, 0.0, 0.0], [-3.0, -3.0, -3.0], [-6.0, -6.0, -6.0]]
    );

    // Vector - singleton Matrix.
    let res = &vector - &singleton_matrix;
    let vector_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(*vector_s_matrix_res, array![[0.0, 1.0, 2.0]]);

    // Singleton Matrix - Vector.
    let res = &singleton_matrix - &vector;
    let vector_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(*vector_s_matrix_res, array![[0.0, -1.0, -2.0]]);

    // Matrix - Scalar.
    let res = &matrix - &scalar;
    let matrix_scalar_res: &Array2<f32> = &res.array;
    assert_eq!(*matrix_scalar_res, -scalar_matrix_res);

    // Matrix - Vector.
    let res = &matrix - &vector;
    let matrix_vector_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_vector_res,
        array![[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]]
    );

    // Matrix - Matrix.
    let res = &matrix
        - &Tensor {
            array: array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        };
    let matrix_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_matrix_res,
        array![[-8.0, -6.0, -4.0], [-2.0, 0.0, 2.0], [4.0, 6.0, 8.0]]
    );

    // Matrix - Matrix broadcast.
    let res = &matrix
        - &Tensor {
            array: array![[1.0, 2.0, 3.0]],
        };
    let matrix_matrix_b_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_matrix_b_res,
        array![[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]]
    );

    // Singleton Matrix - Matrix.
    let res = &singleton_matrix - &matrix;
    let matrix_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_s_matrix_res,
        -array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    );
}

#[test]
fn mul_test() {
    let scalar = Tensor { array: array![3.0] };
    let vector = Tensor {
        array: array![1.0, 2.0, 3.0],
    };
    let matrix = Tensor {
        array: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    };
    let singleton_matrix = Tensor {
        array: array![[1.0]],
    };

    // Scalar * Scalar.
    let scalar_res: Array1<f32> = (&scalar * &Tensor { array: array![-3.] }).array;
    assert_eq!(scalar_res[0], -9.0);

    // Scalar * Vector.
    let res = &scalar * &vector;
    let scalar_vector_res: &Array1<f32> = &res.array;
    assert_eq!(*scalar_vector_res, array![3.0, 6.0, 9.0]);

    // Scalar * Matrix.
    let res = &scalar * &matrix;
    let scalar_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *scalar_matrix_res,
        array![[3.0, 6.0, 9.0], [12.0, 15.0, 18.0], [21.0, 24.0, 27.0]]
    );

    // Vector * Scalar.
    let res = &vector * &scalar;
    let vector_scalar_res: &Array1<f32> = &res.array;
    assert_eq!(*vector_scalar_res, *scalar_vector_res);

    // Vector * Vector.
    let res = &vector
        * &Tensor {
            array: array![3.0, 2.0, 1.0],
        };
    let vector_vector_res: &Array1<f32> = &res.array;
    assert_eq!(*vector_vector_res, array![3.0, 4.0, 3.0]);

    // Vector * Matrix
    let res = &vector * &matrix;
    let vector_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *vector_matrix_res,
        array![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0], [7.0, 16.0, 27.0]]
    );

    // Vector * singleton Matrix.
    let res = &vector * &singleton_matrix;
    let vector_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(*vector_s_matrix_res, array![[1.0, 2.0, 3.0]]);

    // Singleton Matrix * Vector.
    let res = &singleton_matrix * &vector;
    let vector_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(*vector_s_matrix_res, array![[1.0, 2.0, 3.0]]);

    // Matrix * Scalar.
    let res = &matrix * &scalar;
    let matrix_scalar_res: &Array2<f32> = &res.array;
    assert_eq!(*matrix_scalar_res, *scalar_matrix_res);

    // Matrix * Vector.
    let res = &matrix * &vector;
    let matrix_vector_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_vector_res,
        array![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0], [7.0, 16.0, 27.0]]
    );

    // Matrix * Matrix.
    let res = &matrix
        * &Tensor {
            array: array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        };
    let matrix_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_matrix_res,
        array![[9.0, 16.0, 21.0], [24.0, 25.0, 24.0], [21.0, 16.0, 9.0]]
    );

    // Matrix * Matrix broadcast.
    let res = &matrix
        * &Tensor {
            array: array![[1.0, 2.0, 3.0]],
        };
    let matrix_matrix_b_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_matrix_b_res,
        array![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0], [7.0, 16.0, 27.0]]
    );

    // Singleton Matrix * Matrix.
    let res = &singleton_matrix * &matrix;
    let matrix_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_s_matrix_res,
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    );
}

#[test]
fn div_test() {
    let scalar = Tensor { array: array![3.0] };
    let vector = Tensor {
        array: array![3.0, 3.0, 3.0],
    };
    let matrix = Tensor {
        array: array![[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]],
    };
    let singleton_matrix = Tensor {
        array: array![[1.0]],
    };

    // Scalar / Scalar.
    let scalar_res: Array1<f32> = (&scalar / &Tensor { array: array![-3.] }).array;
    assert_eq!(scalar_res[0], -1.0);

    // Scalar / Vector.
    let res = &scalar / &vector;
    let scalar_vector_res: &Array1<f32> = &res.array;
    assert_eq!(*scalar_vector_res, array![1.0, 1.0, 1.0]);

    // Scalar / Matrix.
    let res = &scalar / &matrix;
    let scalar_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *scalar_matrix_res,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Vector / Scalar.
    let res = &vector / &scalar;
    let vector_scalar_res: &Array1<f32> = &res.array;
    assert_eq!(*vector_scalar_res, *scalar_vector_res);

    // Vector / Vector.
    let res = &vector
        / &Tensor {
            array: array![1.0, 3.0, 6.0],
        };
    let vector_vector_res: &Array1<f32> = &res.array;
    assert_eq!(*vector_vector_res, array![3.0, 1.0, 0.5]);

    // Vector / Matrix
    let res = &vector / &matrix;
    let vector_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *vector_matrix_res,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Vector / singleton Matrix.
    let res = &vector / &singleton_matrix;
    let vector_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(*vector_s_matrix_res, array![[3.0, 3.0, 3.0]]);

    // Singleton Matrix / Vector.
    let res = &singleton_matrix / &vector;
    let vector_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *vector_s_matrix_res,
        array![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]
    );

    // Matrix / Scalar.
    let res = &matrix / &scalar;
    let matrix_scalar_res: &Array2<f32> = &res.array;
    assert_eq!(*matrix_scalar_res, *scalar_matrix_res);

    // Matrix / Vector.
    let res = &matrix / &vector;
    let matrix_vector_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_vector_res,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix / Matrix.
    let res = &matrix
        / &Tensor {
            array: array![[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]],
        };
    let matrix_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_matrix_res,
        array![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    );

    // Matrix / Matrix broadcast.
    let res = &matrix
        / &Tensor {
            array: array![[3.0, 6.0, 3.0]],
        };
    let matrix_matrix_b_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_matrix_b_res,
        array![[1.0, 0.5, 1.0], [1.0, 0.5, 1.0], [1.0, 0.5, 1.0]]
    );

    // Singleton Matrix / Matrix.
    let res = &singleton_matrix / &matrix;
    let matrix_s_matrix_res: &Array2<f32> = &res.array;
    assert_eq!(
        *matrix_s_matrix_res,
        array![
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        ]
    );
}

#[test]
fn concatenations() {
    let fst = Tensor::zeros(3);
    let snd = Tensor::zeros(2);
    assert_eq!(Tensor::concatenate(Axis(0), &[fst, snd]), Tensor::zeros(5));

    let fst = Tensor::zeros((3, 2, 1));
    let snd = Tensor::zeros((3, 2, 3));
    assert_eq!(
        Tensor::concatenate(Axis(2), &[fst, snd]),
        Tensor::zeros((3, 2, 4))
    );

    let fst = Tensor::zeros((3, 2, 1));
    let snd = Tensor::zeros((3, 8, 1));
    assert_eq!(
        Tensor::concatenate(Axis(1), &[fst, snd]),
        Tensor::zeros((3, 10, 1))
    );
}

#[test]
fn stackings() {
    let fst = Tensor::zeros(3);
    let snd = Tensor::zeros(3);
    assert_eq!(Tensor::stack(&[fst, snd]), Tensor::zeros((2, 3)));

    let fst = Tensor::zeros((3, 2, 3));
    let snd = Tensor::zeros((3, 2, 3));
    assert_eq!(Tensor::stack(&[fst, snd]), Tensor::zeros((2, 3, 2, 3)));

    let fst = Tensor::zeros((3, 2, 1));
    let snd = Tensor::zeros((3, 2, 1));
    assert_eq!(Tensor::stack(&[fst, snd]), Tensor::zeros((2, 3, 2, 1)));
}

#[test]
fn mat_mat_mul_test() {
    // 2 rows 3 cols floating point matrix.
    let matrix_2x3 = Tensor {
        array: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    };
    // 3 rows 2 cols floating point matrix.
    let matrix_3x2 = Tensor {
        array: array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    };
    // 2 rows 2 cols floating point matrix.
    let mut matrix_2x2 = Tensor {
        array: array![[0.0, 0.0], [0.0, 0.0]],
    };
    // 3 rows 3 cols floating point matrix.
    let mut matrix_3x3 = Tensor {
        array: array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    };

    // Output should be: 2x3 mul_fwd 3x2 -> 2x2.
    matrix_2x3.mat_mul(&matrix_3x2, &mut matrix_2x2, 1.0, 0.0, false, false);
    assert_eq!(matrix_2x2.array.dim(), (2, 2));

    // Output should be: 2x3 mult (2x3)^t -> 2x2.
    matrix_2x3.mat_mul(&matrix_2x3, &mut matrix_2x2, 1.0, 0.0, false, true);
    assert_eq!(matrix_2x2.array.dim(), (2, 2));

    // Output should be: (2x3)^t mult 3x2 -> 3x3.
    matrix_2x3.mat_mul(&matrix_2x3, &mut matrix_3x3, 1.0, 0.0, true, false);
    assert_eq!(matrix_3x3.array.dim(), (3, 3));
}

#[test]
fn mat_vec_mul_test() {
    // 3 rows 3 cols floating point matrix.
    let matrix_3x3 = Tensor {
        array: array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    };
    // 3-dim vector.
    let vector = Tensor {
        array: array![1.0, 1.0, 1.0],
    };
    // 3-dim vector.
    let mut res = Tensor {
        array: array![0.0, 0.0, 0.0],
    };

    matrix_3x3.mat_vec_mul(&vector, &mut res, 1.0, 0.0, false);
    assert_eq!(res.array, array![6.0, 15.0, 24.0]);
}
