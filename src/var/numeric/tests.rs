use super::{
    add, add_assign, assign, div, div_assign_pow, mat_mat_mul, mat_vec_mul, mul, scaled_add_assign,
    scaled_assign, sub,
};
use super::{DataRepr, Matrix, Vector};
use ndarray::prelude::array;

#[test]
fn add_test() {
    let scalar = DataRepr::Scalar(1.0);
    let vector = DataRepr::Vector(array![1.0, 2.0, 3.0]);
    let matrix = DataRepr::Matrix(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let singleton_matrix = DataRepr::Matrix(array![[1.0]]);

    // Scalar + Scalar.
    let scalar_res: f32 = (&scalar + &DataRepr::Scalar(9.0)).scalar();
    assert_eq!(scalar_res, 10.0);

    // Scalar + Scalar forward.
    let mut forward_scalar_res = DataRepr::Scalar(0.0);
    add(&mut forward_scalar_res, &scalar, &DataRepr::Scalar(9.0));
    assert_eq!(forward_scalar_res.scalar(), scalar_res);

    // Scalar + Vector.
    let res = &scalar + &vector;
    let scalar_vector_res: &Vector = (res).vector();
    assert_eq!(*scalar_vector_res, array![2.0, 3.0, 4.0]);

    // Scalar + Vector forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    add(&mut forward_vec_res, &scalar, &vector);
    assert_eq!(*forward_vec_res.vector(), *scalar_vector_res);

    // Scalar + Matrix.
    let res = &scalar + &matrix;
    let scalar_matrix_res: &Matrix = (res).matrix();
    assert_eq!(
        *scalar_matrix_res,
        array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    );

    // Scalar + Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    add(&mut forward_mat_res, &scalar, &matrix);
    assert_eq!(*forward_mat_res.matrix(), *scalar_matrix_res);

    // Vector + Scalar.
    let res = &vector + &scalar;
    let vector_scalar_res: &Vector = (res).vector();
    assert_eq!(*vector_scalar_res, *scalar_vector_res);

    // Vector + Scalar forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    add(&mut forward_vec_res, &vector, &scalar);
    assert_eq!(*forward_vec_res.vector(), *vector_scalar_res);

    // Vector + Vector.
    let res = &vector + &DataRepr::Vector(array![3.0, 2.0, 1.0]);
    let vector_vector_res: &Vector = res.vector();
    assert_eq!(*vector_vector_res, array![4.0, 4.0, 4.0]);

    // Vector + Vector forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    add(
        &mut forward_vec_res,
        &vector,
        &DataRepr::Vector(array![3.0, 2.0, 1.0]),
    );
    assert_eq!(*forward_vec_res.vector(), *vector_vector_res);

    // Vector + Matrix
    let res = &vector + &matrix;
    let vector_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *vector_matrix_res,
        array![[2.0, 4.0, 6.0], [5.0, 7.0, 9.0], [8.0, 10.0, 12.0]]
    );

    // Vector + Matrix forward.
    let mut forward_vec_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    add(&mut forward_vec_res, &vector, &matrix);
    assert_eq!(*forward_vec_res.matrix(), *vector_matrix_res);

    // Vector + singleton Matrix.
    let res = &vector + &singleton_matrix;
    let vector_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(*vector_s_matrix_res, array![[2.0, 3.0, 4.0]]);

    // Vector + singleton Matrix forward.
    let mut forward_mat_res = DataRepr::Matrix(array![[0.0, 0.0, 0.0]]);
    add(&mut forward_mat_res, &vector, &singleton_matrix);
    assert_eq!(*forward_mat_res.matrix(), *vector_s_matrix_res);

    // Singleton Matrix + Vector.
    let res = &singleton_matrix + &vector;
    let vector_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(*vector_s_matrix_res, array![[2.0, 3.0, 4.0]]);

    // Singleton Matrix + Vector forward.
    let mut forward_mat_res = DataRepr::Matrix(array![[0.0, 0.0, 0.0]]);
    add(&mut forward_mat_res, &singleton_matrix, &vector);
    assert_eq!(*forward_mat_res.matrix(), *vector_s_matrix_res);

    // Matrix + Scalar.
    let res = &matrix + &scalar;
    let matrix_scalar_res: &Matrix = res.matrix();
    assert_eq!(*matrix_scalar_res, *scalar_matrix_res);

    // Matrix + Scalar forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    add(&mut forward_mat_res, &matrix, &scalar);
    assert_eq!(*forward_mat_res.matrix(), *matrix_scalar_res);

    // Matrix + Vector.
    let res = &matrix + &vector;
    let matrix_vector_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_vector_res,
        array![[2.0, 4.0, 6.0], [5.0, 7.0, 9.0], [8.0, 10.0, 12.0]]
    );

    // Matrix + Vector forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    add(&mut forward_mat_res, &matrix, &vector);
    assert_eq!(*forward_mat_res.matrix(), *matrix_vector_res);

    // Matrix + Matrix.
    let res =
        &matrix + &DataRepr::Matrix(array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    let matrix_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_matrix_res,
        array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    );

    // Matrix + Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    add(
        &mut forward_mat_res,
        &matrix,
        &DataRepr::Matrix(array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
    );
    assert_eq!(*forward_mat_res.matrix(), *matrix_matrix_res);

    // Matrix + Matrix broadcast.
    let res = &matrix + &DataRepr::Matrix(array![[1.0, 2.0, 3.0]]);
    let matrix_matrix_b_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_matrix_b_res,
        array![[2.0, 4.0, 6.0], [5.0, 7.0, 9.0], [8.0, 10.0, 12.0]]
    );

    // Matrix + Matrix forward broadcast.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    add(
        &mut forward_mat_res,
        &matrix,
        &DataRepr::Matrix(array![[1.0, 2.0, 3.0]]),
    );
    assert_eq!(*forward_mat_res.matrix(), *matrix_matrix_b_res);

    // Singleton Matrix + Matrix.
    let res = &singleton_matrix + &matrix;
    let matrix_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_s_matrix_res,
        array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]
    );

    // Singleton Matrix + Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    add(&mut forward_mat_res, &singleton_matrix, &matrix);
    assert_eq!(*forward_mat_res.matrix(), *matrix_s_matrix_res);
}

#[test]
fn sub_test() {
    let scalar = DataRepr::Scalar(1.0);
    let vector = DataRepr::Vector(array![1.0, 2.0, 3.0]);
    let matrix = DataRepr::Matrix(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let singleton_matrix = DataRepr::Matrix(array![[1.0]]);

    // Scalar - Scalar.
    let scalar_res: f32 = (&scalar - &DataRepr::Scalar(1.0)).scalar();
    assert_eq!(scalar_res, 0.0);

    // Scalar - Scalar forward.
    let mut forward_scalar_res = DataRepr::Scalar(0.0);
    sub(&mut forward_scalar_res, &scalar, &DataRepr::Scalar(1.0));
    assert_eq!(forward_scalar_res.scalar(), scalar_res);

    // Scalar - Vector.
    let res = &scalar - &vector;
    let scalar_vector_res: &Vector = (res).vector();
    assert_eq!(*scalar_vector_res, array![0.0, -1.0, -2.0]);

    // Scalar - Vector forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    sub(&mut forward_vec_res, &scalar, &vector);
    assert_eq!(*forward_vec_res.vector(), *scalar_vector_res);

    // Scalar - Matrix.
    let res = &scalar - &matrix;
    let scalar_matrix_res: &Matrix = (res).matrix();
    assert_eq!(
        *scalar_matrix_res,
        array![[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0], [-6.0, -7.0, -8.0]]
    );

    // Scalar - Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    sub(&mut forward_mat_res, &scalar, &matrix);
    assert_eq!(*forward_mat_res.matrix(), *scalar_matrix_res);

    // Vector - Scalar.
    let res = &vector - &scalar;
    let vector_scalar_res: &Vector = (res).vector();
    assert_eq!(*vector_scalar_res, -scalar_vector_res);

    // Vector - Scalar forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    sub(&mut forward_vec_res, &vector, &scalar);
    assert_eq!(*forward_vec_res.vector(), *vector_scalar_res);

    // Vector - Vector.
    let res = &vector - &DataRepr::Vector(array![3.0, 2.0, 1.0]);
    let vector_vector_res: &Vector = res.vector();
    assert_eq!(*vector_vector_res, array![-2.0, 0.0, 2.0]);

    // Vector - Vector forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    sub(
        &mut forward_vec_res,
        &vector,
        &DataRepr::Vector(array![3.0, 2.0, 1.0]),
    );
    assert_eq!(*forward_vec_res.vector(), *vector_vector_res);

    // Vector - Matrix
    let res = &vector - &matrix;
    let vector_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *vector_matrix_res,
        array![[0.0, 0.0, 0.0], [-3.0, -3.0, -3.0], [-6.0, -6.0, -6.0]]
    );

    // Vector - Matrix forward.
    let mut forward_vec_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    sub(&mut forward_vec_res, &vector, &matrix);
    assert_eq!(*forward_vec_res.matrix(), *vector_matrix_res);

    // Vector - singleton Matrix.
    let res = &vector - &singleton_matrix;
    let vector_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(*vector_s_matrix_res, array![[0.0, 1.0, 2.0]]);

    // Vector - singleton Matrix forward.
    let mut forward_mat_res = DataRepr::Matrix(array![[0.0, 0.0, 0.0]]);
    sub(&mut forward_mat_res, &vector, &singleton_matrix);
    assert_eq!(*forward_mat_res.matrix(), *vector_s_matrix_res);

    // Singleton Matrix - Vector.
    let res = &singleton_matrix - &vector;
    let vector_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(*vector_s_matrix_res, array![[0.0, -1.0, -2.0]]);

    // Singleton Matrix - Vector forward.
    let mut forward_mat_res = DataRepr::Matrix(array![[0.0, 0.0, 0.0]]);
    sub(&mut forward_mat_res, &singleton_matrix, &vector);
    assert_eq!(*forward_mat_res.matrix(), *vector_s_matrix_res);

    // Matrix - Scalar.
    let res = &matrix - &scalar;
    let matrix_scalar_res: &Matrix = res.matrix();
    assert_eq!(*matrix_scalar_res, -scalar_matrix_res);

    // Matrix - Scalar forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    sub(&mut forward_mat_res, &matrix, &scalar);
    assert_eq!(*forward_mat_res.matrix(), *matrix_scalar_res);

    // Matrix - Vector.
    let res = &matrix - &vector;
    let matrix_vector_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_vector_res,
        array![[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]]
    );

    // Matrix - Vector forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    sub(&mut forward_mat_res, &matrix, &vector);
    assert_eq!(*forward_mat_res.matrix(), *matrix_vector_res);

    // Matrix - Matrix.
    let res =
        &matrix - &DataRepr::Matrix(array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    let matrix_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_matrix_res,
        array![[-8.0, -6.0, -4.0], [-2.0, 0.0, 2.0], [4.0, 6.0, 8.0]]
    );

    // Matrix - Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    sub(
        &mut forward_mat_res,
        &matrix,
        &DataRepr::Matrix(array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
    );
    assert_eq!(*forward_mat_res.matrix(), *matrix_matrix_res);

    // Matrix - Matrix broadcast.
    let res = &matrix - &DataRepr::Matrix(array![[1.0, 2.0, 3.0]]);
    let matrix_matrix_b_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_matrix_b_res,
        array![[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [6.0, 6.0, 6.0]]
    );

    // Matrix - Matrix forward broadcast.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    sub(
        &mut forward_mat_res,
        &matrix,
        &DataRepr::Matrix(array![[1.0, 2.0, 3.0]]),
    );
    assert_eq!(*forward_mat_res.matrix(), *matrix_matrix_b_res);

    // Singleton Matrix - Matrix.
    let res = &singleton_matrix - &matrix;
    let matrix_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_s_matrix_res,
        -array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    );

    // Singleton Matrix - Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    sub(&mut forward_mat_res, &singleton_matrix, &matrix);
    assert_eq!(*forward_mat_res.matrix(), *matrix_s_matrix_res);
}

#[test]
fn mul_test() {
    let scalar = DataRepr::Scalar(3.0);
    let vector = DataRepr::Vector(array![1.0, 2.0, 3.0]);
    let matrix = DataRepr::Matrix(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let singleton_matrix = DataRepr::Matrix(array![[1.0]]);

    // Scalar * Scalar.
    let scalar_res: f32 = (&scalar * &DataRepr::Scalar(-3.0)).scalar();
    assert_eq!(scalar_res, -9.0);

    // Scalar * Scalar forward.
    let mut forward_scalar_res = DataRepr::Scalar(0.0);
    mul(&mut forward_scalar_res, &scalar, &DataRepr::Scalar(-3.0));
    assert_eq!(forward_scalar_res.scalar(), scalar_res);

    // Scalar * Vector.
    let res = &scalar * &vector;
    let scalar_vector_res: &Vector = (res).vector();
    assert_eq!(*scalar_vector_res, array![3.0, 6.0, 9.0]);

    // Scalar * Vector forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    mul(&mut forward_vec_res, &scalar, &vector);
    assert_eq!(*forward_vec_res.vector(), *scalar_vector_res);

    // Scalar * Matrix.
    let res = &scalar * &matrix;
    let scalar_matrix_res: &Matrix = (res).matrix();
    assert_eq!(
        *scalar_matrix_res,
        array![[3.0, 6.0, 9.0], [12.0, 15.0, 18.0], [21.0, 24.0, 27.0]]
    );

    // Scalar * Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    mul(&mut forward_mat_res, &scalar, &matrix);
    assert_eq!(*forward_mat_res.matrix(), *scalar_matrix_res);

    // Vector * Scalar.
    let res = &vector * &scalar;
    let vector_scalar_res: &Vector = (res).vector();
    assert_eq!(*vector_scalar_res, *scalar_vector_res);

    // Vector * Scalar forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    mul(&mut forward_vec_res, &vector, &scalar);
    assert_eq!(*forward_vec_res.vector(), *vector_scalar_res);

    // Vector * Vector.
    let res = &vector * &DataRepr::Vector(array![3.0, 2.0, 1.0]);
    let vector_vector_res: &Vector = res.vector();
    assert_eq!(*vector_vector_res, array![3.0, 4.0, 3.0]);

    // Vector * Vector forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    mul(
        &mut forward_vec_res,
        &vector,
        &DataRepr::Vector(array![3.0, 2.0, 1.0]),
    );
    assert_eq!(*forward_vec_res.vector(), *vector_vector_res);

    // Vector * Matrix
    let res = &vector * &matrix;
    let vector_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *vector_matrix_res,
        array![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0], [7.0, 16.0, 27.0]]
    );

    // Vector * Matrix forward.
    let mut forward_vec_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    mul(&mut forward_vec_res, &vector, &matrix);
    assert_eq!(*forward_vec_res.matrix(), *vector_matrix_res);

    // Vector * singleton Matrix.
    let res = &vector * &singleton_matrix;
    let vector_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(*vector_s_matrix_res, array![[1.0, 2.0, 3.0]]);

    // Vector * singleton Matrix forward.
    let mut forward_mat_res = DataRepr::Matrix(array![[0.0, 0.0, 0.0]]);
    mul(&mut forward_mat_res, &vector, &singleton_matrix);
    assert_eq!(*forward_mat_res.matrix(), *vector_s_matrix_res);

    // Singleton Matrix * Vector.
    let res = &singleton_matrix * &vector;
    let vector_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(*vector_s_matrix_res, array![[1.0, 2.0, 3.0]]);

    // Singleton Matrix * Vector forward.
    let mut forward_mat_res = DataRepr::Matrix(array![[0.0, 0.0, 0.0]]);
    mul(&mut forward_mat_res, &singleton_matrix, &vector);
    assert_eq!(*forward_mat_res.matrix(), *vector_s_matrix_res);

    // Matrix * Scalar.
    let res = &matrix * &scalar;
    let matrix_scalar_res: &Matrix = res.matrix();
    assert_eq!(*matrix_scalar_res, *scalar_matrix_res);

    // Matrix * Scalar forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    mul(&mut forward_mat_res, &matrix, &scalar);
    assert_eq!(*forward_mat_res.matrix(), *matrix_scalar_res);

    // Matrix * Vector.
    let res = &matrix * &vector;
    let matrix_vector_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_vector_res,
        array![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0], [7.0, 16.0, 27.0]]
    );

    // Matrix * Vector forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    mul(&mut forward_mat_res, &matrix, &vector);
    assert_eq!(*forward_mat_res.matrix(), *matrix_vector_res);

    // Matrix * Matrix.
    let res =
        &matrix * &DataRepr::Matrix(array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    let matrix_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_matrix_res,
        array![[9.0, 16.0, 21.0], [24.0, 25.0, 24.0], [21.0, 16.0, 9.0]]
    );

    // Matrix * Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    mul(
        &mut forward_mat_res,
        &matrix,
        &DataRepr::Matrix(array![[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
    );
    assert_eq!(*forward_mat_res.matrix(), *matrix_matrix_res);

    // Matrix * Matrix broadcast.
    let res = &matrix * &DataRepr::Matrix(array![[1.0, 2.0, 3.0]]);
    let matrix_matrix_b_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_matrix_b_res,
        array![[1.0, 4.0, 9.0], [4.0, 10.0, 18.0], [7.0, 16.0, 27.0]]
    );

    // Matrix * Matrix forward broadcast.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    mul(
        &mut forward_mat_res,
        &matrix,
        &DataRepr::Matrix(array![[1.0, 2.0, 3.0]]),
    );
    assert_eq!(*forward_mat_res.matrix(), *matrix_matrix_b_res);

    // Singleton Matrix * Matrix.
    let res = &singleton_matrix * &matrix;
    let matrix_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_s_matrix_res,
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    );

    // Singleton Matrix * Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    mul(&mut forward_mat_res, &singleton_matrix, &matrix);
    assert_eq!(*forward_mat_res.matrix(), *matrix_s_matrix_res);
}

#[test]
fn div_test() {
    let scalar = DataRepr::Scalar(3.0);
    let vector = DataRepr::Vector(array![3.0, 3.0, 3.0]);
    let matrix = DataRepr::Matrix(array![[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]);
    let singleton_matrix = DataRepr::Matrix(array![[1.0]]);

    // Scalar / Scalar.
    let scalar_res: f32 = (&scalar / &DataRepr::Scalar(-3.0)).scalar();
    assert_eq!(scalar_res, -1.0);

    // Scalar / Scalar forward.
    let mut forward_scalar_res = DataRepr::Scalar(0.0);
    div(&mut forward_scalar_res, &scalar, &DataRepr::Scalar(-3.0));
    assert_eq!(forward_scalar_res.scalar(), scalar_res);

    // Scalar / Vector.
    let res = &scalar / &vector;
    let scalar_vector_res: &Vector = (res).vector();
    assert_eq!(*scalar_vector_res, array![1.0, 1.0, 1.0]);

    // Scalar / Vector forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    div(&mut forward_vec_res, &scalar, &vector);
    assert_eq!(*forward_vec_res.vector(), *scalar_vector_res);

    // Scalar / Matrix.
    let res = &scalar / &matrix;
    let scalar_matrix_res: &Matrix = (res).matrix();
    assert_eq!(
        *scalar_matrix_res,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Scalar / Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    div(&mut forward_mat_res, &scalar, &matrix);
    assert_eq!(*forward_mat_res.matrix(), *scalar_matrix_res);

    // Vector / Scalar.
    let res = &vector / &scalar;
    let vector_scalar_res: &Vector = (res).vector();
    assert_eq!(*vector_scalar_res, *scalar_vector_res);

    // Vector / Scalar forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    div(&mut forward_vec_res, &vector, &scalar);
    assert_eq!(*forward_vec_res.vector(), *vector_scalar_res);

    // Vector / Vector.
    let res = &vector / &DataRepr::Vector(array![1.0, 3.0, 6.0]);
    let vector_vector_res: &Vector = res.vector();
    assert_eq!(*vector_vector_res, array![3.0, 1.0, 0.5]);

    // Vector / Vector forward.
    let mut forward_vec_res = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    div(
        &mut forward_vec_res,
        &vector,
        &DataRepr::Vector(array![1.0, 3.0, 6.0]),
    );
    assert_eq!(*forward_vec_res.vector(), *vector_vector_res);

    // Vector / Matrix
    let res = &vector / &matrix;
    let vector_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *vector_matrix_res,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Vector / Matrix forward.
    let mut forward_vec_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    div(&mut forward_vec_res, &vector, &matrix);
    assert_eq!(*forward_vec_res.matrix(), *vector_matrix_res);

    // Vector / singleton Matrix.
    let res = &vector / &singleton_matrix;
    let vector_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(*vector_s_matrix_res, array![[3.0, 3.0, 3.0]]);

    // Vector / singleton Matrix forward.
    let mut forward_mat_res = DataRepr::Matrix(array![[0.0, 0.0, 0.0]]);
    div(&mut forward_mat_res, &vector, &singleton_matrix);
    assert_eq!(*forward_mat_res.matrix(), *vector_s_matrix_res);

    // Singleton Matrix / Vector.
    let res = &singleton_matrix / &vector;
    let vector_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *vector_s_matrix_res,
        array![[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]
    );

    // Singleton Matrix / Vector forward.
    let mut forward_mat_res = DataRepr::Matrix(array![[0.0, 0.0, 0.0]]);
    div(&mut forward_mat_res, &singleton_matrix, &vector);
    assert_eq!(*forward_mat_res.matrix(), *vector_s_matrix_res);

    // Matrix / Scalar.
    let res = &matrix / &scalar;
    let matrix_scalar_res: &Matrix = res.matrix();
    assert_eq!(*matrix_scalar_res, *scalar_matrix_res);

    // Matrix / Scalar forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    div(&mut forward_mat_res, &matrix, &scalar);
    assert_eq!(*forward_mat_res.matrix(), *matrix_scalar_res);

    // Matrix / Vector.
    let res = &matrix / &vector;
    let matrix_vector_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_vector_res,
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix / Vector forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    div(&mut forward_mat_res, &matrix, &vector);
    assert_eq!(*forward_mat_res.matrix(), *matrix_vector_res);

    // Matrix / Matrix.
    let res =
        &matrix / &DataRepr::Matrix(array![[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]]);
    let matrix_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_matrix_res,
        array![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
    );

    // Matrix / Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    div(
        &mut forward_mat_res,
        &matrix,
        &DataRepr::Matrix(array![[6.0, 6.0, 6.0], [6.0, 6.0, 6.0], [6.0, 6.0, 6.0]]),
    );
    assert_eq!(*forward_mat_res.matrix(), *matrix_matrix_res);

    // Matrix / Matrix broadcast.
    let res = &matrix / &DataRepr::Matrix(array![[3.0, 6.0, 3.0]]);
    let matrix_matrix_b_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_matrix_b_res,
        array![[1.0, 0.5, 1.0], [1.0, 0.5, 1.0], [1.0, 0.5, 1.0]]
    );

    // Matrix / Matrix forward broadcast.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    div(
        &mut forward_mat_res,
        &matrix,
        &DataRepr::Matrix(array![[3.0, 6.0, 3.0]]),
    );
    assert_eq!(*forward_mat_res.matrix(), *matrix_matrix_b_res);

    // Singleton Matrix / Matrix.
    let res = &singleton_matrix / &matrix;
    let matrix_s_matrix_res: &Matrix = res.matrix();
    assert_eq!(
        *matrix_s_matrix_res,
        array![
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        ]
    );

    // Singleton Matrix / Matrix forward.
    let mut forward_mat_res =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
    div(&mut forward_mat_res, &singleton_matrix, &matrix);
    assert_eq!(*forward_mat_res.matrix(), *matrix_s_matrix_res);
}

#[test]
fn assign_test() {
    let mut scalar_trgt = DataRepr::Scalar(0.0);
    let mut vector_trgt = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    let mut matrix_trgt =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);

    let scalar = DataRepr::Scalar(1.0);
    let vector = DataRepr::Vector(array![1.0, 1.0, 1.0]);
    let matrix = DataRepr::Matrix(array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);

    // Scalar scalar assignment.
    assign(&mut scalar_trgt, &scalar);
    assert_eq!(scalar_trgt.scalar(), scalar.scalar());

    // Scalar scalar vector.
    assign(&mut scalar_trgt, &vector);
    assert_eq!(scalar_trgt.scalar(), 3.0);

    // Scalar scalar matrix.
    assign(&mut scalar_trgt, &matrix);
    assert_eq!(scalar_trgt.scalar(), 9.0);

    // Vector scalar assignment.
    assign(&mut vector_trgt, &scalar);
    assert_eq!(*vector_trgt.vector(), array![1.0, 1.0, 1.0]);

    // Vector vector assignment.
    assign(&mut vector_trgt, &vector);
    assert_eq!(*vector_trgt.vector(), array![1.0, 1.0, 1.0]);

    // Vector matrix assignment.
    assign(&mut vector_trgt, &matrix);
    assert_eq!(*vector_trgt.vector(), array![3.0, 3.0, 3.0]);

    // Matrix scalar assignment.
    assign(&mut matrix_trgt, &scalar);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix vector assignment.
    assign(&mut matrix_trgt, &vector);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix matrix assignment.
    assign(&mut matrix_trgt, &matrix);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );
}

#[test]
fn scaled_assign_test() {
    let mut scalar_trgt = DataRepr::Scalar(0.0);
    let mut vector_trgt = DataRepr::Vector(array![0.0, 0.0, 0.0]);
    let mut matrix_trgt =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);

    let scalar = DataRepr::Scalar(1.0);
    let vector = DataRepr::Vector(array![1.0, 1.0, 1.0]);
    let matrix = DataRepr::Matrix(array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);

    // Scalar scalar assignment.
    scaled_assign(&mut scalar_trgt, &scalar, -1.0);
    assert_eq!(scalar_trgt.scalar(), -scalar.scalar());

    // Scalar scalar vector.
    scaled_assign(&mut scalar_trgt, &vector, -1.0);
    assert_eq!(scalar_trgt.scalar(), -3.0);

    // Scalar scalar matrix.
    scaled_assign(&mut scalar_trgt, &matrix, -1.0);
    assert_eq!(scalar_trgt.scalar(), -9.0);

    // Vector scalar assignment.
    scaled_assign(&mut vector_trgt, &scalar, -1.0);
    assert_eq!(*vector_trgt.vector(), -array![1.0, 1.0, 1.0]);

    // Vector vector assignment.
    scaled_assign(&mut vector_trgt, &vector, -1.0);
    assert_eq!(*vector_trgt.vector(), -array![1.0, 1.0, 1.0]);

    // Vector matrix assignment.
    scaled_assign(&mut vector_trgt, &matrix, -1.0);
    assert_eq!(*vector_trgt.vector(), -array![3.0, 3.0, 3.0]);

    // Matrix scalar assignment.
    scaled_assign(&mut matrix_trgt, &scalar, -1.0);
    assert_eq!(
        *matrix_trgt.matrix(),
        -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix vector assignment.
    scaled_assign(&mut matrix_trgt, &vector, -1.0);
    assert_eq!(
        *matrix_trgt.matrix(),
        -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );

    // Matrix matrix assignment.
    scaled_assign(&mut matrix_trgt, &matrix, -1.0);
    assert_eq!(
        *matrix_trgt.matrix(),
        -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    );
}

#[test]
fn add_assign_test() {
    let mut scalar_trgt = DataRepr::Scalar(5.0);
    let mut vector_trgt = DataRepr::Vector(array![5.0, 5.0, 5.0]);
    let mut matrix_trgt =
        DataRepr::Matrix(array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]);

    let scalar = DataRepr::Scalar(5.0);
    let vector = DataRepr::Vector(array![5.0, 5.0, 5.0]);
    let matrix = DataRepr::Matrix(array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]);

    // Scalar scalar assignment.
    add_assign(&mut scalar_trgt, &scalar);
    assert_eq!(scalar_trgt.scalar(), 10.0);

    // Scalar scalar vector.
    add_assign(&mut scalar_trgt, &vector);
    assert_eq!(scalar_trgt.scalar(), 25.0);

    // Scalar scalar matrix.
    add_assign(&mut scalar_trgt, &matrix);
    assert_eq!(scalar_trgt.scalar(), 70.0);

    // Vector scalar assignment.
    add_assign(&mut vector_trgt, &scalar);
    assert_eq!(*vector_trgt.vector(), array![10.0, 10.0, 10.0]);

    // Vector vector assignment.
    add_assign(&mut vector_trgt, &vector);
    assert_eq!(*vector_trgt.vector(), array![15.0, 15.0, 15.0]);

    // Vector matrix assignment.
    add_assign(&mut vector_trgt, &matrix);
    assert_eq!(*vector_trgt.vector(), array![30.0, 30.0, 30.0]);

    // Matrix scalar assignment.
    add_assign(&mut matrix_trgt, &scalar);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    );

    // Matrix vector assignment.
    add_assign(&mut matrix_trgt, &vector);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[15.0, 15.0, 15.0], [15.0, 15.0, 15.0], [15.0, 15.0, 15.0]]
    );

    // Matrix matrix assignment.
    add_assign(&mut matrix_trgt, &matrix);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[20.0, 20.0, 20.0], [20.0, 20.0, 20.0], [20.0, 20.0, 20.0]]
    );
}

#[test]
fn scaled_add_assign_test() {
    let mut scalar_trgt = DataRepr::Scalar(5.0);
    let mut vector_trgt = DataRepr::Vector(array![5.0, 5.0, 5.0]);
    let mut matrix_trgt =
        DataRepr::Matrix(array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]);

    let scalar = DataRepr::Scalar(5.0);
    let vector = DataRepr::Vector(array![5.0, 5.0, 5.0]);
    let matrix = DataRepr::Matrix(array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]);

    // Scalar scalar assignment.
    scaled_add_assign(&mut scalar_trgt, &scalar, -1.0);
    assert_eq!(scalar_trgt.scalar(), 0.0);
    scalar_trgt.set_zero();

    // Scalar scalar vector.
    scaled_add_assign(&mut scalar_trgt, &vector, -1.0);
    assert_eq!(scalar_trgt.scalar(), -15.0);
    scalar_trgt.set_zero();

    // Scalar scalar matrix.
    scaled_add_assign(&mut scalar_trgt, &matrix, -1.0);
    assert_eq!(scalar_trgt.scalar(), -45.0);
    scalar_trgt.set_zero();

    // Vector scalar assignment.
    scaled_add_assign(&mut vector_trgt, &scalar, -1.0);
    assert_eq!(*vector_trgt.vector(), array![0.0, 0.0, 0.0]);
    vector_trgt.set_zero();

    // Vector vector assignment.
    scaled_add_assign(&mut vector_trgt, &vector, -1.0);
    assert_eq!(*vector_trgt.vector(), array![-5.0, -5.0, -5.0]);
    vector_trgt.set_zero();

    // Vector matrix assignment.
    scaled_add_assign(&mut vector_trgt, &matrix, -1.0);
    assert_eq!(*vector_trgt.vector(), array![-15.0, -15.0, -15.0]);
    vector_trgt.set_zero();

    // Matrix scalar assignment.
    scaled_add_assign(&mut matrix_trgt, &scalar, -1.0);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    );
    matrix_trgt.set_zero();

    // Matrix vector assignment.
    scaled_add_assign(&mut matrix_trgt, &vector, -1.0);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]
    );
    matrix_trgt.set_zero();

    // Matrix matrix assignment.
    scaled_add_assign(&mut matrix_trgt, &matrix, -1.0);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]
    );
    matrix_trgt.set_zero();
}

#[test]
fn div_assign_pow_test() {
    // Scalar target.
    let mut scalar_trgt = DataRepr::Scalar(1.0);
    // Vector target.
    let mut vector_trgt = DataRepr::Vector(array![1.0, 1.0, 1.0]);
    // Matrix target.
    let mut matrix_trgt =
        DataRepr::Matrix(array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);

    // The source is a one row matrix.
    let src = DataRepr::Matrix(array![[2.0, 2.0, 2.0]]);

    div_assign_pow(&mut scalar_trgt, &src, 2);
    assert_eq!(scalar_trgt.scalar(), 1.0 / 12.0);

    div_assign_pow(&mut vector_trgt, &src, 2);
    assert_eq!(*vector_trgt.vector(), array![0.25, 0.25, 0.25]);

    // Assign to a matrix a one-row matrix.
    div_assign_pow(&mut matrix_trgt, &src, 2);
    assert_eq!(
        *matrix_trgt.matrix(),
        array![[0.25, 0.25, 0.25], [0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]
    );
}

#[test]
fn mat_mat_mul_test() {
    // 2 rows 3 cols floating point matrix.
    let matrix_2x3 = DataRepr::Matrix(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    // 3 rows 2 cols floating point matrix.
    let matrix_3x2 = DataRepr::Matrix(array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    // 2 rows 2 cols floating point matrix.
    let mut matrix_2x2 = DataRepr::Matrix(array![[0.0, 0.0], [0.0, 0.0]]);
    // 3 rows 3 cols floating point matrix.
    let mut matrix_3x3 =
        DataRepr::Matrix(array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);

    // Output should be: 2x3 mul 3x2 -> 2x2.
    mat_mat_mul(
        &mut matrix_2x2,
        1.0,
        &matrix_2x3,
        &matrix_3x2,
        0.0,
        false,
        false,
    );
    assert_eq!(matrix_2x2.matrix().dim(), (2, 2));

    // Output should be: 2x3 mult (2x3)^t -> 2x2.
    mat_mat_mul(
        &mut matrix_2x2,
        1.0,
        &matrix_2x3,
        &matrix_2x3,
        0.0,
        false,
        true,
    );
    assert_eq!(matrix_2x2.matrix().dim(), (2, 2));

    // Output should be: (2x3)^t mult 3x2 -> 3x3.
    mat_mat_mul(
        &mut matrix_3x3,
        1.0,
        &matrix_2x3,
        &matrix_2x3,
        0.0,
        true,
        false,
    );
    assert_eq!(matrix_3x3.matrix().dim(), (3, 3));
}

#[test]
fn mat_vec_mul_test() {
    // 3 rows 3 cols floating point matrix.
    let matrix_3x3 = DataRepr::Matrix(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    // 3-dim vector.
    let vector = DataRepr::Vector(array![1.0, 1.0, 1.0]);
    // 3-dim vector.
    let mut res = DataRepr::Vector(array![0.0, 0.0, 0.0]);

    mat_vec_mul(&mut res, 1.0, &matrix_3x3, &vector, 0.0, false);
    assert_eq!(*res.vector(), array![6.0, 15.0, 24.0]);
}
