use super::{PaddingMode, Zero};

use ndarray::{self, Array, IntoDimension, Ix4};

#[test]
fn test() {
    let padding = Zero;

    let base = Array::range(0.0, 25.0, 1.0).into_shape((5, 5)).unwrap();
    let mut padded = Array::<f32, _>::zeros((7, 9));

    PaddingMode::<Ix4>::pad(
        &padding,
        &mut padded.view_mut(),
        &base.view(),
        [1, 2].into_dimension(),
    );

    assert_eq!(
        padded,
        ndarray::array![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0., 1., 2., 3., 4., 0.0, 0.0],
            [0.0, 0.0, 5., 6., 7., 8., 9., 0.0, 0.0],
            [0.0, 0.0, 10., 11., 12., 13., 14., 0.0, 0.0],
            [0.0, 0.0, 15., 16., 17., 18., 19., 0.0, 0.0],
            [0.0, 0.0, 20., 21., 22., 23., 24., 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    );
}
