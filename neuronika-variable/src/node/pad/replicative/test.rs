use super::{PaddingMode, Replicative};

use ndarray::{self, Array, IntoDimension, Ix3, Ix4, Ix5};

#[test]
fn test_1d() {
    let padding = Replicative;

    let base = Array::range(0.0, 5.0, 1.0);
    let mut padded = Array::<f32, _>::zeros(9);

    PaddingMode::<Ix3>::pad(
        &padding,
        &mut padded.view_mut(),
        &base.view(),
        [2].into_dimension(),
    );

    assert_eq!(padded, ndarray::array![0., 0., 0., 1., 2., 3., 4., 4., 4.],);
}

#[test]
fn test_2d() {
    let padding = Replicative;

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
            [0., 0., 0., 1., 2., 3., 4., 4., 4.],
            [0., 0., 0., 1., 2., 3., 4., 4., 4.],
            [5., 5., 5., 6., 7., 8., 9., 9., 9.],
            [10., 10., 10., 11., 12., 13., 14., 14., 14.],
            [15., 15., 15., 16., 17., 18., 19., 19., 19.],
            [20., 20., 20., 21., 22., 23., 24., 24., 24.],
            [20., 20., 20., 21., 22., 23., 24., 24., 24.],
        ]
    );
}

#[test]
fn test_3d() {
    let padding = Replicative;

    let base = Array::range(0.0, 125.0, 1.0).into_shape((5, 5, 5)).unwrap();

    let mut padded = Array::<f32, _>::zeros((7, 9, 11));

    PaddingMode::<Ix5>::pad(
        &padding,
        &mut padded.view_mut(),
        &base.view(),
        [1, 2, 3].into_dimension(),
    );

    assert_eq!(
        padded,
        ndarray::array![
            [
                [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                [5., 5., 5., 5., 6., 7., 8., 9., 9., 9., 9.],
                [10., 10., 10., 10., 11., 12., 13., 14., 14., 14., 14.],
                [15., 15., 15., 15., 16., 17., 18., 19., 19., 19., 19.],
                [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.]
            ],
            [
                [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                [5., 5., 5., 5., 6., 7., 8., 9., 9., 9., 9.],
                [10., 10., 10., 10., 11., 12., 13., 14., 14., 14., 14.],
                [15., 15., 15., 15., 16., 17., 18., 19., 19., 19., 19.],
                [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.]
            ],
            [
                [25., 25., 25., 25., 26., 27., 28., 29., 29., 29., 29.],
                [25., 25., 25., 25., 26., 27., 28., 29., 29., 29., 29.],
                [25., 25., 25., 25., 26., 27., 28., 29., 29., 29., 29.],
                [30., 30., 30., 30., 31., 32., 33., 34., 34., 34., 34.],
                [35., 35., 35., 35., 36., 37., 38., 39., 39., 39., 39.],
                [40., 40., 40., 40., 41., 42., 43., 44., 44., 44., 44.],
                [45., 45., 45., 45., 46., 47., 48., 49., 49., 49., 49.],
                [45., 45., 45., 45., 46., 47., 48., 49., 49., 49., 49.],
                [45., 45., 45., 45., 46., 47., 48., 49., 49., 49., 49.]
            ],
            [
                [50., 50., 50., 50., 51., 52., 53., 54., 54., 54., 54.],
                [50., 50., 50., 50., 51., 52., 53., 54., 54., 54., 54.],
                [50., 50., 50., 50., 51., 52., 53., 54., 54., 54., 54.],
                [55., 55., 55., 55., 56., 57., 58., 59., 59., 59., 59.],
                [60., 60., 60., 60., 61., 62., 63., 64., 64., 64., 64.],
                [65., 65., 65., 65., 66., 67., 68., 69., 69., 69., 69.],
                [70., 70., 70., 70., 71., 72., 73., 74., 74., 74., 74.],
                [70., 70., 70., 70., 71., 72., 73., 74., 74., 74., 74.],
                [70., 70., 70., 70., 71., 72., 73., 74., 74., 74., 74.]
            ],
            [
                [75., 75., 75., 75., 76., 77., 78., 79., 79., 79., 79.],
                [75., 75., 75., 75., 76., 77., 78., 79., 79., 79., 79.],
                [75., 75., 75., 75., 76., 77., 78., 79., 79., 79., 79.],
                [80., 80., 80., 80., 81., 82., 83., 84., 84., 84., 84.],
                [85., 85., 85., 85., 86., 87., 88., 89., 89., 89., 89.],
                [90., 90., 90., 90., 91., 92., 93., 94., 94., 94., 94.],
                [95., 95., 95., 95., 96., 97., 98., 99., 99., 99., 99.],
                [95., 95., 95., 95., 96., 97., 98., 99., 99., 99., 99.],
                [95., 95., 95., 95., 96., 97., 98., 99., 99., 99., 99.]
            ],
            [
                [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                [105., 105., 105., 105., 106., 107., 108., 109., 109., 109., 109.],
                [110., 110., 110., 110., 111., 112., 113., 114., 114., 114., 114.],
                [115., 115., 115., 115., 116., 117., 118., 119., 119., 119., 119.],
                [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.]
            ],
            [
                [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                [105., 105., 105., 105., 106., 107., 108., 109., 109., 109., 109.],
                [110., 110., 110., 110., 111., 112., 113., 114., 114., 114., 114.],
                [115., 115., 115., 115., 116., 117., 118., 119., 119., 119., 119.],
                [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.]
            ]
        ]
    );
}
