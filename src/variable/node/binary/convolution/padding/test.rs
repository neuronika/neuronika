use super::{Constant, PaddingMode, Reflective, Replicative, Zero};

#[test]
fn constant_pad() {
    let padding = Constant::new(8.);
    let arr = ndarray::Array::range(0., 25., 1.)
        .into_shape((5, 5))
        .unwrap();
    let padded = padding.pad(&arr, [1, 2]);
    assert_eq!(
        padded,
        ndarray::array![
            [8., 8., 8., 8., 8., 8., 8., 8., 8.],
            [8., 8., 0., 1., 2., 3., 4., 8., 8.],
            [8., 8., 5., 6., 7., 8., 9., 8., 8.],
            [8., 8., 10., 11., 12., 13., 14., 8., 8.],
            [8., 8., 15., 16., 17., 18., 19., 8., 8.],
            [8., 8., 20., 21., 22., 23., 24., 8., 8.],
            [8., 8., 8., 8., 8., 8., 8., 8., 8.],
        ]
    );
}

#[test]
fn zero_pad() {
    let padding = Zero;
    let arr = ndarray::Array::range(0., 25., 1.)
        .into_shape((5, 5))
        .unwrap();
    let padded = padding.pad(&arr, [1, 2]);
    assert_eq!(
        padded,
        ndarray::array![
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 2., 3., 4., 0., 0.],
            [0., 0., 5., 6., 7., 8., 9., 0., 0.],
            [0., 0., 10., 11., 12., 13., 14., 0., 0.],
            [0., 0., 15., 16., 17., 18., 19., 0., 0.],
            [0., 0., 20., 21., 22., 23., 24., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
        ]
    );
}

#[test]
fn replication_pad_1d() {
    let padding = Replicative;
    let arr = ndarray::Array::range(0., 5., 1.);
    let padded = padding.pad(&arr, [2]);
    assert_eq!(padded, ndarray::array![0., 0., 0., 1., 2., 3., 4., 4., 4.],);
}

#[test]
fn replication_pad_2d() {
    let padding = Replicative;
    let arr = ndarray::Array::range(0., 25., 1.)
        .into_shape((5, 5))
        .unwrap();
    let padded = padding.pad(&arr, [1, 2]);
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
fn replication_pad_3d() {
    let padding = Replicative;
    let arr = ndarray::Array::range(0., 125., 1.)
        .into_shape((5, 5, 5))
        .unwrap();
    let padded = padding.pad(&arr, [1, 2, 3]);
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

#[test]
fn reflection_pad_1d() {
    let padding = Reflective;
    let arr = ndarray::Array::range(0., 5., 1.);
    let padded = padding.pad(&arr, [2]);
    assert_eq!(padded, ndarray::array![2., 1., 0., 1., 2., 3., 4., 3., 2.],);
}

#[test]
fn reflection_pad_2d() {
    let padding = Reflective;
    let arr = ndarray::Array::range(0., 25., 1.)
        .into_shape((5, 5))
        .unwrap();
    let padded = padding.pad(&arr, [1, 2]);
    assert_eq!(
        padded,
        ndarray::array![
            [7., 6., 5., 6., 7., 8., 9., 8., 7.],
            [2., 1., 0., 1., 2., 3., 4., 3., 2.],
            [7., 6., 5., 6., 7., 8., 9., 8., 7.],
            [12., 11., 10., 11., 12., 13., 14., 13., 12.],
            [17., 16., 15., 16., 17., 18., 19., 18., 17.],
            [22., 21., 20., 21., 22., 23., 24., 23., 22.],
            [17., 16., 15., 16., 17., 18., 19., 18., 17.]
        ]
    );
}

#[test]
fn reflection_pad_3d() {
    let padding = Reflective;
    let arr = ndarray::Array::range(0., 125., 1.)
        .into_shape((5, 5, 5))
        .unwrap();
    let padded = padding.pad(&arr, [1, 2, 3]);
    assert_eq!(
        padded,
        ndarray::array![
            [
                [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                [28., 27., 26., 25., 26., 27., 28., 29., 28., 27., 26.],
                [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                [48., 47., 46., 45., 46., 47., 48., 49., 48., 47., 46.],
                [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.]
            ],
            [
                [13., 12., 11., 10., 11., 12., 13., 14., 13., 12., 11.],
                [8., 7., 6., 5., 6., 7., 8., 9., 8., 7., 6.],
                [3., 2., 1., 0., 1., 2., 3., 4., 3., 2., 1.],
                [8., 7., 6., 5., 6., 7., 8., 9., 8., 7., 6.],
                [13., 12., 11., 10., 11., 12., 13., 14., 13., 12., 11.],
                [18., 17., 16., 15., 16., 17., 18., 19., 18., 17., 16.],
                [23., 22., 21., 20., 21., 22., 23., 24., 23., 22., 21.],
                [18., 17., 16., 15., 16., 17., 18., 19., 18., 17., 16.],
                [13., 12., 11., 10., 11., 12., 13., 14., 13., 12., 11.]
            ],
            [
                [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                [28., 27., 26., 25., 26., 27., 28., 29., 28., 27., 26.],
                [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                [48., 47., 46., 45., 46., 47., 48., 49., 48., 47., 46.],
                [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.]
            ],
            [
                [63., 62., 61., 60., 61., 62., 63., 64., 63., 62., 61.],
                [58., 57., 56., 55., 56., 57., 58., 59., 58., 57., 56.],
                [53., 52., 51., 50., 51., 52., 53., 54., 53., 52., 51.],
                [58., 57., 56., 55., 56., 57., 58., 59., 58., 57., 56.],
                [63., 62., 61., 60., 61., 62., 63., 64., 63., 62., 61.],
                [68., 67., 66., 65., 66., 67., 68., 69., 68., 67., 66.],
                [73., 72., 71., 70., 71., 72., 73., 74., 73., 72., 71.],
                [68., 67., 66., 65., 66., 67., 68., 69., 68., 67., 66.],
                [63., 62., 61., 60., 61., 62., 63., 64., 63., 62., 61.]
            ],
            [
                [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                [78., 77., 76., 75., 76., 77., 78., 79., 78., 77., 76.],
                [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                [98., 97., 96., 95., 96., 97., 98., 99., 98., 97., 96.],
                [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.]
            ],
            [
                [113., 112., 111., 110., 111., 112., 113., 114., 113., 112., 111.],
                [108., 107., 106., 105., 106., 107., 108., 109., 108., 107., 106.],
                [103., 102., 101., 100., 101., 102., 103., 104., 103., 102., 101.],
                [108., 107., 106., 105., 106., 107., 108., 109., 108., 107., 106.],
                [113., 112., 111., 110., 111., 112., 113., 114., 113., 112., 111.],
                [118., 117., 116., 115., 116., 117., 118., 119., 118., 117., 116.],
                [123., 122., 121., 120., 121., 122., 123., 124., 123., 122., 121.],
                [118., 117., 116., 115., 116., 117., 118., 119., 118., 117., 116.],
                [113., 112., 111., 110., 111., 112., 113., 114., 113., 112., 111.]
            ],
            [
                [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                [78., 77., 76., 75., 76., 77., 78., 79., 78., 77., 76.],
                [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                [98., 97., 96., 95., 96., 97., 98., 99., 98., 97., 96.],
                [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.]
            ]
        ]
    )
}
