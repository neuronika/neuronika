use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, BCEWithLogitsLoss,
    BCEWithLogitsLossBackward, Backward, Data, Forward, Gradient, Reduction,
};
use ndarray::arr0;

#[test]
fn mean() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
    let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
    let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Mean);

    loss.forward();
    assert_almost_equals(&*loss.data(), &arr0(8.));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let input_diff = new_backward_input((3, 3), vec![0.; 9]);
    let loss_backward =
        BCEWithLogitsLossBackward::new(input_diff.clone(), input, target, Reduction::Mean);
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = arr0(1.);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &new_tensor(
            (3, 3),
            vec![
                -0.0000050465264,
                -0.0000018543667,
                0.11111042,
                0.11111086,
                0.111111015,
                -0.00000003973643,
                0.1111111,
                0.11111111,
                0.,
            ],
        ),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &(&new_tensor(
            (3, 3),
            vec![
                -0.0000050465264,
                -0.0000018543667,
                0.11111042,
                0.11111086,
                0.111111015,
                -0.00000003973643,
                0.1111111,
                0.11111111,
                0.,
            ],
        ) * 2.),
    );
}

#[test]
fn sum() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
    let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
    let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Sum);

    loss.forward();
    assert_almost_equals(&*loss.data(), &arr0(72.0001));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let input_diff = new_backward_input((3, 3), vec![0.; 9]);
    let loss_backward =
        BCEWithLogitsLossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = arr0(1.);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &new_tensor(
            (3, 3),
            vec![
                -0.00004541874,
                -0.0000166893,
                0.9999938,
                0.99999774,
                0.99999917,
                -0.00000035762787,
                0.9999999,
                1.,
                0.,
            ],
        ),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &(&new_tensor(
            (3, 3),
            vec![
                -0.00004541874,
                -0.0000166893,
                0.9999938,
                0.99999774,
                0.99999917,
                -0.00000035762787,
                0.9999999,
                1.,
                0.,
            ],
        ) * 2.),
    );
}

#[test]
fn debug_forward() {
    let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
    let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
    let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Mean);

    let output = "BCEWithLogitsLoss { data: 0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0, reduction: Mean, computed: false }";

    assert_eq!(output, format!("{:?}", loss));
}

#[test]
fn display_forward() {
    let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
    let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
    let loss = BCEWithLogitsLoss::new(input.clone(), target.clone(), Reduction::Mean);

    assert_eq!(format!("{}", loss.data()), format!("{}", loss));
}

#[test]
fn debug_backward() {
    let loss = BCEWithLogitsLossBackward::new(
        new_backward_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        Reduction::Mean,
    );

    let output = "BCEWithLogitsLossBackward { gradient: Some(0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0), reduction: Mean, overwrite: true }";

    assert_eq!(output, format!("{:?}", loss));
}

#[test]
fn display_backward() {
    let loss = BCEWithLogitsLossBackward::new(
        new_backward_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        Reduction::Mean,
    );

    assert_eq!(format!("{}", loss.gradient()), format!("{}", loss));
}

#[test]
fn no_grad() {
    // BCEWithLogitsLossBackward
    let node = BCEWithLogitsLossBackward::new(
        new_backward_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        Reduction::Mean,
    );

    node.no_grad();
    assert!(node.gradient.borrow().is_none());

    node.with_grad();
    assert_eq!(&*node.gradient(), arr0(0.));
}
