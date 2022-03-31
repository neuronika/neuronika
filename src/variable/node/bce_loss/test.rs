use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, BCELoss, BCELossBackward,
    Backward, Data, Forward, Gradient, Reduction,
};
use ndarray::arr0;

#[test]
fn mean() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
    let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
    let loss = BCELoss::new(input.clone(), target.clone(), Reduction::Mean);

    loss.forward();
    assert_almost_equals(&*loss.data(), &arr0(22.9244));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let input_diff = new_backward_input((3, 3), vec![0.; 9]);
    let loss_backward = BCELossBackward::new(input_diff.clone(), input, target, Reduction::Mean);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = arr0(1.);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &new_tensor(
            (3, 3),
            vec![
                -1.1111e+00,
                -1.2346e-01,
                1.1111e+00,
                0.0000e+00,
                0.0000e+00,
                -9.32067e+05,
                5.5556e-01,
                0.0000e+00,
                -9.32067e+05,
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
                -1.1111e+00,
                -1.2346e-01,
                1.1111e+00,
                0.0000e+00,
                0.0000e+00,
                -9.32067e+05,
                5.5556e-01,
                0.0000e+00,
                -9.32067e+05,
            ],
        ) * 2.),
    );
}

#[test]
fn sum() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
    let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
    let loss = BCELoss::new(input.clone(), target.clone(), Reduction::Sum);

    loss.forward();
    assert_almost_equals(&*loss.data(), &arr0(206.3199));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let input_diff = new_backward_input((3, 3), vec![0.; 9]);
    let loss_backward = BCELossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = arr0(1.);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &new_tensor(
            (3, 3),
            vec![
                -1.0000e+01,
                -1.1111e+00,
                1.0000e+01,
                0.0000e+00,
                0.0000e+00,
                -8.3886e+6,
                5.0000e+00,
                0.0000e+00,
                -8.3886e+6,
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
                -1.0000e+01,
                -1.1111e+00,
                1.0000e+01,
                0.0000e+00,
                0.0000e+00,
                -8.3886e+6,
                5.0000e+00,
                0.0000e+00,
                -8.3886e+6,
            ],
        ) * 2.),
    );
}

#[test]
fn no_grad() {
    // BCELossBackward
    let node = BCELossBackward::new(
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

#[test]
fn debug_forward() {
    let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
    let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
    let loss = BCELoss::new(input.clone(), target.clone(), Reduction::Mean);

    let output = "BCELoss { data: 0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0, reduction: Mean, computed: false }";

    assert_eq!(output, format!("{:?}", loss));
}

#[test]
fn display_forward() {
    let target = new_input((3, 3), vec![1., 1., 0., 0., 0., 1., 0., 0., 1.]);
    let input = new_input((3, 3), vec![0.1, 0.9, 0.9, 0., 0., 0., 0.8, 0., 0.]);
    let loss = BCELoss::new(input.clone(), target.clone(), Reduction::Mean);

    assert_eq!(format!("{}", loss.data()), format!("{}", loss));
}

#[test]
fn debug_backward() {
    let loss = BCELossBackward::new(
        new_backward_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        Reduction::Mean,
    );

    let output = "BCELossBackward { gradient: Some(0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0), reduction: Mean, overwrite: true }";

    assert_eq!(output, format!("{:?}", loss));
}

#[test]
fn display_backward() {
    let loss = BCELossBackward::new(
        new_backward_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        Reduction::Mean,
    );

    assert_eq!(format!("{}", loss.gradient()), format!("{}", loss));
}
