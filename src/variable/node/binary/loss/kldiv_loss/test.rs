use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, KLDivLoss, KLDivLossBackward, Reduction,
};
use ndarray::arr0;

#[test]
fn mean() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input((2, 3), vec![0.2, 0.5, 0.3, 0.6, 0.0, 0.4]);
    let v: Vec<f32> = vec![0.4, 0.5, 0.1, 0.6, 0.1, 0.3]
        .iter()
        .map(|&el: &f32| el.ln())
        .collect();
    let input = new_input((2, 3), v);
    input.forward();

    let loss = KLDivLoss::new(input, target.clone(), Reduction::Mean);

    loss.forward();
    assert_almost_equals(&*loss.data(), &arr0(0.1530));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let input_diff = new_backward_input((2, 3), vec![0.; 6]);
    let loss_backward = KLDivLossBackward::new(input_diff.clone(), target, Reduction::Mean);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = arr0(1.);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &new_tensor(
            (2, 3),
            vec![-0.1000, -0.2500, -0.1500, -0.3000, 0.0000, -0.2000],
        ),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &(&new_tensor(
            (2, 3),
            vec![-0.1000, -0.2500, -0.1500, -0.3000, 0.0000, -0.2000],
        ) * 2.),
    );
}

#[test]
fn sum() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input((2, 3), vec![0.2, 0.5, 0.3, 0.6, 0.0, 0.4]);
    let v: Vec<f32> = vec![0.4, 0.5, 0.1, 0.6, 0.1, 0.3]
        .iter()
        .map(|&el: &f32| el.ln())
        .collect();
    let input = new_input((2, 3), v);
    input.forward();

    let loss = KLDivLoss::new(input, target.clone(), Reduction::Sum);

    loss.forward();
    assert_almost_equals(&*loss.data(), &arr0(0.3060));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let input_diff = new_backward_input((2, 3), vec![0.; 6]);
    let loss_backward = KLDivLossBackward::new(input_diff.clone(), target, Reduction::Sum);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = arr0(1.);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &new_tensor(
            (2, 3),
            vec![-0.2000, -0.5000, -0.3000, -0.6000, 0.0000, -0.4000],
        ),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &(&new_tensor(
            (2, 3),
            vec![-0.2000, -0.5000, -0.3000, -0.6000, 0.0000, -0.4000],
        ) * 2.),
    );
}

#[test]
fn debug_forward() {
    let target = new_input((2, 3), vec![0.2, 0.5, 0.3, 0.6, 0.0, 0.4]);
    let v: Vec<f32> = vec![0.4, 0.5, 0.1, 0.6, 0.1, 0.3]
        .iter()
        .map(|&el: &f32| el.ln())
        .collect();
    let input = new_input((2, 3), v);

    let loss = KLDivLoss::new(input, target.clone(), Reduction::Mean);

    let output = "KLDivLoss { data: 0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0, reduction: Mean, computed: false }";

    assert_eq!(output, format!("{:?}", loss));
}

#[test]
fn display_forward() {
    let target = new_input((2, 3), vec![0.2, 0.5, 0.3, 0.6, 0.0, 0.4]);
    let v: Vec<f32> = vec![0.4, 0.5, 0.1, 0.6, 0.1, 0.3]
        .iter()
        .map(|&el: &f32| el.ln())
        .collect();
    let input = new_input((2, 3), v);

    let loss = KLDivLoss::new(input, target.clone(), Reduction::Mean);

    assert_eq!(format!("{}", loss.data()), format!("{}", loss));
}

#[test]
fn debug_backward() {
    let target = new_input((2, 3), vec![0.2, 0.5, 0.3, 0.6, 0.0, 0.4]);
    let input_diff = new_backward_input((2, 3), vec![0.; 6]);
    let loss = KLDivLossBackward::new(input_diff.clone(), target, Reduction::Mean);

    let output = "KLDivLossBackward { gradient: Some(0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0), reduction: Mean, overwrite: true }";

    assert_eq!(output, format!("{:?}", loss));
}

#[test]
fn display_backward() {
    let target = new_input((2, 3), vec![0.2, 0.5, 0.3, 0.6, 0.0, 0.4]);
    let input_diff = new_backward_input((2, 3), vec![0.; 6]);
    let loss = KLDivLossBackward::new(input_diff.clone(), target, Reduction::Mean);

    assert_eq!(format!("{}", loss.gradient()), format!("{}", loss));
}

#[test]
fn no_grad() {
    // KLDivLossBackward
    let node = KLDivLossBackward::new(
        new_backward_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        Reduction::Mean,
    );

    node.no_grad();
    assert!(node.gradient.borrow().is_none());

    node.with_grad();
    assert_eq!(&*node.gradient(), arr0(0.));
}
