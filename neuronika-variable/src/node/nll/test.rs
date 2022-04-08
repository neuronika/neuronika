use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, NLLLoss, NLLLossBackward, Rc, Reduction,
};
use crate::variable::node::LogSoftmax;
use ndarray::arr0;

#[test]
fn mean() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input(3, vec![2., 0., 4.]);
    let input = Rc::new(LogSoftmax::new(
        new_input(
            (3, 5),
            vec![
                0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0., 0.2, 0.5,
            ],
        ),
        1,
    ));
    input.forward();

    let loss = NLLLoss::new(input, target.clone(), Reduction::Mean);

    loss.forward();
    assert_almost_equals(&*loss.data(), &arr0(1.52222));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let input_diff = new_backward_input((3, 5), vec![0.; 15]);
    let loss_backward = NLLLossBackward::new(input_diff.clone(), target, Reduction::Mean);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = arr0(1.);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &new_tensor(
            (3, 5),
            vec![
                0.0000, 0.0000, -0.3333, 0.0000, 0.0000, -0.3333, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, -0.3333,
            ],
        ),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &(&new_tensor(
            (3, 5),
            vec![
                0.0000, 0.0000, -0.3333, 0.0000, 0.0000, -0.3333, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, -0.3333,
            ],
        ) * 2.),
    );
}

#[test]
fn sum() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input(3, vec![2., 0., 4.]);
    let input = Rc::new(LogSoftmax::new(
        new_input(
            (3, 5),
            vec![
                0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0., 0.2, 0.5,
            ],
        ),
        1,
    ));
    input.forward();

    let loss = NLLLoss::new(input, target.clone(), Reduction::Sum);

    loss.forward();
    assert_almost_equals(&*loss.data(), &arr0(4.56666));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let input_diff = new_backward_input((3, 5), vec![0.; 15]);
    let loss_backward = NLLLossBackward::new(input_diff.clone(), target, Reduction::Sum);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = arr0(1.);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &new_tensor(
            (3, 5),
            vec![
                0.0000, 0.0000, -1.0000, 0.0000, 0.0000, -1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, -1.0000,
            ],
        ),
    );

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(
        &*input_diff.gradient(),
        &(&new_tensor(
            (3, 5),
            vec![
                0.0000, 0.0000, -1.0000, 0.0000, 0.0000, -1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                0.0000, 0.0000, 0.0000, 0.0000, -1.0000,
            ],
        ) * 2.),
    );
}

#[test]
fn debug_forward() {
    let target = new_input(3, vec![2., 0., 4.]);
    let input = Rc::new(LogSoftmax::new(
        new_input(
            (3, 5),
            vec![
                0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0., 0.2, 0.5,
            ],
        ),
        1,
    ));

    let loss = NLLLoss::new(input, target.clone(), Reduction::Mean);

    let output = "NLLLoss { data: 0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0, reduction: Mean, computed: false }";

    assert_eq!(output, format!("{:?}", loss));
}

#[test]
fn display_forward() {
    let target = new_input(3, vec![2., 0., 4.]);
    let input = Rc::new(LogSoftmax::new(
        new_input(
            (3, 5),
            vec![
                0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0.4, 0.2, 0.1, 0., 0.3, 0., 0.2, 0.5,
            ],
        ),
        1,
    ));

    let loss = NLLLoss::new(input, target.clone(), Reduction::Mean);

    assert_eq!(format!("{}", loss.data()), format!("{}", loss));
}

#[test]
fn debug_backward() {
    let input_diff = new_backward_input((3, 5), vec![0.; 15]);
    let target = new_input(3, vec![2., 0., 4.]);

    let loss = NLLLossBackward::new(input_diff.clone(), target, Reduction::Mean);

    let output = "NLLLossBackward { gradient: Some(0.0, shape=[], strides=[], layout=CFcf (0xf), const ndim=0), reduction: Mean, overwrite: true }";

    assert_eq!(output, format!("{:?}", loss));
}

#[test]
fn display_backward() {
    let input_diff = new_backward_input((3, 5), vec![0.; 15]);
    let target = new_input(3, vec![2., 0., 4.]);

    let loss = NLLLossBackward::new(input_diff.clone(), target, Reduction::Mean);

    assert_eq!(format!("{}", loss.gradient()), format!("{}", loss));
}

#[test]
fn no_grad() {
    // NLLLossBackward
    let node = NLLLossBackward::new(
        new_backward_input((3, 3), vec![0.; 9]),
        new_input(3, vec![0.; 3]),
        Reduction::Mean,
    );

    node.no_grad();
    assert!(node.gradient.borrow().is_none());

    node.with_grad();
    assert_eq!(&*node.gradient(), arr0(0.));
}
