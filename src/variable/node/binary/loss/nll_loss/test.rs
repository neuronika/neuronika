use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, NLLLoss, NLLLossBackward, Rc, Reduction, Tensor,
};

#[test]
fn mean() {
    use crate::variable::node::LogSoftmax;

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
    assert_almost_equals(&*loss.data(), &new_tensor(1, vec![1.52222]));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let input_diff = new_backward_input((3, 5), vec![0.; 15]);
    let loss_backward = NLLLossBackward::new(input_diff.clone(), target, Reduction::Mean);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

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
    use crate::variable::node::LogSoftmax;

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
    assert_almost_equals(&*loss.data(), &new_tensor(1, vec![4.56666]));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    let input_diff = new_backward_input((3, 5), vec![0.; 15]);
    let loss_backward = NLLLossBackward::new(input_diff.clone(), target, Reduction::Sum);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

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
    assert_eq!(&*node.gradient(), Tensor::zeros(1));
}
