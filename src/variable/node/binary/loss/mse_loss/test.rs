use super::{
    assert_almost_equals, new_backward_input, new_input, new_tensor, Backward, Data, Forward,
    Gradient, MSELoss, MSELossBackward, Reduction, Tensor,
};

#[test]
fn mean() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
    let loss = MSELoss::new(input.clone(), target.clone(), Reduction::Mean);

    loss.forward();
    assert_almost_equals(&*loss.data(), &new_tensor(1, vec![81.]));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let input_diff = new_backward_input((3, 3), vec![0.; 9]);
    let loss_backward = MSELossBackward::new(input_diff.clone(), input, target, Reduction::Mean);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![2.; 9]));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![4.; 9]));
}

#[test]
fn sum() {
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let target = new_input((3, 3), vec![1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    let input = new_input((3, 3), vec![10., 11., 12., 13., 14., 15., 16., 17., 18.]);
    let loss = MSELoss::new(input.clone(), target.clone(), Reduction::Sum);

    loss.forward();
    assert_almost_equals(&*loss.data(), &new_tensor(1, vec![729.]));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward Pass ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    let input_diff = new_backward_input((3, 3), vec![0.; 9]);
    let loss_backward = MSELossBackward::new(input_diff.clone(), input, target, Reduction::Sum);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Seed Gradient ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    *loss_backward.gradient_mut() = new_tensor(1, vec![1.]);

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![18.; 9]));

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2nd Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    loss_backward.backward();
    assert_almost_equals(&*input_diff.gradient(), &new_tensor((3, 3), vec![36.; 9]));
}

#[test]
fn no_grad() {
    // MSELossBackward
    let node = MSELossBackward::new(
        new_backward_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        new_input(3, vec![0.; 3]),
        Reduction::Mean,
    );

    node.no_grad();
    assert!(node.gradient.borrow().is_none());

    node.with_grad();
    assert_eq!(&*node.gradient(), Tensor::zeros(1));
}
