use std::{cell::Cell, rc::Rc};

#[test]
fn data_mut() {
    // Var
    let x = crate::ones((2, 2));
    *x.data_mut() += 1.;
    assert_eq!(*x.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // VarDiff
    let x = crate::ones((2, 2)).requires_grad();
    *x.data_mut() += 1.;
    assert_eq!(*x.data(), ndarray::array![[2., 2.,], [2., 2.,]]);
}

#[test]
fn grad_mut() {
    // Only VarDiff has a gradient.
    let x = crate::ones((2, 2)).requires_grad();
    *x.grad_mut() += 1.;
    assert_eq!(*x.grad(), ndarray::array![[1., 1.,], [1., 1.,]]);
}

#[test]
fn add_scalar() {
    // Var - f32
    let x = crate::ones((2, 2));
    let y = x + 1.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // f32 - Var
    let x = crate::ones((2, 2));
    let y = 1. + x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // VarDiff - f32
    let x = crate::ones((2, 2)).requires_grad();
    let y = x + 1.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // f32 - VarDiff
    let x = crate::ones((2, 2)).requires_grad();
    let y = 1. + x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);
}

#[test]
fn sub_scalar() {
    // Var - f32
    let x = crate::ones((2, 2));
    let y = x - 1.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[0., 0.], [0., 0.]]);

    // f32 - Var
    let x = crate::ones((2, 2));
    let y = 1. - x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[0., 0.], [0., 0.]]);

    // VarDiff - f32
    let x = crate::ones((2, 2)).requires_grad();
    let y = x - 1.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[0., 0.], [0., 0.]]);

    // f32 - VarDiff
    let x = crate::ones((2, 2)).requires_grad();
    let y = 1. - x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[0., 0.], [0., 0.]]);
}

#[test]
fn mul_scalar() {
    // Var - f32
    let x = crate::ones((2, 2));
    let y = x * 2.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // f32 - Var
    let x = crate::ones((2, 2));
    let y = 2. * x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // VarDiff - f32
    let x = crate::ones((2, 2)).requires_grad();
    let y = x * 2.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // f32 - VarDiff
    let x = crate::ones((2, 2)).requires_grad();
    let y = 2. * x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);
}

#[test]
fn div_scalar() {
    // Var - f32
    let x = crate::full((2, 2), 9.);
    let y = x / 3.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[3., 3.], [3., 3.]]);

    // f32 - Var
    let x = crate::full((2, 2), 3.);
    let y = 9. / x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[3., 3.], [3., 3.]]);

    // VarDiff - f32
    let x = crate::full((2, 2), 9.).requires_grad();
    let y = x / 3.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[3., 3.], [3., 3.]]);

    // f32 - VarDiff
    let x = crate::full((2, 2), 3.).requires_grad();
    let y = 9. / x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[3., 3.], [3., 3.]]);
}

#[test]
fn differentiate_loop() {
    let mut x = crate::ones(()).requires_grad();
    let y = x.clone();

    for _ in 0..5 {
        x = x.clone() * 4.0;
    }

    x.forward();
    x.backward(1.0);

    assert_eq!(x.data()[()], 1024.);
    assert_eq!(y.grad()[()], 1024.);
}

#[test]
fn sum() {
    let input = crate::ones((2, 2));
    let sum = input.sum();

    assert_eq!(sum.history.len(), 1);
}

#[test]
fn sum_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let sum = input.sum();

    assert_eq!(sum.history.len(), 1);
}

#[test]
fn mean() {
    let input = crate::ones((2, 2));
    let mean = input.mean();

    assert_eq!(mean.history.len(), 1);
}

#[test]
fn mean_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let mean = input.mean();

    assert_eq!(mean.history.len(), 1);
}

#[test]
fn pow() {
    let input = crate::ones((2, 2));
    let pow = input.pow(2);

    assert_eq!(pow.history.len(), 1);
}

#[test]
fn pow_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let pow = input.pow(2);

    assert_eq!(pow.history.len(), 1);
}

#[test]
fn sqrt() {
    let input = crate::ones((2, 2));
    let relu = input.sqrt();

    assert_eq!(relu.history.len(), 1);
}

#[test]
fn sqrt_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let relu = input.sqrt();

    assert_eq!(relu.history.len(), 1);
}

#[test]
fn relu() {
    let input = crate::ones((2, 2));
    let relu = input.relu();

    assert_eq!(relu.history.len(), 1);
}

#[test]
fn relu_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let relu = input.relu();

    assert_eq!(relu.history.len(), 1);
}

#[test]
fn leaky_relu() {
    let input = crate::ones((2, 2));
    let leaky_relu = input.leaky_relu();

    assert_eq!(leaky_relu.history.len(), 1);
}

#[test]
fn leaky_relu_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let leaky_relu = input.leaky_relu();

    assert_eq!(leaky_relu.history.len(), 1);
}

#[test]
fn softplus() {
    let input = crate::ones((2, 2));
    let softplus = input.softplus();

    assert_eq!(softplus.history.len(), 1);
}

#[test]
fn softplus_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let softplus = input.softplus();

    assert_eq!(softplus.history.len(), 1);
}

#[test]
fn sigmoid() {
    let input = crate::ones((2, 2));
    let sigmoid = input.sigmoid();

    assert_eq!(sigmoid.history.len(), 1);
}

#[test]
fn sigmoid_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let sigmoid = input.sigmoid();

    assert_eq!(sigmoid.history.len(), 1);
}

#[test]
fn tanh() {
    let input = crate::ones((2, 2));
    let tanh = input.tanh();

    assert_eq!(tanh.history.len(), 1);
}

#[test]
fn tanh_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let tanh = input.tanh();

    assert_eq!(tanh.history.len(), 1);
}

#[test]
fn ln() {
    let input = crate::ones((2, 2));
    let ln = input.ln();

    assert_eq!(ln.history.len(), 1);
}

#[test]
fn ln_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let ln = input.ln();

    assert_eq!(ln.history.len(), 1);
}

#[test]
fn exp() {
    let input = crate::ones((2, 2));
    let exp = input.exp();

    assert_eq!(exp.history.len(), 1);
}

#[test]
fn exp_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let exp = input.exp();

    assert_eq!(exp.history.len(), 1);
}

#[test]
fn softmax() {
    let input = crate::ones((2, 2));
    let softmax = input.softmax(1);

    assert_eq!(softmax.history.len(), 1);
}

#[test]
fn softmax_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let softmax = input.softmax(1);

    assert_eq!(softmax.history.len(), 1);
}

#[test]
fn log_softmax() {
    let input = crate::ones((2, 2));
    let log_softmax = input.log_softmax(1);

    assert_eq!(log_softmax.history.len(), 1);
}

#[test]
fn log_softmax_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let log_softmax = input.log_softmax(1);

    assert_eq!(log_softmax.history.len(), 1);
}

#[test]
fn t() {
    let input = crate::ones((2, 2));
    let t = input.t();

    assert_eq!(t.history.len(), 1);
}

#[test]
fn t_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let t = input.t();

    assert_eq!(t.history.len(), 1);
}

#[test]
fn dropout() {
    let input = crate::ones((2, 2));
    let status = Rc::new(Cell::default());
    let dropout = input.dropout(0.5, status);

    assert_eq!(dropout.history.len(), 1);
}

#[test]
fn dropout_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let status = Rc::new(Cell::default());
    let dropout = input.dropout(0.5, status);

    assert_eq!(dropout.history.len(), 1);
}

#[test]
fn chunks() {
    let input = crate::ones((2, 2));
    let chunks = input.chunks((1, 1));

    assert_eq!(chunks[0].history.len(), 1);
    assert_eq!(chunks[1].history.len(), 1);
}

#[test]
fn chunks_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let chunks = input.chunks((1, 1));

    assert_eq!(chunks[0].history.len(), 1);
    assert_eq!(chunks[1].history.len(), 1);
}

#[test]
fn unsqueeze() {
    let input = crate::ones((2, 2));
    let unsqueeze = input.unsqueeze(0);

    assert_eq!(unsqueeze.history.len(), 1);
}

#[test]
fn unsqueeze_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let unsqueeze = input.unsqueeze(0);

    assert_eq!(unsqueeze.history.len(), 1);
}

#[test]
fn cat() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let cat = super::Cat::cat(lhs, rhs, 1);

    assert_eq!(cat.history.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let cat = super::Cat::cat(lhs, rhs, 1);

    assert_eq!(cat.history.len(), 1);
}

#[test]
fn cat_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let cat = super::Cat::cat(lhs, rhs, 1);

    assert_eq!(cat.history.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let cat = super::Cat::cat(lhs, rhs, 1);

    assert_eq!(cat.history.len(), 1);
}

#[test]
fn multi_cat() {
    let a = crate::ones((2, 2)) + 1.;
    let b = 18. / crate::full((2, 2), 9.);
    let c = crate::full((2, 1), 3.) * 4.;

    let d = a.cat(&[b, c], 1);
    assert_eq!(d.history.len(), 4);
}

#[test]
fn multi_cat_diff() {
    let a = crate::ones((2, 2)).requires_grad() + 1.;
    let b = 18. / crate::full((2, 2), 9.).requires_grad();
    let c = crate::full((2, 1), 3.).requires_grad() * 4.;

    let d = a.cat(&[b, c], 1);
    assert_eq!(d.history.len(), 4);
}

#[test]
fn stack() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let stack = super::Stack::stack(lhs, rhs, 1);

    assert_eq!(stack.history.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let cat = super::Stack::stack(lhs, rhs, 1);

    assert_eq!(cat.history.len(), 1);
}

#[test]
fn stack_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let cat = super::Stack::stack(lhs, rhs, 1);

    assert_eq!(cat.history.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let stack = super::Stack::stack(lhs, rhs, 1);

    assert_eq!(stack.history.len(), 1);
}

#[test]
fn multi_stack() {
    let a = crate::ones((2, 2)) + 1.;
    let b = 18. / crate::full((2, 2), 9.);
    let c = crate::full((2, 2), 3.) * 4.;

    let d = a.stack(&[b, c], 0);
    d.forward();

    print!("{d}");
    //assert_eq!(d.history.len(), 4);
}

#[test]
fn multi_stack_diff() {
    let a = crate::ones((2, 2)).requires_grad() + 1.;
    let b = 18. / crate::full((2, 2), 9.).requires_grad();
    let c = crate::full((2, 2), 3.).requires_grad() * 4.;

    let d = a.stack(&[b, c], 0);
    assert_eq!(d.history.len(), 4);
}

#[test]
fn neg() {
    let input = crate::ones((2, 2));
    let neg = -input;

    assert_eq!(neg.history.len(), 1);
}

#[test]
fn neg_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let neg = -input;

    assert_eq!(neg.history.len(), 1);
}

#[test]
fn add() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let add = lhs + rhs;

    assert_eq!(add.history.len(), 1);
}

#[test]
fn add_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let add = lhs + rhs;

    assert_eq!(add.history.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let add = lhs + rhs;

    assert_eq!(add.history.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let add = lhs + rhs;

    assert_eq!(add.history.len(), 1);
}

#[test]
fn sub() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let sub = lhs - rhs;

    assert_eq!(sub.history.len(), 1);
}

#[test]
fn sub_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let sub = lhs - rhs;

    assert_eq!(sub.history.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let sub = lhs - rhs;

    assert_eq!(sub.history.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let sub = lhs - rhs;

    assert_eq!(sub.history.len(), 1);
}

#[test]
fn mul() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let mul = lhs * rhs;

    assert_eq!(mul.history.len(), 1);
}

#[test]
fn mul_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let mul = lhs * rhs;

    assert_eq!(mul.history.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let mul = lhs * rhs;

    assert_eq!(mul.history.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let mul = lhs * rhs;

    assert_eq!(mul.history.len(), 1);
}

#[test]
fn div() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let div = lhs / rhs;

    assert_eq!(div.history.len(), 1);
}

#[test]
fn div_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let div = lhs / rhs;

    assert_eq!(div.history.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let div = lhs / rhs;

    assert_eq!(div.history.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let div = lhs / rhs;

    assert_eq!(div.history.len(), 1);
}

#[test]
fn vv() {
    let lhs = crate::ones(2);
    let rhs = crate::zeros(2);
    let vv = lhs.vv(rhs);

    assert_eq!(vv.history.len(), 1);
}

#[test]
fn vv_diff() {
    let lhs = crate::ones(2);
    let rhs = crate::zeros(2).requires_grad();
    let vv = lhs.vv(rhs);

    assert_eq!(vv.history.len(), 1);

    let lhs = crate::ones(2).requires_grad();
    let rhs = crate::zeros(2);
    let vv = lhs.vv(rhs);

    assert_eq!(vv.history.len(), 1);

    let lhs = crate::ones(2).requires_grad();
    let rhs = crate::zeros(2).requires_grad();
    let vv = lhs.vv(rhs);

    assert_eq!(vv.history.len(), 1);
}

#[test]
fn vm() {
    let lhs = crate::ones(2);
    let rhs = crate::zeros((2, 2));
    let vm = lhs.vm(rhs);

    assert_eq!(vm.history.len(), 1);
}

#[test]
fn vm_diff() {
    let lhs = crate::ones(2);
    let rhs = crate::zeros((2, 2)).requires_grad();
    let vm = lhs.vm(rhs);

    assert_eq!(vm.history.len(), 1);

    let lhs = crate::ones(2).requires_grad();
    let rhs = crate::zeros((2, 2));
    let vm = lhs.vm(rhs);

    assert_eq!(vm.history.len(), 1);

    let lhs = crate::ones(2).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let vm = lhs.vm(rhs);

    assert_eq!(vm.history.len(), 1);
}

#[test]
fn mv() {
    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones(2);
    let mv = lhs.mv(rhs);

    assert_eq!(mv.history.len(), 1);
}

#[test]
fn mv_diff() {
    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones(2);
    let mv = lhs.mv(rhs);

    assert_eq!(mv.history.len(), 1);

    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones(2).requires_grad();
    let mv = lhs.mv(rhs);

    assert_eq!(mv.history.len(), 1);

    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones(2).requires_grad();
    let mv = lhs.mv(rhs);

    assert_eq!(mv.history.len(), 1);
}

#[test]
fn mm() {
    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones((2, 2));
    let mm = lhs.mm(rhs);

    assert_eq!(mm.history.len(), 1);
}

#[test]
fn mm_diff() {
    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones((2, 2));
    let mm = lhs.mm(rhs);

    assert_eq!(mm.history.len(), 1);

    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones((2, 2)).requires_grad();
    let mm = lhs.mm(rhs);

    assert_eq!(mm.history.len(), 1);

    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones((2, 2)).requires_grad();
    let mm = lhs.mm(rhs);

    assert_eq!(mm.history.len(), 1);
}

#[test]
fn mm_t() {
    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones((2, 2));
    let mm_t = lhs.mm_t(rhs);

    assert_eq!(mm_t.history.len(), 1);
}

#[test]
fn mm_t_diff() {
    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones((2, 2));
    let mm_t = lhs.mm_t(rhs);

    assert_eq!(mm_t.history.len(), 1);

    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones((2, 2)).requires_grad();
    let mm_t = lhs.mm_t(rhs);

    assert_eq!(mm_t.history.len(), 1);

    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones((2, 2)).requires_grad();
    let mm_t = lhs.mm_t(rhs);

    assert_eq!(mm_t.history.len(), 1);
}

// #[test]
// fn convolve() {
//     use crate::Convolve;

//     let kernel = crate::zeros((2, 2, 2, 2));
//     let input = crate::ones((4, 2, 6, 6));
//     let convolve = super::Var::convolve(
//         input,
//         kernel,
//         &[1, 1],
//         &[1, 1],
//         &[0, 0],
//         crate::variable::Zero,
//     );

//     assert_eq!(convolve.history.len(), 1);
//     assert!(convolve.history.changeables.is_empty());
// }

// #[test]
// fn convolve_diff() {
//     use crate::Convolve;

//     let kernel = crate::zeros((2, 2, 2, 2)).requires_grad();
//     let input = crate::ones((4, 2, 6, 6));
//     let convolve = super::Var::convolve(
//         input,
//         kernel,
//         &[1, 1],
//         &[1, 1],
//         &[0, 0],
//         crate::variable::Zero,
//     );

//     assert_eq!(convolve.history.len(), 1);
//     assert_eq!(convolve.history.parameters.len(), 1);

//     let kernel = crate::zeros((2, 2, 2, 2)).requires_grad();
//     let input = crate::ones((4, 2, 6, 6)).requires_grad();
//     let convolve = super::VarDiff::convolve(
//         input,
//         kernel,
//         &[1, 1],
//         &[1, 1],
//         &[0, 0],
//         crate::variable::Zero,
//     );

//     assert_eq!(convolve.history.len(), 1);
//     assert_eq!(convolve.history.parameters.len(), 2);
// }

// #[test]
// fn convolve_groups() {
//     use crate::ConvolveWithGroups;

//     let kernel = crate::zeros((2, 2, 2, 2));
//     let input = crate::ones((4, 2, 6, 6));
//     let convolve = super::Var::convolve_with_groups(
//         input,
//         kernel,
//         &[1, 1],
//         &[1, 1],
//         &[0, 0],
//         crate::variable::Zero,
//         2,
//     );

//     assert_eq!(convolve.history.len(), 1);
//     assert!(convolve.history.changeables.is_empty());
// }

// #[test]
// fn convolve_groups_diff() {
//     use crate::ConvolveWithGroups;

//     let kernel = crate::zeros((2, 2, 2, 2)).requires_grad();
//     let input = crate::ones((4, 2, 6, 6));
//     let convolve = super::Var::convolve_with_groups(
//         input,
//         kernel,
//         &[1, 1],
//         &[1, 1],
//         &[0, 0],
//         crate::variable::Zero,
//         2,
//     );

//     assert_eq!(convolve.history.len(), 1);
//     assert_eq!(convolve.history.parameters.len(), 1);

//     let kernel = crate::zeros((2, 2, 2, 2)).requires_grad();
//     let input = crate::ones((4, 2, 6, 6)).requires_grad();
//     let convolve = super::VarDiff::convolve_with_groups(
//         input,
//         kernel,
//         &[1, 1],
//         &[1, 1],
//         &[0, 0],
//         crate::variable::Zero,
//         2,
//     );

//     assert_eq!(convolve.history.len(), 1);
//     assert_eq!(convolve.history.parameters.len(), 2);
// }
