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
    let mut y = x + 1.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // f32 - Var
    let x = crate::ones((2, 2));
    let mut y = 1. + x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // VarDiff - f32
    let x = crate::ones((2, 2)).requires_grad();
    let mut y = x + 1.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // f32 - VarDiff
    let x = crate::ones((2, 2)).requires_grad();
    let mut y = 1. + x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);
}

#[test]
fn param_test() {
    use super::Param;

    let mut data = ndarray::array![[1., 2.], [3., 4.]];
    let mut grad = ndarray::array![[4., 5.], [6., 7.]];

    let param = Param::new(data.as_mut_ptr(), grad.as_mut_ptr(), data.shape().to_vec());

    let (mut data_view, mut grad_view) = param.get();
    data_view += 1.;
    grad_view += 1.;

    assert_eq!(data, ndarray::array![[2., 3.], [4., 5.]]);
    assert_eq!(grad, ndarray::array![[5., 6.], [7., 8.]]);
}

#[test]
fn sub_scalar() {
    // Var - f32
    let x = crate::ones((2, 2));
    let mut y = x - 1.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[0., 0.], [0., 0.]]);

    // f32 - Var
    let x = crate::ones((2, 2));
    let mut y = 1. - x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[0., 0.], [0., 0.]]);

    // VarDiff - f32
    let x = crate::ones((2, 2)).requires_grad();
    let mut y = x - 1.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[0., 0.], [0., 0.]]);

    // f32 - VarDiff
    let x = crate::ones((2, 2)).requires_grad();
    let mut y = 1. - x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[0., 0.], [0., 0.]]);
}

#[test]
fn mul_scalar() {
    // Var - f32
    let x = crate::ones((2, 2));
    let mut y = x * 2.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // f32 - Var
    let x = crate::ones((2, 2));
    let mut y = 2. * x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // VarDiff - f32
    let x = crate::ones((2, 2)).requires_grad();
    let mut y = x * 2.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);

    // f32 - VarDiff
    let x = crate::ones((2, 2)).requires_grad();
    let mut y = 2. * x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[2., 2.], [2., 2.]]);
}

#[test]
fn div_scalar() {
    // Var - f32
    let x = crate::full((2, 2), 9.);
    let mut y = x / 3.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[3., 3.], [3., 3.]]);

    // f32 - Var
    let x = crate::full((2, 2), 3.);
    let mut y = 9. / x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[3., 3.], [3., 3.]]);

    // VarDiff - f32
    let x = crate::full((2, 2), 9.).requires_grad();
    let mut y = x / 3.;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[3., 3.], [3., 3.]]);

    // f32 - VarDiff
    let x = crate::full((2, 2), 3.).requires_grad();
    let mut y = 9. / x;
    y.forward();

    assert_eq!(*y.data(), ndarray::array![[3., 3.], [3., 3.]]);
}

#[test]
fn parameters_test() {
    let x = crate::rand((2, 2)).requires_grad();
    let y = crate::rand((2, 2)).requires_grad();
    let z = crate::rand((1, 1)).requires_grad();

    let w = x.clone() + y + z + x;

    assert_eq!(w.parameters().len(), 3);
}

#[test]
fn sum() {
    let input = crate::ones((2, 2));
    let sum = input.sum();

    assert_eq!(sum.past.len(), 1);
    assert!(sum.past.changeables.is_empty());
}

#[test]
fn sum_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let sum = input.sum();

    assert_eq!(sum.past.len(), 1);
    assert_eq!(sum.past.parameters.len(), 1);
}

#[test]
fn mean() {
    let input = crate::ones((2, 2));
    let mean = input.mean();

    assert_eq!(mean.past.len(), 1);
    assert!(mean.past.changeables.is_empty());
}

#[test]
fn mean_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let mean = input.mean();

    assert_eq!(mean.past.len(), 1);
    assert_eq!(mean.past.parameters.len(), 1);
}

#[test]
fn pow() {
    let input = crate::ones((2, 2));
    let pow = input.pow(2);

    assert_eq!(pow.past.len(), 1);
    assert!(pow.past.changeables.is_empty());
}

#[test]
fn pow_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let pow = input.pow(2);

    assert_eq!(pow.past.len(), 1);
    assert_eq!(pow.past.parameters.len(), 1);
}

#[test]
fn sqrt() {
    let input = crate::ones((2, 2));
    let relu = input.sqrt();

    assert_eq!(relu.past.len(), 1);
    assert!(relu.past.changeables.is_empty());
}

#[test]
fn sqrt_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let relu = input.sqrt();

    assert_eq!(relu.past.len(), 1);
    assert_eq!(relu.past.parameters.len(), 1);
}

#[test]
fn relu() {
    let input = crate::ones((2, 2));
    let relu = input.relu();

    assert_eq!(relu.past.len(), 1);
    assert!(relu.past.changeables.is_empty());
}

#[test]
fn relu_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let relu = input.relu();

    assert_eq!(relu.past.len(), 1);
    assert_eq!(relu.past.parameters.len(), 1);
}

#[test]
fn leaky_relu() {
    let input = crate::ones((2, 2));
    let leaky_relu = input.leaky_relu();

    assert_eq!(leaky_relu.past.len(), 1);
    assert!(leaky_relu.past.changeables.is_empty());
}

#[test]
fn leaky_relu_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let leaky_relu = input.leaky_relu();

    assert_eq!(leaky_relu.past.len(), 1);
    assert_eq!(leaky_relu.past.parameters.len(), 1);
}

#[test]
fn softplus() {
    let input = crate::ones((2, 2));
    let softplus = input.softplus();

    assert_eq!(softplus.past.len(), 1);
    assert!(softplus.past.changeables.is_empty());
}

#[test]
fn softplus_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let softplus = input.softplus();

    assert_eq!(softplus.past.len(), 1);
    assert_eq!(softplus.past.parameters.len(), 1);
}

#[test]
fn sigmoid() {
    let input = crate::ones((2, 2));
    let sigmoid = input.sigmoid();

    assert_eq!(sigmoid.past.len(), 1);
    assert!(sigmoid.past.changeables.is_empty());
}

#[test]
fn sigmoid_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let sigmoid = input.sigmoid();

    assert_eq!(sigmoid.past.len(), 1);
    assert_eq!(sigmoid.past.parameters.len(), 1);
}

#[test]
fn tanh() {
    let input = crate::ones((2, 2));
    let tanh = input.tanh();

    assert_eq!(tanh.past.len(), 1);
    assert!(tanh.past.changeables.is_empty());
}

#[test]
fn tanh_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let tanh = input.tanh();

    assert_eq!(tanh.past.len(), 1);
    assert_eq!(tanh.past.parameters.len(), 1);
}

#[test]
fn ln() {
    let input = crate::ones((2, 2));
    let ln = input.ln();

    assert_eq!(ln.past.len(), 1);
    assert!(ln.past.changeables.is_empty());
}

#[test]
fn ln_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let ln = input.ln();

    assert_eq!(ln.past.len(), 1);
    assert_eq!(ln.past.parameters.len(), 1);
}

#[test]
fn exp() {
    let input = crate::ones((2, 2));
    let exp = input.exp();

    assert_eq!(exp.past.len(), 1);
    assert!(exp.past.changeables.is_empty());
}

#[test]
fn exp_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let exp = input.exp();

    assert_eq!(exp.past.len(), 1);
    assert_eq!(exp.past.parameters.len(), 1);
}

#[test]
fn softmax() {
    let input = crate::ones((2, 2));
    let softmax = input.softmax(1);

    assert_eq!(softmax.past.len(), 1);
    assert!(softmax.past.changeables.is_empty());
}

#[test]
fn softmax_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let softmax = input.softmax(1);

    assert_eq!(softmax.past.len(), 1);
    assert_eq!(softmax.past.parameters.len(), 1);
}

#[test]
fn log_softmax() {
    let input = crate::ones((2, 2));
    let log_softmax = input.log_softmax(1);

    assert_eq!(log_softmax.past.len(), 1);
    assert!(log_softmax.past.changeables.is_empty());
}

#[test]
fn log_softmax_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let log_softmax = input.log_softmax(1);

    assert_eq!(log_softmax.past.len(), 1);
    assert_eq!(log_softmax.past.parameters.len(), 1);
}

#[test]
fn t() {
    let input = crate::ones((2, 2));
    let t = input.t();

    assert_eq!(t.past.len(), 1);
    assert!(t.past.changeables.is_empty());
}

#[test]
fn t_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let t = input.t();

    assert_eq!(t.past.len(), 1);
    assert_eq!(t.past.parameters.len(), 1);
}

#[test]
fn dropout() {
    let input = crate::ones((2, 2));
    let dropout = input.dropout(0.5);

    assert_eq!(dropout.past.len(), 1);
    assert_eq!(dropout.past.changeables.len(), 1);
}

#[test]
fn dropout_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let dropout = input.dropout(0.5);

    assert_eq!(dropout.past.len(), 1);
    assert_eq!(dropout.past.parameters.len(), 1);
}

#[test]
fn chunks() {
    let input = crate::ones((2, 2));
    let chunks = input.chunks((1, 1));

    assert_eq!(chunks[0].past.len(), 1);
    assert_eq!(chunks[1].past.len(), 1);

    assert!(chunks[0].past.changeables.is_empty());
    assert!(chunks[1].past.changeables.is_empty());
}

#[test]
fn chunks_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let chunks = input.chunks((1, 1));

    assert_eq!(chunks[0].past.len(), 1);
    assert_eq!(chunks[1].past.len(), 1);

    assert_eq!(chunks[0].past.parameters.len(), 1);
    assert_eq!(chunks[1].past.parameters.len(), 1);
}

#[test]
fn unsqueeze() {
    let input = crate::ones((2, 2));
    let unsqueeze = input.unsqueeze(0);

    assert_eq!(unsqueeze.past.len(), 1);
    assert!(unsqueeze.past.changeables.is_empty());
}

#[test]
fn unsqueeze_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let unsqueeze = input.unsqueeze(0);

    assert_eq!(unsqueeze.past.len(), 1);
    assert_eq!(unsqueeze.past.parameters.len(), 1);
}

#[test]
fn cat() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let cat = super::Cat::cat(lhs, rhs, 1);

    assert_eq!(cat.past.len(), 1);
    assert!(cat.past.changeables.is_empty());

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let cat = super::Cat::cat(lhs, rhs, 1);

    assert_eq!(cat.past.len(), 1);
    assert_eq!(cat.past.parameters.len(), 1);
}

#[test]
fn cat_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let cat = super::Cat::cat(lhs, rhs, 1);

    assert_eq!(cat.past.len(), 1);
    assert_eq!(cat.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let cat = super::Cat::cat(lhs, rhs, 1);

    assert_eq!(cat.past.len(), 1);
    assert_eq!(cat.past.parameters.len(), 2);
}

#[test]
fn multi_cat() {
    use std::boxed::Box;

    let a = crate::ones((2, 2)) + 1.;
    let b = 18. / crate::full((2, 2), 9.);
    let c = crate::full((2, 1), 3.) * 4.;

    let d = a.cat(&[Box::new(b), Box::new(c)], 1);
    assert_eq!(d.past.len(), 4);
    assert!(d.past.changeables.is_empty());
}

#[test]
fn multi_cat_diff() {
    use std::boxed::Box;

    let a = crate::ones((2, 2)).requires_grad() + 1.;
    let b = 18. / crate::full((2, 2), 9.).requires_grad();
    let c = crate::full((2, 1), 3.).requires_grad() * 4.;

    let d = a.cat(&[Box::new(b), Box::new(c)], 1);
    assert_eq!(d.past.len(), 4);
    assert_eq!(d.past.parameters.len(), 3);
}

#[test]
fn stack() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let stack = super::Stack::stack(lhs, rhs, 1);

    assert_eq!(stack.past.len(), 1);
    assert!(stack.past.changeables.is_empty());

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let cat = super::Stack::stack(lhs, rhs, 1);

    assert_eq!(cat.past.len(), 1);
    assert_eq!(cat.past.parameters.len(), 1);
}

#[test]
fn stack_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let cat = super::Stack::stack(lhs, rhs, 1);

    assert_eq!(cat.past.len(), 1);
    assert_eq!(cat.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let stack = super::Stack::stack(lhs, rhs, 1);

    assert_eq!(stack.past.len(), 1);
    assert_eq!(stack.past.parameters.len(), 2);
}

#[test]
fn multi_stack() {
    use std::boxed::Box;

    let a = crate::ones((2, 2)) + 1.;
    let b = 18. / crate::full((2, 2), 9.);
    let c = crate::full((2, 2), 3.) * 4.;

    let d = a.stack(&[Box::new(b), Box::new(c)], 0);
    assert_eq!(d.past.len(), 4);
    assert!(d.past.changeables.is_empty());
}

#[test]
fn multi_stack_diff() {
    use std::boxed::Box;

    let a = crate::ones((2, 2)).requires_grad() + 1.;
    let b = 18. / crate::full((2, 2), 9.).requires_grad();
    let c = crate::full((2, 2), 3.).requires_grad() * 4.;

    let d = a.stack(&[Box::new(b), Box::new(c)], 0);
    assert_eq!(d.past.len(), 4);
    assert_eq!(d.past.parameters.len(), 3);
}

#[test]
fn neg() {
    let input = crate::ones((2, 2));
    let neg = -input;

    assert_eq!(neg.past.len(), 1);
    assert!(neg.past.changeables.is_empty());
}

#[test]
fn neg_diff() {
    let input = crate::ones((2, 2)).requires_grad();
    let neg = -input;

    assert_eq!(neg.past.len(), 1);
    assert_eq!(neg.past.parameters.len(), 1);
}

#[test]
fn add() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let add = lhs + rhs;

    assert_eq!(add.past.len(), 1);
    assert!(add.past.changeables.is_empty());
}

#[test]
fn add_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let add = lhs + rhs;

    assert_eq!(add.past.len(), 1);
    assert_eq!(add.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let add = lhs + rhs;

    assert_eq!(add.past.len(), 1);
    assert_eq!(add.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let add = lhs + rhs;

    assert_eq!(add.past.len(), 1);
    assert_eq!(add.past.parameters.len(), 2);
}

#[test]
fn sub() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let sub = lhs - rhs;

    assert_eq!(sub.past.len(), 1);
    assert!(sub.past.changeables.is_empty());
}

#[test]
fn sub_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let sub = lhs - rhs;

    assert_eq!(sub.past.len(), 1);
    assert_eq!(sub.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let sub = lhs - rhs;

    assert_eq!(sub.past.len(), 1);
    assert_eq!(sub.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let sub = lhs - rhs;

    assert_eq!(sub.past.len(), 1);
    assert_eq!(sub.past.parameters.len(), 2);
}

#[test]
fn mul() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let mul = lhs * rhs;

    assert_eq!(mul.past.len(), 1);
    assert!(mul.past.changeables.is_empty());
}

#[test]
fn mul_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let mul = lhs * rhs;

    assert_eq!(mul.past.len(), 1);
    assert_eq!(mul.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let mul = lhs * rhs;

    assert_eq!(mul.past.len(), 1);
    assert_eq!(mul.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let mul = lhs * rhs;

    assert_eq!(mul.past.len(), 1);
    assert_eq!(mul.past.parameters.len(), 2);
}

#[test]
fn div() {
    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2));
    let div = lhs / rhs;

    assert_eq!(div.past.len(), 1);
    assert!(div.past.changeables.is_empty());
}

#[test]
fn div_diff() {
    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2));
    let div = lhs / rhs;

    assert_eq!(div.past.len(), 1);
    assert_eq!(div.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2));
    let rhs = crate::zeros((2, 2)).requires_grad();
    let div = lhs / rhs;

    assert_eq!(div.past.len(), 1);
    assert_eq!(div.past.parameters.len(), 1);

    let lhs = crate::ones((2, 2)).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let div = lhs / rhs;

    assert_eq!(div.past.len(), 1);
    assert_eq!(div.past.parameters.len(), 2);
}

#[test]
fn vv() {
    let lhs = crate::ones(2);
    let rhs = crate::zeros(2);
    let vv = lhs.vv(rhs);

    assert_eq!(vv.past.len(), 1);
    assert!(vv.past.changeables.is_empty());
}

#[test]
fn vv_diff() {
    let lhs = crate::ones(2);
    let rhs = crate::zeros(2).requires_grad();
    let vv = lhs.vv(rhs);

    assert_eq!(vv.past.len(), 1);
    assert_eq!(vv.past.parameters.len(), 1);

    let lhs = crate::ones(2).requires_grad();
    let rhs = crate::zeros(2);
    let vv = lhs.vv(rhs);

    assert_eq!(vv.past.len(), 1);
    assert_eq!(vv.past.parameters.len(), 1);

    let lhs = crate::ones(2).requires_grad();
    let rhs = crate::zeros(2).requires_grad();
    let vv = lhs.vv(rhs);

    assert_eq!(vv.past.len(), 1);
    assert_eq!(vv.past.parameters.len(), 2);
}

#[test]
fn vm() {
    let lhs = crate::ones(2);
    let rhs = crate::zeros((2, 2));
    let vm = lhs.vm(rhs);

    assert_eq!(vm.past.len(), 1);
    assert!(vm.past.changeables.is_empty());
}

#[test]
fn vm_diff() {
    let lhs = crate::ones(2);
    let rhs = crate::zeros((2, 2)).requires_grad();
    let vm = lhs.vm(rhs);

    assert_eq!(vm.past.len(), 1);
    assert_eq!(vm.past.parameters.len(), 1);

    let lhs = crate::ones(2).requires_grad();
    let rhs = crate::zeros((2, 2));
    let vm = lhs.vm(rhs);

    assert_eq!(vm.past.len(), 1);
    assert_eq!(vm.past.parameters.len(), 1);

    let lhs = crate::ones(2).requires_grad();
    let rhs = crate::zeros((2, 2)).requires_grad();
    let vm = lhs.vm(rhs);

    assert_eq!(vm.past.len(), 1);
    assert_eq!(vm.past.parameters.len(), 2);
}

#[test]
fn mv() {
    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones(2);
    let mv = lhs.mv(rhs);

    assert_eq!(mv.past.len(), 1);
    assert!(mv.past.changeables.is_empty());
}

#[test]
fn mv_diff() {
    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones(2);
    let mv = lhs.mv(rhs);

    assert_eq!(mv.past.len(), 1);
    assert_eq!(mv.past.parameters.len(), 1);

    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones(2).requires_grad();
    let mv = lhs.mv(rhs);

    assert_eq!(mv.past.len(), 1);
    assert_eq!(mv.past.parameters.len(), 1);

    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones(2).requires_grad();
    let mv = lhs.mv(rhs);

    assert_eq!(mv.past.len(), 1);
    assert_eq!(mv.past.parameters.len(), 2);
}

#[test]
fn mm() {
    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones((2, 2));
    let mm = lhs.mm(rhs);

    assert_eq!(mm.past.len(), 1);
    assert!(mm.past.changeables.is_empty());
}

#[test]
fn mm_diff() {
    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones((2, 2));
    let mm = lhs.mm(rhs);

    assert_eq!(mm.past.len(), 1);
    assert_eq!(mm.past.parameters.len(), 1);

    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones((2, 2)).requires_grad();
    let mm = lhs.mm(rhs);

    assert_eq!(mm.past.len(), 1);
    assert_eq!(mm.past.parameters.len(), 1);

    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones((2, 2)).requires_grad();
    let mm = lhs.mm(rhs);

    assert_eq!(mm.past.len(), 1);
    assert_eq!(mm.past.parameters.len(), 2);
}

#[test]
fn mm_t() {
    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones((2, 2));
    let mm_t = lhs.mm_t(rhs);

    assert_eq!(mm_t.past.len(), 1);
    assert!(mm_t.past.changeables.is_empty());
}

#[test]
fn mm_t_diff() {
    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones((2, 2));
    let mm_t = lhs.mm_t(rhs);

    assert_eq!(mm_t.past.len(), 1);
    assert_eq!(mm_t.past.parameters.len(), 1);

    let lhs = crate::zeros((2, 2));
    let rhs = crate::ones((2, 2)).requires_grad();
    let mm_t = lhs.mm_t(rhs);

    assert_eq!(mm_t.past.len(), 1);
    assert_eq!(mm_t.past.parameters.len(), 1);

    let lhs = crate::zeros((2, 2)).requires_grad();
    let rhs = crate::ones((2, 2)).requires_grad();
    let mm_t = lhs.mm_t(rhs);

    assert_eq!(mm_t.past.len(), 1);
    assert_eq!(mm_t.past.parameters.len(), 2);
}

#[test]
fn convolve() {
    use crate::Convolve;

    let kernel = crate::zeros((2, 2, 2, 2));
    let input = crate::ones((4, 2, 6, 6));
    let convolve = super::Var::convolve(
        input,
        kernel,
        &[1, 1],
        &[1, 1],
        &[0, 0],
        crate::variable::Zero,
    );

    assert_eq!(convolve.past.len(), 1);
    assert!(convolve.past.changeables.is_empty());
}

#[test]
fn convolve_diff() {
    use crate::Convolve;

    let kernel = crate::zeros((2, 2, 2, 2)).requires_grad();
    let input = crate::ones((4, 2, 6, 6));
    let convolve = super::Var::convolve(
        input,
        kernel,
        &[1, 1],
        &[1, 1],
        &[0, 0],
        crate::variable::Zero,
    );

    assert_eq!(convolve.past.len(), 1);
    assert_eq!(convolve.past.parameters.len(), 1);

    let kernel = crate::zeros((2, 2, 2, 2)).requires_grad();
    let input = crate::ones((4, 2, 6, 6)).requires_grad();
    let convolve = super::VarDiff::convolve(
        input,
        kernel,
        &[1, 1],
        &[1, 1],
        &[0, 0],
        crate::variable::Zero,
    );

    assert_eq!(convolve.past.len(), 1);
    assert_eq!(convolve.past.parameters.len(), 2);
}

#[test]
fn convolve_groups() {
    use crate::ConvolveWithGroups;

    let kernel = crate::zeros((2, 2, 2, 2));
    let input = crate::ones((4, 2, 6, 6));
    let convolve = super::Var::convolve_with_groups(
        input,
        kernel,
        &[1, 1],
        &[1, 1],
        &[0, 0],
        crate::variable::Zero,
        2,
    );

    assert_eq!(convolve.past.len(), 1);
    assert!(convolve.past.changeables.is_empty());
}

#[test]
fn convolve_groups_diff() {
    use crate::ConvolveWithGroups;

    let kernel = crate::zeros((2, 2, 2, 2)).requires_grad();
    let input = crate::ones((4, 2, 6, 6));
    let convolve = super::Var::convolve_with_groups(
        input,
        kernel,
        &[1, 1],
        &[1, 1],
        &[0, 0],
        crate::variable::Zero,
        2,
    );

    assert_eq!(convolve.past.len(), 1);
    assert_eq!(convolve.past.parameters.len(), 1);

    let kernel = crate::zeros((2, 2, 2, 2)).requires_grad();
    let input = crate::ones((4, 2, 6, 6)).requires_grad();
    let convolve = super::VarDiff::convolve_with_groups(
        input,
        kernel,
        &[1, 1],
        &[1, 1],
        &[0, 0],
        crate::variable::Zero,
        2,
    );

    assert_eq!(convolve.past.len(), 1);
    assert_eq!(convolve.past.parameters.len(), 2);
}
