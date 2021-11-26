use super::{super::L2, AMSGrad};

#[test]
fn creation() {
    let optim = AMSGrad::new(Vec::new(), 1e-2, (0.9, 0.999), L2::new(1e-2), 1e-8);

    assert_eq!(optim.params.borrow().len(), 0);
    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert_eq!(optim.get_betas(), (0.9, 0.999));
    assert!((optim.get_eps() - 1e-8).abs() <= f32::EPSILON);
}

#[test]
fn set_lr() {
    let optim = AMSGrad::new(Vec::new(), 1e-2, (0.9, 0.999), L2::new(1e-2), 1e-8);

    optim.set_lr(1e-3);
    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);
}

#[test]
fn set_betas() {
    let optim = AMSGrad::new(Vec::new(), 1e-2, (0.9, 0.999), L2::new(1e-2), 1e-8);

    optim.set_betas((0.91, 0.9991));
    assert_eq!(optim.get_betas(), (0.91, 0.9991));
}

#[test]
fn set_eps() {
    let optim = AMSGrad::new(Vec::new(), 1e-2, (0.9, 0.999), L2::new(1e-2), 1e-8);

    optim.set_eps(1e-9);
    assert!((optim.get_eps() - 1e-9).abs() <= f32::EPSILON);
}

const EPOCHS: usize = 200;

#[test]
fn step() {
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y);

    let w = crate::rand((3, 3)).requires_grad();
    let loss = (x.mm(w) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.data()[0];
    let optim = AMSGrad::new(loss.parameters(), 0.01, (0.9, 0.999), L2::new(0.0), 1e-8);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < first_value);
}
