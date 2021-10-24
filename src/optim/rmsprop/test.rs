use super::{super::L2, RMSProp};

#[test]
fn creation() {
    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8);

    assert_eq!(optim.params.borrow().len(), 0);
    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert!((optim.get_alpha() - 1e-3).abs() <= f32::EPSILON);

    let optim = optim.with_momentum(0.5);

    assert_eq!(optim.params.borrow().len(), 0);
    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert!((optim.get_alpha() - 1e-3).abs() <= f32::EPSILON);
    assert!((optim.get_momentum() - 0.5).abs() <= f32::EPSILON);

    let optim = optim.centered();

    assert_eq!(optim.params.borrow().len(), 0);
    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert!((optim.get_alpha() - 1e-3).abs() <= f32::EPSILON);
    assert!((optim.get_momentum() - 0.5).abs() <= f32::EPSILON);
}

#[test]
fn set_lr() {
    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8);

    optim.set_lr(1e-3);
    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);

    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).with_momentum(0.5);

    optim.set_lr(1e-3);
    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);

    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).centered();

    optim.set_lr(1e-3);
    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);

    let optim =
        RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).centered_with_momentum(0.5);

    optim.set_lr(1e-3);
    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);
}

#[test]
fn set_alpha() {
    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8);

    optim.set_alpha(1e-2);
    assert!((optim.get_alpha() - 1e-2).abs() <= f32::EPSILON);

    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).with_momentum(0.5);

    optim.set_alpha(1e-2);
    assert!((optim.get_alpha() - 1e-2).abs() <= f32::EPSILON);

    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).centered();

    optim.set_alpha(1e-2);
    assert!((optim.get_alpha() - 1e-2).abs() <= f32::EPSILON);

    let optim =
        RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).centered_with_momentum(0.5);

    optim.set_alpha(1e-2);
    assert!((optim.get_alpha() - 1e-2).abs() <= f32::EPSILON);
}

#[test]
fn set_eps() {
    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8);

    optim.set_eps(1e-9);
    assert!((optim.get_eps() - 1e-9).abs() <= f32::EPSILON);

    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).with_momentum(0.5);

    optim.set_eps(1e-9);
    assert!((optim.get_eps() - 1e-9).abs() <= f32::EPSILON);

    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).centered();

    optim.set_eps(1e-9);
    assert!((optim.get_eps() - 1e-9).abs() <= f32::EPSILON);

    let optim =
        RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).centered_with_momentum(0.5);

    optim.set_eps(1e-9);
    assert!((optim.get_eps() - 1e-9).abs() <= f32::EPSILON);
}

#[test]
fn set_momentum() {
    let optim = RMSProp::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-8).with_momentum(0.5);

    optim.set_momentum(0.8);
    assert!((optim.get_momentum() - 0.8).abs() <= f32::EPSILON);

    let optim = optim.centered();

    optim.set_momentum(0.5);
    assert!((optim.get_momentum() - 0.5).abs() <= f32::EPSILON);
}

const EPOCHS: usize = 2000;
const TOL: f32 = 1e-3;

#[test]
fn step() {
    // RMSProp.
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y);

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w) - z).pow(2).sum();

    let optim = RMSProp::new(loss.parameters(), 0.01, 0.7, L2::new(0.0), 1e-8);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < TOL);
}

#[test]
fn step_centered() {
    // RMSProp centered.
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y);

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w) - z).pow(2).sum();

    let optim = RMSProp::new(loss.parameters(), 0.01, 0.7, L2::new(0.0), 1e-8).centered();

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < TOL);
}

#[test]
fn step_with_momentum() {
    // RMSProp with momentum.
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y);

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w) - z).pow(2).sum();

    let optim = RMSProp::new(loss.parameters(), 0.001, 0.99, L2::new(0.0), 1e-8).with_momentum(0.5);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < TOL);
}

#[test]
fn step_centered_with_momentum() {
    // RMSProp centered with momentum.
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y);

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w) - z).pow(2).sum();

    let optim = RMSProp::new(loss.parameters(), 0.001, 0.99, L2::new(0.0), 1e-8)
        .centered_with_momentum(0.5);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < TOL);
}
