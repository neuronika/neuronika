use super::{super::L2, SGD};

#[test]
fn creation() {
    let optim = SGD::new(Vec::new(), 1e-2, L2::new(1e-2));

    assert_eq!(optim.params.borrow().len(), 0);
    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);

    let optim = optim.with_momentum(0.5, 0.0, true);

    assert_eq!(optim.params.borrow().len(), 0);
    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert!((optim.get_momentum() - 0.5).abs() <= f32::EPSILON);
    assert!(optim.get_dampening().abs() <= f32::EPSILON);
    assert!(optim.get_nesterov());
}

#[test]
fn set_lr() {
    let optim = SGD::new(Vec::new(), 1e-2, L2::new(1e-2));
    optim.set_lr(1e-3);

    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);

    let optim = SGD::new(Vec::new(), 1e-2, L2::new(1e-2)).with_momentum(0.5, 0.0, true);
    optim.set_lr(1e-3);

    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);
}

#[test]
fn set_dampening() {
    let optim = SGD::new(Vec::new(), 1e-2, L2::new(1e-2)).with_momentum(0.5, 0.0, true);
    optim.set_dampening(1.0);

    assert!((optim.get_dampening() - 1.0).abs() <= f32::EPSILON);
}

#[test]
fn set_momentum() {
    let optim = SGD::new(Vec::new(), 1e-2, L2::new(1e-2)).with_momentum(0.5, 0.0, true);
    optim.set_momentum(0.3);

    assert!((optim.get_momentum() - 0.3).abs() <= f32::EPSILON);
}

#[test]
fn set_nesterov() {
    let optim = SGD::new(Vec::new(), 1e-2, L2::new(1e-2)).with_momentum(0.5, 0.0, false);
    optim.set_nesterov(true);

    assert!(optim.get_nesterov());
}

const EPOCHS: usize = 200;

#[test]
fn step() {
    // SGD.
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y);

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.data()[0];
    let optim = SGD::new(loss.parameters(), 0.1, L2::new(0.));

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < first_value);
}

#[test]
fn step_with_momentum() {
    // SGD with momentum.
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y);

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.data()[0];
    let optim = SGD::new(loss.parameters(), 0.1, L2::new(0.)).with_momentum(0.7, 0.0, false);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < first_value);
}

#[test]
fn step_with_nesterov_momentum() {
    // SGD with momentum.
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y);

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.data()[0];
    let optim = SGD::new(loss.parameters(), 0.1, L2::new(0.)).with_momentum(0.7, 0.0, true);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < first_value);
}
