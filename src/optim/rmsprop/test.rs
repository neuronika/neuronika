use super::{super::L2, RMSProp};

#[test]
fn creation() {
    let optim = RMSProp::new(1e-2, L2::new(1e-2), 1e-3, None, false, 1e-8);

    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert!((optim.status().get_alpha().unwrap() - 1e-3).abs() <= f32::EPSILON);
}

#[test]
#[should_panic]
fn creation_invalid() {
    let _ = RMSProp::new(1e-2, L2::new(1e-2), 2.0, None, false, 1e-8);
}

#[test]
fn set_lr() {
    let optim = RMSProp::new(1e-2, L2::new(1e-2), 1e-3, None, false, 1e-8);

    optim.set_lr(1e-3);
    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);
}

#[test]
fn set_alpha() {
    let optim = RMSProp::new(1e-2, L2::new(1e-2), 1e-3, None, false, 1e-8);

    optim.status().set_alpha(1e-2);
    assert!((optim.status().get_alpha().unwrap() - 1e-2).abs() <= f32::EPSILON);
}

#[test]
fn set_eps() {
    let optim = RMSProp::new(1e-2, L2::new(1e-2), 1e-3, None, false, 1e-8);

    optim.status().set_eps(1e-9);
    assert!((optim.status().get_eps() - 1e-9).abs() <= f32::EPSILON);
}

#[test]
fn set_momentum() {
    let optim = RMSProp::new(1e-2, L2::new(1e-2), 1e-3, None, false, 1e-8);

    optim.status().set_momentum(0.8);
    assert!((optim.status().get_momentum().unwrap() - 0.8).abs() <= f32::EPSILON);
}

const EPOCHS: usize = 10;

#[test]
fn step() {
    let x = crate::rand((3, 3)).requires_grad();
    let y = crate::rand((3, 3));
    let z = crate::rand((3, 3));

    let loss = (x.clone().mm(y) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.item();
    let optim = RMSProp::new(1e-4, L2::new(0.0), 0.99, None, false, 1e-8);
    optim.register(x.clone());

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }

    assert!(loss.item() < first_value.clone());
}

#[test]
fn step_centered() {
    let x = crate::rand((3, 3)).requires_grad();
    let y = crate::rand((3, 3));
    let z = crate::rand((3, 3));

    let loss = (x.clone().mm(y) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.item();
    let optim = RMSProp::new(1e-4, L2::new(0.0), 0.99, None, true, 1e-8);
    optim.register(x);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }

    assert!(loss.item() < first_value.clone());
}

#[test]
fn step_with_momentum() {
    let x = crate::rand((3, 3)).requires_grad();
    let y = crate::rand((3, 3));
    let z = crate::rand((3, 3));

    let loss = (x.clone().mm(y) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.item();
    let optim = RMSProp::new(1e-4, L2::new(0.0), 0.99, 0.5, false, 1e-8);
    optim.register(x);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }

    assert!(loss.item() < first_value.clone());
}

#[test]
fn step_centered_with_momentum() {
    let x = crate::rand((3, 3)).requires_grad();
    let y = crate::rand((3, 3));
    let z = crate::rand((3, 3));

    let loss = (x.clone().mm(y) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.item();
    let optim = RMSProp::new(1e-4, L2::new(0.0), 0.99, 0.5, true, 1e-8);
    optim.register(x);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }

    assert!(loss.item() < first_value.clone());
}
