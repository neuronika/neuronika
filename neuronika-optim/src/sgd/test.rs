use neuronika_variable;

use super::{super::L2, StochasticGD};

#[test]
fn creation() {
    let optim = StochasticGD::new(1e-2, L2::new(1e-2), None, None, false);

    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);

    optim.status().set_momentum(0.5);
    optim.status().set_nesterov(true);

    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert!((optim.status().get_momentum().unwrap() - 0.5).abs() <= f32::EPSILON);
    assert!(optim.status().get_dampening() == None);
    assert!(optim.status().get_nesterov());
}

#[test]
#[should_panic]
fn creation_invalid_nesterov() {
    let _ = StochasticGD::new(1e-2, L2::new(1e-2), None, None, true);
}

#[test]
#[should_panic]
fn creation_invalid_dampening() {
    let _ = StochasticGD::new(1e-2, L2::new(1e-2), 0.5, 3.0, false);
}

#[test]
fn set_lr() {
    let optim = StochasticGD::new(1e-2, L2::new(1e-2), None, None, false);
    optim.set_lr(1e-3);

    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);
}

#[test]
fn set_dampening() {
    let optim = StochasticGD::new(1e-2, L2::new(1e-2), None, None, false);
    optim.status().set_dampening(1.0);

    assert!((optim.status().get_dampening().unwrap() - 1.0).abs() <= f32::EPSILON);
}

#[test]
fn set_momentum() {
    let optim = StochasticGD::new(1e-2, L2::new(1e-2), None, None, false);
    optim.status().set_momentum(0.3);

    assert!((optim.status().get_momentum().unwrap() - 0.3).abs() <= f32::EPSILON);
}

#[test]
fn set_nesterov() {
    let optim = StochasticGD::new(1e-2, L2::new(1e-2), None, None, false);
    optim.status().set_nesterov(true);

    assert!(optim.status().get_nesterov());
}

const EPOCHS: usize = 10;

#[test]
fn step() {
    let x = neuronika_variable::rand((3, 3)).requires_grad();
    let y = neuronika_variable::rand((3, 3));
    let z = neuronika_variable::rand((3, 3));

    let loss = (x.clone().mm(y) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.item();
    let optim = StochasticGD::new(1e-4, L2::new(0.0), None, None, false);
    optim.register(x);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }

    assert!(loss.item() < first_value);
}

#[test]
fn step_with_momentum() {
    let x = neuronika_variable::rand((3, 3)).requires_grad();
    let y = neuronika_variable::rand((3, 3));
    let z = neuronika_variable::rand((3, 3));

    let loss = (x.clone().mm(y) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.item();
    let optim = StochasticGD::new(1e-4, L2::new(0.0), 0.7, 0.3, false);
    optim.register(x);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }

    assert!(loss.item() < first_value);
}

#[test]
fn step_with_nesterov_momentum() {
    let x = neuronika_variable::rand((3, 3)).requires_grad();
    let y = neuronika_variable::rand((3, 3));
    let z = neuronika_variable::rand((3, 3));

    let loss = (x.clone().mm(y) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.item();
    let optim = StochasticGD::new(1e-4, L2::new(0.), 0.7, 0.3, true);
    optim.register(x);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }

    assert!(loss.item() < first_value);
}
