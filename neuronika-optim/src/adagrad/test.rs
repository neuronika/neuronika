use neuronika_variable;

use super::{super::L2, Adagrad};

#[test]
fn creation() {
    let optim = Adagrad::new(1e-2, 1e-3, L2::new(1e-2), 1e-10);

    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert!((optim.status().get_lr_decay() - 1e-3).abs() <= f32::EPSILON);
    assert!((optim.status().get_eps() - 1e-10).abs() <= f32::EPSILON);
}

#[test]
fn set_lr() {
    let optim = Adagrad::new(1e-2, 1e-3, L2::new(1e-2), 1e-10);

    optim.set_lr(1e-3);
    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);
}

#[test]
fn set_lr_decay() {
    let optim = Adagrad::new(1e-2, 1e-3, L2::new(1e-2), 1e-10);

    optim.status().set_lr_decay(1e-4);
    assert!((optim.status().get_lr_decay() - 1e-4).abs() <= f32::EPSILON);
}

#[test]
fn set_eps() {
    let optim = Adagrad::new(1e-2, 1e-3, L2::new(1e-2), 1e-10);

    optim.status().set_eps(1e-9);
    assert!((optim.status().get_eps() - 1e-9).abs() <= f32::EPSILON);
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

    let optim = Adagrad::new(1e-4, 1e-9, L2::new(0.0), 1e-10);
    optim.register(x);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }

    assert!(loss.item() < first_value);
}
