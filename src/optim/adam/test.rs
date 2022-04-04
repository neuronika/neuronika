use super::{super::L2, Adam};

#[test]
fn creation() {
    let optim = Adam::new(1e-2, 0.9, 0.999, L2::new(1e-2), 1e-8);

    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert!((optim.status().get_beta1() - 0.9).abs() <= f32::EPSILON);
    assert!((optim.status().get_beta2() - 0.999).abs() <= f32::EPSILON);
    assert!((optim.status().get_eps() - 1e-8).abs() <= f32::EPSILON);
}

#[test]
fn set_lr() {
    let optim = Adam::new(1e-2, 0.9, 0.999, L2::new(1e-2), 1e-8);

    optim.set_lr(1e-3);
    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);
}

#[test]
fn set_betas() {
    let optim = Adam::new(1e-2, 0.9, 0.999, L2::new(1e-2), 1e-8);

    optim.status().set_beta1(0.91);
    optim.status().set_beta2(0.9991);
    assert!((optim.status().get_beta1() - 0.91).abs() <= f32::EPSILON);
    assert!((optim.status().get_beta2() - 0.9991).abs() <= f32::EPSILON);
}

#[test]
fn set_eps() {
    let optim = Adam::new(1e-2, 0.9, 0.999, L2::new(1e-2), 1e-8);

    optim.status().set_eps(1e-9);
    assert!((optim.status().get_eps() - 1e-9).abs() <= f32::EPSILON);
}

const EPOCHS: usize = 200;

#[test]
fn step() {
    let x = crate::rand((3, 3)).requires_grad();
    let y = crate::rand((3, 3));
    let z = crate::rand((3, 3));

    let loss = (x.clone().mm(y) - z).pow(2).sum();
    loss.forward();

    let first_value = loss.item();

    let optim = Adam::new(0.001, 0.9, 0.999, L2::new(0.0), 1e-8);
    optim.register(x);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }

    assert!(loss.item() < first_value.clone());
}
