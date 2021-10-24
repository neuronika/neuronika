use super::{super::L2, Adagrad};

#[test]
fn creation() {
    let optim = Adagrad::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-10);

    assert_eq!(optim.params.borrow().len(), 0);
    assert!((optim.get_lr() - 1e-2).abs() <= f32::EPSILON);
    assert!((optim.get_lr_decay() - 1e-3).abs() <= f32::EPSILON);
    assert!((optim.get_eps() - 1e-10).abs() <= f32::EPSILON);
}

#[test]
fn set_lr() {
    let optim = Adagrad::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-10);

    optim.set_lr(1e-3);
    assert!((optim.get_lr() - 1e-3).abs() <= f32::EPSILON);
}

#[test]
fn set_lr_decay() {
    let optim = Adagrad::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-10);

    optim.set_lr_decay(1e-4);
    assert!((optim.get_lr_decay() - 1e-4).abs() <= f32::EPSILON);
}

#[test]
fn set_eps() {
    let optim = Adagrad::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-10);

    optim.set_eps(1e-9);
    assert!((optim.get_eps() - 1e-9).abs() <= f32::EPSILON);
}

const EPOCHS: usize = 2000;
const TOL: f32 = 1e-3;

#[test]
fn step() {
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y);

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w) - z).pow(2).sum();

    let optim = Adagrad::new(loss.parameters(), 0.075, 1e-9, L2::new(0.0), 1e-10);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < TOL);
}
