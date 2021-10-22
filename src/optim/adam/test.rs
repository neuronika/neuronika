use super::{super::L2, Adam};

#[test]
fn creation() {
    let optim = Adam::new(Vec::new(), 1e-2, (0.9, 0.999), L2::new(1e-2), 1e-8);

    assert_eq!(optim.params.borrow().len(), 0);
    assert_eq!(optim.get_lr(), 1e-2);
    assert_eq!(optim.get_betas(), (0.9, 0.999));
    assert_eq!(optim.get_eps(), 1e-8);
}

#[test]
fn set_lr() {
    let optim = Adam::new(Vec::new(), 1e-2, (0.9, 0.999), L2::new(1e-2), 1e-8);

    optim.set_lr(1e-3);
    assert_eq!(optim.get_lr(), 1e-3);
}

#[test]
fn set_betas() {
    let optim = Adam::new(Vec::new(), 1e-2, (0.9, 0.999), L2::new(1e-2), 1e-8);

    optim.set_betas((0.91, 0.9991));
    assert_eq!(optim.get_betas(), (0.91, 0.9991));
}

#[test]
fn set_eps() {
    let optim = Adam::new(Vec::new(), 1e-2, (0.9, 0.999), L2::new(1e-2), 1e-8);

    optim.set_eps(1e-9);
    assert_eq!(optim.get_eps(), 1e-9);
}

const EPOCHS: usize = 2000;
const TOL: f32 = 1e-3;

#[test]
fn step() {
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y.clone());

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w.clone()) - z).pow(2).sum();

    let optim = Adam::new(loss.parameters(), 0.01, (0.9, 0.999), L2::new(0.0), 1e-8);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < TOL);
}
