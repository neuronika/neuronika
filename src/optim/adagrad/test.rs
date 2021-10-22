use super::{super::L2, Adagrad};

#[test]
fn creation() {
    let optim = Adagrad::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-10);

    assert_eq!(optim.params.borrow().len(), 0);
    assert_eq!(optim.get_lr(), 1e-2);
    assert_eq!(optim.get_lr_decay(), 1e-3);
    assert_eq!(optim.get_eps(), 1e-10);
}

#[test]
fn set_lr() {
    let optim = Adagrad::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-10);

    optim.set_lr(1e-3);
    assert_eq!(optim.get_lr(), 1e-3);
}

#[test]
fn set_lr_decay() {
    let optim = Adagrad::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-10);

    optim.set_lr_decay(1e-4);
    assert_eq!(optim.get_lr_decay(), 1e-4);
}

#[test]
fn set_eps() {
    let optim = Adagrad::new(Vec::new(), 1e-2, 1e-3, L2::new(1e-2), 1e-10);

    optim.set_eps(1e-9);
    assert_eq!(optim.get_eps(), 1e-9);
}

const EPOCHS: usize = 1000;
const TOL: f32 = 1e-3;

#[test]
fn step() {
    let x = crate::rand((3, 3));
    let y = crate::rand((3, 3));
    let z = x.clone().mm(y.clone());

    let w = crate::rand((3, 3)).requires_grad();
    let mut loss = (x.mm(w.clone()) - z).pow(2).sum();

    let optim = Adagrad::new(loss.parameters(), 0.1, 1e-9, L2::new(0.), 1e-10);

    for _ in 0..EPOCHS {
        loss.forward();
        loss.backward(1.0);

        optim.step();
        optim.zero_grad();
    }
    assert!(loss.data()[0] < TOL);
}
