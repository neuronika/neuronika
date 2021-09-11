use super::super::{Optimizer, L2, SGD};
use super::{ExponentialLR, LRScheduler, LambdaLR, MultiStepLR, MultiplicativeLR, StepLR};

#[test]
fn lambda_lr() {
    const EPOCHS: usize = 5;
    let optim = SGD::new(vec![], 1., L2::new(0.1));
    let scheduler = LambdaLR::new(&optim, |epoch| epoch as f32);

    for epoch in 0..EPOCHS {
        optim.zero_grad();
        optim.step();
        assert_eq!(scheduler.get_current_epoch(), epoch);
        scheduler.step();
        assert_eq!(scheduler.get_current_lr(), epoch as f32);
    }
}

#[test]
fn multiplicative_lr() {
    const EPOCHS: usize = 5;
    let optim = SGD::new(vec![], 1., L2::new(0.1));
    let scheduler = MultiplicativeLR::new(&optim, |epoch| epoch as f32);
    scheduler.set_current_epoch(1);

    for epoch in 1..=EPOCHS {
        optim.zero_grad();
        assert_eq!(scheduler.get_current_epoch(), epoch);
        optim.step();
        scheduler.step();
    }

    assert_eq!(scheduler.get_current_lr(), 120.0); // Should be 5!.
}

#[test]
fn step_lr() {
    const EPOCHS: usize = 4;
    let optim = SGD::new(vec![], 1., L2::new(0.1));
    let scheduler = StepLR::new(&optim, 1, 2.);
    scheduler.set_current_epoch(1);

    for epoch in 1..=EPOCHS {
        optim.zero_grad();
        optim.step();
        assert_eq!(scheduler.get_current_epoch(), epoch);
        scheduler.step();
        assert_eq!(scheduler.get_current_lr(), 2.0_f32.powi(epoch as i32));
    }
}

#[test]
fn multistep_lr() {
    const EPOCHS: usize = 5;
    let optim = SGD::new(vec![], 1., L2::new(0.1));
    let scheduler = MultiStepLR::new(&optim, [0, 1, 2, 3, 4], 2.);

    for epoch in 0..EPOCHS {
        optim.zero_grad();
        optim.step();
        assert_eq!(scheduler.get_current_epoch(), epoch);
        scheduler.step();
    }
    assert_eq!(scheduler.get_current_lr(), 32.); // Should be 2^5.
}

#[test]
fn exponential_lr() {
    const EPOCHS: usize = 5;
    let optim = SGD::new(vec![], 1., L2::new(0.1));
    let scheduler = ExponentialLR::new(&optim, 5.);

    for epoch in 0..EPOCHS {
        optim.zero_grad();
        assert_eq!(scheduler.get_current_epoch(), epoch);
        optim.step();
        scheduler.step();
    }

    assert_eq!(scheduler.get_current_lr(), 5.0_f32.powi(5)); // Should be 5^5.
}
