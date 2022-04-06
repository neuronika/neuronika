use crate::optim::{StochasticGD, L2};

use super::ExponentialLR;

#[test]
fn exponential_lr() {
    const EPOCHS: usize = 5;
    let optim = StochasticGD::new(1.0, L2::new(0.1), None, None, false);
    let scheduler = ExponentialLR::new(&optim, 5.);

    scheduler.set_current_epoch(5);
    assert_eq!(scheduler.get_current_epoch(), 5);
    scheduler.set_current_epoch(0);
    assert_eq!(scheduler.get_current_epoch(), 0);

    for epoch in 0..EPOCHS {
        optim.zero_grad();
        assert_eq!(scheduler.get_current_epoch(), epoch);
        optim.step();
        scheduler.step();
        scheduler.print_lr();
    }
    assert!((scheduler.get_last_lr() - 5_f32.powi(4)).abs() <= f32::EPSILON);
    assert!((scheduler.get_current_lr() - 5_f32.powi(5)).abs() <= f32::EPSILON);
    // Should be 5^5.
}
