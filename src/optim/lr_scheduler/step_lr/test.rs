use crate::optim::{StochasticGD, L2};

use super::StepLR;

#[test]
fn step_lr() {
    const EPOCHS: usize = 5;
    let optim = StochasticGD::new(1.0, L2::new(0.1), None, None, false);
    let scheduler = StepLR::new(&optim, 1, 2.);

    scheduler.set_current_epoch(5);
    assert_eq!(scheduler.get_current_epoch(), 5);
    scheduler.set_current_epoch(0);
    assert_eq!(scheduler.get_current_epoch(), 0);

    for epoch in 0..EPOCHS {
        assert!((scheduler.get_current_lr() - 2_f32.powi(epoch as i32)).abs() <= f32::EPSILON);
        optim.zero_grad();
        optim.step();
        assert_eq!(scheduler.get_current_epoch(), epoch);
        scheduler.step();
        scheduler.print_lr();
    }
    assert!((scheduler.get_last_lr() - 2_f32.powi(4)).abs() <= f32::EPSILON);
}
