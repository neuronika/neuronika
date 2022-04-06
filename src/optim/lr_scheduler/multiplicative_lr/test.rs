use crate::optim::{StochasticGD, L2};

use super::MultiplicativeLR;

#[test]
fn multiplicative_lr() {
    const EPOCHS: usize = 5;
    let optim = StochasticGD::new(1.0, L2::new(0.1), None, None, false);
    let scheduler = MultiplicativeLR::new(&optim, |epoch| epoch as f32);

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

    assert!((scheduler.get_last_lr() - 24_f32).abs() <= f32::EPSILON);
    assert!((scheduler.get_current_lr() - 120_f32).abs() <= f32::EPSILON);
    // Should be 5!.
}
