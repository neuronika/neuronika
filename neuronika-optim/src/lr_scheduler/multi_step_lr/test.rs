use crate::{StochasticGD, L2};

use super::MultiStepLR;

#[test]
fn multistep_lr() {
    const EPOCHS: usize = 5;
    let optim = StochasticGD::new(1.0, L2::new(0.1), None, None, false);
    let scheduler = MultiStepLR::new(&optim, vec![1, 2, 3, 4], 2.);

    scheduler.set_current_epoch(5);
    assert_eq!(scheduler.get_current_epoch(), 5);
    scheduler.set_current_epoch(0);
    assert_eq!(scheduler.get_current_epoch(), 0);

    for epoch in 0..EPOCHS {
        optim.zero_grad();
        optim.step();
        assert_eq!(scheduler.get_current_epoch(), epoch);
        scheduler.step();
        scheduler.print_lr();
    }
    assert!((scheduler.get_last_lr() - 16_f32).abs() <= f32::EPSILON);
    assert!((scheduler.get_current_lr() - 16_f32).abs() <= f32::EPSILON); // Should be 2^4.
}
