use crate::{StochasticGD, L2};

use super::LambdaLR;

#[test]
fn lambda_lr() {
    const EPOCHS: usize = 5;
    let optim = StochasticGD::new(1.0, L2::new(0.1), None, None, false);
    let scheduler = LambdaLR::new(&optim, |epoch| epoch as f32);

    scheduler.set_current_epoch(5);
    assert_eq!(scheduler.get_current_epoch(), 5);
    scheduler.set_current_epoch(0);
    assert_eq!(scheduler.get_current_epoch(), 0);

    for epoch in 0..EPOCHS {
        if epoch > 0 {
            assert!((scheduler.get_current_lr() - epoch as f32).abs() <= f32::EPSILON);
        }
        optim.zero_grad();
        optim.step();
        assert_eq!(scheduler.get_current_epoch(), epoch);
        scheduler.step();
        scheduler.print_lr();
    }
    assert!((scheduler.get_last_lr() - 4_f32).abs() <= f32::EPSILON);
}
