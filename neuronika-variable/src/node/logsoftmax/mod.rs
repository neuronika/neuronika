use std::rc::Rc;

use ndarray::{Array, Axis, Dimension, Zip};

use crate::{
    autograd::{Backward, Forward},
    gradient::Gradient,
    utils::Shared,
};

pub(crate) struct LogSoftmax<D>
where
    D: Dimension,
{
    operand_data: Shared<Array<f32, D>>,
    data: Shared<Array<f32, D>>,
    axis: Axis,
}

impl<D> LogSoftmax<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_data: Shared<Array<f32, D>>,
        data: Shared<Array<f32, D>>,
        axis: usize,
    ) -> Self {
        Self {
            operand_data,
            data,
            axis: Axis(axis),
        }
    }
}

impl<D> Forward for LogSoftmax<D>
where
    D: Dimension,
{
    fn forward(&self) {
        Zip::from(self.data.borrow_mut().lanes_mut(self.axis))
            .and(self.operand_data.borrow().lanes(self.axis))
            .for_each(|lane_v, lane_o| {
                let max = lane_o.fold(f32::MIN, |x, &y| x.max(y));
                let exp = &lane_o.map(|&el| (el - max).exp());
                let log_sum_exp = exp.sum().ln();
                Zip::from(lane_v)
                    .and(lane_o)
                    .for_each(|lane_v_el, &lane_o_el| *lane_v_el = lane_o_el - log_sum_exp - max);
            });
    }
}

pub(crate) struct LogSoftmaxBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
    data: Shared<Array<f32, D>>,
    gradient: Rc<Gradient<Array<f32, D>, D>>,
    axis: Axis,
}

impl<D> LogSoftmaxBackward<D>
where
    D: Dimension,
{
    pub(crate) fn new(
        operand_gradient: Rc<Gradient<Array<f32, D>, D>>,
        data: Shared<Array<f32, D>>,
        gradient: Rc<Gradient<Array<f32, D>, D>>,
        axis: usize,
    ) -> Self {
        Self {
            operand_gradient,
            data,
            gradient,
            axis: Axis(axis),
        }
    }
}

impl<D> Backward for LogSoftmaxBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        Zip::from(self.operand_gradient.borrow_mut().lanes_mut(self.axis))
            .and(self.gradient.borrow().lanes(self.axis))
            .and(self.data.borrow().lanes(self.axis))
            .for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let gradient_sum = grad_lane.sum();
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, &grad_el, &data_el| {
                        *op_grad_el += grad_el - data_el.exp() * gradient_sum
                    })
            });
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
