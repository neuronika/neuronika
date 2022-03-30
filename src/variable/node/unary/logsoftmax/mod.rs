use super::{Backward, Forward, OptionalTensor, Tensor};
use ndarray::{Axis, Dimension, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct LogSoftmax<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    axis: usize,
    computed: Cell<bool>,
}

impl<D> LogSoftmax<D>
where
    D: Dimension,
{
    pub fn new(
        operand_data: Rc<RefCell<Tensor<D>>>,
        data: Rc<RefCell<Tensor<D>>>,
        axis: usize,
    ) -> Self {
        Self {
            operand_data,
            data,
            axis,
            computed: Cell::default(),
        }
    }
}

impl<D> Forward for LogSoftmax<D>
where
    D: Dimension,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let axis = self.axis;
        Zip::from(self.data.borrow_mut().lanes_mut(Axis(axis)))
            .and(self.operand_data.borrow().lanes(Axis(axis)))
            .for_each(|lane_v, lane_o| {
                let max = lane_o.fold(f32::MIN, |x, y| x.max(*y));
                let exp = &lane_o.map(|el| (el - max).exp());
                let log_sum_exp = exp.sum().ln();
                Zip::from(lane_v)
                    .and(lane_o)
                    .for_each(|lane_v_el, lane_o_el| *lane_v_el = lane_o_el - log_sum_exp - max);
            });
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct LogSoftmaxBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<OptionalTensor<D>>,
    data: Rc<RefCell<Tensor<D>>>,
    gradient: Rc<OptionalTensor<D>>,
    axis: Axis,
}

impl<D> LogSoftmaxBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<OptionalTensor<D>>,
        data: Rc<RefCell<Tensor<D>>>,
        gradient: Rc<OptionalTensor<D>>,
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
        Zip::from(self.operand_gradient.content_mut().lanes_mut(self.axis))
            .and(self.gradient.content().lanes(self.axis))
            .and(self.data.borrow().lanes(self.axis))
            .for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let gradient_sum = grad_lane.sum();
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, grad_el, data_el| {
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
