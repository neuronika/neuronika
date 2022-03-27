#[cfg(test)]
use super::{assert_almost_equals, new_tensor};
use super::{expect_tensor, expect_tensor_mut, Backward, Forward, Tensor};
use ndarray::{Axis, Dimension, Zip};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

pub struct Softmax<D>
where
    D: Dimension,
{
    operand_data: Rc<RefCell<Tensor<D>>>,
    data: Rc<RefCell<Tensor<D>>>,
    axis: usize,
    computed: Cell<bool>,
}

impl<D> Softmax<D>
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

impl<D> Forward for Softmax<D>
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
                let max = lane_o.fold(std::f32::MIN, |x, y| x.max(*y));
                let num = &lane_o.map(|el| (el - max).exp());
                let den = num.sum();
                Zip::from(lane_v)
                    .and(num)
                    .for_each(|lane_v_el, num_el| *lane_v_el = *num_el / den);
            });
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

pub struct SoftmaxBackward<D>
where
    D: Dimension,
{
    operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
    data: Rc<RefCell<Tensor<D>>>,
    gradient: Rc<RefCell<Option<Tensor<D>>>>,
    shape: D,
    axis: usize,
}

impl<D> SoftmaxBackward<D>
where
    D: Dimension,
{
    pub fn new(
        operand_gradient: Rc<RefCell<Option<Tensor<D>>>>,
        data: Rc<RefCell<Tensor<D>>>,
        gradient: Rc<RefCell<Option<Tensor<D>>>>,
        shape: D,
        axis: usize,
    ) -> Self {
        Self {
            operand_gradient,
            data,
            gradient,
            shape,
            axis,
        }
    }
}

impl<D> Backward for SoftmaxBackward<D>
where
    D: Dimension,
{
    fn backward(&self) {
        let mut operand_gradient = expect_tensor_mut(&self.operand_gradient);
        let gradient = expect_tensor(&self.gradient);
        let data = self.data.borrow();
        let axis = self.axis;
        Zip::from(operand_gradient.lanes_mut(Axis(axis)))
            .and(gradient.lanes(Axis(axis)))
            .and(data.lanes(Axis(axis)))
            .for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let sum = Zip::from(grad_lane)
                    .and(data_lane)
                    .fold(0., |acc, grad_el, data_el| acc + grad_el * data_el);
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, grad_el, data_el| {
                        *op_grad_el += data_el * (grad_el - sum)
                    })
            });
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// #[cfg(test)]
// mod test;
