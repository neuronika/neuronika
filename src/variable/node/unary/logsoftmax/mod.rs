#[cfg(test)]
use super::{assert_almost_equals, new_backward_input, new_input, new_tensor};
use super::{
    expect_tensor, expect_tensor_mut, Backward, Data, Forward, Gradient, Overwrite, Tensor,
};
use ndarray::{Axis, Zip};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSoftmax ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct LogSoftmax<T: Data> {
    operand: Rc<T>,
    data: RefCell<Tensor<T::Dim>>,
    axis: usize,
    computed: Cell<bool>,
}

impl<T: Data> LogSoftmax<T> {
    pub fn new(operand: Rc<T>, axis: usize) -> Self {
        let data = RefCell::new(Tensor::zeros(operand.data().raw_dim()));

        Self {
            operand,
            data,
            axis,
            computed: Cell::new(false),
        }
    }
}

impl<T: Data> Forward for LogSoftmax<T> {
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let axis = self.axis;
        Zip::from(self.data.borrow_mut().lanes_mut(Axis(axis)))
            .and(self.operand.data().lanes(Axis(axis)))
            .for_each(|lane_v, lane_o| {
                let max = lane_o.fold(std::f32::MIN, |x, y| x.max(*y));
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

impl<T: Data> Data for LogSoftmax<T> {
    type Dim = T::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
    }
}

impl<T: Data> Debug for LogSoftmax<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogSoftmax")
            .field("data", &self.data.borrow())
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<T: Data> Display for LogSoftmax<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LogSoftmaxBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    gradient: RefCell<Option<Tensor<T::Dim>>>,
    shape: T::Dim,
    overwrite: Cell<bool>,
    diff_operand: Rc<T>,
    no_diff_operand: Rc<U>,
    axis: usize,
}

impl<T, U> LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    pub fn new(diff_operand: Rc<T>, no_diff_operand: Rc<U>, axis: usize) -> Self {
        let shape = diff_operand.gradient().raw_dim();

        Self {
            gradient: RefCell::new(Some(Tensor::zeros(shape.clone()))),
            shape,
            overwrite: Cell::new(true),
            diff_operand,
            no_diff_operand,
            axis,
        }
    }
}

impl<T, U> Gradient for LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    type Dim = T::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<T, U> Overwrite for LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<T, U> Backward for LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn backward(&self) {
        let mut op_grad = self.diff_operand.gradient_mut();
        let data = self.no_diff_operand.data();
        let grad = self.gradient();
        let axis = self.axis;

        let zip = Zip::from(op_grad.lanes_mut(Axis(axis)))
            .and(grad.lanes(Axis(axis)))
            .and(data.lanes(Axis(axis)));
        if self.diff_operand.can_overwrite() {
            zip.for_each(|mut op_grad_lane, grad_lane, data_lane| {
                let gradient_sum = grad_lane.sum();
                Zip::from(&mut op_grad_lane)
                    .and(&grad_lane)
                    .and(&data_lane)
                    .for_each(|op_grad_el, grad_el, data_el| {
                        *op_grad_el = grad_el - data_el.exp() * gradient_sum
                    })
            });
            self.diff_operand.set_overwrite(false);
        } else {
            zip.for_each(|mut op_grad_lane, grad_lane, data_lane| {
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

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

impl<T, U> Debug for LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogSoftmaxBackward")
            .field("gradient", &self.gradient.borrow())
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<T, U> Display for LogSoftmaxBackward<T, U>
where
    T: Gradient + Overwrite,
    U: Data<Dim = T::Dim>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
