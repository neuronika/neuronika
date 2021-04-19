use super::{BroadTensor, Broadcasted, GraphBuilder, ParamDim, Parameters, Tensor};
use ndarray::{
    concatenate, linalg::general_mat_mul, linalg::general_mat_vec_mul, stack, Array2, ArrayView1,
    Axis, DimMax, Dimension, Ix1, Ix2, RemoveAxis, Zip,
};
use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

// ==================================================== Utils ====================================================

fn broadcasted_zeros<Lhs, Rhs>(left: &Tensor<Lhs>, right: &Tensor<Rhs>) -> BroadTensor<Lhs, Rhs>
where
    Lhs: Dimension + DimMax<Rhs>,
    Rhs: Dimension,
{
    let (bigger, smaller) = if left.ndim() >= right.ndim() {
        (left.shape(), right.shape())
    } else {
        (right.shape(), left.shape())
    };
    let b_dim = {
        let mut empty_d = <Lhs as DimMax<Rhs>>::Output::zeros(bigger.len());
        let empty_d_slice = empty_d.slice_mut();
        empty_d_slice
            .iter_mut()
            .zip(bigger.iter())
            .for_each(|(e_el, b_el)| *e_el = *b_el);
        empty_d_slice
            .iter_mut()
            .rev()
            .zip(smaller.iter().rev())
            .for_each(|(l, r)| *l = std::cmp::max(*l, *r));
        empty_d
    };
    Tensor::zeros(b_dim)
}

trait DotDim<Rhs>
where
    Self: Dimension,
    Rhs: Dimension,
{
    type Output: Dimension;
    fn shape(lhs: Self, rhs: Rhs) -> <Self as DotDim<Rhs>>::Output;
}

impl DotDim<Ix1> for Ix1 {
    type Output = Ix1;
    fn shape(_: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        let mut res_shape = Ix1::zeros(1);
        res_shape[0] = 1;
        res_shape
    }
}

impl DotDim<Ix2> for Ix1 {
    type Output = Ix1;
    fn shape(_: Self, rhs: Ix2) -> <Self as DotDim<Ix1>>::Output {
        let mut res_shape = Ix1::zeros(1);
        res_shape[0] = rhs.last_elem();
        res_shape
    }
}

impl DotDim<Ix1> for Ix2 {
    type Output = Ix1;
    fn shape(lhs: Self, _: Ix1) -> <Self as DotDim<Ix1>>::Output {
        let mut res_shape = Ix1::zeros(1);
        res_shape[0] = lhs[0];
        res_shape
    }
}

impl DotDim<Ix2> for Ix2 {
    type Output = Ix2;
    fn shape(lhs: Self, rhs: Ix2) -> <Self as DotDim<Ix2>>::Output {
        let mut res_shape = Ix2::zeros(2);
        res_shape[0] = lhs[0];
        res_shape[1] = rhs[1];
        res_shape
    }
}

fn sum_axis_inplace(arr: &mut ndarray::ArrayD<f32>, axis: ndarray::Axis) {
    let (first, rest) = arr.view_mut().split_at(axis, 1);
    ndarray::Zip::from(first.remove_axis(axis))
        .and(rest.lanes(axis))
        .for_each(|dst, src| *dst += src.sum());
    arr.index_axis_inplace(axis, 0);
}

pub fn reduce<D: ndarray::Dimension, E: ndarray::Dimension>(
    dest: &mut ndarray::Array<f32, D>,
    src: &ndarray::Array<f32, E>,
) {
    let mut dyn_rhs = src.clone().into_dyn();
    let static_rhs = unsafe {
        while (*(&dyn_rhs as *const ndarray::ArrayD<f32>)).ndim() > dest.ndim() {
            sum_axis_inplace(&mut dyn_rhs, ndarray::Axis(0));
        }
        for (axis, size) in dest.shape().iter().enumerate() {
            if *size == 1 {
                sum_axis_inplace(&mut dyn_rhs, ndarray::Axis(axis));
                dyn_rhs.insert_axis_inplace(ndarray::Axis(axis));
            }
        }
        dyn_rhs.as_standard_layout()
    };
    ndarray::Zip::from(dest)
        .and_broadcast(&static_rhs)
        .for_each(|dest_el, src_el| *dest_el = *src_el);
}

// ===================================== Computational Graph Components Trait =====================================

/// Node of a computational graph.
pub trait Node: Debug + 'static {
    type Dim: Dimension;

    /// Returns the data of the node.
    fn data(&self) -> Ref<Tensor<Self::Dim>>;

    /// Checks wether the node requires the computation of the gradient.
    fn requires_grad(&self) -> bool;
}

// ====================================== Computational Graph Component: Learnable Parameter ======================================

#[derive(Debug)]
pub struct Parameter<D>
where
    D: Dimension,
{
    pub(crate) data: RefCell<Tensor<D>>,
    pub(crate) grad: RefCell<Tensor<D>>,
}

impl<D> Parameter<D>
where
    D: ParamDim,
{
    pub fn new<'a>(data: Tensor<D>) -> GraphBuilder<Self> {
        let grad = Tensor::zeros(data.raw_dim());
        let node = Rc::new(Parameter {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
        });
        let mut upstream = Parameters::new();
        D::insert(
            GraphBuilder::new(Rc::clone(&node), Parameters::new()),
            &mut upstream,
        );
        GraphBuilder::new(node, upstream)
    }

    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.grad.borrow()
    }

    pub fn zero_grad(&self) {
        self.grad.borrow_mut().map_inplace(|el| *el = 0.0);
    }
}

impl<D> Node for Parameter<D>
where
    D: Dimension + 'static,
{
    type Dim = D;

    // fn forward(&self) {
    //     // Nothing
    // }

    // fn backward(&self, gradient: &Ref<Tensor<Self::Dim>>) {
    //     accumulate(
    //         &mut self.grad.borrow_mut(),
    //         gradient,
    //         1.0,
    //         &BackwardAction::Increment,
    //     );
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        true
    }
}

// ============================================= Computational Graph Component: Input =============================================

#[derive(Debug)]
pub struct Input<D>
where
    D: Dimension,
{
    pub(crate) data: RefCell<Tensor<D>>,
}

impl<'a, D: Dimension + 'static> Input<D> {
    pub fn new(data: Tensor<D>) -> GraphBuilder<Self> {
        GraphBuilder::new(
            Rc::new(Input {
                data: RefCell::new(data),
            }),
            Parameters::new(),
        )
    }
}

impl<D: Dimension + 'static> Node for Input<D> {
    type Dim = D;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        false
    }
}

// ============================================ Computational Graph Internal Component: Negation  ============================================

#[derive(Debug)]
pub struct Negation<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Negation<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Negation<T> {
    type Dim = T::Dim;

    // fn forward(&self) {
    //     if self.counter.forward_action() == ForwardAction::Cached {
    //         return;
    //     }

    //     self.operand.forward();

    //     let mut self_data = self.data.borrow_mut();
    //     let operand_data = self.operand.data();

    //     Zip::from(self_data.deref_mut())
    //         .and(operand_data.deref())
    //         .par_for_each(|self_data_el, operand_data_el| *self_data_el = -operand_data_el);
    // }

    // fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //     accumulate(
    //         &mut self.grad.borrow_mut(),
    //         grad,
    //         -1.0,
    //         &self.counter.backward_action(),
    //     );

    //     if self.counter.recurse_backward() {
    //         self.operand.backward(&self.grad.borrow());
    //     }
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================================== Computational Graph Internal Component: Addition ==============================================

#[derive(Debug)]
pub struct Addition<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    lhs_grad: RefCell<Tensor<Lhs::Dim>>,
    rhs_grad: RefCell<Tensor<Rhs::Dim>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    requires_grad: bool,
}

impl<Lhs, Rhs> Addition<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = RefCell::new(broadcasted_zeros(lhs.data().deref(), rhs.data().deref()));
        let lhs_grad = RefCell::new(Tensor::zeros(lhs.data().raw_dim()));
        let rhs_grad = RefCell::new(Tensor::zeros(rhs.data().raw_dim()));

        Self {
            data,
            lhs_grad,
            rhs_grad,
            lhs,
            rhs,
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for Addition<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    // fn forward(&self) {
    //     if self.counter.forward_action() == ForwardAction::Cached {
    //         return;
    //     }

    //     self.lhs.forward();
    //     self.rhs.forward();

    //     Zip::from(self.data.borrow_mut().deref_mut())
    //         .and_broadcast(self.lhs.data().deref())
    //         .and_broadcast(self.rhs.data().deref())
    //         .par_for_each(|self_data_el, lhs_data_el, rhs_data_el| {
    //             *self_data_el = *lhs_data_el + *rhs_data_el
    //         });
    // }

    // fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //     let action = self.counter.backward_action();

    //     accumulate(&mut self.lhs_grad.borrow_mut(), grad.deref(), 1.0, &action);
    //     accumulate(&mut self.rhs_grad.borrow_mut(), grad.deref(), 1.0, &action);

    //     if self.counter.recurse_backward() {
    //         self.lhs.backward(&self.lhs_grad.borrow());
    //         self.rhs.backward(&self.rhs_grad.borrow());
    //     }
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ========================================= Computational Graph Internal Component: Subtraction  =========================================

#[derive(Debug)]
pub struct Subtraction<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    lhs_grad: RefCell<Tensor<Lhs::Dim>>,
    rhs_grad: RefCell<Tensor<Rhs::Dim>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    requires_grad: bool,
}

impl<Lhs, Rhs> Subtraction<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = RefCell::new(broadcasted_zeros(lhs.data().deref(), rhs.data().deref()));
        let lhs_grad = RefCell::new(Tensor::zeros(lhs.data().raw_dim()));
        let rhs_grad = RefCell::new(Tensor::zeros(rhs.data().raw_dim()));

        Self {
            data,
            lhs_grad,
            rhs_grad,
            lhs,
            rhs,
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for Subtraction<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    // fn forward(&self) {
    //     if self.counter.forward_action() == ForwardAction::Cached {
    //         return;
    //     }

    //     self.lhs.forward();
    //     self.rhs.forward();

    //     let mut self_data = self.data.borrow_mut();
    //     let lhs_data = self.lhs.data();
    //     let rhs_data = self.rhs.data();

    //     Zip::from(self_data.deref_mut())
    //         .and_broadcast(lhs_data.deref())
    //         .and_broadcast(rhs_data.deref())
    //         .par_for_each(|self_data_el, lhs_data_el, rhs_data_el| {
    //             *self_data_el = *lhs_data_el - *rhs_data_el
    //         });
    // }

    // fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //     let action = self.counter.backward_action();

    //     accumulate(&mut self.lhs_grad.borrow_mut(), grad.deref(), 1.0, &action);
    //     accumulate(&mut self.rhs_grad.borrow_mut(), grad.deref(), -1.0, &action);

    //     if self.counter.recurse_backward() {
    //         self.lhs.backward(&self.lhs_grad.borrow());
    //         self.rhs.backward(&self.rhs_grad.borrow());
    //     }
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ========================================= Computational Graph Internal Component: Multiplication  =========================================

#[derive(Debug)]
pub struct Multiplication<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    lhs_grad: RefCell<Tensor<Lhs::Dim>>,
    rhs_grad: RefCell<Tensor<Rhs::Dim>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    requires_grad: bool,
}

impl<Lhs, Rhs> Multiplication<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = RefCell::new(broadcasted_zeros(lhs.data().deref(), rhs.data().deref()));
        let lhs_grad = RefCell::new(Tensor::zeros(lhs.data().raw_dim()));
        let rhs_grad = RefCell::new(Tensor::zeros(rhs.data().raw_dim()));

        Self {
            data,
            lhs_grad,
            rhs_grad,
            lhs,
            rhs,
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for Multiplication<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    // fn forward(&self) {
    //     if self.counter.forward_action() == ForwardAction::Cached {
    //         return;
    //     }

    //     self.lhs.forward();
    //     self.rhs.forward();

    //     let mut self_data = self.data.borrow_mut();
    //     let lhs_data = self.lhs.data();
    //     let rhs_data = self.rhs.data();

    //     Zip::from(self_data.deref_mut())
    //         .and_broadcast(lhs_data.deref())
    //         .and_broadcast(rhs_data.deref())
    //         .par_for_each(|self_data_el, lhs_data_el, rhs_data_el| {
    //             *self_data_el = *lhs_data_el * *rhs_data_el
    //         });
    // }

    // fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //     let action = self.counter.backward_action();
    //     let rhs_data = self.rhs.data();
    //     let lhs_data = self.lhs.data();
    //     let down_grad = grad.deref();
    //     let mut tmp = Tensor::zeros(down_grad.raw_dim());

    //     Zip::from(&mut tmp)
    //         .and(down_grad)
    //         .and_broadcast(rhs_data.deref())
    //         .par_for_each(|res, grad, rhs| *res = *grad * *rhs);
    //     accumulate(&mut self.lhs_grad.borrow_mut(), &tmp, 1.0, &action);

    //     Zip::from(&mut tmp)
    //         .and(down_grad)
    //         .and_broadcast(lhs_data.deref())
    //         .par_for_each(|res, grad, lhs| *res = *grad * *lhs);
    //     accumulate(&mut self.rhs_grad.borrow_mut(), &tmp, 1.0, &action);

    //     if self.counter.recurse_backward() {
    //         self.lhs.backward(&self.lhs_grad.borrow());
    //         self.rhs.backward(&self.rhs_grad.borrow());
    //     }
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // =========================================== Computational Graph Internal Component: Division  ===========================================

#[derive(Debug)]
pub struct Division<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    data: RefCell<BroadTensor<Lhs::Dim, Rhs::Dim>>,
    lhs_grad: RefCell<Tensor<Lhs::Dim>>,
    rhs_grad: RefCell<Tensor<Rhs::Dim>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    requires_grad: bool,
}

impl<Lhs, Rhs> Division<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let data = RefCell::new(broadcasted_zeros(lhs.data().deref(), rhs.data().deref()));
        let lhs_grad = RefCell::new(Tensor::zeros(lhs.data().raw_dim()));
        let rhs_grad = RefCell::new(Tensor::zeros(rhs.data().raw_dim()));

        Self {
            data,
            lhs_grad,
            rhs_grad,
            lhs,
            rhs,
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for Division<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node,
    Lhs::Dim: Dimension + DimMax<Rhs::Dim>,
{
    type Dim = Broadcasted<Lhs::Dim, Rhs::Dim>;

    // fn forward(&self) {
    //     if self.counter.forward_action() == ForwardAction::Cached {
    //         return;
    //     }

    //     self.lhs.forward();
    //     self.rhs.forward();

    //     let mut self_data = self.data.borrow_mut();
    //     let lhs_data = self.lhs.data();
    //     let rhs_data = self.rhs.data();

    //     Zip::from(self_data.deref_mut())
    //         .and_broadcast(lhs_data.deref())
    //         .and_broadcast(rhs_data.deref())
    //         .par_for_each(|self_data_el, lhs_data_el, rhs_data_el| {
    //             *self_data_el = *lhs_data_el / *rhs_data_el
    //         });
    // }

    // fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //     let action = self.counter.backward_action();
    //     let rhs_data = self.rhs.data();
    //     let lhs_data = self.lhs.data();
    //     let down_grad = grad.deref();
    //     let mut tmp = Tensor::zeros(down_grad.raw_dim());

    //     Zip::from(&mut tmp)
    //         .and(down_grad)
    //         .and_broadcast(rhs_data.deref())
    //         .par_for_each(|res, grad, rhs| *res = *grad / *rhs);
    //     accumulate(&mut self.lhs_grad.borrow_mut(), &tmp, 1.0, &action);

    //     Zip::from(&mut tmp)
    //         .and(down_grad)
    //         .and_broadcast(lhs_data.deref())
    //         .and_broadcast(rhs_data.deref())
    //         .par_for_each(|res, grad, lhs, rhs| *res = *grad * *lhs / rhs.powi(2));
    //     accumulate(&mut self.rhs_grad.borrow_mut(), &tmp, -1.0, &action);

    //     if self.counter.recurse_backward() {
    //         self.lhs.backward(&self.lhs_grad.borrow());
    //         self.rhs.backward(&self.rhs_grad.borrow());
    //     }
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Matrix Mult.  ============================

#[derive(Debug)]
pub struct MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix2>,
    Rhs: Node<Dim = Ix2>,
{
    data: RefCell<Tensor<Ix2>>,
    grad: RefCell<Tensor<Ix2>>,
    lhs_grad: RefCell<Tensor<Ix2>>,
    rhs_grad: RefCell<Tensor<Ix2>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    requires_grad: bool,
}

impl<Lhs, Rhs> MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix2>,
    Rhs: Node<Dim = Ix2>,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(lhs.data().raw_dim(), rhs.data().raw_dim());
        let data = RefCell::new(Tensor::zeros((shape[0], shape[1])));
        let grad = RefCell::new(Tensor::zeros((shape[0], shape[1])));
        let lhs_grad = RefCell::new(Tensor::zeros(lhs.data().raw_dim()));
        let rhs_grad = RefCell::new(Tensor::zeros(rhs.data().raw_dim()));
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        Self {
            data,
            grad,
            lhs_grad,
            rhs_grad,
            lhs,
            rhs,
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for MatrixMatrixMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix2>,
    Rhs: Node<Dim = Ix2>,
{
    type Dim = Ix2;

    // fn forward(&self) {
    //     if self.counter.forward_action() == ForwardAction::Cached {
    //         return;
    //     }

    //     self.lhs.forward();
    //     self.rhs.forward();

    //     let lhs_data = self.lhs.data();
    //     let rhs_data = self.rhs.data();
    //     let mut res_data = self.data.borrow_mut();

    //     general_mat_mul(
    //         1.0,
    //         lhs_data.deref(),
    //         rhs_data.deref(),
    //         0.0,
    //         res_data.deref_mut(),
    //     );
    // }

    // fn backward(&self, input_grad: &Ref<Tensor<Self::Dim>>) {
    //     let action = self.counter.backward_action();
    //     {
    //         let mut self_grad = self.grad.borrow_mut();
    //         let down_grad = input_grad.deref();

    //         accumulate(self_grad.deref_mut(), down_grad, 1.0, &action);
    //     }

    //     if self.counter.recurse_backward() {
    //         {
    //             let lhs_data = self.lhs.data();
    //             let mut lhs_grad = self.lhs_grad.borrow_mut();
    //             let rhs_data = self.rhs.data();
    //             let mut rhs_grad = self.rhs_grad.borrow_mut();

    //             let grad = self.grad.borrow();

    //             general_mat_mul(
    //                 1.0,
    //                 grad.deref(),
    //                 &rhs_data.deref().t(),
    //                 0.0,
    //                 lhs_grad.deref_mut(),
    //             );
    //             general_mat_mul(
    //                 1.0,
    //                 &lhs_data.deref().t(),
    //                 grad.deref(),
    //                 0.0,
    //                 rhs_grad.deref_mut(),
    //             );
    //         }
    //         self.lhs.backward(&self.lhs_grad.borrow());
    //         self.rhs.backward(&self.rhs_grad.borrow());
    //     }
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Mat. Vec. Prod.  ============================

#[derive(Debug)]
pub struct MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix2>,
    Rhs: Node<Dim = Ix1>,
{
    data: RefCell<Tensor<Ix1>>,
    grad: RefCell<Tensor<Ix1>>,
    lhs_grad: RefCell<Tensor<Ix2>>,
    rhs_grad: RefCell<Tensor<Ix1>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    requires_grad: bool,
}

impl<Lhs, Rhs> MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix2>,
    Rhs: Node<Dim = Ix1>,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(lhs.data().raw_dim(), rhs.data().raw_dim());
        let data = RefCell::new(Tensor::zeros(shape[0]));
        let grad = RefCell::new(Tensor::zeros(shape[0]));
        let lhs_grad = RefCell::new(Tensor::zeros(lhs.data().raw_dim()));
        let rhs_grad = RefCell::new(Tensor::zeros(rhs.data().raw_dim()));
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        Self {
            data,
            grad,
            lhs_grad,
            rhs_grad,
            lhs,
            rhs,
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for MatrixVectorMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix2>,
    Rhs: Node<Dim = Ix1>,
{
    type Dim = Ix1;

    // fn forward(&self) {
    //     if self.counter.forward_action() == ForwardAction::Cached {
    //         return;
    //     }

    //     self.lhs.forward();
    //     self.rhs.forward();

    //     let lhs_data = self.lhs.data();
    //     let rhs_data = self.rhs.data();
    //     let mut self_data = self.data.borrow_mut();

    //     general_mat_vec_mul(
    //         1.0,
    //         lhs_data.deref(),
    //         rhs_data.deref(),
    //         0.0,
    //         self_data.deref_mut(),
    //     );
    // }

    // fn backward(&self, input_grad: &Ref<Tensor<Self::Dim>>) {
    //     let action = self.counter.backward_action();

    //     let mut self_grad = self.grad.borrow_mut();
    //     let down_grad = input_grad.deref();

    //     accumulate(self_grad.deref_mut(), down_grad, 1.0, &action);

    //     if self.counter.recurse_backward() {
    //         {
    //             let lhs_data = self.lhs.data();
    //             let mut lhs_grad = self.lhs_grad.borrow_mut();
    //             let rhs_data = self.rhs.data();
    //             let mut rhs_grad = self.rhs_grad.borrow_mut();
    //             let grad = self.grad.borrow();

    //             Zip::from(lhs_grad.rows_mut())
    //                 .and(grad.deref())
    //                 .for_each(|row, grad_el| {
    //                     Zip::from(row)
    //                         .and(rhs_data.deref())
    //                         .for_each(|row_el, rhs_data_el| *row_el = *rhs_data_el * *grad_el);
    //                 });

    //             general_mat_vec_mul(
    //                 1.0,
    //                 &lhs_data.deref().t(),
    //                 grad.deref(),
    //                 0.0,
    //                 rhs_grad.deref_mut(),
    //             );
    //         }

    //         self.lhs.backward(&self.lhs_grad.borrow());
    //         self.rhs.backward(&self.rhs_grad.borrow());
    //     }
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Inner Prod.  ============================

#[derive(Debug)]
pub struct VectorVectorMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix1>,
    Rhs: Node<Dim = Ix1>,
{
    data: RefCell<Tensor<Ix1>>,
    lhs_grad: RefCell<Tensor<Ix1>>,
    rhs_grad: RefCell<Tensor<Ix1>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    requires_grad: bool,
}

impl<Lhs, Rhs> VectorVectorMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix1>,
    Rhs: Node<Dim = Ix1>,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(lhs.data().raw_dim(), rhs.data().raw_dim());
        let data = RefCell::new(Tensor::zeros(shape[0]));
        let lhs_grad = RefCell::new(Tensor::zeros(lhs.data().raw_dim()));
        let rhs_grad = RefCell::new(Tensor::zeros(rhs.data().raw_dim()));
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        Self {
            data,
            lhs_grad,
            rhs_grad,
            lhs,
            rhs,
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for VectorVectorMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix1>,
    Rhs: Node<Dim = Ix1>,
{
    type Dim = Ix1;

    // fn forward(&self) {
    //     if self.counter.forward_action() == ForwardAction::Cached {
    //         return;
    //     }

    //     self.lhs.forward();
    //     self.rhs.forward();

    //     let lhs_data = self.lhs.data();
    //     let rhs_data = self.rhs.data();
    //     let mut self_data = self.data.borrow_mut();

    //     self_data[0] = lhs_data.dot(rhs_data.deref());
    // }

    // fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //     let action = self.counter.backward_action();

    //     {
    //         let lhs_data = self.lhs.data();
    //         let mut lhs_grad = self.lhs_grad.borrow_mut();
    //         let rhs_data = self.rhs.data();
    //         let mut rhs_grad = self.rhs_grad.borrow_mut();
    //         let down_grad = grad.deref();

    //         accumulate(
    //             lhs_grad.deref_mut(),
    //             rhs_data.deref(),
    //             down_grad[0],
    //             &action,
    //         );
    //         accumulate(
    //             rhs_grad.deref_mut(),
    //             lhs_data.deref(),
    //             down_grad[0],
    //             &action,
    //         );
    //     }
    //     if self.counter.recurse_backward() {
    //         self.lhs.backward(&self.lhs_grad.borrow());
    //         self.rhs.backward(&self.rhs_grad.borrow());
    //     }
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Inner Prod.  ============================

#[derive(Debug)]
pub struct VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix1>,
    Rhs: Node<Dim = Ix2>,
{
    data: RefCell<Tensor<Ix1>>,
    lhs_grad: RefCell<Tensor<Ix1>>,
    rhs_grad: RefCell<Tensor<Ix2>>,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    requires_grad: bool,
}

impl<Lhs, Rhs> VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix1>,
    Rhs: Node<Dim = Ix2>,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>) -> Self {
        let shape = DotDim::shape(lhs.data().raw_dim(), rhs.data().raw_dim());
        let data = RefCell::new(Tensor::zeros(shape[0]));
        let lhs_grad = RefCell::new(Tensor::zeros(lhs.data().raw_dim()));
        let rhs_grad = RefCell::new(Tensor::zeros(rhs.data().raw_dim()));
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();

        Self {
            data,
            lhs_grad,
            rhs_grad,
            lhs,
            rhs,
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for VectorMatrixMul<Lhs, Rhs>
where
    Lhs: Node<Dim = Ix1>,
    Rhs: Node<Dim = Ix2>,
{
    type Dim = Ix1;

    // fn forward(&self) {
    //     if self.counter.forward_action() == ForwardAction::Cached {
    //         return;
    //     }

    //     self.lhs.forward();
    //     self.rhs.forward();

    //     let lhs_data = self.lhs.data();
    //     let rhs_data = self.rhs.data();
    //     let mut self_data = self.data.borrow_mut();

    //     *self_data = lhs_data.dot(rhs_data.deref());
    // }

    // fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //     let action = self.counter.backward_action();

    //     {
    //         let lhs_data = self.lhs.data();
    //         let mut lhs_grad = self.lhs_grad.borrow_mut();
    //         let rhs_data = self.rhs.data();
    //         let mut rhs_grad = self.rhs_grad.borrow_mut();
    //         let down_grad = grad.deref();

    //         accumulate(
    //             lhs_grad.deref_mut(),
    //             rhs_data.deref(),
    //             down_grad[0],
    //             &action,
    //         );
    //         accumulate(
    //             rhs_grad.deref_mut(),
    //             lhs_data.deref(),
    //             down_grad[0],
    //             &action,
    //         );
    //     }

    //     if self.counter.recurse_backward() {
    //         self.lhs.backward(&self.lhs_grad.borrow());
    //         self.rhs.backward(&self.rhs_grad.borrow());
    //     }
    // }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Power  ============================

#[derive(Debug)]
pub struct Power<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    exp: i32,
    requires_grad: bool,
}

impl<T: Node> Power<T> {
    pub fn new(operand: Rc<T>, exp: i32) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            exp,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Power<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let (mut self_data, operand_data, exp) =
    //             { (self.data.borrow_mut(), self.operand.data(), self.exp) };

    //         Zip::from(self_data.deref_mut())
    //             .and(operand_data.deref())
    //             .par_for_each(|self_data_el, operand_data_el| {
    //                 *self_data_el = operand_data_el.powi(exp)
    //             });
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let operand_data = self.operand.data();
    //             let down_grad = grad.deref();
    //             let exp = self.exp;

    //             let zip = Zip::from(self_grad.deref_mut())
    //                 .and(down_grad)
    //                 .and(operand_data.deref());

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip.for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el = *down_grad_el * operand_data_el.powi(exp - 1) * exp as f32
    //                     });
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip.for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el += *down_grad_el * operand_data_el.powi(exp - 1) * exp as f32
    //                     });
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Sum Reduction  ============================

#[derive(Debug)]
pub struct Sum<T: Node> {
    data: RefCell<Tensor<Ix1>>,
    grad: RefCell<Tensor<Ix1>>,
    op_grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Sum<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let requires_grad = operand.requires_grad();
        let (data, grad, op_grad) = {
            let operand_data = operand.data();
            (
                Tensor::<Ix1>::from(vec![operand_data.sum()]),
                Tensor::zeros(1),
                Tensor::zeros(operand_data.raw_dim()),
            )
        };

        Self {
            data: RefCell::new(data),
            grad: RefCell::new(grad),
            op_grad: RefCell::new(op_grad),
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Sum<T> {
    type Dim = Ix1;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();

    //         self_data[0] = operand_data.sum();
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         let action = self.counter.backward_action();
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let mut op_grad = self.op_grad.borrow_mut();
    //             let down_grad = grad.deref();

    //             accumulate(self_grad.deref_mut(), down_grad, 1.0, &action);
    //             accumulate(op_grad.deref_mut(), self_grad.deref(), 1.0, &action);
    //         }
    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.op_grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Natural Log.  ============================

#[derive(Debug)]
pub struct Logn<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Logn<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Logn<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();

    //         Zip::from(self_data.deref_mut())
    //             .and(operand_data.deref())
    //             .par_for_each(|self_data_el, operand_data_el| *self_data_el = operand_data_el.ln());
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let operand_data = self.operand.data();
    //             let down_grad = grad.deref();

    //             let zip = Zip::from(self_grad.deref_mut())
    //                 .and(down_grad.deref())
    //                 .and(operand_data.deref());

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el = *down_grad_el / *operand_data_el
    //                     });
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el += *down_grad_el / *operand_data_el
    //                     });
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: ReLU  ============================

#[derive(Debug)]
pub struct Relu<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Relu<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Relu<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();

    //         Zip::from(self_data.deref_mut())
    //             .and(operand_data.deref())
    //             .par_for_each(|self_data_el, operand_data_el| {
    //                 *self_data_el = if *operand_data_el > 0.0 {
    //                     *operand_data_el
    //                 } else {
    //                     0.0
    //                 }
    //             });
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let operand_data = self.operand.data();
    //             let down_grad = grad;

    //             let zip = Zip::from(self_grad.deref_mut())
    //                 .and(down_grad.deref())
    //                 .and(operand_data.deref());

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el = if *operand_data_el > 0.0 {
    //                             *down_grad_el
    //                         } else {
    //                             0.0
    //                         }
    //                     });
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el += if *operand_data_el > 0.0 {
    //                             *down_grad_el
    //                         } else {
    //                             0.0
    //                         }
    //                     });
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: LeakyReLU  ============================

#[derive(Debug)]
pub struct LeakyRelu<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> LeakyRelu<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for LeakyRelu<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();

    //         Zip::from(self_data.deref_mut())
    //             .and(operand_data.deref())
    //             .par_for_each(|self_data_el, operand_data_el| {
    //                 *self_data_el = if *operand_data_el > 0.0 {
    //                     *operand_data_el
    //                 } else {
    //                     0.01 * operand_data_el
    //                 }
    //             });
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let operand_data = self.operand.data();
    //             let down_grad = grad.deref();

    //             let zip = Zip::from(self_grad.deref_mut())
    //                 .and(down_grad)
    //                 .and(operand_data.deref());

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el = if *operand_data_el > 0.0 {
    //                             *down_grad_el
    //                         } else {
    //                             0.01 * down_grad_el
    //                         }
    //                     });
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el += if *operand_data_el > 0.0 {
    //                             *down_grad_el
    //                         } else {
    //                             0.01 * down_grad_el
    //                         }
    //                     });
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Softplus  ============================

#[derive(Debug)]
pub struct Softplus<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Softplus<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Softplus<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();

    //         Zip::from(self_data.deref_mut())
    //             .and(operand_data.deref())
    //             .par_for_each(|self_data_el, operand_data_el| {
    //                 *self_data_el = if *operand_data_el < -15.0 {
    //                     0.0
    //                 } else if *operand_data_el > 15.0 {
    //                     *operand_data_el
    //                 } else {
    //                     (1.0 + operand_data_el.exp()).ln()
    //                 }
    //             });
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let operand_data = self.operand.data();
    //             let down_grad = grad.deref();

    //             let zip = Zip::from(self_grad.deref_mut())
    //                 .and(down_grad)
    //                 .and(operand_data.deref());

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el = if *operand_data_el >= 15.0 {
    //                             *down_grad_el
    //                         } else if *operand_data_el <= -15.0 {
    //                             0.0
    //                         } else {
    //                             down_grad_el / (1.0 + (-*operand_data_el).exp())
    //                         }
    //                     });
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el += if *operand_data_el >= 15.0 {
    //                             *down_grad_el
    //                         } else if *operand_data_el <= -15.0 {
    //                             0.0
    //                         } else {
    //                             down_grad_el / (1.0 + (-*operand_data_el).exp())
    //                         }
    //                     });
    //                 }
    //             }
    //         }
    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Sigmoid  ============================

#[derive(Debug)]
pub struct Sigmoid<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Sigmoid<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Sigmoid<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();

    //         Zip::from(self_data.deref_mut())
    //             .and(operand_data.deref())
    //             .par_for_each(|self_data_el, operand_data_el| {
    //                 *self_data_el = if *operand_data_el >= 15.0 {
    //                     1.0
    //                 } else if *operand_data_el <= -15.0 {
    //                     0.0
    //                 } else {
    //                     1.0 / (1.0 + (-*operand_data_el).exp())
    //                 }
    //             });
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let operand_data = self.operand.data();
    //             let down_grad = grad.deref();

    //             let zip = Zip::from(self_grad.deref_mut())
    //                 .and(down_grad)
    //                 .and(operand_data.deref());

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el = *down_grad_el * *operand_data_el * (1.0 - *operand_data_el)
    //                     });
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el += *down_grad_el * *operand_data_el * (1.0 - *operand_data_el)
    //                     });
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Hyper. Tangent  ============================

#[derive(Debug)]
pub struct Tanh<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Tanh<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Tanh<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();

    //         Zip::from(self_data.deref_mut())
    //             .and(operand_data.deref())
    //             .par_for_each(|self_data_el, operand_data_el| *self_data_el = operand_data_el.tanh());
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let operand_data = self.operand.data();
    //             let down_grad = grad.deref();

    //             let zip = Zip::from(self_grad.deref_mut())
    //                 .and(down_grad)
    //                 .and(operand_data.deref());

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el = *down_grad_el * (1.0 - operand_data_el.powi(2))
    //                     });
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el += *down_grad_el * (1.0 - operand_data_el.powi(2))
    //                     });
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Exponential  ============================

#[derive(Debug)]
pub struct Exp<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Exp<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Exp<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();

    //         Zip::from(self_data.deref_mut())
    //             .and(operand_data.deref())
    //             .par_for_each(|self_data_el, operand_data_el| *self_data_el = operand_data_el.exp());
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let operand_data = self.operand.data();
    //             let down_grad = grad.deref();

    //             let zip = Zip::from(self_grad.deref_mut())
    //                 .and(down_grad)
    //                 .and(operand_data.deref());

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el = *down_grad_el * *operand_data_el
    //                     });
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip.par_for_each(|self_grad_el, down_grad_el, operand_data_el| {
    //                         *self_grad_el += *down_grad_el * *operand_data_el
    //                     });
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Softmax  ============================

#[derive(Debug)]
pub struct Softmax<T: Node> {
    axis: usize,
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    jacobian: RefCell<Array2<f32>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Softmax<T> {
    pub fn new(operand: Rc<T>, axis: usize) -> Self {
        let shape = operand.data().raw_dim();
        let jacobian = RefCell::new(Tensor::zeros((shape[axis], shape[axis])));
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            axis,
            data,
            grad,
            jacobian,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Softmax<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();
    //         let axis = self.axis;

    //         Zip::from(operand_data.lanes(Axis(axis)))
    //             .and(self_data.lanes_mut(Axis(axis)))
    //             .for_each(|lane_self, lane_new| {
    //                 let max = lane_self.fold(std::f32::MIN, |x, y| x.max(*y));
    //                 let num = &lane_self.map(|el| (el - max).exp());
    //                 let den = num.sum();
    //                 Zip::from(lane_new)
    //                     .and(num)
    //                     .for_each(|lane_new_el, num_el| *lane_new_el = *num_el / den);
    //             });
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let operand_data = self.operand.data();
    //             let mut jacobian = self.jacobian.borrow_mut();
    //             let axis = self.axis;

    //             fn fill_jacobian(jacobian: &mut Array2<f32>, array: &ArrayView1<f32>) {
    //                 for (row_idx, (mut row, row_val)) in jacobian
    //                     .rows_mut()
    //                     .into_iter()
    //                     .zip(array.iter())
    //                     .enumerate()
    //                 {
    //                     for (col_idx, (grad, col_val)) in row
    //                         .as_slice_mut()
    //                         .unwrap()
    //                         .iter_mut()
    //                         .zip(array.iter())
    //                         .enumerate()
    //                     {
    //                         if row_idx == col_idx {
    //                             *grad = row_val * (1.0 - col_val);
    //                         } else {
    //                             *grad = -row_val * col_val;
    //                         }
    //                     }
    //                 }
    //             }

    //             let beta = match self.counter.backward_action() {
    //                 BackwardAction::Set => 0.0,
    //                 BackwardAction::Increment => 1.0,
    //             };

    //             Zip::from(self_grad.lanes_mut(Axis(axis)))
    //                 .and(operand_data.lanes(Axis(axis)))
    //                 .and(grad.lanes(Axis(axis)))
    //                 .for_each(|mut d_g_col, data_col, grad_col| {
    //                     fill_jacobian(&mut jacobian, &data_col);
    //                     general_mat_vec_mul(1.0, &jacobian, &grad_col, beta, &mut d_g_col);
    //                 });
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Transposition  ============================

#[derive(Debug)]
pub struct Transpose<T: Node> {
    data: RefCell<Tensor<T::Dim>>,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T: Node> Transpose<T> {
    pub fn new(operand: Rc<T>) -> Self {
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.clone()));
        let grad = RefCell::new(Tensor::zeros(shape));
        let requires_grad = operand.requires_grad();

        Self {
            data,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T: Node> Node for Transpose<T> {
    type Dim = T::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let operand_data = self.operand.data();

    //         self_data.assign(&operand_data.t());
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut self_grad = self.grad.borrow_mut();
    //             let down_grad = grad.deref();

    //             let zip = Zip::from(self_grad.deref_mut()).and(down_grad.t());

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => zip.par_for_each(|dest, src| *dest = *src),
    //                 BackwardAction::Increment => zip.par_for_each(|dest, src| *dest = *src),
    //             };
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Concatenate  ============================

#[derive(Debug)]
pub struct Concatenate<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node<Dim = Lhs::Dim>,
    Lhs::Dim: RemoveAxis,
{
    data: RefCell<Tensor<Lhs::Dim>>,
    axis: usize,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    lhs_grad: RefCell<Tensor<Lhs::Dim>>,
    rhs_grad: RefCell<Tensor<Lhs::Dim>>,
    requires_grad: bool,
}

impl<Lhs, Rhs> Concatenate<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node<Dim = Lhs::Dim>,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>, axis: usize) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let lhs_grad = Tensor::zeros(lhs.data().raw_dim());
        let rhs_grad = Tensor::zeros(rhs.data().raw_dim());
        let data = concatenate(Axis(axis), &[lhs_grad.view(), rhs_grad.view()]).unwrap();

        Self {
            data: RefCell::new(data),
            axis,
            lhs,
            rhs,
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for Concatenate<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node<Dim = Lhs::Dim>,
    Lhs::Dim: RemoveAxis,
{
    type Dim = Lhs::Dim;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.lhs.forward();
    //         self.rhs.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let lhs_data = self.lhs.data();
    //         let rhs_data = self.rhs.data();
    //         let axis = self.axis;

    //         let (mut lhs_, mut rhs_) = self_data
    //             .view_mut()
    //             .split_at(Axis(axis), lhs_data.len_of(Axis(axis)));
    //         Zip::from(lhs_data.deref())
    //             .and(&mut lhs_)
    //             .for_each(|single_el, fused_el| *fused_el = *single_el);
    //         Zip::from(rhs_data.deref())
    //             .and(&mut rhs_)
    //             .for_each(|single_el, fused_el| *fused_el = *single_el);
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut lhs_grad = self.lhs_grad.borrow_mut();
    //             let mut rhs_grad = self.rhs_grad.borrow_mut();
    //             let axis = self.axis;

    //             let (lhs_, rhs_) = grad
    //                 .view()
    //                 .split_at(Axis(axis), lhs_grad.len_of(Axis(axis)));

    //             let zip_lhs = Zip::from(lhs_grad.deref_mut()).and(&lhs_);
    //             let zip_rhs = Zip::from(rhs_grad.deref_mut()).and(&rhs_);

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip_lhs.for_each(|single_el, fused_el| *single_el = *fused_el);
    //                     zip_rhs.for_each(|single_el, fused_el| *single_el = *fused_el);
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip_lhs.for_each(|single_el, fused_el| *single_el += *fused_el);
    //                     zip_rhs.for_each(|single_el, fused_el| *single_el += *fused_el);
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.lhs.backward(&self.lhs_grad.borrow());
    //             self.rhs.backward(&self.rhs_grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Stack  ============================

#[derive(Debug)]
pub struct Stack<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node<Dim = Lhs::Dim>,
    Lhs::Dim: RemoveAxis,
{
    data: RefCell<Tensor<<Lhs::Dim as Dimension>::Larger>>,
    axis: usize,
    lhs: Rc<Lhs>,
    rhs: Rc<Rhs>,
    lhs_grad: RefCell<Tensor<Lhs::Dim>>,
    rhs_grad: RefCell<Tensor<Lhs::Dim>>,
    requires_grad: bool,
}

impl<Lhs, Rhs> Stack<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node<Dim = Lhs::Dim>,
    Lhs::Dim: RemoveAxis,
{
    pub fn new(lhs: Rc<Lhs>, rhs: Rc<Rhs>, axis: usize) -> Self {
        let requires_grad = lhs.requires_grad() || rhs.requires_grad();
        let lhs_grad = Tensor::zeros(lhs.data().raw_dim());
        let rhs_grad = Tensor::zeros(rhs.data().raw_dim());
        let data = stack(Axis(axis), &[lhs_grad.view(), rhs_grad.view()]).unwrap();

        Self {
            data: RefCell::new(data),
            axis,
            lhs,
            rhs,
            lhs_grad: RefCell::new(lhs_grad),
            rhs_grad: RefCell::new(rhs_grad),
            requires_grad,
        }
    }
}

impl<Lhs, Rhs> Node for Stack<Lhs, Rhs>
where
    Lhs: Node,
    Rhs: Node<Dim = Lhs::Dim>,
    Lhs::Dim: RemoveAxis,
{
    type Dim = <Lhs::Dim as Dimension>::Larger;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.lhs.forward();
    //         self.rhs.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let lhs_data = self.lhs.data();
    //         let rhs_data = self.rhs.data();
    //         let axis = self.axis;

    //         let mut subview_iter = self_data.axis_iter_mut(Axis(axis));

    //         let subview_left = subview_iter
    //             .next()
    //             .unwrap()
    //             .into_dimensionality::<Lhs::Dim>()
    //             .unwrap();
    //         let subview_right = subview_iter
    //             .next()
    //             .unwrap()
    //             .into_dimensionality::<Lhs::Dim>()
    //             .unwrap();

    //         Zip::from(lhs_data.deref())
    //             .and(subview_left)
    //             .for_each(|single_el, fused_el| *fused_el = *single_el);
    //         Zip::from(rhs_data.deref())
    //             .and(subview_right)
    //             .for_each(|single_el, fused_el| *fused_el = *single_el);
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             let mut lhs_grad = self.lhs_grad.borrow_mut();
    //             let mut rhs_grad = self.rhs_grad.borrow_mut();
    //             let axis = self.axis;

    //             let mut subview_iter = grad.axis_iter(Axis(axis));

    //             let subview_left = subview_iter
    //                 .next()
    //                 .unwrap()
    //                 .into_dimensionality::<Lhs::Dim>()
    //                 .unwrap();
    //             let subview_right = subview_iter
    //                 .next()
    //                 .unwrap()
    //                 .into_dimensionality::<Lhs::Dim>()
    //                 .unwrap();

    //             let zip_lhs = Zip::from(lhs_grad.deref_mut()).and(subview_left);
    //             let zip_rhs = Zip::from(rhs_grad.deref_mut()).and(subview_right);

    //             match self.counter.backward_action() {
    //                 BackwardAction::Set => {
    //                     zip_lhs.for_each(|single_el, fused_el| *single_el = *fused_el);
    //                     zip_rhs.for_each(|single_el, fused_el| *single_el = *fused_el);
    //                 }
    //                 BackwardAction::Increment => {
    //                     zip_lhs.for_each(|single_el, fused_el| *single_el += *fused_el);
    //                     zip_rhs.for_each(|single_el, fused_el| *single_el += *fused_el);
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.lhs.backward(&self.lhs_grad.borrow());
    //             self.rhs.backward(&self.rhs_grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// // ============================ Computational Graph Internal Component: Unsqueeze  ============================

#[derive(Debug)]
pub struct Unsqueeze<T>
where
    T: Node,
    T::Dim: RemoveAxis,
{
    data: RefCell<Tensor<<T::Dim as Dimension>::Larger>>,
    axis: usize,
    grad: RefCell<Tensor<T::Dim>>,
    operand: Rc<T>,
    requires_grad: bool,
}

impl<T> Unsqueeze<T>
where
    T: Node,
    T::Dim: RemoveAxis,
{
    pub fn new(operand: Rc<T>, axis: usize) -> Self {
        let requires_grad = operand.requires_grad();
        let shape = operand.data().raw_dim();
        let data = RefCell::new(Tensor::zeros(shape.insert_axis(Axis(axis))));
        let grad = RefCell::new(Tensor::zeros(shape));

        Self {
            data,
            axis,
            grad,
            operand,
            requires_grad,
        }
    }
}

impl<T> Node for Unsqueeze<T>
where
    T: Node,
    T::Dim: RemoveAxis,
{
    type Dim = <T::Dim as Dimension>::Larger;

    //     fn forward(&self) {
    //         if self.counter.forward_action() == ForwardAction::Cached {
    //             return;
    //         }

    //         self.operand.forward();

    //         let mut self_data = self.data.borrow_mut();
    //         let mut mut_array = self_data
    //             .axis_iter_mut(Axis(self.axis))
    //             .next()
    //             .unwrap()
    //             .into_dimensionality::<T::Dim>()
    //             .unwrap();
    //         let operand_data = self.operand.data();

    //         Zip::from(&mut mut_array)
    //             .and(operand_data.deref())
    //             .par_for_each(|self_data_el, operand_data_el| *self_data_el = *operand_data_el);
    //     }

    //     fn backward(&self, grad: &Ref<Tensor<Self::Dim>>) {
    //         {
    //             {
    //                 let mut self_grad = self.grad.borrow_mut();
    //                 let axis = self.axis;
    //                 let down_grad = grad
    //                     .axis_iter(Axis(axis))
    //                     .next()
    //                     .unwrap()
    //                     .into_dimensionality::<T::Dim>()
    //                     .unwrap();

    //                 let zip = Zip::from(self_grad.deref_mut()).and(&down_grad);

    //                 match self.counter.backward_action() {
    //                     BackwardAction::Set => {
    //                         zip.par_for_each(|self_grad_el, down_grad_el| *self_grad_el = *down_grad_el)
    //                     }
    //                     BackwardAction::Increment => zip
    //                         .par_for_each(|self_grad_el, down_grad_el| *self_grad_el += *down_grad_el),
    //                 }
    //             }
    //         }

    //         if self.counter.recurse_backward() {
    //             self.operand.backward(&self.grad.borrow());
    //         }
    //     }

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

// #[cfg(test)]
// mod tests {
//     use super::{accumulate, BackwardAction};
//     use ndarray::array;

//     #[test]
//     fn assign_test() {
//         let mut scalar_trgt = array![0.0];
//         let mut vector_trgt = array![0.0, 0.0, 0.0];
//         let mut matrix_trgt = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

//         let scalar = array![1.0];
//         let vector = array![1.0, 1.0, 1.0];
//         let matrix = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];

//         // Scalar scalar assignment.
//         accumulate(&mut scalar_trgt, &scalar, 1.0, &BackwardAction::Set);
//         assert!(scalar_trgt[0] - scalar[0] <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 0.0);

//         // Scalar scalar vector.
//         accumulate(&mut scalar_trgt, &vector, 1.0, &BackwardAction::Set);
//         assert!(scalar_trgt[0] - 3.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 0.0);

//         // Scalar scalar matrix.
//         accumulate(&mut scalar_trgt, &matrix, 1.0, &BackwardAction::Set);
//         assert!(scalar_trgt[0] - 9.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 0.0);

//         // Vector scalar assignment.
//         accumulate(&mut vector_trgt, &scalar, 1.0, &BackwardAction::Set);
//         assert_eq!(vector_trgt, array![1.0, 1.0, 1.0]);
//         vector_trgt.map_inplace(|el| *el = 0.0);

//         // Vector vector assignment.
//         accumulate(&mut vector_trgt, &vector, 1.0, &BackwardAction::Set);
//         assert_eq!(vector_trgt, array![1.0, 1.0, 1.0]);
//         vector_trgt.map_inplace(|el| *el = 0.0);

//         // Vector matrix assignment.
//         accumulate(&mut vector_trgt, &matrix, 1.0, &BackwardAction::Set);
//         assert_eq!(vector_trgt, array![3.0, 3.0, 3.0]);
//         vector_trgt.map_inplace(|el| *el = 0.0);

//         // Matrix scalar assignment.
//         accumulate(&mut matrix_trgt, &scalar, 1.0, &BackwardAction::Set);
//         assert_eq!(
//             matrix_trgt,
//             array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 0.0);

//         // Matrix vector assignment.
//         accumulate(&mut matrix_trgt, &vector, 1.0, &BackwardAction::Set);
//         assert_eq!(
//             matrix_trgt,
//             array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 0.0);

//         // Matrix matrix assignment.
//         accumulate(&mut matrix_trgt, &matrix, 1.0, &BackwardAction::Set);
//         assert_eq!(
//             matrix_trgt,
//             array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 0.0);
//     }

//     #[test]
//     fn scaled_assign_test() {
//         let mut scalar_trgt = array![0.0];
//         let mut vector_trgt = array![0.0, 0.0, 0.0];
//         let mut matrix_trgt = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];

//         let scalar = array![1.0];
//         let vector = array![1.0, 1.0, 1.0];
//         let matrix = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];

//         // Scalar scalar assignment.
//         accumulate(&mut scalar_trgt, &scalar, -1.0, &BackwardAction::Set);
//         assert!(scalar_trgt[0] - scalar[0] <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 0.0);

//         // Scalar scalar vector.
//         accumulate(&mut scalar_trgt, &vector, -1.0, &BackwardAction::Set);
//         assert!(scalar_trgt[0] - 3.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 0.0);

//         // Scalar scalar matrix.
//         accumulate(&mut scalar_trgt, &matrix, -1.0, &BackwardAction::Set);
//         assert!(scalar_trgt[0] - 9.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 0.0);

//         // Vector scalar assignment.
//         accumulate(&mut vector_trgt, &scalar, -1.0, &BackwardAction::Set);
//         assert_eq!(vector_trgt, -array![1.0, 1.0, 1.0]);
//         vector_trgt.map_inplace(|el| *el = 0.0);

//         // Vector vector assignment.
//         accumulate(&mut vector_trgt, &vector, -1.0, &BackwardAction::Set);
//         assert_eq!(vector_trgt, -array![1.0, 1.0, 1.0]);
//         vector_trgt.map_inplace(|el| *el = 0.0);

//         // Vector matrix assignment.
//         accumulate(&mut vector_trgt, &matrix, -1.0, &BackwardAction::Set);
//         assert_eq!(vector_trgt, -array![3.0, 3.0, 3.0]);
//         vector_trgt.map_inplace(|el| *el = 0.0);

//         // Matrix scalar assignment.
//         accumulate(&mut matrix_trgt, &scalar, -1.0, &BackwardAction::Set);
//         assert_eq!(
//             matrix_trgt,
//             -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 0.0);

//         // Matrix vector assignment.
//         accumulate(&mut matrix_trgt, &vector, -1.0, &BackwardAction::Set);
//         assert_eq!(
//             matrix_trgt,
//             -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 0.0);

//         // Matrix matrix assignment.
//         accumulate(&mut matrix_trgt, &matrix, -1.0, &BackwardAction::Set);
//         assert_eq!(
//             matrix_trgt,
//             -array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 0.0);
//     }

//     #[test]
//     fn add_assign_test() {
//         let mut scalar_trgt = array![5.0];
//         let mut vector_trgt = array![5.0, 5.0, 5.0];
//         let mut matrix_trgt = array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];

//         let scalar = array![5.0];
//         let vector = array![5.0, 5.0, 5.0];
//         let matrix = array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];

//         // Scalar scalar assignment.
//         accumulate(&mut scalar_trgt, &scalar, 1.0, &BackwardAction::Increment);
//         assert!(scalar_trgt[0] - 10.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 5.0);

//         // Scalar scalar vector.
//         accumulate(&mut scalar_trgt, &vector, 1.0, &BackwardAction::Increment);
//         assert!(scalar_trgt[0] - 20.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 5.0);

//         // Scalar scalar matrix.
//         accumulate(&mut scalar_trgt, &matrix, 1.0, &BackwardAction::Increment);
//         assert!(scalar_trgt[0] - 50.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 5.0);

//         // Vector scalar assignment.
//         accumulate(&mut vector_trgt, &scalar, 1.0, &BackwardAction::Increment);
//         assert_eq!(vector_trgt, array![10.0, 10.0, 10.0]);
//         vector_trgt.map_inplace(|el| *el = 5.0);

//         // Vector vector assignment.
//         accumulate(&mut vector_trgt, &vector, 1.0, &BackwardAction::Increment);
//         assert_eq!(vector_trgt, array![10.0, 10.0, 10.0]);
//         vector_trgt.map_inplace(|el| *el = 5.0);

//         // Vector matrix assignment.
//         accumulate(&mut vector_trgt, &matrix, 1.0, &BackwardAction::Increment);
//         assert_eq!(vector_trgt, array![20.0, 20.0, 20.0]);
//         vector_trgt.map_inplace(|el| *el = 5.0);

//         // Matrix scalar assignment.
//         accumulate(&mut matrix_trgt, &scalar, 1.0, &BackwardAction::Increment);
//         assert_eq!(
//             matrix_trgt,
//             array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 5.0);

//         // Matrix vector assignment.
//         accumulate(&mut matrix_trgt, &vector, 1.0, &BackwardAction::Increment);
//         assert_eq!(
//             matrix_trgt,
//             array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 5.0);

//         // Matrix matrix assignment.
//         accumulate(&mut matrix_trgt, &matrix, 1.0, &BackwardAction::Increment);
//         assert_eq!(
//             matrix_trgt,
//             array![[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 5.0);
//     }

//     #[test]
//     fn scaled_add_assign_test() {
//         let mut scalar_trgt = array![5.0];
//         let mut vector_trgt = array![5.0, 5.0, 5.0];
//         let mut matrix_trgt = array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];

//         let scalar = array![5.0];
//         let vector = array![5.0, 5.0, 5.0];
//         let matrix = array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];

//         // Scalar scalar assignment.
//         accumulate(&mut scalar_trgt, &scalar, -1.0, &BackwardAction::Increment);
//         assert!(scalar_trgt[0] - 0.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 5.0);

//         // Scalar scalar vector.
//         accumulate(&mut scalar_trgt, &vector, -1.0, &BackwardAction::Increment);
//         assert!(scalar_trgt[0] - 10.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 5.0);

//         // Scalar scalar matrix.
//         accumulate(&mut scalar_trgt, &matrix, -1.0, &BackwardAction::Increment);
//         assert!(scalar_trgt[0] - 40.0 <= f32::EPSILON);
//         scalar_trgt.map_inplace(|el| *el = 5.0);

//         // Vector scalar assignment.
//         accumulate(&mut vector_trgt, &scalar, -1.0, &BackwardAction::Increment);
//         assert_eq!(vector_trgt, array![0.0, 0.0, 0.0]);
//         vector_trgt.map_inplace(|el| *el = 5.0);

//         // Vector vector assignment.
//         accumulate(&mut vector_trgt, &vector, -1.0, &BackwardAction::Increment);
//         assert_eq!(vector_trgt, array![-0.0, -0.0, -0.0]);
//         vector_trgt.map_inplace(|el| *el = 5.0);

//         // Vector matrix assignment.
//         accumulate(&mut vector_trgt, &matrix, -1.0, &BackwardAction::Increment);
//         assert_eq!(vector_trgt, array![-10.0, -10.0, -10.0]);
//         vector_trgt.map_inplace(|el| *el = 5.0);

//         // Matrix scalar assignment.
//         accumulate(&mut matrix_trgt, &scalar, -1.0, &BackwardAction::Increment);
//         assert_eq!(
//             matrix_trgt,
//             array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 5.0);

//         // Matrix vector assignment.
//         accumulate(&mut matrix_trgt, &vector, -1.0, &BackwardAction::Increment);
//         assert_eq!(
//             matrix_trgt,
//             array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 5.0);

//         // Matrix matrix assignment.
//         accumulate(&mut matrix_trgt, &matrix, -1.0, &BackwardAction::Increment);
//         assert_eq!(
//             matrix_trgt,
//             array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
//         );
//         matrix_trgt.map_inplace(|el| *el = 5.0);
//     }
// }
