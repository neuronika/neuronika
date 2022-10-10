use ndarray::{DimMax, Dimension};

use cudnn::{BinaryOp, BinaryOpTensorDescriptor, NanPropagation, TensorDescriptor};

use crate::{
    autograd::Forward,
    cuda::cuarray::CuArray,
    utils::{Broadcast, Shared},
};

pub(crate) struct BinaryOperation<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    left_data: Shared<CuArray<f32, D>>,
    right_data: Shared<CuArray<f32, E>>,
    data: Shared<CuArray<f32, Broadcast<D, E>>>,
    op_desc: BinaryOpTensorDescriptor<f32>,
    left_desc: TensorDescriptor<f32>,
    right_desc: TensorDescriptor<f32>,
    data_desc: TensorDescriptor<f32>,
}

impl<D, E> BinaryOperation<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    pub(crate) fn new(
        op: BinaryOp,
        left_data: Shared<CuArray<f32, D>>,
        right_data: Shared<CuArray<f32, E>>,
        data: Shared<CuArray<f32, Broadcast<D, E>>>,
    ) -> Self {
        let nan_opt = NanPropagation::PropagateNaN;
        let op_desc = BinaryOpTensorDescriptor::new(op, nan_opt).unwrap();

        let left_desc = left_data.borrow().cudnn_tensor_desc();
        let right_desc = right_data.borrow().cudnn_tensor_desc();
        let data_desc = data.borrow().cudnn_tensor_desc();

        Self {
            left_data,
            right_data,
            data,
            op_desc,
            left_desc,
            right_desc,
            data_desc,
        }
    }
}

impl<D, E> Forward for BinaryOperation<D, E>
where
    D: Dimension + DimMax<E>,
    E: Dimension,
{
    fn forward(&self) {
        let left_data = self.left_data.borrow();
        let right_data = self.right_data.borrow();
        let mut data = self.data.borrow_mut();

        let cudnn = left_data.device().cudnn();

        cudnn
            .binary_tensor_op(
                &self.op_desc,
                1.0,
                &self.left_desc,
                left_data.buffer(),
                1.0,
                &self.right_desc,
                right_data.buffer(),
                0.0,
                &self.data_desc,
                data.buffer_mut(),
            )
            .unwrap()
    }
}
