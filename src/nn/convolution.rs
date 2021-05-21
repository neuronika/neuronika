use ndarray::{
    iter::{AxisChunksIter, AxisChunksIterMut},
    Array, ArrayBase, ArrayView, ArrayViewMut, Axis, Data, DataMut, Dimension, IntoDimension, Ix1,
    Ix2, Ix3, IxDyn, RawData, RemoveAxis, ShapeBuilder, Slice, ViewRepr, Zip,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

use crate::variable::{
    expect_tensor, expect_tensor_mut,
    node::{Backward, Data as NData, Forward, Gradient, Overwrite},
    Tensor, Var, VarDiff,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Padding Modes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// This trait defines the behaviour of the **padding modes**. All padding modes must implement it.
pub trait PaddingMode: Send + Sync + Clone {
    fn pad<D: ReflPad + ReplPad, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        array: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    );
}

/// Zero padding.
///
/// See also [`constant_pad`] for more informations.
#[derive(Clone)]
pub struct Zero;
/// Constant padding.
///
/// See also [`constant_pad`] for more informations.
#[derive(Clone)]
pub struct Constant(f32);
/// Reflective padding.
///
/// See also [`reflection_pad`] for more informations.
#[derive(Clone)]
pub struct Reflective;
/// Replicative padding.
///
/// See also [`replication_pad`] for more informations.
#[derive(Clone)]
pub struct Replicative;

impl PaddingMode for Zero {
    fn pad<D: Dimension, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        array: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    ) {
        constant_pad_inplace(array, original, padding, 0.);
    }
}

impl PaddingMode for Constant {
    fn pad<D: Dimension, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        array: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    ) {
        let value = self.0;
        constant_pad_inplace(array, original, padding, value);
    }
}

impl PaddingMode for Reflective {
    fn pad<D: ReflPad, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        array: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    ) {
        D::reflection_pad_inplace(array, original, padding);
    }
}

impl PaddingMode for Replicative {
    fn pad<D: ReplPad, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        &self,
        array: &mut ArrayBase<S, D>,
        original: &ArrayBase<T, D>,
        padding: &[usize],
    ) {
        D::replication_pad_inplace(array, original, padding);
    }
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolve Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait Convolve<Inp, Ker, Pad: PaddingMode> {
    type Output;

    fn convolve(
        input: Inp,
        kernel: Ker,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
    ) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolve Trait Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolve with non differentiable variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl<F1, F2, Pad> Convolve<Self, Var<F2>, Pad> for Var<F1>
where
    F1: NData + 'static,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    F2: NData<Dim = F1::Dim> + 'static,
    Pad: PaddingMode + 'static,
{
    type Output = Var<Convolution<F1, F2, Pad>>;
    fn convolve(
        mut input: Self,
        kernel: Var<F2>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
    ) -> Self::Output {
        input.past.merge(kernel.past);
        Var::from(
            Convolution::new(
                input.node,
                kernel.node,
                stride,
                dilation,
                padding,
                padding_mode,
            ),
            input.past,
        )
    }
}

// ~~~~~~~~~~~~~~~ Convolve with differentiable kernel and not differentiable input ~~~~~~~~~~~~~~~~
impl<F1, F2, B2, Pad> Convolve<Self, VarDiff<F2, B2>, Pad> for Var<F1>
where
    F1: NData + 'static,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    F2: NData<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = F2::Dim> + Overwrite,
    Pad: PaddingMode + 'static,
{
    type Output = VarDiff<Convolution<F1, F2, Pad>, ConvolutionUnaryBackward<F1, F2, B2, Pad>>;

    fn convolve(
        input: Self,
        kernel: VarDiff<F2, B2>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
    ) -> Self::Output {
        let node = ConvolutionUnaryBackward::new(
            kernel.node,
            input.node.clone(),
            kernel.var.node.clone(),
            stride,
            dilation,
            padding,
            padding_mode.clone(),
        );
        VarDiff::from(
            node,
            kernel.past,
            Var::convolve(input, kernel.var, stride, dilation, padding, padding_mode),
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolve with differentiable variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl<F1, B1, F2, B2, Pad> Convolve<Self, VarDiff<F2, B2>, Pad> for VarDiff<F1, B1>
where
    F1: NData + 'static,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    B1: Gradient<Dim = F1::Dim> + Overwrite,
    F2: NData<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = F2::Dim> + Overwrite,
    Pad: PaddingMode + 'static,
{
    type Output = VarDiff<Convolution<F1, F2, Pad>, ConvolutionBackward<F1, B1, F2, B2, Pad>>;

    fn convolve(
        mut input: Self,
        kernel: VarDiff<F2, B2>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
    ) -> Self::Output {
        input.past.merge(kernel.past);
        let node = ConvolutionBackward::new(
            input.node,
            kernel.node,
            input.var.node.clone(),
            kernel.var.node.clone(),
            stride,
            dilation,
            padding,
            padding_mode.clone(),
        );
        VarDiff::from(
            node,
            input.past,
            Var::convolve(
                input.var,
                kernel.var,
                stride,
                dilation,
                padding,
                padding_mode,
            ),
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolve with Groups Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait ConvolveWithGroups<Inp, Ker, Pad: PaddingMode> {
    type Output;

    fn convolve_with_groups(
        input: Inp,
        kernel: Ker,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
        groups: usize,
    ) -> Self::Output;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolve with Groups Trait Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolve with non differentiable variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl<F1, F2, Pad> ConvolveWithGroups<Self, Var<F2>, Pad> for Var<F1>
where
    F1: NData + 'static,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    F2: NData<Dim = F1::Dim> + 'static,
    Pad: PaddingMode + 'static,
{
    type Output = Var<GroupedConvolution<F1, F2, Pad>>;
    fn convolve_with_groups(
        mut input: Self,
        kernel: Var<F2>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
        groups: usize,
    ) -> Self::Output {
        input.past.merge(kernel.past);
        Var::from(
            GroupedConvolution::new(
                input.node,
                kernel.node,
                stride,
                dilation,
                padding,
                padding_mode,
                groups,
            ),
            input.past,
        )
    }
}

// ~~~~~~~~~~~~~~~ Convolve with differentiable kernel and not differentiable input ~~~~~~~~~~~~~~~~
impl<F1, F2, B2, Pad> ConvolveWithGroups<Self, VarDiff<F2, B2>, Pad> for Var<F1>
where
    F1: NData + 'static,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    F2: NData<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = F2::Dim> + Overwrite,
    Pad: PaddingMode + 'static,
{
    type Output =
        VarDiff<GroupedConvolution<F1, F2, Pad>, GroupedConvolutionUnaryBackward<F1, F2, B2, Pad>>;

    fn convolve_with_groups(
        input: Self,
        kernel: VarDiff<F2, B2>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
        groups: usize,
    ) -> Self::Output {
        let node = GroupedConvolutionUnaryBackward::new(
            kernel.node,
            input.node.clone(),
            kernel.var.node.clone(),
            stride,
            dilation,
            padding,
            padding_mode.clone(),
            groups,
        );
        VarDiff::from(
            node,
            kernel.past,
            Var::convolve_with_groups(
                input,
                kernel.var,
                stride,
                dilation,
                padding,
                padding_mode,
                groups,
            ),
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolve with differentiable variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl<F1, B1, F2, B2, Pad> ConvolveWithGroups<Self, VarDiff<F2, B2>, Pad> for VarDiff<F1, B1>
where
    F1: NData + 'static,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    B1: Gradient<Dim = F1::Dim> + Overwrite,
    F2: NData<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = F2::Dim> + Overwrite,
    Pad: PaddingMode + 'static,
{
    type Output =
        VarDiff<GroupedConvolution<F1, F2, Pad>, GroupedConvolutionBackward<F1, B1, F2, B2, Pad>>;

    fn convolve_with_groups(
        mut input: Self,
        kernel: VarDiff<F2, B2>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
        groups: usize,
    ) -> Self::Output {
        input.past.merge(kernel.past);
        let node = GroupedConvolutionBackward::new(
            input.node,
            kernel.node,
            input.var.node.clone(),
            kernel.var.node.clone(),
            stride,
            dilation,
            padding,
            padding_mode.clone(),
            groups,
        );
        VarDiff::from(
            node,
            input.past,
            Var::convolve_with_groups(
                input.var,
                kernel.var,
                stride,
                dilation,
                padding,
                padding_mode,
                groups,
            ),
        )
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolution Forward Structs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Convolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    input: Rc<Inp>,
    kernel: Rc<Ker>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    padding_mode: Pad,
    data: RefCell<Tensor<Inp::Dim>>,
    computed: Cell<bool>,
}

impl<Inp, Ker, Pad> Convolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    pub fn new(
        input: Rc<Inp>,
        kernel: Rc<Ker>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
    ) -> Self {
        // Computes the shape of the output feature map.
        let shape: Inp::Dim = {
            let (input_data, kernel_data) = (input.data(), kernel.data());
            conv_out_shape(
                input_data.shape(),
                kernel_data.shape(),
                &padding,
                stride,
                dilation,
            )
        };

        let (stride, dilation, padding) = (stride.to_vec(), dilation.to_vec(), padding.to_vec());
        let data = RefCell::new(Tensor::zeros(shape));

        Self {
            input,
            kernel,
            data,
            stride,
            dilation,
            padding,
            padding_mode,
            computed: Cell::new(false),
        }
    }
}

impl<Inp, Ker, Pad> NData for Convolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    type Dim = Inp::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Inp, Ker, Pad> Forward for Convolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
    <<Inp as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<Inp as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (input, kernel, mut output_map, stride, dilation, padding, padding_mode) = (
            self.input.data(),
            self.kernel.data(),
            self.data.borrow_mut(),
            &self.stride,
            &self.dilation,
            &self.padding,
            &self.padding_mode,
        );
        check_conv_args(input.shape(), kernel.shape(), padding, stride, dilation);

        // If there's no padding just performs the convolution.
        if padding.iter().all(|pad| *pad == 0) {
            convolution(&input, &kernel, &mut *output_map, stride, dilation);
        } else {
            // If there's padding to be applied, pads the input and then it performs the
            // convolution. Do note that here memory is allocated and then freed.
            let padded_input = pad(&*input, padding, padding_mode);
            convolution(&padded_input, &*kernel, &mut *output_map, stride, dilation);
        }
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Grouped Convolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct GroupedConvolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    input: Rc<Inp>,
    kernel: Rc<Ker>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    padding_mode: Pad,
    groups: usize,
    data: RefCell<Tensor<Inp::Dim>>,
    computed: Cell<bool>,
}

impl<Inp, Ker, Pad> GroupedConvolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    pub fn new(
        input: Rc<Inp>,
        kernel: Rc<Ker>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
        groups: usize,
    ) -> Self {
        let shape: Inp::Dim = {
            let (input_data, kernel_data) = (input.data(), kernel.data());
            conv_out_shape(
                input_data.shape(),
                kernel_data.shape(),
                padding,
                stride,
                dilation,
            )
        };
        let (stride, dilation, padding) = (stride.to_vec(), dilation.to_vec(), padding.to_vec());
        let data = RefCell::new(Tensor::zeros(shape));

        Self {
            input,
            kernel,
            data,
            stride,
            dilation,
            padding,
            padding_mode,
            groups,
            computed: Cell::new(false),
        }
    }
}

impl<Inp, Ker, Pad> NData for GroupedConvolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    type Dim = Inp::Dim;

    fn data(&self) -> Ref<Tensor<Self::Dim>> {
        self.data.borrow()
    }
}

impl<Inp, Ker, Pad> Forward for GroupedConvolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    <<Inp as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<Inp as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
    Pad: PaddingMode,
{
    fn forward(&self) {
        if self.was_computed() {
            return;
        }

        self.computed.set(true);
        let (input, kernel, mut output_map, stride, dilation, padding, padding_mode, groups) = (
            self.input.data(),
            self.kernel.data(),
            self.data.borrow_mut(),
            &self.stride,
            &self.dilation,
            &self.padding,
            &self.padding_mode,
            &self.groups,
        );
        check_conv_args(input.shape(), kernel.shape(), padding, stride, dilation);
        check_groups_args(input.shape(), kernel.shape(), *groups);

        // If there's no padding just performs the convolution.
        if padding.iter().all(|pad| *pad == 0) {
            convolution_with_groups(
                &*input,
                &*kernel,
                &mut *output_map,
                stride,
                dilation,
                *groups,
            );
        } else {
            // If there's padding, pads the input and then it performs the convolution.
            // Do note that here memory is allocated and then freed.
            let padded_input = pad(&*input, padding, padding_mode);
            convolution_with_groups(
                &padded_input,
                &*kernel,
                &mut *output_map,
                stride,
                dilation,
                *groups,
            );
        }
    }

    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolution Backward Structs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    input_grad: Rc<InpG>,
    kernel_grad: Rc<KerG>,
    gradient: RefCell<Option<Tensor<InpG::Dim>>>,
    input: Rc<InpD>,
    kernel: Rc<KerD>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    padding_mode: Pad,
    shape: InpD::Dim,
    overwrite: Cell<bool>,
}

impl<InpD, InpG, KerD, KerG, Pad> ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    #[allow(clippy::clippy::too_many_arguments)]
    pub fn new(
        input_grad: Rc<InpG>,
        kernel_grad: Rc<KerG>,
        input: Rc<InpD>,
        kernel: Rc<KerD>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
    ) -> Self {
        let shape: InpD::Dim = conv_out_shape(
            input.data().shape(),
            kernel.data().shape(),
            &padding,
            &stride,
            &dilation,
        );
        let gradient = RefCell::new(Some(Tensor::zeros(shape.clone())));
        let (stride, dilation, padding) = (stride.to_vec(), dilation.to_vec(), padding.to_vec());

        Self {
            input_grad,
            kernel_grad,
            gradient,
            shape,
            input,
            kernel,
            stride,
            dilation,
            padding,
            padding_mode,
            overwrite: Cell::new(true),
        }
    }
}

impl<InpD, InpG, KerD, KerG, Pad> Gradient for ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    type Dim = InpG::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<InpD, InpG, KerD, KerG, Pad> Overwrite for ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<InpD, InpG, KerD, KerG, Pad> Backward for ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
    <<InpD as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<InpD as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    fn backward(&self) {
        let gradient = self.gradient();

        let (
            mut input_grad,
            mut kernel_grad,
            input,
            kernel,
            padding,
            padding_mode,
            stride,
            dilation,
        ) = (
            self.input_grad.gradient_mut(),
            self.kernel_grad.gradient_mut(),
            self.input.data(),
            self.kernel.data(),
            &self.padding,
            &self.padding_mode,
            &self.stride,
            &self.dilation,
        );
        let (overwrite_input_grad, overwrite_kernel_grad) = (
            self.input_grad.can_overwrite(),
            self.kernel_grad.can_overwrite(),
        );

        if padding.iter().all(|pad| *pad == 0) {
            convolution_backward(
                &mut *input_grad,
                &mut *kernel_grad,
                &*gradient,
                &*input,
                &*kernel,
                padding,
                stride,
                dilation,
                overwrite_input_grad,
                overwrite_kernel_grad,
            );
        } else {
            let padded_input = pad(&input, padding, padding_mode);
            convolution_backward(
                &mut *input_grad,
                &mut *kernel_grad,
                &*gradient,
                &padded_input,
                &*kernel,
                padding,
                stride,
                dilation,
                overwrite_input_grad,
                overwrite_kernel_grad,
            );
        }

        if overwrite_input_grad {
            self.input_grad.set_overwrite(false);
        }
        if overwrite_kernel_grad {
            self.kernel_grad.set_overwrite(false);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolution Unary Backward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct ConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    kernel_grad: Rc<KerG>,
    gradient: RefCell<Option<Tensor<KerG::Dim>>>,
    input: Rc<InpD>,
    kernel: Rc<KerD>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    padding_mode: Pad,
    shape: InpD::Dim,
    overwrite: Cell<bool>,
}

impl<InpD, KerD, KerG, Pad> ConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    pub fn new(
        kernel_grad: Rc<KerG>,
        input: Rc<InpD>,
        kernel: Rc<KerD>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
    ) -> Self {
        let shape: InpD::Dim = conv_out_shape(
            input.data().shape(),
            kernel.data().shape(),
            &padding,
            &stride,
            &dilation,
        );
        let gradient = RefCell::new(Some(Tensor::zeros(shape.clone())));
        let (stride, dilation, padding) = (stride.to_vec(), dilation.to_vec(), padding.to_vec());

        Self {
            kernel_grad,
            gradient,
            shape,
            input,
            kernel,
            stride,
            dilation,
            padding,
            padding_mode,
            overwrite: Cell::new(true),
        }
    }
}

impl<InpD, KerD, KerG, Pad> Gradient for ConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    type Dim = KerG::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<InpD, KerD, KerG, Pad> Overwrite for ConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<InpD, KerD, KerG, Pad> Backward for ConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
    <<InpD as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<InpD as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    fn backward(&self) {
        let gradient = self.gradient();

        let (mut kernel_grad, input, kernel, padding, padding_mode, stride, dilation) = (
            self.kernel_grad.gradient_mut(),
            self.input.data(),
            self.kernel.data(),
            &self.padding,
            &self.padding_mode,
            &self.stride,
            &self.dilation,
        );
        let overwrite_kernel_grad = self.kernel_grad.can_overwrite();

        if padding.iter().all(|pad| *pad == 0) {
            convolution_unary_backward(
                &mut *kernel_grad,
                &*gradient,
                &*input,
                &*kernel,
                stride,
                dilation,
                overwrite_kernel_grad,
            );
        } else {
            let padded_input = pad(&input, padding, padding_mode);
            convolution_unary_backward(
                &mut *kernel_grad,
                &*gradient,
                &padded_input,
                &*kernel,
                stride,
                dilation,
                overwrite_kernel_grad,
            );
        }

        if overwrite_kernel_grad {
            self.kernel_grad.set_overwrite(false);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Grouped Convolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    input_grad: Rc<InpG>,
    kernel_grad: Rc<KerG>,
    gradient: RefCell<Option<Tensor<InpG::Dim>>>,
    input: Rc<InpD>,
    kernel: Rc<KerD>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    padding_mode: Pad,
    groups: usize,
    shape: InpD::Dim,
    overwrite: Cell<bool>,
}

impl<InpD, InpG, KerD, KerG, Pad> GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    #[allow(clippy::clippy::too_many_arguments)]
    pub fn new(
        input_grad: Rc<InpG>,
        kernel_grad: Rc<KerG>,
        input: Rc<InpD>,
        kernel: Rc<KerD>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
        groups: usize,
    ) -> Self {
        let shape: InpD::Dim = conv_out_shape(
            input.data().shape(),
            kernel.data().shape(),
            &padding,
            &stride,
            &dilation,
        );
        let gradient = RefCell::new(Some(Tensor::zeros(shape.clone())));
        let (stride, dilation, padding) = (stride.to_vec(), dilation.to_vec(), padding.to_vec());

        Self {
            input_grad,
            kernel_grad,
            gradient,
            shape,
            input,
            kernel,
            stride,
            dilation,
            padding,
            padding_mode,
            groups,
            overwrite: Cell::new(true),
        }
    }
}

impl<InpD, InpG, KerD, KerG, Pad> Gradient
    for GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    type Dim = InpG::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<InpD, InpG, KerD, KerG, Pad> Overwrite
    for GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<InpD, InpG, KerD, KerG, Pad> Backward
    for GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim> + Overwrite,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
    <<InpD as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<InpD as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    fn backward(&self) {
        let gradient = self.gradient();

        let (
            mut input_grad,
            mut kernel_grad,
            input,
            kernel,
            padding,
            padding_mode,
            stride,
            dilation,
            groups,
        ) = (
            self.input_grad.gradient_mut(),
            self.kernel_grad.gradient_mut(),
            self.input.data(),
            self.kernel.data(),
            &self.padding,
            &self.padding_mode,
            &self.stride,
            &self.dilation,
            &self.groups,
        );
        let (overwrite_input_grad, overwrite_kernel_grad) = (
            self.input_grad.can_overwrite(),
            self.kernel_grad.can_overwrite(),
        );

        if padding.iter().all(|pad| *pad == 0) {
            convolution_with_groups_backward(
                &mut *input_grad,
                &mut *kernel_grad,
                &*gradient,
                &*input,
                &*kernel,
                padding,
                stride,
                dilation,
                *groups,
                overwrite_input_grad,
                overwrite_kernel_grad,
            );
        } else {
            let padded_input = pad(&input, padding, padding_mode);
            convolution_with_groups_backward(
                &mut *input_grad,
                &mut *kernel_grad,
                &*gradient,
                &padded_input,
                &*kernel,
                padding,
                stride,
                dilation,
                *groups,
                overwrite_input_grad,
                overwrite_kernel_grad,
            );
        }

        if overwrite_input_grad {
            self.input_grad.set_overwrite(false);
        }
        if overwrite_kernel_grad {
            self.kernel_grad.set_overwrite(false);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Grouped Convolution Unary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct GroupedConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    kernel_grad: Rc<KerG>,
    gradient: RefCell<Option<Tensor<KerG::Dim>>>,
    input: Rc<InpD>,
    kernel: Rc<KerD>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    padding_mode: Pad,
    groups: usize,
    shape: InpD::Dim,
    overwrite: Cell<bool>,
}

impl<InpD, KerD, KerG, Pad> GroupedConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    #[allow(clippy::clippy::too_many_arguments)]
    pub fn new(
        kernel_grad: Rc<KerG>,
        input: Rc<InpD>,
        kernel: Rc<KerD>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
        groups: usize,
    ) -> Self {
        let shape: InpD::Dim = conv_out_shape(
            input.data().shape(),
            kernel.data().shape(),
            &padding,
            &stride,
            &dilation,
        );
        let gradient = RefCell::new(Some(Tensor::zeros(shape.clone())));
        let (stride, dilation, padding) = (stride.to_vec(), dilation.to_vec(), padding.to_vec());

        Self {
            kernel_grad,
            gradient,
            shape,
            input,
            kernel,
            stride,
            dilation,
            padding,
            padding_mode,
            groups,
            overwrite: Cell::new(true),
        }
    }
}

impl<InpD, KerD, KerG, Pad> Gradient for GroupedConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    type Dim = KerG::Dim;

    fn gradient(&self) -> Ref<Tensor<Self::Dim>> {
        expect_tensor(&self.gradient)
    }

    fn gradient_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        expect_tensor_mut(&self.gradient)
    }
}

impl<InpD, KerD, KerG, Pad> Overwrite for GroupedConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<InpD, KerD, KerG, Pad> Backward for GroupedConvolutionUnaryBackward<InpD, KerD, KerG, Pad>
where
    InpD: NData,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim> + Overwrite,
    Pad: PaddingMode,
    <<InpD as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<InpD as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    fn backward(&self) {
        let gradient = self.gradient();

        let (mut kernel_grad, input, kernel, padding, padding_mode, stride, dilation, groups) = (
            self.kernel_grad.gradient_mut(),
            self.input.data(),
            self.kernel.data(),
            &self.padding,
            &self.padding_mode,
            &self.stride,
            &self.dilation,
            &self.groups,
        );
        let overwrite_kernel_grad = self.kernel_grad.can_overwrite();

        if padding.iter().all(|pad| *pad == 0) {
            convolution_with_groups_unary_backward(
                &mut *kernel_grad,
                &*gradient,
                &*input,
                &*kernel,
                stride,
                dilation,
                *groups,
                overwrite_kernel_grad,
            );
        } else {
            let padded_input = pad(&input, padding, padding_mode);
            convolution_with_groups_unary_backward(
                &mut *kernel_grad,
                &*gradient,
                &padded_input,
                &*kernel,
                stride,
                dilation,
                *groups,
                overwrite_kernel_grad,
            );
        }

        if overwrite_kernel_grad {
            self.kernel_grad.set_overwrite(false);
        }
    }

    fn no_grad(&self) {
        *self.gradient.borrow_mut() = None;
    }

    fn with_grad(&self) {
        *self.gradient.borrow_mut() = Some(Tensor::zeros(self.shape.clone()));
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Utility Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Checks that the arguments are correct for the given **convolution**. It verifies that the
/// `padding`, `stride` and `dilation` slices are of the right length; their length must match the
/// dimensionality of the convolution. It also check that `kernel` and `input` are of the same
/// dimension and that the kernel size, after dilation is applied, **is not bigger** that the actual
/// input size.
///
/// # Arguments
///
/// * `input_shape` - the shape of the input map of the convolution
/// * `kernel_shape` - the shape of the kernel
/// * `padding` - the padding to be applied to the input
/// * `stride` - the stride for the cross-correlation
/// * `dilation` - the spacing between the kernel points
fn check_conv_args(
    input_shape: &[usize],
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) {
    // The type of convolution can be extrapolated by considering the number of input's dimension
    // skipping the first two, that are the batch size and input channels. The first two axes of
    // the input are always for the batch size and the number of input channels.
    let conv_dim = input_shape.len() - 2;
    assert_eq!(
        conv_dim,
        padding.len(),
        "error: invalid padding {:?} for {}d conv.",
        padding,
        conv_dim
    );

    assert_eq!(
        conv_dim,
        stride.len(),
        "error: invalid stride {:?} for {}d conv.",
        stride,
        conv_dim
    );

    assert_eq!(
        conv_dim,
        dilation.len(),
        "error: invalid dilation {:?} for {}d conv.",
        dilation,
        conv_dim
    );

    assert_eq!(
        kernel_shape.len(),
        input_shape.len(),
        "error: invalid kernel's shape {:?} for {}d conv",
        &kernel_shape,
        conv_dim
    );

    let dilated_kernel_size: Vec<usize> = kernel_shape
        .iter()
        .skip(2)
        .zip(dilation.iter())
        .map(|(kernel_dim, dilation_dim)| (kernel_dim - 1) * dilation_dim + 1)
        .collect();
    let padded_input_size: Vec<usize> = input_shape
        .iter()
        .skip(2)
        .zip(padding.iter())
        .map(|(input_dim, padding)| input_dim + padding * 2)
        .collect();
    padded_input_size
        .iter()
        .zip(dilated_kernel_size.iter())
        .for_each(|(padded_input_dim, dilated_kernel_dim)| {
            assert!(
                padded_input_dim >= dilated_kernel_dim,
                "Calculated padded input size per channel: {:?}. Kernel size: {:?}. The kernel size can't be greater than actual input size.",
                padded_input_size,
                dilated_kernel_size
            )
        });
}

/// Checks that the arguments are correct for the given **grouped convolution**. This function
/// should most of the time be used together with `check_conv_args`.
///
/// It enforces that both the number of **input channels** and **output channels** are divisible
/// by `groups`.
///
/// # Arguments
/// * `input_shape` - the shape of the input map of the convolution
/// * `kernel_shape` - the shape of the kernel
/// * `groups` -  the connections between inputs and outputs.
fn check_groups_args(input_shape: &[usize], kernel_shape: &[usize], groups: usize) {
    assert_eq!(
        input_shape[1] % groups,
        0,
        "error: in_channels {} is not disible by groups {}",
        input_shape[1],
        groups
    );
    assert_eq!(
        kernel_shape[0] % groups,
        0,
        "error: out_channels {} is not disible by groups {}",
        kernel_shape[0],
        groups
    );
}

/// Computes the shape of the array resulting from the **n**-dimensional convolution
/// performed with the given parameters.
///
/// **n** can be either **1**, **2** or **3**
///
/// # Arguments
///
/// * `input_shape` - the shape of the input
/// * `kernel_shape` - the shape of the kernel
/// * `padding` - the padding around the input
/// * `stride` - the stride
/// * `dilation` - the dilation
///
/// ## 1-dimensional convolution
///
/// The **input** must be of shape **(N, Cin, L)**
/// * **N** is the batch size
/// * **Cin** is the number of input channels
/// * **L** is the **length** of the input
///
/// The **kernel** must be of shape **(Cout, Cin, Lk)**
/// * **Cout** is the number of output channels
/// * **Cin** is the number of input channels
/// * **Lk** is the **length** of the kernel
///
/// The resulting output shape will be **(N, Cout, Lout)**
///
/// ## 2-dimensional convolution
/// The **input** must be of shape **(N, Cin, H, W)**
/// * **N** is the batch size
/// * **Cin** is the number of input channels
/// * **H** is the **height** of the input
/// * **W** is the **width** of the input
///
/// The **kernel** must be of shape **(Cout, Cin, Hk, Wk)**
/// * **Cout** is the number of output channels
/// * **Cin** is the number of input channels
/// * **Hk** is the **height** of the kernel
/// * **Wk** is the **width** of the kernel
///
/// The resulting output shape will be **(N, Cout, Hout, Wout)**
///
/// ## 3-dimensional convolution
///
/// The **input** must be of shape **(N, Cin, D, H, W)**
/// * **N** is the batch size
/// * **Cin** is the number of input channels
/// * **D** is the **depth** of the input
/// * **H** is the **height** of the input
/// * **W** is the **width** of the input
///
/// The **kernel** must be of shape **(Cout, Cin, Dk,  Hk, Wk)**
/// * **Cout** is the number of output channels
/// * **Cin** is the number of input channels
/// * **Dk** is the **depth** of the kernel
/// * **Hk** is the **height** of the kernel
/// * **Wk** is the **width** of the kernel
///
/// The resulting output shape will be **(N, Cout, Dout, Hout, Wout)**
fn conv_out_shape<D: Dimension>(
    input_shape: &[usize],
    kernel_shape: &[usize],
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> D {
    let in_shape_len = input_shape.len();
    // Initialize the dimension to be all 0s.
    let mut map_shape = D::zeros(in_shape_len);
    let map_shape_slice = map_shape.slice_mut();
    // Sets the batch size. The batch size doesn't change.
    map_shape_slice[0] = input_shape[0];
    // Sets the output channels.
    map_shape_slice[1] = kernel_shape[0];
    // First two components of the shape are always
    // the batch size and channels.
    itertools::izip!(
        map_shape_slice.iter_mut().skip(2), // Skips bacth size and out channels.
        input_shape.iter().skip(2),         // Skips batch size and out channels.
        kernel_shape.iter().skip(2),        // Skips out channels and in channels.
        padding,
        stride,
        dilation
    )
    .for_each(|(map_s, in_s, k_s, pd, stri, dil)| {
        *map_s = (in_s + 2 * pd - dil * (k_s - 1) - 1) / stri + 1
    });
    map_shape
}

/// Computes the shape of the array resulting from the **n**-dimensional convolution
/// performed with the given parameters. `input_shape` is assumed to be the shape of an **already**
/// padded input.
///
/// # Arguments
/// * `input_shape` - the shape of the input
/// * `kernel_shape` - the shape of the kernel
/// * `stride` - the stride
/// * `dilation` - the dilation
fn conv_out_shape_padded<D: Dimension>(
    input_shape: &[usize],
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> D {
    let in_shape_len = input_shape.len();
    // Initialize the dimension to be all 0s.
    let mut map_shape = D::zeros(in_shape_len);
    let map_shape_slice = map_shape.slice_mut();
    // Sets the batch size. The batch size doesn't change.
    map_shape_slice[0] = input_shape[0];
    // Sets the output channels.
    map_shape_slice[1] = kernel_shape[0];
    // First two components of the shape are always
    // the batch size and channels.
    itertools::izip!(
        map_shape_slice.iter_mut().skip(2), // Skips bacth size and out channels.
        input_shape.iter().skip(2),         // Skips batch size and out channels.
        kernel_shape.iter().skip(2),        // Skips out channels and in channels.
        stride,
        dilation
    )
    .for_each(|(map_s, in_s, k_s, stri, dil)| *map_s = (in_s - dil * (k_s - 1) - 1) / stri + 1);
    map_shape
}

/// Computes the shape of the **input** after the padding is applied.
///
/// # Arguments
///
/// * `input_shape` - the shape of the input
/// * `padding` - the padding around the input
fn padded_shape<D: Dimension>(input_shape: &[usize], padding: &[usize]) -> D {
    let in_shape_len = input_shape.len();
    let mut padded_input_shape = D::zeros(in_shape_len);
    padded_input_shape[0] = input_shape[0];
    padded_input_shape[1] = input_shape[1];
    padded_input_shape
        .slice_mut()
        .iter_mut()
        .skip(2)
        .zip(input_shape.iter().skip(2))
        .zip(padding.iter())
        .for_each(|((padded_s, original_s), pad)| *padded_s = original_s + 2 * pad);
    padded_input_shape
}

/// Pads `array` accordingly to `padding` and `padding_mode`. Returns the padded array.
///
/// # Arguments
///
/// * `array` - the array to be padded
/// * `padding` - the padding around to be applied to input
/// * `padding_mode` - the type of padding
fn pad<D: Dimension, T: PaddingMode>(
    array: &Array<f32, D>,
    padding: &[usize],
    padding_mode: &T,
) -> Array<f32, D>
where
    <D as Dimension>::Smaller: RemoveAxis,
    <<D as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    let mut padded = {
        let padded_shape = padded_shape::<D>(array.shape(), padding);
        Array::<f32, D>::zeros(padded_shape)
    };
    let (padded_raw_dim, original_raw_dim) = (padded.raw_dim(), array.raw_dim());
    // The dimension of a single raw sample in the batch.
    let (padded_inner_dimensions, original_inner_dimensions) = (
        padded_raw_dim.slice().iter().skip(2),
        original_raw_dim.slice().iter().skip(2),
    );
    // The number of single raw samples in the batch.
    let outer_dimension: usize = original_raw_dim.slice().iter().take(2).product();

    // Reshapes by removing an axis, so that all samples can be iterated on and padded.
    let (mut input_view_dimension, mut original_view_dimension): (
        <D as Dimension>::Smaller,
        <D as Dimension>::Smaller,
    ) = (
        <D as Dimension>::Smaller::zeros(padded_raw_dim.ndim() - 1),
        <D as Dimension>::Smaller::zeros(original_raw_dim.ndim() - 1),
    );
    input_view_dimension[0] = outer_dimension;
    original_view_dimension[0] = outer_dimension;

    input_view_dimension
        .slice_mut()
        .iter_mut()
        .skip(1)
        .zip(padded_inner_dimensions)
        .for_each(|(view_dim, inner_dim)| *view_dim = *inner_dim);
    original_view_dimension
        .slice_mut()
        .iter_mut()
        .skip(1)
        .zip(original_inner_dimensions)
        .for_each(|(view_dim, inner_dim)| *view_dim = *inner_dim);

    let (mut input_view_mut, original_view) = (
        padded.view_mut().into_shape(input_view_dimension).unwrap(),
        array.view().into_shape(original_view_dimension).unwrap(),
    );

    input_view_mut
        .outer_iter_mut()
        .into_par_iter()
        .zip(original_view.outer_iter())
        .for_each(|(mut pad_sample, original_sample)| {
            padding_mode.pad(&mut pad_sample, &original_sample, padding)
        });
    padded
}

/// Returns an **unpadded** view of `array`. This method is supposed to be used only with arrays
/// of dimension greater or equal than *3*. The length of the padding slice must be *dim - 2* where
/// *dim* is the array's dimensionality.
fn unpad<'a, S: Data<Elem = f32> + 'a, D: Dimension + 'a>(
    array: &'a ArrayBase<S, D>,
    padding: &[usize],
) -> ArrayBase<ViewRepr<&'a f32>, D> {
    array.slice_each_axis(|ax| {
        let (ax_index, ax_len) = (ax.axis.index(), array.len_of(ax.axis));
        let range = {
            if ax_index > 1 && padding[ax_index - 2] != 0 {
                padding[ax_index - 2] as isize..-(padding[ax_index - 2] as isize)
            } else {
                0..ax_len as isize
            }
        };
        Slice::from(range)
    })
}

/// Computes the shape of a rolling window view.
///
/// # Arguments
/// * `input` - input array
/// * `window_shape` - the shape of each of the windows
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
fn compute_rolling_window_shape<D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Vec<usize> {
    let mut win_indices_shape: D =
        conv_out_shape_padded(input.shape(), window_shape, stride, dilation);
    win_indices_shape[1] = 1;
    win_indices_shape
        .slice()
        .iter()
        .chain(window_shape.iter().skip(1))
        .cloned()
        .collect()
}

/// Computes the strides of a rolling window view.
///
/// # Arguments
/// * `input` - input array
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
fn compute_rolling_window_strides<D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    stride: &[usize],
    dilation: &[usize],
) -> Vec<usize> {
    let indexing_strides: Vec<isize> = {
        let view = input.slice_each_axis(|ax| {
            let axis_index = ax.axis.index();
            if axis_index == 0 || axis_index == 1 {
                Slice::new(0, None, 1) // Batch stride and channel stride
            } else {
                Slice::new(0, None, stride[ax.axis.index() - 2] as isize)
            }
        });
        let view_strides: &[isize] = view.strides();
        view_strides.to_vec()
    };
    // Number of in channels doesn't count for the window's strides,
    // it must be left unchanged.
    let window_strides: Vec<isize> = input
        .strides()
        .iter()
        .skip(1) // Skip out channels
        .enumerate()
        .map(|(i, is)| {
            if i < 1 {
                *is
            } else {
                *is * (dilation[i - 1] as isize)
            }
        })
        .collect();
    indexing_strides
        .iter()
        .chain(window_strides.iter())
        .map(|s| *s as usize)
        .collect()
}

/// Returns a **rolling window view** of the input array.
///
/// # Arguments
/// * `input` - input array
/// * `window_shape` - the shape of each of the windows
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
fn as_windows<'a, D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayView<'a, f32, IxDyn> {
    let rolling_window_shape: Vec<usize> =
        compute_rolling_window_shape(input, window_shape, stride, dilation);
    let rolling_window_strides: Vec<usize> =
        compute_rolling_window_strides(input, stride, dilation);

    unsafe {
        ArrayView::from_shape_ptr(
            rolling_window_shape.strides(rolling_window_strides),
            input.as_ptr(),
        )
    }
}

/// Returns a **mutable rolling window view** of the input array.
///
/// # Arguments
/// * `input` - input array
/// * `window_shape` - the shape of each of the windows
/// * `stride` - the stride
/// * `dilation` - the spacing between each element of the windows
fn as_windows_mut<'a, D: Dimension, S: DataMut<Elem = f32>>(
    input: &mut ArrayBase<S, D>,
    window_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> ArrayViewMut<'a, f32, IxDyn> {
    let rolling_window_shape: Vec<usize> =
        compute_rolling_window_shape(input, window_shape, stride, dilation);
    let rolling_window_strides: Vec<usize> =
        compute_rolling_window_strides(input, stride, dilation);

    unsafe {
        ArrayViewMut::from_shape_ptr(
            rolling_window_shape.strides(rolling_window_strides),
            input.as_mut_ptr(),
        )
    }
}

/// Computes **sig2col**, **im2col** and **vol2col**.
///
/// # Arguments
/// * `input` - input array
/// * `kernel_shape` - the shape of the kernel
/// * `padding` - the padding to be applied to `input`
/// * `stride` - the stride
/// * `dilation` - the dilation
fn to_col<D: Dimension, S: Data<Elem = f32>>(
    input: &ArrayBase<S, D>,
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) -> Array<f32, Ix2> {
    let mut o_shape = conv_out_shape_padded::<D>(input.shape(), kernel_shape, stride, dilation);
    o_shape[1] = 1;
    let (im2col_h, im2col_w): (usize, usize) = {
        (
            kernel_shape.iter().skip(1).product(),
            o_shape.slice().iter().product(),
        )
    };
    as_windows(&input, kernel_shape, stride, dilation)
        .into_owned()
        .into_shape((im2col_w, im2col_h))
        .unwrap()
        .reversed_axes()
}

/// Reshapes the array in input into a matrix so that only the dimension of
/// axis 0 is preserved.
///
/// **Panics** if `array` is not standard layout.
fn flatten<D: Dimension, S: RawData>(array: ArrayBase<S, D>) -> ArrayBase<S, Ix2> {
    let (kernel_shape, mut new_shape) = (array.raw_dim(), Ix2::zeros(2));
    new_shape[0] = kernel_shape[0];
    new_shape[1] = kernel_shape.slice().iter().skip(1).product();
    array.into_shape(new_shape).unwrap()
}

/// Puts the axis corresponding to the output channels of the feature map `array`
/// in the last position and returns the input with swapped axes.
fn permute_channels<D: Dimension>(
    array: ArrayBase<ViewRepr<&f32>, D>,
) -> ArrayBase<ViewRepr<&f32>, D> {
    let mut dim = array.raw_dim();
    dim.set_last_elem(0);
    let dim_len = dim.ndim() - 1;
    dim.slice_mut()
        .iter_mut()
        .zip(1..=dim_len)
        .for_each(|(ax, el)| *ax = el);
    array.permuted_axes(dim)
}

/// Assigns the **2-dimensional** convolution result to the **n-dimensional** feature map.
fn assign_to_output_map<D: Dimension, S: DataMut<Elem = f32>>(
    out_map: &mut ArrayBase<S, D>,
    flat_result: Array<f32, Ix2>,
) {
    let batch_size = out_map.shape()[0];
    let mut sample_size = out_map.raw_dim();
    sample_size[0] = 1;

    let convolved_samples =
        flat_result.axis_chunks_iter(Axis(1), flat_result.len_of(Axis(1)) / batch_size);
    let samples = out_map.axis_chunks_iter_mut(Axis(0), 1);

    samples
        .into_iter()
        .zip(convolved_samples.into_iter())
        .for_each(|(mut sample, incoming_result)| {
            Zip::from(&mut sample)
                .and(
                    &incoming_result
                        .as_standard_layout()
                        .into_shape(sample_size.clone())
                        .unwrap(),
                )
                .for_each(|sample_el, incoming_el| *sample_el = *incoming_el)
        });
}

/// Assigns to the **n**-dimensional feature map's gradient `dest` the **2**-dimensional
/// array `columns`. This method encapsulates the functionalities of **col2sig**, **col2im** and
/// **col2vol**.
///
/// **n** can be either 3, 4 or 5.
fn assign_from_cols<D: Dimension, S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
    dest: &mut ArrayBase<S, D>,
    columns: ArrayBase<T, Ix2>,
    kernel_shape: &[usize],
    stride: &[usize],
    dilation: &[usize],
) {
    let mut dest_windows_mut = as_windows_mut(dest, kernel_shape, stride, dilation);
    let from_cols = columns.into_shape(dest_windows_mut.raw_dim()).unwrap();

    Zip::from(&mut dest_windows_mut)
        .and(&from_cols)
        .par_for_each(|dest_el, src_el| *dest_el += *src_el);
}

/// Partitions the **flattened input**, the **flattened kernel** and the **output map**
/// so that they can be used in a grouped convolution.
fn group_inputs<'a, D: Dimension, S: Data<Elem = f32>, T: DataMut<Elem = f32>>(
    input: &'a ArrayBase<S, D>,
    kernel: &'a ArrayBase<S, D>,
    out_map: &'a mut ArrayBase<T, D>,
    groups: usize,
) -> (
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIterMut<'a, f32, D>,
) {
    // Splits the input map along the channels.
    let input_groups = input.axis_chunks_iter(Axis(1), input.len_of(Axis(1)) / groups);
    // Splits the kernel along the output channels.
    let kernel_groups = kernel.axis_chunks_iter(Axis(0), kernel.len_of(Axis(0)) / groups);
    // Splits the outputmap along the channels.
    let out_map_groups = out_map.axis_chunks_iter_mut(Axis(1), out_map.len_of(Axis(1)) / groups);

    (input_groups, kernel_groups, out_map_groups)
}

/// Iterators needed for the **backward pass** of a grouped convolution.
type GroupedBackwardArgs<'a, D> = (
    AxisChunksIterMut<'a, f32, D>,
    AxisChunksIterMut<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
);

/// Iterators needed for the **backward pass** of a grouped convolution where the kernel
/// is the only differentiable variable.
type GroupedBackwardArgsUnary<'a, D> = (
    AxisChunksIterMut<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
    AxisChunksIter<'a, f32, D>,
);

/// Partitions the **input gradient**, the **kernel gradient** and the **output map
/// gradient** so that they can be used in the backward pass of the grouped convolution.
fn group_gradients<'a, D: Dimension, S: DataMut<Elem = f32>, U: Data<Elem = f32>>(
    input_grad: &'a mut ArrayBase<S, D>,
    kernel_grad: &'a mut ArrayBase<S, D>,
    out_map_grad: &'a ArrayBase<U, D>,
    input: &'a ArrayBase<U, D>,
    kernel: &'a ArrayBase<U, D>,
    groups: usize,
) -> GroupedBackwardArgs<'a, D> {
    let input_grad_groups =
        input_grad.axis_chunks_iter_mut(Axis(1), input_grad.len_of(Axis(1)) / groups);
    let kernel_grad_groups =
        kernel_grad.axis_chunks_iter_mut(Axis(0), kernel_grad.len_of(Axis(0)) / groups);
    let out_map_grad_groups =
        out_map_grad.axis_chunks_iter(Axis(1), out_map_grad.len_of(Axis(1)) / groups);
    let input_groups = input.axis_chunks_iter(Axis(1), input.len_of(Axis(1)) / groups);
    let kernel_groups = kernel.axis_chunks_iter(Axis(0), kernel.len_of(Axis(0)) / groups);

    (
        input_grad_groups,
        kernel_grad_groups,
        out_map_grad_groups,
        input_groups,
        kernel_groups,
    )
}

fn group_gradients_unary<'a, D: Dimension, S: DataMut<Elem = f32>, U: Data<Elem = f32>>(
    kernel_grad: &'a mut ArrayBase<S, D>,
    out_map_grad: &'a ArrayBase<U, D>,
    input: &'a ArrayBase<U, D>,
    kernel: &'a ArrayBase<U, D>,
    groups: usize,
) -> GroupedBackwardArgsUnary<'a, D> {
    let kernel_grad_groups =
        kernel_grad.axis_chunks_iter_mut(Axis(0), kernel_grad.len_of(Axis(0)) / groups);
    let out_map_grad_groups =
        out_map_grad.axis_chunks_iter(Axis(1), out_map_grad.len_of(Axis(1)) / groups);
    let input_groups = input.axis_chunks_iter(Axis(1), input.len_of(Axis(1)) / groups);
    let kernel_groups = kernel.axis_chunks_iter(Axis(0), kernel.len_of(Axis(0)) / groups);

    (
        kernel_grad_groups,
        out_map_grad_groups,
        input_groups,
        kernel_groups,
    )
}

/// Performs an **n-dimensional** convolution where **n** can be either *1*, *2* or *3*.
/// Do note that this function doesn't take into account *groups* nor *padding*. The padding is
/// assumed to be already applied to the input map when this function is called.
///
/// The resulting output map is stored in `output`.
/// # Arguments
///
/// * `input` - the input map
/// * `kernel` - the kernel
/// * `output` - the output map where the convolution result will be stored
/// * `stride` - the stride controls the stride for the cross-correlation
/// * `dilation` - the dilation controls the spacing between the kernel points
fn convolution<D: Dimension, S: Data<Elem = f32>, T: DataMut<Elem = f32>>(
    input: &ArrayBase<S, D>,
    kernel: &ArrayBase<S, D>,
    output: &mut ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
) {
    let (flat_kernel, flat_input) = (
        flatten(kernel.view()),
        to_col(&input, kernel.shape(), stride, dilation),
    );
    let convolution_result = flat_kernel.dot(&flat_input);
    assign_to_output_map(output, convolution_result);
}

/// Performs the **backpropagation** for an an **n-dimensional** convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// # Arguments
/// * `input_grad` - the gradient of the input map
/// * `kernel_grad` - the gradient of the kernel
/// * `grad` - the incoming gradient **d_out**.
/// * `input` - the input map
/// * `kernel` - the kernel
/// * `padding` - the padding that must be taken into account while accumulating the input's
/// gradient
/// * `stride` - the stride
/// * `dilation` - the dilation
/// * `overwrite_input_grad`  - specifies the kind of accumulation operation to be performed on
/// the input's gradient
/// * `overwrite_kernel_grad` - specifies the kind of accumulation operation to be performed on
/// the kernel's gradient
#[allow(clippy::too_many_arguments)]
fn convolution_backward<
    D: Dimension,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
    U: Data<Elem = f32>,
>(
    input_grad: &mut ArrayBase<S, D>,
    kernel_grad: &mut ArrayBase<S, D>,
    grad: &ArrayBase<T, D>,
    input: &ArrayBase<U, D>,
    kernel: &ArrayBase<T, D>,
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
    overwrite_input_grad: bool,
    overwrite_kernel_grad: bool,
) {
    // Flattens the incoming gradient.
    let gradient = permute_channels(grad.view());
    let gradient_as_standard = gradient.as_standard_layout();
    let flat_gradient = flatten(gradient_as_standard);

    // Computes the kernel's gradient.
    let kernel_gradient = flat_gradient
        .dot(&to_col(input, kernel.shape(), stride, dilation).t())
        .into_shape(kernel_grad.raw_dim())
        .unwrap();

    // Assigns the kernel's gradient.
    let kernel_grad_zip = Zip::from(kernel_grad).and(&kernel_gradient);
    if overwrite_kernel_grad {
        kernel_grad_zip
            .for_each(|kernel_grad_el, incoming_grad_el| *kernel_grad_el = *incoming_grad_el);
    } else {
        kernel_grad_zip
            .for_each(|kernel_grad_el, incoming_grad_el| *kernel_grad_el += *incoming_grad_el);
    }

    // Computes the input's gradient.
    let flat_kernel = flatten(kernel.view());
    let input_gradient = flat_kernel.t().dot(&flat_gradient);

    // If padding is not present, just assigns the gradient to the input.
    if padding.iter().all(|pad| *pad == 0) {
        assign_from_cols(input_grad, input_gradient, kernel.shape(), stride, dilation);
    } else {
        // If padding is present a buffer is needed to accumulate the input's incoming gradient.
        let mut buffer: Array<f32, D> =
            Array::zeros(padded_shape::<D>(input_grad.shape(), padding));
        assign_from_cols(
            &mut buffer,
            input_gradient,
            kernel.shape(),
            stride,
            dilation,
        );

        // The actual input's incoming gradient is extracted from the buffer and assigned.
        let actual_gradient = unpad(&buffer, padding);
        let input_gradient_zip = Zip::from(input_grad).and(actual_gradient);
        if overwrite_input_grad {
            input_gradient_zip
                .for_each(|input_grad_el, incoming_grad_el| *input_grad_el = *incoming_grad_el);
        } else {
            input_gradient_zip
                .for_each(|input_grad_el, incoming_grad_el| *input_grad_el += *incoming_grad_el);
        }
    }
}

/// Performs the **backpropagation** for an an **n-dimensional** convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// This function should be used in those circumstances in which the kernel is the only
/// differentiable variable, such as the first layer of a CNN module.
fn convolution_unary_backward<
    D: Dimension,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
    U: Data<Elem = f32>,
>(
    kernel_grad: &mut ArrayBase<S, D>,
    grad: &ArrayBase<T, D>,
    input: &ArrayBase<U, D>,
    kernel: &ArrayBase<T, D>,
    stride: &[usize],
    dilation: &[usize],
    overwrite_kernel_grad: bool,
) {
    // Flattens the incoming gradient.
    let gradient = permute_channels(grad.view());
    let gradient_as_standard = gradient.as_standard_layout();
    let flat_gradient = flatten(gradient_as_standard);

    // Computes the kernel's gradient.
    let kernel_gradient = flat_gradient
        .dot(&to_col(input, kernel.shape(), stride, dilation).t())
        .into_shape(kernel_grad.raw_dim())
        .unwrap();

    // Assigns the kernel's gradient.
    let kernel_grad_zip = Zip::from(kernel_grad).and(&kernel_gradient);
    if overwrite_kernel_grad {
        kernel_grad_zip
            .for_each(|kernel_grad_el, incoming_grad_el| *kernel_grad_el = *incoming_grad_el);
    } else {
        kernel_grad_zip
            .for_each(|kernel_grad_el, incoming_grad_el| *kernel_grad_el += *incoming_grad_el);
    }
}

/// Performs an **n-dimensional grouped** convolution where **n** can be either *1*, *2* or *3*.
///
/// Do note that this function doesn't take into account *padding*. The padding is
/// assumed to be already applied to the input map when this function is called.
///
/// # Arguments
/// * `input` - the input map
/// * `kernel` - the kernel
/// * `output` - the output map where the convolution result will be stored
/// * `stride` - the stride
/// * `dilation` - the dilation
/// * `groups` - the number of groups
fn convolution_with_groups<D: Dimension>(
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    output: &mut Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
) {
    let (input_groups, kernel_groups, output_buffer_groups) =
        group_inputs(input, kernel, output, groups);
    kernel_groups
        .into_par_iter()
        .zip(input_groups.into_iter())
        .zip(output_buffer_groups.into_iter())
        .for_each(|((kernel, input), mut output)| {
            convolution(&input, &kernel, &mut output, stride, dilation);
        });
}

/// Performs the **backpropagation** for an an **n-dimensional** grouped convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// # Arguments
/// * `input_grad` - the gradient of the input map
/// * `kernel_grad` - the gradient of the kernel
/// * `grad` - the incoming gradient **d_out**.
/// * `input` - the input map
/// * `kernel` - the kernel
/// * `padding` - the padding that must be taken into account while accumulating the input's
/// * `stride` - the stride
/// * `dilation` - the dilation
/// * `groups` - the number of groups
/// * `overwrite_input_grad`  - specifies the kind of accumulation operation to be performed on
/// the input gradient
/// * `overwrite_kernel_grad` - specifies the kind of accumulation operation to be performed on
/// the kernel gradient
#[allow(clippy::too_many_arguments)]
fn convolution_with_groups_backward<D: Dimension>(
    input_grad: &mut Array<f32, D>,
    kernel_grad: &mut Array<f32, D>,
    grad: &Array<f32, D>,
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    padding: &[usize],
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    overwrite_input_grad: bool,
    overwrite_kernel_grad: bool,
) {
    let (input_grad_groups, kernel_grad_groups, grad_groups, input_groups, kernel_groups) =
        group_gradients(input_grad, kernel_grad, &grad, &input, &kernel, groups);

    grad_groups
        .into_par_iter()
        .zip(kernel_grad_groups.into_iter())
        .zip(input_grad_groups.into_iter())
        .zip(kernel_groups.into_iter())
        .zip(input_groups.into_iter())
        .for_each(
            |((((gradient, mut kernel_gradient), mut input_gradient), kernel), input)| {
                convolution_backward(
                    &mut input_gradient,
                    &mut kernel_gradient,
                    &gradient,
                    &input,
                    &kernel,
                    padding,
                    stride,
                    dilation,
                    overwrite_input_grad,
                    overwrite_kernel_grad,
                )
            },
        );
}

/// Performs the **backpropagation** for an an **n-dimensional** grouped convolution where
/// **n** can be either *1*, *2* or *3*.
///
/// This function should be used in those circumstances in which the kernel is the only
/// differentiable variable, such as the first layer of a CNN module.
#[allow(clippy::clippy::too_many_arguments)]
fn convolution_with_groups_unary_backward<D: Dimension>(
    kernel_grad: &mut Array<f32, D>,
    grad: &Array<f32, D>,
    input: &Array<f32, D>,
    kernel: &Array<f32, D>,
    stride: &[usize],
    dilation: &[usize],
    groups: usize,
    overwrite_kernel_grad: bool,
) {
    let (kernel_grad_groups, grad_groups, input_groups, kernel_groups) =
        group_gradients_unary(kernel_grad, &grad, &input, &kernel, groups);

    grad_groups
        .into_par_iter()
        .zip(kernel_grad_groups.into_iter())
        .zip(kernel_groups.into_iter())
        .zip(input_groups.into_iter())
        .for_each(|(((gradient, mut kernel_gradient), kernel), input)| {
            convolution_unary_backward(
                &mut kernel_gradient,
                &gradient,
                &input,
                &kernel,
                stride,
                dilation,
                overwrite_kernel_grad,
            )
        });
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Paddings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// A [`ndarray::Dimension`] that supports **reflective padding**.
pub trait ReflPad: Dimension {
    fn reflection_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Self>,
        padding: &[usize],
    ) -> Array<f32, Self>;

    fn reflection_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    );
}

/// A [`ndarray::Dimension`] that supports **replicative padding**.
pub trait ReplPad: Dimension {
    fn replication_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Self>,
        padding: &[usize],
    ) -> Array<f32, Self>;

    fn replication_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    );
}
/// Pads the input array with a constant value.
///
/// # Arguments
///
/// * `input` - the array to be padded.
///
/// * `padding` - the amount of padding for each dimension.
///
/// * `value` - the value for the padding.
///
/// # Examples
///
/// ```
/// use neuronika::nn;
///
/// let arr = ndarray::array![
///    [1., 2., 3.],
///    [4., 5., 6.],
///    [7., 8., 9.]
/// ];
/// let padded = nn::constant_pad(&arr, (1, 1), 0.);
/// let result = ndarray::array![
///    [0., 0., 0., 0., 0.],
///    [0., 1., 2., 3., 0.],
///    [0., 4., 5., 6., 0.],
///    [0., 7., 8., 9., 0.],
///    [0., 0., 0., 0., 0.]
/// ];
///
/// assert_eq!(padded, result);
/// ```
pub fn constant_pad<S, D, E>(input: &ArrayBase<S, D>, padding: E, val: f32) -> Array<f32, D>
where
    D: Dimension,
    S: DataMut<Elem = f32>,
    E: IntoDimension<Dim = D>,
{
    let padding_into_dim = padding.into_dimension();
    let padded_shape = {
        let mut padded_shape = input.raw_dim();
        padded_shape
            .slice_mut()
            .iter_mut()
            .zip(padding_into_dim.slice().iter())
            .for_each(|(ax_len, pad)| *ax_len += pad * 2);
        padded_shape
    };
    let mut padded = Array::zeros(padded_shape);
    constant_pad_inplace(&mut padded, &input, padding_into_dim.slice(), val);
    padded
}

/// Pads the input array with a constant value. The operation is done inplace.
fn constant_pad_inplace<S, T, D>(
    input: &mut ArrayBase<S, D>,
    original: &ArrayBase<T, D>,
    padding: &[usize],
    val: f32,
) where
    D: Dimension,
    S: DataMut<Elem = f32>,
    T: Data<Elem = f32>,
{
    input.map_inplace(|el| *el = val);
    let mut orig_portion = input.view_mut();
    orig_portion.slice_each_axis_inplace(|ax| {
        let (ax_index, ax_len) = (ax.axis.index(), original.len_of(ax.axis));
        let range = {
            if padding[ax_index] != 0 {
                padding[ax_index] as isize..-(padding[ax_index] as isize)
            } else {
                0..ax_len as isize
            }
        };
        Slice::from(range)
    });
    orig_portion.assign(original);
}

/// Pads the input array using the **reflection** of the input boundary.
///
/// Only **1**, **2** and **3** dimensional arrays support reflective padding.
///
/// # Arguments
///
/// * `input` - the array to be padded.
///
/// * `padding` - the amount of padding for each dimension.
///
/// # Examples
///
/// ```
/// use neuronika::nn;
///
/// let arr = ndarray::array![
///    [1., 2., 3.],
///    [4., 5., 6.],
///    [7., 8., 9.]
/// ];
/// let padded = nn::reflection_pad(&arr, (1, 1));
/// let result = ndarray::array![
///    [5., 4., 5., 6., 5.],
///    [2., 1., 2., 3., 2.],
///    [5., 4., 5., 6., 5.],
///    [8., 7., 8., 9., 8.],
///    [5., 4., 5., 6., 5.]
/// ];
///
/// assert_eq!(padded, result);
/// ```
pub fn reflection_pad<D, E>(input: &Array<f32, D>, padding: E) -> Array<f32, D>
where
    D: ReflPad,
    E: IntoDimension<Dim = D>,
{
    D::reflection_pad(&input, padding.into_dimension().slice())
}

/// Pads the input array using the **replication** of the input boundary.
///
/// Only **1**, **2** and **3** dimensional arrays support replicative padding.
///
/// # Arguments
///
/// * `input` - the array to be padded.
///
/// * `padding` - the amount of padding for each dimension.
///
/// # Examples
///
/// ```
/// use neuronika::nn;
///
/// let arr = ndarray::array![
///    [1., 2., 3.],
///    [4., 5., 6.],
///    [7., 8., 9.]
/// ];
/// let padded = nn::replication_pad(&arr, (1, 1));
/// let result = ndarray::array![
///    [1., 1., 2., 3., 3.],
///    [1., 1., 2., 3., 3.],
///    [4., 4., 5., 6., 6.],
///    [7., 7., 8., 9., 9.],
///    [7., 7., 8., 9., 9.]
/// ];
///
/// assert_eq!(padded, result);
/// ```
pub fn replication_pad<D, E>(input: &Array<f32, D>, padding: E) -> Array<f32, D>
where
    D: ReplPad,
    E: IntoDimension<Dim = D>,
{
    D::replication_pad(&input, padding.into_dimension().slice())
}

impl ReflPad for Ix1 {
    fn reflection_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix1>,
        padding: &[usize],
    ) -> Array<f32, Ix1> {
        let out_len = {
            let len = input.len();
            let pad = padding[0];
            len + pad * 2
        };
        let mut out = Array::<f32, _>::zeros(out_len);
        Self::reflection_pad_inplace(&mut out, input, padding);
        out
    }

    fn reflection_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Ix1>,
        input: &ArrayBase<T, Ix1>,
        padding: &[usize],
    ) {
        let mut pos;
        let (in_len, out_len, pad) = { (input.len(), to_pad.len(), padding[0]) };
        let (in_slice, out_slice) = (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap());
        for (i, out_slice_el) in out_slice.iter_mut().enumerate().take(out_len) {
            if i < pad {
                pos = pad * 2 - i;
            } else if i >= pad && i < in_len + pad {
                pos = i;
            } else {
                pos = (in_len + pad - 1) * 2 - i;
            }
            pos -= pad;
            *out_slice_el = in_slice[pos];
        }
    }
}

impl ReflPad for Ix2 {
    fn reflection_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix2>,
        padding: &[usize],
    ) -> Array<f32, Ix2> {
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let mut out = Array::<f32, _>::zeros((out_len_x, out_len_y));
        Self::reflection_pad_inplace(&mut out, &input, &padding);
        out
    }

    fn reflection_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Ix2>,
        input: &ArrayBase<T, Ix2>,
        padding: &[usize],
    ) {
        let (mut pos_x, mut pos_y);
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap()) };
        for i in 0..out_len_x {
            for j in 0..out_len_y {
                if j < pad_y {
                    pos_x = pad_y * 2 - j;
                } else if j >= pad_y && j < len_y + pad_y {
                    pos_x = j;
                } else {
                    pos_x = (len_y + pad_y - 1) * 2 - j;
                }
                pos_x -= pad_y;

                if i < pad_x {
                    pos_y = pad_x * 2 - i;
                } else if i >= pad_x && i < len_x + pad_x {
                    pos_y = i;
                } else {
                    pos_y = (len_x + pad_x - 1) * 2 - i;
                }
                pos_y -= pad_x;
                slice_out[i * out_len_y + j] = slice_in[pos_y * len_y + pos_x];
            }
        }
    }
}

impl ReflPad for Ix3 {
    fn reflection_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix3>,
        padding: &[usize],
    ) -> Array<f32, Ix3> {
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let mut out = Array::<f32, _>::zeros((out_len_z, out_len_x, out_len_y));
        Self::reflection_pad_inplace(&mut out, &input, padding);
        out
    }

    fn reflection_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    ) {
        let (mut pos_x, mut pos_y, mut pos_z);
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap()) };

        for z in 0..out_len_z {
            for i in 0..out_len_x {
                for j in 0..out_len_y {
                    if j < pad_y {
                        pos_x = pad_y * 2 - j;
                    } else if j >= pad_y && j < len_y + pad_y {
                        pos_x = j;
                    } else {
                        pos_x = (len_y + pad_y - 1) * 2 - j;
                    }
                    pos_x -= pad_y;

                    if i < pad_x {
                        pos_y = pad_x * 2 - i;
                    } else if i >= pad_x && i < len_x + pad_x {
                        pos_y = i;
                    } else {
                        pos_y = (len_x + pad_x - 1) * 2 - i;
                    }
                    pos_y -= pad_x;

                    if z < pad_z {
                        pos_z = pad_z * 2 - z;
                    } else if z >= pad_z && z < len_z + pad_z {
                        pos_z = z;
                    } else {
                        pos_z = (len_z + pad_z - 1) * 2 - z;
                    }
                    pos_z -= pad_z;
                    slice_out[z * out_len_y * out_len_x + i * out_len_y + j] =
                        slice_in[pos_z * len_y * len_x + pos_y * len_y + pos_x];
                }
            }
        }
    }
}

impl ReplPad for Ix1 {
    fn replication_pad<S: Data<Elem = f32>>(
        input: &ArrayBase<S, Ix1>,
        padding: &[usize],
    ) -> Array<f32, Ix1> {
        let out_len = {
            let len = input.len();
            let pad = padding[0];
            len + pad * 2
        };
        let mut out = Array::<f32, _>::zeros(out_len);
        Self::replication_pad_inplace(&mut out, &input, padding);
        out
    }

    fn replication_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    ) {
        let mut pos;
        let (in_len, out_len, pad) = (input.len(), to_pad.len(), padding[0]);
        let (in_slice, out_slice) = (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap());
        for (j, out_slice_el) in out_slice.iter_mut().enumerate().take(out_len) {
            if j < pad {
                pos = pad;
            } else if j >= pad && j < in_len + pad {
                pos = j;
            } else {
                pos = in_len + pad - 1;
            }
            pos -= pad;
            *out_slice_el = in_slice[pos];
        }
    }
}

impl ReplPad for Ix2 {
    fn replication_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix2>,
        padding: &[usize],
    ) -> Array<f32, Ix2> {
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let mut out = Array::<f32, _>::zeros((out_len_x, out_len_y));
        Self::replication_pad_inplace(&mut out, &input, padding);
        out
    }

    fn replication_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    ) {
        let (mut pos_x, mut pos_y);
        let (len_x, len_y) = {
            let in_sp = input.shape();
            (in_sp[0], in_sp[1])
        };
        let (pad_x, pad_y) = (padding[0], padding[1]);
        let (out_len_x, out_len_y) = (len_x + pad_x * 2, len_y + pad_y * 2);
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap()) };
        for i in 0..out_len_x {
            for j in 0..out_len_y {
                if j < pad_y {
                    pos_x = pad_y;
                } else if j >= pad_y && j < len_y + pad_y {
                    pos_x = j;
                } else {
                    pos_x = len_y + pad_y - 1;
                }
                pos_x -= pad_y;

                if i < pad_x {
                    pos_y = pad_x;
                } else if i >= pad_x && i < len_x + pad_x {
                    pos_y = i;
                } else {
                    pos_y = len_x + pad_x - 1;
                }
                pos_y -= pad_x;
                slice_out[i * out_len_y + j] = slice_in[pos_y * len_y + pos_x];
            }
        }
    }
}

impl ReplPad for Ix3 {
    fn replication_pad<S: DataMut<Elem = f32>>(
        input: &ArrayBase<S, Ix3>,
        padding: &[usize],
    ) -> Array<f32, Ix3> {
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let mut out = Array::<f32, _>::zeros((out_len_z, out_len_x, out_len_y));
        Self::replication_pad_inplace(&mut out, &input, padding);
        out
    }

    fn replication_pad_inplace<S: DataMut<Elem = f32>, T: Data<Elem = f32>>(
        to_pad: &mut ArrayBase<S, Self>,
        input: &ArrayBase<T, Self>,
        padding: &[usize],
    ) {
        let (mut pos_x, mut pos_y, mut pos_z);
        let (len_x, len_y, len_z) = {
            let in_sp = input.shape();
            (in_sp[1], in_sp[2], in_sp[0])
        };
        let (pad_x, pad_y, pad_z) = (padding[1], padding[2], padding[0]);
        let (out_len_x, out_len_y, out_len_z) =
            (len_x + pad_x * 2, len_y + pad_y * 2, len_z + pad_z * 2);
        let (slice_in, slice_out) = { (input.as_slice().unwrap(), to_pad.as_slice_mut().unwrap()) };
        for z in 0..out_len_z {
            for i in 0..out_len_x {
                for j in 0..out_len_y {
                    if j < pad_y {
                        pos_x = pad_y;
                    } else if j >= pad_y && j < len_y + pad_y {
                        pos_x = j;
                    } else {
                        pos_x = len_y + pad_y - 1;
                    }
                    pos_x -= pad_y;

                    if i < pad_x {
                        pos_y = pad_x;
                    } else if i >= pad_x && i < len_x + pad_x {
                        pos_y = i;
                    } else {
                        pos_y = len_x + pad_x - 1;
                    }
                    pos_y -= pad_x;

                    if z < pad_z {
                        pos_z = pad_z;
                    } else if z >= pad_z && z < len_z + pad_z {
                        pos_z = z;
                    } else {
                        pos_z = len_z + pad_z - 1;
                    }
                    pos_z -= pad_z;

                    slice_out[z * out_len_y * out_len_x + i * out_len_y + j] =
                        slice_in[pos_z * len_y * len_x + pos_y * len_y + pos_x];
                }
            }
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod tests {
    #[test]
    fn im2col() {
        let input = ndarray::array![
            [
                [0., 1., 2., 3.],
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
                [12., 13., 14., 15.]
            ],
            [
                [0., 1., 2., 3.],
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
                [12., 13., 14., 15.]
            ],
            [
                [0., 1., 2., 3.],
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
                [12., 13., 14., 15.]
            ],
        ];
        // We must reshape the input, consider it as a bidimensional signal
        // with 3 channels each of 4 x 4.
        let d = input.clone().into_shape((1, 3, 4, 4)).unwrap();

        let im2col = ndarray::array![
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 2.0, 5.0, 6.0],
            [2.0, 3.0, 6.0, 7.0],
            [4.0, 5.0, 8.0, 9.0],
            [5.0, 6.0, 9.0, 10.0],
            [6.0, 7.0, 10.0, 11.0],
            [8.0, 9.0, 12.0, 13.0],
            [9.0, 10.0, 13.0, 14.0],
            [10.0, 11.0, 14.0, 15.0],
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 2.0, 5.0, 6.0],
            [2.0, 3.0, 6.0, 7.0],
            [4.0, 5.0, 8.0, 9.0],
            [5.0, 6.0, 9.0, 10.0],
            [6.0, 7.0, 10.0, 11.0],
            [8.0, 9.0, 12.0, 13.0],
            [9.0, 10.0, 13.0, 14.0],
            [10.0, 11.0, 14.0, 15.0],
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 2.0, 5.0, 6.0],
            [2.0, 3.0, 6.0, 7.0],
            [4.0, 5.0, 8.0, 9.0],
            [5.0, 6.0, 9.0, 10.0],
            [6.0, 7.0, 10.0, 11.0],
            [8.0, 9.0, 12.0, 13.0],
            [9.0, 10.0, 13.0, 14.0],
            [10.0, 11.0, 14.0, 15.0]
        ];
        assert_eq!(im2col, super::to_col(&d, &[1, 3, 3, 3], &[1, 1], &[1, 1]));

        // Now let's increase the batch size by 1.
        let input_batch = ndarray::stack(ndarray::Axis(0), &[input.view(), input.view()]).unwrap();
        // We must reshape the input, consider it as 2 bidimensional signals
        // with 3 channels each of 4 x 4.
        let d = input_batch.into_shape((2, 3, 4, 4)).unwrap();

        // The im2col's result. Note that the im2col of signals
        // from the batch are concatenated along the columns.
        assert_eq!(
            ndarray::concatenate(ndarray::Axis(1), &[im2col.view(), im2col.view()]).unwrap(),
            super::to_col(&d, &[1, 3, 3, 3], &[1, 1], &[1, 1])
        );
        // The nice thing about to_col is that it works for 1d, 2d, and 3d convolutions.
    }

    #[test]
    fn flatten() {
        use super::*;
        use ndarray::stack;
        // This is a kernel of 4 filters, reshaped output should be of shape (4,9).
        let kernel1 = (0..9)
            .map(|el| el as f32)
            .collect::<Array<f32, _>>()
            .into_shape((3, 3))
            .unwrap();
        let kernel2 = (9..18)
            .map(|el| el as f32)
            .collect::<Array<f32, _>>()
            .into_shape((3, 3))
            .unwrap();
        let kernel3 = (18..27)
            .map(|el| el as f32)
            .collect::<Array<f32, _>>()
            .into_shape((3, 3))
            .unwrap();
        let flattened = stack(Axis(0), &[kernel1.view(), kernel2.view(), kernel3.view()]).unwrap();
        assert_eq!(
            flatten(flattened),
            (0..27)
                .map(|el| el as f32)
                .collect::<Array<f32, _>>()
                .into_shape((3, 9))
                .unwrap()
        );
    }

    #[test]
    fn conv_args_ok() {
        // This is the input of a two dimensional convolution.
        // It is formed by 1 signal having 2 channels each of 4 x 4.
        let conv_input = ndarray::Array::<f32, _>::zeros((1, 2, 4, 4));
        super::check_conv_args(conv_input.shape(), &[1, 2, 2, 2], &[0, 0], &[1, 1], &[1, 1]);
    }

    #[test]
    #[should_panic(expected = "error: invalid kernel's shape [1, 2, 2] for 2d conv")]
    fn conv_args_invalid_kernel() {
        // This is the input of a two dimensional convolution.
        // It is formed by 1 signal having 2 channels each of 4 x 4.
        let conv_input = ndarray::Array::<f32, _>::zeros((1, 2, 4, 4));
        super::check_conv_args(conv_input.shape(), &[1, 2, 2], &[0, 0], &[1, 1], &[1, 1]);
    }

    #[test]
    fn constant_pad() {
        let arr = ndarray::Array::range(0., 25., 1.)
            .into_shape((5, 5))
            .unwrap();
        let padded = super::constant_pad(&arr, [1, 2], 8.);
        assert_eq!(
            padded,
            ndarray::array![
                [8., 8., 8., 8., 8., 8., 8., 8., 8.],
                [8., 8., 0., 1., 2., 3., 4., 8., 8.],
                [8., 8., 5., 6., 7., 8., 9., 8., 8.],
                [8., 8., 10., 11., 12., 13., 14., 8., 8.],
                [8., 8., 15., 16., 17., 18., 19., 8., 8.],
                [8., 8., 20., 21., 22., 23., 24., 8., 8.],
                [8., 8., 8., 8., 8., 8., 8., 8., 8.],
            ]
        );
    }

    #[test]
    fn replication_pad_1d() {
        let arr = ndarray::Array::range(0., 5., 1.);
        let padded = super::replication_pad(&arr, [2]);
        assert_eq!(padded, ndarray::array![0., 0., 0., 1., 2., 3., 4., 4., 4.],);
    }

    #[test]
    fn replication_pad_2d() {
        let arr = ndarray::Array::range(0., 25., 1.)
            .into_shape((5, 5))
            .unwrap();
        let padded = super::replication_pad(&arr, [1, 2]);
        assert_eq!(
            padded,
            ndarray::array![
                [0., 0., 0., 1., 2., 3., 4., 4., 4.],
                [0., 0., 0., 1., 2., 3., 4., 4., 4.],
                [5., 5., 5., 6., 7., 8., 9., 9., 9.],
                [10., 10., 10., 11., 12., 13., 14., 14., 14.],
                [15., 15., 15., 16., 17., 18., 19., 19., 19.],
                [20., 20., 20., 21., 22., 23., 24., 24., 24.],
                [20., 20., 20., 21., 22., 23., 24., 24., 24.],
            ]
        );
    }

    #[test]
    fn replication_pad_3d() {
        let arr = ndarray::Array::range(0., 125., 1.)
            .into_shape((5, 5, 5))
            .unwrap();
        let padded = super::replication_pad(&arr, [1, 2, 3]);
        assert_eq!(
            padded,
            ndarray::array![
                [
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [5., 5., 5., 5., 6., 7., 8., 9., 9., 9., 9.],
                    [10., 10., 10., 10., 11., 12., 13., 14., 14., 14., 14.],
                    [15., 15., 15., 15., 16., 17., 18., 19., 19., 19., 19.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.]
                ],
                [
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [0., 0., 0., 0., 1., 2., 3., 4., 4., 4., 4.],
                    [5., 5., 5., 5., 6., 7., 8., 9., 9., 9., 9.],
                    [10., 10., 10., 10., 11., 12., 13., 14., 14., 14., 14.],
                    [15., 15., 15., 15., 16., 17., 18., 19., 19., 19., 19.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.],
                    [20., 20., 20., 20., 21., 22., 23., 24., 24., 24., 24.]
                ],
                [
                    [25., 25., 25., 25., 26., 27., 28., 29., 29., 29., 29.],
                    [25., 25., 25., 25., 26., 27., 28., 29., 29., 29., 29.],
                    [25., 25., 25., 25., 26., 27., 28., 29., 29., 29., 29.],
                    [30., 30., 30., 30., 31., 32., 33., 34., 34., 34., 34.],
                    [35., 35., 35., 35., 36., 37., 38., 39., 39., 39., 39.],
                    [40., 40., 40., 40., 41., 42., 43., 44., 44., 44., 44.],
                    [45., 45., 45., 45., 46., 47., 48., 49., 49., 49., 49.],
                    [45., 45., 45., 45., 46., 47., 48., 49., 49., 49., 49.],
                    [45., 45., 45., 45., 46., 47., 48., 49., 49., 49., 49.]
                ],
                [
                    [50., 50., 50., 50., 51., 52., 53., 54., 54., 54., 54.],
                    [50., 50., 50., 50., 51., 52., 53., 54., 54., 54., 54.],
                    [50., 50., 50., 50., 51., 52., 53., 54., 54., 54., 54.],
                    [55., 55., 55., 55., 56., 57., 58., 59., 59., 59., 59.],
                    [60., 60., 60., 60., 61., 62., 63., 64., 64., 64., 64.],
                    [65., 65., 65., 65., 66., 67., 68., 69., 69., 69., 69.],
                    [70., 70., 70., 70., 71., 72., 73., 74., 74., 74., 74.],
                    [70., 70., 70., 70., 71., 72., 73., 74., 74., 74., 74.],
                    [70., 70., 70., 70., 71., 72., 73., 74., 74., 74., 74.]
                ],
                [
                    [75., 75., 75., 75., 76., 77., 78., 79., 79., 79., 79.],
                    [75., 75., 75., 75., 76., 77., 78., 79., 79., 79., 79.],
                    [75., 75., 75., 75., 76., 77., 78., 79., 79., 79., 79.],
                    [80., 80., 80., 80., 81., 82., 83., 84., 84., 84., 84.],
                    [85., 85., 85., 85., 86., 87., 88., 89., 89., 89., 89.],
                    [90., 90., 90., 90., 91., 92., 93., 94., 94., 94., 94.],
                    [95., 95., 95., 95., 96., 97., 98., 99., 99., 99., 99.],
                    [95., 95., 95., 95., 96., 97., 98., 99., 99., 99., 99.],
                    [95., 95., 95., 95., 96., 97., 98., 99., 99., 99., 99.]
                ],
                [
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [105., 105., 105., 105., 106., 107., 108., 109., 109., 109., 109.],
                    [110., 110., 110., 110., 111., 112., 113., 114., 114., 114., 114.],
                    [115., 115., 115., 115., 116., 117., 118., 119., 119., 119., 119.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.]
                ],
                [
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [100., 100., 100., 100., 101., 102., 103., 104., 104., 104., 104.],
                    [105., 105., 105., 105., 106., 107., 108., 109., 109., 109., 109.],
                    [110., 110., 110., 110., 111., 112., 113., 114., 114., 114., 114.],
                    [115., 115., 115., 115., 116., 117., 118., 119., 119., 119., 119.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.],
                    [120., 120., 120., 120., 121., 122., 123., 124., 124., 124., 124.]
                ]
            ]
        );
    }

    #[test]
    fn reflection_pad_1d() {
        let arr = ndarray::Array::range(0., 5., 1.);
        let padded = super::reflection_pad(&arr, [2]);
        assert_eq!(padded, ndarray::array![2., 1., 0., 1., 2., 3., 4., 3., 2.],);
    }

    #[test]
    fn reflection_pad_2d() {
        let arr = ndarray::Array::range(0., 25., 1.)
            .into_shape((5, 5))
            .unwrap();
        let padded = super::reflection_pad(&arr, [1, 2]);
        assert_eq!(
            padded,
            ndarray::array![
                [7., 6., 5., 6., 7., 8., 9., 8., 7.],
                [2., 1., 0., 1., 2., 3., 4., 3., 2.],
                [7., 6., 5., 6., 7., 8., 9., 8., 7.],
                [12., 11., 10., 11., 12., 13., 14., 13., 12.],
                [17., 16., 15., 16., 17., 18., 19., 18., 17.],
                [22., 21., 20., 21., 22., 23., 24., 23., 22.],
                [17., 16., 15., 16., 17., 18., 19., 18., 17.]
            ]
        );
    }

    #[test]
    fn reflection_pad_3d() {
        let arr = ndarray::Array::range(0., 125., 1.)
            .into_shape((5, 5, 5))
            .unwrap();
        let padded = super::reflection_pad(&arr, [1, 2, 3]);
        assert_eq!(
            padded,
            ndarray::array![
                [
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                    [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                    [28., 27., 26., 25., 26., 27., 28., 29., 28., 27., 26.],
                    [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                    [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                    [48., 47., 46., 45., 46., 47., 48., 49., 48., 47., 46.],
                    [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.]
                ],
                [
                    [13., 12., 11., 10., 11., 12., 13., 14., 13., 12., 11.],
                    [8., 7., 6., 5., 6., 7., 8., 9., 8., 7., 6.],
                    [3., 2., 1., 0., 1., 2., 3., 4., 3., 2., 1.],
                    [8., 7., 6., 5., 6., 7., 8., 9., 8., 7., 6.],
                    [13., 12., 11., 10., 11., 12., 13., 14., 13., 12., 11.],
                    [18., 17., 16., 15., 16., 17., 18., 19., 18., 17., 16.],
                    [23., 22., 21., 20., 21., 22., 23., 24., 23., 22., 21.],
                    [18., 17., 16., 15., 16., 17., 18., 19., 18., 17., 16.],
                    [13., 12., 11., 10., 11., 12., 13., 14., 13., 12., 11.]
                ],
                [
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                    [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                    [28., 27., 26., 25., 26., 27., 28., 29., 28., 27., 26.],
                    [33., 32., 31., 30., 31., 32., 33., 34., 33., 32., 31.],
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.],
                    [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                    [48., 47., 46., 45., 46., 47., 48., 49., 48., 47., 46.],
                    [43., 42., 41., 40., 41., 42., 43., 44., 43., 42., 41.],
                    [38., 37., 36., 35., 36., 37., 38., 39., 38., 37., 36.]
                ],
                [
                    [63., 62., 61., 60., 61., 62., 63., 64., 63., 62., 61.],
                    [58., 57., 56., 55., 56., 57., 58., 59., 58., 57., 56.],
                    [53., 52., 51., 50., 51., 52., 53., 54., 53., 52., 51.],
                    [58., 57., 56., 55., 56., 57., 58., 59., 58., 57., 56.],
                    [63., 62., 61., 60., 61., 62., 63., 64., 63., 62., 61.],
                    [68., 67., 66., 65., 66., 67., 68., 69., 68., 67., 66.],
                    [73., 72., 71., 70., 71., 72., 73., 74., 73., 72., 71.],
                    [68., 67., 66., 65., 66., 67., 68., 69., 68., 67., 66.],
                    [63., 62., 61., 60., 61., 62., 63., 64., 63., 62., 61.]
                ],
                [
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                    [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                    [78., 77., 76., 75., 76., 77., 78., 79., 78., 77., 76.],
                    [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                    [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                    [98., 97., 96., 95., 96., 97., 98., 99., 98., 97., 96.],
                    [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.]
                ],
                [
                    [113., 112., 111., 110., 111., 112., 113., 114., 113., 112., 111.],
                    [108., 107., 106., 105., 106., 107., 108., 109., 108., 107., 106.],
                    [103., 102., 101., 100., 101., 102., 103., 104., 103., 102., 101.],
                    [108., 107., 106., 105., 106., 107., 108., 109., 108., 107., 106.],
                    [113., 112., 111., 110., 111., 112., 113., 114., 113., 112., 111.],
                    [118., 117., 116., 115., 116., 117., 118., 119., 118., 117., 116.],
                    [123., 122., 121., 120., 121., 122., 123., 124., 123., 122., 121.],
                    [118., 117., 116., 115., 116., 117., 118., 119., 118., 117., 116.],
                    [113., 112., 111., 110., 111., 112., 113., 114., 113., 112., 111.]
                ],
                [
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                    [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                    [78., 77., 76., 75., 76., 77., 78., 79., 78., 77., 76.],
                    [83., 82., 81., 80., 81., 82., 83., 84., 83., 82., 81.],
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.],
                    [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                    [98., 97., 96., 95., 96., 97., 98., 99., 98., 97., 96.],
                    [93., 92., 91., 90., 91., 92., 93., 94., 93., 92., 91.],
                    [88., 87., 86., 85., 86., 87., 88., 89., 88., 87., 86.]
                ]
            ]
        )
    }

    #[test]
    fn apply_remove_padding_test() {
        use super::*;

        let to_pad = (0..625)
            .map(|el| el as f32)
            .collect::<Array<f32, _>>()
            .into_shape((5, 5, 5, 5))
            .unwrap();
        let padding = &[4, 2];

        let padded_shape = (5, 5, 13, 9);
        let padded = pad(&to_pad, padding, &Reflective);

        let real_padded_elems = vec![
            22., 21., 20., 21., 22., 23., 24., 23., 22., 17., 16., 15., 16., 17., 18., 19., 18.,
            17., 12., 11., 10., 11., 12., 13., 14., 13., 12., 7., 6., 5., 6., 7., 8., 9., 8., 7.,
            2., 1., 0., 1., 2., 3., 4., 3., 2., 7., 6., 5., 6., 7., 8., 9., 8., 7., 12., 11., 10.,
            11., 12., 13., 14., 13., 12., 17., 16., 15., 16., 17., 18., 19., 18., 17., 22., 21.,
            20., 21., 22., 23., 24., 23., 22., 17., 16., 15., 16., 17., 18., 19., 18., 17., 12.,
            11., 10., 11., 12., 13., 14., 13., 12., 7., 6., 5., 6., 7., 8., 9., 8., 7., 2., 1., 0.,
            1., 2., 3., 4., 3., 2., 47., 46., 45., 46., 47., 48., 49., 48., 47., 42., 41., 40.,
            41., 42., 43., 44., 43., 42., 37., 36., 35., 36., 37., 38., 39., 38., 37., 32., 31.,
            30., 31., 32., 33., 34., 33., 32., 27., 26., 25., 26., 27., 28., 29., 28., 27., 32.,
            31., 30., 31., 32., 33., 34., 33., 32., 37., 36., 35., 36., 37., 38., 39., 38., 37.,
            42., 41., 40., 41., 42., 43., 44., 43., 42., 47., 46., 45., 46., 47., 48., 49., 48.,
            47., 42., 41., 40., 41., 42., 43., 44., 43., 42., 37., 36., 35., 36., 37., 38., 39.,
            38., 37., 32., 31., 30., 31., 32., 33., 34., 33., 32., 27., 26., 25., 26., 27., 28.,
            29., 28., 27., 72., 71., 70., 71., 72., 73., 74., 73., 72., 67., 66., 65., 66., 67.,
            68., 69., 68., 67., 62., 61., 60., 61., 62., 63., 64., 63., 62., 57., 56., 55., 56.,
            57., 58., 59., 58., 57., 52., 51., 50., 51., 52., 53., 54., 53., 52., 57., 56., 55.,
            56., 57., 58., 59., 58., 57., 62., 61., 60., 61., 62., 63., 64., 63., 62., 67., 66.,
            65., 66., 67., 68., 69., 68., 67., 72., 71., 70., 71., 72., 73., 74., 73., 72., 67.,
            66., 65., 66., 67., 68., 69., 68., 67., 62., 61., 60., 61., 62., 63., 64., 63., 62.,
            57., 56., 55., 56., 57., 58., 59., 58., 57., 52., 51., 50., 51., 52., 53., 54., 53.,
            52., 97., 96., 95., 96., 97., 98., 99., 98., 97., 92., 91., 90., 91., 92., 93., 94.,
            93., 92., 87., 86., 85., 86., 87., 88., 89., 88., 87., 82., 81., 80., 81., 82., 83.,
            84., 83., 82., 77., 76., 75., 76., 77., 78., 79., 78., 77., 82., 81., 80., 81., 82.,
            83., 84., 83., 82., 87., 86., 85., 86., 87., 88., 89., 88., 87., 92., 91., 90., 91.,
            92., 93., 94., 93., 92., 97., 96., 95., 96., 97., 98., 99., 98., 97., 92., 91., 90.,
            91., 92., 93., 94., 93., 92., 87., 86., 85., 86., 87., 88., 89., 88., 87., 82., 81.,
            80., 81., 82., 83., 84., 83., 82., 77., 76., 75., 76., 77., 78., 79., 78., 77., 122.,
            121., 120., 121., 122., 123., 124., 123., 122., 117., 116., 115., 116., 117., 118.,
            119., 118., 117., 112., 111., 110., 111., 112., 113., 114., 113., 112., 107., 106.,
            105., 106., 107., 108., 109., 108., 107., 102., 101., 100., 101., 102., 103., 104.,
            103., 102., 107., 106., 105., 106., 107., 108., 109., 108., 107., 112., 111., 110.,
            111., 112., 113., 114., 113., 112., 117., 116., 115., 116., 117., 118., 119., 118.,
            117., 122., 121., 120., 121., 122., 123., 124., 123., 122., 117., 116., 115., 116.,
            117., 118., 119., 118., 117., 112., 111., 110., 111., 112., 113., 114., 113., 112.,
            107., 106., 105., 106., 107., 108., 109., 108., 107., 102., 101., 100., 101., 102.,
            103., 104., 103., 102., 147., 146., 145., 146., 147., 148., 149., 148., 147., 142.,
            141., 140., 141., 142., 143., 144., 143., 142., 137., 136., 135., 136., 137., 138.,
            139., 138., 137., 132., 131., 130., 131., 132., 133., 134., 133., 132., 127., 126.,
            125., 126., 127., 128., 129., 128., 127., 132., 131., 130., 131., 132., 133., 134.,
            133., 132., 137., 136., 135., 136., 137., 138., 139., 138., 137., 142., 141., 140.,
            141., 142., 143., 144., 143., 142., 147., 146., 145., 146., 147., 148., 149., 148.,
            147., 142., 141., 140., 141., 142., 143., 144., 143., 142., 137., 136., 135., 136.,
            137., 138., 139., 138., 137., 132., 131., 130., 131., 132., 133., 134., 133., 132.,
            127., 126., 125., 126., 127., 128., 129., 128., 127., 172., 171., 170., 171., 172.,
            173., 174., 173., 172., 167., 166., 165., 166., 167., 168., 169., 168., 167., 162.,
            161., 160., 161., 162., 163., 164., 163., 162., 157., 156., 155., 156., 157., 158.,
            159., 158., 157., 152., 151., 150., 151., 152., 153., 154., 153., 152., 157., 156.,
            155., 156., 157., 158., 159., 158., 157., 162., 161., 160., 161., 162., 163., 164.,
            163., 162., 167., 166., 165., 166., 167., 168., 169., 168., 167., 172., 171., 170.,
            171., 172., 173., 174., 173., 172., 167., 166., 165., 166., 167., 168., 169., 168.,
            167., 162., 161., 160., 161., 162., 163., 164., 163., 162., 157., 156., 155., 156.,
            157., 158., 159., 158., 157., 152., 151., 150., 151., 152., 153., 154., 153., 152.,
            197., 196., 195., 196., 197., 198., 199., 198., 197., 192., 191., 190., 191., 192.,
            193., 194., 193., 192., 187., 186., 185., 186., 187., 188., 189., 188., 187., 182.,
            181., 180., 181., 182., 183., 184., 183., 182., 177., 176., 175., 176., 177., 178.,
            179., 178., 177., 182., 181., 180., 181., 182., 183., 184., 183., 182., 187., 186.,
            185., 186., 187., 188., 189., 188., 187., 192., 191., 190., 191., 192., 193., 194.,
            193., 192., 197., 196., 195., 196., 197., 198., 199., 198., 197., 192., 191., 190.,
            191., 192., 193., 194., 193., 192., 187., 186., 185., 186., 187., 188., 189., 188.,
            187., 182., 181., 180., 181., 182., 183., 184., 183., 182., 177., 176., 175., 176.,
            177., 178., 179., 178., 177., 222., 221., 220., 221., 222., 223., 224., 223., 222.,
            217., 216., 215., 216., 217., 218., 219., 218., 217., 212., 211., 210., 211., 212.,
            213., 214., 213., 212., 207., 206., 205., 206., 207., 208., 209., 208., 207., 202.,
            201., 200., 201., 202., 203., 204., 203., 202., 207., 206., 205., 206., 207., 208.,
            209., 208., 207., 212., 211., 210., 211., 212., 213., 214., 213., 212., 217., 216.,
            215., 216., 217., 218., 219., 218., 217., 222., 221., 220., 221., 222., 223., 224.,
            223., 222., 217., 216., 215., 216., 217., 218., 219., 218., 217., 212., 211., 210.,
            211., 212., 213., 214., 213., 212., 207., 206., 205., 206., 207., 208., 209., 208.,
            207., 202., 201., 200., 201., 202., 203., 204., 203., 202., 247., 246., 245., 246.,
            247., 248., 249., 248., 247., 242., 241., 240., 241., 242., 243., 244., 243., 242.,
            237., 236., 235., 236., 237., 238., 239., 238., 237., 232., 231., 230., 231., 232.,
            233., 234., 233., 232., 227., 226., 225., 226., 227., 228., 229., 228., 227., 232.,
            231., 230., 231., 232., 233., 234., 233., 232., 237., 236., 235., 236., 237., 238.,
            239., 238., 237., 242., 241., 240., 241., 242., 243., 244., 243., 242., 247., 246.,
            245., 246., 247., 248., 249., 248., 247., 242., 241., 240., 241., 242., 243., 244.,
            243., 242., 237., 236., 235., 236., 237., 238., 239., 238., 237., 232., 231., 230.,
            231., 232., 233., 234., 233., 232., 227., 226., 225., 226., 227., 228., 229., 228.,
            227., 272., 271., 270., 271., 272., 273., 274., 273., 272., 267., 266., 265., 266.,
            267., 268., 269., 268., 267., 262., 261., 260., 261., 262., 263., 264., 263., 262.,
            257., 256., 255., 256., 257., 258., 259., 258., 257., 252., 251., 250., 251., 252.,
            253., 254., 253., 252., 257., 256., 255., 256., 257., 258., 259., 258., 257., 262.,
            261., 260., 261., 262., 263., 264., 263., 262., 267., 266., 265., 266., 267., 268.,
            269., 268., 267., 272., 271., 270., 271., 272., 273., 274., 273., 272., 267., 266.,
            265., 266., 267., 268., 269., 268., 267., 262., 261., 260., 261., 262., 263., 264.,
            263., 262., 257., 256., 255., 256., 257., 258., 259., 258., 257., 252., 251., 250.,
            251., 252., 253., 254., 253., 252., 297., 296., 295., 296., 297., 298., 299., 298.,
            297., 292., 291., 290., 291., 292., 293., 294., 293., 292., 287., 286., 285., 286.,
            287., 288., 289., 288., 287., 282., 281., 280., 281., 282., 283., 284., 283., 282.,
            277., 276., 275., 276., 277., 278., 279., 278., 277., 282., 281., 280., 281., 282.,
            283., 284., 283., 282., 287., 286., 285., 286., 287., 288., 289., 288., 287., 292.,
            291., 290., 291., 292., 293., 294., 293., 292., 297., 296., 295., 296., 297., 298.,
            299., 298., 297., 292., 291., 290., 291., 292., 293., 294., 293., 292., 287., 286.,
            285., 286., 287., 288., 289., 288., 287., 282., 281., 280., 281., 282., 283., 284.,
            283., 282., 277., 276., 275., 276., 277., 278., 279., 278., 277., 322., 321., 320.,
            321., 322., 323., 324., 323., 322., 317., 316., 315., 316., 317., 318., 319., 318.,
            317., 312., 311., 310., 311., 312., 313., 314., 313., 312., 307., 306., 305., 306.,
            307., 308., 309., 308., 307., 302., 301., 300., 301., 302., 303., 304., 303., 302.,
            307., 306., 305., 306., 307., 308., 309., 308., 307., 312., 311., 310., 311., 312.,
            313., 314., 313., 312., 317., 316., 315., 316., 317., 318., 319., 318., 317., 322.,
            321., 320., 321., 322., 323., 324., 323., 322., 317., 316., 315., 316., 317., 318.,
            319., 318., 317., 312., 311., 310., 311., 312., 313., 314., 313., 312., 307., 306.,
            305., 306., 307., 308., 309., 308., 307., 302., 301., 300., 301., 302., 303., 304.,
            303., 302., 347., 346., 345., 346., 347., 348., 349., 348., 347., 342., 341., 340.,
            341., 342., 343., 344., 343., 342., 337., 336., 335., 336., 337., 338., 339., 338.,
            337., 332., 331., 330., 331., 332., 333., 334., 333., 332., 327., 326., 325., 326.,
            327., 328., 329., 328., 327., 332., 331., 330., 331., 332., 333., 334., 333., 332.,
            337., 336., 335., 336., 337., 338., 339., 338., 337., 342., 341., 340., 341., 342.,
            343., 344., 343., 342., 347., 346., 345., 346., 347., 348., 349., 348., 347., 342.,
            341., 340., 341., 342., 343., 344., 343., 342., 337., 336., 335., 336., 337., 338.,
            339., 338., 337., 332., 331., 330., 331., 332., 333., 334., 333., 332., 327., 326.,
            325., 326., 327., 328., 329., 328., 327., 372., 371., 370., 371., 372., 373., 374.,
            373., 372., 367., 366., 365., 366., 367., 368., 369., 368., 367., 362., 361., 360.,
            361., 362., 363., 364., 363., 362., 357., 356., 355., 356., 357., 358., 359., 358.,
            357., 352., 351., 350., 351., 352., 353., 354., 353., 352., 357., 356., 355., 356.,
            357., 358., 359., 358., 357., 362., 361., 360., 361., 362., 363., 364., 363., 362.,
            367., 366., 365., 366., 367., 368., 369., 368., 367., 372., 371., 370., 371., 372.,
            373., 374., 373., 372., 367., 366., 365., 366., 367., 368., 369., 368., 367., 362.,
            361., 360., 361., 362., 363., 364., 363., 362., 357., 356., 355., 356., 357., 358.,
            359., 358., 357., 352., 351., 350., 351., 352., 353., 354., 353., 352., 397., 396.,
            395., 396., 397., 398., 399., 398., 397., 392., 391., 390., 391., 392., 393., 394.,
            393., 392., 387., 386., 385., 386., 387., 388., 389., 388., 387., 382., 381., 380.,
            381., 382., 383., 384., 383., 382., 377., 376., 375., 376., 377., 378., 379., 378.,
            377., 382., 381., 380., 381., 382., 383., 384., 383., 382., 387., 386., 385., 386.,
            387., 388., 389., 388., 387., 392., 391., 390., 391., 392., 393., 394., 393., 392.,
            397., 396., 395., 396., 397., 398., 399., 398., 397., 392., 391., 390., 391., 392.,
            393., 394., 393., 392., 387., 386., 385., 386., 387., 388., 389., 388., 387., 382.,
            381., 380., 381., 382., 383., 384., 383., 382., 377., 376., 375., 376., 377., 378.,
            379., 378., 377., 422., 421., 420., 421., 422., 423., 424., 423., 422., 417., 416.,
            415., 416., 417., 418., 419., 418., 417., 412., 411., 410., 411., 412., 413., 414.,
            413., 412., 407., 406., 405., 406., 407., 408., 409., 408., 407., 402., 401., 400.,
            401., 402., 403., 404., 403., 402., 407., 406., 405., 406., 407., 408., 409., 408.,
            407., 412., 411., 410., 411., 412., 413., 414., 413., 412., 417., 416., 415., 416.,
            417., 418., 419., 418., 417., 422., 421., 420., 421., 422., 423., 424., 423., 422.,
            417., 416., 415., 416., 417., 418., 419., 418., 417., 412., 411., 410., 411., 412.,
            413., 414., 413., 412., 407., 406., 405., 406., 407., 408., 409., 408., 407., 402.,
            401., 400., 401., 402., 403., 404., 403., 402., 447., 446., 445., 446., 447., 448.,
            449., 448., 447., 442., 441., 440., 441., 442., 443., 444., 443., 442., 437., 436.,
            435., 436., 437., 438., 439., 438., 437., 432., 431., 430., 431., 432., 433., 434.,
            433., 432., 427., 426., 425., 426., 427., 428., 429., 428., 427., 432., 431., 430.,
            431., 432., 433., 434., 433., 432., 437., 436., 435., 436., 437., 438., 439., 438.,
            437., 442., 441., 440., 441., 442., 443., 444., 443., 442., 447., 446., 445., 446.,
            447., 448., 449., 448., 447., 442., 441., 440., 441., 442., 443., 444., 443., 442.,
            437., 436., 435., 436., 437., 438., 439., 438., 437., 432., 431., 430., 431., 432.,
            433., 434., 433., 432., 427., 426., 425., 426., 427., 428., 429., 428., 427., 472.,
            471., 470., 471., 472., 473., 474., 473., 472., 467., 466., 465., 466., 467., 468.,
            469., 468., 467., 462., 461., 460., 461., 462., 463., 464., 463., 462., 457., 456.,
            455., 456., 457., 458., 459., 458., 457., 452., 451., 450., 451., 452., 453., 454.,
            453., 452., 457., 456., 455., 456., 457., 458., 459., 458., 457., 462., 461., 460.,
            461., 462., 463., 464., 463., 462., 467., 466., 465., 466., 467., 468., 469., 468.,
            467., 472., 471., 470., 471., 472., 473., 474., 473., 472., 467., 466., 465., 466.,
            467., 468., 469., 468., 467., 462., 461., 460., 461., 462., 463., 464., 463., 462.,
            457., 456., 455., 456., 457., 458., 459., 458., 457., 452., 451., 450., 451., 452.,
            453., 454., 453., 452., 497., 496., 495., 496., 497., 498., 499., 498., 497., 492.,
            491., 490., 491., 492., 493., 494., 493., 492., 487., 486., 485., 486., 487., 488.,
            489., 488., 487., 482., 481., 480., 481., 482., 483., 484., 483., 482., 477., 476.,
            475., 476., 477., 478., 479., 478., 477., 482., 481., 480., 481., 482., 483., 484.,
            483., 482., 487., 486., 485., 486., 487., 488., 489., 488., 487., 492., 491., 490.,
            491., 492., 493., 494., 493., 492., 497., 496., 495., 496., 497., 498., 499., 498.,
            497., 492., 491., 490., 491., 492., 493., 494., 493., 492., 487., 486., 485., 486.,
            487., 488., 489., 488., 487., 482., 481., 480., 481., 482., 483., 484., 483., 482.,
            477., 476., 475., 476., 477., 478., 479., 478., 477., 522., 521., 520., 521., 522.,
            523., 524., 523., 522., 517., 516., 515., 516., 517., 518., 519., 518., 517., 512.,
            511., 510., 511., 512., 513., 514., 513., 512., 507., 506., 505., 506., 507., 508.,
            509., 508., 507., 502., 501., 500., 501., 502., 503., 504., 503., 502., 507., 506.,
            505., 506., 507., 508., 509., 508., 507., 512., 511., 510., 511., 512., 513., 514.,
            513., 512., 517., 516., 515., 516., 517., 518., 519., 518., 517., 522., 521., 520.,
            521., 522., 523., 524., 523., 522., 517., 516., 515., 516., 517., 518., 519., 518.,
            517., 512., 511., 510., 511., 512., 513., 514., 513., 512., 507., 506., 505., 506.,
            507., 508., 509., 508., 507., 502., 501., 500., 501., 502., 503., 504., 503., 502.,
            547., 546., 545., 546., 547., 548., 549., 548., 547., 542., 541., 540., 541., 542.,
            543., 544., 543., 542., 537., 536., 535., 536., 537., 538., 539., 538., 537., 532.,
            531., 530., 531., 532., 533., 534., 533., 532., 527., 526., 525., 526., 527., 528.,
            529., 528., 527., 532., 531., 530., 531., 532., 533., 534., 533., 532., 537., 536.,
            535., 536., 537., 538., 539., 538., 537., 542., 541., 540., 541., 542., 543., 544.,
            543., 542., 547., 546., 545., 546., 547., 548., 549., 548., 547., 542., 541., 540.,
            541., 542., 543., 544., 543., 542., 537., 536., 535., 536., 537., 538., 539., 538.,
            537., 532., 531., 530., 531., 532., 533., 534., 533., 532., 527., 526., 525., 526.,
            527., 528., 529., 528., 527., 572., 571., 570., 571., 572., 573., 574., 573., 572.,
            567., 566., 565., 566., 567., 568., 569., 568., 567., 562., 561., 560., 561., 562.,
            563., 564., 563., 562., 557., 556., 555., 556., 557., 558., 559., 558., 557., 552.,
            551., 550., 551., 552., 553., 554., 553., 552., 557., 556., 555., 556., 557., 558.,
            559., 558., 557., 562., 561., 560., 561., 562., 563., 564., 563., 562., 567., 566.,
            565., 566., 567., 568., 569., 568., 567., 572., 571., 570., 571., 572., 573., 574.,
            573., 572., 567., 566., 565., 566., 567., 568., 569., 568., 567., 562., 561., 560.,
            561., 562., 563., 564., 563., 562., 557., 556., 555., 556., 557., 558., 559., 558.,
            557., 552., 551., 550., 551., 552., 553., 554., 553., 552., 597., 596., 595., 596.,
            597., 598., 599., 598., 597., 592., 591., 590., 591., 592., 593., 594., 593., 592.,
            587., 586., 585., 586., 587., 588., 589., 588., 587., 582., 581., 580., 581., 582.,
            583., 584., 583., 582., 577., 576., 575., 576., 577., 578., 579., 578., 577., 582.,
            581., 580., 581., 582., 583., 584., 583., 582., 587., 586., 585., 586., 587., 588.,
            589., 588., 587., 592., 591., 590., 591., 592., 593., 594., 593., 592., 597., 596.,
            595., 596., 597., 598., 599., 598., 597., 592., 591., 590., 591., 592., 593., 594.,
            593., 592., 587., 586., 585., 586., 587., 588., 589., 588., 587., 582., 581., 580.,
            581., 582., 583., 584., 583., 582., 577., 576., 575., 576., 577., 578., 579., 578.,
            577., 622., 621., 620., 621., 622., 623., 624., 623., 622., 617., 616., 615., 616.,
            617., 618., 619., 618., 617., 612., 611., 610., 611., 612., 613., 614., 613., 612.,
            607., 606., 605., 606., 607., 608., 609., 608., 607., 602., 601., 600., 601., 602.,
            603., 604., 603., 602., 607., 606., 605., 606., 607., 608., 609., 608., 607., 612.,
            611., 610., 611., 612., 613., 614., 613., 612., 617., 616., 615., 616., 617., 618.,
            619., 618., 617., 622., 621., 620., 621., 622., 623., 624., 623., 622., 617., 616.,
            615., 616., 617., 618., 619., 618., 617., 612., 611., 610., 611., 612., 613., 614.,
            613., 612., 607., 606., 605., 606., 607., 608., 609., 608., 607., 602., 601., 600.,
            601., 602., 603., 604., 603., 602.,
        ];

        assert_eq!(
            padded,
            Array::from_shape_vec(padded_shape, real_padded_elems).unwrap()
        );
        assert_eq!(unpad(&padded, padding), to_pad);
    }

    #[test]
    fn conv1d() {
        use ndarray::prelude::*;

        use super::*;
        use ndarray::Ix3;

        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((5, 3, 10)).unwrap();
        let kernel = Array::<f32, _>::ones((6, 3, 5));
        let stride = &[1];
        let padding = &[0];
        let dilation = &[1];

        let conv_out_shape =
            conv_out_shape::<Ix3>(input.shape(), kernel.shape(), padding, stride, dilation);

        let true_output_elems = vec![
            180., 195., 210., 225., 240., 255., 180., 195., 210., 225., 240., 255., 180., 195.,
            210., 225., 240., 255., 180., 195., 210., 225., 240., 255., 180., 195., 210., 225.,
            240., 255., 180., 195., 210., 225., 240., 255., 630., 645., 660., 675., 690., 705.,
            630., 645., 660., 675., 690., 705., 630., 645., 660., 675., 690., 705., 630., 645.,
            660., 675., 690., 705., 630., 645., 660., 675., 690., 705., 630., 645., 660., 675.,
            690., 705., 1080., 1095., 1110., 1125., 1140., 1155., 1080., 1095., 1110., 1125.,
            1140., 1155., 1080., 1095., 1110., 1125., 1140., 1155., 1080., 1095., 1110., 1125.,
            1140., 1155., 1080., 1095., 1110., 1125., 1140., 1155., 1080., 1095., 1110., 1125.,
            1140., 1155., 1530., 1545., 1560., 1575., 1590., 1605., 1530., 1545., 1560., 1575.,
            1590., 1605., 1530., 1545., 1560., 1575., 1590., 1605., 1530., 1545., 1560., 1575.,
            1590., 1605., 1530., 1545., 1560., 1575., 1590., 1605., 1530., 1545., 1560., 1575.,
            1590., 1605., 1980., 1995., 2010., 2025., 2040., 2055., 1980., 1995., 2010., 2025.,
            2040., 2055., 1980., 1995., 2010., 2025., 2040., 2055., 1980., 1995., 2010., 2025.,
            2040., 2055., 1980., 1995., 2010., 2025., 2040., 2055., 1980., 1995., 2010., 2025.,
            2040., 2055.,
        ];

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_input_grad_elems = vec![
            6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18.,
            12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30.,
            24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24.,
            30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12.,
            18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6.,
            6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18.,
            12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24., 30., 30.,
            24., 18., 12., 6., 6., 12., 18., 24., 30., 30., 24., 18., 12., 6., 6., 12., 18., 24.,
            30., 30., 24., 18., 12., 6.,
        ];

        let true_kernel_grad_elems = array![
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
            [
                [1875., 1905., 1935., 1965., 1995.],
                [2175., 2205., 2235., 2265., 2295.],
                [2475., 2505., 2535., 2565., 2595.],
            ],
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(kernel_grad, true_kernel_grad_elems);
    }

    #[test]
    fn conv2d() {
        use super::*;
        use ndarray::Ix4;

        // This is an input with a batch size of 3, 2 input channels each of 5 by 5.
        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((3, 2, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((3, 2, 2, 2));

        let stride = &[1, 1];
        let padding = &[0, 0];
        let dilation = &[1, 1];

        let conv_out_shape =
            conv_out_shape::<Ix4>(input.shape(), kernel.shape(), padding, stride, dilation);

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);
        let true_output_elems: Vec<f32> = vec![
            124., 132., 140., 148., 164., 172., 180., 188., 204., 212., 220., 228., 244., 252.,
            260., 268., 124., 132., 140., 148., 164., 172., 180., 188., 204., 212., 220., 228.,
            244., 252., 260., 268., 124., 132., 140., 148., 164., 172., 180., 188., 204., 212.,
            220., 228., 244., 252., 260., 268., 524., 532., 540., 548., 564., 572., 580., 588.,
            604., 612., 620., 628., 644., 652., 660., 668., 524., 532., 540., 548., 564., 572.,
            580., 588., 604., 612., 620., 628., 644., 652., 660., 668., 524., 532., 540., 548.,
            564., 572., 580., 588., 604., 612., 620., 628., 644., 652., 660., 668., 924., 932.,
            940., 948., 964., 972., 980., 988., 1004., 1012., 1020., 1028., 1044., 1052., 1060.,
            1068., 924., 932., 940., 948., 964., 972., 980., 988., 1004., 1012., 1020., 1028.,
            1044., 1052., 1060., 1068., 924., 932., 940., 948., 964., 972., 980., 988., 1004.,
            1012., 1020., 1028., 1044., 1052., 1060., 1068.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros((3, 2, 5, 5));
        let mut kernel_grad = Array::<f32, _>::zeros((3, 2, 2, 2));
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_input_grad_elems: Vec<f32> = vec![
            3., 6., 6., 6., 3., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 6., 12., 12., 12.,
            6., 3., 6., 6., 6., 3., 3., 6., 6., 6., 3., 6., 12., 12., 12., 6., 6., 12., 12., 12.,
            6., 6., 12., 12., 12., 6., 3., 6., 6., 6., 3., 3., 6., 6., 6., 3., 6., 12., 12., 12.,
            6., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 3., 6., 6., 6., 3., 3., 6., 6., 6.,
            3., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 3., 6., 6.,
            6., 3., 3., 6., 6., 6., 3., 6., 12., 12., 12., 6., 6., 12., 12., 12., 6., 6., 12., 12.,
            12., 6., 3., 6., 6., 6., 3., 3., 6., 6., 6., 3., 6., 12., 12., 12., 6., 6., 12., 12.,
            12., 6., 6., 12., 12., 12., 6., 3., 6., 6., 6., 3.,
        ];
        let true_kernel_grad_elems: Vec<f32> = vec![
            2832., 2880., 3072., 3120., 4032., 4080., 4272., 4320., 2832., 2880., 3072., 3120.,
            4032., 4080., 4272., 4320., 2832., 2880., 3072., 3120., 4032., 4080., 4272., 4320.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }

    #[test]
    fn conv3d() {
        use super::*;

        let input_elems = Array::<f32, _>::from_iter((0..750).map(|el| el as f32));
        let input = input_elems.into_shape((2, 3, 5, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((4, 3, 2, 2, 2));

        let stride = &[1, 1, 1];
        let padding = &[0, 0, 0];
        let dilation = &[1, 1, 1];

        let conv_out_shape = conv_out_shape::<ndarray::Ix5>(
            input.shape(),
            kernel.shape(),
            padding,
            stride,
            dilation,
        );

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        let true_output_elems = vec![
            3372., 3396., 3420., 3444., 3492., 3516., 3540., 3564., 3612., 3636., 3660., 3684.,
            3732., 3756., 3780., 3804., 3972., 3996., 4020., 4044., 4092., 4116., 4140., 4164.,
            4212., 4236., 4260., 4284., 4332., 4356., 4380., 4404., 4572., 4596., 4620., 4644.,
            4692., 4716., 4740., 4764., 4812., 4836., 4860., 4884., 4932., 4956., 4980., 5004.,
            5172., 5196., 5220., 5244., 5292., 5316., 5340., 5364., 5412., 5436., 5460., 5484.,
            5532., 5556., 5580., 5604., 3372., 3396., 3420., 3444., 3492., 3516., 3540., 3564.,
            3612., 3636., 3660., 3684., 3732., 3756., 3780., 3804., 3972., 3996., 4020., 4044.,
            4092., 4116., 4140., 4164., 4212., 4236., 4260., 4284., 4332., 4356., 4380., 4404.,
            4572., 4596., 4620., 4644., 4692., 4716., 4740., 4764., 4812., 4836., 4860., 4884.,
            4932., 4956., 4980., 5004., 5172., 5196., 5220., 5244., 5292., 5316., 5340., 5364.,
            5412., 5436., 5460., 5484., 5532., 5556., 5580., 5604., 3372., 3396., 3420., 3444.,
            3492., 3516., 3540., 3564., 3612., 3636., 3660., 3684., 3732., 3756., 3780., 3804.,
            3972., 3996., 4020., 4044., 4092., 4116., 4140., 4164., 4212., 4236., 4260., 4284.,
            4332., 4356., 4380., 4404., 4572., 4596., 4620., 4644., 4692., 4716., 4740., 4764.,
            4812., 4836., 4860., 4884., 4932., 4956., 4980., 5004., 5172., 5196., 5220., 5244.,
            5292., 5316., 5340., 5364., 5412., 5436., 5460., 5484., 5532., 5556., 5580., 5604.,
            3372., 3396., 3420., 3444., 3492., 3516., 3540., 3564., 3612., 3636., 3660., 3684.,
            3732., 3756., 3780., 3804., 3972., 3996., 4020., 4044., 4092., 4116., 4140., 4164.,
            4212., 4236., 4260., 4284., 4332., 4356., 4380., 4404., 4572., 4596., 4620., 4644.,
            4692., 4716., 4740., 4764., 4812., 4836., 4860., 4884., 4932., 4956., 4980., 5004.,
            5172., 5196., 5220., 5244., 5292., 5316., 5340., 5364., 5412., 5436., 5460., 5484.,
            5532., 5556., 5580., 5604., 12372., 12396., 12420., 12444., 12492., 12516., 12540.,
            12564., 12612., 12636., 12660., 12684., 12732., 12756., 12780., 12804., 12972., 12996.,
            13020., 13044., 13092., 13116., 13140., 13164., 13212., 13236., 13260., 13284., 13332.,
            13356., 13380., 13404., 13572., 13596., 13620., 13644., 13692., 13716., 13740., 13764.,
            13812., 13836., 13860., 13884., 13932., 13956., 13980., 14004., 14172., 14196., 14220.,
            14244., 14292., 14316., 14340., 14364., 14412., 14436., 14460., 14484., 14532., 14556.,
            14580., 14604., 12372., 12396., 12420., 12444., 12492., 12516., 12540., 12564., 12612.,
            12636., 12660., 12684., 12732., 12756., 12780., 12804., 12972., 12996., 13020., 13044.,
            13092., 13116., 13140., 13164., 13212., 13236., 13260., 13284., 13332., 13356., 13380.,
            13404., 13572., 13596., 13620., 13644., 13692., 13716., 13740., 13764., 13812., 13836.,
            13860., 13884., 13932., 13956., 13980., 14004., 14172., 14196., 14220., 14244., 14292.,
            14316., 14340., 14364., 14412., 14436., 14460., 14484., 14532., 14556., 14580., 14604.,
            12372., 12396., 12420., 12444., 12492., 12516., 12540., 12564., 12612., 12636., 12660.,
            12684., 12732., 12756., 12780., 12804., 12972., 12996., 13020., 13044., 13092., 13116.,
            13140., 13164., 13212., 13236., 13260., 13284., 13332., 13356., 13380., 13404., 13572.,
            13596., 13620., 13644., 13692., 13716., 13740., 13764., 13812., 13836., 13860., 13884.,
            13932., 13956., 13980., 14004., 14172., 14196., 14220., 14244., 14292., 14316., 14340.,
            14364., 14412., 14436., 14460., 14484., 14532., 14556., 14580., 14604., 12372., 12396.,
            12420., 12444., 12492., 12516., 12540., 12564., 12612., 12636., 12660., 12684., 12732.,
            12756., 12780., 12804., 12972., 12996., 13020., 13044., 13092., 13116., 13140., 13164.,
            13212., 13236., 13260., 13284., 13332., 13356., 13380., 13404., 13572., 13596., 13620.,
            13644., 13692., 13716., 13740., 13764., 13812., 13836., 13860., 13884., 13932., 13956.,
            13980., 14004., 14172., 14196., 14220., 14244., 14292., 14316., 14340., 14364., 14412.,
            14436., 14460., 14484., 14532., 14556., 14580., 14604.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_input_grad_elems = vec![
            4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16., 16., 32., 32.,
            32., 16., 16., 32., 32., 32., 16., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 16.,
            32., 32., 32., 16., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 16.,
            32., 32., 32., 16., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8.,
            8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4.,
            8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8.,
            4., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 16., 32.,
            32., 32., 16., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16.,
            16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 8.,
            16., 16., 16., 8., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8.,
            8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8.,
            8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 8., 16., 16., 16.,
            8., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 8., 16.,
            16., 16., 8., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16.,
            16., 32., 32., 32., 16., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 16., 32., 32.,
            32., 16., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 8., 16., 16., 16., 8., 4.,
            8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8.,
            4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8.,
            8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 16., 32., 32., 32.,
            16., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 8., 16., 16., 16., 8., 8., 16.,
            16., 16., 8., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 16., 32., 32., 32.,
            16., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16., 16., 32.,
            32., 32., 16., 16., 32., 32., 32., 16., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 8.,
            16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4.,
            4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16., 16., 32., 32.,
            32., 16., 16., 32., 32., 32., 16., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 16.,
            32., 32., 32., 16., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 16.,
            32., 32., 32., 16., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8.,
            8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4.,
            8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8.,
            4., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 16., 32.,
            32., 32., 16., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 16., 32., 32., 32., 16.,
            16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 16., 32., 32., 32., 16., 8.,
            16., 16., 16., 8., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8.,
            8., 16., 16., 16., 8., 4., 8., 8., 8., 4.,
        ];

        let true_kernel_grad_elems = vec![
            29952., 30080., 30592., 30720., 33152., 33280., 33792., 33920., 45952., 46080., 46592.,
            46720., 49152., 49280., 49792., 49920., 61952., 62080., 62592., 62720., 65152., 65280.,
            65792., 65920., 29952., 30080., 30592., 30720., 33152., 33280., 33792., 33920., 45952.,
            46080., 46592., 46720., 49152., 49280., 49792., 49920., 61952., 62080., 62592., 62720.,
            65152., 65280., 65792., 65920., 29952., 30080., 30592., 30720., 33152., 33280., 33792.,
            33920., 45952., 46080., 46592., 46720., 49152., 49280., 49792., 49920., 61952., 62080.,
            62592., 62720., 65152., 65280., 65792., 65920., 29952., 30080., 30592., 30720., 33152.,
            33280., 33792., 33920., 45952., 46080., 46592., 46720., 49152., 49280., 49792., 49920.,
            61952., 62080., 62592., 62720., 65152., 65280., 65792., 65920.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input_grad.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel_grad.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }

    #[test]
    fn conv1d_strided() {
        use ndarray::prelude::*;

        use super::*;
        use ndarray::Ix3;

        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((5, 3, 10)).unwrap();
        let kernel = Array::<f32, _>::ones((6, 3, 5));
        let stride = &[2];
        let padding = &[0];
        let dilation = &[1];

        let conv_out_shape =
            conv_out_shape::<Ix3>(input.shape(), kernel.shape(), padding, stride, dilation);

        let true_output_elems = vec![
            180., 210., 240., 180., 210., 240., 180., 210., 240., 180., 210., 240., 180., 210.,
            240., 180., 210., 240., 630., 660., 690., 630., 660., 690., 630., 660., 690., 630.,
            660., 690., 630., 660., 690., 630., 660., 690., 1080., 1110., 1140., 1080., 1110.,
            1140., 1080., 1110., 1140., 1080., 1110., 1140., 1080., 1110., 1140., 1080., 1110.,
            1140., 1530., 1560., 1590., 1530., 1560., 1590., 1530., 1560., 1590., 1530., 1560.,
            1590., 1530., 1560., 1590., 1530., 1560., 1590., 1980., 2010., 2040., 1980., 2010.,
            2040., 1980., 2010., 2040., 1980., 2010., 2040., 1980., 2010., 2040., 1980., 2010.,
            2040.,
        ];

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_input_grad_elems = vec![
            6., 6., 12., 12., 18., 12., 12., 6., 6., 0., 6., 6., 12., 12., 18., 12., 12., 6., 6.,
            0., 6., 6., 12., 12., 18., 12., 12., 6., 6., 0., 6., 6., 12., 12., 18., 12., 12., 6.,
            6., 0., 6., 6., 12., 12., 18., 12., 12., 6., 6., 0., 6., 6., 12., 12., 18., 12., 12.,
            6., 6., 0., 6., 6., 12., 12., 18., 12., 12., 6., 6., 0., 6., 6., 12., 12., 18., 12.,
            12., 6., 6., 0., 6., 6., 12., 12., 18., 12., 12., 6., 6., 0., 6., 6., 12., 12., 18.,
            12., 12., 6., 6., 0., 6., 6., 12., 12., 18., 12., 12., 6., 6., 0., 6., 6., 12., 12.,
            18., 12., 12., 6., 6., 0., 6., 6., 12., 12., 18., 12., 12., 6., 6., 0., 6., 6., 12.,
            12., 18., 12., 12., 6., 6., 0., 6., 6., 12., 12., 18., 12., 12., 6., 6., 0.,
        ];

        let true_kernel_grad_elems = array![
            [
                [930., 945., 960., 975., 990.],
                [1080., 1095., 1110., 1125., 1140.],
                [1230., 1245., 1260., 1275., 1290.]
            ],
            [
                [930., 945., 960., 975., 990.],
                [1080., 1095., 1110., 1125., 1140.],
                [1230., 1245., 1260., 1275., 1290.]
            ],
            [
                [930., 945., 960., 975., 990.],
                [1080., 1095., 1110., 1125., 1140.],
                [1230., 1245., 1260., 1275., 1290.]
            ],
            [
                [930., 945., 960., 975., 990.],
                [1080., 1095., 1110., 1125., 1140.],
                [1230., 1245., 1260., 1275., 1290.]
            ],
            [
                [930., 945., 960., 975., 990.],
                [1080., 1095., 1110., 1125., 1140.],
                [1230., 1245., 1260., 1275., 1290.]
            ],
            [
                [930., 945., 960., 975., 990.],
                [1080., 1095., 1110., 1125., 1140.],
                [1230., 1245., 1260., 1275., 1290.]
            ]
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(kernel_grad, true_kernel_grad_elems);
    }

    #[test]
    fn conv2d_strided() {
        use super::*;
        use ndarray::Ix4;

        // This is an input with a batch size of 3, 2 input channels each of 5 by 5.
        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((3, 2, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((3, 2, 2, 2));

        let stride = &[2, 2];
        let padding = &[0, 0];
        let dilation = &[1, 1];

        let conv_out_shape =
            conv_out_shape::<Ix4>(input.shape(), kernel.shape(), padding, stride, dilation);

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        let true_output_elems: Vec<f32> = vec![
            124., 140., 204., 220., 124., 140., 204., 220., 124., 140., 204., 220., 524., 540.,
            604., 620., 524., 540., 604., 620., 524., 540., 604., 620., 924., 940., 1004., 1020.,
            924., 940., 1004., 1020., 924., 940., 1004., 1020.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros((3, 2, 5, 5));
        let mut kernel_grad = Array::<f32, _>::zeros((3, 2, 2, 2));
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_input_grad_elems: Vec<f32> = vec![
            3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 0., 0.,
            0., 0., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3.,
            0., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3.,
            3., 3., 3., 0., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3.,
            3., 0., 3., 3., 3., 3., 0., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0.,
            3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 0., 0., 0., 0., 0., 3., 3., 3., 3., 0., 3., 3.,
            3., 3., 0., 3., 3., 3., 3., 0., 3., 3., 3., 3., 0., 0., 0., 0., 0., 0.,
        ];

        let true_kernel_grad_elems: Vec<f32> = vec![
            672., 684., 732., 744., 972., 984., 1032., 1044., 672., 684., 732., 744., 972., 984.,
            1032., 1044., 672., 684., 732., 744., 972., 984., 1032., 1044.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }

    #[test]
    fn conv3d_strided() {
        use super::*;

        let input_elems = Array::<f32, _>::from_iter((0..750).map(|el| el as f32));
        let input = input_elems.into_shape((2, 3, 5, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((4, 3, 2, 2, 2));

        let stride = &[1, 2, 3];
        let padding = &[0, 0, 0];
        let dilation = &[1, 1, 1];

        let conv_out_shape = conv_out_shape::<ndarray::Ix5>(
            input.shape(),
            kernel.shape(),
            padding,
            stride,
            dilation,
        );

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        let true_output_elems = vec![
            3372., 3444., 3612., 3684., 3972., 4044., 4212., 4284., 4572., 4644., 4812., 4884.,
            5172., 5244., 5412., 5484., 3372., 3444., 3612., 3684., 3972., 4044., 4212., 4284.,
            4572., 4644., 4812., 4884., 5172., 5244., 5412., 5484., 3372., 3444., 3612., 3684.,
            3972., 4044., 4212., 4284., 4572., 4644., 4812., 4884., 5172., 5244., 5412., 5484.,
            3372., 3444., 3612., 3684., 3972., 4044., 4212., 4284., 4572., 4644., 4812., 4884.,
            5172., 5244., 5412., 5484., 12372., 12444., 12612., 12684., 12972., 13044., 13212.,
            13284., 13572., 13644., 13812., 13884., 14172., 14244., 14412., 14484., 12372., 12444.,
            12612., 12684., 12972., 13044., 13212., 13284., 13572., 13644., 13812., 13884., 14172.,
            14244., 14412., 14484., 12372., 12444., 12612., 12684., 12972., 13044., 13212., 13284.,
            13572., 13644., 13812., 13884., 14172., 14244., 14412., 14484., 12372., 12444., 12612.,
            12684., 12972., 13044., 13212., 13284., 13572., 13644., 13812., 13884., 14172., 14244.,
            14412., 14484.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_input_grad_elems = vec![
            4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 0., 0.,
            0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8.,
            8., 0., 0., 0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8.,
            8., 0., 8., 8., 0., 0., 0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0.,
            8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4.,
            4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 0., 0., 0., 0., 0., 4., 4., 0., 4., 4., 4., 4.,
            0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 0., 0., 0., 0., 0., 8., 8., 0., 8.,
            8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 8.,
            8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0.,
            0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8.,
            0., 0., 0., 0., 0., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4.,
            0., 4., 4., 0., 0., 0., 0., 0., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4.,
            4., 4., 4., 0., 4., 4., 0., 0., 0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8.,
            8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0.,
            8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 8., 8., 0., 8., 8.,
            8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 4., 4.,
            0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 0., 0., 0., 0.,
            0., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 0.,
            0., 0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0.,
            8., 8., 0., 0., 0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8.,
            8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8.,
            0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 4., 4., 0., 4., 4., 4., 4., 0., 4.,
            4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 0., 0., 0., 0., 0., 4., 4., 0., 4., 4., 4.,
            4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 0., 0., 0., 0., 0., 8., 8., 0.,
            8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0.,
            8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0.,
            0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8.,
            8., 0., 0., 0., 0., 0., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4.,
            4., 0., 4., 4., 0., 0., 0., 0., 0., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0.,
            4., 4., 4., 4., 0., 4., 4., 0., 0., 0., 0., 0., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8.,
            8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 8., 8., 0., 8., 8., 8., 8.,
            0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 8., 8., 0., 8.,
            8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 8., 8., 0., 8., 8., 0., 0., 0., 0., 0., 4.,
            4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 4., 4., 0., 4., 4., 0., 0., 0.,
            0., 0.,
        ];

        let true_kernel_grad_elems = vec![
            7408., 7440., 7568., 7600., 8208., 8240., 8368., 8400., 11408., 11440., 11568., 11600.,
            12208., 12240., 12368., 12400., 15408., 15440., 15568., 15600., 16208., 16240., 16368.,
            16400., 7408., 7440., 7568., 7600., 8208., 8240., 8368., 8400., 11408., 11440., 11568.,
            11600., 12208., 12240., 12368., 12400., 15408., 15440., 15568., 15600., 16208., 16240.,
            16368., 16400., 7408., 7440., 7568., 7600., 8208., 8240., 8368., 8400., 11408., 11440.,
            11568., 11600., 12208., 12240., 12368., 12400., 15408., 15440., 15568., 15600., 16208.,
            16240., 16368., 16400., 7408., 7440., 7568., 7600., 8208., 8240., 8368., 8400., 11408.,
            11440., 11568., 11600., 12208., 12240., 12368., 12400., 15408., 15440., 15568., 15600.,
            16208., 16240., 16368., 16400.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input_grad.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel_grad.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }

    #[test]
    fn conv1d_dilated() {
        use ndarray::prelude::*;

        use super::*;
        use ndarray::Ix3;

        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((5, 3, 10)).unwrap();
        let kernel = Array::<f32, _>::ones((6, 3, 5));
        let stride = &[2];
        let padding = &[0];
        let dilation = &[2];

        let conv_out_shape =
            conv_out_shape::<Ix3>(input.shape(), kernel.shape(), padding, stride, dilation);

        let true_output_elems = vec![
            210., 210., 210., 210., 210., 210., 660., 660., 660., 660., 660., 660., 1110., 1110.,
            1110., 1110., 1110., 1110., 1560., 1560., 1560., 1560., 1560., 1560., 2010., 2010.,
            2010., 2010., 2010., 2010.,
        ];

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_input_grad_elems = vec![
            6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0.,
            6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0.,
            6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0.,
            6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0.,
            6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0.,
            6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0.,
            6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0., 6., 0.,
        ];

        let true_kernel_grad_elems = array![
            [
                [300., 310., 320., 330., 340.],
                [350., 360., 370., 380., 390.],
                [400., 410., 420., 430., 440.]
            ],
            [
                [300., 310., 320., 330., 340.],
                [350., 360., 370., 380., 390.],
                [400., 410., 420., 430., 440.]
            ],
            [
                [300., 310., 320., 330., 340.],
                [350., 360., 370., 380., 390.],
                [400., 410., 420., 430., 440.]
            ],
            [
                [300., 310., 320., 330., 340.],
                [350., 360., 370., 380., 390.],
                [400., 410., 420., 430., 440.]
            ],
            [
                [300., 310., 320., 330., 340.],
                [350., 360., 370., 380., 390.],
                [400., 410., 420., 430., 440.]
            ],
            [
                [300., 310., 320., 330., 340.],
                [350., 360., 370., 380., 390.],
                [400., 410., 420., 430., 440.]
            ]
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(kernel_grad, true_kernel_grad_elems);
    }

    #[test]
    fn conv2d_dilated() {
        use super::*;
        use ndarray::Ix4;

        // This is an input with a batch size of 3, 2 input channels each of 5 by 5.
        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((3, 2, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((3, 2, 2, 2));

        let stride = &[2, 2];
        let padding = &[0, 0];
        let dilation = &[2, 2];

        let conv_out_shape =
            conv_out_shape::<Ix4>(input.shape(), kernel.shape(), padding, stride, dilation);

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        let true_output_elems: Vec<f32> = vec![
            148., 164., 228., 244., 148., 164., 228., 244., 148., 164., 228., 244., 548., 564.,
            628., 644., 548., 564., 628., 644., 548., 564., 628., 644., 948., 964., 1028., 1044.,
            948., 964., 1028., 1044., 948., 964., 1028., 1044.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros((3, 2, 5, 5));
        let mut kernel_grad = Array::<f32, _>::zeros((3, 2, 2, 2));
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_input_grad_elems: Vec<f32> = vec![
            3., 0., 6., 0., 3., 0., 0., 0., 0., 0., 6., 0., 12., 0., 6., 0., 0., 0., 0., 0., 3.,
            0., 6., 0., 3., 3., 0., 6., 0., 3., 0., 0., 0., 0., 0., 6., 0., 12., 0., 6., 0., 0.,
            0., 0., 0., 3., 0., 6., 0., 3., 3., 0., 6., 0., 3., 0., 0., 0., 0., 0., 6., 0., 12.,
            0., 6., 0., 0., 0., 0., 0., 3., 0., 6., 0., 3., 3., 0., 6., 0., 3., 0., 0., 0., 0., 0.,
            6., 0., 12., 0., 6., 0., 0., 0., 0., 0., 3., 0., 6., 0., 3., 3., 0., 6., 0., 3., 0.,
            0., 0., 0., 0., 6., 0., 12., 0., 6., 0., 0., 0., 0., 0., 3., 0., 6., 0., 3., 3., 0.,
            6., 0., 3., 0., 0., 0., 0., 0., 6., 0., 12., 0., 6., 0., 0., 0., 0., 0., 3., 0., 6.,
            0., 3.,
        ];

        let true_kernel_grad_elems: Vec<f32> = vec![
            672., 696., 792., 816., 972., 996., 1092., 1116., 672., 696., 792., 816., 972., 996.,
            1092., 1116., 672., 696., 792., 816., 972., 996., 1092., 1116.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }

    #[test]
    fn conv3d_dilated() {
        use super::*;

        let input_elems = Array::<f32, _>::from_iter((0..750).map(|el| el as f32));
        let input = input_elems.into_shape((2, 3, 5, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((4, 3, 2, 2, 2));

        let stride = &[1, 2, 3];
        let padding = &[0, 0, 0];
        let dilation = &[1, 2, 2];

        let conv_out_shape = conv_out_shape::<ndarray::Ix5>(
            input.shape(),
            kernel.shape(),
            padding,
            stride,
            dilation,
        );

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&input, &kernel, &mut conv_out, stride, dilation);

        let true_output_elems = vec![
            3444., 3684., 4044., 4284., 4644., 4884., 5244., 5484., 3444., 3684., 4044., 4284.,
            4644., 4884., 5244., 5484., 3444., 3684., 4044., 4284., 4644., 4884., 5244., 5484.,
            3444., 3684., 4044., 4284., 4644., 4884., 5244., 5484., 12444., 12684., 13044., 13284.,
            13644., 13884., 14244., 14484., 12444., 12684., 13044., 13284., 13644., 13884., 14244.,
            14484., 12444., 12684., 13044., 13284., 13644., 13884., 14244., 14484., 12444., 12684.,
            13044., 13284., 13644., 13884., 14244., 14484.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_input_grad_elems = vec![
            4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0.,
            4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0.,
            0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0.,
            0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0.,
            16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0.,
            0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16.,
            0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0.,
            0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8.,
            0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0.,
            0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4.,
            0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0.,
            0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0.,
            16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0.,
            0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0.,
            4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0.,
            0.,
        ];

        let true_kernel_grad_elems = vec![
            3680., 3712., 3840., 3872., 4080., 4112., 4240., 4272., 5680., 5712., 5840., 5872.,
            6080., 6112., 6240., 6272., 7680., 7712., 7840., 7872., 8080., 8112., 8240., 8272.,
            3680., 3712., 3840., 3872., 4080., 4112., 4240., 4272., 5680., 5712., 5840., 5872.,
            6080., 6112., 6240., 6272., 7680., 7712., 7840., 7872., 8080., 8112., 8240., 8272.,
            3680., 3712., 3840., 3872., 4080., 4112., 4240., 4272., 5680., 5712., 5840., 5872.,
            6080., 6112., 6240., 6272., 7680., 7712., 7840., 7872., 8080., 8112., 8240., 8272.,
            3680., 3712., 3840., 3872., 4080., 4112., 4240., 4272., 5680., 5712., 5840., 5872.,
            6080., 6112., 6240., 6272., 7680., 7712., 7840., 7872., 8080., 8112., 8240., 8272.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input_grad.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel_grad.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }

    #[test]
    fn conv2d_padded() {
        use super::*;
        use ndarray::Ix4;

        // This is an input with a batch size of 3, 2 input channels each of 5 by 5.
        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((3, 2, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((3, 2, 2, 2));

        let stride = &[1, 1];
        let padding = &[3, 1];
        let dilation = &[1, 1];

        let padded_input = pad(&input, padding, &Zero);
        assert_eq!(padded_input.shape(), &[3, 2, 11, 7]);

        let conv_out_shape =
            conv_out_shape_padded::<Ix4>(padded_input.shape(), kernel.shape(), stride, dilation);

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution(&padded_input, &kernel, &mut conv_out, stride, dilation);

        let true_output_elems = vec![
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 25., 52., 56., 60., 64., 33., 60.,
            124., 132., 140., 148., 76., 80., 164., 172., 180., 188., 96., 100., 204., 212., 220.,
            228., 116., 120., 244., 252., 260., 268., 136., 65., 132., 136., 140., 144., 73., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 25., 52., 56., 60., 64., 33., 60., 124., 132., 140., 148., 76., 80., 164., 172.,
            180., 188., 96., 100., 204., 212., 220., 228., 116., 120., 244., 252., 260., 268.,
            136., 65., 132., 136., 140., 144., 73., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 25., 52., 56., 60., 64., 33., 60.,
            124., 132., 140., 148., 76., 80., 164., 172., 180., 188., 96., 100., 204., 212., 220.,
            228., 116., 120., 244., 252., 260., 268., 136., 65., 132., 136., 140., 144., 73., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 125., 252., 256., 260., 264., 133., 260., 524., 532., 540., 548., 276., 280., 564.,
            572., 580., 588., 296., 300., 604., 612., 620., 628., 316., 320., 644., 652., 660.,
            668., 336., 165., 332., 336., 340., 344., 173., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 125., 252., 256., 260., 264.,
            133., 260., 524., 532., 540., 548., 276., 280., 564., 572., 580., 588., 296., 300.,
            604., 612., 620., 628., 316., 320., 644., 652., 660., 668., 336., 165., 332., 336.,
            340., 344., 173., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 125., 252., 256., 260., 264., 133., 260., 524., 532., 540.,
            548., 276., 280., 564., 572., 580., 588., 296., 300., 604., 612., 620., 628., 316.,
            320., 644., 652., 660., 668., 336., 165., 332., 336., 340., 344., 173., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 225.,
            452., 456., 460., 464., 233., 460., 924., 932., 940., 948., 476., 480., 964., 972.,
            980., 988., 496., 500., 1004., 1012., 1020., 1028., 516., 520., 1044., 1052., 1060.,
            1068., 536., 265., 532., 536., 540., 544., 273., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 225., 452., 456., 460.,
            464., 233., 460., 924., 932., 940., 948., 476., 480., 964., 972., 980., 988., 496.,
            500., 1004., 1012., 1020., 1028., 516., 520., 1044., 1052., 1060., 1068., 536., 265.,
            532., 536., 540., 544., 273., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 225., 452., 456., 460., 464., 233., 460., 924.,
            932., 940., 948., 476., 480., 964., 972., 980., 988., 496., 500., 1004., 1012., 1020.,
            1028., 516., 520., 1044., 1052., 1060., 1068., 536., 265., 532., 536., 540., 544.,
            273., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        convolution_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &padded_input,
            &kernel,
            padding,
            stride,
            dilation,
            true,
            true,
        );

        let true_kernel_grad_elems = vec![
            4650., 4650., 4650., 4650., 6525., 6525., 6525., 6525., 4650., 4650., 4650., 4650.,
            6525., 6525., 6525., 6525., 4650., 4650., 4650., 4650., 6525., 6525., 6525., 6525.,
        ];

        let true_input_grad_elems = vec![
            12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
            12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
            12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
            12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
            12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
            12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
            12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
            12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
            12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12., 12.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input_grad.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel_grad.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }

    #[test]
    fn grouped_conv1d() {
        use super::*;
        use ndarray::prelude::*;
        use ndarray::Ix3;

        let input_elems = (0..150).map(|el| el as f32).collect::<Array<f32, _>>();
        let input = input_elems.into_shape((5, 3, 10)).unwrap();
        let kernel = Array::<f32, _>::ones((6, 1, 5));
        let stride = &[2];
        let padding = &[0];
        let dilation = &[2];
        let groups = 3;

        let conv_out_shape =
            conv_out_shape::<Ix3>(input.shape(), kernel.shape(), padding, stride, dilation);

        let true_output_elems = vec![
            20., 20., 70., 70., 120., 120., 170., 170., 220., 220., 270., 270., 320., 320., 370.,
            370., 420., 420., 470., 470., 520., 520., 570., 570., 620., 620., 670., 670., 720.,
            720.,
        ];

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution_with_groups(&input, &kernel, &mut conv_out, stride, dilation, groups);

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_with_groups_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            groups,
            true,
            true,
        );

        let true_input_grad_elems = vec![
            2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.,
            2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.,
            2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.,
            2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.,
            2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.,
            2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.,
            2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0., 2., 0.,
        ];

        let true_kernel_grad_elems = array![
            [[300., 310., 320., 330., 340.]],
            [[300., 310., 320., 330., 340.]],
            [[350., 360., 370., 380., 390.]],
            [[350., 360., 370., 380., 390.]],
            [[400., 410., 420., 430., 440.]],
            [[400., 410., 420., 430., 440.]]
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(kernel_grad, true_kernel_grad_elems);
    }

    #[test]
    fn grouped_conv2d() {
        use super::*;
        use ndarray::Ix4;
        // This is an input with a batch size of 4, 8 input channels each of 5 by 5.
        // Constructing an input.
        let input: Array<f32, Ix4> = (0..800)
            .map(|el| el as f32)
            .collect::<Array<f32, _>>()
            .into_shape((4, 8, 5, 5))
            .unwrap();

        // Both output and input channels need to be divisible by group.
        // Group is 2 so we must divide the input channels by 2.
        let kernel = Array::<f32, _>::ones((8, 4, 2, 2));
        let stride = &[1, 1];
        let padding = &[0, 0];
        let dilation = &[1, 1];
        let groups = 2;

        let conv_out_shape =
            conv_out_shape::<Ix4>(input.shape(), kernel.shape(), padding, stride, dilation);
        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution_with_groups(&input, &kernel, &mut conv_out, stride, dilation, groups);

        let true_output_elems = vec![
            648., 664., 680., 696., 728., 744., 760., 776., 808., 824., 840., 856., 888., 904.,
            920., 936., 648., 664., 680., 696., 728., 744., 760., 776., 808., 824., 840., 856.,
            888., 904., 920., 936., 648., 664., 680., 696., 728., 744., 760., 776., 808., 824.,
            840., 856., 888., 904., 920., 936., 648., 664., 680., 696., 728., 744., 760., 776.,
            808., 824., 840., 856., 888., 904., 920., 936., 2248., 2264., 2280., 2296., 2328.,
            2344., 2360., 2376., 2408., 2424., 2440., 2456., 2488., 2504., 2520., 2536., 2248.,
            2264., 2280., 2296., 2328., 2344., 2360., 2376., 2408., 2424., 2440., 2456., 2488.,
            2504., 2520., 2536., 2248., 2264., 2280., 2296., 2328., 2344., 2360., 2376., 2408.,
            2424., 2440., 2456., 2488., 2504., 2520., 2536., 2248., 2264., 2280., 2296., 2328.,
            2344., 2360., 2376., 2408., 2424., 2440., 2456., 2488., 2504., 2520., 2536., 3848.,
            3864., 3880., 3896., 3928., 3944., 3960., 3976., 4008., 4024., 4040., 4056., 4088.,
            4104., 4120., 4136., 3848., 3864., 3880., 3896., 3928., 3944., 3960., 3976., 4008.,
            4024., 4040., 4056., 4088., 4104., 4120., 4136., 3848., 3864., 3880., 3896., 3928.,
            3944., 3960., 3976., 4008., 4024., 4040., 4056., 4088., 4104., 4120., 4136., 3848.,
            3864., 3880., 3896., 3928., 3944., 3960., 3976., 4008., 4024., 4040., 4056., 4088.,
            4104., 4120., 4136., 5448., 5464., 5480., 5496., 5528., 5544., 5560., 5576., 5608.,
            5624., 5640., 5656., 5688., 5704., 5720., 5736., 5448., 5464., 5480., 5496., 5528.,
            5544., 5560., 5576., 5608., 5624., 5640., 5656., 5688., 5704., 5720., 5736., 5448.,
            5464., 5480., 5496., 5528., 5544., 5560., 5576., 5608., 5624., 5640., 5656., 5688.,
            5704., 5720., 5736., 5448., 5464., 5480., 5496., 5528., 5544., 5560., 5576., 5608.,
            5624., 5640., 5656., 5688., 5704., 5720., 5736., 7048., 7064., 7080., 7096., 7128.,
            7144., 7160., 7176., 7208., 7224., 7240., 7256., 7288., 7304., 7320., 7336., 7048.,
            7064., 7080., 7096., 7128., 7144., 7160., 7176., 7208., 7224., 7240., 7256., 7288.,
            7304., 7320., 7336., 7048., 7064., 7080., 7096., 7128., 7144., 7160., 7176., 7208.,
            7224., 7240., 7256., 7288., 7304., 7320., 7336., 7048., 7064., 7080., 7096., 7128.,
            7144., 7160., 7176., 7208., 7224., 7240., 7256., 7288., 7304., 7320., 7336., 8648.,
            8664., 8680., 8696., 8728., 8744., 8760., 8776., 8808., 8824., 8840., 8856., 8888.,
            8904., 8920., 8936., 8648., 8664., 8680., 8696., 8728., 8744., 8760., 8776., 8808.,
            8824., 8840., 8856., 8888., 8904., 8920., 8936., 8648., 8664., 8680., 8696., 8728.,
            8744., 8760., 8776., 8808., 8824., 8840., 8856., 8888., 8904., 8920., 8936., 8648.,
            8664., 8680., 8696., 8728., 8744., 8760., 8776., 8808., 8824., 8840., 8856., 8888.,
            8904., 8920., 8936., 10248., 10264., 10280., 10296., 10328., 10344., 10360., 10376.,
            10408., 10424., 10440., 10456., 10488., 10504., 10520., 10536., 10248., 10264., 10280.,
            10296., 10328., 10344., 10360., 10376., 10408., 10424., 10440., 10456., 10488., 10504.,
            10520., 10536., 10248., 10264., 10280., 10296., 10328., 10344., 10360., 10376., 10408.,
            10424., 10440., 10456., 10488., 10504., 10520., 10536., 10248., 10264., 10280., 10296.,
            10328., 10344., 10360., 10376., 10408., 10424., 10440., 10456., 10488., 10504., 10520.,
            10536., 11848., 11864., 11880., 11896., 11928., 11944., 11960., 11976., 12008., 12024.,
            12040., 12056., 12088., 12104., 12120., 12136., 11848., 11864., 11880., 11896., 11928.,
            11944., 11960., 11976., 12008., 12024., 12040., 12056., 12088., 12104., 12120., 12136.,
            11848., 11864., 11880., 11896., 11928., 11944., 11960., 11976., 12008., 12024., 12040.,
            12056., 12088., 12104., 12120., 12136., 11848., 11864., 11880., 11896., 11928., 11944.,
            11960., 11976., 12008., 12024., 12040., 12056., 12088., 12104., 12120., 12136.,
        ];

        assert_eq!(
            conv_out,
            Array::from_shape_vec(conv_out.raw_dim(), true_output_elems).unwrap()
        );

        // // Backward pass
        let mut input_grad = Array::<f32, _>::zeros((4, 8, 5, 5));
        let mut kernel_grad = Array::<f32, _>::zeros((8, 4, 2, 2));
        let d_out = Array::<f32, _>::ones(conv_out.raw_dim());

        convolution_with_groups_backward(
            &mut input_grad,
            &mut kernel_grad,
            &d_out,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            groups,
            true,
            true,
        );

        let true_kernel_grad_elems: Vec<f32> = vec![
            19776., 19840., 20096., 20160., 21376., 21440., 21696., 21760., 22976., 23040., 23296.,
            23360., 24576., 24640., 24896., 24960., 19776., 19840., 20096., 20160., 21376., 21440.,
            21696., 21760., 22976., 23040., 23296., 23360., 24576., 24640., 24896., 24960., 19776.,
            19840., 20096., 20160., 21376., 21440., 21696., 21760., 22976., 23040., 23296., 23360.,
            24576., 24640., 24896., 24960., 19776., 19840., 20096., 20160., 21376., 21440., 21696.,
            21760., 22976., 23040., 23296., 23360., 24576., 24640., 24896., 24960., 26176., 26240.,
            26496., 26560., 27776., 27840., 28096., 28160., 29376., 29440., 29696., 29760., 30976.,
            31040., 31296., 31360., 26176., 26240., 26496., 26560., 27776., 27840., 28096., 28160.,
            29376., 29440., 29696., 29760., 30976., 31040., 31296., 31360., 26176., 26240., 26496.,
            26560., 27776., 27840., 28096., 28160., 29376., 29440., 29696., 29760., 30976., 31040.,
            31296., 31360., 26176., 26240., 26496., 26560., 27776., 27840., 28096., 28160., 29376.,
            29440., 29696., 29760., 30976., 31040., 31296., 31360.,
        ];
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel_grad.raw_dim(), true_kernel_grad_elems).unwrap()
        );

        let true_input_grad_elems: Vec<f32> = vec![
            4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8.,
            4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8.,
            8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8.,
            8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8.,
            8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16.,
            16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16.,
            16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16.,
            16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8.,
            8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4.,
            8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8.,
            16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8.,
            16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8.,
            16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4.,
            4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8.,
            4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8.,
            8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16.,
            16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8.,
            8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8.,
            8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16.,
            16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16.,
            16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16.,
            16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8.,
            8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4.,
            8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8.,
            16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8.,
            16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8.,
            16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4.,
            4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 4., 8., 8., 8., 4., 4., 8., 8., 8., 4., 8., 16., 16., 16., 8., 8., 16., 16., 16.,
            8., 8., 16., 16., 16., 8., 4., 8., 8., 8., 4.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input_grad.raw_dim(), true_input_grad_elems).unwrap()
        );
    }

    #[test]
    fn grouped_conv3d() {
        use super::*;

        let input_elems = Array::<f32, _>::from_iter((0..2000).map(|el| el as f32));
        let input = input_elems.into_shape((2, 8, 5, 5, 5)).unwrap();
        let kernel = Array::<f32, _>::ones((16, 2, 2, 2, 2));

        let stride = &[1, 2, 3];
        let padding = &[0, 0, 0];
        let dilation = &[1, 2, 2];
        let groups = 4;

        let conv_out_shape = conv_out_shape::<ndarray::Ix5>(
            input.shape(),
            kernel.shape(),
            padding,
            stride,
            dilation,
        );

        // Convolution result
        let mut conv_out = Array::<f32, _>::zeros(conv_out_shape);

        convolution_with_groups(&input, &kernel, &mut conv_out, stride, dilation, groups);

        let true_output_elems = vec![
            1296., 1456., 1696., 1856., 2096., 2256., 2496., 2656., 1296., 1456., 1696., 1856.,
            2096., 2256., 2496., 2656., 1296., 1456., 1696., 1856., 2096., 2256., 2496., 2656.,
            1296., 1456., 1696., 1856., 2096., 2256., 2496., 2656., 5296., 5456., 5696., 5856.,
            6096., 6256., 6496., 6656., 5296., 5456., 5696., 5856., 6096., 6256., 6496., 6656.,
            5296., 5456., 5696., 5856., 6096., 6256., 6496., 6656., 5296., 5456., 5696., 5856.,
            6096., 6256., 6496., 6656., 9296., 9456., 9696., 9856., 10096., 10256., 10496., 10656.,
            9296., 9456., 9696., 9856., 10096., 10256., 10496., 10656., 9296., 9456., 9696., 9856.,
            10096., 10256., 10496., 10656., 9296., 9456., 9696., 9856., 10096., 10256., 10496.,
            10656., 13296., 13456., 13696., 13856., 14096., 14256., 14496., 14656., 13296., 13456.,
            13696., 13856., 14096., 14256., 14496., 14656., 13296., 13456., 13696., 13856., 14096.,
            14256., 14496., 14656., 13296., 13456., 13696., 13856., 14096., 14256., 14496., 14656.,
            17296., 17456., 17696., 17856., 18096., 18256., 18496., 18656., 17296., 17456., 17696.,
            17856., 18096., 18256., 18496., 18656., 17296., 17456., 17696., 17856., 18096., 18256.,
            18496., 18656., 17296., 17456., 17696., 17856., 18096., 18256., 18496., 18656., 21296.,
            21456., 21696., 21856., 22096., 22256., 22496., 22656., 21296., 21456., 21696., 21856.,
            22096., 22256., 22496., 22656., 21296., 21456., 21696., 21856., 22096., 22256., 22496.,
            22656., 21296., 21456., 21696., 21856., 22096., 22256., 22496., 22656., 25296., 25456.,
            25696., 25856., 26096., 26256., 26496., 26656., 25296., 25456., 25696., 25856., 26096.,
            26256., 26496., 26656., 25296., 25456., 25696., 25856., 26096., 26256., 26496., 26656.,
            25296., 25456., 25696., 25856., 26096., 26256., 26496., 26656., 29296., 29456., 29696.,
            29856., 30096., 30256., 30496., 30656., 29296., 29456., 29696., 29856., 30096., 30256.,
            30496., 30656., 29296., 29456., 29696., 29856., 30096., 30256., 30496., 30656., 29296.,
            29456., 29696., 29856., 30096., 30256., 30496., 30656.,
        ];

        assert_eq!(
            conv_out,
            Array::<f32, _>::from_shape_vec(conv_out_shape, true_output_elems).unwrap()
        );

        let mut input_grad = Array::<f32, _>::zeros(input.raw_dim());
        let mut kernel_grad = Array::<f32, _>::zeros(kernel.raw_dim());
        let conv_out_grad = Array::<f32, _>::ones(conv_out_shape);

        // Backward pass.
        convolution_with_groups_backward(
            &mut input_grad,
            &mut kernel_grad,
            &conv_out_grad,
            &input,
            &kernel,
            padding,
            stride,
            dilation,
            groups,
            true,
            true,
        );

        let true_input_grad_elems = vec![
            4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0.,
            4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0.,
            0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0.,
            0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0.,
            16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0.,
            0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16.,
            0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0.,
            0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8.,
            0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0.,
            0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4.,
            0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0.,
            0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0.,
            16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0.,
            0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0.,
            4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0.,
            0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4.,
            0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0.,
            16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0.,
            0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16.,
            0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0.,
            0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8.,
            0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0.,
            0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4.,
            0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0.,
            0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0.,
            16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0.,
            0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0.,
            4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0.,
            0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4.,
            0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0.,
            16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0.,
            0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16.,
            0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0.,
            0., 0., 0., 8., 0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 0., 0., 0., 0., 0., 4., 0., 4., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 8., 0., 8., 0.,
            0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0.,
            8., 0., 8., 0., 0., 0., 0., 0., 0., 0., 16., 0., 16., 0., 0., 0., 0., 0., 0., 0., 8.,
            0., 8., 0., 0., 4., 0., 4., 0., 0., 0., 0., 0., 0., 0., 8., 0., 8., 0., 0., 0., 0., 0.,
            0., 0., 4., 0., 4., 0., 0.,
        ];

        let true_kernel_grad_elems = vec![
            8680., 8712., 8840., 8872., 9080., 9112., 9240., 9272., 10680., 10712., 10840., 10872.,
            11080., 11112., 11240., 11272., 8680., 8712., 8840., 8872., 9080., 9112., 9240., 9272.,
            10680., 10712., 10840., 10872., 11080., 11112., 11240., 11272., 8680., 8712., 8840.,
            8872., 9080., 9112., 9240., 9272., 10680., 10712., 10840., 10872., 11080., 11112.,
            11240., 11272., 8680., 8712., 8840., 8872., 9080., 9112., 9240., 9272., 10680., 10712.,
            10840., 10872., 11080., 11112., 11240., 11272., 12680., 12712., 12840., 12872., 13080.,
            13112., 13240., 13272., 14680., 14712., 14840., 14872., 15080., 15112., 15240., 15272.,
            12680., 12712., 12840., 12872., 13080., 13112., 13240., 13272., 14680., 14712., 14840.,
            14872., 15080., 15112., 15240., 15272., 12680., 12712., 12840., 12872., 13080., 13112.,
            13240., 13272., 14680., 14712., 14840., 14872., 15080., 15112., 15240., 15272., 12680.,
            12712., 12840., 12872., 13080., 13112., 13240., 13272., 14680., 14712., 14840., 14872.,
            15080., 15112., 15240., 15272., 16680., 16712., 16840., 16872., 17080., 17112., 17240.,
            17272., 18680., 18712., 18840., 18872., 19080., 19112., 19240., 19272., 16680., 16712.,
            16840., 16872., 17080., 17112., 17240., 17272., 18680., 18712., 18840., 18872., 19080.,
            19112., 19240., 19272., 16680., 16712., 16840., 16872., 17080., 17112., 17240., 17272.,
            18680., 18712., 18840., 18872., 19080., 19112., 19240., 19272., 16680., 16712., 16840.,
            16872., 17080., 17112., 17240., 17272., 18680., 18712., 18840., 18872., 19080., 19112.,
            19240., 19272., 20680., 20712., 20840., 20872., 21080., 21112., 21240., 21272., 22680.,
            22712., 22840., 22872., 23080., 23112., 23240., 23272., 20680., 20712., 20840., 20872.,
            21080., 21112., 21240., 21272., 22680., 22712., 22840., 22872., 23080., 23112., 23240.,
            23272., 20680., 20712., 20840., 20872., 21080., 21112., 21240., 21272., 22680., 22712.,
            22840., 22872., 23080., 23112., 23240., 23272., 20680., 20712., 20840., 20872., 21080.,
            21112., 21240., 21272., 22680., 22712., 22840., 22872., 23080., 23112., 23240., 23272.,
        ];

        assert_eq!(
            input_grad,
            Array::from_shape_vec(input_grad.raw_dim(), true_input_grad_elems).unwrap(),
        );
        assert_eq!(
            kernel_grad,
            Array::from_shape_vec(kernel_grad.raw_dim(), true_kernel_grad_elems).unwrap(),
        );
    }
}
