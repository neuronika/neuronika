use crate::variable::{
    expect_tensor, expect_tensor_mut, Backward, Data as NData, Forward, Gradient, Overwrite,
    Tensor, Var, VarDiff,
};
use ndarray::{Dimension, RemoveAxis};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Padding Modes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mod padding;
pub use padding::{Constant, PaddingMode, Reflective, Replicative, Zero};
use padding::{ReflPad, ReplPad};
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Utility Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mod numeric;
use numeric::{
    check_conv_args, check_groups_args, conv_out_shape, convolution, convolution_backward,
    convolution_unary_backward, convolution_with_groups, convolution_with_groups_backward,
    convolution_with_groups_unary_backward, pad,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolve Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/// Convolution.
pub trait Convolve<Inp, Ker, Pad: PaddingMode> {
    /// The type of the convolution's result. See the [*differentiability arithmetic*] for more
    /// details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Applies a *n*-dimensional convolution with the given parameters. *n* can be either 1, 2 or
    /// 3.
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
/// Grouped convolution.
pub trait ConvolveWithGroups<Inp, Ker, Pad: PaddingMode> {
    /// The type of the grouped convolution's result. See the [*differentiability arithmetic*] for
    /// more details.
    ///
    /// [*differentiability arithmetic*]: index.html#differentiability-arithmetic
    type Output;

    /// Applies a *n*-dimensional grouped convolution with the given parameters. *n* can be either
    /// 1, 2 or 3.
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

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
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

    fn data_mut(&self) -> RefMut<Tensor<Self::Dim>> {
        self.data.borrow_mut()
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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

#[cfg(test)]
mod tests {
    mod convolution_node {
        use crate::variable::{Differentiable, Input, InputBackward};
        use ndarray::StrideShape;

        use super::super::*;

        fn new_input<D, Sh>(shape: Sh, elems: Vec<f32>) -> Rc<Input<D>>
        where
            D: Dimension + 'static,
            Sh: Into<StrideShape<D>>,
        {
            Input::new(new_tensor(shape, elems)).node
        }

        fn new_backward_input<D, Sh>(shape: Sh, elems: Vec<f32>) -> Rc<InputBackward<D>>
        where
            D: Dimension + 'static,
            Sh: Into<StrideShape<D>>,
        {
            Rc::new(Input::new(new_tensor(shape, elems)).node.differentiable())
        }

        fn new_tensor<D, Sh>(shape: Sh, elems: Vec<f32>) -> Tensor<D>
        where
            D: Dimension + 'static,
            Sh: Into<StrideShape<D>>,
        {
            Tensor::from_shape_vec(shape, elems).unwrap()
        }
        mod forward {
            use super::*;

            #[test]
            fn creation() {
                let input = new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
                let kernel = new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);
                let node = Convolution::new(input, kernel, &[1, 1], &[1, 1], &[0, 0], Zero);

                let outshape: ndarray::Ix4 =
                    conv_out_shape(&[4, 4, 6, 6], &[4, 4, 2, 2], &[0, 0], &[1, 1], &[1, 1]);
                assert_eq!(*node.data(), Tensor::from_elem(outshape, 0.));
                assert!(!node.was_computed());
            }

            #[test]
            fn computation_was_computed_transition() {
                let input = new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
                let kernel = new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);
                let node = Convolution::new(input, kernel, &[1, 1], &[1, 1], &[0, 0], Zero);

                node.forward();
                assert!(node.was_computed());

                node.forward();
                assert!(node.was_computed());

                node.reset_computation();
                assert!(!node.was_computed());

                node.reset_computation();
                assert!(!node.was_computed());
            }
        }

        mod forward_grouped {
            use super::*;

            #[test]
            fn creation() {
                let input = new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
                let kernel = new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);
                let node =
                    GroupedConvolution::new(input, kernel, &[1, 1], &[1, 1], &[0, 0], Zero, 2);

                let outshape: ndarray::Ix4 =
                    conv_out_shape(&[4, 4, 6, 6], &[4, 4, 2, 2], &[0, 0], &[1, 1], &[1, 1]);
                assert_eq!(*node.data(), Tensor::from_elem(outshape, 0.));
                assert!(!node.was_computed());
            }

            #[test]
            fn computation_was_computed_transition() {
                let input = new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
                let kernel = new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);
                let node =
                    GroupedConvolution::new(input, kernel, &[1, 1], &[1, 1], &[0, 0], Zero, 2);

                node.forward();
                assert!(node.was_computed());

                node.forward();
                assert!(node.was_computed());

                node.reset_computation();
                assert!(!node.was_computed());

                node.reset_computation();
                assert!(!node.was_computed());
            }
        }

        mod backward {
            use super::*;
            #[test]
            fn creation() {
                let node = ConvolutionBackward::new(
                    new_backward_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
                    new_backward_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
                    new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
                    new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
                    &[1, 1],
                    &[1, 1],
                    &[0, 0],
                    Zero,
                );

                let outshape: ndarray::Ix4 =
                    conv_out_shape(&[4, 4, 6, 6], &[4, 4, 2, 2], &[0, 0], &[1, 1], &[1, 1]);

                assert_eq!(*node.gradient(), Tensor::from_elem(outshape, 0.));
                assert_eq!(*node.gradient_mut(), Tensor::from_elem(outshape, 0.));
                assert!(node.can_overwrite());
            }

            #[test]
            fn computation_state_transition() {
                let input_grad = new_backward_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
                let kernel_grad = new_backward_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);

                let node = ConvolutionBackward::new(
                    input_grad.clone(),
                    kernel_grad.clone(),
                    new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
                    new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
                    &[1, 1],
                    &[1, 1],
                    &[0, 0],
                    Zero,
                );

                node.backward();
                assert!(node.can_overwrite());
                assert!(!input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                node.backward();
                assert!(node.can_overwrite());
                assert!(!input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                input_grad.set_overwrite(true);
                assert!(node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                input_grad.set_overwrite(true);
                assert!(node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                kernel_grad.set_overwrite(true);
                assert!(node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(kernel_grad.can_overwrite());

                kernel_grad.set_overwrite(true);
                assert!(node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(kernel_grad.can_overwrite());

                node.set_overwrite(false);
                assert!(!node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(kernel_grad.can_overwrite());

                node.set_overwrite(false);
                assert!(!node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(kernel_grad.can_overwrite());

                node.backward();
                assert!(!node.can_overwrite());
                assert!(!input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                node.backward();
                assert!(!node.can_overwrite());
                assert!(!input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());
            }
        }

        mod backward_gropued {
            use super::*;
            #[test]
            fn creation() {
                let node = GroupedConvolutionBackward::new(
                    new_backward_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
                    new_backward_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
                    new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
                    new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
                    &[1, 1],
                    &[1, 1],
                    &[0, 0],
                    Zero,
                    2,
                );

                let outshape: ndarray::Ix4 =
                    conv_out_shape(&[4, 4, 6, 6], &[4, 4, 2, 2], &[0, 0], &[1, 1], &[1, 1]);

                assert_eq!(*node.gradient(), Tensor::from_elem(outshape, 0.));
                assert_eq!(*node.gradient_mut(), Tensor::from_elem(outshape, 0.));
                assert!(node.can_overwrite());
            }

            #[test]
            fn computation_state_transition() {
                let input_grad = new_backward_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]);
                let kernel_grad = new_backward_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]);

                let node = GroupedConvolutionBackward::new(
                    input_grad.clone(),
                    kernel_grad.clone(),
                    new_input((4, 4, 6, 6), vec![0.; 4 * 4 * 6 * 6]),
                    new_input((4, 4, 2, 2), vec![0.; 4 * 4 * 2 * 2]),
                    &[1, 1],
                    &[1, 1],
                    &[0, 0],
                    Zero,
                    2,
                );

                node.backward();
                assert!(node.can_overwrite());
                assert!(!input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                node.backward();
                assert!(node.can_overwrite());
                assert!(!input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                input_grad.set_overwrite(true);
                assert!(node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                input_grad.set_overwrite(true);
                assert!(node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                kernel_grad.set_overwrite(true);
                assert!(node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(kernel_grad.can_overwrite());

                kernel_grad.set_overwrite(true);
                assert!(node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(kernel_grad.can_overwrite());

                node.set_overwrite(false);
                assert!(!node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(kernel_grad.can_overwrite());

                node.set_overwrite(false);
                assert!(!node.can_overwrite());
                assert!(input_grad.can_overwrite());
                assert!(kernel_grad.can_overwrite());

                node.backward();
                assert!(!node.can_overwrite());
                assert!(!input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());

                node.backward();
                assert!(!node.can_overwrite());
                assert!(!input_grad.can_overwrite());
                assert!(!kernel_grad.can_overwrite());
            }
        }
    }
}
