#[cfg(test)]
use super::{new_backward_input, new_input};
use crate::variable::{
    expect_tensor, expect_tensor_mut, Backward, Cache, Data as NData, Forward, Gradient, Overwrite,
    Tensor, Var, VarDiff,
};
use ndarray::{Dimension, RemoveAxis};
use std::{
    cell::{Cell, Ref, RefCell, RefMut},
    fmt::{Debug, Display},
    rc::Rc,
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Padding Modes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mod padding;
pub use padding::{Constant, PaddingMode, Reflective, Replicative, Zero};
use padding::{ReflPad, ReplPad};

mod numeric;
use numeric::{
    check_conv_args, check_groups_args, conv_out_shape, convolution, convolution_backward_input,
    convolution_backward_kernel, convolution_with_groups, convolution_with_groups_backward,
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

impl<F1: ?Sized, F2: ?Sized, Pad> Convolve<Self, Var<F2>, Pad> for Var<F1>
where
    F1: NData + 'static,
    F1::Dim: RemoveAxis,
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

impl<F1: ?Sized, F2: ?Sized, B2: ?Sized, Pad> Convolve<Self, VarDiff<F2, B2>, Pad> for Var<F1>
where
    F1: NData + 'static,
    F1::Dim: RemoveAxis,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    F2: NData<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = F2::Dim>,
    Pad: PaddingMode + 'static,
{
    type Output = VarDiff<Convolution<F1, F2, Pad>, ConvolutionBackwardUnary<F1, B2, Pad>>;

    fn convolve(
        input: Self,
        kernel: VarDiff<F2, B2>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
    ) -> Self::Output {
        let node = ConvolutionBackwardUnary::new(
            kernel.node,
            input.node.clone(),
            kernel.var.node.clone(),
            stride,
            dilation,
            padding,
            padding_mode,
        );
        VarDiff::from(
            node,
            kernel.past,
            Var::convolve(input, kernel.var, stride, dilation, padding, padding_mode),
        )
    }
}

impl<F1: ?Sized, B1: ?Sized, F2: ?Sized, B2: ?Sized, Pad> Convolve<Self, VarDiff<F2, B2>, Pad>
    for VarDiff<F1, B1>
where
    F1: NData + 'static,
    F1::Dim: RemoveAxis,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    B1: Gradient<Dim = F1::Dim> + Overwrite,
    F2: NData<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = F2::Dim>,
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
            padding_mode,
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

impl<F1: ?Sized, F2: ?Sized, Pad> ConvolveWithGroups<Self, Var<F2>, Pad> for Var<F1>
where
    F1: NData + 'static,
    F1::Dim: RemoveAxis,
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

impl<F1: ?Sized, F2: ?Sized, B2: ?Sized, Pad> ConvolveWithGroups<Self, VarDiff<F2, B2>, Pad>
    for Var<F1>
where
    F1: NData + 'static,
    F1::Dim: RemoveAxis,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    F2: NData<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = F2::Dim>,
    Pad: PaddingMode + 'static,
{
    type Output =
        VarDiff<GroupedConvolution<F1, F2, Pad>, GroupedConvolutionBackwardUnary<F1, B2, Pad>>;

    fn convolve_with_groups(
        input: Self,
        kernel: VarDiff<F2, B2>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
        groups: usize,
    ) -> Self::Output {
        let node = GroupedConvolutionBackwardUnary::new(
            kernel.node,
            input.node.clone(),
            kernel.var.node.clone(),
            stride,
            dilation,
            padding,
            padding_mode,
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

impl<F1: ?Sized, B1: ?Sized, F2: ?Sized, B2: ?Sized, Pad>
    ConvolveWithGroups<Self, VarDiff<F2, B2>, Pad> for VarDiff<F1, B1>
where
    F1: NData + 'static,
    F1::Dim: RemoveAxis,
    <F1::Dim as Dimension>::Smaller: RemoveAxis,
    <<F1::Dim as Dimension>::Smaller as Dimension>::Smaller: ReflPad + ReplPad,
    B1: Gradient<Dim = F1::Dim> + Overwrite,
    F2: NData<Dim = F1::Dim> + 'static,
    B2: Gradient<Dim = F2::Dim>,
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
            padding_mode,
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Convolution<Inp: ?Sized, Ker: ?Sized, Pad>
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

impl<Inp: ?Sized, Ker: ?Sized, Pad> Convolution<Inp, Ker, Pad>
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
            computed: Cell::new(false),
        }
    }
}

impl<Inp: ?Sized, Ker: ?Sized, Pad> NData for Convolution<Inp, Ker, Pad>
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

impl<Inp: ?Sized, Ker: ?Sized, Pad> Cache for Convolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<Inp: ?Sized, Ker: ?Sized, Pad> Forward for Convolution<Inp, Ker, Pad>
where
    Inp: NData,
    Inp::Dim: RemoveAxis,
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
}

impl<Inp: ?Sized, Ker: ?Sized, Pad> Debug for Convolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Convolution")
            .field("data", &self.data.borrow())
            .field("stride", &self.stride)
            .field("dilation", &self.dilation)
            .field("padding", &self.padding)
            .field("padding_mode", &self.padding_mode)
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<Inp: ?Sized, Ker: ?Sized, Pad> Display for Convolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GroupedConvolution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct GroupedConvolution<Inp: ?Sized, Ker: ?Sized, Pad>
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

impl<Inp: ?Sized, Ker: ?Sized, Pad> GroupedConvolution<Inp, Ker, Pad>
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

impl<Inp: ?Sized, Ker: ?Sized, Pad> Cache for GroupedConvolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    fn was_computed(&self) -> bool {
        self.computed.get()
    }

    fn reset_computation(&self) {
        self.computed.set(false);
    }
}

impl<Inp: ?Sized, Ker: ?Sized, Pad> NData for GroupedConvolution<Inp, Ker, Pad>
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

impl<Inp: ?Sized, Ker: ?Sized, Pad> Forward for GroupedConvolution<Inp, Ker, Pad>
where
    Inp: NData,
    Inp::Dim: RemoveAxis,
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
}

impl<Inp: ?Sized, Ker: ?Sized, Pad> Debug for GroupedConvolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedConvolution")
            .field("data", &self.data.borrow())
            .field("stride", &self.stride)
            .field("dilation", &self.dilation)
            .field("padding", &self.padding)
            .field("padding_mode", &self.padding_mode)
            .field("groups", &self.groups)
            .field("computed", &self.computed.get())
            .finish()
    }
}

impl<Inp: ?Sized, Ker: ?Sized, Pad> Display for GroupedConvolution<Inp, Ker, Pad>
where
    Inp: NData,
    Ker: NData<Dim = Inp::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}", &self.data.borrow())
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Convolution Backward Structs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConvolutionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct ConvolutionBackward<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad: ?Sized>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
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

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad>
    ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
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
            padding,
            stride,
            dilation,
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

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Gradient
    for ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
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

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Overwrite
    for ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
    Pad: PaddingMode,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Backward
    for ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpD::Dim: RemoveAxis,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
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
        convolution_backward_input(
            &mut *input_grad,
            &*gradient,
            &*kernel,
            padding,
            stride,
            dilation,
            overwrite_input_grad,
        );
        if padding.iter().all(|pad| *pad == 0) {
            convolution_backward_kernel(
                &mut *kernel_grad,
                &*gradient,
                &input,
                stride,
                dilation,
                overwrite_kernel_grad,
            );
        } else {
            let padded_input = pad(&input, padding, padding_mode);
            convolution_backward_kernel(
                &mut *kernel_grad,
                &*gradient,
                &padded_input,
                stride,
                dilation,
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

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Debug
    for ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvolutionBackward")
            .field("gradient", &self.gradient.borrow())
            .field("stride", &self.stride)
            .field("dilation", &self.dilation)
            .field("padding", &self.padding)
            .field("padding_mode", &self.padding_mode)
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Display
    for ConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConvolutionBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct ConvolutionBackwardUnary<InpD: ?Sized, KerG: ?Sized, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
{
    kernel_grad: Rc<KerG>,
    gradient: RefCell<Option<Tensor<KerG::Dim>>>,
    input: Rc<InpD>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    padding_mode: Pad,
    shape: InpD::Dim,
    overwrite: Cell<bool>,
}

impl<InpD: ?Sized, KerG: ?Sized, Pad> ConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
{
    pub fn new<KerD: ?Sized>(
        kernel_grad: Rc<KerG>,
        input: Rc<InpD>,
        kernel: Rc<KerD>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
    ) -> Self
    where
        KerD: NData<Dim = KerG::Dim>,
    {
        let shape: InpD::Dim = conv_out_shape(
            input.data().shape(),
            kernel.data().shape(),
            padding,
            stride,
            dilation,
        );
        let gradient = RefCell::new(Some(Tensor::zeros(shape.clone())));
        let (stride, dilation, padding) = (stride.to_vec(), dilation.to_vec(), padding.to_vec());

        Self {
            kernel_grad,
            gradient,
            shape,
            input,
            stride,
            dilation,
            padding,
            padding_mode,
            overwrite: Cell::new(true),
        }
    }
}

impl<InpD: ?Sized, KerG: ?Sized, Pad> Gradient for ConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
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

impl<InpD: ?Sized, KerG: ?Sized, Pad> Overwrite for ConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<InpD: ?Sized, KerG: ?Sized, Pad> Backward for ConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    InpD::Dim: RemoveAxis,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
    <<InpD as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<InpD as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    fn backward(&self) {
        let gradient = self.gradient();

        let (mut kernel_grad, input, padding, padding_mode, stride, dilation) = (
            self.kernel_grad.gradient_mut(),
            self.input.data(),
            &self.padding,
            &self.padding_mode,
            &self.stride,
            &self.dilation,
        );
        let overwrite_kernel_grad = self.kernel_grad.can_overwrite();

        if padding.iter().all(|pad| *pad == 0) {
            convolution_backward_kernel(
                &mut *kernel_grad,
                &*gradient,
                &*input,
                stride,
                dilation,
                overwrite_kernel_grad,
            );
        } else {
            let padded_input = pad(&input, padding, padding_mode);
            convolution_backward_kernel(
                &mut *kernel_grad,
                &*gradient,
                &padded_input,
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

impl<InpD: ?Sized, KerG: ?Sized, Pad> Debug for ConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvolutionBackwardUnary")
            .field("gradient", &self.gradient.borrow())
            .field("stride", &self.stride)
            .field("dilation", &self.dilation)
            .field("padding", &self.padding)
            .field("padding_mode", &self.padding_mode)
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<InpD: ?Sized, KerG: ?Sized, Pad> Display for ConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GroupedConvolutionBackward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct GroupedConvolutionBackward<
    InpD: ?Sized,
    InpG: ?Sized,
    KerD: ?Sized,
    KerG: ?Sized,
    Pad: ?Sized,
> where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
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

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad>
    GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
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
            padding,
            stride,
            dilation,
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

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Gradient
    for GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
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

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Overwrite
    for GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
    Pad: PaddingMode,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Backward
    for GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpD::Dim: RemoveAxis,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
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

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Debug
    for GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedConvolutionBackward")
            .field("gradient", &self.gradient.borrow())
            .field("stride", &self.stride)
            .field("dilation", &self.dilation)
            .field("padding", &self.padding)
            .field("padding_mode", &self.padding_mode)
            .field("groups", &self.groups)
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<InpD: ?Sized, InpG: ?Sized, KerD: ?Sized, KerG: ?Sized, Pad> Display
    for GroupedConvolutionBackward<InpD, InpG, KerD, KerG, Pad>
where
    InpD: NData,
    InpG: Gradient<Dim = InpD::Dim>,
    KerD: NData<Dim = InpD::Dim>,
    KerG: Gradient<Dim = KerD::Dim>,
    Pad: PaddingMode,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GroupedConvolutionBackwardUnary ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct GroupedConvolutionBackwardUnary<InpD: ?Sized, KerG: ?Sized, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
{
    kernel_grad: Rc<KerG>,
    gradient: RefCell<Option<Tensor<KerG::Dim>>>,
    input: Rc<InpD>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    padding: Vec<usize>,
    padding_mode: Pad,
    groups: usize,
    shape: InpD::Dim,
    overwrite: Cell<bool>,
}

impl<InpD: ?Sized, KerG: ?Sized, Pad> GroupedConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new<KerD: ?Sized>(
        kernel_grad: Rc<KerG>,
        input: Rc<InpD>,
        kernel: Rc<KerD>,
        stride: &[usize],
        dilation: &[usize],
        padding: &[usize],
        padding_mode: Pad,
        groups: usize,
    ) -> Self
    where
        KerD: NData,
    {
        let shape: InpD::Dim = conv_out_shape(
            input.data().shape(),
            kernel.data().shape(),
            padding,
            stride,
            dilation,
        );
        let gradient = RefCell::new(Some(Tensor::zeros(shape.clone())));
        let (stride, dilation, padding) = (stride.to_vec(), dilation.to_vec(), padding.to_vec());

        Self {
            kernel_grad,
            gradient,
            shape,
            input,
            stride,
            dilation,
            padding,
            padding_mode,
            groups,
            overwrite: Cell::new(true),
        }
    }
}

impl<InpD: ?Sized, KerG: ?Sized, Pad> Gradient for GroupedConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
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

impl<InpD: ?Sized, KerG: ?Sized, Pad> Overwrite for GroupedConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
{
    fn can_overwrite(&self) -> bool {
        self.overwrite.get()
    }

    fn set_overwrite(&self, state: bool) {
        self.overwrite.set(state);
    }
}

impl<InpD: ?Sized, KerG: ?Sized, Pad> Backward for GroupedConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    InpD::Dim: RemoveAxis,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
    <<InpD as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<InpD as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    fn backward(&self) {
        let gradient = self.gradient();

        let (mut kernel_grad, input, padding, padding_mode, stride, dilation, groups) = (
            self.kernel_grad.gradient_mut(),
            self.input.data(),
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

impl<InpD: ?Sized, KerG: ?Sized, Pad> Debug for GroupedConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    InpD::Dim: RemoveAxis,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
    <<InpD as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<InpD as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedConvolutionBackwardUnary")
            .field("gradient", &self.gradient.borrow())
            .field("stride", &self.stride)
            .field("dilation", &self.dilation)
            .field("padding", &self.padding)
            .field("padding_mode", &self.padding_mode)
            .field("groups", &self.groups)
            .field("overwrite", &self.overwrite.get())
            .finish()
    }
}

impl<InpD: ?Sized, KerG: ?Sized, Pad> Display for GroupedConvolutionBackwardUnary<InpD, KerG, Pad>
where
    InpD: NData,
    InpD::Dim: RemoveAxis,
    KerG: Gradient<Dim = InpD::Dim>,
    Pad: PaddingMode,
    <<InpD as NData>::Dim as Dimension>::Smaller: RemoveAxis,
    <<<InpD as NData>::Dim as Dimension>::Smaller as Dimension>::Smaller: ReplPad + ReflPad,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &*self.gradient.borrow() {
            Some(gradient) => write!(f, "{}", &gradient),
            None => write!(f, "None"),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tests ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[cfg(test)]
mod test;
