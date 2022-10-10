use ndarray::{Array, Dimension, Ix4, Ix5, ShapeBuilder};

use cust::memory::{bytemuck::Zeroable, DeviceBuffer, DeviceCopy};

use cudnn::{DataType, FilterDescriptor, ScalarC, TensorDescriptor};

use crate::cuda::device::Device;

/// An array allocated on a CUDA capable device.
pub struct CuArray<T, D>
where
    T: DeviceCopy,
    D: Dimension,
{
    buffer: DeviceBuffer<T>,
    dim: D,
    strides: D,
    device: Device,
}

impl<T, D> CuArray<T, D>
where
    T: DeviceCopy + Zeroable,
    D: Dimension,
{
    /// Creates a new cuda array with zeroed data.
    ///
    /// # Arguments
    ///
    /// * `size` - number of elements.
    ///
    /// * `dim` - dimension.
    ///
    /// * `device` -  CUDA capable device to create the array on.
    pub(crate) fn zeroed(size: usize, dim: D, device: Device) -> Self {
        let buffer = DeviceBuffer::zeroed(size).unwrap();
        let strides = dim.default_strides();

        Self {
            buffer,
            dim,
            strides,
            device,
        }
    }
}

impl<T, D> CuArray<T, D>
where
    T: DeviceCopy,
    D: Dimension,
{
    /// Creates a new cuda array from a slice and a dimension.
    ///
    /// # Arguments
    ///
    /// * `slice` - host slice containing the data.
    ///
    /// * `dim` - dimension.
    ///
    /// * `device` - CUDA capable device to create the array on.
    pub(crate) fn from_slice(slice: &[T], dim: D, device: Device) -> Self {
        let buffer = DeviceBuffer::from_slice(slice).unwrap();
        let strides = dim.default_strides();

        Self {
            buffer,
            dim,
            strides,
            device,
        }
    }

    /// Returns the dimension of the CUDA array.
    pub(crate) fn dimension(&self) -> D {
        self.dim.clone()
    }

    /// Returns the CUDA capable device associated with the array.
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// Returns an immutable reference to the underlying CUDA device buffer.
    pub(crate) fn buffer(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }

    /// Returns a mutable reference to the underlying CUDA device buffer.
    pub(crate) fn buffer_mut(&mut self) -> &mut DeviceBuffer<T> {
        &mut self.buffer
    }
}

impl<T, D> CuArray<T, D>
where
    T: DeviceCopy + Default,
    D: Dimension,
{
    /// Returns the content of the array as a ndarray array.
    pub fn as_ndarray(&self) -> Array<T, D> {
        let host_vec = self.buffer.as_host_vec().unwrap();

        Array::from_shape_vec(self.dim.clone().strides(self.strides.clone()), host_vec).unwrap()
    }

    /// Creates a new array from a ndarray one.
    ///
    /// # Arguments
    ///
    /// * `array` - a ndarray array.
    ///
    /// * `device` - CUDA capable device to create the array on.
    pub fn from_ndarray(array: &Array<T, D>, device: Device) -> Self {
        Self::from_slice(array.as_slice().unwrap(), array.raw_dim(), device)
    }
}

impl<T, D> CuArray<T, D>
where
    T: DeviceCopy + DataType,
    D: Dimension,
{
    /// Returns a cuDNN tensor descriptor describing this array.
    pub(crate) fn cudnn_tensor_desc(&self) -> TensorDescriptor<T> {
        let ndim = self.dim.ndim();

        if ndim > 5 {
            panic!("Only tensors up to 5 dimensions are supported.")
        }

        // cuDNN accepts only tensors with more than 4 dimensions.
        let mut shape = vec![1; ndim.max(4)];
        for (r, d) in shape.iter_mut().rev().zip(self.dim.slice()) {
            *r = *d as i32;
        }

        // This is equivalent to standard layout.
        TensorDescriptor::new_format(&shape, ScalarC::Nchw).unwrap()
    }
}

impl<T> CuArray<T, Ix4>
where
    T: DeviceCopy + DataType,
{
    /// Returns a cuDNN filter descriptor for a 1-d or 2-d convolution.
    pub(crate) fn cudnn_filter_descriptor(&self) -> FilterDescriptor<T> {
        let mut shape = [0_i32; 4];
        for (s, d) in shape.iter_mut().zip(self.dim.slice()) {
            *s = *d as i32;
        }

        FilterDescriptor::new(&shape, ScalarC::Nchw).unwrap()
    }
}

impl<T> CuArray<T, Ix5>
where
    T: DeviceCopy + DataType,
{
    /// Returns a cuDNN filter descriptor for a 3-d convolution.
    pub(crate) fn cudnn_filter_descriptor(&self) -> FilterDescriptor<T> {
        let mut shape = [0_i32; 5];
        for (s, d) in shape.iter_mut().zip(self.dim.slice()) {
            *s = *d as i32;
        }

        FilterDescriptor::new(&shape, ScalarC::Nchw).unwrap()
    }
}
