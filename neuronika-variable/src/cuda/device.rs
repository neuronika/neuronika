use std::rc::Rc;

use cust::context::{Context as CudaContext, CurrentContext};

use cudnn::CudnnContext;

use blastoff::CublasContext;

/// Handle to a CUDA capable device.
#[derive(Clone, Debug)]
pub struct Device {
    cuda: Rc<CudaContext>,
    cublas: Rc<CublasContext>,
    cudnn: Rc<CudnnContext>,
}

impl Device {
    /// Creates a handle to a CUDA capable device.
    ///
    /// # Arguments
    ///
    /// `device` - device index to select.
    ///
    /// # Panics
    ///
    /// If the requested device is not found.
    ///
    /// # Examples
    ///
    /// ```
    /// # use neuronika_variable as neuronika;
    /// let device = neuronika::cuda::Device::new(0);
    /// ```
    pub fn new(device: u32) -> Self {
        // Initialize CUDA driver API.
        cust::init(cust::CudaFlags::empty()).unwrap();

        let device = cust::device::Device::get_device(device).unwrap();
        let cuda = CudaContext::new(device).unwrap();

        // Binds the current context to the selected device.
        CurrentContext::set_current(&cuda).unwrap();

        let cublas = CublasContext::new().unwrap();
        let cudnn = CudnnContext::new().unwrap();

        Self {
            cuda: Rc::new(cuda),
            cublas: Rc::new(cublas),
            cudnn: Rc::new(cudnn),
        }
    }

    /// Returns a reference to the cuda context associate with this device.
    pub(crate) fn cuda(&self) -> &CudaContext {
        &self.cuda
    }

    /// Returns a reference to the cuBLAS context associated to this device.
    pub(crate) fn cublas(&self) -> &CublasContext {
        &self.cublas
    }

    /// Returns a reference to the cuDNN context associated to this device.
    pub(crate) fn cudnn(&self) -> &CudnnContext {
        &self.cudnn
    }
}

impl Default for Device {
    /// Creates a handle to the 0-th CUDA device.
    fn default() -> Self {
        Self::new(0)
    }
}
