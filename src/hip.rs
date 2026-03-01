use std::os::raw::{c_void, c_int};
use anyhow::{Result, bail};

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum HipError {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidImage = 200,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    ArrayIsMapped = 207,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    NotMappedAsArray = 212,
    NotMappedAsPointer = 213,
    ECCNotCorrectable = 214,
    UnsupportedLimit = 215,
    ContextAlreadyInUse = 216,
    PeerAccessUnsupported = 217,
    InvalidKernelFile = 300,
    InvalidGraphicsContext = 301,
    SharedObjectSymbolNotFound = 302,
    SharedObjectInitFailed = 303,
    OperatingSystemError = 304,
    InvalidHandle = 400,
    NotFound = 500,
    NotReady = 600,
    IllegalAddress = 700,
    LaunchOutOfResources = 701,
    LaunchTimeOut = 702,
    PeerAccessAlreadyEnabled = 704,
    PeerAccessNotEnabled = 705,
    SetOnActiveProcess = 708,
    ContextIsDestroyed = 709,
    Assert = 710,
    HostMemoryAlreadyRegistered = 712,
    HostMemoryNotRegistered = 713,
    LaunchFailure = 719,
    CooperativeLaunchTooLarge = 720,
    NotSupported = 801,
    Unknown = 999,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum HipMemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4,
}

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Uint3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Uint3 {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GgmlGluOp {
    Reglu = 0,
    Geglu = 1,
    Swiglu = 2,
    SwigluOai = 3,
    GegluErf = 4,
    GegluQuick = 5,
    Count = 6,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GgmlCudaMmFusionArgsDevice {
    pub x_bias: *const c_void,
    pub gate: *const c_void,
    pub gate_bias: *const c_void,
    pub glu_op: u32, // The enum is a uint32_t sized int in C++
    pub padding: u32, // Implicit 32-bit padding safely enforces 8-byte alignment
}

impl Default for GgmlCudaMmFusionArgsDevice {
    fn default() -> Self {
        Self {
            x_bias: std::ptr::null(),
            gate: std::ptr::null(),
            gate_bias: std::ptr::null(),
            glu_op: 0,
            padding: 0,
        }
    }
}

extern "C" {
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError;
    pub fn hipFree(ptr: *mut c_void) -> HipError;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size_bytes: usize, kind: HipMemcpyKind) -> HipError;
    pub fn hipGetDeviceCount(count: *mut c_int) -> HipError;
    pub fn hipSetDevice(device: c_int) -> HipError;
    pub fn hipDeviceSynchronize() -> HipError;
    pub fn hipGetLastError() -> HipError;

    pub fn hipModuleLoad(module: *mut *mut c_void, fname: *const std::os::raw::c_char) -> HipError;
    pub fn hipModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> HipError;
    pub fn hipModuleUnload(module: *mut c_void) -> HipError;
    pub fn hipModuleGetFunction(function: *mut *mut c_void, module: *mut c_void, kname: *const std::os::raw::c_char) -> HipError;
    pub fn hipModuleLaunchKernel(
        f: *mut c_void,
        gridDimX: u32, gridDimY: u32, gridDimZ: u32,
        blockDimX: u32, blockDimY: u32, blockDimZ: u32,
        sharedMemBytes: u32, stream: *mut c_void,
        kernelParams: *mut *mut c_void, extra: *mut *mut c_void
    ) -> HipError;
}

// rocBLAS types and functions for GEMM
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RocblasOperation {
    None = 111,      // No transpose
    Transpose = 112, // Transpose
    Conjugate = 113, // Conjugate transpose
}

type RocblasHandle = *mut c_void;
type RocblasStatus = i32;

extern "C" {
    pub fn rocblas_create_handle(handle: *mut RocblasHandle) -> RocblasStatus;
    pub fn rocblas_destroy_handle(handle: RocblasHandle) -> RocblasStatus;
    pub fn rocblas_set_stream(handle: RocblasHandle, stream: *mut c_void) -> RocblasStatus;
    
    pub fn rocblas_sgemm(
        handle: RocblasHandle,
        transA: RocblasOperation,
        transB: RocblasOperation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        B: *const f32,
        ldb: c_int,
        beta: *const f32,
        C: *mut f32,
        ldc: c_int,
    ) -> RocblasStatus;
}

/// Global rocBLAS handle (initialized once)
static mut ROCBLAS_HANDLE: Option<RocblasHandle> = None;

pub fn get_rocblas_handle() -> Result<RocblasHandle> {
    unsafe {
        if ROCBLAS_HANDLE.is_none() {
            let mut handle: RocblasHandle = std::ptr::null_mut();
            let status = rocblas_create_handle(&mut handle);
            if status != 0 {
                bail!("rocblas_create_handle failed with status: {}", status);
            }
            ROCBLAS_HANDLE = Some(handle);
        }
        Ok(ROCBLAS_HANDLE.unwrap())
    }
}

/// Copy device memory to device memory
pub fn hip_memcpy_dto_d(dst: *mut c_void, src: *const c_void, size: usize) {
    unsafe {
        hipMemcpy(dst, src, size, HipMemcpyKind::DeviceToDevice);
    }
}

pub struct HipModule {
    module: *mut c_void,
}

impl HipModule {
    pub fn load_data(data: &[u8]) -> Result<Self> {
        let mut module = std::ptr::null_mut();
        let err = unsafe { hipModuleLoadData(&mut module, data.as_ptr() as *const _) };
        if err != HipError::Success {
            bail!("hipModuleLoadData failed with error: {:?}", err);
        }
        Ok(Self { module })
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut module = std::ptr::null_mut();
        let c_path = std::ffi::CString::new(path)?;
        let err = unsafe { hipModuleLoad(&mut module, c_path.as_ptr()) };
        if err != HipError::Success {
            bail!("hipModuleLoad failed with error: {:?}", err);
        }
        Ok(Self { module })
    }

    pub fn get_function(&self, name: &str) -> Result<HipFunction> {
        let mut function = std::ptr::null_mut();
        let c_name = std::ffi::CString::new(name)?;
        let err = unsafe { hipModuleGetFunction(&mut function, self.module, c_name.as_ptr()) };
        if err != HipError::Success {
            bail!("hipModuleGetFunction failed with error: {:?}", err);
        }
        Ok(HipFunction { function })
    }
}

impl Drop for HipModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe { hipModuleUnload(self.module); }
        }
    }
}

pub struct HipFunction {
    function: *mut c_void,
}

impl HipFunction {
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        params: &[*mut c_void],
    ) -> Result<()> {
        let err = hipModuleLaunchKernel(
            self.function,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            0, std::ptr::null_mut(),
            params.as_ptr() as *mut *mut c_void,
            std::ptr::null_mut(),
        );
        if err != HipError::Success {
            // Get additional error info
            let last_err = hipGetLastError();
            bail!(
                "hipModuleLaunchKernel failed: {:?} (last error: {:?})\n\
                 Grid: ({}, {}, {}), Block: ({}, {}, {}), Params: {} args",
                err, last_err,
                grid.0, grid.1, grid.2,
                block.0, block.1, block.2,
                params.len()
            );
        }
        Ok(())
    }
    
    pub unsafe fn launch_with_shared(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &[*mut c_void],
    ) -> Result<()> {
        let err = hipModuleLaunchKernel(
            self.function,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            shared_mem_bytes, std::ptr::null_mut(),
            params.as_ptr() as *mut *mut c_void,
            std::ptr::null_mut(),
        );
        if err != HipError::Success {
            let last_err = hipGetLastError();
            bail!(
                "hipModuleLaunchKernel failed: {:?} (last error: {:?})\n\
                 Grid: ({}, {}, {}), Block: ({}, {}, {}), Shared: {} bytes, Params: {} args",
                err, last_err,
                grid.0, grid.1, grid.2,
                block.0, block.1, block.2,
                shared_mem_bytes, params.len()
            );
        }
        Ok(())
    }
}

pub struct DevicePtr {
    ptr: *mut c_void,
    size: usize,
}

impl DevicePtr {
    pub fn alloc(size: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let err = unsafe { hipMalloc(&mut ptr, size) };
        if err != HipError::Success {
            bail!("hipMalloc failed with error: {:?}", err);
        }
        Ok(Self { ptr, size })
    }

    pub fn copy_from_host(&self, host_ptr: *const c_void, size: usize) -> Result<()> {
        if size > self.size {
            bail!("Copy size exceeds allocated device memory");
        }
        let err = unsafe { hipMemcpy(self.ptr, host_ptr, size, HipMemcpyKind::HostToDevice) };
        if err != HipError::Success {
            bail!("hipMemcpy (H2D) failed with error: {:?}", err);
        }
        Ok(())
    }

    pub fn copy_to_host(&self, host_ptr: *mut c_void, size: usize) -> Result<()> {
        if size > self.size {
            bail!("Copy size exceeds allocated device memory");
        }
        let err = unsafe { hipMemcpy(host_ptr, self.ptr, size, HipMemcpyKind::DeviceToHost) };
        if err != HipError::Success {
            bail!("hipMemcpy (D2H) failed with error: {:?}", err);
        }
        Ok(())
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for DevicePtr {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                hipFree(self.ptr);
            }
        }
    }
}

pub fn init_gpu() -> Result<()> {
    let mut count = 0;
    let err = unsafe { hipGetDeviceCount(&mut count) };
    if err != HipError::Success {
        bail!("hipGetDeviceCount failed: {:?}", err);
    }
    if count == 0 {
        bail!("No HIP devices found");
    }
    let err = unsafe { hipSetDevice(0) };
    if err != HipError::Success {
        bail!("hipSetDevice(0) failed: {:?}", err);
    }
    // println!("GPU initialized: found {} device(s)", count);
    Ok(())
}
