//! CUDA Driver API FFI — libloading-based direct binding to nvcuda.dll.

use std::ffi::{c_char, c_void, CString};
use std::sync::Mutex;
use libloading::{Library, Symbol};

type CUresult = i32;
type CUdevice = i32;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;
type CUstream = *mut c_void;
const CUDA_SUCCESS: CUresult = 0;

type CuInitFn = unsafe extern "C" fn(u32) -> CUresult;
type CuDeviceGetFn = unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult;
type CuCtxCreateFn = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
type CuModuleLoadDataFn = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type CuModuleGetFunctionFn = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type CuLaunchKernelFn = unsafe extern "C" fn(
    CUfunction, u32, u32, u32, u32, u32, u32, u32,
    CUstream, *mut *mut c_void, *mut *mut c_void,
) -> CUresult;
type CuMemAllocFn = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type CuMemcpyHtoDFn = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type CuMemcpyDtoHFn = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type CuCtxSynchronizeFn = unsafe extern "C" fn() -> CUresult;
type CuMemFreeFn = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type CuCtxDestroyFn = unsafe extern "C" fn(CUcontext) -> CUresult;

// ═══ Global ══════════════════════════════════════════════

static CUDA: Mutex<Option<CudaDriver>> = Mutex::new(None);

struct CudaDriver {
    _lib: &'static Library,
    _ctx: CUcontext,
    module: CUmodule,
    launch: Symbol<'static, CuLaunchKernelFn>,
    alloc: Symbol<'static, CuMemAllocFn>,
    htod: Symbol<'static, CuMemcpyHtoDFn>,
    dtoh: Symbol<'static, CuMemcpyDtoHFn>,
    sync: Symbol<'static, CuCtxSynchronizeFn>,
    free: Symbol<'static, CuMemFreeFn>,
    get_func_sym: Symbol<'static, CuModuleGetFunctionFn>,
}

unsafe impl Send for CudaDriver {}

// ═══ Public API ══════════════════════════════════════════

pub fn init() -> bool {
    let mut guard = CUDA.lock().unwrap();
    if guard.is_some() { return true; }
    match try_init() {
        Ok(driver) => {
            tracing::info!("CUDA initialized via Driver API");
            *guard = Some(driver);
            true
        }
        Err(e) => {
            tracing::warn!("CUDA init failed: {e}. CPU fallback active.");
            false
        }
    }
}

pub fn available() -> bool { CUDA.lock().unwrap().is_some() }

pub struct GpuBuffer { ptr: CUdeviceptr, len: usize }

impl GpuBuffer {
    pub fn alloc(len: usize, elem_size: usize) -> Result<Self, String> {
        let guard = CUDA.lock().unwrap();
        let d = guard.as_ref().ok_or("CUDA not initialized")?;
        let mut ptr: CUdeviceptr = 0;
        let r = unsafe { (d.alloc)(&mut ptr, len * elem_size) };
        if r != CUDA_SUCCESS { return Err(format!("cuMemAlloc: {r}")); }
        Ok(GpuBuffer { ptr, len })
    }
    pub fn copy_host_to_device<T>(&self, data: &[T]) -> Result<(), String> {
        let guard = CUDA.lock().unwrap();
        let d = guard.as_ref().ok_or("CUDA not initialized")?;
        let r = unsafe { (d.htod)(self.ptr, data.as_ptr() as *const c_void, std::mem::size_of_val(data)) };
        if r != CUDA_SUCCESS { return Err(format!("cuMemcpyHtoD: {r}")); }
        Ok(())
    }
    pub fn copy_device_to_host<T: Clone + Default>(&self, data: &mut [T]) -> Result<(), String> {
        let guard = CUDA.lock().unwrap();
        let d = guard.as_ref().ok_or("CUDA not initialized")?;
        let r = unsafe { (d.dtoh)(data.as_mut_ptr() as *mut c_void, self.ptr, std::mem::size_of_val(data)) };
        if r != CUDA_SUCCESS { return Err(format!("cuMemcpyDtoH: {r}")); }
        Ok(())
    }
    pub fn free(self) -> Result<(), String> {
        let guard = CUDA.lock().unwrap();
        let d = guard.as_ref().ok_or("CUDA not initialized")?;
        let r = unsafe { (d.free)(self.ptr) };
        if r != CUDA_SUCCESS { return Err(format!("cuMemFree: {r}")); }
        Ok(())
    }
}

pub unsafe fn launch_kernel(
    name: &str, grid: (u32, u32, u32), block: (u32, u32, u32),
    args: &[*mut c_void],
) -> Result<(), String> {
    let guard = CUDA.lock().unwrap();
    let d = guard.as_ref().ok_or("CUDA not initialized")?;
    let cn = CString::new(name).map_err(|e| format!("Invalid name: {e}"))?;
    let mut func: CUfunction = std::ptr::null_mut();
    let r = (d.get_func_sym)(&mut func, d.module, cn.as_ptr());
    if r != CUDA_SUCCESS { return Err(format!("cuModuleGetFunction({name}): {r}")); }
    let r = (d.launch)(
        func, grid.0, grid.1, grid.2, block.0, block.1, block.2,
        0, std::ptr::null_mut(), args.as_ptr() as *mut *mut c_void, std::ptr::null_mut(),
    );
    if r != CUDA_SUCCESS { return Err(format!("cuLaunchKernel({name}): {r}")); }
    Ok(())
}

pub fn synchronize() -> Result<(), String> {
    let guard = CUDA.lock().unwrap();
    let d = guard.as_ref().ok_or("CUDA not initialized")?;
    let r = unsafe { (d.sync)() };
    if r != CUDA_SUCCESS { return Err(format!("cuCtxSynchronize: {r}")); }
    Ok(())
}

// ═══ Internal Init ═══════════════════════════════════════

fn try_init() -> Result<CudaDriver, String> {
    // Try common CUDA driver paths
    let cuda_paths = [
        "nvcuda.dll",
        "C:\\Windows\\System32\\nvcuda.dll",
    ];
    let lib = cuda_paths.iter().find_map(|p| unsafe { Library::new(p).ok() })
        .ok_or_else(|| "nvcuda.dll not found in system paths".to_string())?;
    let lib = Box::leak(Box::new(lib));

    macro_rules! sym {
        ($name:expr, $ty:ty) => {
            unsafe { lib.get::<$ty>($name) }
                .map_err(|e| format!("symbol {}: {e}", String::from_utf8_lossy($name)))
        };
    }

    // 1. cuInit
    let cu_init = sym!(b"cuInit\0", CuInitFn)?;
    let r = unsafe { cu_init(0) };
    if r != CUDA_SUCCESS { return Err(format!("cuInit: {r}")); }

    // 2. cuDeviceGet
    let cu_device_get = sym!(b"cuDeviceGet\0", CuDeviceGetFn)?;
    let mut device: CUdevice = 0;
    let r = unsafe { cu_device_get(&mut device, 0) };
    if r != CUDA_SUCCESS { return Err(format!("cuDeviceGet: {r}")); }

    // 3. cuCtxCreate
    let cu_ctx_create = sym!(b"cuCtxCreate_v2\0", CuCtxCreateFn)?;
    let mut ctx: CUcontext = std::ptr::null_mut();
    let r = unsafe { cu_ctx_create(&mut ctx, 0, device) };
    if r != CUDA_SUCCESS { return Err(format!("cuCtxCreate: {r}")); }

    // 4. cuModuleLoadData
    let cu_module_load = sym!(b"cuModuleLoadData\0", CuModuleLoadDataFn)?;
    let ptx = crate::gpu::PTX_BYTES;
    let mut module: CUmodule = std::ptr::null_mut();
    let r = unsafe { cu_module_load(&mut module, ptx.as_ptr() as *const c_void) };
    if r != CUDA_SUCCESS { return Err(format!("cuModuleLoadData: {r}")); }

    // 5. Load all function pointers
    let launch = sym!(b"cuLaunchKernel\0", CuLaunchKernelFn)?;
    let alloc = sym!(b"cuMemAlloc_v2\0", CuMemAllocFn)?;
    let htod = sym!(b"cuMemcpyHtoD_v2\0", CuMemcpyHtoDFn)?;
    let dtoh = sym!(b"cuMemcpyDtoH_v2\0", CuMemcpyDtoHFn)?;
    let sync = sym!(b"cuCtxSynchronize\0", CuCtxSynchronizeFn)?;
    let free = sym!(b"cuMemFree_v2\0", CuMemFreeFn)?;
    let get_func_sym = sym!(b"cuModuleGetFunction\0", CuModuleGetFunctionFn)?;

    Ok(CudaDriver {
        _lib: lib, _ctx: ctx, module,
        launch, alloc, htod, dtoh, sync, free, get_func_sym,
    })
}
