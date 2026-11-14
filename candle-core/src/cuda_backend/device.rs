use crate::backend::BackendDevice;
use crate::{CpuStorage, CpuStorageRef, DType, Layout, Result, Shape};
pub use candle_kernels as kernels;
pub use cudarc;
use cudarc::driver::{
    CudaFunction, CudaSlice, DevicePtrMut, DeviceRepr, LaunchAsync, LaunchConfig,
};
use half::{bf16, f16};
use std::sync::{Arc, Mutex};

use super::{CudaError, CudaStorage, CudaStorageSlice, WrapErr};

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

struct CudaRng(cudarc::curand::CudaRng);
unsafe impl Send for CudaRng {}

/// Persistent cuBLAS workspace allocated with `cuMemAlloc` (outside any memory
/// pool / stream-ordered allocator).
///
/// Without a user-provided workspace, cuBLAS allocates its internal workspace
/// lazily via stream-ordered allocation. When that first happens *during CUDA
/// graph capture*, the workspace becomes a graph-owned memory node whose
/// address is cached in the cuBLAS handle and baked into every subsequently
/// captured graph. With `AUTO_FREE_ON_LAUNCH` graph pools the address is
/// recycled once the owning graph replays, so workspace-writing kernels (e.g.
/// split-K `dot_kernel`) corrupt unrelated tensors on replay.
///
/// Setting a stable workspace up front (per handle/stream) avoids any
/// workspace allocation during capture. This mirrors what PyTorch does for
/// CUDA graph support.
struct BlasWorkspace {
    ptr: cudarc::driver::sys::CUdeviceptr,
    size: usize,
}

// The workspace is a plain device allocation shared by all clones of the
// device; it is only read/written by cuBLAS on the device's own stream.
unsafe impl Send for BlasWorkspace {}
unsafe impl Sync for BlasWorkspace {}

impl BlasWorkspace {
    /// 32 MiB matches the upper range of common cuBLAS workspace configs and
    /// is large enough for split-K reductions used by skinny decode matmuls.
    const SIZE: usize = 32 * 1024 * 1024;

    fn new(device: &cudarc::driver::CudaDevice) -> Result<Self> {
        use cudarc::driver::result;
        device.bind_to_thread().w()?;
        let ptr = unsafe { result::malloc_sync(Self::SIZE) }.w()?;
        Ok(Self {
            ptr,
            size: Self::SIZE,
        })
    }

    fn attach(&self, blas: &cudarc::cublas::CudaBlas) -> Result<()> {
        let status = unsafe {
            cudarc::cublas::sys::lib().cublasSetWorkspace_v2(
                *blas.handle(),
                self.ptr as *mut core::ffi::c_void,
                self.size,
            )
        };
        if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(CudaError::BlasWorkspace(status)).w();
        }
        Ok(())
    }
}

impl Drop for BlasWorkspace {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::result::free_sync(self.ptr);
        }
    }
}

#[derive(Clone)]
pub struct CudaDevice {
    id: DeviceId,
    device: Arc<cudarc::driver::CudaDevice>,
    pub(crate) blas: Arc<cudarc::cublas::CudaBlas>,
    blas_workspace: Arc<BlasWorkspace>,
    curand: Arc<Mutex<CudaRng>>,
}

impl std::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaDevice({:?})", self.id)
    }
}

impl std::ops::Deref for CudaDevice {
    type Target = Arc<cudarc::driver::CudaDevice>;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl CudaDevice {
    pub fn cuda_device(&self) -> Arc<cudarc::driver::CudaDevice> {
        self.device.clone()
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Host→device copy without a device-wide synchronize.
    ///
    /// Safer than `htod_sync_copy` for bulk weight loading: async H2D is ordered
    /// with later work on the same stream. Callers must keep `src` alive until a
    /// subsequent same-stream op or an explicit `synchronize()` completes.
    fn htod_copy_slice_nosync<T: DeviceRepr>(&self, src: &[T]) -> Result<CudaSlice<T>> {
        let mut dst = unsafe { self.alloc(src.len()) }.w()?;
        self.bind_to_thread().w()?;
        let stream = *self.cu_stream();
        let bytes = std::mem::size_of_val(src);
        if let Some(staged) = pinned_stage(stream, src.as_ptr() as *const u8, bytes)? {
            // The staged bytes are a copy of `src` in page-locked memory.
            let staged = unsafe { std::slice::from_raw_parts(staged as *const T, src.len()) };
            unsafe {
                cudarc::driver::result::memcpy_htod_async(*dst.device_ptr_mut(), staged, stream)
            }
            .w()?;
        } else {
            unsafe {
                cudarc::driver::result::memcpy_htod_async(*dst.device_ptr_mut(), src, stream)
            }
            .w()?;
        }
        Ok(dst)
    }

    fn const_impl(&self, v: f64, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let slice = match dtype {
            DType::U8 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<u8>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_u8", kernels::FILL)?;
                let params = (&data, v as u8, elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<u32>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_u32", kernels::FILL)?;
                let params = (&data, v as u32, elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<i64>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_i64", kernels::FILL)?;
                let params = (&data, v as i64, elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<bf16>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_bf16", kernels::FILL)?;
                let params = (&data, bf16::from_f64(v), elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f16>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_f16", kernels::FILL)?;
                let params = (&data, f16::from_f64(v), elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f32>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_f32", kernels::FILL)?;
                let params = (&data, v as f32, elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                // SAFETY: Set later by running the fill kernel.
                let data = unsafe { self.alloc::<f64>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_f64", kernels::FILL)?;
                let params = (&data, v, elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E8M0 => {
                let data = unsafe { self.alloc::<u8>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_u8", kernels::FILL)?;
                let params = (&data, v as u8, elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                CudaStorageSlice::U8(data)
            }
            DType::F8E4M3 => {
                let data = unsafe { self.alloc::<u8>(elem_count) }.w()?;
                let func = self.get_or_load_func("fill_u8", kernels::FILL)?;
                let params = (&data, v as u8, elem_count);
                unsafe { func.launch(cfg, params) }.w()?;
                CudaStorageSlice::U8(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    pub fn get_or_load_func(&self, module_name: &str, ptx: &'static str) -> Result<CudaFunction> {
        if !self.has_func(module_name, module_name) {
            // Leaking the string here is a bit sad but we need a &'static str and this is only
            // done once per kernel name.
            let static_module_name = Box::leak(module_name.to_string().into_boxed_str());
            self.load_ptx(ptx.into(), module_name, &[static_module_name])
                .map_err(|cuda| CudaError::Load {
                    cuda,
                    module_name: module_name.to_string(),
                })
                .w()?;
        }
        self.get_func(module_name, module_name)
            // Clippy recommends this `ok_or` rather than `ok_or_else` so hopefully the compiler is
            // able to only build the error value if needed.
            .ok_or(CudaError::MissingKernel {
                module_name: module_name.to_string(),
            })
            .w()
    }
}

impl CudaDevice {
    pub fn new_with_stream(ordinal: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new_with_stream(ordinal).w()?;
        let blas = cudarc::cublas::CudaBlas::new(device.clone()).w()?;
        let blas_workspace = BlasWorkspace::new(&device)?;
        blas_workspace.attach(&blas)?;
        let curand = cudarc::curand::CudaRng::new(299792458, device.clone()).w()?;
        Ok(Self {
            id: DeviceId::new(),
            device,
            blas: Arc::new(blas),
            blas_workspace: Arc::new(blas_workspace),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
        })
    }
}

impl BackendDevice for CudaDevice {
    type Storage = CudaStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(ordinal).w()?;
        let blas = cudarc::cublas::CudaBlas::new(device.clone()).w()?;
        let blas_workspace = BlasWorkspace::new(&device)?;
        blas_workspace.attach(&blas)?;
        let curand = cudarc::curand::CudaRng::new(299792458, device.clone()).w()?;
        Ok(Self {
            id: DeviceId::new(),
            device,
            blas: Arc::new(blas),
            blas_workspace: Arc::new(blas_workspace),
            curand: Arc::new(Mutex::new(CudaRng(curand))),
        })
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        // We do not call set_seed but instead create a new curand object. This ensures that the
        // state will be identical and the same random numbers will be generated.
        let mut curand = self.curand.lock().unwrap();
        curand.0 = cudarc::curand::CudaRng::new(seed, self.device.clone()).w()?;
        Ok(())
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Cuda {
            gpu_id: self.device.ordinal(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.id == rhs.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType, sync_alloc: bool) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc_zeros::<u8>(elem_count, sync_alloc).w()?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc_zeros::<u32>(elem_count, sync_alloc).w()?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                let data = self.alloc_zeros::<i64>(elem_count, sync_alloc).w()?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc_zeros::<bf16>(elem_count, sync_alloc).w()?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc_zeros::<f16>(elem_count, sync_alloc).w()?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc_zeros::<f32>(elem_count, sync_alloc).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc_zeros::<f64>(elem_count, sync_alloc).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E8M0 => {
                let data = self.alloc_zeros::<u8>(elem_count, sync_alloc).w()?;
                CudaStorageSlice::U8(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc_zeros::<u8>(elem_count, sync_alloc).w()?;
                CudaStorageSlice::U8(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, up: f64) -> Result<CudaStorage> {
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        let slice = match dtype {
            // TODO: Add support for F16 and BF16 though this is likely to require some upstream
            // cudarc changes.
            DType::U8
            | DType::U32
            | DType::I64
            | DType::F16
            | DType::BF16
            | DType::F8E8M0
            | DType::F8E4M3 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_uniform",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count) }.w()?;
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count) }.w()?;
                curand.0.fill_with_uniform(&mut data).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        let slice = if lo == 0. && up == 1.0 {
            slice
        } else {
            use super::utils::Map1;
            let layout = Layout::contiguous(shape);
            super::Affine(up - lo, lo).map(&slice, self, &layout)?
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<CudaStorage> {
        // TODO: Add support for F16 and BF16 though this is likely to require some upstream
        // cudarc changes.
        let elem_count = shape.elem_count();
        let curand = self.curand.lock().unwrap();
        // curand can only generate an odd number of values.
        // https://github.com/huggingface/candle/issues/734
        let elem_count_round = if elem_count % 2 == 1 {
            elem_count + 1
        } else {
            elem_count
        };
        let slice = match dtype {
            DType::U8
            | DType::U32
            | DType::I64
            | DType::F16
            | DType::BF16
            | DType::F8E8M0
            | DType::F8E4M3 => Err(CudaError::UnsupportedDtype {
                dtype,
                op: "rand_normal",
            })
            .w()?,
            DType::F32 => {
                let mut data = unsafe { self.alloc::<f32>(elem_count_round) }.w()?;
                curand
                    .0
                    .fill_with_normal(&mut data, mean as f32, std as f32)
                    .w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let mut data = unsafe { self.alloc::<f64>(elem_count_round) }.w()?;
                curand.0.fill_with_normal(&mut data, mean, std).w()?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<CudaStorage> {
        self.const_impl(1., shape, dtype)
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let slice = match dtype {
            DType::U8 => {
                let data = self.alloc::<u8>(elem_count).w()?;
                CudaStorageSlice::U8(data)
            }
            DType::U32 => {
                let data = self.alloc::<u32>(elem_count).w()?;
                CudaStorageSlice::U32(data)
            }
            DType::I64 => {
                let data = self.alloc::<i64>(elem_count).w()?;
                CudaStorageSlice::I64(data)
            }
            DType::BF16 => {
                let data = self.alloc::<bf16>(elem_count).w()?;
                CudaStorageSlice::BF16(data)
            }
            DType::F16 => {
                let data = self.alloc::<f16>(elem_count).w()?;
                CudaStorageSlice::F16(data)
            }
            DType::F32 => {
                let data = self.alloc::<f32>(elem_count).w()?;
                CudaStorageSlice::F32(data)
            }
            DType::F64 => {
                let data = self.alloc::<f64>(elem_count).w()?;
                CudaStorageSlice::F64(data)
            }
            DType::F8E8M0 => {
                let data = self.alloc::<u8>(elem_count).w()?;
                CudaStorageSlice::U8(data)
            }
            DType::F8E4M3 => {
                let data = self.alloc::<u8>(elem_count).w()?;
                CudaStorageSlice::U8(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_slice<T: crate::WithDType>(&self, s: &[T]) -> Result<Self::Storage> {
        let slice = match T::cpu_storage_ref(s) {
            CpuStorageRef::U8(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorageRef::U32(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorageRef::I64(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorageRef::BF16(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorageRef::F16(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorageRef::F32(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorageRef::F64(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.htod_copy_slice_nosync(storage)?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<CudaStorage> {
        let slice = match storage {
            CpuStorage::U8(storage) => {
                let data = self.htod_copy_slice_nosync(&storage)?;
                CudaStorageSlice::U8(data)
            }
            CpuStorage::U32(storage) => {
                let data = self.htod_copy_slice_nosync(&storage)?;
                CudaStorageSlice::U32(data)
            }
            CpuStorage::I64(storage) => {
                let data = self.htod_copy_slice_nosync(&storage)?;
                CudaStorageSlice::I64(data)
            }
            CpuStorage::BF16(storage) => {
                let data = self.htod_copy_slice_nosync(&storage)?;
                CudaStorageSlice::BF16(data)
            }
            CpuStorage::F16(storage) => {
                let data = self.htod_copy_slice_nosync(&storage)?;
                CudaStorageSlice::F16(data)
            }
            CpuStorage::F32(storage) => {
                let data = self.htod_copy_slice_nosync(&storage)?;
                CudaStorageSlice::F32(data)
            }
            CpuStorage::F64(storage) => {
                let data = self.htod_copy_slice_nosync(&storage)?;
                CudaStorageSlice::F64(data)
            }
        };
        Ok(CudaStorage {
            slice,
            device: self.clone(),
        })
    }

    fn synchronize(&self) -> Result<()> {
        self.device.synchronize().map_err(crate::Error::wrap)?;
        Ok(())
    }
}

/// Pinned staging for small host-to-device copies.
///
/// A copy from pageable memory blocks the host until the DMA runs, and the
/// DMA runs in stream order, so every small upload (token ids, positions,
/// penalty indices) waited for all queued kernels: the host could never get
/// ahead of the GPU. Staging through page-locked memory makes the copy
/// asynchronous. The ring is reused in laps; on each wrap the host waits once
/// for the previous lap's copies (an event recorded on the stream), which at
/// decode rates is once per thousands of steps. Copies above
/// `PINNED_MAX_COPY`, and any issued while the stream is capturing a CUDA
/// graph, keep the pageable path. `CANDLE_PINNED_HTOD=0` turns it off.
const PINNED_RING_BYTES: usize = 32 << 20;
const PINNED_MAX_COPY: usize = 1 << 20;

struct PinnedRing {
    base: usize,
    head: usize,
    wrap_event: cudarc::driver::sys::CUevent,
}

// The ring is only touched under `PINNED_RINGS`'s lock.
unsafe impl Send for PinnedRing {}

static PINNED_RINGS: std::sync::OnceLock<Mutex<std::collections::HashMap<usize, PinnedRing>>> =
    std::sync::OnceLock::new();

/// Copies `bytes` from `src` into the stream's pinned ring and returns the
/// staged host pointer, or `None` when the caller should copy from `src`.
fn pinned_stage(
    stream: cudarc::driver::sys::CUstream,
    src: *const u8,
    bytes: usize,
) -> Result<Option<*const u8>> {
    use cudarc::driver::result;
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| !std::env::var("CANDLE_PINNED_HTOD").is_ok_and(|v| v == "0")) {
        return Ok(None);
    }
    if bytes == 0 || bytes > PINNED_MAX_COPY {
        return Ok(None);
    }
    if cudarc::driver::capture_status(stream)
        != Ok(cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE)
    {
        return Ok(None);
    }
    let rings = PINNED_RINGS.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut rings = rings.lock().unwrap_or_else(|e| e.into_inner());
    let ring = match rings.entry(stream as usize) {
        std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
        std::collections::hash_map::Entry::Vacant(e) => {
            // Portable (flag 1) so the pages stay valid in every context.
            let base = unsafe { result::malloc_host(PINNED_RING_BYTES, 1) }.w()? as usize;
            let wrap_event =
                result::event::create(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING)
                    .w()?;
            e.insert(PinnedRing {
                base,
                head: 0,
                wrap_event,
            })
        }
    };
    let len = (bytes + 255) & !255;
    if ring.head + len > PINNED_RING_BYTES {
        // Everything staged in the previous lap must be read before reuse.
        unsafe {
            result::event::record(ring.wrap_event, stream).w()?;
            result::event::synchronize(ring.wrap_event).w()?;
        }
        ring.head = 0;
    }
    let dst = (ring.base + ring.head) as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(src, dst, bytes) };
    ring.head += len;
    Ok(Some(dst as *const u8))
}
