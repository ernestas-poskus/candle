use crate::{DType, InferenceMode, Result};
use candle_metal_kernels::Kernels;
use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder, MTLResourceOptions, NSUInteger,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};

use super::MetalError;

/// Unique identifier for cuda devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub(crate) fn new() -> Self {
        // https://users.rust-lang.org/t/idiomatic-rust-way-to-generate-unique-id/33805
        use std::sync::atomic;
        static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
        Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
    }
}

type BufferMap = HashMap<(NSUInteger, MTLResourceOptions), Vec<Arc<Buffer>>>;

pub(crate) struct Commands {
    /// Single command queue for the entire device.
    command_queue: CommandQueue,
    /// One command buffer at a time.
    /// The scheduler works by allowing multiple
    /// [ComputeCommandEncoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc)
    /// on a single command buffer. Using a single command buffer would be fastest on the GPU but
    /// prevents overlapping of CPU and GPU commands (because command buffer needs to be committed
    /// to start to work).
    /// Despite what the documentation says, command buffers are NOT ordered. They are ordered
    /// for their START time, but there's no guarantee that command buffer1 will finish before
    /// command buffer2 starts (or there are metal bugs there)
    command_buffer: CommandBuffer,
    /// Keeps track of the current amount of compute command encoders on the current
    /// command buffer
    /// Arc, RwLock because of the interior mutability.
    command_buffer_index: usize,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
    current_encoder: Option<ComputeCommandEncoder>,
}

impl Commands {
    pub(crate) fn new(command_queue: CommandQueue) -> Result<Self> {
        let command_buffer = command_queue.new_command_buffer().to_owned();
        command_buffer.enqueue();
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val.parse()?,
            _ => 512,
        };
        Ok(Self {
            command_queue,
            command_buffer,
            command_buffer_index: 0,
            compute_per_buffer,
            current_encoder: None,
        })
    }

    pub fn command_buffer(&mut self) -> Result<(bool, CommandBuffer)> {
        self.end_current_encoder();
        let mut command_buffer = self.command_buffer.to_owned();
        let mut flushed = false;
        if self.command_buffer_index > self.compute_per_buffer {
            self.command_buffer.commit();
            command_buffer = self.command_queue.new_command_buffer().to_owned();
            self.command_buffer = command_buffer.clone();
            self.command_buffer_index = 0;
            flushed = true;
        }
        self.command_buffer_index += 1;
        Ok((flushed, command_buffer))
    }

    pub fn command_encoder(&mut self) -> Result<bool> {
        let mut flushed = false;
        if self.command_buffer_index > self.compute_per_buffer {
            self.end_current_encoder();
            self.command_buffer.commit();
            self.command_buffer = self.command_queue.new_command_buffer().to_owned();
            self.command_buffer_index = 0;
            flushed = true;
        }
        if self.current_encoder.is_none() {
            self.current_encoder =
                Some(self.command_buffer.new_compute_command_encoder().to_owned());
        }
        self.command_buffer_index += 1;
        Ok(flushed)
    }

    fn end_current_encoder(&mut self) {
        if let Some(encoder) = self.current_encoder.take() {
            encoder.end_encoding();
        }
    }

    pub fn wait_until_completed(&mut self) -> Result<()> {
        self.end_current_encoder();
        match self.command_buffer.status() {
            metal::MTLCommandBufferStatus::NotEnqueued
            | metal::MTLCommandBufferStatus::Enqueued => self.command_buffer.commit(),
            metal::MTLCommandBufferStatus::Committed
            | metal::MTLCommandBufferStatus::Scheduled
            | metal::MTLCommandBufferStatus::Completed => {}
            metal::MTLCommandBufferStatus::Error => {}
        }
        self.command_buffer.wait_until_completed();
        self.command_buffer = self.command_queue.new_command_buffer().to_owned();
        self.command_buffer_index = 0;

        Ok(())
    }
}

impl Drop for Commands {
    /// Metal asserts (`-[_MTLCommandEncoder dealloc]:134: failed assertion
    /// 'Command encoder released without endEncoding'`) if a compute command
    /// encoder is released while still open, which aborts the process. The
    /// encoder here stays open between kernel launches on purpose -- see
    /// `command_encoder` -- so every path that stops using the device has to
    /// close it. `wait_until_completed` does that, but nothing guarantees a
    /// caller reaches it before the last `MetalDevice` clone goes away: a
    /// program that computes a tensor and then returns drops `Commands` with
    /// an encoder still open. Closing it here makes the open encoder an
    /// invariant of the type rather than of the call sequence.
    fn drop(&mut self) {
        self.end_current_encoder();
    }
}

pub struct CommandEncoderGuard<'a> {
    commands: std::sync::RwLockWriteGuard<'a, Commands>,
}

impl AsRef<metal::ComputeCommandEncoderRef> for CommandEncoderGuard<'_> {
    fn as_ref(&self) -> &metal::ComputeCommandEncoderRef {
        self.commands
            .current_encoder
            .as_ref()
            .expect("compute encoder guard has no active encoder")
    }
}

pub struct PersistentEncoder<'a>(&'a metal::ComputeCommandEncoderRef);

impl AsRef<metal::ComputeCommandEncoderRef> for PersistentEncoder<'_> {
    fn as_ref(&self) -> &metal::ComputeCommandEncoderRef {
        self.0
    }
}

impl candle_metal_kernels::utils::EncoderProvider for &CommandEncoderGuard<'_> {
    type Encoder<'a>
        = PersistentEncoder<'a>
    where
        Self: 'a;

    fn encoder(&self) -> Self::Encoder<'_> {
        PersistentEncoder(
            self.commands
                .current_encoder
                .as_ref()
                .expect("compute encoder guard has no active encoder"),
        )
    }
}

impl CommandEncoderGuard<'_> {
    pub fn set_label(&self, label: &str) {
        self.commands
            .current_encoder
            .as_ref()
            .expect("compute encoder guard has no active encoder")
            .set_label(label);
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    /// Unique identifier, the registryID is not sufficient as it identifies the GPU rather than
    /// the device itself.
    pub(crate) id: DeviceId,

    /// Raw metal device: <https://developer.apple.com/documentation/metal/mtldevice?language=objc>
    pub(crate) device: metal::Device,

    pub(crate) commands: Arc<RwLock<Commands>>,

    /// Simple allocator struct.
    /// The buffers are stored in size buckets since ML tends to use similar shapes over and over.
    /// We store the buffers in [`Arc`] because it's much faster than Obj-c internal ref counting
    /// (could be linked to FFI communication overhead).
    ///
    /// Whenever a buffer has a strong_count==1, we can reuse it, it means it was dropped in the
    /// graph calculation, and only we the allocator kept a reference to it, therefore it's free
    /// to be reused. However, in order for this to work, we need to guarantee the order of
    /// operation, so that this buffer is not being used by another kernel at the same time.
    /// Arc is the CPU reference count, it doesn't mean anything on the GPU side of things.
    ///
    /// Whenever we actually allocate a new buffer, we make a full sweep to clean up unused buffers
    /// (strong_count = 1).
    pub(crate) buffers: Arc<RwLock<BufferMap>>,

    /// Simple keeper struct to keep track of the already compiled kernels so we can reuse them.
    /// Heavily used by [`candle_metal_kernels`]
    pub(crate) kernels: Arc<Kernels>,
    /// Seed for random number generation.
    pub(crate) seed: Arc<Mutex<Buffer>>,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.id)
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = metal::DeviceRef;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn compile(
        &self,
        func_name: &'static str,
        kernel: ug::lang::ssa::Kernel,
    ) -> Result<metal::ComputePipelineState> {
        let mut buf = vec![];
        ug_metal::code_gen::gen(&mut buf, func_name, &kernel)?;
        let metal_code = String::from_utf8(buf)?;
        let lib = self
            .device
            .new_library_with_source(&metal_code, &metal::CompileOptions::new())
            .map_err(MetalError::from)?;
        let func = lib
            .get_function(func_name, None)
            .map_err(MetalError::from)?;
        let pl = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(MetalError::from)?;
        Ok(pl)
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn metal_device(&self) -> &metal::Device {
        &self.device
    }

    fn drop_unused_buffers(&self) -> Result<()> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        for subbuffers in buffers.values_mut() {
            let newbuffers = subbuffers
                .iter()
                .filter(|s| Arc::strong_count(*s) > 1)
                .map(Arc::clone)
                .collect();
            *subbuffers = newbuffers;
        }
        Ok(())
    }

    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        let mut commands = self.commands.write().map_err(MetalError::from)?;
        let (flushed, command_buffer) = commands.command_buffer()?;
        if flushed {
            self.drop_unused_buffers()?
        }
        Ok(command_buffer)
    }

    pub fn command_encoder(&self) -> Result<CommandEncoderGuard<'_>> {
        let mut commands = self.commands.write().map_err(MetalError::from)?;
        let flushed = commands.command_encoder()?;
        if flushed {
            self.drop_unused_buffers()?;
        }
        Ok(CommandEncoderGuard { commands })
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        let mut commands = self.commands.write().map_err(MetalError::from)?;
        commands.wait_until_completed()
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    /// Creates a new buffer (not necessarily zeroed).
    /// The buffer is [MTLPrivate](https://developer.apple.com/documentation/metal/mtlstoragemode)
    /// This means the buffer data cannot be read on the CPU directly.
    ///
    /// [`name`] is only used to keep track of the resource origin in case of bugs
    pub fn new_buffer(
        &self,
        element_count: usize,
        dtype: DType,
        name: &str,
    ) -> Result<Arc<Buffer>> {
        let size = (element_count * dtype.size_in_bytes()) as NSUInteger;
        self.allocate_buffer(size, MTLResourceOptions::StorageModePrivate, name)
    }

    /// Creates a new buffer (not necessarily zeroed).
    /// The buffer is [MTLManaged](https://developer.apple.com/documentation/metal/mtlstoragemode)
    /// This means the buffer can be read on the CPU but will require manual
    /// synchronization when the CPU memory is modified
    /// Used as a bridge to gather data back from the GPU
    pub fn new_buffer_managed(&self, size: NSUInteger) -> Result<Arc<Buffer>> {
        self.allocate_buffer(size, MTLResourceOptions::StorageModeManaged, "managed")
    }

    /// Creates a new buffer from data.
    /// The buffer is [MTLManaged](https://developer.apple.com/documentation/metal/mtlstoragemode)
    ///
    /// Does not require synchronization, as [newBufferWithBytes](https://developer.apple.com/documentation/metal/mtldevice/1433429-newbufferwithbytes)
    /// allocates the buffer and copies over the existing data before returning the MTLBuffer.
    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data) as NSUInteger;
        let new_buffer = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            size,
            MTLResourceOptions::StorageModeManaged,
        );
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;

        let subbuffers = buffers
            .entry((size, MTLResourceOptions::StorageModeManaged))
            .or_insert(vec![]);

        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        Ok(new_buffer)
    }

    /// Uploads CPU data into an exact-size private buffer.
    ///
    /// This is intended for long-lived weight buffers where keeping a managed
    /// copy would unnecessarily inflate unified-memory usage.
    pub fn new_private_buffer_with_data<T>(&self, data: &[T], _name: &str) -> Result<Arc<Buffer>> {
        let size = core::mem::size_of_val(data) as NSUInteger;

        let staging = self.device.new_buffer_with_data(
            data.as_ptr().cast(),
            size,
            MTLResourceOptions::StorageModeShared,
        );

        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        let subbuffers = buffers
            .entry((size, MTLResourceOptions::StorageModePrivate))
            .or_insert(vec![]);
        let private = if let Some(existing) = subbuffers
            .iter()
            .find(|buffer| Arc::strong_count(buffer) == 1 && buffer.length() == size)
        {
            existing.clone()
        } else {
            let buffer = Arc::new(
                self.device
                    .new_buffer(size, MTLResourceOptions::StorageModePrivate),
            );
            subbuffers.push(buffer.clone());
            buffer
        };
        drop(buffers);

        let command_buffer = self.command_buffer()?;
        command_buffer.set_label("upload_private");
        let blit = command_buffer.new_blit_command_encoder();
        blit.set_label("upload_private_blit");
        blit.copy_from_buffer(&staging, 0, &private, 0, size);
        blit.end_encoding();

        Ok(private)
    }

    pub fn allocate_zeros(&self, size_in_bytes: usize, shared: bool) -> Result<Arc<Buffer>> {
        let buffer = if shared {
            let new_buffer = self.device.new_buffer(
                size_in_bytes as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );
            Arc::new(new_buffer)
        } else {
            self.allocate_buffer(
                size_in_bytes as NSUInteger,
                MTLResourceOptions::StorageModePrivate,
                "allocate_zeros",
            )?
        };
        let command_buffer = self.command_buffer()?;
        command_buffer.set_label("zeros");
        let blit = command_buffer.new_blit_command_encoder();
        blit.fill_buffer(
            &buffer,
            metal::NSRange {
                location: 0,
                length: buffer.length(),
            },
            0,
        );
        blit.end_encoding();
        Ok(buffer)
    }

    /// The critical allocator algorithm
    fn allocate_buffer(
        &self,
        size: NSUInteger,
        option: MTLResourceOptions,
        _name: &str,
    ) -> Result<Arc<Buffer>> {
        let mut buffers = self.buffers.write().map_err(MetalError::from)?;
        let exact_size = InferenceMode::is_enabled();
        if let Some(b) = find_available_buffer(size, option, &buffers, exact_size) {
            // Cloning also ensures we increment the strong count
            return Ok(b.clone());
        }

        let size = if exact_size { size } else { buf_size(size) };
        let subbuffers = buffers.entry((size, option)).or_insert(vec![]);

        let new_buffer = self.device.new_buffer(size as NSUInteger, option);
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());

        Ok(new_buffer)
    }

    /// Create a metal GPU capture trace on [`path`].
    pub fn capture<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let capture = metal::CaptureManager::shared();
        let descriptor = metal::CaptureDescriptor::new();
        descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_capture_device(self);
        // The [set_output_url] call requires an absolute path so we convert it if needed.
        if path.as_ref().is_absolute() {
            descriptor.set_output_url(path);
        } else {
            let path = std::env::current_dir()?.join(path);
            descriptor.set_output_url(path);
        }

        capture
            .start_capture(&descriptor)
            .map_err(MetalError::from)?;
        Ok(())
    }
}

fn buf_size(size: NSUInteger) -> NSUInteger {
    size.saturating_sub(1).next_power_of_two() as NSUInteger
}

fn find_available_buffer(
    size: NSUInteger,
    option: MTLResourceOptions,
    buffers: &BufferMap,
    exact_size: bool,
) -> Option<Arc<Buffer>> {
    let mut best_buffer: Option<&Arc<Buffer>> = None;
    let mut best_buffer_size: NSUInteger = NSUInteger::MAX;
    for ((buffer_size, buffer_option), subbuffers) in buffers.iter() {
        let size_match = if exact_size {
            *buffer_size == size
        } else {
            *buffer_size >= size
        };
        if size_match && buffer_size < &best_buffer_size && buffer_option == &option {
            for sub in subbuffers {
                if Arc::strong_count(sub) == 1 {
                    best_buffer = Some(sub);
                    best_buffer_size = *buffer_size;
                }
            }
        }
    }
    best_buffer.cloned()
}
