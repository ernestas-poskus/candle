use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, cuda_backend::WrapErr};
use crate::{CudaDevice, CudaStorage, Result};
use half::f16;

use cudarc::driver::{CudaSlice, CudaView, DeviceSlice};

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

/// Software-pipelined `..._pf_cuda*` q4_K matvecs (moss rtf Phase C1) are
/// ON by default after the shipped A/B (RTX 3090 350 W, same binary:
/// −5.1% wall at 8 concurrent sequences, neutral at 4; bit-exact
/// checksums and fused-GLU equivalence green).
/// `MOSS_CANDLE_MMVQ_PREFETCH=0` restores the standard kernels for A/Bs.
fn mmvq_prefetch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("MOSS_CANDLE_MMVQ_PREFETCH").is_ok_and(|v| v == "0"))
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_0(
    src: &CudaSlice<f32>,
    dst: &CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    use cudarc::driver::LaunchAsync;
    use cudarc::driver::LaunchConfig;
    // N is `ky`, K is `k`
    let total_rows = ky; // This will be the grid_dim.y

    // padded K
    let kx_padded = pad(k, GgmlDType::Q8_0.block_size());

    // blocks per row
    let num_blocks_per_row = ceil_div(kx_padded, GgmlDType::Q8_0.block_size()); // This will be grid_dim.x
    let func = dev.get_or_load_func("quantize_q8_0", candle_kernels::QUANTIZED)?;

    // The launch configuration
    let cfg = LaunchConfig {
        // grid_dim: (blocks_per_row, total_rows, 1)
        grid_dim: (num_blocks_per_row as u32, total_rows as u32, 1),
        // block_dim: (threads_per_block, 1, 1)
        block_dim: (GgmlDType::Q8_0.block_size() as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let params = (src, dst, k as i32, kx_padded as i32);
    unsafe { func.launch(cfg, params) }.w()?;

    Ok(())
}

fn quantize_q8_1(
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    use cudarc::driver::LaunchAsync;
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let params = (&src_chunk, &dst_chunk, k as i32, kx_padded as i32);
        unsafe { func.clone().launch(cfg, params) }.w()?;
        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use cudarc::driver::LaunchAsync;

    let nb = (elem_count + 255) / 256;
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        GgmlDType::IQ2_XXS => ("dequantize_block_iq2_xxs_f32", true, 32, nb),
        GgmlDType::IQ2_XS => ("dequantize_block_iq2_xs_f32", true, 32, nb),
        GgmlDType::IQ3_XXS => ("dequantize_block_iq3_xxs_f32", true, 32, nb),
        GgmlDType::IQ4_XS => ("dequantize_block_iq4_xs_f32", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count).w()? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let params = (&data.inner, &dst);
        unsafe { func.launch(cfg, params) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let params = (&data.inner, &dst, nb32 as i32);
        unsafe { func.launch(cfg, params) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use cudarc::driver::LaunchAsync;

    let nb = (elem_count + 255) / 256;
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        GgmlDType::IQ2_XXS => ("dequantize_block_iq2_xxs_f16", true, 32, nb),
        GgmlDType::IQ2_XS => ("dequantize_block_iq2_xs_f16", true, 32, nb),
        GgmlDType::IQ3_XXS => ("dequantize_block_iq3_xxs_f16", true, 32, nb),
        GgmlDType::IQ4_XS => ("dequantize_block_iq4_xs_f16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count).w()? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let params = (&data.inner, &dst);
        unsafe { func.launch(cfg, params) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let params = (&data.inner, &dst, nb32 as i32);
        unsafe { func.launch(cfg, params) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaSlice<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use cudarc::driver::LaunchAsync;

    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows).w()? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let params = (&data.inner, y, &dst, ncols as i32, nrows as i32);
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaSlice<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use cudarc::driver::LaunchAsync;

    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes).w()? };
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    // Phase C1 (moss rtf): software-pipelined q4_K variant — x-side loads
    // register-double-buffered one iteration ahead. Same math, same launch
    // geometry; opt-in for same-binary A/Bs. ncols>=5 only: measured
    // -8.6%/launch at ncols=8 but +7% at ncols=4 (the win scales with the
    // per-thread serial chain length; at short chains the extra register
    // pressure dominates).
    let kernel_name = if dtype == GgmlDType::Q4K && b_size >= 5 && mmvq_prefetch_enabled() {
        format!("mul_mat_vec_q4_K_q8_1_pf_cuda{b_size}")
    } else {
        format!("{kernel_name}{b_size}")
    };
    let func = dev.get_or_load_func(&kernel_name, candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows * b_size).w()? };
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32 + 1) / 2, 4),
        5..=8 => ((nrows as u32 + 1) / 2, 4),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let params = (
        &data.inner,
        &y_q8_1,
        &dst,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32,
    );
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Fused SwiGLU matvec: `silu(gate·y) * (up·y)` with the activation
/// quantized to q8_1 once for both weight matrices, SiLU and the product
/// applied in the kernel epilogue (`mul_mat_vec_q4_K_q8_1_glu_cuda*`).
/// Q4K-only by design — callers must check dtype and fall back to the
/// unfused path otherwise.
#[allow(clippy::too_many_arguments)]
fn mul_mat_vec_glu_via_q8_1(
    gate: &PaddedCudaSlice,
    up: &PaddedCudaSlice,
    y: &CudaSlice<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use cudarc::driver::LaunchAsync;

    if dtype != GgmlDType::Q4K {
        crate::bail!("fused glu matvec only supports q4_K, got {dtype:?}")
    }
    for (name, data) in [("gate", gate), ("up", up)] {
        let data_elems = data.len / dtype.type_size() * dtype.block_size();
        if data_elems < ncols * nrows {
            crate::bail!("unexpected {name} size {}, ncols {ncols} {nrows}", data_elems)
        }
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} b_size {b_size}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Quantize y once, shared by both matvecs.
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes).w()? };
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    // GLU chains are 2x longer per iteration (gate + up computes), so the
    // prefetch variant applies at every batch width when enabled.
    let kernel_name = if mmvq_prefetch_enabled() {
        format!("mul_mat_vec_q4_K_q8_1_glu_pf_cuda{b_size}")
    } else {
        format!("mul_mat_vec_q4_K_q8_1_glu_cuda{b_size}")
    };
    let func = dev.get_or_load_func(&kernel_name, candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows * b_size).w()? };
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32 + 1) / 2, 4),
        5..=8 => ((nrows as u32 + 1) / 2, 4),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let params = (
        &gate.inner,
        &up.inner,
        &y_q8_1,
        &dst,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32,
    );
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaSlice<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use cudarc::driver::LaunchAsync;

    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes).w()? };
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, candle_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(x_rows * y_cols).w()? };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };

    let params = (
        /* vx */ &data.inner,
        /* vy */ &y_q8_1,
        /* dst */ &dst,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32,
    );
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn quantize_q4_k(
    src: &CudaSlice<f32>,
    dst: &CudaSlice<u8>,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<()> {
    use cudarc::driver::LaunchAsync;

    let kx = elem_count;
    let kx_padded = pad(kx, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);
    let func = dev.get_or_load_func("quantize_q4_k", candle_kernels::QUANTIZED)?;
    let blocks_per_grid = (num_blocks + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (blocks_per_grid as u32, 1, 1),
        block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let params = (src, dst, num_blocks);
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(())
}

fn quantize_q6_k(
    src: &CudaSlice<f32>,
    dst: &CudaSlice<u8>,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<()> {
    use cudarc::driver::LaunchAsync;

    assert!(
        elem_count % CUDA_QUANTIZE_BLOCK_SIZE == 0,
        "invalid tensor for quantization!"
    );
    let nb_k = elem_count / CUDA_QUANTIZE_BLOCK_SIZE; // Number of blocks along the k-dimension
    let func = dev.get_or_load_func("quantize_q6_k", candle_kernels::QUANTIZED)?;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nb_k as u32, 1, 1),
        block_dim: (64 as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let params = (src, dst, 1, elem_count);
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(())
}

impl QCudaStorage {
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes, false).w()?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
        })
    }

    pub(crate) fn from_cpu_data(
        device: &CudaDevice,
        elem_count: usize,
        dtype: GgmlDType,
        data: &[u8],
    ) -> Result<Self> {
        let size_in_bytes = ceil_div(elem_count, dtype.block_size()) * dtype.type_size();
        if data.len() != size_in_bytes {
            crate::bail!(
                "invalid quantized cuda upload size: got {}, expected {}",
                data.len(),
                size_in_bytes
            )
        }
        let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
        let mut inner = unsafe { device.alloc::<u8>(padded_len).w()? };
        device
            .htod_sync_copy_into(data, &mut inner.slice_mut(..data.len()))
            .w()?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: data.len(),
            },
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        use cudarc::driver::DevicePtr;
        Ok(*self.data.inner.device_ptr() as *const u8)
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) -> Result<()> {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
                | GgmlDType::IQ2_XXS
                | GgmlDType::IQ2_XS
                | GgmlDType::IQ3_XXS
                | GgmlDType::IQ4_XS
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .dtoh_sync_copy(&self.data.inner.slice(..self.data.len))
            .w()?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out)?,
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out)?,
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out)?,
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out)?,
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out)?,
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out)?,
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out)?,
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out)?,
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out)?,
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out)?,
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out)?,
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out)?,
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out)?,
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out)?,
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out)?,
            GgmlDType::IQ2_XXS => {
                deq::<crate::quantized::BlockIQ2XXS>(&buffer, block_len, &mut out)?
            }
            GgmlDType::IQ2_XS => deq::<crate::quantized::BlockIQ2XS>(&buffer, block_len, &mut out)?,
            GgmlDType::IQ3_XXS => {
                deq::<crate::quantized::BlockIQ3XXS>(&buffer, block_len, &mut out)?
            }
            GgmlDType::IQ4_XS => deq::<crate::quantized::BlockIQ4XS>(&buffer, block_len, &mut out)?,
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => {
                //dedicated gpu kernel for q4k quantization
                if self.dtype == GgmlDType::Q4K {
                    return quantize_q4_k(&data, &self.data.inner, data.len(), &self.device);
                } else if self.dtype == GgmlDType::Q6K {
                    return quantize_q6_k(&data, &self.data.inner, data.len(), &self.device);
                } else if self.dtype == GgmlDType::Q8_0 {
                    return quantize_q8_0(&data, &self.data.inner, data.len(), 1, &self.device);
                } else if self.dtype == GgmlDType::Q8_1 {
                    return quantize_q8_1(&data, &mut self.data.inner, data.len(), 1, &self.device);
                }
                self.device.dtoh_sync_copy(data).w()?
            }
            _ => crate::bail!("only f32 can be quantized"),
        };
        // Run the quantization on cpu.
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len).w()? };
        self.device
            .htod_sync_copy_into(data.as_ref(), &mut inner.slice_mut(..data.len()))
            .w()?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let is_iq = matches!(
            self.dtype,
            GgmlDType::IQ2_XXS | GgmlDType::IQ2_XS | GgmlDType::IQ3_XXS | GgmlDType::IQ4_XS
        );
        if is_iq {
            return self.dequantize_matmul(self_shape, storage, layout);
        }
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            // mmvq-vs-mmq crossover. mmvq's per-launch time grows ~linearly
            // with ncols (per-column serial dot chains), flattening batched
            // decode throughput past ~4 concurrent sequences (moss rtf
            // phase A, RTX 3090); the MMQ tile kernels amortize weights
            // across the whole tile instead. Default stays 8; overridable
            // for measurement via MOSS_CANDLE_MMVQ_MAX_BM.
            static MAX_BM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            *MAX_BM.get_or_init(|| {
                std::env::var("MOSS_CANDLE_MMVQ_MAX_BM")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(8)
            })
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    /// Fused SwiGLU forward over two same-shape Q4K weights (`self` = gate,
    /// `up`): `silu(gate·y) * (up·y)`, activation quantized once. Callers
    /// (see `quantized::fused_swiglu_forward`) are responsible for
    /// eligibility checks (dtype Q4K, decode-width batch, contiguous rhs);
    /// this errors rather than falling back.
    pub fn fwd_glu(
        &self,
        up: &QCudaStorage,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        match rhs_l.contiguous_offsets() {
            Some((o1, _)) => {
                if o1 != 0 {
                    crate::bail!("sliced input is not supported in fused glu matmul")
                }
            }
            None => Err(crate::Error::RequiresContiguous { op: "qglu-matmul" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in fused glu {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on glu matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }
        let out = mul_mat_vec_glu_via_q8_1(
            &self.data,
            &up.data,
            &rhs,
            self.dtype,
            ncols,
            nrows,
            b_size,
            self.device(),
        )?;
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }
}

fn mul_mat_vec_via_q8_1_with_out(
    weight: &CudaView<u8>,
    input: &CudaView<u8>,
    dst: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize, //k
    nrows: usize, //n
    b_size: usize,
    dev: &CudaDevice,
) -> Result<()> {
    use cudarc::driver::LaunchAsync;
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, candle_kernels::QUANTIZED)?;
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32 + 1) / 2, 4),
        5..=8 => ((nrows as u32 + 1) / 2, 4),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let params = (
        weight,
        input,
        dst,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32,
    );
    unsafe { func.launch(cfg, params) }.w()?;
    Ok(())
}

fn indexed_moe_forward_fused_q8_1_input(
    weight: &CudaView<u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: &CudaSlice<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &CudaSlice<u32>,
    idx_shape: &crate::Shape, //[batch, topk]
    dev: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
    use cudarc::driver::LaunchAsync;
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];

    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    //quant input into q8_1
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = k_padded / q8_1_block_size;
    let in_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * in_row_size_bytes;
    let mut input_quant = unsafe { dev.alloc::<u8>(y_size_in_bytes).w()? };

    quantize_q8_1(&input, &mut input_quant, k, total_rows, dev)?;
    // output buffer
    let outsize = batch * topk * n;
    let out = unsafe { dev.alloc::<f32>(outsize).w()? };

    let kernel_name = match w_dtype {
        GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
        GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
        GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
        GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
        GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
        GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
        _ => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
    };

    const GROUP_SIZE: usize = 8; // ncols_y (warps per block)
    let use_batch_kernel = batch % GROUP_SIZE == 0;
    let kernel_name = if use_batch_kernel {
        format!("{}_b8", kernel_name)
    } else {
        kernel_name.into()
    };

    const CHUNK_SIZE: usize = 8192; // avoid gridDim.y limit
    let func = dev.get_or_load_func(&kernel_name, candle_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < batch {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = batch - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the input source (q8_1) tensor by bytes ---
        let src_start_elem = rows_processed * input_dim1 * in_row_size_bytes;
        let src_num_elems = rows_in_chunk * input_dim1 * in_row_size_bytes;
        let src_chunk = input_quant.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the index (u32) tensor by bytes ---
        let idx_start_elem = rows_processed * topk;
        let idx_num_elems = rows_in_chunk * topk;
        let idx_chunk = ids.slice(idx_start_elem..(idx_start_elem + idx_num_elems));

        // --- slice the destination (f32) tensor by bytes ---
        let dst_start_elem = rows_processed * topk * n;
        let dst_num_elem = rows_in_chunk * topk * n;
        let dst_chunk = out.slice(dst_start_elem..(dst_start_elem + dst_num_elem));

        let cfg = if use_batch_kernel {
            let rows_per_block = 1usize; // keep in sync with kernel
            let n_blocks_x = (n + rows_per_block - 1) / rows_per_block;
            let num_task_groups = (rows_in_chunk + GROUP_SIZE - 1) / GROUP_SIZE; // ceil_div(batch, 8)
            let n_warps = GROUP_SIZE as u32; // 8 warps
            cudarc::driver::LaunchConfig {
                grid_dim: (n_blocks_x as u32, num_task_groups as u32, topk as u32),
                block_dim: (WARP_SIZE as u32, n_warps, 1), // (32, 8)
                shared_mem_bytes: 0,
            }
        } else {
            let (nblocks, nwarps) = (n as u32, 4);
            cudarc::driver::LaunchConfig {
                grid_dim: (nblocks, rows_in_chunk as u32, topk as u32),
                block_dim: (WARP_SIZE as u32, nwarps, 1),
                shared_mem_bytes: 0,
            }
        };

        let params = (
            weight,
            &src_chunk,
            &idx_chunk,
            &dst_chunk,
            n as i32,
            k as i32,
            rows_in_chunk as i32,
            topk as i32,
            k_padded as i32,
            input_dim1 as i32,
        );
        unsafe { func.clone().launch(cfg, params) }.w()?;
        rows_processed += rows_in_chunk;
    }

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

impl QCudaStorage {
    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &CudaStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &CudaStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        if matches!(
            self.dtype(),
            GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        ) {
            let input_storage = input.as_cuda_slice::<f32>()?;
            let ids_storage = ids.as_cuda_slice::<u32>()?;
            return indexed_moe_forward_fused_q8_1_input(
                &self.data.inner.slice(0..),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                &input_storage,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_storage,
                ids_l.shape(), //[batch, topk]
                &self.device,
            );
        }

        //fallback to kernel iteration
        let (num_experts, n, k) = self_shape.dims3()?;
        let batch = input_l.shape().dims()[0];
        let input_dim1 = input_l.shape().dims()[1];

        let topk = ids_l.shape().dims()[1];
        assert!(batch == ids_l.shape().dims()[0], "batch dim not match!");
        let outsize = batch * topk * n;
        let out = unsafe { self.device.alloc::<f32>(outsize).w()? };

        let w_stride = (n * k) * self.dtype.type_size() / self.dtype.block_size();

        let input_storage = input.as_cuda_slice::<f32>()?;
        let k_padded = pad(k, MATRIX_ROW_PADDING);
        let y_size_in_bytes = batch * input_dim1 * k_padded * GgmlDType::Q8_1.type_size()
            / GgmlDType::Q8_1.block_size();
        let mut input_quant = unsafe { self.device.alloc::<u8>(y_size_in_bytes).w()? };
        quantize_q8_1(
            &input_storage,
            &mut input_quant,
            k,
            batch * input_dim1,
            &self.device,
        )?;
        let in_stride = k_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let out_stride = n;

        let ids_values = match &ids.slice {
            crate::cuda_backend::CudaStorageSlice::U32(data) => {
                self.device.dtoh_sync_copy(data).w()?
            }
            _ => crate::bail!("only f32 can be quantized"),
        };

        for b in 0..batch {
            for i in 0..topk {
                let idx = b * topk + i;
                let input_idx = if input_dim1 == 1 { b } else { idx };
                let cur_input =
                    input_quant.slice(input_idx * in_stride..(input_idx + 1) * in_stride);
                let expert_id = ids_values[idx] as usize;
                assert!(expert_id < num_experts);
                assert!(
                    expert_id * w_stride < self.data.inner.len(),
                    "slice out of weight bound {}!",
                    self.data.inner.len()
                );
                let cur_weight = self
                    .data
                    .inner
                    .slice(expert_id * w_stride..(expert_id + 1) * w_stride);
                let cur_out = out.slice(idx * out_stride..(idx + 1) * out_stride);
                mul_mat_vec_via_q8_1_with_out(
                    &cur_weight,
                    &cur_input,
                    &cur_out,
                    self.dtype,
                    k,
                    n,
                    1,
                    self.device(),
                )?;
            }
        }
        let mut out_shape = input_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        out_shape[1] = topk;
        Ok((
            CudaStorage::wrap_cuda_slice(out, self.device.clone()),
            out_shape.into(),
        ))
    }

    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        match rhs_l.contiguous_offsets() {
            Some((o1, _)) => assert!(
                o1 == 0,
                "sliced input is not supported in quantized matmul!"
            ),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let use_dequant_path = FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed)
            || matches!(
                self.dtype,
                GgmlDType::IQ2_XXS | GgmlDType::IQ2_XS | GgmlDType::IQ3_XXS | GgmlDType::IQ4_XS
            );
        let out = if use_dequant_path {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let storage = storage.as_cuda_slice::<f32>()?;
            match layout.contiguous_offsets() {
                Some((o1, _)) => assert!(
                    o1 == 0,
                    "sliced input is not supported in quantized matmul!"
                ),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = unsafe { device.alloc::<u8>(padded_len).w()? };
    device
        .htod_sync_copy_into(data, &mut inner.slice_mut(..data.len()))
        .w()?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes).w()? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.htod_sync_copy(&vs).w()?;
        quantize_q8_1(&y, &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.htod_sync_copy(&vs).w()?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y,
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.dtoh_sync_copy(&vs.slice(..)).unwrap();
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y,
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.dtoh_sync_copy(&vs.slice(..)).unwrap();
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.htod_sync_copy(&vs).w()?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y,
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.dtoh_sync_copy(&vs.slice(..)).unwrap();

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.htod_sync_copy(&vs).w()?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y,
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.dtoh_sync_copy(&vs.slice(..)).unwrap();
        Ok(())
    }
}
