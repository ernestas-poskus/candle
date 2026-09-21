#![cfg(feature = "cuda")]
//! Equivalence gate for the fused SwiGLU matvec (`fused_swiglu_forward`):
//! the fused kernel walks both weight matrices with the same vec_dot and
//! block iteration as the unfused mmvq kernels, so the projections are
//! bit-comparable and only the silu/mul epilogue (f32) may differ by ulps
//! from the composed tensor-op reference.

use candle_core::quantized::{fused_swiglu_forward, GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Result, Tensor};

fn silu_ref(x: &Tensor) -> Result<Tensor> {
    // silu(x) = x / (1 + e^-x), composed from primitive ops so this test
    // does not depend on candle-nn.
    let sig = ((x.neg()?.exp()? + 1.0)?).recip()?;
    x * sig
}

#[test]
fn fused_swiglu_matches_unfused_q4k() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    // Production-shaped reduction width (hidden=4096); row count reduced
    // for test speed — the kernel tiles rows independently, so 512 rows
    // exercise the same code paths as 12288.
    let (n, k) = (512usize, 4096usize);
    let gate_w = Tensor::randn(0f32, 1f32, (n, k), &dev)?;
    let up_w = Tensor::randn(0f32, 1f32, (n, k), &dev)?;
    let gate = QMatMul::from_qtensor(QTensor::quantize(&gate_w, GgmlDType::Q4K)?)?;
    let up = QMatMul::from_qtensor(QTensor::quantize(&up_w, GgmlDType::Q4K)?)?;

    for b in [1usize, 2, 5, 8] {
        let xs = Tensor::randn(0f32, 1f32, (b, k), &dev)?;
        let fused = fused_swiglu_forward(&gate, &up, &xs)?
            .expect("cuda + q4k + b<=8 must take the fused path");
        let reference = (silu_ref(&gate.forward(&xs)?)? * up.forward(&xs)?)?;

        let diff = (&fused - &reference)?.abs()?.flatten_all()?.max(0)?;
        let denom = reference.abs()?.flatten_all()?.max(0)?;
        let max_diff = diff.to_vec0::<f32>()?;
        let max_ref = denom.to_vec0::<f32>()?.max(1e-3);
        assert!(
            max_diff / max_ref < 1e-4,
            "fused vs unfused diverged at b={b}: max_diff={max_diff}, max_ref={max_ref}"
        );
    }
    Ok(())
}

#[test]
fn fused_swiglu_declines_wide_batches() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let (n, k) = (256usize, 1024usize);
    let gate = QMatMul::from_qtensor(QTensor::quantize(
        &Tensor::randn(0f32, 1f32, (n, k), &dev)?,
        GgmlDType::Q4K,
    )?)?;
    let up = QMatMul::from_qtensor(QTensor::quantize(
        &Tensor::randn(0f32, 1f32, (n, k), &dev)?,
        GgmlDType::Q4K,
    )?)?;
    // b=9 exceeds the mmvq decode-kernel width — must decline, not error,
    // so callers keep the unfused fallback.
    let xs = Tensor::randn(0f32, 1f32, (9usize, k), &dev)?;
    assert!(fused_swiglu_forward(&gate, &up, &xs)?.is_none());
    Ok(())
}
