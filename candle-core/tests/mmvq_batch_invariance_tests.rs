#![cfg(feature = "cuda")]
//! Batched decode must give every sequence the bits it would get alone: the
//! matvec kernels for ncols 1-8 (including the four-row `_r4_` kernels the
//! dispatch picks for wide-K matvecs at ncols >= 5) keep the same per-column
//! thread mapping and summation order, so a column's result may not depend
//! on how many other columns share the launch. A row-mapping slip in a
//! kernel variant shows up here as a mismatch against the single-column run,
//! and the dense reference catches a variant that is consistently wrong.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Result, Tensor};

fn pseudo(len: usize, mul: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * mul) as u32 as f32 / u32::MAX as f32) - 0.5)
        .collect()
}

#[test]
fn each_column_matches_its_single_column_run() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    // MOSS-TTS decode shapes: q/o, k/v, and ffn_down (K = 12288, the shape
    // the four-row kernels serve).
    for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
        for (n, k) in [(1024usize, 4096usize), (4096, 4096), (4096, 12288)] {
            let w = Tensor::from_vec(pseudo(n * k, 2_654_435_761), (n, k), &dev)?;
            let q = QTensor::quantize(&w, dtype)?;
            let dense = q.dequantize(&dev)?;
            let q = QMatMul::from_qtensor(q)?;
            let x = pseudo(8 * k, 40_503 * 65_537);
            // The CUDA matvec reads its activation from offset 0 of the
            // storage, so every batch gets its own buffer, not a row view.
            let rows = |from: usize, b: usize| {
                Tensor::from_vec(x[from * k..(from + b) * k].to_vec(), (b, k), &dev)
            };

            let solo: Vec<Vec<f32>> = (0..8)
                .map(|c| q.forward(&rows(c, 1)?)?.flatten_all()?.to_vec1())
                .collect::<Result<_>>()?;
            let reference = rows(0, 8)?.matmul(&dense.t()?)?.to_vec2::<f32>()?;
            for (c, (got, want)) in solo.iter().zip(&reference).enumerate() {
                let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
                let worst = got
                    .iter()
                    .zip(want)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    worst <= 0.01 * scale,
                    "{dtype:?} n={n} k={k} column {c}: off the dense reference by {worst} (scale {scale})"
                );
            }

            for b in 2..=8 {
                let batched = q.forward(&rows(0, b)?)?.to_vec2::<f32>()?;
                for (c, row) in batched.iter().enumerate() {
                    let same = row
                        .iter()
                        .zip(&solo[c])
                        .all(|(a, s)| a.to_bits() == s.to_bits());
                    assert!(same, "{dtype:?} n={n} k={k} b={b}: column {c} differs from its solo run");
                }
            }
        }
    }
    Ok(())
}
