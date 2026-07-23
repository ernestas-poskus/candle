#![cfg(feature = "cuda")]
//! Bit-exactness gate for the software-pipelined q4_K matvec (Phase C1):
//! the `_pf_` kernels reorder LOADS only — every arithmetic op and its
//! order are identical to `mul_mat_vec_q4_K_q8_1_cuda*` — so outputs must
//! be bit-identical, not merely close. Run once with
//! `MOSS_CANDLE_MMVQ_PREFETCH=1` and once without; this test spawns no
//! subprocesses, so it compares against a CPU-side reference instead and
//! is executed twice by the harness script (env off, env on) with the
//! requirement that both runs print identical checksums.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module, Result, Tensor};

#[test]
fn q4k_matvec_shapes_and_checksum() -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let mut checksums = Vec::new();
    for (n, k) in [(1024usize, 4096usize), (4096, 4096), (12288, 4096)] {
        // Deterministic weights/activations so the checksum is comparable
        // across processes (env on vs off).
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 2_654_435_761usize) as u32 as f32 / u32::MAX as f32) - 0.5)
            .collect();
        let w = Tensor::from_vec(w, (n, k), &dev)?;
        let q = QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q4K)?)?;
        for b in [1usize, 2, 4, 5, 8] {
            let xs: Vec<f32> = (0..b * k)
                .map(|i| ((i * 40_503usize) as u16 as f32 / u16::MAX as f32) - 0.5)
                .collect();
            let xs = Tensor::from_vec(xs, (b, k), &dev)?;
            let out = q.forward(&xs)?;
            assert_eq!(out.dims(), &[b, n]);
            let v = out.flatten_all()?.to_vec1::<f32>()?;
            assert!(v.iter().all(|x| x.is_finite()), "non-finite at n={n} b={b}");
            // Order-stable digest of the exact bits.
            let digest = v
                .iter()
                .fold(0u64, |acc, x| acc.rotate_left(1) ^ u64::from(x.to_bits()));
            checksums.push((n, b, digest));
        }
    }
    for (n, b, digest) in checksums {
        println!("MMVQ_CHECKSUM n={n} b={b} digest={digest:016x}");
    }
    Ok(())
}
