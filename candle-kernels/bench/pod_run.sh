# Run on a CUDA pod from the repository root: the matvec bench, then the
# quantized CUDA tests. Output goes to stdout.
set -u
nvidia-smi --query-gpu=name,power.limit --format=csv,noheader
[ -f /root/bench_done ] || (cd candle-kernels/bench && nvcc -O3 -arch=sm_86 -std=c++17 -I../src mmvq_bench.cu -o /root/b 2>&1 | grep -E "error" ; /root/b 2>&1; touch /root/bench_done)
command -v cargo >/dev/null || { curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal >/dev/null 2>&1; }
. "$HOME/.cargo/env"
export CUDA_COMPUTE_CAP=86
cargo test -q --release --no-fail-fast -p candle-core --features cuda \
  --test mmvq_batch_invariance_tests --test fused_swiglu_tests --test mmvq_prefetch_tests --test quantized_tests 2>&1 \
  | grep -iE -A3 "^test |test result|panicked|error|FAILED|MMVQ_CHECKSUM" | head -120
# Same quantized tests at the base revision, once, to tell regressions from
# failures that were already there.
if [ ! -f /root/base_done ]; then
  echo "=== base 5b4b272 quantized_tests"
  [ -d /root/base ] || git clone -q https://github.com/ernestas-poskus/candle.git /root/base
  (cd /root/base && git checkout -q 5b4b272156975a7d227b54fb6fac01c3578851cb && \
   cargo test -q --release -p candle-core --features cuda --test quantized_tests 2>&1 \
   | grep -iE -A3 "test result|panicked|FAILED" | head -40)
  touch /root/base_done
fi
