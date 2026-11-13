// Cold-DRAM timing of the decode matvec kernels at MOSS-TTS backbone shapes
// (Qwen3 8B, Q4_K_M: q/o 4096x4096, k/v 1024x4096, gate/up 12288x4096,
// down 4096x12288; attn_v and ffn_down are q6_K on half the layers).
// Each launch reads a different slice of a 2 GB weight pool, so weights come
// from DRAM as in decode. Prints us/launch and achieved GB/s of weight reads.
//
// nvcc -O3 -arch=sm_86 -std=c++17 -I../src mmvq_bench.cu -o mmvq_bench
#include "quantized.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cmath>


// ---- Variants: R rows per CTA (R output rows share each activation load), NW warps.
template <int ncols_y, int R, int NW, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t dot>
__global__ void mmvq_r(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {
  const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
  const int row0 = R*blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / QK8_1;
  constexpr int blocks_per_iter = vdr * NW*WARP_SIZE / qi;
  float tmp[ncols_y][R] = {0.0f};
  const block_q_t * x = (const block_q_t *) vx;
  const block_q8_1 * y = (const block_q8_1 *) vy;
  for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);
    const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
    for (int j = 0; j < ncols_y; ++j)
#pragma unroll
      for (int i = 0; i < R; ++i)
        tmp[j][i] += dot(&x[kbx + (row0 + i)*blocks_per_row_x], &y[j*blocks_per_col_y + kby], kqs);
  }
  __shared__ float sh[NW-1 > 0 ? NW-1 : 1][ncols_y][R][WARP_SIZE];
  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_y; ++j)
#pragma unroll
      for (int i = 0; i < R; ++i) sh[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
      for (int l = 0; l < NW-1; ++l) tmp[j][i] += sh[l][j][i][threadIdx.x];
      tmp[j][i] = warp_reduce_sum(tmp[j][i]);
    }
    if (threadIdx.x < R) {
      float v = tmp[j][0];
#pragma unroll
      for (int i = 1; i < R; ++i) if (threadIdx.x == i) v = tmp[j][i];
      dst[j*nrows_dst + row0 + threadIdx.x] = v;
    }
  }
}

template <int ncols_y, int R, int NW>
__global__ void glu_r(const void * __restrict__ vg, const void * __restrict__ vu, const void * __restrict__ vy,
    float * __restrict__ dst, const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {
  constexpr int qk = QK_K, qi = QI4_K, vdr = VDR_Q4_K_Q8_1_MMVQ;
  const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
  const int row0 = R*blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / QK8_1;
  constexpr int blocks_per_iter = vdr * NW*WARP_SIZE / qi;
  float tg[ncols_y][R] = {0.0f}, tu[ncols_y][R] = {0.0f};
  const block_q4_K * xg = (const block_q4_K *) vg;
  const block_q4_K * xu = (const block_q4_K *) vu;
  const block_q8_1 * y = (const block_q8_1 *) vy;
  for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);
    const int kqs = vdr * (tid % (qi/vdr));
#pragma unroll
    for (int j = 0; j < ncols_y; ++j)
#pragma unroll
      for (int i = 0; i < R; ++i) {
        tg[j][i] += vec_dot_q4_K_q8_1(&xg[kbx + (row0 + i)*blocks_per_row_x], &y[j*blocks_per_col_y + kby], kqs);
        tu[j][i] += vec_dot_q4_K_q8_1(&xu[kbx + (row0 + i)*blocks_per_row_x], &y[j*blocks_per_col_y + kby], kqs);
      }
  }
  __shared__ float sg[NW-1 > 0 ? NW-1 : 1][ncols_y][R][WARP_SIZE];
  __shared__ float su[NW-1 > 0 ? NW-1 : 1][ncols_y][R][WARP_SIZE];
  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_y; ++j)
#pragma unroll
      for (int i = 0; i < R; ++i) { sg[threadIdx.y-1][j][i][threadIdx.x] = tg[j][i]; su[threadIdx.y-1][j][i][threadIdx.x] = tu[j][i]; }
  }
  __syncthreads();
  if (threadIdx.y > 0) return;
#pragma unroll
  for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
    for (int i = 0; i < R; ++i) {
#pragma unroll
      for (int l = 0; l < NW-1; ++l) { tg[j][i] += sg[l][j][i][threadIdx.x]; tu[j][i] += su[l][j][i][threadIdx.x]; }
      tg[j][i] = warp_reduce_sum(tg[j][i]); tu[j][i] = warp_reduce_sum(tu[j][i]);
    }
    if (threadIdx.x < R) {
      float g = tg[j][0], u = tu[j][0];
#pragma unroll
      for (int i = 1; i < R; ++i) if (threadIdx.x == i) { g = tg[j][i]; u = tu[j][i]; }
      dst[j*nrows_dst + row0 + threadIdx.x] = g / (1.0f + __expf(-g)) * u;
    }
  }
}

#define Q4R(N,R,NW) mmvq_r<N,R,NW,QK_K,QI4_K,block_q4_K,VDR_Q4_K_Q8_1_MMVQ,vec_dot_q4_K_q8_1>
#define Q6R(N,R,NW) mmvq_r<N,R,NW,QK_K,QI6_K,block_q6_K,VDR_Q6_K_Q8_1_MMVQ,vec_dot_q6_K_q8_1>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

typedef void (*mmvq_t)(const void*, const void*, float*, int, int, int, int);
typedef void (*glu_t)(const void*, const void*, const void*, float*, int, int, int, int);

static const size_t POOL = 2ull << 30;

static void fill_pool(char *dev, size_t block_bytes, size_t d_off, size_t d_len) {
  std::vector<char> h(64 << 20);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (char)(rand() & 0xff);
  for (size_t b = 0; b + block_bytes <= h.size(); b += block_bytes)
    for (size_t j = 0; j < d_len; j += 2) { h[b + d_off + j] = 0x00; h[b + d_off + j + 1] = 0x20; }
  size_t chunk = (h.size() / block_bytes) * block_bytes;
  for (size_t o = 0; o + chunk <= POOL; o += chunk) CK(cudaMemcpy(dev + o, h.data(), chunk, cudaMemcpyHostToDevice));
}

struct Ctx { char *q4k, *q6k; char *y; float *dst; cudaEvent_t a, b; };

template <typename F>
static double time_us(Ctx &c, int iters, F launch) {
  for (int i = 0; i < 20; ++i) launch(i);
  CK(cudaEventRecord(c.a));
  for (int i = 0; i < iters; ++i) launch(i);
  CK(cudaEventRecord(c.b));
  CK(cudaEventSynchronize(c.b));
  float ms; CK(cudaEventElapsedTime(&ms, c.a, c.b));
  CK(cudaGetLastError());
  return ms * 1e3 / iters;
}

struct Var { const char *name; int R, NW; mmvq_t q4[9], q6[9]; glu_t glu[9]; };
#define VROW(N,R,NW) v.q4[N]=Q4R(N,R,NW); v.q6[N]=Q6R(N,R,NW); v.glu[N]=glu_r<N,R,NW>;
#define VAR(R,NW) { Var v{"r" #R "w" #NW, R, NW}; VROW(1,R,NW) VROW(4,R,NW) VROW(5,R,NW) VROW(8,R,NW) vars.push_back(v); }
static std::vector<Var> vars;
static mmvq_t q4k_std[9], q4k_pf[9], q6k_std[9];
static glu_t q4k_glu[9], q4k_glu_pf[9];

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 400;
  q4k_std[1]=mul_mat_vec_q4_K_q8_1_cuda1; q4k_std[2]=mul_mat_vec_q4_K_q8_1_cuda2; q4k_std[3]=mul_mat_vec_q4_K_q8_1_cuda3; q4k_std[4]=mul_mat_vec_q4_K_q8_1_cuda4;
  q4k_std[5]=mul_mat_vec_q4_K_q8_1_cuda5; q4k_std[6]=mul_mat_vec_q4_K_q8_1_cuda6; q4k_std[7]=mul_mat_vec_q4_K_q8_1_cuda7; q4k_std[8]=mul_mat_vec_q4_K_q8_1_cuda8;
  q4k_pf[1]=mul_mat_vec_q4_K_q8_1_pf_cuda1; q4k_pf[2]=mul_mat_vec_q4_K_q8_1_pf_cuda2; q4k_pf[3]=mul_mat_vec_q4_K_q8_1_pf_cuda3; q4k_pf[4]=mul_mat_vec_q4_K_q8_1_pf_cuda4;
  q4k_pf[5]=mul_mat_vec_q4_K_q8_1_pf_cuda5; q4k_pf[6]=mul_mat_vec_q4_K_q8_1_pf_cuda6; q4k_pf[7]=mul_mat_vec_q4_K_q8_1_pf_cuda7; q4k_pf[8]=mul_mat_vec_q4_K_q8_1_pf_cuda8;
  q6k_std[1]=mul_mat_vec_q6_K_q8_1_cuda1; q6k_std[2]=mul_mat_vec_q6_K_q8_1_cuda2; q6k_std[3]=mul_mat_vec_q6_K_q8_1_cuda3; q6k_std[4]=mul_mat_vec_q6_K_q8_1_cuda4;
  q6k_std[5]=mul_mat_vec_q6_K_q8_1_cuda5; q6k_std[6]=mul_mat_vec_q6_K_q8_1_cuda6; q6k_std[7]=mul_mat_vec_q6_K_q8_1_cuda7; q6k_std[8]=mul_mat_vec_q6_K_q8_1_cuda8;
  q4k_glu[1]=mul_mat_vec_q4_K_q8_1_glu_cuda1; q4k_glu[2]=mul_mat_vec_q4_K_q8_1_glu_cuda2; q4k_glu[3]=mul_mat_vec_q4_K_q8_1_glu_cuda3; q4k_glu[4]=mul_mat_vec_q4_K_q8_1_glu_cuda4;
  q4k_glu[5]=mul_mat_vec_q4_K_q8_1_glu_cuda5; q4k_glu[6]=mul_mat_vec_q4_K_q8_1_glu_cuda6; q4k_glu[7]=mul_mat_vec_q4_K_q8_1_glu_cuda7; q4k_glu[8]=mul_mat_vec_q4_K_q8_1_glu_cuda8;
  q4k_glu_pf[1]=mul_mat_vec_q4_K_q8_1_glu_pf_cuda1; q4k_glu_pf[2]=mul_mat_vec_q4_K_q8_1_glu_pf_cuda2; q4k_glu_pf[3]=mul_mat_vec_q4_K_q8_1_glu_pf_cuda3; q4k_glu_pf[4]=mul_mat_vec_q4_K_q8_1_glu_pf_cuda4;
  q4k_glu_pf[5]=mul_mat_vec_q4_K_q8_1_glu_pf_cuda5; q4k_glu_pf[6]=mul_mat_vec_q4_K_q8_1_glu_pf_cuda6; q4k_glu_pf[7]=mul_mat_vec_q4_K_q8_1_glu_pf_cuda7; q4k_glu_pf[8]=mul_mat_vec_q4_K_q8_1_glu_pf_cuda8;

  VAR(2,4) VAR(4,4) VAR(8,4) VAR(4,2) VAR(8,2)
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device %s, %d SMs, L2 %d KB\n", p.name, p.multiProcessorCount, p.l2CacheSize >> 10);
  Ctx c;
  CK(cudaMalloc(&c.q4k, POOL)); CK(cudaMalloc(&c.q6k, POOL));
  CK(cudaMalloc(&c.y, 8 * 12288 / 32 * sizeof(block_q8_1)));
  CK(cudaMalloc(&c.dst, 8 * 12288 * sizeof(float)));
  CK(cudaEventCreate(&c.a)); CK(cudaEventCreate(&c.b));
  fill_pool(c.q4k, sizeof(block_q4_K), 0, 4);
  fill_pool(c.q6k, sizeof(block_q6_K), offsetof(block_q6_K, d), 2);
  CK(cudaMemset(c.y, 0x11, 8 * 12288 / 32 * sizeof(block_q8_1)));

  // Peak: a plain device-to-device copy of 512 MB.
  double cp = time_us(c, 10, [&](int) { CK(cudaMemcpyAsync(c.q6k, c.q4k, 512 << 20, cudaMemcpyDeviceToDevice)); });
  printf("memcpy D2D %.0f GB/s (read+write)\n", 2.0 * (512 << 20) / cp / 1e3);

  struct Shape { const char *name; int n, k; bool q6; bool glu; };
  Shape shapes[] = {
    {"q/o q4K", 4096, 4096, false, false}, {"k/v q4K", 1024, 4096, false, false},
    {"v q6K", 1024, 4096, true, false}, {"gate+up glu q4K", 12288, 4096, false, true},
    {"down q4K", 4096, 12288, false, false}, {"down q6K", 4096, 12288, true, false},
  };
  int cols[] = {4, 5, 8};
  printf("%-16s %5s %-8s %9s %8s\n", "shape", "ncols", "kernel", "us", "GB/s");
  for (auto &s : shapes) {
    size_t row_bytes = (size_t)(s.k / 256) * (s.q6 ? sizeof(block_q6_K) : sizeof(block_q4_K));
    size_t mbytes = row_bytes * s.n;
    size_t slots = POOL / mbytes - (s.glu ? 1 : 0);
    char *pool = s.q6 ? c.q6k : c.q4k;
    int nrows_y = s.k; // multiple of 512
    for (int nc : cols) {
      int rpb = nc == 1 ? 1 : 2;
      dim3 grid((s.n + rpb - 1) / rpb), block(32, 4);
      std::vector<float> ref(s.n * nc), got(s.n * nc);
      auto run = [&](const char *kname, auto launch) {
        double us = time_us(c, iters, launch);
        // Correctness: rerun slot 0 and compare against the reference kernel's output.
        CK(cudaMemset(c.dst, 0, s.n * nc * sizeof(float)));
        launch(0); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), c.dst, s.n * nc * sizeof(float), cudaMemcpyDeviceToHost));
        if (!strcmp(kname, "std")) ref = got;
        double md = 0; for (size_t i = 0; i < got.size(); ++i) { double d = fabs(got[i] - ref[i]) / (fabs(ref[i]) + 1e-3); if (!(d <= md)) md = d; }
        double gb = (s.glu ? 2.0 : 1.0) * mbytes / us / 1e3;
        printf("%-16s %5d %-8s %9.2f %8.0f  maxrel %.1e\n", s.name, nc, kname, us, gb, md);
      };
      if (s.glu) {
        run("std", [&](int i) { char *g = pool + (i % slots) * mbytes; q4k_glu[nc]<<<grid, block>>>(g, g + mbytes, c.y, c.dst, s.k, s.n, nrows_y, s.n); });
        run("pf", [&](int i) { char *g = pool + (i % slots) * mbytes; q4k_glu_pf[nc]<<<grid, block>>>(g, g + mbytes, c.y, c.dst, s.k, s.n, nrows_y, s.n); });
        for (auto &v : vars) { if (nc == 1 && v.R > 1) continue; dim3 g2(s.n / v.R), b2(32, v.NW);
          run(v.name, [&](int i) { char *g = pool + (i % slots) * mbytes; v.glu[nc]<<<g2, b2>>>(g, g + mbytes, c.y, c.dst, s.k, s.n, nrows_y, s.n); }); }
      } else if (s.q6) {
        run("std", [&](int i) { q6k_std[nc]<<<grid, block>>>(pool + (i % slots) * mbytes, c.y, c.dst, s.k, s.n, nrows_y, s.n); });
        for (auto &v : vars) { if (nc == 1 && v.R > 1) continue; dim3 g2(s.n / v.R), b2(32, v.NW);
          run(v.name, [&](int i) { v.q6[nc]<<<g2, b2>>>(pool + (i % slots) * mbytes, c.y, c.dst, s.k, s.n, nrows_y, s.n); }); }
      } else {
        run("std", [&](int i) { q4k_std[nc]<<<grid, block>>>(pool + (i % slots) * mbytes, c.y, c.dst, s.k, s.n, nrows_y, s.n); });
        run("pf", [&](int i) { q4k_pf[nc]<<<grid, block>>>(pool + (i % slots) * mbytes, c.y, c.dst, s.k, s.n, nrows_y, s.n); });
        for (auto &v : vars) { if (nc == 1 && v.R > 1) continue; dim3 g2(s.n / v.R), b2(32, v.NW);
          run(v.name, [&](int i) { v.q4[nc]<<<g2, b2>>>(pool + (i % slots) * mbytes, c.y, c.dst, s.k, s.n, nrows_y, s.n); }); }
      }
    }
  }
  return 0;
}
