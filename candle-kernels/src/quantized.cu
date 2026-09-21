// Kernels adapted from llama.cpp ggml-cuda.cu
// https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda.cu
#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

#define GGML_UNUSED(x) (void)(x)
#define GGML_CUDA_ASSUME(x)

#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

#undef GGML_CUDA_F16
#define GGML_CUDA_DMMV_X 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define CUDA_DEQUANTIZE_BLOCK_SIZE 256
#define K_QUANTS_PER_ITERATION 2

typedef uint16_t ggml_fp16_t;
typedef float dfloat; // dequantize float
typedef float2 dfloat2;
typedef void (*dequantize_kernel_t)(const void * vx, const int ib, const int iqs, dfloat2 & v);

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t * x8, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) (x8 + sizeof(int) * i32); // assume at least 2 byte alignment

    int x32 = 0;
    x32 |= x16[0] <<  0;
    x32 |= x16[1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}

static __device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t * x8, const int & i32) {
    return *((const int *) (x8 + sizeof(int) * i32)); // assume at least 4 byte alignment
}


#define WARP_SIZE 32
#define CUDART_HMAX     11070 // CUDA 11.7, min. ver. for which __hmax and __hmax2 are known to work (may be higher than needed)

#define CC_PASCAL     600
#define MIN_CC_DP4A   610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products
#define CC_VOLTA      700
#define CC_OFFSET_AMD 1000000
#define CC_RDNA1      (CC_OFFSET_AMD + 1010)
#define CC_RDNA2      (CC_OFFSET_AMD + 1030)
#define CC_RDNA3      (CC_OFFSET_AMD + 1100)

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A
    return __dp4a(a, b, c);
#else // __CUDA_ARCH__ >= MIN_CC_DP4A
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif // __CUDA_ARCH__ >= MIN_CC_DP4A
}


#define  MMQ_X_Q4_0_RDNA2  64
#define  MMQ_Y_Q4_0_RDNA2  128
#define NWARPS_Q4_0_RDNA2  8
#define  MMQ_X_Q4_0_RDNA1  64
#define  MMQ_Y_Q4_0_RDNA1  64
#define NWARPS_Q4_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_0_AMPERE 4
#define  MMQ_Y_Q4_0_AMPERE 32
#define NWARPS_Q4_0_AMPERE 4
#else
#define  MMQ_X_Q4_0_AMPERE 64
#define  MMQ_Y_Q4_0_AMPERE 128
#define NWARPS_Q4_0_AMPERE 4
#endif
#define  MMQ_X_Q4_0_PASCAL 64
#define  MMQ_Y_Q4_0_PASCAL 64
#define NWARPS_Q4_0_PASCAL 8

#define  MMQ_X_Q4_1_RDNA2  64
#define  MMQ_Y_Q4_1_RDNA2  128
#define NWARPS_Q4_1_RDNA2  8
#define  MMQ_X_Q4_1_RDNA1  64
#define  MMQ_Y_Q4_1_RDNA1  64
#define NWARPS_Q4_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_1_AMPERE 4
#define  MMQ_Y_Q4_1_AMPERE 32
#define NWARPS_Q4_1_AMPERE 4
#else
#define  MMQ_X_Q4_1_AMPERE 64
#define  MMQ_Y_Q4_1_AMPERE 128
#define NWARPS_Q4_1_AMPERE 4
#endif
#define  MMQ_X_Q4_1_PASCAL 64
#define  MMQ_Y_Q4_1_PASCAL 64
#define NWARPS_Q4_1_PASCAL 8

#define  MMQ_X_Q5_0_RDNA2  64
#define  MMQ_Y_Q5_0_RDNA2  128
#define NWARPS_Q5_0_RDNA2  8
#define  MMQ_X_Q5_0_RDNA1  64
#define  MMQ_Y_Q5_0_RDNA1  64
#define NWARPS_Q5_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_0_AMPERE 4
#define  MMQ_Y_Q5_0_AMPERE 32
#define NWARPS_Q5_0_AMPERE 4
#else
#define  MMQ_X_Q5_0_AMPERE 128
#define  MMQ_Y_Q5_0_AMPERE 64
#define NWARPS_Q5_0_AMPERE 4
#endif
#define  MMQ_X_Q5_0_PASCAL 64
#define  MMQ_Y_Q5_0_PASCAL 64
#define NWARPS_Q5_0_PASCAL 8

#define  MMQ_X_Q5_1_RDNA2  64
#define  MMQ_Y_Q5_1_RDNA2  128
#define NWARPS_Q5_1_RDNA2  8
#define  MMQ_X_Q5_1_RDNA1  64
#define  MMQ_Y_Q5_1_RDNA1  64
#define NWARPS_Q5_1_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_1_AMPERE 4
#define  MMQ_Y_Q5_1_AMPERE 32
#define NWARPS_Q5_1_AMPERE 4
#else
#define  MMQ_X_Q5_1_AMPERE 128
#define  MMQ_Y_Q5_1_AMPERE 64
#define NWARPS_Q5_1_AMPERE 4
#endif
#define  MMQ_X_Q5_1_PASCAL 64
#define  MMQ_Y_Q5_1_PASCAL 64
#define NWARPS_Q5_1_PASCAL 8

#define  MMQ_X_Q8_0_RDNA2  64
#define  MMQ_Y_Q8_0_RDNA2  128
#define NWARPS_Q8_0_RDNA2  8
#define  MMQ_X_Q8_0_RDNA1  64
#define  MMQ_Y_Q8_0_RDNA1  64
#define NWARPS_Q8_0_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q8_0_AMPERE 4
#define  MMQ_Y_Q8_0_AMPERE 32
#define NWARPS_Q8_0_AMPERE 4
#else
#define  MMQ_X_Q8_0_AMPERE 128
#define  MMQ_Y_Q8_0_AMPERE 64
#define NWARPS_Q8_0_AMPERE 4
#endif
#define  MMQ_X_Q8_0_PASCAL 64
#define  MMQ_Y_Q8_0_PASCAL 64
#define NWARPS_Q8_0_PASCAL 8

#define  MMQ_X_Q2_K_RDNA2  64
#define  MMQ_Y_Q2_K_RDNA2  128
#define NWARPS_Q2_K_RDNA2  8
#define  MMQ_X_Q2_K_RDNA1  128
#define  MMQ_Y_Q2_K_RDNA1  32
#define NWARPS_Q2_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q2_K_AMPERE 4
#define  MMQ_Y_Q2_K_AMPERE 32
#define NWARPS_Q2_K_AMPERE 4
#else
#define  MMQ_X_Q2_K_AMPERE 64
#define  MMQ_Y_Q2_K_AMPERE 128
#define NWARPS_Q2_K_AMPERE 4
#endif
#define  MMQ_X_Q2_K_PASCAL 64
#define  MMQ_Y_Q2_K_PASCAL 64
#define NWARPS_Q2_K_PASCAL 8

#define  MMQ_X_Q3_K_RDNA2  128
#define  MMQ_Y_Q3_K_RDNA2  64
#define NWARPS_Q3_K_RDNA2  8
#define  MMQ_X_Q3_K_RDNA1  32
#define  MMQ_Y_Q3_K_RDNA1  128
#define NWARPS_Q3_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q3_K_AMPERE 4
#define  MMQ_Y_Q3_K_AMPERE 32
#define NWARPS_Q3_K_AMPERE 4
#else
#define  MMQ_X_Q3_K_AMPERE 128
#define  MMQ_Y_Q3_K_AMPERE 128
#define NWARPS_Q3_K_AMPERE 4
#endif
#define  MMQ_X_Q3_K_PASCAL 64
#define  MMQ_Y_Q3_K_PASCAL 64
#define NWARPS_Q3_K_PASCAL 8

#define  MMQ_X_Q4_K_RDNA2  64
#define  MMQ_Y_Q4_K_RDNA2  128
#define NWARPS_Q4_K_RDNA2  8
#define  MMQ_X_Q4_K_RDNA1  32
#define  MMQ_Y_Q4_K_RDNA1  64
#define NWARPS_Q4_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q4_K_AMPERE 4
#define  MMQ_Y_Q4_K_AMPERE 32
#define NWARPS_Q4_K_AMPERE 4
#else
#define  MMQ_X_Q4_K_AMPERE 64
#define  MMQ_Y_Q4_K_AMPERE 128
#define NWARPS_Q4_K_AMPERE 4
#endif
#define  MMQ_X_Q4_K_PASCAL 64
#define  MMQ_Y_Q4_K_PASCAL 64
#define NWARPS_Q4_K_PASCAL 8

#define  MMQ_X_Q5_K_RDNA2  64
#define  MMQ_Y_Q5_K_RDNA2  128
#define NWARPS_Q5_K_RDNA2  8
#define  MMQ_X_Q5_K_RDNA1  32
#define  MMQ_Y_Q5_K_RDNA1  64
#define NWARPS_Q5_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q5_K_AMPERE 4
#define  MMQ_Y_Q5_K_AMPERE 32
#define NWARPS_Q5_K_AMPERE 4
#else
#define  MMQ_X_Q5_K_AMPERE 64
#define  MMQ_Y_Q5_K_AMPERE 128
#define NWARPS_Q5_K_AMPERE 4
#endif
#define  MMQ_X_Q5_K_PASCAL 64
#define  MMQ_Y_Q5_K_PASCAL 64
#define NWARPS_Q5_K_PASCAL 8

#define  MMQ_X_Q6_K_RDNA2  64
#define  MMQ_Y_Q6_K_RDNA2  128
#define NWARPS_Q6_K_RDNA2  8
#define  MMQ_X_Q6_K_RDNA1  32
#define  MMQ_Y_Q6_K_RDNA1  64
#define NWARPS_Q6_K_RDNA1  8
#if defined(CUDA_USE_TENSOR_CORES)
#define  MMQ_X_Q6_K_AMPERE 4
#define  MMQ_Y_Q6_K_AMPERE 32
#define NWARPS_Q6_K_AMPERE 4
#else
#define  MMQ_X_Q6_K_AMPERE 64
#define  MMQ_Y_Q6_K_AMPERE 64
#define NWARPS_Q6_K_AMPERE 4
#endif
#define  MMQ_X_Q6_K_PASCAL 64
#define  MMQ_Y_Q6_K_PASCAL 64
#define NWARPS_Q6_K_PASCAL 8


// QK = number of values after dequantization
// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
    half    d;              // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
    half2   dm;             // dm.x = delta, dm.y = min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == sizeof(ggml_fp16_t) * 2 + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
    half d;                 // delta
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
    half2 dm;               // dm.x = delta, dm.y = min
    uint8_t qh[4];          // 5-th bit of quants
    uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
    half    d;              // delta
    int8_t  qs[QK8_0];      // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct {
    half2   ds;             // ds.x = delta, ds.y = sum
    int8_t  qs[QK8_0];      // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_fp16_t) + QK8_0, "wrong q8_1 block size/padding");

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);
typedef void (*allocate_tiles_cuda_t)(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc);
typedef void (*load_tiles_cuda_t)(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row);
typedef float (*vec_dot_q_mul_mat_cuda_t)(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ms, const int & i, const int & j, const int & k);

#define QR2_K 4
#define QI2_K (QK_K / (4*QR2_K))
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    half2 dm;                // super-block scale for quantized scales/mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_fp16_t) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

#define QR3_K 4
#define QI3_K (QK_K / (4*QR3_K))
typedef struct {
    uint8_t hmask[QK_K/8];     // quants - high bit
    uint8_t qs[QK_K/4];        // quants - low 2 bits
#ifdef GGML_QKK_64
    uint8_t scales[2]; // scales, quantized with 8 bits
#else
    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits
#endif
    half d;             // super-block scale
} block_q3_K;
//static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + K_SCALE_SIZE, "wrong q3_K block size/padding");

#define QR4_K 2
#define QI4_K (QK_K / (4*QR4_K))
#ifdef GGML_QKK_64
typedef struct {
    half    dm[2];             // super-block scales/mins
    uint8_t scales[2];         // 4-bit block scales/mins
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == sizeof(half2) + QK_K/2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {
    half2 dm;                  // super-block scale for quantized scales/mins
    uint8_t scales[3*QK_K/64]; // scales, quantized with 6 bits
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_fp16_t) + 3*QK_K/64 + QK_K/2, "wrong q4_K block size/padding");
#endif

#define QR5_K 2
#define QI5_K (QK_K / (4*QR5_K))
#ifdef GGML_QKK_64
typedef struct {
    half d;                  // super-block scale
    int8_t scales[QK_K/16];  // block scales
    uint8_t qh[QK_K/8];      // quants, high bit
    uint8_t qs[QK_K/2];      // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == sizeof(ggml_fp16_t) + QK_K/2 + QK_K/8 + QK_K/16, "wrong q5_K block size/padding");
#else
typedef struct {
    half2 dm;                     // super-block scale for quantized scales/mins
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");
#endif

#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
typedef struct {
    uint8_t ql[QK_K/2];   // quants, lower 4 bits
    uint8_t qh[QK_K/4];   // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales
    half    d;         // delta
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + 13*QK_K/16, "wrong q6_K block size/padding");

// In llama.cpp this is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K block size/padding");


template <int qk, int qr, int qi, bool need_sum, typename block_q_t, int mmq_x, int mmq_y, int nwarps,
              allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles, int vdr, vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void mul_mat_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = nrows_y / QK8_1;
    const int blocks_per_warp = WARP_SIZE / qi;

    const int & ncols_dst = ncols_y;

    const int row_dst_0 = blockIdx.x*mmq_y;
    const int & row_x_0 = row_dst_0;

    const int col_dst_0 = blockIdx.y*mmq_x;
    const int & col_y_0 = col_dst_0;

    int   * tile_x_ql = nullptr;
    half2 * tile_x_dm = nullptr;
    int   * tile_x_qh = nullptr;
    int   * tile_x_sc = nullptr;

    allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

    __shared__ int    tile_y_qs[mmq_x * WARP_SIZE];
    __shared__ half2  tile_y_ds[mmq_x * WARP_SIZE/QI8_1];

    float sum[mmq_y/WARP_SIZE][mmq_x/nwarps] = {{0.0f}};

    for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {

        load_tiles(x + row_x_0*blocks_per_row_x + ib0, tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc,
                   threadIdx.y, nrows_x-row_x_0-1, threadIdx.x, blocks_per_row_x);

#pragma unroll
        for (int ir = 0; ir < qr; ++ir) {
            const int kqs = ir*WARP_SIZE + threadIdx.x;
            const int kbxd = kqs / QI8_1;

#pragma unroll
            for (int i = 0; i < mmq_x; i += nwarps) {
                const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y-1); // to prevent out-of-bounds memory accesses

                const block_q8_1 * by0 = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + kbxd];

                const int index_y = (threadIdx.y + i) * WARP_SIZE + kqs % WARP_SIZE;
                tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
            }

#pragma unroll
            for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
                const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE/QI8_1)) % mmq_x;
                const int kby = threadIdx.x % (WARP_SIZE/QI8_1);
                const int col_y_eff = min(col_y_0 + ids, ncols_y-1);

                // if the sum is not needed it's faster to transform the scale to f32 ahead of time
                const half2 * dsi_src = &y[col_y_eff*blocks_per_col_y + ib0 * (qk/QK8_1) + ir*(WARP_SIZE/QI8_1) + kby].ds;
                half2       * dsi_dst = &tile_y_ds[ids * (WARP_SIZE/QI8_1) + kby];
                if (need_sum) {
                    *dsi_dst = *dsi_src;
                } else {
                    float * dfi_dst = (float *) dsi_dst;
                    *dfi_dst = __low2half(*dsi_src);
                }
            }

            __syncthreads();

// #pragma unroll // unrolling this loop causes too much register pressure
            for (int k = ir*WARP_SIZE/qr; k < (ir+1)*WARP_SIZE/qr; k += vdr) {
#pragma unroll
                for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
                    for (int i = 0; i < mmq_y; i += WARP_SIZE) {
                        sum[i/WARP_SIZE][j/nwarps] += vec_dot(
                            tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds,
                            threadIdx.x + i, threadIdx.y + j, k);
                    }
                }
            }

            __syncthreads();
        }
    }

#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
        const int col_dst = col_dst_0 + j + threadIdx.y;

        if (col_dst >= ncols_dst) {
            return;
        }

#pragma unroll
        for (int i = 0; i < mmq_y; i += WARP_SIZE) {
            const int row_dst = row_dst_0 + threadIdx.x + i;

            if (row_dst >= nrows_dst) {
                continue;
            }

            dst[col_dst*nrows_dst + row_dst] = sum[i/WARP_SIZE][j/nwarps];
        }
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    (void)x_qh; (void)x_sc;

    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const block_q4_0 * bx0 = (const block_q4_0 *) vx;

    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
        // x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbx] = bxi->d;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = bxi->d;
    }
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_1;
    const int kqsx = k % QI4_1;

    const block_q4_1 * bx0 = (const block_q4_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_1) {
        int i = i0 + i_offset * QI4_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI4_1) + i / QI4_1 + kbxd] = bxi->dm;
    }
}


template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    (void)x_qh; (void)x_sc;

    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI4_0) + mmq_y/QI4_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int   tile_x_qs[mmq_y * (WARP_SIZE) +     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI4_1) + mmq_y/QI4_1];

    *x_ql = tile_x_qs;
    *x_dm = tile_x_dm;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {16.0f, 16.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}


template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __device__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = 2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

template<typename dst_t>
static __device__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static __device__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

//================================== k-quants

template<typename dst_t>
static __device__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int tid = threadIdx.x;
#if QK_K == 256
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#else
    const int is = tid/16;  // 0 or 1
    const int il = tid%16;  // 0...15
    const uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;
    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

template<typename dst_t>
static __device__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

#if QK_K == 256
    const int r = threadIdx.x/4;
    const int tid = r/2;
    const int is0 = r%2;
    const int l0 = 16*is0 + 4*(threadIdx.x%4);
    const int n = tid / 4;
    const int j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#else
    const int tid = threadIdx.x;
    const int is  = tid/16;  // 0 or 1
    const int il  = tid%16;  // 0...15
    const int im  = il/8;    // 0...1
    const int in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    const uint8_t q = x[i].qs[il] >> (2*is);
    const uint8_t h = x[i].hmask[in] >> (2*is + im);
    const float   d = (float)x[i].d;

    if (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } else {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

#if QK_K == 256
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

template<typename dst_t>
static __device__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int i = blockIdx.x;

#if QK_K == 256
    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
#else
    const int tid = threadIdx.x;
    const uint8_t * q = x[i].qs;
    dst_t * y = yy + i*QK_K;
    const float d = (float)x[i].dm[0];
    const float m = (float)x[i].dm[1];
    y[tid+ 0] = d * (x[i].scales[0] & 0xF) * (q[tid] & 0xF) - m * (x[i].scales[0] >> 4);
    y[tid+32] = d * (x[i].scales[1] & 0xF) * (q[tid] >>  4) - m * (x[i].scales[1] >> 4);
#endif
}

template<typename dst_t>
static __device__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int i = blockIdx.x;

#if QK_K == 256
    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#else
    const int tid = threadIdx.x;
    const uint8_t q = x[i].qs[tid];
    const int im = tid/8;  // 0...3
    const int in = tid%8;  // 0...7
    const int is = tid/16; // 0 or 1
    const uint8_t h = x[i].qh[in] >> im;
    const float d = x[i].d;
    dst_t * y = yy + i*QK_K + tid;
    y[ 0] = d * x[i].scales[is+0] * ((q & 0xF) - ((h >> 0) & 1 ? 0 : 16));
    y[32] = d * x[i].scales[is+2] * ((q >>  4) - ((h >> 4) & 1 ? 0 : 16));
#endif
}

template<typename dst_t>
static __device__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;
#if QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#else

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/16;         // 0 or 1
    const int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    const float d = x[i].d;

    const uint8_t   ql = x[i].ql[16*ip + il];
    const uint8_t   qh = x[i].qh[il] >> (2*ip);
    const int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}

template<typename dst_t>
static __device__ void dequantize_block_q8_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int i = blockIdx.x;

    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 8*il;

    const block_q8_0 * x = (const block_q8_0 *)vx + ib;
    const float d = __half2float(x->d);

    const int8_t * q = x->qs + 8*il;

    for (int l = 0; l < 8; ++l) {
        y[l] = d * q[l];
    }
}

template<typename dst_t>
static __device__ void dequantize_block_q8_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q8_K * x = (const block_q8_K *) vx;

    const int i = blockIdx.x;

#if QK_K == 256
    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int n   = 8;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const int8_t * q = x[i].qs + 64*il + n*ir;

    for (int l = 0; l < n; ++l) {
        y[l] = q[l] * x[i].d;
    }
#else
    const int tid = threadIdx.x;
    const uint8_t * q = x[i].qs;
    float * y = yy + i*QK_K;
    y[tid] = x[i].d * x[i].scales[0];
#endif
}

template<typename dst_t>
static __device__ void dequantize_block_q5_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
  return dequantize_block<QK5_0, QR5_0, dequantize_q5_0>(vx, yy, nb32);
}

template<typename dst_t>
static __device__ void dequantize_block_q5_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
  return dequantize_block<QK5_1, QR5_1, dequantize_q5_1>(vx, yy, nb32);
}

#define DEQUANTIZE_K(QNAME) \
extern "C" __global__ void dequantize_block_##QNAME##_f32(const void * __restrict__ vx, float * __restrict__ y) { \
  dequantize_block_##QNAME(vx, y); \
} \
extern "C" __global__ void dequantize_block_##QNAME##_f16(const void * __restrict__ vx, half * __restrict__ y) { \
  dequantize_block_##QNAME(vx, y); \
} \

#define DEQUANTIZE(QNAME) \
extern "C" __global__ void dequantize_block_##QNAME##_f32(const void * __restrict__ vx, float * __restrict__ y, const int k) { \
  dequantize_block_##QNAME(vx, y, k); \
} \
extern "C" __global__ void dequantize_block_##QNAME##_f16(const void * __restrict__ vx, half * __restrict__ y, const int k) { \
  dequantize_block_##QNAME(vx, y, k); \
} \

DEQUANTIZE_K(q2_K)
DEQUANTIZE_K(q3_K)
DEQUANTIZE_K(q4_K)
DEQUANTIZE_K(q5_K)
DEQUANTIZE_K(q6_K)
DEQUANTIZE_K(q8_K)
DEQUANTIZE(q4_0)
DEQUANTIZE(q4_1)
DEQUANTIZE(q5_0)
DEQUANTIZE(q5_1)
DEQUANTIZE(q8_0)

template <int qk, int qr, dequantize_kernel_t dequantize_kernel>
static __device__ void dequantize_mul_mat_vec(const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst, const int ncols, const int nrows) {
    // qk = quantized weights per x block
    // qr = number of quantized weights per data value in x block
    const int row = blockIdx.x*blockDim.y + threadIdx.y;

    if (row >= nrows) {
        return;
    }

    const int tid = threadIdx.x;

    const int iter_stride = 2*GGML_CUDA_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE; // num quantized vals per thread and i iter
    const int y_offset = qr == 1 ? 1 : qk/2;

// partial sum for each thread
#ifdef GGML_CUDA_F16
    half2 tmp = {0.0f, 0.0f}; // two sums for f16 to take advantage of half2 intrinsics
#else
    float tmp = 0.0f;
#endif // GGML_CUDA_F16

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/qk; // x block index
        const int iqs = (col%qk)/qr; // x quant index
        const int iybs = col - col%qk; // y block start index

// processing >2 values per i iter is faster for fast GPUs
#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            // process 2 vals per j iter

            // dequantize
            // for qr = 2 the iqs needs to increase by 1 per j iter because 2 weights per data val
            dfloat2 v;
            dequantize_kernel(vx, ib, iqs + j/qr, v);

            // matrix multiplication
            // for qr = 2 the y index needs to increase by 1 per j iter because of y_offset = qk/2
#ifdef GGML_CUDA_F16
            tmp += __hmul2(v, {
                y[iybs + iqs + j/qr + 0],
                y[iybs + iqs + j/qr + y_offset]
            });
#else
            tmp += v.x * y[iybs + iqs + j/qr + 0];
            tmp += v.y * y[iybs + iqs + j/qr + y_offset];
#endif // GGML_CUDA_F16
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
#ifdef GGML_CUDA_F16
        dst[row] = tmp.x + tmp.y;
#else
        dst[row] = tmp;
#endif // GGML_CUDA_F16
    }
}

extern "C" __global__ void dequantize_mul_mat_vec_q4_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows) {
    dequantize_mul_mat_vec<QK4_0, QR4_0, dequantize_q4_0>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void dequantize_mul_mat_vec_q4_1_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows) {
    dequantize_mul_mat_vec<QK4_1, QR4_1, dequantize_q4_1>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void dequantize_mul_mat_vec_q5_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows) {
    dequantize_mul_mat_vec<QK5_0, QR5_0, dequantize_q5_0>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void dequantize_mul_mat_vec_q5_1_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows) {
    dequantize_mul_mat_vec<QK5_1, QR5_1, dequantize_q5_1>(vx, y, dst, ncols, nrows);
}
extern "C" __global__ void dequantize_mul_mat_vec_q8_0_cuda(const void * vx, const dfloat * y, float * dst, const int ncols, const int nrows) {
    dequantize_mul_mat_vec<QK8_0, QR8_0, dequantize_q8_0>(vx, y, dst, ncols, nrows);
}

extern "C" __global__ void dequantize_mul_mat_vec_q2_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q2_K * x = (const block_q2_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...15
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int step = 16/K_QUANTS_PER_ITERATION;

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15 or 0...14 in steps of 2
    const int q_offset = 32*im + l0;
    const int s_offset = 8*im;
    const int y_offset = 128*im + l0;

    uint32_t aux[4];
    const uint8_t * d = (const uint8_t *)aux;
    const uint8_t * m = (const uint8_t *)(aux + 2);

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

        const uint32_t * a = (const uint32_t *)(x[i].scales + s_offset);
        aux[0] = a[0] & 0x0f0f0f0f;
        aux[1] = a[1] & 0x0f0f0f0f;
        aux[2] = (a[0] >> 4) & 0x0f0f0f0f;
        aux[3] = (a[1] >> 4) & 0x0f0f0f0f;

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            sum1 += y[l+ 0] * d[0] * ((q[l+ 0] >> 0) & 3)
                  + y[l+32] * d[2] * ((q[l+ 0] >> 2) & 3)
                  + y[l+64] * d[4] * ((q[l+ 0] >> 4) & 3)
                  + y[l+96] * d[6] * ((q[l+ 0] >> 6) & 3)
                  + y[l+16] * d[1] * ((q[l+16] >> 0) & 3)
                  + y[l+48] * d[3] * ((q[l+16] >> 2) & 3)
                  + y[l+80] * d[5] * ((q[l+16] >> 4) & 3)
                  +y[l+112] * d[7] * ((q[l+16] >> 6) & 3);
            sum2 += y[l+ 0] * m[0] + y[l+32] * m[2] + y[l+64] * m[4] + y[ l+96] * m[6]
                  + y[l+16] * m[1] + y[l+48] * m[3] + y[l+80] * m[5] + y[l+112] * m[7];

        }
        tmp += dall * sum1 - dmin * sum2;

    }
#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;

    uint32_t uaux[2];
    const uint8_t * d = (const uint8_t *)uaux;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint32_t * s = (const uint32_t *)x[i].scales;

        uaux[0] = s[0] & 0x0f0f0f0f;
        uaux[1] = (s[0] >> 4) & 0x0f0f0f0f;

        const float2 dall = __half22float2(x[i].dm);

        float sum1 = 0, sum2 = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t ql = q[l];
            sum1 += y[l+ 0] * d[0] * ((ql >> 0) & 3)
                  + y[l+16] * d[1] * ((ql >> 2) & 3)
                  + y[l+32] * d[2] * ((ql >> 4) & 3)
                  + y[l+48] * d[3] * ((ql >> 6) & 3);
            sum2 += y[l+0] * d[4] + y[l+16] * d[5] + y[l+32] * d[6] + y[l+48] * d[7];
        }
        tmp += dall.x * sum1 - dall.y * sum2;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

extern "C" __global__ void dequantize_mul_mat_vec_q3_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q3_K * x = (const block_q3_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256

    const uint16_t kmask1 = 0x0303;
    const uint16_t kmask2 = 0x0f0f;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int n  = K_QUANTS_PER_ITERATION;               // iterations in the inner loop
    const int step = 16/K_QUANTS_PER_ITERATION;
    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0....15 or 0...7

    const uint8_t m = 1 << (4*im);

    const int l0 = n*in;                                 // 0...15 or 0...14 in steps of 2
    const int q_offset =  32*im + l0;
    const int y_offset = 128*im + l0;

    uint16_t utmp[4];
    const int8_t * s = (const int8_t *)utmp;

    const uint16_t s_shift = 4*im;

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * q = x[i].qs + q_offset;
        const uint8_t * h = x[i].hmask + l0;

        const uint16_t * a = (const uint16_t *)x[i].scales;
        utmp[0] = ((a[0] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 0)) & kmask1) << 4);
        utmp[1] = ((a[1] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 0)) & kmask1) << 4);
        utmp[2] = ((a[2] >> s_shift) & kmask2) | (((a[4] >> (s_shift + 2)) & kmask1) << 4);
        utmp[3] = ((a[3] >> s_shift) & kmask2) | (((a[5] >> (s_shift + 2)) & kmask1) << 4);

        const float d = x[i].d;

        float sum = 0;
        for (int l = 0; l < n; ++l) {
            sum += y[l+ 0] * (s[0] - 32) * (((q[l] >> 0) & 3) - (h[l] & (m << 0) ? 0 : 4))
                 + y[l+32] * (s[2] - 32) * (((q[l] >> 2) & 3) - (h[l] & (m << 1) ? 0 : 4))
                 + y[l+64] * (s[4] - 32) * (((q[l] >> 4) & 3) - (h[l] & (m << 2) ? 0 : 4))
                 + y[l+96] * (s[6] - 32) * (((q[l] >> 6) & 3) - (h[l] & (m << 3) ? 0 : 4));
            sum += y[l+16] * (s[1] - 32) * (((q[l+16] >> 0) & 3) - (h[l+16] & (m << 0) ? 0 : 4))
                 + y[l+48] * (s[3] - 32) * (((q[l+16] >> 2) & 3) - (h[l+16] & (m << 1) ? 0 : 4))
                 + y[l+80] * (s[5] - 32) * (((q[l+16] >> 4) & 3) - (h[l+16] & (m << 2) ? 0 : 4))
                + y[l+112] * (s[7] - 32) * (((q[l+16] >> 6) & 3) - (h[l+16] & (m << 3) ? 0 : 4));
        }
        tmp += d * sum;

    }
#else

    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15 or 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0....1 or 0...3
    const int offset = tid * K_QUANTS_PER_ITERATION;         // 0...15 or 0...14
    const int in = offset/8;                                 // 0 or 1
    const int im = offset%8;                                 // 0...7

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y = yy + i * QK_K + offset;
        const uint8_t * q = x[i].qs + offset;
        const uint8_t * s = x[i].scales;

        const float dall = (float)x[i].d;

        float sum = 0;
        for (int l = 0; l < K_QUANTS_PER_ITERATION; ++l) {
            const uint8_t hl = x[i].hmask[im+l] >> in;
            const uint8_t ql = q[l];
            sum += y[l+ 0] * dall * ((s[0] & 0xF) - 8) * ((int8_t)((ql >> 0) & 3) - ((hl >> 0) & 1 ? 0 : 4))
                 + y[l+16] * dall * ((s[0] >>  4) - 8) * ((int8_t)((ql >> 2) & 3) - ((hl >> 2) & 1 ? 0 : 4))
                 + y[l+32] * dall * ((s[1] & 0xF) - 8) * ((int8_t)((ql >> 4) & 3) - ((hl >> 4) & 1 ? 0 : 4))
                 + y[l+48] * dall * ((s[1] >>  4) - 8) * ((int8_t)((ql >> 6) & 3) - ((hl >> 6) & 1 ? 0 : 4));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

extern "C" __global__ void dequantize_mul_mat_vec_q4_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q4_K * x = (const block_q4_K *)vx + ib0;

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0,1

    const int step = 8/K_QUANTS_PER_ITERATION;           // 8 or 4

    const int il  = tid/step;                            // 0...3
    const int ir  = tid - step*il;                       // 0...7 or 0...3
    const int n   = 2 * K_QUANTS_PER_ITERATION;          // 2 or 4

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

#if K_QUANTS_PER_ITERATION == 2
    uint32_t q32[4];
    const uint8_t * q4 = (const uint8_t *)q32;
#else
    uint16_t q16[4];
    const uint8_t * q4 = (const uint8_t *)q16;
#endif

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y1 = yy + i*QK_K + y_offset;
        const float   * y2 = y1 + 128;

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

#if K_QUANTS_PER_ITERATION == 2
        const uint32_t * q1 = (const uint32_t *)(x[i].qs + q_offset);
        const uint32_t * q2 = q1 + 16;

        q32[0] = q1[0] & 0x0f0f0f0f;
        q32[1] = q1[0] & 0xf0f0f0f0;
        q32[2] = q2[0] & 0x0f0f0f0f;
        q32[3] = q2[0] & 0xf0f0f0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 4; ++l) {
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+ 4];
            s.z += y2[l] * q4[l+8]; s.w += y2[l+32] * q4[l+12];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
#else
        const uint16_t * q1 = (const uint16_t *)(x[i].qs + q_offset);
        const uint16_t * q2 = q1 + 32;

        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[0] & 0xf0f0;
        q16[2] = q2[0] & 0x0f0f;
        q16[3] = q2[0] & 0xf0f0;

        float4 s = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        for (int l = 0; l < 2; ++l) {
            s.x += y1[l] * q4[l+0]; s.y += y1[l+32] * q4[l+2];
            s.z += y2[l] * q4[l+4]; s.w += y2[l+32] * q4[l+6];
            smin += y1[l] * sc[2] + y1[l+32] * sc[3] + y2[l] * sc[6] + y2[l+32] * sc[7];
        }
        tmp += dall * (s.x * sc[0] + s.y * sc[1] * 1.f/16.f + s.z * sc[4] + s.w * sc[5] * 1.f/16.f) - dmin * smin;
#endif

    }
#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);

    const int step = tid * K_QUANTS_PER_ITERATION;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    float tmp = 0;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const float   * y = yy + i*QK_K + step;
        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux16[0] = a[0] & 0x0f0f;
        aux16[1] = (a[0] >> 4) & 0x0f0f;
        const float d = (float)x[i].dm[0];
        const float m = (float)x[i].dm[1];
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * (d * s[0] * (q[j+ 0] & 0xF) - m * s[2])
                 + y[j+16] * (d * s[0] * (q[j+16] & 0xF) - m * s[2])
                 + y[j+32] * (d * s[1] * (q[j+ 0] >>  4) - m * s[3])
                 + y[j+48] * (d * s[1] * (q[j+16] >>  4) - m * s[3]);
        }
        tmp += sum;
    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

extern "C" __global__ void dequantize_mul_mat_vec_q5_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols) {

    const int row = blockIdx.x;
    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q5_K * x = (const block_q5_K *)vx + ib0;

    float tmp = 0; // partial sum for thread in warp

#if QK_K == 256
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int tid = threadIdx.x/2;  // 0...15
    const int ix  = threadIdx.x%2;

    const int il  = tid/4;     // 0...3
    const int ir  = tid - 4*il;// 0...3
    const int n   = 2;

    const int im = il/2;  // 0 or 1. 0 computes 0,32 + 128,160, 1 computes 64,96 + 192,224
    const int in = il%2;

    const int l0 = n*(2*ir + in);
    const int q_offset = 32*im + l0;
    const int y_offset = 64*im + l0;

    const uint8_t hm1  = 1 << (2*im);
    const uint8_t hm2  = hm1 << 4;

    uint16_t aux[4];
    const uint8_t * sc = (const uint8_t *)aux;

    uint16_t q16[8];
    const uint8_t * q4 = (const uint8_t *)q16;

    for (int i = ix; i < num_blocks_per_row; i += 2) {

        const uint8_t * ql1 = x[i].qs + q_offset;
        const uint8_t * qh  = x[i].qh + l0;
        const float   * y1  = yy + i*QK_K + y_offset;
        const float   * y2  = y1 + 128;

        const float dall = __low2half(x[i].dm);
        const float dmin = __high2half(x[i].dm);

        const uint16_t * a = (const uint16_t *)x[i].scales;
        aux[0] = a[im+0] & kmask1;
        aux[1] = a[im+2] & kmask1;
        aux[2] = ((a[im+4] >> 0) & kmask2) | ((a[im+0] & kmask3) >> 2);
        aux[3] = ((a[im+4] >> 4) & kmask2) | ((a[im+2] & kmask3) >> 2);

        float4 sum = {0.f, 0.f, 0.f, 0.f};
        float smin = 0;
        const uint16_t * q1 = (const uint16_t *)ql1;
        const uint16_t * q2 = q1 + 32;
        q16[0] = q1[0] & 0x0f0f;
        q16[1] = q1[8] & 0x0f0f;
        q16[2] = (q1[0] >> 4) & 0x0f0f;
        q16[3] = (q1[8] >> 4) & 0x0f0f;
        q16[4] = q2[0] & 0x0f0f;
        q16[5] = q2[8] & 0x0f0f;
        q16[6] = (q2[0] >> 4) & 0x0f0f;
        q16[7] = (q2[8] >> 4) & 0x0f0f;
        for (int l = 0; l < n; ++l) {
            sum.x += y1[l+ 0] * (q4[l +0] + (qh[l+ 0] & (hm1 << 0) ? 16 : 0))
                   + y1[l+16] * (q4[l +2] + (qh[l+16] & (hm1 << 0) ? 16 : 0));
            sum.y += y1[l+32] * (q4[l +4] + (qh[l+ 0] & (hm1 << 1) ? 16 : 0))
                   + y1[l+48] * (q4[l +6] + (qh[l+16] & (hm1 << 1) ? 16 : 0));
            sum.z += y2[l+ 0] * (q4[l +8] + (qh[l+ 0] & (hm2 << 0) ? 16 : 0))
                   + y2[l+16] * (q4[l+10] + (qh[l+16] & (hm2 << 0) ? 16 : 0));
            sum.w += y2[l+32] * (q4[l+12] + (qh[l+ 0] & (hm2 << 1) ? 16 : 0))
                   + y2[l+48] * (q4[l+14] + (qh[l+16] & (hm2 << 1) ? 16 : 0));
            smin += (y1[l] + y1[l+16]) * sc[2] + (y1[l+32] + y1[l+48]) * sc[3]
                  + (y2[l] + y2[l+16]) * sc[6] + (y2[l+32] + y2[l+48]) * sc[7];
        }
        tmp += dall * (sum.x * sc[0] + sum.y * sc[1] + sum.z * sc[4] + sum.w * sc[5]) - dmin * smin;
    }

#else
    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...15
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);
    const int step = tid * K_QUANTS_PER_ITERATION;
    const int im = step/8;
    const int in = step%8;

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {
        const uint8_t * q = x[i].qs + step;
        const int8_t  * s = x[i].scales;
        const float   * y = yy + i*QK_K + step;
        const float     d = x[i].d;
        float sum = 0.f;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            const uint8_t h = x[i].qh[in+j] >> im;
            sum += y[j+ 0] * d * s[0] * ((q[j+ 0] & 0xF) - ((h >> 0) & 1 ? 0 : 16))
                 + y[j+16] * d * s[1] * ((q[j+16] & 0xF) - ((h >> 2) & 1 ? 0 : 16))
                 + y[j+32] * d * s[2] * ((q[j+ 0] >>  4) - ((h >> 4) & 1 ? 0 : 16))
                 + y[j+48] * d * s[3] * ((q[j+16] >>  4) - ((h >> 6) & 1 ? 0 : 16));
        }
        tmp += sum;
    }
#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

extern "C" __global__ void dequantize_mul_mat_vec_q6_k(const void * __restrict__ vx, const float * __restrict__ yy, float * __restrict__ dst, const int ncols, int nrows) {

    static_assert(16%K_QUANTS_PER_ITERATION == 0, "16 must be divisible by K_QUANTS_PER_ITERATION");

    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    if (row > nrows) return;

    const int num_blocks_per_row = ncols / QK_K;
    const int ib0 = row*num_blocks_per_row;

    const block_q6_K * x = (const block_q6_K *)vx + ib0;

#if QK_K == 256

    const int tid = threadIdx.x/K_QUANTS_PER_ITERATION;  // 0...31 or 0...16
    const int ix  = threadIdx.x%K_QUANTS_PER_ITERATION;  // 0 or 0, 1

    const int step = 16/K_QUANTS_PER_ITERATION;          // 16 or 8

    const int im = tid/step;                             // 0 or 1. 0 computes 0..., 1 computes 128...
    const int in = tid - step*im;                        // 0...15 or 0...7

#if K_QUANTS_PER_ITERATION == 1
    const int l0 = K_QUANTS_PER_ITERATION*in;            // 0...15
    const int is = 0;
#else
    const int l0 = 4 * in;                               // 0, 4, 8, ..., 28
    const int is = in / 4;
#endif
    const int ql_offset = 64*im + l0;
    const int qh_offset = 32*im + l0;
    const int s_offset  =  8*im + is;
    const int y_offset = 128*im + l0;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + y_offset;
        const uint8_t * ql = x[i].ql + ql_offset;
        const uint8_t * qh = x[i].qh + qh_offset;
        const int8_t  * s  = x[i].scales + s_offset;

        const float d = x[i].d;

#if K_QUANTS_PER_ITERATION == 1
        float sum = y[ 0] * s[0] * d * ((int8_t)((ql[ 0] & 0xF) | ((qh[ 0] & 0x03) << 4)) - 32)
                  + y[16] * s[1] * d * ((int8_t)((ql[16] & 0xF) | ((qh[16] & 0x03) << 4)) - 32)
                  + y[32] * s[2] * d * ((int8_t)((ql[32] & 0xF) | ((qh[ 0] & 0x0c) << 2)) - 32)
                  + y[48] * s[3] * d * ((int8_t)((ql[48] & 0xF) | ((qh[16] & 0x0c) << 2)) - 32)
                  + y[64] * s[4] * d * ((int8_t)((ql[ 0]  >> 4) | ((qh[ 0] & 0x30) >> 0)) - 32)
                  + y[80] * s[5] * d * ((int8_t)((ql[16]  >> 4) | ((qh[16] & 0x30) >> 0)) - 32)
                  + y[96] * s[6] * d * ((int8_t)((ql[32]  >> 4) | ((qh[ 0] & 0xc0) >> 2)) - 32)
                  +y[112] * s[7] * d * ((int8_t)((ql[48]  >> 4) | ((qh[16] & 0xc0) >> 2)) - 32);
        tmp += sum;
#else
        float sum = 0;
        for (int l = 0; l < 4; ++l) {
            sum += y[l+ 0] * s[0] * d * ((int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32)
                 + y[l+32] * s[2] * d * ((int8_t)((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32)
                 + y[l+64] * s[4] * d * ((int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32)
                 + y[l+96] * s[6] * d * ((int8_t)((ql[l+32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        tmp += sum;
#endif

    }

#else

    const int tid = threadIdx.x/(2*K_QUANTS_PER_ITERATION);  // 0...7
    const int ix  = threadIdx.x%(2*K_QUANTS_PER_ITERATION);  // 0...3

    const int step = tid * K_QUANTS_PER_ITERATION;

    float tmp = 0; // partial sum for thread in warp

    for (int i = ix; i < num_blocks_per_row; i += 2*K_QUANTS_PER_ITERATION) {

        const float   * y  = yy + i * QK_K + step;
        const uint8_t * ql = x[i].ql + step;
        const uint8_t * qh = x[i].qh + step;
        const int8_t  * s  = x[i].scales;

        const float d = x[i+0].d;

        float sum = 0;
        for (int j = 0; j < K_QUANTS_PER_ITERATION; ++j) {
            sum += y[j+ 0] * s[0] * d * ((int8_t)((ql[j+ 0] & 0xF) | ((qh[j] & 0x03) << 4)) - 32)
                 + y[j+16] * s[1] * d * ((int8_t)((ql[j+16] & 0xF) | ((qh[j] & 0x0c) << 2)) - 32)
                 + y[j+32] * s[2] * d * ((int8_t)((ql[j+ 0] >>  4) | ((qh[j] & 0x30) >> 0)) - 32)
                 + y[j+48] * s[3] * d * ((int8_t)((ql[j+16] >>  4) | ((qh[j] & 0xc0) >> 2)) - 32);
        }
        tmp += sum;

    }

#endif

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_0_q8_1_impl(
    const int * v, const int * u, const float & d4, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 8 from each quant value
    return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q4_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm4, const half2 & ds8) {
    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm4, ds8));
    const float d4d8 = tmp.x;
    const float m4s8 = tmp.y;
#else
    const float2 dm4f = __half22float2(dm4);
    const float2 ds8f = __half22float2(ds8);
    const float d4d8 = dm4f.x * ds8f.x;
    const float m4s8 = dm4f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_0_q8_1_impl(
    const int * vl, const int * vh, const int * u, const float & d5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

    const float2 ds8f = __half22float2(ds8);

    // second part effectively subtracts 16 from each quant value
    return d5 * (sumi * ds8f.x - (16*vdr/QI5_0) * ds8f.y);
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <int vdr> static __device__ __forceinline__ float vec_dot_q5_1_q8_1_impl(
    const int * vl, const int * vh, const int * u, const half2 & dm5, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        int vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); // SIMD dot product of quantized values

        int vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm5, ds8));
    const float d5d8 = tmp.x;
    const float m5s8 = tmp.y;
#else
    const float2 dm5f = __half22float2(dm5);
    const float2 ds8f = __half22float2(ds8);
    const float d5d8 = dm5f.x * ds8f.x;
    const float m5s8 = dm5f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    return sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const float & d8_0, const float & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * sumi;
}

template <int vdr> static __device__ __forceinline__ float vec_dot_q8_1_q8_1_impl(
    const int * v, const int * u, const half2 & dm8, const half2 & ds8) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_CUDA_F16
    const float2 tmp = __half22float2(__hmul2(dm8, ds8));
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
#else
    const float2 dm8f = __half22float2(dm8);
    const float2 ds8f = __half22float2(ds8);
    const float d8d8 = dm8f.x * ds8f.x;
    const float m8s8 = dm8f.y * ds8f.y;
#endif // GGML_CUDA_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    return sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q2_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = __half22float2(dm2);

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float & d8) {

    int sumi_d = 0;
    int sumi_m = 0;

#pragma unroll
    for (int i0 = 0; i0 < QI8_1; i0 += QI8_1/2) {
        int sumi_d_sc = 0;

        const int sc = scales[i0 / (QI8_1/2)];

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;

#pragma unroll
        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_d_sc = ggml_cuda_dp4a(v[i], u[i], sumi_d_sc); // SIMD dot product
            sumi_m    = ggml_cuda_dp4a(m,    u[i], sumi_m); // multiply sum of q8_1 values with m
        }

        sumi_d += sumi_d_sc * (sc & 0xF);
    }

    const float2 dm2f = __half22float2(dm2);

    return d8 * (dm2f.x*sumi_d - dm2f.y*sumi_m);
}

#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMQ  2

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        const int vi = __vsubss4(vil, vih);

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d3 * sumf;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d3, const float & d8) {

    int sumi = 0;

#pragma unroll
    for (int i0 = 0; i0 < QR3_K*VDR_Q3_K_Q8_1_MMQ; i0 += QI8_1/2) {
        int sumi_sc = 0;

        for (int i = i0; i < i0 + QI8_1/2; ++i) {
            sumi_sc = ggml_cuda_dp4a(v[i], u[i], sumi_sc); // SIMD dot product
        }

        sumi += sumi_sc * scales[i0 / (QI8_1/2)];
    }

    return d3*d8 * sumi;
}

#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q4_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K*VDR_Q4_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a((v[j] >> (4*i)) & 0x0F0F0F0F, u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const half2 * __restrict__ ds8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K*VDR_Q5_K_Q8_1_MMQ/QI8_1; ++i) {
        int sumi_d = 0;

#pragma unroll
        for (int j = 0; j < QI8_1; ++j) {
            sumi_d = ggml_cuda_dp4a(v[i*QI8_1 + j], u[i*QI8_1 + j], sumi_d); // SIMD dot product
        }

        const float2 ds8f = __half22float2(ds8[i]);

        sumf_d += ds8f.x * (sc[i] * sumi_d);
        sumf_m += ds8f.y *   m[i]; // sum of q8_1 block * q4_K min val
    }

    const float2 dm4f = __half22float2(dm4);

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

#define VDR_Q6_K_Q8_1_MMVQ 1
#define VDR_Q6_K_Q8_1_MMQ  8

// contiguous v/x values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

// contiguous u/y values
static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const int8_t * __restrict__ sc,
    const float & d6, const float * __restrict__ d8) {

    float sumf_d = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < VDR_Q6_K_Q8_1_MMQ; i0 += 4) {
        int2 sumi_d = {0, 0}; // 2 q6_K scales per q8_1 scale

#pragma unroll
        for (int i = i0; i < i0 + 2; ++i) {
            sumi_d.x = ggml_cuda_dp4a(v[2*i+0], u[2*i+0], sumi_d.x); // SIMD dot product
            sumi_d.x = ggml_cuda_dp4a(v[2*i+1], u[2*i+1], sumi_d.x); // SIMD dot product

            sumi_d.y = ggml_cuda_dp4a(v[2*i+4], u[2*i+4], sumi_d.y); // SIMD dot product
            sumi_d.y = ggml_cuda_dp4a(v[2*i+5], u[2*i+5], sumi_d.y); // SIMD dot product
        }

        sumf_d += d8[i0/4] * (sc[i0/2+0]*sumi_d.x + sc[i0/2+1]*sumi_d.y);
    }

    return d6 * sumf_d;
}

static __device__ __forceinline__ float vec_dot_q4_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq;

    int v[VDR_Q4_0_Q8_1_MMVQ];
    int u[2*VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]     = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}


static __device__ __forceinline__ float vec_dot_q4_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q4_1 * bq4_1 = (const block_q4_1 *) vbq;

    int v[VDR_Q4_1_Q8_1_MMVQ];
    int u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_0 * bq5_0 = (const block_q5_0 *) vbq;

    int vl[VDR_Q5_0_Q8_1_MMVQ];
    int vh[VDR_Q5_0_Q8_1_MMVQ];
    int  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q5_1 * bq5_1 = (const block_q5_1 *) vbq;

    int vl[VDR_Q5_1_Q8_1_MMVQ];
    int vh[VDR_Q5_1_Q8_1_MMVQ];
    int  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d, __low2half(bq8_1->ds));
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = bq3_K->d;

    const int vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + i].ds);
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#ifndef GGML_QKK_64
    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    // Branchless extraction lets nvcc hoist the scale work out of loops in
    // callers.  This is the current llama.cpp Q4_K MMVQ formulation.
    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);

#else

    const block_q4_K * bq4_K = (const block_q4_K *) vbq;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    uint16_t aux16[2];
    const uint8_t * s = (const uint8_t *)aux16;

    const uint16_t * a = (const uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    const float dall = bq4_K->dm[0];
    const float dmin = bq4_K->dm[1];

    const float d8_1 = __low2float(bq8_1[0].ds);
    const float d8_2 = __low2float(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * q4 = (const int *)bq4_K->qs + (iqs/2);
    const int v1 = q4[0];
    const int v2 = q4[4];

    const int dot1 = ggml_cuda_dp4a(ui2, v2 & 0x0f0f0f0f, ggml_cuda_dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    const int dot2 = ggml_cuda_dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, ggml_cuda_dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    const int dot3 = ggml_cuda_dp4a(0x01010101, ui2, ggml_cuda_dp4a(0x01010101, ui1, 0));
    const int dot4 = ggml_cuda_dp4a(0x01010101, ui4, ggml_cuda_dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    return dall * sumf_d - dmin * sumf_m;
#endif
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

#ifndef GGML_QKK_64
    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#else

    const block_q5_K * bq5_K = (const block_q5_K *) vbq;

    const int8_t * s = bq5_K->scales;

    const float d = bq5_K->d;

    const float d8_1 = __low2half(bq8_1[0].ds);
    const float d8_2 = __low2half(bq8_1[1].ds);

    const int ui1 = *((const int *)bq8_1[0].qs + (iqs/2));
    const int ui2 = *((const int *)bq8_1[0].qs + (iqs/2) + 4);
    const int ui3 = *((const int *)bq8_1[1].qs + (iqs/2));
    const int ui4 = *((const int *)bq8_1[1].qs + (iqs/2) + 4);

    const int * ql = (const int *)bq5_K->qs + (iqs/2);
    const int vl1 = ql[0];
    const int vl2 = ql[4];

    const int step = 4 * (iqs/2); // 0, 4, 8, 12
    const int im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    const int in = step%8; // 0, 4, 0, 4
    const int vh = (*((const int *)(bq5_K->qh + in))) >> im;

    const int v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    const int v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    const int v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    const int v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    const float sumf_d = d8_1 * (ggml_cuda_dp4a(ui1, v1, 0) * s[0] + ggml_cuda_dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (ggml_cuda_dp4a(ui3, v3, 0) * s[2] + ggml_cuda_dp4a(ui4, v4, 0) * s[3]);

    return d * sumf_d;
#endif
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_from_uint8(bq6_K->ql, iqs);
    const int vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

// https://github.com/ggerganov/llama.cpp/blob/c50a82ce0f71558cbb8e555146ba124251504b38/ggml-cuda/mmvq.cu#L4
typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs);

template <int ncols_y, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda,
          int nwarps_override = 0>
static __device__ void mul_mat_vec_q(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && (defined(RDNA2) || defined(RDNA3))
    constexpr int nwarps              = 1;
    constexpr int rows_per_cuda_block = 1;
#else
    constexpr int nwarps              = nwarps_override != 0 ? nwarps_override : (ncols_y <= 4 ? 4 : 2);
    constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__) && !defined(RDNA2) && !defined(RDNA3)

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

// partial sum for each thread
    float tmp[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp[j][i] += vec_dot_q_cuda(
                    &x[kbx + (row0 + i)*blocks_per_row_x], &y[j*blocks_per_col_y + kby], kqs);
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum(tmp[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block) {
            dst[j*nrows_dst + row0 + threadIdx.x] = tmp[j][threadIdx.x];
        }
    }
}

// batch size = 1
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda1(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<1, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// Volta/Turing (SM70/SM75) have a different latency/occupancy balance from
// Ampere and later.  Current llama.cpp uses two warps for single-token K-quants
// on these architectures, while SM80+ keeps four.  Separate entry points let
// the host select the tuning without baking one architecture into the PTX.
#define DECL_MMVQ_K_W2(NAME, QI, BLOCK, VDR, DOT)                                      \
extern "C" __global__ void NAME##_w2(                                                  \
    const void * vx, const void * vy, float * dst,                                      \
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {      \
    mul_mat_vec_q<1, QK_K, QI, BLOCK, VDR, DOT, 2>                                      \
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);                            \
}

DECL_MMVQ_K_W2(mul_mat_vec_q2_K_q8_1_cuda1, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
DECL_MMVQ_K_W2(mul_mat_vec_q3_K_q8_1_cuda1, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
DECL_MMVQ_K_W2(mul_mat_vec_q4_K_q8_1_cuda1, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
DECL_MMVQ_K_W2(mul_mat_vec_q5_K_q8_1_cuda1, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
DECL_MMVQ_K_W2(mul_mat_vec_q6_K_q8_1_cuda1, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)

#undef DECL_MMVQ_K_W2

// batch size = 2
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda2(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<2, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 3
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda3(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<3, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 4
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda4(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<4, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 5
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda5(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<5, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 6
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda6(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<6, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 7
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda7(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<7, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// batch size = 8
extern "C" __global__ void mul_mat_vec_q4_0_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_1_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_0_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_1_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q8_0_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q2_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q3_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q4_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q5_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

extern "C" __global__ void mul_mat_vec_q6_K_q8_1_cuda8(
    const void * vx, const void * vy, float * dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    mul_mat_vec_q<8, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (vx, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst);
}

// Fused SwiGLU matvec: dst = silu(gate·y) * (up·y), both weight matrices
// walked in lockstep over one shared quantized-activation buffer, SiLU and
// the elementwise product applied in the epilogue. Same launch geometry and
// tuning parameters as mul_mat_vec_q (which is at parity with upstream
// llama.cpp's mmvq for these shapes; llama.cpp ships the same fusion as
// mmvq's `has_fusion` path). Replaces, per FFN call: one activation
// quantization, one kernel launch, the separate SiLU and mul kernels, and
// one intermediate-activation round trip.
template <int ncols_y, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __device__ void mul_mat_vec_q_glu(
    const void * __restrict__ vgate, const void * __restrict__ vup,
    const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) {

    constexpr int nwarps              = ncols_y <= 4 ? 4 : 2;
    constexpr int rows_per_cuda_block = ncols_y == 1 ? 1 : 2;

    const     int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    const     int blocks_per_col_y = nrows_y / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps*WARP_SIZE / qi;

    float tmp_g[ncols_y][rows_per_cuda_block] = {0.0f};
    float tmp_u[ncols_y][rows_per_cuda_block] = {0.0f};

    const block_q_t  * xg = (const block_q_t  *) vgate;
    const block_q_t  * xu = (const block_q_t  *) vup;
    const block_q8_1 * y  = (const block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_g[j][i] += vec_dot_q_cuda(
                    &xg[kbx + (row0 + i)*blocks_per_row_x], &y[j*blocks_per_col_y + kby], kqs);
                tmp_u[j][i] += vec_dot_q_cuda(
                    &xu[kbx + (row0 + i)*blocks_per_row_x], &y[j*blocks_per_col_y + kby], kqs);
            }
        }
    }

    __shared__ float smem_g[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    __shared__ float smem_u[nwarps-1 > 0 ? nwarps-1 : 1][ncols_y][rows_per_cuda_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                smem_g[threadIdx.y-1][j][i][threadIdx.x] = tmp_g[j][i];
                smem_u[threadIdx.y-1][j][i][threadIdx.x] = tmp_u[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < ncols_y; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp_g[j][i] += smem_g[l][j][i][threadIdx.x];
                tmp_u[j][i] += smem_u[l][j][i][threadIdx.x];
            }
            tmp_g[j][i] = warp_reduce_sum(tmp_g[j][i]);
            tmp_u[j][i] = warp_reduce_sum(tmp_u[j][i]);
        }

        if (threadIdx.x < rows_per_cuda_block) {
            const float g = tmp_g[j][threadIdx.x];
            const float u = tmp_u[j][threadIdx.x];
            const float silu_g = g / (1.0f + __expf(-g));
            dst[j*nrows_dst + row0 + threadIdx.x] = silu_g * u;
        }
    }
}

#define MMVQ_GLU_Q4K_ENTRY(N) \
extern "C" __global__ void mul_mat_vec_q4_K_q8_1_glu_cuda##N( \
    const void * vgate, const void * vup, const void * vy, float * dst, \
    const int ncols_x, const int nrows_x, const int nrows_y, const int nrows_dst) { \
    mul_mat_vec_q_glu<N, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1> \
        (vgate, vup, vy, dst, ncols_x, nrows_x, nrows_y, nrows_dst); \
}

MMVQ_GLU_Q4K_ENTRY(1)
MMVQ_GLU_Q4K_ENTRY(2)
MMVQ_GLU_Q4K_ENTRY(3)
MMVQ_GLU_Q4K_ENTRY(4)
MMVQ_GLU_Q4K_ENTRY(5)
MMVQ_GLU_Q4K_ENTRY(6)
MMVQ_GLU_Q4K_ENTRY(7)
MMVQ_GLU_Q4K_ENTRY(8)

#undef MMVQ_GLU_Q4K_ENTRY

extern "C" __global__ void quantize_q8_1(const float * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    const int ix = blockDim.x*blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int iy = blockDim.y*blockIdx.y + threadIdx.y;

    const int i_padded = iy*kx_padded + ix;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int ib = i_padded / QK8_1; // block index
    const int iqs = i_padded % QK8_1; // quant index

    const float xi = ix < kx ? x[iy*kx + ix] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

// Kernels from https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/mmq.cu

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int  tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI5_0) + mmq_y/QI5_0];

    *x_ql = tile_x_ql;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_0;
    const int kqsx = k % QI5_0;

    const block_q5_0 * bx0 = (const block_q5_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8(bxi->qs, kqsx);
        const int qh = get_int_from_uint8(bxi->qh, 0) >> (4 * (k % QI5_0));

        int qs0 = (ql >>  0)   & 0x0F0F0F0F;
        qs0    |= (qh <<  4)   & 0x00000010;  // 0 ->  4
        qs0    |= (qh << 11)   & 0x00001000;  // 1 -> 12
        qs0    |= (qh << 18)   & 0x00100000;  // 2 -> 20
        qs0    |= (qh << 25)   & 0x10000000;  // 3 -> 28
        qs0     = __vsubss4(qs0, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4)   & 0x0F0F0F0F;
        qs1    |= (qh >> 12)   & 0x00000010;  // 16 ->  4
        qs1    |= (qh >>  5)   & 0x00001000;  // 17 -> 12
        qs1    |= (qh <<  2)   & 0x00100000;  // 18 -> 20
        qs1    |= (qh <<  9)   & 0x10000000;  // 19 -> 28
        qs1     = __vsubss4(qs1, 0x10101010); // subtract 16

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_0;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_0) {
        int i = i0 + i_offset * QI5_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI5_0) + i / QI5_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q5_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_0) + i/QI5_0 + k/QI5_0;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    int u[2*VDR_Q5_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_0) % WARP_SIZE];
    }

    return vec_dot_q8_0_q8_1_impl<QR5_0*VDR_Q5_0_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dmf[index_bx], y_df[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}


template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_1(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_1) + mmq_y/QI5_1];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_1(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset < nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_1;
    const int kqsx = k % QI5_1;

    const block_q5_1 * bx0 = (const block_q5_1 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int qh = get_int_from_uint8_aligned(bxi->qh, 0) >> (4 * (k % QI5_1));

        int qs0 = (ql >>  0) & 0x0F0F0F0F;
        qs0    |= (qh <<  4) & 0x00000010; // 0 ->  4
        qs0    |= (qh << 11) & 0x00001000; // 1 -> 12
        qs0    |= (qh << 18) & 0x00100000; // 2 -> 20
        qs0    |= (qh << 25) & 0x10000000; // 3 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+0] = qs0;

        int qs1 = (ql >>  4) & 0x0F0F0F0F;
        qs1    |= (qh >> 12) & 0x00000010; // 16 ->  4
        qs1    |= (qh >>  5) & 0x00001000; // 17 -> 12
        qs1    |= (qh <<  2) & 0x00100000; // 18 -> 20
        qs1    |= (qh <<  9) & 0x10000000; // 19 -> 28

        x_ql[i * (2*WARP_SIZE + 1) + 2*k+1] = qs1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_1;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_1) {
        int i = i0 + i_offset * QI5_1 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_1 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI5_1) + i / QI5_1 + kbxd] = bxi->dm;
    }
}

static __device__ __forceinline__ float vec_dot_q5_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const int index_bx = i * (WARP_SIZE/QI5_1) + + i/QI5_1 + k/QI5_1;

    int u[2*VDR_Q5_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q5_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI5_1) % WARP_SIZE];
    }

    return vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
        (&x_ql[i * (2*WARP_SIZE + 1) + 2 * k], u, x_dm[index_bx], y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q8_0(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    __shared__ int  tile_x_qs[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ float tile_x_d[mmq_y * (WARP_SIZE/QI8_0) + mmq_y/QI8_0];

    *x_ql = tile_x_qs;
    *x_dm = (half2 *) tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q8_0(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI8_0;
    const int kqsx = k % QI8_0;
    float * x_dmf = (float *) x_dm;

    const block_q8_0 * bx0 = (const block_q8_0 *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_int8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI8_0;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI8_0) {
        int i = i0 + i_offset * QI8_0 + k / blocks_per_tile_x_row;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q8_0 * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI8_0) + i / QI8_0 + kbxd] = bxi->d;
    }
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[j * WARP_SIZE + k], x_dmf[i * (WARP_SIZE/QI8_0) + i/QI8_0 + k/QI8_0],
         y_df[j * (WARP_SIZE/QI8_1) + k/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q2_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI2_K) + mmq_y/QI2_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q2_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI2_K;
    const int kqsx = k % QI2_K;

    const block_q2_K * bx0 = (const block_q2_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI2_K;
    const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI2_K) {
        int i = (i0 + i_offset * QI2_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dm[i * (WARP_SIZE/QI2_K) + i / QI2_K + kbxd] = bxi->dm;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q2_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI2_K/4);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = get_int_from_uint8_aligned(bxi->scales, k % (QI2_K/4));
    }
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const int kbx = k / QI2_K;
    const int ky  = (k % QI2_K) * QR2_K;
    const float * y_df = (const float *) y_ds;

    int v[QR2_K*VDR_Q2_K_Q8_1_MMQ];

    const int kqsx = i * (WARP_SIZE + 1) + kbx*QI2_K + (QI2_K/2) * (ky/(2*QI2_K)) + ky % (QI2_K/2);
    const int shift = 2 * ((ky % (2*QI2_K)) / (QI2_K/2));

#pragma unroll
    for (int l = 0; l < QR2_K*VDR_Q2_K_Q8_1_MMQ; ++l) {
        v[l] = (x_ql[kqsx + l] >> shift) & 0x03030303;
    }

    const uint8_t * scales = ((const uint8_t *) &x_sc[i * (WARP_SIZE/4) + i/4 + kbx*4]) + ky/4;

    const int index_y = j * WARP_SIZE + (QR2_K*k) % WARP_SIZE;
    return vec_dot_q2_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dm[i * (WARP_SIZE/QI2_K) + i/QI2_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q3_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI3_K) + mmq_y/QI3_K];
    __shared__ int   tile_x_qh[mmq_y * (WARP_SIZE/2)     + mmq_y/2];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/4)     + mmq_y/4];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_qh = tile_x_qh;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q3_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI3_K;
    const int kqsx = k % QI3_K;

    const block_q3_K * bx0 = (const block_q3_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI3_K;
    const int kbxd = k % blocks_per_tile_x_row;
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI3_K) {
        int i = (i0 + i_offset * QI3_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI3_K) + i / QI3_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 2) {
        int i = i0 + i_offset * 2 + k / (WARP_SIZE/2);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/2)) / (QI3_K/2);

        // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
        x_qh[i * (WARP_SIZE/2) + i / 2 + k % (WARP_SIZE/2)] = ~get_int_from_uint8(bxi->hmask, k % (QI3_K/2));
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
        int i = i0 + i_offset * 4 + k / (WARP_SIZE/4);

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q3_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/4)) / (QI3_K/4);

        const int ksc = k % (QI3_K/4);

        const int ksc_low = ksc % (QI3_K/8);
        const int shift_low = 4 * (ksc / (QI3_K/8));
        const int sc_low = (get_int_from_uint8(bxi->scales, ksc_low) >> shift_low) & 0x0F0F0F0F;

        const int ksc_high = QI3_K/8;
        const int shift_high = 2 * ksc;
        const int sc_high = ((get_int_from_uint8(bxi->scales, ksc_high) >> shift_high) << 4) & 0x30303030;

        const int sc = __vsubss4(sc_low | sc_high, 0x20202020);

        x_sc[i * (WARP_SIZE/4) + i / 4 + k % (WARP_SIZE/4)] = sc;
    }
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const int kbx  = k / QI3_K;
    const int ky  = (k % QI3_K) * QR3_K;
    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * scales = ((const int8_t *) (x_sc + i * (WARP_SIZE/4) + i/4 + kbx*4)) + ky/4;

    int v[QR3_K*VDR_Q3_K_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < QR3_K*VDR_Q3_K_Q8_1_MMQ; ++l) {
        const int kqsx = i * (WARP_SIZE + 1) + kbx*QI3_K + (QI3_K/2) * (ky/(2*QI3_K)) + ky % (QI3_K/2);
        const int shift = 2 * ((ky % 32) / 8);
        const int vll = (x_ql[kqsx + l] >> shift) & 0x03030303;

        const int vh = x_qh[i * (WARP_SIZE/2) + i/2 + kbx * (QI3_K/2) + (ky+l)%8] >> ((ky+l) / 8);
        const int vlh = (vh << 2) & 0x04040404;

        v[l] = __vsubss4(vll, vlh);
    }

    const int index_y = j * WARP_SIZE + (k*QR3_K) % WARP_SIZE;
    return vec_dot_q3_K_q8_1_impl_mmq(v, &y_qs[index_y], scales, x_dmf[i * (WARP_SIZE/QI3_K) + i/QI3_K + kbx], y_df[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q4_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (WARP_SIZE)       + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI4_K) + mmq_y/QI4_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q4_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI4_K; // == 0 if QK_K == 256
    const int kqsx = k % QI4_K; // == k if QK_K == 256

    const block_q4_K * bx0 = (const block_q4_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbx;

        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI4_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_K) {
        int i = (i0 + i_offset * QI4_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = bxi->dm;
#else
        x_dm[i * (WARP_SIZE/QI4_K) + i / QI4_K + kbxd] = {bxi->dm[0], bxi->dm[1]};
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q4_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI4_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2*((k % 16) / 8);

    const int index_y = j * WARP_SIZE + (QR4_K*k) % WARP_SIZE;
    return vec_dot_q4_K_q8_1_impl_mmq(&x_ql[i * (WARP_SIZE + 1) + k], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI4_K) + i/QI4_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q5_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI5_K) + mmq_y/QI5_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q5_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI5_K; // == 0 if QK_K == 256
    const int kqsx = k % QI5_K; // == k if QK_K == 256

    const block_q5_K * bx0 = (const block_q5_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR5_K*kqsx;

        const int ql = get_int_from_uint8_aligned(bxi->qs, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8_aligned(bxi->qh, kqsx % (QI5_K/4));
        const int qh0 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 0)) << 4) & 0x10101010;
        const int qh1 = ((qh >> (2 * (kqsx / (QI5_K/4)) + 1)) << 4) & 0x10101010;

        const int kq0 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + 0;
        const int kq1 = ky - ky % (QI5_K/2) + k % (QI5_K/4) + (QI5_K/4);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = ql0 | qh0;
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = ql1 | qh1;
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI5_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI5_K) {
        int i = (i0 + i_offset * QI5_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + kbxd;

#if QK_K == 256
        x_dm[i * (WARP_SIZE/QI5_K) + i / QI5_K + kbxd] = bxi->dm;
#endif
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q5_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / (QI5_K/8);

        const int * scales = (const int *) bxi->scales;

        const int ksc = k % (WARP_SIZE/8);

        // scale arrangement after the following two lines: sc0,...,sc3, sc4,...,sc7, m0,...,m3, m4,...,m8
        int scales8 = (scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F; // lower 4 bits
        scales8    |= (scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030; // upper 2 bits

        x_sc[i * (WARP_SIZE/8) + i / 8 + ksc] = scales8;
    }
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const uint8_t * sc = ((const uint8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/16]) + 2 * ((k % 16) / 8);

    const int index_x = i * (QR5_K*WARP_SIZE + 1) +  QR5_K*k;
    const int index_y = j * WARP_SIZE             + (QR5_K*k) % WARP_SIZE;
    return vec_dot_q5_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, sc+8,
                                      x_dm[i * (WARP_SIZE/QI5_K) + i/QI5_K], &y_ds[index_y/QI8_1]);
}

template <int mmq_y> static __device__ __forceinline__ void allocate_tiles_q6_K(int ** x_ql, half2 ** x_dm, int ** x_qh, int ** x_sc) {
    GGML_UNUSED(x_qh);

    __shared__ int   tile_x_ql[mmq_y * (2*WARP_SIZE)     + mmq_y];
    __shared__ half2 tile_x_dm[mmq_y * (WARP_SIZE/QI6_K) + mmq_y/QI6_K];
    __shared__ int   tile_x_sc[mmq_y * (WARP_SIZE/8)     + mmq_y/8];

    *x_ql = tile_x_ql;
    *x_dm = tile_x_dm;
    *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check> static __device__ __forceinline__ void load_tiles_q6_K(
    const void * __restrict__ vx, int * __restrict__ x_ql, half2 * __restrict__ x_dm, int * __restrict__ x_qh,
    int * __restrict__ x_sc, const int & i_offset, const int & i_max, const int & k, const int & blocks_per_row) {
    GGML_UNUSED(x_qh);

    GGML_CUDA_ASSUME(i_offset >= 0);
    GGML_CUDA_ASSUME(i_offset <  nwarps);
    GGML_CUDA_ASSUME(k >= 0);
    GGML_CUDA_ASSUME(k <  WARP_SIZE);

    const int kbx  = k / QI6_K; // == 0 if QK_K == 256
    const int kqsx = k % QI6_K; // == k if QK_K == 256

    const block_q6_K * bx0 = (const block_q6_K *) vx;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbx;
        const int ky = QR6_K*kqsx;

        const int ql = get_int_from_uint8(bxi->ql, kqsx);
        const int ql0 = (ql >> 0) & 0x0F0F0F0F;
        const int ql1 = (ql >> 4) & 0x0F0F0F0F;

        const int qh = get_int_from_uint8(bxi->qh, (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4));
        const int qh0 = ((qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4)))) << 4) & 0x30303030;
        const int qh1 =  (qh >> (2 * ((kqsx % (QI6_K/2)) / (QI6_K/4))))       & 0x30303030;

        const int kq0 = ky - ky % QI6_K + k % (QI6_K/2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K/2) + (QI6_K/2);

        x_ql[i * (2*WARP_SIZE + 1) + kq0] = __vsubss4(ql0 | qh0, 0x20202020);
        x_ql[i * (2*WARP_SIZE + 1) + kq1] = __vsubss4(ql1 | qh1, 0x20202020);
    }

    const int blocks_per_tile_x_row = WARP_SIZE / QI6_K; // == 1 if QK_K == 256
    const int kbxd = k % blocks_per_tile_x_row;          // == 0 if QK_K == 256
    float * x_dmf = (float *) x_dm;

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI6_K) {
        int i = (i0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + kbxd;

        x_dmf[i * (WARP_SIZE/QI6_K) + i / QI6_K + kbxd] = bxi->d;
    }

#pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
        int i = (i0 + i_offset * 8 + k / (WARP_SIZE/8)) % mmq_y;

        if (need_check) {
            i = min(i, i_max);
        }

        const block_q6_K * bxi = bx0 + i*blocks_per_row + (k % (WARP_SIZE/8)) / 4;

        x_sc[i * (WARP_SIZE/8) + i / 8 + k % (WARP_SIZE/8)] = get_int_from_int8(bxi->scales, k % (QI6_K/8));
    }
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh);

    const float * x_dmf = (const float *) x_dm;
    const float * y_df  = (const float *) y_ds;

    const int8_t * sc = ((const int8_t *) &x_sc[i * (WARP_SIZE/8) + i/8 + k/8]);

    const int index_x = i * (QR6_K*WARP_SIZE + 1) +  QR6_K*k;
    const int index_y = j * WARP_SIZE             + (QR6_K*k) % WARP_SIZE;
    return vec_dot_q6_K_q8_1_impl_mmq(&x_ql[index_x], &y_qs[index_y], sc, x_dmf[i * (WARP_SIZE/QI6_K) + i/QI6_K], &y_df[index_y/QI8_1]);
}


static __device__ __forceinline__ float vec_dot_q4_0_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));
    const float * x_dmf = (const float *) x_dm;

    int u[2*VDR_Q4_0_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_0) % WARP_SIZE];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}

static __device__ __forceinline__ float vec_dot_q4_1_q8_1_mul_mat(
    const int * __restrict__ x_ql, const half2 * __restrict__ x_dm, const int * __restrict__ x_qh, const int * __restrict__ x_sc,
    const int * __restrict__ y_qs, const half2 * __restrict__ y_ds, const int & i, const int & j, const int & k) {
    GGML_UNUSED(x_qh); GGML_UNUSED(x_sc);

    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_1_Q8_1_MMQ];

#pragma unroll
    for (int l = 0; l < VDR_Q4_1_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * WARP_SIZE + (kyqs + l)         % WARP_SIZE];
        u[2*l+1] = y_qs[j * WARP_SIZE + (kyqs + l + QI4_1) % WARP_SIZE];
    }

    return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
        (&x_ql[i * (WARP_SIZE + 1) + k], u, x_dm[i * (WARP_SIZE/QI4_1) + i/QI4_1 + k/QI4_1],
         y_ds[j * (WARP_SIZE/QI8_1) + (2*k/QI8_1) % (WARP_SIZE/QI8_1)]);
}


extern "C" __global__ void
    mul_mat_q4_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_0_AMPERE;
    const int nwarps = NWARPS_Q4_0_AMPERE;

    mul_mat_q<QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps, allocate_tiles_q4_0<mmq_y>,
        load_tiles_q4_0<mmq_y, nwarps, true>, VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q4_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_1_AMPERE;
    const int nwarps = NWARPS_Q4_1_AMPERE;

    mul_mat_q<QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps, allocate_tiles_q4_1<mmq_y>,
        load_tiles_q4_1<mmq_y, nwarps, true>, VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}


extern "C" __global__ void
    mul_mat_q5_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_0_AMPERE;
    const int nwarps = NWARPS_Q5_0_AMPERE;

    mul_mat_q<QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps, allocate_tiles_q5_0<mmq_y>,
        load_tiles_q5_0<mmq_y, nwarps, true>, VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
mul_mat_q5_1(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_1_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_1_AMPERE;
    const int nwarps = NWARPS_Q5_1_AMPERE;

    mul_mat_q<QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps, allocate_tiles_q5_1<mmq_y>,
        load_tiles_q5_1<mmq_y, nwarps, true>, VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q8_0(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q8_0_AMPERE;
    const int mmq_y  =  MMQ_Y_Q8_0_AMPERE;
    const int nwarps = NWARPS_Q8_0_AMPERE;

    mul_mat_q<QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps, allocate_tiles_q8_0<mmq_y>,
        load_tiles_q8_0<mmq_y, nwarps, true>, VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
mul_mat_q2_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q2_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q2_K_AMPERE;
    const int nwarps = NWARPS_Q2_K_AMPERE;
    mul_mat_q<QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps, allocate_tiles_q2_K<mmq_y>,
        load_tiles_q2_K<mmq_y, nwarps, true>, VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q3_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q3_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q3_K_AMPERE;
    const int nwarps = NWARPS_Q3_K_AMPERE;
    mul_mat_q<QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps, allocate_tiles_q3_K<mmq_y>,
        load_tiles_q3_K<mmq_y, nwarps, true>, VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q4_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q4_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q4_K_AMPERE;
    const int nwarps = NWARPS_Q4_K_AMPERE;
    mul_mat_q<QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps, allocate_tiles_q4_K<mmq_y>,
        load_tiles_q4_K<mmq_y, nwarps, true>, VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
mul_mat_q5_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q5_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q5_K_AMPERE;
    const int nwarps = NWARPS_Q5_K_AMPERE;
    mul_mat_q<QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps, allocate_tiles_q5_K<mmq_y>,
        load_tiles_q5_K<mmq_y, nwarps, true>, VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

extern "C" __global__ void
    mul_mat_q6_K(
    const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y, const int nrows_dst) {
    const int mmq_x  =  MMQ_X_Q6_K_AMPERE;
    const int mmq_y  =  MMQ_Y_Q6_K_AMPERE;
    const int nwarps = NWARPS_Q6_K_AMPERE;
    mul_mat_q<QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps, allocate_tiles_q6_K<mmq_y>,
        load_tiles_q6_K<mmq_y, nwarps, true>, VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>
        (vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

typedef struct {
    half    d;             // super-block scales/mins
    half    dmin;
    uint8_t scales[3*QK_K/64];
    uint8_t qs[QK_K/2];        // 4--bit quants
} block_q4_K_;

// Unpacks the 6-bit scale and 6-bit min for a sub-block.
__device__ void get_scale_min_k4_(int j, const uint8_t* __restrict__ q, uint8_t* __restrict__ d, uint8_t* __restrict__ m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j] >> 6) << 4);
    }
}

/**
 * @brief CUDA kernel for quantizing float values into 4-bit blocks using a K-block format.
 *
 * This kernel processes multiple input blocks in parallel and performs:
 *   1. Finding per-sub-block min/max values
 *   2. Computing scales and offsets
 *   3. Packing scales/mins into compressed form
 *   4. Quantizing floats to 4-bit integers
 *   5. Packing quantized values into bytes
 *
 * The quantization follows the formula:
 *   dequantized_value = scale * quantized_value - min
 * where min is stored as a positive offset.
 *
 * @author
 *   Guoqing Bao
 *   Part of the project: https://github.com/guoqingbao/vllm.rs/
 * @param[in]  x           Pointer to the input float array of size `num_blocks * QK_K`.
 *                         Each block contains QK_K floats, subdivided into 8 sub-blocks of 32 elements.
 * @param[out] y           Pointer to the output buffer of type `block_q4_K_[]`.
 *                         Stores quantized values, packed scales, and metadata.
 * @param[in]  num_blocks  Total number of QK_K-sized blocks to process.
 *
 * @note
 *   - This kernel uses a grid-stride loop for flexible execution configuration.
 *   - The packing scheme matches the reference Rust implementation for bit layout.
 */

// Canonical K-quant helper matching Candle's CPU `make_qkx1_quants`.
// Rust f32::round() is ties-away-from-zero.  Do not use CUDA's generic
// roundf here: its lowered instruction may use the device's tie-to-even mode
// on exact half-way values, which changes Q6_K levels for regularly spaced
// negative inputs.
__device__ __forceinline__ int kquant_nearest_int(float x) {
    return (int)(x >= 0.0f ? floorf(x + 0.5f) : ceilf(x - 0.5f));
}

__device__ void make_qkx1_quants_cuda(
    const float* x, const int nmax, float* scale_out, float* min_out) {
    int l[32] = {0};
    float min_val = x[0];
    float max_val = x[0];
    for (int i = 1; i < 32; ++i) {
        min_val = fminf(min_val, x[i]);
        max_val = fmaxf(max_val, x[i]);
    }
    if (max_val == min_val) {
        *scale_out = 0.0f;
        *min_out = 0.0f;
        return;
    }
    min_val = fminf(min_val, 0.0f);
    float iscale = nmax / (max_val - min_val);
    float scale = 1.0f / iscale;
    for (int it = 0; it < 5; ++it) {
        float sumlx = 0.0f;
        float suml2 = 0.0f;
        bool changed = false;
        for (int i = 0; i < 32; ++i) {
            int li = max(0, min(nmax, kquant_nearest_int(iscale * (x[i] - min_val))));
            changed |= li != l[i];
            l[i] = li;
            sumlx += (x[i] - min_val) * li;
            suml2 += li * li;
        }
        if (suml2 == 0.0f) {
            *scale_out = 0.0f;
            *min_out = 0.0f;
            return;
        }
        scale = sumlx / suml2;
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) sum += x[i] - scale * l[i];
        min_val = fminf(sum / 32.0f, 0.0f);
        iscale = 1.0f / scale;
        if (!changed) break;
    }
    *scale_out = scale;
    *min_out = -min_val;
}

extern "C" __global__ void quantize_q4_k(
    const float* __restrict__ x, void* __restrict__ y,
    const int num_blocks, const int elem_count) {
    const int block_id = blockIdx.x;
    if (threadIdx.x != 0 || block_id >= num_blocks) return;
    const float* x_block = x + block_id * QK_K;
    block_q4_K_* y_block = reinterpret_cast<block_q4_K_*>(y) + block_id;

    float scales_f[8], mins_f[8];
    for (int i = 0; i < 8; ++i) {
        make_qkx1_quants_cuda(x_block + i * 32, 15, &scales_f[i], &mins_f[i]);
    }
    float max_scale = 0.0f, max_min = 0.0f;
    for (int i = 0; i < 8; ++i) {
        max_scale = fmaxf(max_scale, scales_f[i]);
        max_min = fmaxf(max_min, mins_f[i]);
    }
    const float inv_scale = max_scale > 0.0f ? 63.0f / max_scale : 0.0f;
    const float inv_min = max_min > 0.0f ? 63.0f / max_min : 0.0f;
    uint8_t packed[K_SCALE_SIZE] = {0};
    for (int j = 0; j < 8; ++j) {
        const uint8_t ls = (uint8_t)min(63, kquant_nearest_int(inv_scale * scales_f[j]));
        const uint8_t lm = (uint8_t)min(63, kquant_nearest_int(inv_min * mins_f[j]));
        if (j < 4) {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }
    for (int j = 0; j < K_SCALE_SIZE; ++j) y_block->scales[j] = packed[j];
    y_block->d = __float2half(max_scale / 63.0f);
    y_block->dmin = __float2half(max_min / 63.0f);

    uint8_t l[QK_K] = {0};
    const int valid = elem_count - block_id * QK_K;
    for (int j = 0; j < 8; ++j) {
        uint8_t sc, mn;
        get_scale_min_k4_(j, packed, &sc, &mn);
        const float d = __half2float(y_block->d) * sc;
        const float dm = __half2float(y_block->dmin) * mn;
        if (d != 0.0f) {
            for (int ii = 0; ii < 32; ++ii) {
                const int idx = j * 32 + ii;
                const float value = idx < valid ? x_block[idx] : 0.0f;
                l[idx] = (uint8_t)max(0, min(15, kquant_nearest_int((value + dm) / d)));
            }
        }
    }
    for (int j = 0; j < QK_K; j += 64) {
        const int offset = (j / 64) * 32;
        for (int i = 0; i < 32; ++i) y_block->qs[offset + i] = l[j + i] | (l[j + 32 + i] << 4);
    }
}

// CUDA kernel for quantizing a matrix of floats to the Q8_0 format.
//
// @author
//   Guoqing Bao
//  Part of the project: https://github.com/guoqingbao/vllm.rs/
// Parameters:
//   x:          Source matrix of f32 values. Assumed to be in contiguous row-major layout.
//   y:          Destination buffer for the quantized q8_0 blocks.
//   k:          The number of elements (columns) in a single row of the source matrix.
//   kx_padded:  The padded row size, which must be a multiple of QK8_0.
extern "C" __global__ void quantize_q8_0(
    const float * __restrict__ x,
    void * __restrict__ y,
    const int k,
    const int kx_padded
) {
    // blockIdx.x corresponds to the block index within the row.
    // blockIdx.y corresponds to the row index.
    // threadIdx.x corresponds to the thread index within the data block (0-31).
    const int block_idx = blockIdx.x;
    const int row_idx   = blockIdx.y;
    const int tid       = threadIdx.x;

    // --- Source Pointer Calculation ---
    // This calculation assumes the source tensor 'x' is a contiguous row-major matrix.
    // The offset to the start of the current row is (row_idx * k).
    // The offset to the start of the current block within that row is (block_idx * QK8_0).
    const float * const x_block_start = x + (size_t)row_idx * k + (size_t)block_idx * QK8_0;

    // --- Destination Pointer Calculation ---
    // The destination buffer has a different stride based on the padded row size.
    // The size of one quantized row in bytes is the number of blocks per padded row
    // multiplied by the size of a single q8_0 block.
    const size_t dst_row_size_bytes = (size_t)(kx_padded / QK8_0) * sizeof(block_q8_0);
    block_q8_0 * const y_block = (block_q8_0 *)((uint8_t*)y + (size_t)row_idx * dst_row_size_bytes) + block_idx;

    // Determine the global element index this thread is responsible for within the original row.
    const int current_idx_in_row = block_idx * QK8_0 + tid;

    // Find the absolute maximum value (amax) in the block ---

    // Statically allocated shared memory for the parallel reduction.
    __shared__ float sdata[QK8_0];

    // Each thread loads one float into shared memory.
    // If the index is outside the original row size 'k', it's padding; load 0.
    if (current_idx_in_row < k) {
        sdata[tid] = fabsf(x_block_start[tid]);
    } else {
        sdata[tid] = 0.0f; // Padding threads must participate with a neutral value.
    }
    __syncthreads();

    // Perform parallel reduction in shared memory to find the max value.
    for (int s = QK8_0 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Calculate and store the scaling factor 'd' ---

    // Use shared memory to hold the final scale 'd' to ensure visibility for all threads.
    __shared__ float d_shared;
    if (tid == 0) {
        const float amax = sdata[0];
        // This logic now matches the CPU implementation exactly.
        // If amax is 0, d will be 0. Otherwise, d = amax / 127.0.
        d_shared = amax / 127.0f;
        y_block->d = __float2half(d_shared);
    }
    __syncthreads(); // Ensure d_shared is written before other threads read it.

    // Quantize and store the 8-bit values ---

    // Calculate the inverse scale. If d is 0, iscale will be 0.
    const float iscale = (d_shared != 0.0f) ? 1.0f / d_shared : 0.0f;

    // Each thread quantizes one value. Handle padding.
    if (current_idx_in_row < k) {
        const float val = x_block_start[tid];
        y_block->qs[tid] = roundf(val * iscale);
    } else {
        y_block->qs[tid] = 0; // Write 0 for padded elements.
    }
}


/**
 * @brief CUDA kernel to quantize one block of 256 floats into the Q6_K format.
 *
 * Each CUDA block of 64 threads processes one Q6_K block (256 floats).
 * @author
 *   Guoqing Bao
 *   Part of the project: https://github.com/guoqingbao/vllm.rs/
 * @param xx   Input tensor pointer (float*).
 * @param vy   Output tensor pointer (void*, assumed to be block_q6_K*).
 * @param n    Number of rows in the input tensor.
 * @param k    Number of columns (features) in the input tensor (must be a multiple of QK_K).
 */

// Canonical Q6_K quantization matching Candle's CPU make_qx_quants(16,32,1).
__device__ float make_qx_quants_q6_cuda(const float* x, int8_t* levels) {
    float max_value = 0.0f, amax = 0.0f;
    for (int i = 0; i < 16; ++i) {
        const float value = x[i];
        if (fabsf(value) > amax) { amax = fabsf(value); max_value = value; }
    }
    if (amax == 0.0f) {
        for (int i = 0; i < 16; ++i) levels[i] = 32;
        return 0.0f;
    }
    // Keep the CPU reference's scalar f32 operation order.  Allowing the
    // compiler to contract these expressions into FMAs changes the discrete
    // level-search decisions on near-boundary Q6_K sub-blocks.
    float iscale = __fdiv_rn(-32.0f, max_value);
    float sumlx = 0.0f, suml2 = 0.0f;
    for (int i = 0; i < 16; ++i) {
        int l = max(-32, min(31, kquant_nearest_int(iscale * x[i])));
        levels[i] = (int8_t)(l + 32);
        const float w = __fmul_rn(x[i], x[i]);
        // Candle's CPU make_qx_quants uses the weighted least-squares
        // numerator w*x*l (rmse_type=1 means w=x*x).  Omitting the x here
        // changes every sub-block scale and produces materially wrong Q6_K
        // tensors even though the packed layout is otherwise valid.
        sumlx = __fadd_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
        suml2 = __fadd_rn(suml2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
    }
    float scale = suml2 != 0.0f ? __fdiv_rn(sumlx, suml2) : 0.0f;
    float best = __fmul_rn(scale, sumlx);
    for (int it = 0; it < 3 && scale != 0.0f; ++it) {
        iscale = __fdiv_rn(1.0f, scale);
        float slx = 0.0f, sl2 = 0.0f;
        bool changed = false;
        for (int i = 0; i < 16; ++i) {
            const int old_l = (int)levels[i] - 32;
            const int l = max(-32, min(31, kquant_nearest_int(iscale * x[i])));
            changed |= l != old_l;
            const float w = __fmul_rn(x[i], x[i]);
            slx = __fadd_rn(slx, __fmul_rn(__fmul_rn(w, x[i]), (float)l));
            sl2 = __fadd_rn(sl2, __fmul_rn(__fmul_rn(w, (float)l), (float)l));
        }
        if (!changed || sl2 == 0.0f || __fmul_rn(slx, slx) <= __fmul_rn(best, sl2)) break;
        for (int i = 0; i < 16; ++i) {
            levels[i] = (int8_t)(32 + max(-32, min(31, kquant_nearest_int(iscale * x[i]))));
        }
        sumlx = slx;
        suml2 = sl2;
        scale = __fdiv_rn(sumlx, suml2);
        best = __fmul_rn(scale, sumlx);
    }
    for (int it = 0; it < 5 && suml2 != 0.0f; ++it) {
        int changed_count = 0;
        for (int i = 0; i < 16; ++i) {
            const float w = __fmul_rn(x[i], x[i]);
            const int old_l = (int)levels[i] - 32;
            float slx = __fsub_rn(sumlx, __fmul_rn(__fmul_rn(w, x[i]), (float)old_l));
            float sl2 = __fsub_rn(suml2, __fmul_rn(__fmul_rn(w, (float)old_l), (float)old_l));
            if (slx > 0.0f) {
                int new_l = max(-32, min(31, kquant_nearest_int(__fdiv_rn(__fmul_rn(x[i], sl2), slx))));
                if (new_l != old_l) {
                    slx = __fadd_rn(slx, __fmul_rn(__fmul_rn(w, x[i]), (float)new_l));
                    sl2 = __fadd_rn(sl2, __fmul_rn(__fmul_rn(w, (float)new_l), (float)new_l));
                    if (sl2 > 0.0f &&
                        __fmul_rn(__fmul_rn(slx, slx), suml2) >
                            __fmul_rn(__fmul_rn(sumlx, sumlx), sl2)) {
                        levels[i] = (int8_t)(new_l + 32);
                        sumlx = slx;
                        suml2 = sl2;
                        scale = __fdiv_rn(sumlx, suml2);
                        best = __fmul_rn(scale, sumlx);
                        changed_count++;
                    }
                }
            }
        }
        if (changed_count == 0) break;
    }
    return scale;
}

extern "C" __global__ void quantize_q6_k(
    const float* __restrict__ xx, void* __restrict__ vy, int n, int k) {
    const int i_block = blockIdx.x;
    const int i_row = blockIdx.y;
    if (threadIdx.x != 0 || i_row >= n || i_block >= k / QK_K) return;
    const float* x = xx + i_row * k + i_block * QK_K;
    block_q6_K* y = ((block_q6_K*)vy) + i_row * (k / QK_K) + i_block;

    int8_t levels[QK_K];
    float sub_scales[QK_K / 16];
    float max_scale = 0.0f, max_abs_scale = 0.0f;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        sub_scales[ib] = make_qx_quants_q6_cuda(x + ib * 16, levels + ib * 16);
        if (fabsf(sub_scales[ib]) > max_abs_scale) {
            max_abs_scale = fabsf(sub_scales[ib]);
            max_scale = sub_scales[ib];
        }
    }
    if (max_scale == 0.0f) {
        y->d = __float2half(0.0f);
        for (int i = 0; i < QK_K / 16; ++i) y->scales[i] = 0;
    } else {
        const float iscale = __fdiv_rn(-128.0f, max_scale);
        y->d = __float2half(__fdiv_rn(1.0f, iscale));
        for (int i = 0; i < QK_K / 16; ++i) {
            y->scales[i] = (int8_t)min(127, kquant_nearest_int(iscale * sub_scales[i]));
        }
        for (int j = 0; j < QK_K / 16; ++j) {
            const float d = __half2float(y->d) * y->scales[j];
            if (d == 0.0f) continue;
            for (int i = 0; i < 16; ++i) {
                const int q = max(-32, min(31, kquant_nearest_int(x[j * 16 + i] / d)));
                levels[j * 16 + i] = (int8_t)(q + 32);
            }
        }
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int i = 0; i < 32; ++i) {
            const int q1 = levels[j + i];
            const int q2 = levels[j + i + 32];
            const int q3 = levels[j + i + 64];
            const int q4 = levels[j + i + 96];
            const int ql = (j / 2) + i;
            y->ql[ql] = (q1 & 0xF) | ((q3 & 0xF) << 4);
            y->ql[ql + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
            y->qh[(j / 4) + i] = ((q1 >> 4) & 3) | (((q2 >> 4) & 3) << 2) |
                                 (((q3 >> 4) & 3) << 4) | (((q4 >> 4) & 3) << 6);
        }
    }
}



/**
 * @brief Performs an indexed, batched matrix-vector multiplication for quantized tensors (for MoE models).
 *
 * This kernel handles a batch of `total_tasks` independent operations. Each task consists
 * of multiplying a Q8_1 quantized input vector with a Q4_K quantized weight matrix selected
 * by an index.
 *
 * Parallelization Strategy:
 * - The grid is 2D: gridDim.y corresponds to the task index, and gridDim.x corresponds to the row blocks of the output matrix.
 * - `blockIdx.y`: Identifies which task to perform from the batch (`0` to `total_tasks - 1`).
 * - `blockIdx.x`: Used internally by `mul_mat_vec_q` to parallelize the dot products across the rows of the weight matrix.
 *
 * @author
 *   Guoqing Bao
 *   Part of the project: https://github.com/guoqingbao/vllm.rs/
 * @param all_weights Pointer to the beginning of the weight tensor [num_experts, n, k].
 * @param all_inputs Pointer to the beginning of the input tensor [batch * topk, k].
 * @param indices Pointer to the expert indices for each task [batch * topk].
 * @param all_outputs Pointer to the beginning of the output tensor [batch * topk, n].
 * @param n The number of output features (rows in the weight matrix).
 * @param k The number of input features (columns in the weight matrix).
 * @param total_tasks The total number of tasks to process, typically batch_size * topk.
 * @param k_padded The value of k padded to a multiple of MATRIX_ROW_PADDING.
 * @param weight_expert_stride_bytes The stride in bytes to get from one expert matrix to the next.
 * @param input_task_stride_bytes The stride in bytes to get from one quantized input vector to the next.
 * @param output_task_stride_elems The stride in elements (f32) to get from one output vector to the next.
 */
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ void indexed_moe_forward(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {

    // `blockIdx.y` corresponds to the batch index (0 to batch_size-1)
    const int current_batch = blockIdx.y;
    // `blockIdx.z` corresponds to the topk index (0 to topk-1)
    const int current_topk = blockIdx.z;

    // `gridDim.z` is the number of blocks in the z-dim, which is `topk`.
    // This correctly flattens the (batch, topk) index into a single task ID.
    const int task_id = current_batch * gridDim.z + current_topk;
    if (task_id >= gridDim.y * gridDim.z) {
        return;
    }
    // If input_dim1 is 1, all experts in a batch use the same input vector.
    // Otherwise, each expert has a unique input vector.
    const int input_idx = (input_dim1 == 1) ? current_batch : task_id;

    // The expert to use is found in the `indices` array at the flattened `task_id`.
    const unsigned int expert_id = indices[task_id];

    // Calculate strides
    const size_t weight_block_size = sizeof(block_q_t);
    const size_t input_block_size = sizeof(block_q8_1);
    const size_t weight_expert_stride_bytes = (size_t)(n * k) / qk * weight_block_size;
    const size_t input_task_stride_bytes = (size_t)k_padded / QK8_1 * input_block_size;
    const size_t output_task_stride_elems = n;

    //data offsets of current task
    const void * current_input_ptr  = (const char *)all_inputs  + input_idx * input_task_stride_bytes;
    const void * current_weight_ptr = (const char *)all_weights + expert_id * weight_expert_stride_bytes;
    float * current_output_ptr = all_outputs + task_id * output_task_stride_elems;

    //fixed for inner compute
    constexpr int ncols_y = 1;
    constexpr int nwarps = 4;
    constexpr int rows_per_cuda_block = 1;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x; // `blockIdx.x` is the row within the task

    if (row0 >= n) {
        return;
    }

    const int blocks_per_row_x = k / qk;
    const int blocks_per_col_y = k_padded / QK8_1;
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

    float tmp = 0.0f;

    const block_q_t * w = (const block_q_t *) current_weight_ptr;
    const block_q8_1 * x = (const block_q8_1 *) current_input_ptr;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
        tmp += vec_dot_q_cuda(&w[kbx + row0 * blocks_per_row_x], &x[kby], kqs);
    }

    // --- Inter-warp reduction using shared memory ---
    __shared__ float tmp_shared[nwarps - 1][WARP_SIZE];
    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();

    if (threadIdx.y == 0) {
        for (int l = 0; l < nwarps - 1; ++l) {
            tmp += tmp_shared[l][threadIdx.x];
        }
        tmp = warp_reduce_sum(tmp);
        if (threadIdx.x == 0) {
            current_output_ptr[row0] = tmp;
        }
    }
}

extern "C" __global__ void indexed_moe_forward_q2k_q8_1(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

extern "C" __global__ void indexed_moe_forward_q3k_q8_1(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

extern "C" __global__ void indexed_moe_forward_q4k_q8_1(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

extern "C" __global__ void indexed_moe_forward_q5k_q8_1(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

extern "C" __global__ void indexed_moe_forward_q6k_q8_1(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

extern "C" __global__ void indexed_moe_forward_q8_0_q8_1(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

/**
 * @brief Performs an indexed, batched matrix-vector multiplication for quantized tensors (for MoE models).
 *
 * This kernel handles a batch of `total_tasks` independent operations. Each task consists
 * of multiplying a Q8_1 quantized input vector with a GGUF quantized weight matrix selected
 * by an index.
 *
 * Parallelization Strategy:
 * - The grid is 2D: gridDim.y corresponds to the task index, and gridDim.x corresponds to the row blocks of the output matrix.
 * - `blockIdx.y`: Identifies which task to perform from the batch (`0` to `total_tasks - 1`).
 * - `blockIdx.x`: Used internally by `mul_mat_vec_q` to parallelize the dot products across the rows of the weight matrix.
 *
 * @author
 *   Guoqing Bao
 *   Part of the project: https://github.com/guoqingbao/vllm.rs/
 * @param all_weights Pointer to the beginning of the weight tensor [num_experts, n, k].
 * @param all_inputs Pointer to the beginning of the input tensor [batch * topk, k].
 * @param indices Pointer to the expert indices for each task [batch * topk].
 * @param all_outputs Pointer to the beginning of the output tensor [batch * topk, n].
 * @param n The number of output features (rows in the weight matrix).
 * @param k The number of input features (columns in the weight matrix).
 * @param total_tasks The total number of tasks to process, typically batch_size * topk.
 * @param k_padded The value of k padded to a multiple of MATRIX_ROW_PADDING.
 * @param weight_expert_stride_bytes The stride in bytes to get from one expert matrix to the next.
 * @param input_task_stride_bytes The stride in bytes to get from one quantized input vector to the next.
 * @param output_task_stride_elems The stride in elements (f32) to get from one output vector to the next.
 */
// Each block handles: up to 8 batch items (one per warp) × 1 expert (grid.z)
// blockDim = (32, 8, 1)   // 8 warps
// grid    = (ceil_div(n, rows_per_block), ceil_div(batch, 8), topk)
template <int ncols_y, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ __forceinline__ void indexed_moe_forward_batch(
    const void * __restrict__ all_weights,    // [num_experts, n, k] in qK
    const void * __restrict__ all_inputs,     // [batch, input_dim1, k] in q8_1
    const unsigned int * __restrict__ indices,// [batch, topk]
    float * __restrict__ all_outputs,         // [batch, topk, n] in f32
    const int n,
    const int k,
    const int batch,
    const int topk, 
    const int k_padded,
    const int input_dim1
) {
    // which expert are we computing?
    const int kk = blockIdx.z;        // 0..topk-1
    if (kk >= topk) return;

    // group of up to 8 batch items
    const int start_b = blockIdx.y * ncols_y;

    // which row(s) of the output
    constexpr int rows_per_block = 1; // can change to 2 if beneficial
    const int row0 = rows_per_block * blockIdx.x;
    if (row0 >= n) return;

    // warp + lane
    const int warp_id = threadIdx.y;      // 0..ncols_y-1
    const int lane_id = threadIdx.x;      // 0..31
    const int b = start_b + warp_id;
    if (warp_id >= ncols_y || b >= batch) {
        return; // idle warp
    }

    // strides
    const size_t weight_expert_stride_bytes = (size_t)(n * k) / qk * sizeof(block_q_t);
    const size_t input_task_stride_bytes    = (size_t)k_padded / QK8_1 * sizeof(block_q8_1);
    const size_t output_task_stride_elems   = (size_t)n;

    // pick expert for (b, kk)
    const unsigned int expert_id = indices[b * topk + kk];
    const block_q_t * __restrict__ w_expert =
        (const block_q_t *)((const char *)all_weights + (size_t)expert_id * weight_expert_stride_bytes);

    // select input row:
    // input_dim1 == 1   -> inputs[b, 0, :]
    // input_dim1 == topk-> inputs[b, kk, :]
    const int input_index = (input_dim1 == 1) ? b : (b * topk + kk);
    const block_q8_1 * __restrict__ y_ptr =
        (const block_q8_1 *)((const char *)all_inputs + (size_t)input_index * input_task_stride_bytes);

    // dot-product tiling along k
    const int blocks_per_row_x = k / qk;
    const int blocks_per_iter  = vdr * WARP_SIZE / qi; // no nwarps factor: one warp per batch item

    // accumulators for rows_per_block rows (usually 1)
    float acc[rows_per_block];
    #pragma unroll
    for (int i = 0; i < rows_per_block; ++i) acc[i] = 0.0f;

    #pragma unroll
    for (int kbx = lane_id / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (lane_id % (qi / vdr));

        #pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
            const int row = row0 + i;
            if (row >= n) break;
            acc[i] += vec_dot_q_cuda(
                &w_expert[kbx + row * blocks_per_row_x],
                &y_ptr[kby],
                kqs);
        }
    }

    // intra-warp reduction and writeout
    #pragma unroll
    for (int i = 0; i < rows_per_block; ++i) {
        float v = warp_reduce_sum(acc[i]);
        if (lane_id == 0 && row0 + i < n) {
            float * __restrict__ out_ptr =
                all_outputs + ((size_t)b * topk + kk) * output_task_stride_elems;
            out_ptr[row0 + i] = v;
        }
    }
}

extern "C" __global__ void indexed_moe_forward_q2k_q8_1_b8(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward_batch<8, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

extern "C" __global__ void indexed_moe_forward_q3k_q8_1_b8(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward_batch<8, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

extern "C" __global__ void indexed_moe_forward_q4k_q8_1_b8(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward_batch<8, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
        all_weights, all_inputs, indices, all_outputs,
        n, k, batch, topk, k_padded, input_dim1);
}

extern "C" __global__ void indexed_moe_forward_q5k_q8_1_b8(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward_batch<8, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

extern "C" __global__ void indexed_moe_forward_q6k_q8_1_b8(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward_batch<8, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

extern "C" __global__ void indexed_moe_forward_q8_0_q8_1_b8(
    const void * __restrict__ all_weights,
    const void * __restrict__ all_inputs,
    const unsigned int * __restrict__ indices,
    float * __restrict__ all_outputs,
    const int n,
    const int k,
    const int batch,
    const int topk,
    const int k_padded,
    const int input_dim1) {
    indexed_moe_forward_batch<8, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>
        (all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1);     
}

// ===================== IQ Quantization Types =====================

// IQ2_XXS
#define QK_IQ2_XXS 256
#define QR_IQ2_XXS 8
#define QI_IQ2_XXS (QK_IQ2_XXS / (4 * QR_IQ2_XXS))

typedef struct {
    half d;
    uint16_t qs[QK_IQ2_XXS / 8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(ggml_fp16_t) + QK_IQ2_XXS/8*2, "wrong iq2_xxs block size");

// IQ2_XS
#define QK_IQ2_XS 256
#define QR_IQ2_XS 8
#define QI_IQ2_XS (QK_IQ2_XS / (4 * QR_IQ2_XS))

typedef struct {
    half d;
    uint16_t qs[QK_IQ2_XS / 8];
    uint8_t  scales[QK_IQ2_XS / 32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == sizeof(ggml_fp16_t) + QK_IQ2_XS/8*2 + QK_IQ2_XS/32, "wrong iq2_xs block size");

// IQ3_XXS
#define QK_IQ3_XXS 256
#define QR_IQ3_XXS 8
#define QI_IQ3_XXS (QK_IQ3_XXS / (4 * QR_IQ3_XXS))

typedef struct {
    half d;
    uint8_t qs[3 * QK_IQ3_XXS / 8];
} block_iq3_xxs;
static_assert(sizeof(block_iq3_xxs) == sizeof(ggml_fp16_t) + 3*(QK_IQ3_XXS/8), "wrong iq3_xxs block size");

// IQ4_XS
#define QK_IQ4_XS 256
#define QR_IQ4_XS 8
#define QI_IQ4_XS (QK_IQ4_XS / (4 * QR_IQ4_XS))

typedef struct {
    half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_IQ4_XS / 64];
    uint8_t  qs[QK_IQ4_XS / 2];
} block_iq4_xs;
static_assert(sizeof(block_iq4_xs) == sizeof(ggml_fp16_t) + 2 + QK_IQ4_XS/64 + QK_IQ4_XS/2, "wrong iq4_xs block size");

// IQ1_S
#define QK_IQ1_S 256
typedef struct {
    half d;
    uint8_t qs[QK_IQ1_S / 8];
    uint16_t qh[QK_IQ1_S / 32];
} block_iq1_s;
static_assert(sizeof(block_iq1_s) == 50, "wrong iq1_s block size");

// IQ4_NL
#define QK4_NL 32
typedef struct {
    half d;
    uint8_t qs[QK4_NL / 2];
} block_iq4_nl;
static_assert(sizeof(block_iq4_nl) == 18, "wrong iq4_nl block size");

// IQ3_S
#define QK_IQ3_S 256
typedef struct {
    half d;
    uint8_t qs[QK_IQ3_S / 4];
    uint8_t qh[QK_IQ3_S / 32];
    uint8_t signs[QK_IQ3_S / 8];
    uint8_t scales[QK_IQ3_S / 64];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == 110, "wrong iq3_s block size");

// IQ2_S
#define QK_IQ2_S 256
typedef struct {
    half d;
    uint8_t qs[QK_IQ2_S / 4];
    uint8_t qh[QK_IQ2_S / 32];
    uint8_t scales[QK_IQ2_S / 32];
} block_iq2_s;
static_assert(sizeof(block_iq2_s) == 82, "wrong iq2_s block size");

// IQ1_M
#define QK_IQ1_M 256
typedef struct {
    uint8_t qs[QK_IQ1_M / 8];
    uint8_t qh[QK_IQ1_M / 16];
    uint8_t scales[QK_IQ1_M / 32];
} block_iq1_m;
static_assert(sizeof(block_iq1_m) == 56, "wrong iq1_m block size");

static __device__ const uint8_t ksigns_iq2xs_cu[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static __device__ const uint8_t kmask_iq2xs_cu[8] = {1, 2, 4, 8, 16, 32, 64, 128};

static __device__ const int8_t kvalues_iq4nl_cu[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

#define QI_IQ1_S (QK_IQ1_S / 32)
#define QI_IQ4_NL (QK4_NL / 8)
#define QI_IQ3_S (QK_IQ3_S / 32)
#define QI_IQ2_S (QK_IQ2_S / 32)
#define QI_IQ1_M (QK_IQ1_M / 32)

#define VDR_IQ2_XXS_Q8_1_MMVQ 1
#define VDR_IQ2_XS_Q8_1_MMVQ 1
#define VDR_IQ3_XXS_Q8_1_MMVQ 1
#define VDR_IQ4_XS_Q8_1_MMVQ 1
#define VDR_IQ1_S_Q8_1_MMVQ 1
#define VDR_IQ4_NL_Q8_1_MMVQ 1
#define VDR_IQ3_S_Q8_1_MMVQ 1
#define VDR_IQ2_S_Q8_1_MMVQ 1
#define VDR_IQ1_M_Q8_1_MMVQ 1

static __device__ const uint64_t iq2xxs_grid_cu[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

static __device__ const uint64_t iq2xs_grid_cu[512] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x080808080819192bULL,
    0x0808080808192b19ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b1919ULL,
    0x08080808082b2b08ULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x080808081908192bULL,
    0x0808080819082b19ULL, 0x0808080819190808ULL, 0x080808081919082bULL, 0x0808080819191919ULL,
    0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL,
    0x080808082b08082bULL, 0x080808082b081919ULL, 0x080808082b082b08ULL, 0x080808082b190819ULL,
    0x080808082b191908ULL, 0x080808082b192b19ULL, 0x080808082b2b0808ULL, 0x0808081908080819ULL,
    0x0808081908081908ULL, 0x080808190808192bULL, 0x0808081908082b19ULL, 0x0808081908190808ULL,
    0x080808190819082bULL, 0x0808081908191919ULL, 0x0808081908192b08ULL, 0x0808081908192b2bULL,
    0x08080819082b0819ULL, 0x08080819082b1908ULL, 0x0808081919080808ULL, 0x080808191908082bULL,
    0x0808081919081919ULL, 0x0808081919082b08ULL, 0x0808081919190819ULL, 0x0808081919191908ULL,
    0x08080819192b0808ULL, 0x08080819192b2b08ULL, 0x080808192b080819ULL, 0x080808192b081908ULL,
    0x080808192b190808ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b08081919ULL,
    0x0808082b08082b08ULL, 0x0808082b08190819ULL, 0x0808082b08191908ULL, 0x0808082b082b0808ULL,
    0x0808082b19080819ULL, 0x0808082b19081908ULL, 0x0808082b19190808ULL, 0x0808082b19191919ULL,
    0x0808082b2b080808ULL, 0x0808082b2b082b2bULL, 0x0808190808080819ULL, 0x0808190808081908ULL,
    0x080819080808192bULL, 0x0808190808082b19ULL, 0x0808190808190808ULL, 0x080819080819082bULL,
    0x0808190808191919ULL, 0x0808190808192b08ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL,
    0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819081919ULL, 0x0808190819082b08ULL,
    0x0808190819190819ULL, 0x0808190819191908ULL, 0x080819081919192bULL, 0x08081908192b0808ULL,
    0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x0808191908080808ULL,
    0x080819190808082bULL, 0x0808191908081919ULL, 0x0808191908082b08ULL, 0x0808191908190819ULL,
    0x0808191908191908ULL, 0x08081919082b0808ULL, 0x0808191919080819ULL, 0x0808191919081908ULL,
    0x0808191919190808ULL, 0x08081919192b0819ULL, 0x080819192b080808ULL, 0x0808192b08080819ULL,
    0x0808192b08081908ULL, 0x0808192b08190808ULL, 0x0808192b082b192bULL, 0x0808192b19080808ULL,
    0x0808192b1908082bULL, 0x0808192b2b081908ULL, 0x08082b0808080808ULL, 0x08082b080808082bULL,
    0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808082b2bULL, 0x08082b0808190819ULL,
    0x08082b0808191908ULL, 0x08082b08082b0808ULL, 0x08082b08082b1919ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b0819192b08ULL, 0x08082b082b080808ULL,
    0x08082b082b2b0808ULL, 0x08082b082b2b2b2bULL, 0x08082b1908080819ULL, 0x08082b1908081908ULL,
    0x08082b1908190808ULL, 0x08082b1919080808ULL, 0x08082b192b080819ULL, 0x08082b192b082b19ULL,
    0x08082b2b08080808ULL, 0x08082b2b082b0808ULL, 0x08082b2b082b2b08ULL, 0x08082b2b2b19192bULL,
    0x08082b2b2b2b0808ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x081908080808192bULL,
    0x0819080808082b19ULL, 0x0819080808190808ULL, 0x081908080819082bULL, 0x0819080808191919ULL,
    0x0819080808192b08ULL, 0x08190808082b0819ULL, 0x08190808082b1908ULL, 0x0819080819080808ULL,
    0x081908081908082bULL, 0x0819080819081919ULL, 0x0819080819082b08ULL, 0x0819080819190819ULL,
    0x0819080819191908ULL, 0x08190808192b0808ULL, 0x08190808192b2b2bULL, 0x081908082b080819ULL,
    0x081908082b081908ULL, 0x081908082b190808ULL, 0x0819081908080808ULL, 0x081908190808082bULL,
    0x0819081908081919ULL, 0x0819081908082b08ULL, 0x0819081908190819ULL, 0x0819081908191908ULL,
    0x08190819082b0808ULL, 0x0819081919080819ULL, 0x0819081919081908ULL, 0x0819081919190808ULL,
    0x081908192b080808ULL, 0x081908192b191908ULL, 0x081908192b19192bULL, 0x0819082b08080819ULL,
    0x0819082b08081908ULL, 0x0819082b0808192bULL, 0x0819082b08190808ULL, 0x0819082b19080808ULL,
    0x0819082b192b0808ULL, 0x0819190808080808ULL, 0x081919080808082bULL, 0x0819190808081919ULL,
    0x0819190808082b08ULL, 0x0819190808190819ULL, 0x0819190808191908ULL, 0x08191908082b0808ULL,
    0x0819190819080819ULL, 0x0819190819081908ULL, 0x0819190819082b19ULL, 0x0819190819190808ULL,
    0x08191908192b1908ULL, 0x081919082b080808ULL, 0x0819191908080819ULL, 0x0819191908081908ULL,
    0x0819191908190808ULL, 0x0819191919080808ULL, 0x0819192b08080808ULL, 0x0819192b08191908ULL,
    0x0819192b19082b19ULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL,
    0x08192b080819082bULL, 0x08192b0819080808ULL, 0x08192b0819191908ULL, 0x08192b082b08192bULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b19192b192bULL, 0x08192b2b19190819ULL,
    0x08192b2b2b2b2b19ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808081919ULL,
    0x082b080808082b08ULL, 0x082b080808082b2bULL, 0x082b080808190819ULL, 0x082b080808191908ULL,
    0x082b0808082b0808ULL, 0x082b080819080819ULL, 0x082b080819081908ULL, 0x082b080819190808ULL,
    0x082b08082b080808ULL, 0x082b08082b2b0808ULL, 0x082b081908080819ULL, 0x082b081908081908ULL,
    0x082b081908190808ULL, 0x082b081919080808ULL, 0x082b081919082b08ULL, 0x082b0819192b1919ULL,
    0x082b082b08080808ULL, 0x082b082b082b082bULL, 0x082b082b2b080808ULL, 0x082b082b2b2b2b08ULL,
    0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b1908082b2b19ULL,
    0x082b190819080808ULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b19191919082bULL,
    0x082b19192b192b19ULL, 0x082b192b08080819ULL, 0x082b192b08192b2bULL, 0x082b192b2b2b192bULL,
    0x082b2b0808080808ULL, 0x082b2b0808082b08ULL, 0x082b2b0808082b2bULL, 0x082b2b08082b0808ULL,
    0x082b2b0819191919ULL, 0x082b2b082b082b08ULL, 0x082b2b082b2b082bULL, 0x082b2b19192b2b08ULL,
    0x082b2b192b190808ULL, 0x082b2b2b08082b08ULL, 0x082b2b2b082b0808ULL, 0x082b2b2b2b08082bULL,
    0x082b2b2b2b082b08ULL, 0x082b2b2b2b082b2bULL, 0x1908080808080819ULL, 0x1908080808081908ULL,
    0x190808080808192bULL, 0x1908080808082b19ULL, 0x1908080808190808ULL, 0x190808080819082bULL,
    0x1908080808191919ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL,
    0x1908080819080808ULL, 0x190808081908082bULL, 0x1908080819081919ULL, 0x1908080819082b08ULL,
    0x1908080819082b2bULL, 0x1908080819190819ULL, 0x1908080819191908ULL, 0x19080808192b0808ULL,
    0x19080808192b1919ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL,
    0x1908081908080808ULL, 0x190808190808082bULL, 0x1908081908081919ULL, 0x1908081908082b08ULL,
    0x1908081908190819ULL, 0x1908081908191908ULL, 0x19080819082b0808ULL, 0x1908081919080819ULL,
    0x1908081919081908ULL, 0x1908081919190808ULL, 0x190808192b080808ULL, 0x190808192b081919ULL,
    0x190808192b2b082bULL, 0x1908082b08080819ULL, 0x1908082b08081908ULL, 0x1908082b08190808ULL,
    0x1908082b0819082bULL, 0x1908082b082b2b19ULL, 0x1908082b19080808ULL, 0x1908190808080808ULL,
    0x190819080808082bULL, 0x1908190808081919ULL, 0x1908190808082b08ULL, 0x1908190808190819ULL,
    0x1908190808191908ULL, 0x1908190808192b19ULL, 0x19081908082b0808ULL, 0x1908190819080819ULL,
    0x1908190819081908ULL, 0x1908190819190808ULL, 0x190819082b080808ULL, 0x190819082b191908ULL,
    0x1908191908080819ULL, 0x1908191908081908ULL, 0x1908191908190808ULL, 0x19081919082b1908ULL,
    0x1908191919080808ULL, 0x190819192b192b2bULL, 0x1908192b08080808ULL, 0x1908192b08082b2bULL,
    0x1908192b19081908ULL, 0x1908192b19190808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL,
    0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b0819191908ULL,
    0x19082b08192b082bULL, 0x19082b1908080808ULL, 0x19082b1908190819ULL, 0x19082b1919081908ULL,
    0x19082b1919190808ULL, 0x19082b19192b2b19ULL, 0x19082b2b08081908ULL, 0x1919080808080808ULL,
    0x191908080808082bULL, 0x1919080808081919ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL,
    0x1919080808191908ULL, 0x19190808082b0808ULL, 0x19190808082b2b08ULL, 0x1919080819080819ULL,
    0x1919080819081908ULL, 0x1919080819190808ULL, 0x191908082b080808ULL, 0x1919081908080819ULL,
    0x1919081908081908ULL, 0x1919081908190808ULL, 0x1919081908191919ULL, 0x1919081919080808ULL,
    0x191908191908082bULL, 0x1919082b08080808ULL, 0x1919082b19081908ULL, 0x1919082b2b2b2b2bULL,
    0x1919190808080819ULL, 0x1919190808081908ULL, 0x1919190808190808ULL, 0x19191908082b0819ULL,
    0x1919190819080808ULL, 0x19191908192b0808ULL, 0x191919082b080819ULL, 0x191919082b2b0819ULL,
    0x1919191908080808ULL, 0x1919191908082b08ULL, 0x191919192b080808ULL, 0x191919192b082b08ULL,
    0x1919192b082b0819ULL, 0x1919192b192b2b08ULL, 0x1919192b2b2b0819ULL, 0x19192b0808080808ULL,
    0x19192b0808191908ULL, 0x19192b0819080819ULL, 0x19192b0819190808ULL, 0x19192b082b192b19ULL,
    0x19192b1908192b2bULL, 0x19192b1919080808ULL, 0x19192b191908082bULL, 0x19192b2b2b081919ULL,
    0x192b080808080819ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL,
    0x192b080819191908ULL, 0x192b0808192b082bULL, 0x192b08082b08192bULL, 0x192b08082b2b2b19ULL,
    0x192b081908080808ULL, 0x192b082b082b1908ULL, 0x192b082b19082b2bULL, 0x192b082b2b19082bULL,
    0x192b190808080808ULL, 0x192b19080819192bULL, 0x192b191908190808ULL, 0x192b191919080808ULL,
    0x192b191919081919ULL, 0x192b19192b2b1908ULL, 0x192b2b0808080819ULL, 0x192b2b08192b2b2bULL,
    0x192b2b19082b1919ULL, 0x192b2b2b0808192bULL, 0x192b2b2b19191908ULL, 0x192b2b2b192b082bULL,
    0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808081919ULL, 0x2b08080808082b08ULL,
    0x2b08080808190819ULL, 0x2b08080808191908ULL, 0x2b080808082b0808ULL, 0x2b080808082b2b2bULL,
    0x2b08080819080819ULL, 0x2b08080819081908ULL, 0x2b08080819190808ULL, 0x2b0808082b080808ULL,
    0x2b0808082b08082bULL, 0x2b0808082b2b2b08ULL, 0x2b0808082b2b2b2bULL, 0x2b08081908080819ULL,
    0x2b08081908081908ULL, 0x2b0808190808192bULL, 0x2b08081908190808ULL, 0x2b08081919080808ULL,
    0x2b08081919190819ULL, 0x2b08081919192b19ULL, 0x2b08082b08080808ULL, 0x2b08082b082b0808ULL,
    0x2b08082b2b080808ULL, 0x2b08082b2b08082bULL, 0x2b08082b2b2b0808ULL, 0x2b08082b2b2b2b08ULL,
    0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b0819080819082bULL,
    0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b0819082b082b19ULL,
    0x2b08191908080808ULL, 0x2b08191919081908ULL, 0x2b0819192b2b1919ULL, 0x2b08192b08192b08ULL,
    0x2b08192b192b2b2bULL, 0x2b082b0808080808ULL, 0x2b082b0808082b08ULL, 0x2b082b08082b1919ULL,
    0x2b082b0819192b2bULL, 0x2b082b082b080808ULL, 0x2b082b082b08082bULL, 0x2b082b082b2b2b08ULL,
    0x2b082b190808192bULL, 0x2b082b2b082b082bULL, 0x2b082b2b2b080808ULL, 0x2b082b2b2b082b08ULL,
    0x2b082b2b2b19192bULL, 0x2b082b2b2b2b2b08ULL, 0x2b19080808080819ULL, 0x2b19080808081908ULL,
    0x2b19080808190808ULL, 0x2b19080819080808ULL, 0x2b1908081919192bULL, 0x2b1908082b081908ULL,
    0x2b19081908080808ULL, 0x2b190819082b082bULL, 0x2b190819192b1908ULL, 0x2b19082b1919192bULL,
    0x2b19082b2b082b19ULL, 0x2b19190808080808ULL, 0x2b19190808081919ULL, 0x2b19190819081908ULL,
    0x2b19190819190808ULL, 0x2b19190819192b08ULL, 0x2b191919082b2b19ULL, 0x2b1919192b190808ULL,
    0x2b1919192b19082bULL, 0x2b19192b19080819ULL, 0x2b192b0819190819ULL, 0x2b192b082b2b192bULL,
    0x2b192b1919082b19ULL, 0x2b192b2b08191919ULL, 0x2b192b2b192b0808ULL, 0x2b2b080808080808ULL,
    0x2b2b08080808082bULL, 0x2b2b080808082b08ULL, 0x2b2b080808082b2bULL, 0x2b2b0808082b0808ULL,
    0x2b2b0808082b2b2bULL, 0x2b2b08082b2b0808ULL, 0x2b2b081919190819ULL, 0x2b2b081919192b19ULL,
    0x2b2b08192b2b192bULL, 0x2b2b082b08080808ULL, 0x2b2b082b0808082bULL, 0x2b2b082b08082b08ULL,
    0x2b2b082b082b2b2bULL, 0x2b2b082b2b080808ULL, 0x2b2b082b2b2b0808ULL, 0x2b2b190819080808ULL,
    0x2b2b19082b191919ULL, 0x2b2b192b192b1919ULL, 0x2b2b192b2b192b08ULL, 0x2b2b2b0808082b2bULL,
    0x2b2b2b08082b0808ULL, 0x2b2b2b08082b082bULL, 0x2b2b2b08082b2b08ULL, 0x2b2b2b082b2b0808ULL,
    0x2b2b2b082b2b2b08ULL, 0x2b2b2b1908081908ULL, 0x2b2b2b192b081908ULL, 0x2b2b2b192b08192bULL,
    0x2b2b2b2b082b2b08ULL, 0x2b2b2b2b082b2b2bULL, 0x2b2b2b2b2b190819ULL, 0x2b2b2b2b2b2b2b2bULL,
};

static __device__ const uint32_t iq3xxs_grid_cu[256] = {
    0x04040404U, 0x04040414U, 0x04040424U, 0x04040c0cU,
    0x04040c1cU, 0x04040c3eU, 0x04041404U, 0x04041414U,
    0x04041c0cU, 0x04042414U, 0x04043e1cU, 0x04043e2cU,
    0x040c040cU, 0x040c041cU, 0x040c0c04U, 0x040c0c14U,
    0x040c140cU, 0x040c142cU, 0x040c1c04U, 0x040c1c14U,
    0x040c240cU, 0x040c2c24U, 0x040c3e04U, 0x04140404U,
    0x04140414U, 0x04140424U, 0x04140c0cU, 0x04141404U,
    0x04141414U, 0x04141c0cU, 0x04141c1cU, 0x04141c3eU,
    0x04142c0cU, 0x04142c3eU, 0x04143e2cU, 0x041c040cU,
    0x041c043eU, 0x041c0c04U, 0x041c0c14U, 0x041c142cU,
    0x041c3e04U, 0x04240c1cU, 0x04241c3eU, 0x04242424U,
    0x04242c3eU, 0x04243e1cU, 0x04243e2cU, 0x042c040cU,
    0x042c043eU, 0x042c1c14U, 0x042c2c14U, 0x04341c2cU,
    0x04343424U, 0x043e0c04U, 0x043e0c24U, 0x043e0c34U,
    0x043e241cU, 0x043e340cU, 0x0c04040cU, 0x0c04041cU,
    0x0c040c04U, 0x0c040c14U, 0x0c04140cU, 0x0c04141cU,
    0x0c041c04U, 0x0c041c14U, 0x0c041c24U, 0x0c04243eU,
    0x0c042c04U, 0x0c0c0404U, 0x0c0c0414U, 0x0c0c0c0cU,
    0x0c0c1404U, 0x0c0c1414U, 0x0c14040cU, 0x0c14041cU,
    0x0c140c04U, 0x0c140c14U, 0x0c14140cU, 0x0c141c04U,
    0x0c143e14U, 0x0c1c0404U, 0x0c1c0414U, 0x0c1c1404U,
    0x0c1c1c0cU, 0x0c1c2434U, 0x0c1c3434U, 0x0c24040cU,
    0x0c24042cU, 0x0c242c04U, 0x0c2c1404U, 0x0c2c1424U,
    0x0c2c2434U, 0x0c2c3e0cU, 0x0c34042cU, 0x0c3e1414U,
    0x0c3e2404U, 0x14040404U, 0x14040414U, 0x14040c0cU,
    0x14040c1cU, 0x14041404U, 0x14041414U, 0x14041434U,
    0x14041c0cU, 0x14042414U, 0x140c040cU, 0x140c041cU,
    0x140c042cU, 0x140c0c04U, 0x140c0c14U, 0x140c140cU,
    0x140c1c04U, 0x140c341cU, 0x140c343eU, 0x140c3e04U,
    0x14140404U, 0x14140414U, 0x14140c0cU, 0x14140c3eU,
    0x14141404U, 0x14141414U, 0x14141c3eU, 0x14142404U,
    0x14142c2cU, 0x141c040cU, 0x141c0c04U, 0x141c0c24U,
    0x141c3e04U, 0x141c3e24U, 0x14241c2cU, 0x14242c1cU,
    0x142c041cU, 0x142c143eU, 0x142c240cU, 0x142c3e24U,
    0x143e040cU, 0x143e041cU, 0x143e0c34U, 0x143e242cU,
    0x1c04040cU, 0x1c040c04U, 0x1c040c14U, 0x1c04140cU,
    0x1c04141cU, 0x1c042c04U, 0x1c04342cU, 0x1c043e14U,
    0x1c0c0404U, 0x1c0c0414U, 0x1c0c1404U, 0x1c0c1c0cU,
    0x1c0c2424U, 0x1c0c2434U, 0x1c14040cU, 0x1c14041cU,
    0x1c140c04U, 0x1c14142cU, 0x1c142c14U, 0x1c143e14U,
    0x1c1c0c0cU, 0x1c1c1c1cU, 0x1c241c04U, 0x1c24243eU,
    0x1c243e14U, 0x1c2c0404U, 0x1c2c0434U, 0x1c2c1414U,
    0x1c2c2c2cU, 0x1c340c24U, 0x1c341c34U, 0x1c34341cU,
    0x1c3e1c1cU, 0x1c3e3404U, 0x24040424U, 0x24040c3eU,
    0x24041c2cU, 0x24041c3eU, 0x24042c1cU, 0x24042c3eU,
    0x240c3e24U, 0x24141404U, 0x24141c3eU, 0x24142404U,
    0x24143404U, 0x24143434U, 0x241c043eU, 0x241c242cU,
    0x24240424U, 0x24242c0cU, 0x24243424U, 0x242c142cU,
    0x242c241cU, 0x242c3e04U, 0x243e042cU, 0x243e0c04U,
    0x243e0c14U, 0x243e1c04U, 0x2c040c14U, 0x2c04240cU,
    0x2c043e04U, 0x2c0c0404U, 0x2c0c0434U, 0x2c0c1434U,
    0x2c0c2c2cU, 0x2c140c24U, 0x2c141c14U, 0x2c143e14U,
    0x2c1c0414U, 0x2c1c2c1cU, 0x2c240c04U, 0x2c24141cU,
    0x2c24143eU, 0x2c243e14U, 0x2c2c0414U, 0x2c2c1c0cU,
    0x2c342c04U, 0x2c3e1424U, 0x2c3e2414U, 0x34041424U,
    0x34042424U, 0x34042434U, 0x34043424U, 0x340c140cU,
    0x340c340cU, 0x34140c3eU, 0x34143424U, 0x341c1c04U,
    0x341c1c34U, 0x34242424U, 0x342c042cU, 0x342c2c14U,
    0x34341c1cU, 0x343e041cU, 0x343e140cU, 0x3e04041cU,
    0x3e04042cU, 0x3e04043eU, 0x3e040c04U, 0x3e041c14U,
    0x3e042c14U, 0x3e0c1434U, 0x3e0c2404U, 0x3e140c14U,
    0x3e14242cU, 0x3e142c14U, 0x3e1c0404U, 0x3e1c0c2cU,
    0x3e1c1c1cU, 0x3e1c3404U, 0x3e24140cU, 0x3e24240cU,
    0x3e2c0404U, 0x3e2c0414U, 0x3e2c1424U, 0x3e341c04U,
};

#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f

static __device__ const uint64_t iq2s_grid_cu[1024] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08, 0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x080808080819192b,
    0x0808080808192b19, 0x08080808082b0808, 0x08080808082b082b, 0x08080808082b1919, 0x08080808082b2b08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192b,
    0x0808080819082b19, 0x0808080819190808, 0x080808081919082b, 0x0808080819191919, 0x0808080819192b08, 0x08080808192b0819, 0x08080808192b1908, 0x08080808192b192b,
    0x08080808192b2b19, 0x080808082b080808, 0x080808082b08082b, 0x080808082b081919, 0x080808082b082b08, 0x080808082b190819, 0x080808082b191908, 0x080808082b2b0808,
    0x080808082b2b1919, 0x080808082b2b2b2b, 0x0808081908080819, 0x0808081908081908, 0x080808190808192b, 0x0808081908082b19, 0x0808081908190808, 0x080808190819082b,
    0x0808081908191919, 0x0808081908192b08, 0x08080819082b0819, 0x08080819082b1908, 0x0808081919080808, 0x080808191908082b, 0x0808081919081919, 0x0808081919082b08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192b, 0x0808081919192b19, 0x08080819192b0808, 0x08080819192b1919, 0x08080819192b2b08, 0x080808192b080819,
    0x080808192b081908, 0x080808192b190808, 0x080808192b19082b, 0x080808192b191919, 0x080808192b2b0819, 0x080808192b2b1908, 0x0808082b08080808, 0x0808082b0808082b,
    0x0808082b08081919, 0x0808082b08082b08, 0x0808082b08190819, 0x0808082b08191908, 0x0808082b082b0808, 0x0808082b082b2b2b, 0x0808082b19080819, 0x0808082b19081908,
    0x0808082b1908192b, 0x0808082b19082b19, 0x0808082b19190808, 0x0808082b19191919, 0x0808082b2b080808, 0x0808082b2b081919, 0x0808082b2b082b2b, 0x0808082b2b191908,
    0x0808082b2b2b082b, 0x0808190808080819, 0x0808190808081908, 0x080819080808192b, 0x0808190808082b19, 0x0808190808190808, 0x080819080819082b, 0x0808190808191919,
    0x0808190808192b08, 0x08081908082b0819, 0x08081908082b1908, 0x08081908082b192b, 0x08081908082b2b19, 0x0808190819080808, 0x080819081908082b, 0x0808190819081919,
    0x0808190819082b08, 0x0808190819082b2b, 0x0808190819190819, 0x0808190819191908, 0x080819081919192b, 0x0808190819192b19, 0x08081908192b0808, 0x08081908192b082b,
    0x08081908192b1919, 0x080819082b080819, 0x080819082b081908, 0x080819082b08192b, 0x080819082b082b19, 0x080819082b190808, 0x080819082b191919, 0x080819082b192b08,
    0x080819082b2b0819, 0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908081919, 0x0808191908082b08, 0x0808191908082b2b, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192b, 0x0808191908192b19, 0x08081919082b0808, 0x08081919082b1919, 0x08081919082b2b08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192b, 0x0808191919082b19, 0x0808191919190808, 0x080819191919082b, 0x0808191919191919, 0x0808191919192b08, 0x08081919192b0819, 0x08081919192b1908,
    0x080819192b080808, 0x080819192b08082b, 0x080819192b081919, 0x080819192b082b08, 0x080819192b190819, 0x080819192b191908, 0x080819192b2b0808, 0x0808192b08080819,
    0x0808192b08081908, 0x0808192b0808192b, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b08191919, 0x0808192b19080808, 0x0808192b19081919, 0x0808192b19082b08,
    0x0808192b19190819, 0x0808192b19191908, 0x0808192b192b0808, 0x0808192b2b080819, 0x0808192b2b081908, 0x0808192b2b190808, 0x08082b0808080808, 0x08082b080808082b,
    0x08082b0808081919, 0x08082b0808082b08, 0x08082b0808190819, 0x08082b0808191908, 0x08082b080819192b, 0x08082b0808192b19, 0x08082b08082b0808, 0x08082b08082b1919,
    0x08082b08082b2b2b, 0x08082b0819080819, 0x08082b0819081908, 0x08082b081908192b, 0x08082b0819082b19, 0x08082b0819190808, 0x08082b081919082b, 0x08082b0819191919,
    0x08082b0819192b08, 0x08082b08192b0819, 0x08082b08192b1908, 0x08082b082b080808, 0x08082b082b081919, 0x08082b082b191908, 0x08082b082b2b2b2b, 0x08082b1908080819,
    0x08082b1908081908, 0x08082b1908190808, 0x08082b190819082b, 0x08082b1908191919, 0x08082b1908192b08, 0x08082b19082b0819, 0x08082b1919080808, 0x08082b1919081919,
    0x08082b1919082b08, 0x08082b1919190819, 0x08082b1919191908, 0x08082b19192b0808, 0x08082b192b080819, 0x08082b192b190808, 0x08082b2b08080808, 0x08082b2b08190819,
    0x08082b2b08191908, 0x08082b2b082b082b, 0x08082b2b082b2b08, 0x08082b2b082b2b2b, 0x08082b2b19190808, 0x08082b2b2b192b19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192b, 0x0819080808082b19, 0x0819080808190808, 0x081908080819082b, 0x0819080808191919, 0x0819080808192b08, 0x08190808082b0819, 0x08190808082b1908,
    0x08190808082b192b, 0x0819080819080808, 0x081908081908082b, 0x0819080819081919, 0x0819080819082b08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192b,
    0x0819080819192b19, 0x08190808192b0808, 0x08190808192b082b, 0x08190808192b1919, 0x08190808192b2b08, 0x081908082b080819, 0x081908082b081908, 0x081908082b08192b,
    0x081908082b190808, 0x081908082b191919, 0x081908082b192b08, 0x081908082b2b0819, 0x081908082b2b1908, 0x0819081908080808, 0x081908190808082b, 0x0819081908081919,
    0x0819081908082b08, 0x0819081908082b2b, 0x0819081908190819, 0x0819081908191908, 0x081908190819192b, 0x0819081908192b19, 0x08190819082b0808, 0x08190819082b082b,
    0x08190819082b1919, 0x08190819082b2b08, 0x0819081919080819, 0x0819081919081908, 0x081908191908192b, 0x0819081919082b19, 0x0819081919190808, 0x081908191919082b,
    0x0819081919191919, 0x0819081919192b08, 0x08190819192b0819, 0x08190819192b1908, 0x081908192b080808, 0x081908192b08082b, 0x081908192b081919, 0x081908192b082b08,
    0x081908192b190819, 0x081908192b191908, 0x0819082b08080819, 0x0819082b08081908, 0x0819082b08082b19, 0x0819082b08190808, 0x0819082b08191919, 0x0819082b082b0819,
    0x0819082b082b1908, 0x0819082b19080808, 0x0819082b19081919, 0x0819082b19190819, 0x0819082b19191908, 0x0819082b2b080819, 0x0819082b2b081908, 0x0819082b2b190808,
    0x0819190808080808, 0x081919080808082b, 0x0819190808081919, 0x0819190808082b08, 0x0819190808190819, 0x0819190808191908, 0x081919080819192b, 0x0819190808192b19,
    0x08191908082b0808, 0x08191908082b1919, 0x08191908082b2b08, 0x0819190819080819, 0x0819190819081908, 0x081919081908192b, 0x0819190819082b19, 0x0819190819190808,
    0x081919081919082b, 0x0819190819191919, 0x0819190819192b08, 0x08191908192b0819, 0x08191908192b1908, 0x081919082b080808, 0x081919082b08082b, 0x081919082b081919,
    0x081919082b082b08, 0x081919082b190819, 0x081919082b191908, 0x081919082b2b0808, 0x0819191908080819, 0x0819191908081908, 0x081919190808192b, 0x0819191908082b19,
    0x0819191908190808, 0x081919190819082b, 0x0819191908191919, 0x0819191908192b08, 0x08191919082b0819, 0x08191919082b1908, 0x0819191919080808, 0x081919191908082b,
    0x0819191919081919, 0x0819191919082b08, 0x0819191919190819, 0x0819191919191908, 0x08191919192b0808, 0x081919192b080819, 0x081919192b081908, 0x081919192b190808,
    0x0819192b08080808, 0x0819192b08081919, 0x0819192b08082b08, 0x0819192b08190819, 0x0819192b08191908, 0x0819192b082b0808, 0x0819192b19080819, 0x0819192b19081908,
    0x0819192b19190808, 0x0819192b2b080808, 0x0819192b2b2b2b2b, 0x08192b0808080819, 0x08192b0808081908, 0x08192b080808192b, 0x08192b0808082b19, 0x08192b0808190808,
    0x08192b0808191919, 0x08192b0808192b08, 0x08192b08082b0819, 0x08192b0819080808, 0x08192b081908082b, 0x08192b0819081919, 0x08192b0819082b08, 0x08192b0819190819,
    0x08192b0819191908, 0x08192b08192b0808, 0x08192b082b080819, 0x08192b082b081908, 0x08192b1908080808, 0x08192b190808082b, 0x08192b1908081919, 0x08192b1908082b08,
    0x08192b1908190819, 0x08192b1908191908, 0x08192b19082b0808, 0x08192b1919080819, 0x08192b1919081908, 0x08192b1919190808, 0x08192b19192b2b19, 0x08192b192b2b082b,
    0x08192b2b08081908, 0x08192b2b08190808, 0x08192b2b19080808, 0x08192b2b1919192b, 0x082b080808080808, 0x082b08080808082b, 0x082b080808081919, 0x082b080808082b08,
    0x082b080808190819, 0x082b080808191908, 0x082b08080819192b, 0x082b080808192b19, 0x082b0808082b0808, 0x082b0808082b1919, 0x082b0808082b2b2b, 0x082b080819080819,
    0x082b080819081908, 0x082b080819190808, 0x082b08081919082b, 0x082b080819191919, 0x082b0808192b1908, 0x082b08082b080808, 0x082b08082b082b2b, 0x082b08082b191908,
    0x082b08082b2b2b2b, 0x082b081908080819, 0x082b081908081908, 0x082b081908190808, 0x082b08190819082b, 0x082b081908191919, 0x082b0819082b0819, 0x082b081919080808,
    0x082b08191908082b, 0x082b081919081919, 0x082b081919190819, 0x082b081919191908, 0x082b0819192b0808, 0x082b08192b080819, 0x082b08192b081908, 0x082b08192b190808,
    0x082b082b08080808, 0x082b082b08082b2b, 0x082b082b082b082b, 0x082b082b082b2b08, 0x082b082b082b2b2b, 0x082b082b19081908, 0x082b082b19190808, 0x082b082b2b082b08,
    0x082b082b2b082b2b, 0x082b082b2b2b2b08, 0x082b190808080819, 0x082b190808081908, 0x082b19080808192b, 0x082b190808082b19, 0x082b190808190808, 0x082b190808191919,
    0x082b190808192b08, 0x082b1908082b0819, 0x082b1908082b1908, 0x082b190819080808, 0x082b19081908082b, 0x082b190819081919, 0x082b190819082b08, 0x082b190819190819,
    0x082b190819191908, 0x082b1908192b0808, 0x082b19082b080819, 0x082b19082b081908, 0x082b19082b190808, 0x082b191908080808, 0x082b191908081919, 0x082b191908082b08,
    0x082b191908190819, 0x082b191908191908, 0x082b1919082b0808, 0x082b191919080819, 0x082b191919081908, 0x082b191919190808, 0x082b1919192b192b, 0x082b19192b080808,
    0x082b192b08080819, 0x082b192b08081908, 0x082b192b08190808, 0x082b192b19080808, 0x082b192b19192b19, 0x082b2b0808080808, 0x082b2b0808081919, 0x082b2b0808190819,
    0x082b2b0808191908, 0x082b2b0819080819, 0x082b2b0819081908, 0x082b2b0819190808, 0x082b2b082b082b2b, 0x082b2b082b2b2b2b, 0x082b2b1908080819, 0x082b2b1908081908,
    0x082b2b1908190808, 0x082b2b192b191919, 0x082b2b2b08082b2b, 0x082b2b2b082b082b, 0x082b2b2b192b1908, 0x082b2b2b2b082b08, 0x082b2b2b2b082b2b, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192b, 0x1908080808082b19, 0x1908080808190808, 0x190808080819082b, 0x1908080808191919, 0x1908080808192b08, 0x1908080808192b2b,
    0x19080808082b0819, 0x19080808082b1908, 0x19080808082b192b, 0x1908080819080808, 0x190808081908082b, 0x1908080819081919, 0x1908080819082b08, 0x1908080819082b2b,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192b, 0x1908080819192b19, 0x19080808192b0808, 0x19080808192b082b, 0x19080808192b1919, 0x190808082b080819,
    0x190808082b081908, 0x190808082b190808, 0x190808082b191919, 0x190808082b192b08, 0x190808082b2b0819, 0x190808082b2b1908, 0x1908081908080808, 0x190808190808082b,
    0x1908081908081919, 0x1908081908082b08, 0x1908081908190819, 0x1908081908191908, 0x190808190819192b, 0x1908081908192b19, 0x19080819082b0808, 0x19080819082b082b,
    0x19080819082b1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192b, 0x1908081919082b19, 0x1908081919190808, 0x190808191919082b, 0x1908081919191919,
    0x1908081919192b08, 0x19080819192b0819, 0x19080819192b1908, 0x190808192b080808, 0x190808192b08082b, 0x190808192b081919, 0x190808192b082b08, 0x190808192b190819,
    0x190808192b191908, 0x190808192b2b0808, 0x1908082b08080819, 0x1908082b08081908, 0x1908082b08190808, 0x1908082b0819082b, 0x1908082b08191919, 0x1908082b08192b08,
    0x1908082b082b1908, 0x1908082b19080808, 0x1908082b19081919, 0x1908082b19082b08, 0x1908082b19190819, 0x1908082b19191908, 0x1908082b192b0808, 0x1908082b2b080819,
    0x1908082b2b081908, 0x1908190808080808, 0x190819080808082b, 0x1908190808081919, 0x1908190808082b08, 0x1908190808082b2b, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192b, 0x1908190808192b19, 0x19081908082b0808, 0x19081908082b082b, 0x19081908082b1919, 0x19081908082b2b08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192b, 0x1908190819082b19, 0x1908190819190808, 0x190819081919082b, 0x1908190819191919, 0x1908190819192b08, 0x19081908192b0819, 0x19081908192b1908,
    0x190819082b080808, 0x190819082b08082b, 0x190819082b081919, 0x190819082b082b08, 0x190819082b190819, 0x190819082b191908, 0x190819082b2b0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192b, 0x1908191908082b19, 0x1908191908190808, 0x190819190819082b, 0x1908191908191919, 0x1908191908192b08, 0x19081919082b0819,
    0x19081919082b1908, 0x1908191919080808, 0x190819191908082b, 0x1908191919081919, 0x1908191919082b08, 0x1908191919190819, 0x1908191919191908, 0x19081919192b0808,
    0x19081919192b2b2b, 0x190819192b080819, 0x190819192b081908, 0x190819192b190808, 0x1908192b08080808, 0x1908192b0808082b, 0x1908192b08081919, 0x1908192b08082b08,
    0x1908192b08190819, 0x1908192b08191908, 0x1908192b082b0808, 0x1908192b19080819, 0x1908192b19081908, 0x1908192b19190808, 0x1908192b2b080808, 0x1908192b2b2b1919,
    0x19082b0808080819, 0x19082b0808081908, 0x19082b0808082b19, 0x19082b0808190808, 0x19082b080819082b, 0x19082b0808191919, 0x19082b0808192b08, 0x19082b08082b0819,
    0x19082b08082b1908, 0x19082b0819080808, 0x19082b081908082b, 0x19082b0819081919, 0x19082b0819082b08, 0x19082b0819190819, 0x19082b0819191908, 0x19082b08192b0808,
    0x19082b082b081908, 0x19082b082b190808, 0x19082b1908080808, 0x19082b190808082b, 0x19082b1908081919, 0x19082b1908082b08, 0x19082b1908190819, 0x19082b1908191908,
    0x19082b19082b0808, 0x19082b1919080819, 0x19082b1919081908, 0x19082b1919190808, 0x19082b192b080808, 0x19082b192b19192b, 0x19082b2b08080819, 0x19082b2b08081908,
    0x19082b2b08190808, 0x19082b2b19080808, 0x1919080808080808, 0x191908080808082b, 0x1919080808081919, 0x1919080808082b08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192b, 0x1919080808192b19, 0x19190808082b0808, 0x19190808082b082b, 0x19190808082b1919, 0x19190808082b2b08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192b, 0x1919080819082b19, 0x1919080819190808, 0x191908081919082b, 0x1919080819191919, 0x1919080819192b08, 0x19190808192b0819, 0x19190808192b1908,
    0x191908082b080808, 0x191908082b08082b, 0x191908082b081919, 0x191908082b082b08, 0x191908082b190819, 0x191908082b191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192b, 0x1919081908082b19, 0x1919081908190808, 0x191908190819082b, 0x1919081908191919, 0x1919081908192b08, 0x19190819082b0819, 0x19190819082b1908,
    0x1919081919080808, 0x191908191908082b, 0x1919081919081919, 0x1919081919082b08, 0x1919081919190819, 0x1919081919191908, 0x19190819192b0808, 0x191908192b080819,
    0x191908192b081908, 0x191908192b190808, 0x1919082b08080808, 0x1919082b08081919, 0x1919082b08082b08, 0x1919082b08190819, 0x1919082b08191908, 0x1919082b082b0808,
    0x1919082b19080819, 0x1919082b19081908, 0x1919082b19190808, 0x1919082b192b2b19, 0x1919082b2b080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192b,
    0x1919190808082b19, 0x1919190808190808, 0x191919080819082b, 0x1919190808191919, 0x1919190808192b08, 0x19191908082b0819, 0x19191908082b1908, 0x1919190819080808,
    0x191919081908082b, 0x1919190819081919, 0x1919190819082b08, 0x1919190819190819, 0x1919190819191908, 0x19191908192b0808, 0x191919082b080819, 0x191919082b081908,
    0x191919082b190808, 0x1919191908080808, 0x191919190808082b, 0x1919191908081919, 0x1919191908082b08, 0x1919191908190819, 0x1919191908191908, 0x19191919082b0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192b080808, 0x1919192b08080819, 0x1919192b08081908, 0x1919192b08190808, 0x1919192b082b192b,
    0x1919192b19080808, 0x19192b0808080808, 0x19192b080808082b, 0x19192b0808081919, 0x19192b0808082b08, 0x19192b0808190819, 0x19192b0808191908, 0x19192b08082b0808,
    0x19192b0819080819, 0x19192b0819081908, 0x19192b0819190808, 0x19192b0819192b2b, 0x19192b082b080808, 0x19192b1908080819, 0x19192b1908081908, 0x19192b1908190808,
    0x19192b1919080808, 0x19192b2b08080808, 0x19192b2b08192b19, 0x19192b2b2b081919, 0x19192b2b2b2b2b08, 0x192b080808080819, 0x192b080808081908, 0x192b08080808192b,
    0x192b080808190808, 0x192b08080819082b, 0x192b080808191919, 0x192b080808192b08, 0x192b0808082b0819, 0x192b0808082b1908, 0x192b080819080808, 0x192b080819081919,
    0x192b080819082b08, 0x192b080819190819, 0x192b080819191908, 0x192b0808192b0808, 0x192b08082b081908, 0x192b08082b190808, 0x192b081908080808, 0x192b08190808082b,
    0x192b081908081919, 0x192b081908082b08, 0x192b081908190819, 0x192b081908191908, 0x192b0819082b0808, 0x192b081919080819, 0x192b081919081908, 0x192b081919190808,
    0x192b08192b080808, 0x192b08192b192b19, 0x192b082b08081908, 0x192b082b08190808, 0x192b082b19080808, 0x192b082b1919192b, 0x192b082b2b2b0819, 0x192b190808080808,
    0x192b190808081919, 0x192b190808082b08, 0x192b190808190819, 0x192b190808191908, 0x192b1908082b0808, 0x192b190819080819, 0x192b190819081908, 0x192b190819190808,
    0x192b19082b080808, 0x192b191908080819, 0x192b191908081908, 0x192b191908190808, 0x192b191919080808, 0x192b191919082b2b, 0x192b1919192b2b08, 0x192b19192b19082b,
    0x192b192b08080808, 0x192b192b2b191908, 0x192b2b0808080819, 0x192b2b0808081908, 0x192b2b0808190808, 0x192b2b08192b1919, 0x192b2b082b192b08, 0x192b2b1908080808,
    0x192b2b19082b2b2b, 0x192b2b2b1908082b, 0x192b2b2b2b2b0819, 0x2b08080808080808, 0x2b0808080808082b, 0x2b08080808081919, 0x2b08080808082b08, 0x2b08080808190819,
    0x2b08080808191908, 0x2b08080808192b19, 0x2b080808082b0808, 0x2b080808082b1919, 0x2b08080819080819, 0x2b08080819081908, 0x2b08080819190808, 0x2b0808081919082b,
    0x2b08080819191919, 0x2b08080819192b08, 0x2b080808192b0819, 0x2b0808082b080808, 0x2b0808082b081919, 0x2b0808082b190819, 0x2b0808082b191908, 0x2b08081908080819,
    0x2b08081908081908, 0x2b08081908082b19, 0x2b08081908190808, 0x2b0808190819082b, 0x2b08081908191919, 0x2b08081908192b08, 0x2b080819082b0819, 0x2b080819082b1908,
    0x2b08081919080808, 0x2b0808191908082b, 0x2b08081919081919, 0x2b08081919082b08, 0x2b08081919190819, 0x2b08081919191908, 0x2b0808192b080819, 0x2b0808192b081908,
    0x2b0808192b190808, 0x2b0808192b2b2b19, 0x2b08082b08080808, 0x2b08082b08081919, 0x2b08082b08082b2b, 0x2b08082b08190819, 0x2b08082b08191908, 0x2b08082b19080819,
    0x2b08082b19081908, 0x2b08082b19190808, 0x2b08190808080819, 0x2b08190808081908, 0x2b0819080808192b, 0x2b08190808082b19, 0x2b08190808190808, 0x2b0819080819082b,
    0x2b08190808191919, 0x2b08190808192b08, 0x2b081908082b0819, 0x2b08190819080808, 0x2b0819081908082b, 0x2b08190819081919, 0x2b08190819082b08, 0x2b08190819190819,
    0x2b08190819191908, 0x2b081908192b0808, 0x2b0819082b080819, 0x2b0819082b081908, 0x2b0819082b190808, 0x2b08191908080808, 0x2b0819190808082b, 0x2b08191908081919,
    0x2b08191908082b08, 0x2b08191908190819, 0x2b08191908191908, 0x2b081919082b0808, 0x2b08191919080819, 0x2b08191919081908, 0x2b08191919190808, 0x2b0819192b080808,
    0x2b0819192b082b2b, 0x2b08192b08080819, 0x2b08192b08081908, 0x2b08192b08190808, 0x2b08192b082b2b19, 0x2b08192b19080808, 0x2b082b0808080808, 0x2b082b0808081919,
    0x2b082b0808190819, 0x2b082b0808191908, 0x2b082b0819080819, 0x2b082b0819081908, 0x2b082b0819190808, 0x2b082b082b2b082b, 0x2b082b1908080819, 0x2b082b1908081908,
    0x2b082b1919080808, 0x2b082b19192b1919, 0x2b082b2b082b082b, 0x2b082b2b19192b08, 0x2b082b2b19192b2b, 0x2b082b2b2b08082b, 0x2b082b2b2b2b082b, 0x2b19080808080819,
    0x2b19080808081908, 0x2b19080808082b19, 0x2b19080808190808, 0x2b1908080819082b, 0x2b19080808191919, 0x2b19080808192b08, 0x2b190808082b1908, 0x2b19080819080808,
    0x2b1908081908082b, 0x2b19080819081919, 0x2b19080819082b08, 0x2b19080819190819, 0x2b19080819191908, 0x2b190808192b0808, 0x2b1908082b080819, 0x2b1908082b081908,
    0x2b1908082b190808, 0x2b19081908080808, 0x2b19081908081919, 0x2b19081908190819, 0x2b19081908191908, 0x2b19081919080819, 0x2b19081919081908, 0x2b19081919190808,
    0x2b19081919192b2b, 0x2b19082b08080819, 0x2b19082b08081908, 0x2b19082b08190808, 0x2b19082b19080808, 0x2b19082b2b2b192b, 0x2b19190808080808, 0x2b1919080808082b,
    0x2b19190808081919, 0x2b19190808082b08, 0x2b19190808190819, 0x2b19190808191908, 0x2b191908082b0808, 0x2b19190819080819, 0x2b19190819081908, 0x2b19190819190808,
    0x2b1919082b080808, 0x2b1919082b19192b, 0x2b19191908080819, 0x2b19191908081908, 0x2b19191908190808, 0x2b19191919080808, 0x2b1919192b192b08, 0x2b1919192b2b0819,
    0x2b19192b08080808, 0x2b19192b1908192b, 0x2b19192b192b1908, 0x2b192b0808080819, 0x2b192b0808081908, 0x2b192b0808190808, 0x2b192b08082b192b, 0x2b192b0819080808,
    0x2b192b082b2b2b19, 0x2b192b1908080808, 0x2b192b1919082b19, 0x2b192b191919082b, 0x2b192b2b2b190808, 0x2b2b080808080808, 0x2b2b080808081919, 0x2b2b080808082b2b,
    0x2b2b080808191908, 0x2b2b0808082b082b, 0x2b2b0808082b2b2b, 0x2b2b080819080819, 0x2b2b080819081908, 0x2b2b080819190808, 0x2b2b08082b2b082b, 0x2b2b08082b2b2b2b,
    0x2b2b081919080808, 0x2b2b0819192b1919, 0x2b2b082b0808082b, 0x2b2b082b08082b2b, 0x2b2b082b082b082b, 0x2b2b082b082b2b08, 0x2b2b082b082b2b2b, 0x2b2b082b2b08082b,
    0x2b2b082b2b082b08, 0x2b2b082b2b082b2b, 0x2b2b082b2b2b2b08, 0x2b2b190808080819, 0x2b2b190808081908, 0x2b2b190808190808, 0x2b2b190819080808, 0x2b2b19082b082b19,
    0x2b2b19082b2b1908, 0x2b2b191908080808, 0x2b2b191908192b19, 0x2b2b192b19190819, 0x2b2b2b0808082b2b, 0x2b2b2b08082b2b08, 0x2b2b2b082b2b082b, 0x2b2b2b1919191908,
    0x2b2b2b192b08192b, 0x2b2b2b2b08082b08, 0x2b2b2b2b08082b2b, 0x2b2b2b2b082b0808, 0x2b2b2b2b082b082b, 0x2b2b2b2b082b2b08, 0x2b2b2b2b2b082b08, 0x2b2b2b2b2b2b2b2b,
};

static __device__ const uint32_t iq3xs_grid_cu[512] = {
    0x04040404, 0x0404040c, 0x04040414, 0x0404042c, 0x0404043e, 0x04040c04, 0x04040c0c, 0x04040c14,
    0x04040c24, 0x04040c34, 0x04041404, 0x0404140c, 0x0404142c, 0x04041c1c, 0x04042404, 0x04042414,
    0x0404242c, 0x0404243e, 0x04042c0c, 0x04042c1c, 0x04043404, 0x04043414, 0x04043e0c, 0x04043e24,
    0x04043e3e, 0x040c0404, 0x040c040c, 0x040c0414, 0x040c0424, 0x040c0c04, 0x040c0c0c, 0x040c0c2c,
    0x040c1404, 0x040c141c, 0x040c143e, 0x040c1c0c, 0x040c1c2c, 0x040c2424, 0x040c340c, 0x040c342c,
    0x040c3e14, 0x04140404, 0x0414040c, 0x0414042c, 0x0414043e, 0x04140c04, 0x04140c1c, 0x04140c34,
    0x0414140c, 0x0414142c, 0x04141c04, 0x04141c24, 0x04142414, 0x0414242c, 0x0414243e, 0x04142c0c,
    0x04142c1c, 0x04143e04, 0x04143e1c, 0x041c041c, 0x041c0c0c, 0x041c0c2c, 0x041c1404, 0x041c1414,
    0x041c1c0c, 0x041c1c1c, 0x041c1c34, 0x041c2424, 0x041c2c04, 0x041c2c14, 0x041c343e, 0x041c3e0c,
    0x041c3e2c, 0x04240404, 0x04240c1c, 0x04240c3e, 0x0424140c, 0x04241424, 0x04241c14, 0x04242404,
    0x0424241c, 0x04242c0c, 0x04243e04, 0x042c0414, 0x042c0424, 0x042c1404, 0x042c1414, 0x042c1434,
    0x042c1c1c, 0x042c240c, 0x042c242c, 0x042c243e, 0x042c3434, 0x042c3e1c, 0x04340434, 0x04340c0c,
    0x04340c1c, 0x04341c0c, 0x04342c14, 0x04343e0c, 0x043e0404, 0x043e0414, 0x043e0424, 0x043e1404,
    0x043e1414, 0x043e1434, 0x043e1c1c, 0x043e2c04, 0x043e2c24, 0x0c040404, 0x0c04040c, 0x0c040414,
    0x0c040424, 0x0c040c04, 0x0c040c0c, 0x0c040c1c, 0x0c040c2c, 0x0c040c3e, 0x0c041404, 0x0c041414,
    0x0c041c0c, 0x0c041c24, 0x0c041c34, 0x0c042c24, 0x0c042c34, 0x0c04340c, 0x0c043e14, 0x0c0c0404,
    0x0c0c040c, 0x0c0c041c, 0x0c0c0434, 0x0c0c0c04, 0x0c0c0c24, 0x0c0c140c, 0x0c0c1c04, 0x0c0c1c1c,
    0x0c0c240c, 0x0c0c2c04, 0x0c0c2c14, 0x0c0c3e04, 0x0c0c3e34, 0x0c140404, 0x0c140c14, 0x0c140c2c,
    0x0c140c3e, 0x0c141404, 0x0c141424, 0x0c141c14, 0x0c142404, 0x0c14241c, 0x0c142c2c, 0x0c143404,
    0x0c143e14, 0x0c1c040c, 0x0c1c0424, 0x0c1c043e, 0x0c1c0c04, 0x0c1c0c1c, 0x0c1c140c, 0x0c1c143e,
    0x0c1c1c04, 0x0c1c1c24, 0x0c1c240c, 0x0c1c3414, 0x0c1c3e04, 0x0c24041c, 0x0c24042c, 0x0c240c14,
    0x0c240c24, 0x0c241c0c, 0x0c241c1c, 0x0c242414, 0x0c242434, 0x0c242c04, 0x0c242c24, 0x0c2c040c,
    0x0c2c0c04, 0x0c2c0c1c, 0x0c2c140c, 0x0c2c1c04, 0x0c2c1c14, 0x0c2c2c0c, 0x0c341404, 0x0c341424,
    0x0c34143e, 0x0c342424, 0x0c342434, 0x0c3e040c, 0x0c3e041c, 0x0c3e0c04, 0x0c3e0c14, 0x0c3e140c,
    0x0c3e1c2c, 0x0c3e240c, 0x0c3e3414, 0x0c3e3e04, 0x14040404, 0x1404040c, 0x1404041c, 0x1404042c,
    0x1404043e, 0x14040c04, 0x14040c14, 0x14040c24, 0x14040c34, 0x1404140c, 0x1404141c, 0x1404143e,
    0x14041c04, 0x14041c14, 0x1404240c, 0x1404241c, 0x1404242c, 0x14042c04, 0x14042c14, 0x1404343e,
    0x14043e04, 0x14043e1c, 0x14043e2c, 0x140c0404, 0x140c0414, 0x140c0c04, 0x140c0c1c, 0x140c0c3e,
    0x140c1414, 0x140c142c, 0x140c1c0c, 0x140c1c24, 0x140c2414, 0x140c2c0c, 0x1414040c, 0x14140424,
    0x1414043e, 0x1414140c, 0x1414141c, 0x14141c04, 0x14141c3e, 0x1414240c, 0x14142c1c, 0x14142c3e,
    0x14143e0c, 0x14143e24, 0x141c0404, 0x141c0414, 0x141c042c, 0x141c0c0c, 0x141c1414, 0x141c1424,
    0x141c1c0c, 0x141c1c1c, 0x141c2414, 0x141c2c04, 0x141c3434, 0x1424040c, 0x1424043e, 0x14241404,
    0x1424141c, 0x14241c14, 0x14241c2c, 0x1424240c, 0x14243e14, 0x14243e2c, 0x142c0424, 0x142c0c0c,
    0x142c1414, 0x142c1c3e, 0x142c2404, 0x142c2c1c, 0x142c3e04, 0x14340404, 0x14340414, 0x1434043e,
    0x1434140c, 0x14342c2c, 0x1434340c, 0x143e042c, 0x143e0c0c, 0x143e1434, 0x143e1c04, 0x143e241c,
    0x143e2c04, 0x1c040414, 0x1c040c0c, 0x1c040c1c, 0x1c040c2c, 0x1c040c3e, 0x1c041414, 0x1c041c0c,
    0x1c041c1c, 0x1c041c2c, 0x1c042414, 0x1c042424, 0x1c04243e, 0x1c042c0c, 0x1c04341c, 0x1c043e0c,
    0x1c0c040c, 0x1c0c041c, 0x1c0c042c, 0x1c0c0c24, 0x1c0c140c, 0x1c0c141c, 0x1c0c2404, 0x1c0c3404,
    0x1c0c3e14, 0x1c0c3e34, 0x1c140404, 0x1c140c14, 0x1c141404, 0x1c141c14, 0x1c141c24, 0x1c142c04,
    0x1c1c040c, 0x1c1c0c04, 0x1c1c0c24, 0x1c1c140c, 0x1c1c141c, 0x1c1c143e, 0x1c1c1c04, 0x1c1c240c,
    0x1c1c241c, 0x1c1c243e, 0x1c1c2c2c, 0x1c1c3e1c, 0x1c24041c, 0x1c240c0c, 0x1c240c34, 0x1c241414,
    0x1c241c0c, 0x1c242c14, 0x1c243404, 0x1c243424, 0x1c2c040c, 0x1c2c0c04, 0x1c2c0c14, 0x1c2c142c,
    0x1c2c1c14, 0x1c2c2424, 0x1c2c2c34, 0x1c2c3e1c, 0x1c340c34, 0x1c34240c, 0x1c3e040c, 0x1c3e041c,
    0x1c3e1404, 0x1c3e1414, 0x1c3e1c2c, 0x24040404, 0x24040424, 0x24040c14, 0x24041404, 0x24041424,
    0x2404143e, 0x24041c14, 0x2404240c, 0x24042c04, 0x24043e04, 0x240c0414, 0x240c043e, 0x240c0c0c,
    0x240c0c1c, 0x240c1414, 0x240c1c04, 0x240c1c2c, 0x240c241c, 0x240c2c0c, 0x240c2c2c, 0x2414040c,
    0x2414041c, 0x24140c04, 0x24140c2c, 0x2414140c, 0x24141c1c, 0x24142404, 0x24142c3e, 0x24143414,
    0x24143e04, 0x241c0424, 0x241c0c0c, 0x241c0c1c, 0x241c1404, 0x241c1414, 0x241c1c0c, 0x241c1c2c,
    0x24240404, 0x24240414, 0x24241424, 0x24241c3e, 0x24242404, 0x24243e0c, 0x242c042c, 0x242c043e,
    0x242c140c, 0x242c3414, 0x24340c1c, 0x24341c24, 0x24343404, 0x243e0c04, 0x243e0c2c, 0x243e1c04,
    0x243e241c, 0x243e2c0c, 0x2c040414, 0x2c040c04, 0x2c040c24, 0x2c041414, 0x2c042404, 0x2c042424,
    0x2c04243e, 0x2c042c14, 0x2c043434, 0x2c043e24, 0x2c0c040c, 0x2c0c041c, 0x2c0c042c, 0x2c0c0c14,
    0x2c0c140c, 0x2c0c1c14, 0x2c0c3e14, 0x2c140404, 0x2c140c0c, 0x2c14141c, 0x2c141c04, 0x2c141c34,
    0x2c142c1c, 0x2c1c0414, 0x2c1c043e, 0x2c1c0c04, 0x2c1c143e, 0x2c1c2424, 0x2c1c2c0c, 0x2c1c342c,
    0x2c1c3e1c, 0x2c24040c, 0x2c240424, 0x2c241404, 0x2c241c14, 0x2c242434, 0x2c2c0c14, 0x2c2c1434,
    0x2c2c2c0c, 0x2c2c2c1c, 0x2c342414, 0x2c3e0414, 0x2c3e0424, 0x2c3e1414, 0x34040c0c, 0x34040c1c,
    0x34040c2c, 0x34041c0c, 0x34041c1c, 0x34043404, 0x340c0404, 0x340c1404, 0x340c143e, 0x340c3424,
    0x34140c14, 0x34141c24, 0x34142414, 0x34142c2c, 0x34143414, 0x34143e04, 0x341c0404, 0x341c0c24,
    0x341c140c, 0x341c2404, 0x3424142c, 0x3424241c, 0x34243414, 0x342c0404, 0x342c041c, 0x342c1c24,
    0x342c3404, 0x3434042c, 0x34342404, 0x343e0c0c, 0x343e0c1c, 0x3e040404, 0x3e040424, 0x3e04043e,
    0x3e041404, 0x3e041414, 0x3e041c34, 0x3e042404, 0x3e042c24, 0x3e043414, 0x3e0c0414, 0x3e0c0c0c,
    0x3e0c1424, 0x3e0c241c, 0x3e0c242c, 0x3e14040c, 0x3e140424, 0x3e140c04, 0x3e140c34, 0x3e14140c,
    0x3e141c04, 0x3e142c0c, 0x3e1c0414, 0x3e1c1c14, 0x3e1c1c2c, 0x3e1c2c1c, 0x3e24040c, 0x3e24042c,
    0x3e240c1c, 0x3e241404, 0x3e242c04, 0x3e2c1414, 0x3e2c2414, 0x3e340414, 0x3e341c0c, 0x3e3e0404,
};

static __device__ const uint64_t iq1s_grid_cu[2048] = {
    0x00000000, 0x00000002, 0x00000101, 0x00000200, 0x00000202, 0x00010001, 0x00010101, 0x00020000,
    0x00020002, 0x00020200, 0x00020202, 0x01000101, 0x01010001, 0x01010100, 0x01010102, 0x01020101,
    0x02000000, 0x02000002, 0x02000200, 0x02000202, 0x02010101, 0x02020000, 0x02020002, 0x02020200,
    0x02020202, 0x00000110, 0x00000111, 0x00010011, 0x00010110, 0x00010112, 0x00010211, 0x00010212,
    0x00020111, 0x01000011, 0x01000112, 0x01000211, 0x01010012, 0x01010111, 0x01010212, 0x01020011,
    0x01020110, 0x01020112, 0x01020210, 0x02000111, 0x02010011, 0x02010110, 0x02010112, 0x02020111,
    0x00000020, 0x00000022, 0x00000220, 0x00000222, 0x00010121, 0x00020020, 0x00020022, 0x00020220,
    0x00020222, 0x01000121, 0x01010021, 0x01010221, 0x01020120, 0x01020221, 0x02000020, 0x02000022,
    0x02000220, 0x02000222, 0x02010021, 0x02010121, 0x02010221, 0x02020020, 0x02020022, 0x02020220,
    0x02020222, 0x00011001, 0x00011100, 0x00011102, 0x00021101, 0x01001001, 0x01001201, 0x01011101,
    0x01011202, 0x01021100, 0x01021101, 0x02011001, 0x02011201, 0x02021101, 0x00001011, 0x00001110,
    0x00001111, 0x00001112, 0x00011111, 0x00011210, 0x00011212, 0x00021211, 0x01001010, 0x01001111,
    0x01001212, 0x01011010, 0x01011011, 0x01011110, 0x01011111, 0x01011112, 0x01011211, 0x01021010,
    0x01021012, 0x01021111, 0x01021210, 0x01021212, 0x02001011, 0x02011011, 0x02011111, 0x02011210,
    0x02011212, 0x02021011, 0x02021110, 0x02021111, 0x02021112, 0x02021211, 0x00011120, 0x00011221,
    0x01001021, 0x01001120, 0x01011020, 0x01011022, 0x01011121, 0x01011220, 0x01021020, 0x01021021,
    0x01021122, 0x01021221, 0x02001121, 0x02011021, 0x02011120, 0x02011221, 0x00002000, 0x00002002,
    0x00002200, 0x00002202, 0x00012101, 0x00022000, 0x00022002, 0x00022200, 0x00022202, 0x01002101,
    0x01012001, 0x01012102, 0x01022101, 0x02002000, 0x02002002, 0x02002200, 0x02002202, 0x02012101,
    0x02022000, 0x02022002, 0x02022200, 0x02022202, 0x00002111, 0x00012011, 0x00012110, 0x00012211,
    0x00022110, 0x00022111, 0x01002011, 0x01012010, 0x01012011, 0x01012111, 0x01022011, 0x01022110,
    0x01022211, 0x02012011, 0x02012110, 0x02012112, 0x02012211, 0x02022111, 0x00002020, 0x00002022,
    0x00002220, 0x00002222, 0x00012121, 0x00022020, 0x00022022, 0x00022220, 0x00022222, 0x01002121,
    0x01012021, 0x01012221, 0x01022021, 0x01022121, 0x02002020, 0x02002022, 0x02002121, 0x02002220,
    0x02002222, 0x02012121, 0x02022020, 0x02022022, 0x02022220, 0x02022222, 0x00110000, 0x00110001,
    0x00110100, 0x00110201, 0x00120100, 0x00120101, 0x01100001, 0x01100100, 0x01110000, 0x01110101,
    0x01110200, 0x01120001, 0x01120100, 0x01120101, 0x01120201, 0x02110001, 0x02110100, 0x02110102,
    0x02120001, 0x02120101, 0x00100011, 0x00100110, 0x00100112, 0x00100211, 0x00110010, 0x00110012,
    0x00110111, 0x00110210, 0x00120011, 0x00120110, 0x00120211, 0x01100111, 0x01100212, 0x01110010,
    0x01110011, 0x01110012, 0x01110110, 0x01110111, 0x01110112, 0x01110211, 0x01120010, 0x01120111,
    0x02100110, 0x02110012, 0x02110111, 0x02120011, 0x02120110, 0x00110021, 0x00110120, 0x00110122,
    0x00120121, 0x01100020, 0x01100122, 0x01100221, 0x01110022, 0x01110121, 0x01110220, 0x01110222,
    0x01120120, 0x01120122, 0x02100121, 0x02110021, 0x02110120, 0x02110122, 0x02120121, 0x00101001,
    0x00101102, 0x00101201, 0x00111100, 0x00111101, 0x00111200, 0x00111201, 0x00121001, 0x00121102,
    0x01101001, 0x01101101, 0x01101102, 0x01101200, 0x01101202, 0x01111001, 0x01111100, 0x01111101,
    0x01111102, 0x01111201, 0x01121002, 0x01121101, 0x01121200, 0x02101100, 0x02101201, 0x02111000,
    0x02111100, 0x02111101, 0x02111200, 0x02111201, 0x02111202, 0x02121001, 0x02121100, 0x02121101,
    0x02121201, 0x00101012, 0x00101111, 0x00101212, 0x00111011, 0x00111110, 0x00111111, 0x00111112,
    0x00111211, 0x00121010, 0x00121012, 0x00121111, 0x00121210, 0x00121212, 0x01101011, 0x01101110,
    0x01101111, 0x01101112, 0x01111011, 0x01111012, 0x01111110, 0x01111111, 0x01111112, 0x01111211,
    0x01111212, 0x01121011, 0x01121110, 0x01121111, 0x01121112, 0x01121211, 0x02101010, 0x02101012,
    0x02101110, 0x02101111, 0x02101210, 0x02101212, 0x02111010, 0x02111011, 0x02111110, 0x02111111,
    0x02111112, 0x02111211, 0x02111212, 0x02121010, 0x02121012, 0x02121111, 0x00101021, 0x00101120,
    0x00101121, 0x00101122, 0x00111121, 0x00111122, 0x00111220, 0x00111222, 0x00121021, 0x00121122,
    0x01101020, 0x01101022, 0x01101120, 0x01101121, 0x01101220, 0x01101222, 0x01111021, 0x01111121,
    0x01111122, 0x01111220, 0x01111221, 0x01121021, 0x01121120, 0x01121121, 0x01121220, 0x01121221,
    0x01121222, 0x02101122, 0x02101222, 0x02111022, 0x02111121, 0x02121120, 0x02121221, 0x00112001,
    0x00112102, 0x00122101, 0x01102001, 0x01102100, 0x01102102, 0x01102201, 0x01112000, 0x01112101,
    0x01112200, 0x01112202, 0x01122000, 0x01122001, 0x01122100, 0x01122102, 0x01122201, 0x02102101,
    0x02112001, 0x02112100, 0x02122101, 0x00112010, 0x00112012, 0x00112111, 0x00112212, 0x00122011,
    0x00122111, 0x01102012, 0x01102110, 0x01102111, 0x01102210, 0x01112011, 0x01112110, 0x01112111,
    0x01112112, 0x01112211, 0x01112212, 0x01122010, 0x01122111, 0x01122212, 0x02102211, 0x02112011,
    0x02112012, 0x02112111, 0x02112210, 0x02122011, 0x02122112, 0x02122211, 0x00102221, 0x00112122,
    0x00122120, 0x00122122, 0x01102120, 0x01102122, 0x01102221, 0x01112020, 0x01112022, 0x01112121,
    0x01112220, 0x01122021, 0x01122122, 0x01122221, 0x02102121, 0x02112021, 0x02112122, 0x02112222,
    0x00200000, 0x00200002, 0x00200200, 0x00200202, 0x00210101, 0x00220000, 0x00220002, 0x00220101,
    0x00220200, 0x00220202, 0x01200101, 0x01210001, 0x01210201, 0x01220001, 0x01220101, 0x02200000,
    0x02200002, 0x02200200, 0x02200202, 0x02210101, 0x02220000, 0x02220002, 0x02220101, 0x02220200,
    0x02220202, 0x00200111, 0x00210011, 0x00210110, 0x00210211, 0x00220111, 0x01200012, 0x01200110,
    0x01200211, 0x01210111, 0x01210210, 0x01210212, 0x01220011, 0x01220110, 0x01220111, 0x01220112,
    0x02200111, 0x02210010, 0x02210112, 0x02210211, 0x02220111, 0x00200021, 0x00200220, 0x00200222,
    0x00210021, 0x00210121, 0x00220020, 0x00220022, 0x00220220, 0x00220222, 0x01200121, 0x01210021,
    0x01210122, 0x01210221, 0x01220121, 0x02200021, 0x02200220, 0x02200222, 0x02210021, 0x02210121,
    0x02220020, 0x02220022, 0x02220220, 0x02220222, 0x00201101, 0x00211100, 0x00211102, 0x00211201,
    0x00221101, 0x01201100, 0x01201101, 0x01201102, 0x01201201, 0x01211002, 0x01211101, 0x01211200,
    0x01211202, 0x01221102, 0x02201101, 0x02211001, 0x02211100, 0x02211201, 0x02221001, 0x02221101,
    0x00201211, 0x00211111, 0x00221011, 0x00221211, 0x01201010, 0x01201111, 0x01201210, 0x01211011,
    0x01211110, 0x01211111, 0x01211211, 0x01221012, 0x01221111, 0x01221210, 0x02201211, 0x02211010,
    0x02211110, 0x02211111, 0x02211210, 0x02211212, 0x02221011, 0x02221110, 0x02221112, 0x02221211,
    0x00201121, 0x00211020, 0x00211022, 0x00211221, 0x00221121, 0x01201021, 0x01201221, 0x01211121,
    0x01221020, 0x01221021, 0x01221221, 0x02201120, 0x02201122, 0x02211020, 0x02211222, 0x00202000,
    0x00202002, 0x00202200, 0x00202202, 0x00212101, 0x00222000, 0x00222002, 0x00222200, 0x00222202,
    0x01202101, 0x01212001, 0x01212100, 0x01222101, 0x02202000, 0x02202002, 0x02202200, 0x02202202,
    0x02222000, 0x02222002, 0x02222200, 0x02222202, 0x00202211, 0x00212011, 0x00212110, 0x00212211,
    0x00222111, 0x01202112, 0x01202211, 0x01212012, 0x01212111, 0x01222011, 0x01222110, 0x01222112,
    0x01222211, 0x02202111, 0x02212010, 0x02212112, 0x02212211, 0x02222110, 0x02222111, 0x00202020,
    0x00202022, 0x00202220, 0x00202222, 0x00222020, 0x00222022, 0x00222220, 0x00222222, 0x01202121,
    0x01212021, 0x01212122, 0x01212221, 0x01222121, 0x02202020, 0x02202022, 0x02202220, 0x02202222,
    0x02212121, 0x02222020, 0x02222022, 0x02222220, 0x02222222, 0x10000101, 0x10010001, 0x10010102,
    0x10020101, 0x11000201, 0x11010002, 0x11010101, 0x11010200, 0x11010202, 0x11020001, 0x11020100,
    0x11020102, 0x12010100, 0x12010201, 0x12020001, 0x12020102, 0x10000010, 0x10000011, 0x10000110,
    0x10000112, 0x10000211, 0x10010012, 0x10010111, 0x10010112, 0x10010210, 0x10010212, 0x10020011,
    0x10020112, 0x10020211, 0x11000111, 0x11000210, 0x11000212, 0x11010011, 0x11010110, 0x11010111,
    0x11010112, 0x11010211, 0x11010212, 0x11020111, 0x11020210, 0x11020212, 0x12000011, 0x12000110,
    0x12000112, 0x12010010, 0x12010012, 0x12010111, 0x12020010, 0x12020011, 0x12020012, 0x10000121,
    0x10010021, 0x10010120, 0x10010122, 0x10020121, 0x11000021, 0x11010022, 0x11010121, 0x11010222,
    0x11020120, 0x11020221, 0x12000221, 0x12010120, 0x12020121, 0x10001001, 0x10011101, 0x10011201,
    0x10021201, 0x11001101, 0x11001200, 0x11001202, 0x11011001, 0x11011100, 0x11011101, 0x11011102,
    0x11021001, 0x11021002, 0x11021101, 0x11021200, 0x11021202, 0x12001001, 0x12001102, 0x12001201,
    0x12011000, 0x12011002, 0x12011101, 0x12021000, 0x12021001, 0x12021201, 0x10001011, 0x10001012,
    0x10001111, 0x10001212, 0x10011011, 0x10011110, 0x10011111, 0x10011112, 0x10011211, 0x10021010,
    0x10021111, 0x10021212, 0x11001011, 0x11001110, 0x11001111, 0x11001112, 0x11001211, 0x11011010,
    0x11011011, 0x11011110, 0x11011111, 0x11011112, 0x11011210, 0x11011211, 0x11021011, 0x11021110,
    0x11021111, 0x11021112, 0x11021211, 0x12001012, 0x12001110, 0x12001111, 0x12001210, 0x12011011,
    0x12011110, 0x12011111, 0x12011112, 0x12011211, 0x12011212, 0x12021111, 0x12021210, 0x12021212,
    0x10001021, 0x10001121, 0x10001221, 0x10011120, 0x10011121, 0x10011220, 0x10011222, 0x10021021,
    0x10021120, 0x10021221, 0x11001020, 0x11001022, 0x11001121, 0x11001220, 0x11011020, 0x11011021,
    0x11011022, 0x11011121, 0x11011122, 0x11011221, 0x11021022, 0x11021121, 0x11021220, 0x12001021,
    0x12001121, 0x12001222, 0x12011120, 0x12011121, 0x12021021, 0x12021120, 0x12021122, 0x10002101,
    0x10012001, 0x10012101, 0x10012202, 0x10022101, 0x11002002, 0x11002201, 0x11012000, 0x11012101,
    0x11012200, 0x11022001, 0x11022100, 0x11022102, 0x11022201, 0x12002101, 0x12012001, 0x12012100,
    0x12012102, 0x12012201, 0x12022101, 0x10002011, 0x10002111, 0x10002112, 0x10002212, 0x10012010,
    0x10012110, 0x10012111, 0x10012210, 0x10022011, 0x10022110, 0x10022112, 0x11002010, 0x11002111,
    0x11002212, 0x11012011, 0x11012012, 0x11012110, 0x11012111, 0x11012112, 0x11012211, 0x11022010,
    0x11022012, 0x11022111, 0x11022112, 0x11022212, 0x12002112, 0x12002211, 0x12012012, 0x12012111,
    0x12012112, 0x12012210, 0x12022011, 0x12022110, 0x12022112, 0x12022211, 0x10012122, 0x11002120,
    0x11002122, 0x11002221, 0x11012121, 0x11012220, 0x11012222, 0x11022120, 0x11022221, 0x12012120,
    0x12022121, 0x10100001, 0x10100100, 0x10100101, 0x10100102, 0x10100201, 0x10110002, 0x10110101,
    0x10110202, 0x10120001, 0x10120100, 0x10120201, 0x11100000, 0x11100101, 0x11100200, 0x11110001,
    0x11110100, 0x11110101, 0x11110102, 0x11110201, 0x11120101, 0x11120200, 0x12100102, 0x12100201,
    0x12110101, 0x12110200, 0x12120000, 0x12120001, 0x12120102, 0x12120201, 0x10100111, 0x10100210,
    0x10100211, 0x10100212, 0x10110011, 0x10110110, 0x10110111, 0x10110112, 0x10110210, 0x10110211,
    0x10120010, 0x10120111, 0x10120112, 0x10120210, 0x10120212, 0x11100011, 0x11100110, 0x11100111,
    0x11100112, 0x11100211, 0x11110010, 0x11110011, 0x11110012, 0x11110110, 0x11110111, 0x11110112,
    0x11110210, 0x11110211, 0x11110212, 0x11120011, 0x11120110, 0x11120111, 0x11120112, 0x11120211,
    0x12100012, 0x12100111, 0x12110011, 0x12110110, 0x12110111, 0x12110112, 0x12110211, 0x12120010,
    0x12120111, 0x12120212, 0x10100021, 0x10100122, 0x10110022, 0x10110121, 0x10110222, 0x10120021,
    0x10120120, 0x11100022, 0x11100121, 0x11100222, 0x11110021, 0x11110120, 0x11110121, 0x11110122,
    0x11110221, 0x11120022, 0x11120121, 0x12100121, 0x12110020, 0x12110022, 0x12110121, 0x12110221,
    0x12110222, 0x12120120, 0x10101100, 0x10101101, 0x10111001, 0x10111100, 0x10111101, 0x10111102,
    0x10111200, 0x10111201, 0x10121001, 0x10121101, 0x10121200, 0x10121202, 0x11101001, 0x11101100,
    0x11101101, 0x11101102, 0x11101201, 0x11101202, 0x11111000, 0x11111001, 0x11111100, 0x11111101,
    0x11111102, 0x11111200, 0x11111201, 0x11111202, 0x11121001, 0x11121002, 0x11121100, 0x11121101,
    0x11121102, 0x11121201, 0x12101000, 0x12101200, 0x12101202, 0x12111001, 0x12111100, 0x12111101,
    0x12111102, 0x12111201, 0x12121001, 0x12121100, 0x12121101, 0x12121202, 0x10101011, 0x10101012,
    0x10101110, 0x10101111, 0x10101112, 0x10101211, 0x10111010, 0x10111011, 0x10111012, 0x10111110,
    0x10111111, 0x10111112, 0x10111211, 0x10111212, 0x10121011, 0x10121110, 0x10121111, 0x10121112,
    0x10121211, 0x11101010, 0x11101011, 0x11101012, 0x11101110, 0x11101111, 0x11101112, 0x11101210,
    0x11101211, 0x11111010, 0x11111011, 0x11111012, 0x11111110, 0x11111111, 0x11111112, 0x11111210,
    0x11111211, 0x11111212, 0x11121010, 0x11121011, 0x11121110, 0x11121111, 0x11121112, 0x11121210,
    0x11121211, 0x11121212, 0x12101011, 0x12101110, 0x12101111, 0x12101211, 0x12101212, 0x12111010,
    0x12111011, 0x12111110, 0x12111111, 0x12111112, 0x12111210, 0x12111211, 0x12121011, 0x12121110,
    0x12121111, 0x12121112, 0x12121211, 0x10101020, 0x10101021, 0x10101022, 0x10101120, 0x10101122,
    0x10101220, 0x10101221, 0x10111021, 0x10111120, 0x10111121, 0x10111220, 0x10111221, 0x10121020,
    0x10121021, 0x10121022, 0x10121120, 0x10121121, 0x10121122, 0x10121220, 0x10121221, 0x11101021,
    0x11101121, 0x11101122, 0x11101220, 0x11101221, 0x11101222, 0x11111020, 0x11111021, 0x11111022,
    0x11111120, 0x11111121, 0x11111122, 0x11111220, 0x11111221, 0x11111222, 0x11121021, 0x11121120,
    0x11121121, 0x11121221, 0x12101022, 0x12101121, 0x12101122, 0x12101220, 0x12101221, 0x12101222,
    0x12111021, 0x12111121, 0x12111222, 0x12121022, 0x12121121, 0x12121122, 0x12121220, 0x12121221,
    0x10102100, 0x10102101, 0x10102102, 0x10102201, 0x10112000, 0x10112101, 0x10112200, 0x10122001,
    0x10122202, 0x11102101, 0x11102200, 0x11102202, 0x11112001, 0x11112100, 0x11112101, 0x11112102,
    0x11112200, 0x11112201, 0x11122000, 0x11122002, 0x11122100, 0x11122101, 0x12102002, 0x12102201,
    0x12112000, 0x12112002, 0x12112101, 0x12112200, 0x12122001, 0x12122201, 0x10102011, 0x10102012,
    0x10102111, 0x10102212, 0x10112011, 0x10112110, 0x10112111, 0x10112112, 0x10112211, 0x10122111,
    0x11102011, 0x11102110, 0x11102111, 0x11102112, 0x11102211, 0x11112010, 0x11112011, 0x11112012,
    0x11112110, 0x11112111, 0x11112112, 0x11112210, 0x11112211, 0x11112212, 0x11122011, 0x11122110,
    0x11122111, 0x11122112, 0x11122211, 0x12102011, 0x12102111, 0x12102211, 0x12112011, 0x12112110,
    0x12112111, 0x12112112, 0x12112210, 0x12112211, 0x12122111, 0x10102120, 0x10102220, 0x10112121,
    0x10112222, 0x10122020, 0x10122121, 0x10122122, 0x10122221, 0x11102121, 0x11102220, 0x11102221,
    0x11112021, 0x11112121, 0x11112122, 0x11112220, 0x11112221, 0x11122022, 0x11122121, 0x11122220,
    0x11122222, 0x12102021, 0x12102222, 0x12112022, 0x12112121, 0x12112122, 0x12112220, 0x12112222,
    0x12122021, 0x10200101, 0x10210100, 0x10210102, 0x10210201, 0x10220101, 0x11200100, 0x11210000,
    0x11210101, 0x11210102, 0x11210200, 0x11210202, 0x11220001, 0x11220100, 0x11220102, 0x11220201,
    0x12200001, 0x12210102, 0x12220101, 0x10200011, 0x10200110, 0x10200112, 0x10200211, 0x10210012,
    0x10210111, 0x10220011, 0x10220012, 0x10220112, 0x10220211, 0x11200111, 0x11200211, 0x11210011,
    0x11210111, 0x11210112, 0x11210211, 0x11220111, 0x11220112, 0x11220212, 0x12200110, 0x12200212,
    0x12210012, 0x12210111, 0x12220011, 0x12220112, 0x12220211, 0x10210021, 0x10210122, 0x10210221,
    0x11200020, 0x11200021, 0x11200122, 0x11210121, 0x11210122, 0x11210220, 0x11220020, 0x12200121,
    0x12210021, 0x12210122, 0x12220121, 0x10211001, 0x10211002, 0x10211101, 0x10211102, 0x10211202,
    0x10221001, 0x10221102, 0x10221201, 0x11201000, 0x11201002, 0x11201101, 0x11201200, 0x11201202,
    0x11211001, 0x11211100, 0x11211101, 0x11211102, 0x11211201, 0x11211202, 0x11221000, 0x11221002,
    0x11221101, 0x12201100, 0x12201101, 0x12201201, 0x12211000, 0x12211002, 0x12211100, 0x12211101,
    0x12211102, 0x12211200, 0x12211202, 0x12221001, 0x12221100, 0x12221201, 0x10201111, 0x10201210,
    0x10201212, 0x10211011, 0x10211111, 0x10211112, 0x10211211, 0x11201110, 0x11201111, 0x11201112,
    0x11201211, 0x11211010, 0x11211011, 0x11211110, 0x11211111, 0x11211112, 0x11211211, 0x11221011,
    0x11221110, 0x11221111, 0x11221112, 0x11221211, 0x12201112, 0x12201211, 0x12201212, 0x12211011,
    0x12211111, 0x12211112, 0x12211211, 0x12211212, 0x12221012, 0x12221111, 0x12221112, 0x12221210,
    0x10201022, 0x10201221, 0x10211121, 0x10221020, 0x10221122, 0x10221220, 0x10221221, 0x11201020,
    0x11201121, 0x11201220, 0x11201222, 0x11211021, 0x11211120, 0x11211121, 0x11211122, 0x11211220,
    0x11211222, 0x11221020, 0x11221121, 0x11221220, 0x12201020, 0x12201022, 0x12201121, 0x12201222,
    0x12211120, 0x12211122, 0x12211220, 0x12211221, 0x12221020, 0x12221120, 0x12221122, 0x12221222,
    0x10212102, 0x10212201, 0x10222101, 0x11202001, 0x11212002, 0x11212101, 0x11212202, 0x11222001,
    0x11222201, 0x12202101, 0x12212001, 0x12212200, 0x12222102, 0x10202011, 0x10202110, 0x10212010,
    0x10212111, 0x10222011, 0x10222110, 0x10222112, 0x10222211, 0x11202010, 0x11202011, 0x11202111,
    0x11202112, 0x11202210, 0x11212011, 0x11212110, 0x11212111, 0x11212112, 0x11212211, 0x11222010,
    0x11222111, 0x11222212, 0x12202012, 0x12202110, 0x12202212, 0x12212111, 0x12222011, 0x12222110,
    0x12222111, 0x12222211, 0x10212021, 0x10212122, 0x10212220, 0x11202021, 0x11202120, 0x11202221,
    0x11212020, 0x11212121, 0x11212220, 0x11212222, 0x11222120, 0x11222121, 0x11222221, 0x12202122,
    0x12212120, 0x12212220, 0x12212222, 0x12222122, 0x20000000, 0x20000002, 0x20000200, 0x20000202,
    0x20020000, 0x20020002, 0x20020200, 0x20020202, 0x21000101, 0x21010000, 0x21010001, 0x21010100,
    0x21010102, 0x21010201, 0x21020101, 0x22000000, 0x22000002, 0x22000200, 0x22000202, 0x22010101,
    0x22020000, 0x22020002, 0x22020200, 0x22020202, 0x20000111, 0x20010011, 0x20010110, 0x20010112,
    0x20010211, 0x20020111, 0x21000011, 0x21000110, 0x21000211, 0x21010010, 0x21010012, 0x21010111,
    0x21010112, 0x21010210, 0x21010211, 0x21020110, 0x21020112, 0x21020211, 0x22000111, 0x22000211,
    0x22010110, 0x22010112, 0x22010211, 0x22020111, 0x20000020, 0x20000022, 0x20000220, 0x20000222,
    0x20010121, 0x20020020, 0x20020022, 0x20020220, 0x20020222, 0x21010021, 0x21010120, 0x21010221,
    0x21020121, 0x22000020, 0x22000022, 0x22000220, 0x22000222, 0x22010121, 0x22020020, 0x22020022,
    0x22020220, 0x22020222, 0x20011100, 0x20011201, 0x21001001, 0x21001100, 0x21011001, 0x21011101,
    0x21011202, 0x21021001, 0x21021100, 0x21021201, 0x22011100, 0x22011201, 0x20001011, 0x20001211,
    0x20011012, 0x20011111, 0x20011212, 0x20021112, 0x20021211, 0x21001010, 0x21001011, 0x21001111,
    0x21001210, 0x21011011, 0x21011110, 0x21011111, 0x21011112, 0x21011211, 0x21011212, 0x21021111,
    0x21021112, 0x21021210, 0x21021212, 0x22001011, 0x22001110, 0x22001112, 0x22001211, 0x22011010,
    0x22011012, 0x22011111, 0x22011210, 0x22021112, 0x20011021, 0x20011122, 0x20011221, 0x20021121,
    0x21001021, 0x21001120, 0x21001221, 0x21001222, 0x21011020, 0x21011121, 0x21011221, 0x21011222,
    0x21021021, 0x21021122, 0x21021222, 0x22001121, 0x22011021, 0x22011222, 0x22021120, 0x20002000,
    0x20002002, 0x20002200, 0x20002202, 0x20012101, 0x20022000, 0x20022002, 0x20022200, 0x20022202,
    0x21002001, 0x21002101, 0x21012001, 0x21012100, 0x21012201, 0x21022101, 0x21022201, 0x22002000,
    0x22002002, 0x22002200, 0x22002202, 0x22012101, 0x22022000, 0x22022002, 0x22022200, 0x22022202,
    0x20002111, 0x20002112, 0x20012011, 0x20012110, 0x20012112, 0x20022111, 0x21002011, 0x21002110,
    0x21002112, 0x21002211, 0x21012010, 0x21012012, 0x21012111, 0x21012212, 0x21022011, 0x21022110,
    0x22002111, 0x22012112, 0x22012211, 0x22022111, 0x20002020, 0x20002022, 0x20002220, 0x20002222,
    0x20012121, 0x20022020, 0x20022022, 0x20022220, 0x20022222, 0x21002121, 0x21012021, 0x21012120,
    0x21012122, 0x22002020, 0x22002022, 0x22002220, 0x22002222, 0x22012121, 0x22022020, 0x22022022,
    0x22022220, 0x22022222, 0x20100101, 0x20110001, 0x20110102, 0x20110200, 0x20110201, 0x20120101,
    0x21100001, 0x21100102, 0x21100201, 0x21110101, 0x21110200, 0x21110202, 0x21120201, 0x21120202,
    0x22100101, 0x22110001, 0x22110100, 0x22110102, 0x22110201, 0x22120101, 0x20100011, 0x20100110,
    0x20100112, 0x20100211, 0x20110010, 0x20110111, 0x20110210, 0x20110212, 0x20120011, 0x20120110,
    0x20120112, 0x20120211, 0x21100010, 0x21100111, 0x21110010, 0x21110011, 0x21110110, 0x21110111,
    0x21110112, 0x21110211, 0x21120012, 0x21120111, 0x22100110, 0x22100112, 0x22110012, 0x22110111,
    0x22110210, 0x22120011, 0x22120110, 0x22120112, 0x22120211, 0x20100121, 0x20110021, 0x20110120,
    0x20110221, 0x20120121, 0x21100120, 0x21100122, 0x21100221, 0x21110020, 0x21110022, 0x21110121,
    0x21110220, 0x21120122, 0x21120221, 0x22100121, 0x22110120, 0x22110122, 0x22120221, 0x20101001,
    0x20101100, 0x20101102, 0x20111000, 0x20111101, 0x20111200, 0x20121102, 0x21101000, 0x21101202,
    0x21111001, 0x21111100, 0x21111101, 0x21111102, 0x21111200, 0x21111201, 0x21121000, 0x21121001,
    0x21121002, 0x21121101, 0x22101100, 0x22101102, 0x22111002, 0x22111100, 0x22111101, 0x22111200,
    0x22121001, 0x22121201, 0x20101010, 0x20101111, 0x20101210, 0x20101212, 0x20111010, 0x20111011,
    0x20111110, 0x20111111, 0x20111112, 0x20111211, 0x20121011, 0x20121111, 0x20121211, 0x20121212,
    0x21101011, 0x21101110, 0x21101111, 0x21101112, 0x21101211, 0x21111010, 0x21111011, 0x21111012,
    0x21111110, 0x21111111, 0x21111112, 0x21111210, 0x21111211, 0x21111212, 0x21121011, 0x21121110,
    0x21121111, 0x21121112, 0x21121211, 0x22101011, 0x22101111, 0x22101210, 0x22111011, 0x22111012,
    0x22111110, 0x22111111, 0x22111112, 0x22111211, 0x22111212, 0x22121010, 0x22121012, 0x22121111,
    0x22121210, 0x22121212, 0x20101021, 0x20101120, 0x20111020, 0x20111121, 0x20111221, 0x20121020,
    0x20121122, 0x20121221, 0x21101121, 0x21101220, 0x21101221, 0x21111021, 0x21111022, 0x21111121,
    0x21111122, 0x21111221, 0x21121121, 0x21121220, 0x22101022, 0x22101120, 0x22101221, 0x22101222,
    0x22111022, 0x22111120, 0x22111121, 0x22121120, 0x22121122, 0x22121221, 0x20102101, 0x20112102,
    0x20112201, 0x20122101, 0x21102001, 0x21102102, 0x21112000, 0x21112002, 0x21112101, 0x21112102,
    0x21112202, 0x21122100, 0x21122101, 0x22102101, 0x22112001, 0x22112102, 0x22112201, 0x22122101,
    0x20102110, 0x20102112, 0x20102211, 0x20112010, 0x20112012, 0x20112111, 0x20112210, 0x20112212,
    0x20122010, 0x20122011, 0x20122110, 0x20122112, 0x21102010, 0x21102012, 0x21102111, 0x21102210,
    0x21102212, 0x21112011, 0x21112110, 0x21112111, 0x21112112, 0x21112211, 0x21122012, 0x21122111,
    0x21122112, 0x21122212, 0x22102011, 0x22102110, 0x22112010, 0x22112012, 0x22112111, 0x22112212,
    0x22122011, 0x22122112, 0x20102121, 0x20112121, 0x20122121, 0x21102120, 0x21102122, 0x21102221,
    0x21112020, 0x21112121, 0x21112220, 0x21122021, 0x22102121, 0x22112021, 0x22112120, 0x22112121,
    0x22112122, 0x20200000, 0x20200002, 0x20200200, 0x20200202, 0x20210101, 0x20220000, 0x20220002,
    0x20220200, 0x20220202, 0x21200101, 0x21210001, 0x21210100, 0x21210102, 0x21210201, 0x22200000,
    0x22200002, 0x22200200, 0x22200202, 0x22210101, 0x22220000, 0x22220002, 0x22220200, 0x22220202,
    0x20200111, 0x20200211, 0x20210011, 0x20210110, 0x20210112, 0x20210211, 0x20210212, 0x21200112,
    0x21200211, 0x21210011, 0x21210111, 0x21210210, 0x21210212, 0x21220011, 0x21220110, 0x22200111,
    0x22210010, 0x22210012, 0x22210112, 0x22210211, 0x20200022, 0x20200220, 0x20200222, 0x20210020,
    0x20210221, 0x20220022, 0x20220220, 0x20220222, 0x21200121, 0x21210021, 0x21210122, 0x21210221,
    0x21220121, 0x22200020, 0x22200022, 0x22200220, 0x22200222, 0x22210121, 0x22220020, 0x22220022,
    0x22220220, 0x22220222, 0x20211201, 0x20221101, 0x21201001, 0x21201100, 0x21211000, 0x21211100,
    0x21211101, 0x21211200, 0x21211202, 0x21221001, 0x21221101, 0x21221102, 0x21221200, 0x21221201,
    0x22201101, 0x20201112, 0x20201211, 0x20211010, 0x20211012, 0x20211111, 0x20211210, 0x20221112,
    0x20221211, 0x21201012, 0x21201111, 0x21211011, 0x21211110, 0x21211111, 0x21211112, 0x21211211,
    0x21221111, 0x21221212, 0x22201011, 0x22201110, 0x22201111, 0x22201112, 0x22201211, 0x22211012,
    0x22211111, 0x22211210, 0x20201121, 0x20211021, 0x20211122, 0x20211222, 0x20221021, 0x20221121,
    0x21201120, 0x21201122, 0x21201222, 0x21211022, 0x21211121, 0x21211122, 0x21211220, 0x21221020,
    0x21221022, 0x22201122, 0x22211020, 0x22211121, 0x22211122, 0x22211221, 0x22221021, 0x22221120,
    0x22221122, 0x20202000, 0x20202002, 0x20202200, 0x20202202, 0x20222000, 0x20222002, 0x20222200,
    0x20222202, 0x21212001, 0x21212100, 0x21212102, 0x21212201, 0x22202000, 0x22202002, 0x22202200,
    0x22202202, 0x22212101, 0x22222000, 0x22222002, 0x22222200, 0x22222202, 0x20202111, 0x20212110,
    0x20212211, 0x20222011, 0x20222111, 0x21202011, 0x21212010, 0x21212111, 0x21212212, 0x21222011,
    0x21222112, 0x21222211, 0x22212010, 0x22212112, 0x20202020, 0x20202022, 0x20202220, 0x20202222,
    0x20222020, 0x20222022, 0x20222220, 0x20222222, 0x21212021, 0x21212120, 0x21212122, 0x22202020,
    0x22202022, 0x22202220, 0x22202222, 0x22212121, 0x22222020, 0x22222022, 0x22222220, 0x22222222,
};


static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq2_xxs * bq = (const block_iq2_xxs *) vbq;
    const uint16_t * q2 = bq->qs + 4 * iqs;
    const uint32_t aux32_0 = q2[0] | ((uint32_t) q2[1] << 16);
    const uint32_t aux32_1 = q2[2] | ((uint32_t) q2[3] << 16);
    const float d = __half2float(bq->d) * (0.5f + (float) (aux32_1 >> 28)) * 0.25f;
    const float d8 = __low2float(bq8_1[iqs].ds);
    const int8_t * q8 = bq8_1[iqs].qs;
    float sum = 0.0f;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *) (iq2xxs_grid_cu + ((aux32_0 >> (8 * l)) & 0xff));
        const uint8_t signs = ksigns_iq2xs_cu[(aux32_1 >> (7 * l)) & 127];
        for (int j = 0; j < 8; ++j) {
            const float value = d * (float) grid[j] *
                (signs & kmask_iq2xs_cu[j] ? -1.0f : 1.0f);
            sum += value * (float) q8[8 * l + j] * d8;
        }
    }
    return sum;
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq2_xs * bq = (const block_iq2_xs *) vbq;
    const uint16_t * q2 = bq->qs + 4 * iqs;
    const uint8_t ls1 = bq->scales[iqs] & 0xf;
    const uint8_t ls2 = bq->scales[iqs] >> 4;
    const int8_t * q8 = bq8_1[iqs].qs;
    int sum1 = 0;
    int sum2 = 0;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *) (iq2xs_grid_cu + (q2[l] & 511));
        const uint8_t signs = ksigns_iq2xs_cu[q2[l] >> 9];
        int * sum = l < 2 ? &sum1 : &sum2;
        for (int j = 0; j < 8; ++j) {
            *sum += (int) q8[8 * l + j] * (int) grid[j] *
                (signs & kmask_iq2xs_cu[j] ? -1 : 1);
        }
    }
    const float d = __half2float(bq->d) * __low2float(bq8_1[iqs].ds) * 0.25f;
    return d * ((0.5f + ls1) * sum1 + (0.5f + ls2) * sum2);
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq3_xxs * bq = (const block_iq3_xxs *) vbq;
    const uint8_t * q3 = bq->qs + 8 * iqs;
    const uint16_t * gas = (const uint16_t *) (bq->qs + QK_IQ3_XXS / 4) + 2 * iqs;
    uint32_t aux32 = gas[0] | ((uint32_t) gas[1] << 16);
    const int8_t * q8 = bq8_1[iqs].qs;
    int sum = 0;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid1 = (const uint8_t *) (iq3xxs_grid_cu + q3[2 * l]);
        const uint8_t * grid2 = (const uint8_t *) (iq3xxs_grid_cu + q3[2 * l + 1]);
        const uint8_t signs = ksigns_iq2xs_cu[aux32 & 127];
        for (int j = 0; j < 4; ++j) {
            sum += (int) q8[8 * l + j] * (int) grid1[j] *
                (signs & kmask_iq2xs_cu[j] ? -1 : 1);
            sum += (int) q8[8 * l + 4 + j] * (int) grid2[j] *
                (signs & kmask_iq2xs_cu[j + 4] ? -1 : 1);
        }
        aux32 >>= 7;
    }
    const float d = __half2float(bq->d) * (0.5f + (float) aux32) *
        __low2float(bq8_1[iqs].ds) * 0.5f;
    return d * sum;
}

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq4_xs * bq = (const block_iq4_xs *) vbq;
    const uint8_t * q4 = bq->qs + 16 * iqs;
    const int8_t * q8 = bq8_1[iqs].qs;
    const int ls = ((bq->scales_l[iqs / 2] >> (4 * (iqs % 2))) & 0xf) |
        (((bq->scales_h >> (2 * iqs)) & 3) << 4);
    const float d = __half2float(bq->d) * (float) (ls - 32) *
        __low2float(bq8_1[iqs].ds);
    int sum = 0;
    for (int j = 0; j < 16; ++j) {
        sum += (int) kvalues_iq4nl_cu[q4[j] & 0xf] * (int) q8[j];
        sum += (int) kvalues_iq4nl_cu[q4[j] >> 4] * (int) q8[j + 16];
    }
    return d * sum;
}

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq2_s * bq = (const block_iq2_s *) vbq;
    const int8_t * q8 = bq8_1[iqs].qs;
    float sum = 0.0f;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid = (const uint8_t *) (iq2s_grid_cu +
            (bq->qs[4 * iqs + l] | ((bq->qh[iqs] << (8 - 2 * l)) & 0x300)));
        const uint8_t signs = bq->qs[QK_IQ2_S / 8 + 4 * iqs + l];
        const float d = __half2float(bq->d) *
            (0.5f + (float) ((bq->scales[iqs] >> (4 * (l / 2))) & 0xf)) * 0.25f;
        for (int j = 0; j < 8; ++j) {
            sum += d * (float) grid[j] * (signs & kmask_iq2xs_cu[j] ? -1.0f : 1.0f) *
                (float) q8[8 * l + j];
        }
    }
    return sum * __low2float(bq8_1[iqs].ds);
}

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq3_s * bq = (const block_iq3_s *) vbq;
    const float d = __half2float(bq->d) *
        (0.5f + (float) ((bq->scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf)) * 0.5f;
    const uint8_t * qs = bq->qs + 8 * iqs;
    const int8_t * q8 = bq8_1[iqs].qs;
    float sum = 0.0f;
    for (int l = 0; l < 4; ++l) {
        const uint8_t * grid1 = (const uint8_t *) (iq3xs_grid_cu +
            (qs[2 * l] | ((bq->qh[iqs] << (8 - 2 * l)) & 256)));
        const uint8_t * grid2 = (const uint8_t *) (iq3xs_grid_cu +
            (qs[2 * l + 1] | ((bq->qh[iqs] << (7 - 2 * l)) & 256)));
        const uint8_t signs = bq->signs[4 * iqs + l];
        for (int j = 0; j < 4; ++j) {
            sum += d * (float) grid1[j] * (signs & kmask_iq2xs_cu[j] ? -1.0f : 1.0f) *
                (float) q8[8 * l + j];
            sum += d * (float) grid2[j] * (signs & kmask_iq2xs_cu[j + 4] ? -1.0f : 1.0f) *
                (float) q8[8 * l + 4 + j];
        }
    }
    return sum * __low2float(bq8_1[iqs].ds);
}

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq1_s * bq = (const block_iq1_s *) vbq;
    const uint16_t qh = bq->qh[iqs];
    const float d = __half2float(bq->d) * (2.0f * ((qh >> 12) & 7) + 1.0f);
    const float delta = qh & 0x8000 ? -1.125f : -0.875f;
    const int8_t * q8 = bq8_1[iqs].qs;
    float sum = 0.0f;
    for (int l = 0; l < 4; ++l) {
        const uint32_t grid = (uint32_t) iq1s_grid_cu[
            bq->qs[4 * iqs + l] | (((qh >> (3 * l)) & 7) << 8)];
        const uint8_t * g = (const uint8_t *) &grid;
        for (int j = 0; j < 4; ++j) {
            sum += ((float) (g[j] & 0xf) + delta) * (float) q8[8 * l + j];
            sum += ((float) (g[j] >> 4) + delta) * (float) q8[8 * l + 4 + j];
        }
    }
    const float2 ds = __half22float2(bq8_1[iqs].ds);
    return d * (ds.x * sum + ds.y * delta);
}

static __device__ __forceinline__ float vec_dot_iq1_m_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq1_m * bq = (const block_iq1_m *) vbq;
    const uint16_t * sc = (const uint16_t *) bq->scales;
    const uint16_t scale_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
        ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float base_d = __half2float(*((const half *) &scale_bits));
    const int8_t * q8 = bq8_1[iqs].qs;
    float sum = 0.0f;
    for (int l = 0; l < 4; ++l) {
        const int ib16 = 2 * iqs + l / 2;
        const float scale = 2.0f * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 7) + 1.0f;
        const uint8_t qh = bq->qh[2 * iqs + l / 2];
        const float delta = qh & (0x08 << (4 * (l % 2))) ? -1.125f : -0.875f;
        const uint32_t grid = (uint32_t) iq1s_grid_cu[
            bq->qs[4 * iqs + l] | (((qh >> (4 * (l % 2))) & 7) << 8)];
        const uint8_t * g = (const uint8_t *) &grid;
        for (int j = 0; j < 4; ++j) {
            sum += scale * ((float) (g[j] & 0xf) + delta) * (float) q8[8 * l + j];
            sum += scale * ((float) (g[j] >> 4) + delta) * (float) q8[8 * l + 4 + j];
        }
    }
    return sum * base_d * __low2float(bq8_1[iqs].ds);
}

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    const block_iq4_nl * bq = (const block_iq4_nl *) vbq;
    const uint8_t * q4 = bq->qs + 4 * iqs;
    const int8_t * q8 = bq8_1->qs;
    float sum = 0.0f;
    for (int j = 0; j < 4; ++j) {
        sum += (float) kvalues_iq4nl_cu[q4[j] & 0xf] * (float) q8[4 * iqs + j];
        sum += (float) kvalues_iq4nl_cu[q4[j] >> 4] * (float) q8[16 + 4 * iqs + j];
    }
    return sum * __half2float(bq->d) * __low2float(bq8_1->ds);
}

template<typename dst_t>
static __device__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + i * QK_IQ2_XXS + 32 * ib + 8 * il;
    const uint16_t * q2 = x[i].qs + 4 * ib;
    const uint8_t * aux8 = (const uint8_t *)q2;
    const uint8_t * grid = (const uint8_t *)(iq2xxs_grid_cu + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (float)(aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs_cu[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) {
        y[j] = d * grid[j] * (signs & kmask_iq2xs_cu[j] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __device__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + i * QK_IQ2_XS + 32 * ib + 8 * il;
    const uint16_t * q2 = x[i].qs + 4 * ib;
    const uint8_t * grid = (const uint8_t *)(iq2xs_grid_cu + (q2[il] & 511));
    const float d = __half2float(x[i].d) * (0.5f + (float)((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs_cu[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) {
        y[j] = d * grid[j] * (signs & kmask_iq2xs_cu[j] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __device__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + i * QK_IQ3_XXS + 32 * ib + 8 * il;
    const uint8_t * q3 = x[i].qs + 8 * ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_IQ3_XXS/4) + 2*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid_cu + q3[2*il+0]);
    const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid_cu + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = __half2float(x[i].d) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs_cu[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs_cu[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs_cu[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __device__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;
    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + i * QK_IQ2_S + 32 * ib + 8 * il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid_cu +
        (x[i].qs[4 * ib + il] | ((x[i].qh[ib] << (8 - 2 * il)) & 0x300)));
    const float d = __half2float(x[i].d) *
        (0.5f + (float)((x[i].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_IQ2_S / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) {
        y[j] = d * grid[j] * (signs & kmask_iq2xs_cu[j] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __device__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;
    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + i * QK_IQ3_S + 32 * ib + 8 * il;
    const uint8_t * qs = x[i].qs + 8 * ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3xs_grid_cu +
        (qs[2 * il + 0] | ((x[i].qh[ib] << (8 - 2 * il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3xs_grid_cu +
        (qs[2 * il + 1] | ((x[i].qh[ib] << (7 - 2 * il)) & 256)));
    const float d = __half2float(x[i].d) *
        (0.5f + (float)((x[i].scales[ib / 2] >> 4 * (ib % 2)) & 0xf)) * 0.5f;
    const uint8_t signs = x[i].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = d * grid1[j] * (signs & kmask_iq2xs_cu[j + 0] ? -1.f : 1.f);
        y[j + 4] = d * grid2[j] * (signs & kmask_iq2xs_cu[j + 4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __device__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;
    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + i * QK_IQ1_S + 32 * ib + 8 * il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1.f - IQ1S_DELTA : -1.f + IQ1S_DELTA;
    const float d = __half2float(x[i].d) * (2.f * ((x[i].qh[ib] >> 12) & 7) + 1.f);
    uint32_t grid = (uint32_t)iq1s_grid_cu[
        x[i].qs[4 * ib + il] | (((x[i].qh[ib] >> (3 * il)) & 7) << 8)];
    const uint8_t * q = (const uint8_t *)&grid;
    for (int j = 0; j < 8; ++j) {
        const int qv = (j < 4) ? ((q[j] >> 0) & 0xf) : ((q[j - 4] >> 4) & 0xf);
        y[j] = d * (qv + delta);
    }
}

template<typename dst_t>
static __device__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;
    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + i * QK_IQ1_M + 32 * ib + 8 * il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    const uint16_t scale_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
        ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = __half2float(*((const half *)&scale_bits)) *
        (2.f * ((sc[(2 * ib + il / 2) / 4] >> (3 * ((2 * ib + il / 2) % 4))) & 7) + 1.f);
    const uint8_t qh = x[i].qh[2 * ib + il / 2];
    const float delta = qh & (0x08 << (4 * (il % 2))) ? -1.f - IQ1M_DELTA : -1.f + IQ1M_DELTA;
    uint32_t grid = (uint32_t)iq1s_grid_cu[
        x[i].qs[4 * ib + il] | (((qh >> (4 * (il % 2))) & 7) << 8)];
    const uint8_t * q = (const uint8_t *)&grid;
    for (int j = 0; j < 8; ++j) {
        const int qv = (j < 4) ? ((q[j] >> 0) & 0xf) : ((q[j - 4] >> 4) & 0xf);
        y[j] = d * (qv + delta);
    }
}

template<typename dst_t>
static __device__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *)vx + i;
    const int tid = threadIdx.x;
    if (tid >= QK4_NL / 2) return;
    const uint8_t q = x->qs[tid];
    dst_t * y = yy + i * QK4_NL;
    const float d = __half2float(x->d);
    y[tid] = d * kvalues_iq4nl_cu[q & 0xf];
    y[tid + QK4_NL / 2] = d * kvalues_iq4nl_cu[q >> 4];
}

template<typename dst_t>
static __device__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int i = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *) vx;
    const int tid = threadIdx.x;
    if (tid >= 32) return;

    const int il = tid / 8;
    const int ib = tid % 8;
    dst_t * y = yy + i * QK_IQ4_XS + 32 * ib + 4 * il;
    const uint8_t * q4 = x[i].qs + 16 * ib + 4 * il;
    const float d = __half2float(x[i].d) * (float)((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl_cu[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl_cu[q4[j] >> 4];
    }
}

#define DEQUANTIZE_IQ(QNAME) \
extern "C" __global__ void dequantize_block_##QNAME##_f32(const void * __restrict__ vx, float * __restrict__ y) { \
    dequantize_block_##QNAME(vx, y); \
} \
extern "C" __global__ void dequantize_block_##QNAME##_f16(const void * __restrict__ vx, half * __restrict__ y) { \
    dequantize_block_##QNAME(vx, y); \
} \

DEQUANTIZE_IQ(iq2_xxs)
DEQUANTIZE_IQ(iq2_xs)
DEQUANTIZE_IQ(iq3_xxs)
DEQUANTIZE_IQ(iq1_s)
DEQUANTIZE_IQ(iq4_nl)
DEQUANTIZE_IQ(iq3_s)
DEQUANTIZE_IQ(iq2_s)
DEQUANTIZE_IQ(iq4_xs)
DEQUANTIZE_IQ(iq1_m)

#define DEFINE_INDEXED_IQ_KERNELS(name, qk, qi, block_q_t, vdr, vec_dot_q_cuda) \
extern "C" __global__ void indexed_moe_forward_##name( \
    const void * all_weights, const void * all_inputs, const unsigned int * indices, \
    float * all_outputs, const int n, const int k, const int batch, const int topk, \
    const int k_padded, const int input_dim1) { \
    indexed_moe_forward<qk, qi, block_q_t, vdr, vec_dot_q_cuda>( \
        all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1); \
} \
extern "C" __global__ void indexed_moe_forward_##name##_b8( \
    const void * all_weights, const void * all_inputs, const unsigned int * indices, \
    float * all_outputs, const int n, const int k, const int batch, const int topk, \
    const int k_padded, const int input_dim1) { \
    indexed_moe_forward_batch<8, qk, qi, block_q_t, vdr, vec_dot_q_cuda>( \
        all_weights, all_inputs, indices, all_outputs, n, k, batch, topk, k_padded, input_dim1); \
}

DEFINE_INDEXED_IQ_KERNELS(iq2_xxs_q8_1, QK_IQ2_XXS, QI_IQ2_XXS, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1)
DEFINE_INDEXED_IQ_KERNELS(iq2_xs_q8_1, QK_IQ2_XS, QI_IQ2_XS, block_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1)
DEFINE_INDEXED_IQ_KERNELS(iq3_xxs_q8_1, QK_IQ3_XXS, QI_IQ3_XXS, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1)
DEFINE_INDEXED_IQ_KERNELS(iq4_xs_q8_1, QK_IQ4_XS, QI_IQ4_XS, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1)
DEFINE_INDEXED_IQ_KERNELS(iq1_s_q8_1, QK_IQ1_S, QI_IQ1_S, block_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1)
DEFINE_INDEXED_IQ_KERNELS(iq4_nl_q8_1, QK4_NL, QI_IQ4_NL, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1)
DEFINE_INDEXED_IQ_KERNELS(iq3_s_q8_1, QK_IQ3_S, QI_IQ3_S, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1)
DEFINE_INDEXED_IQ_KERNELS(iq2_s_q8_1, QK_IQ2_S, QI_IQ2_S, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1)
DEFINE_INDEXED_IQ_KERNELS(iq1_m_q8_1, QK_IQ1_M, QI_IQ1_M, block_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1)

#undef DEFINE_INDEXED_IQ_KERNELS


/**
 * @brief A lookup table for converting a 4-bit MXFP4 value to a 32-bit float.
 *
 * MXFP4 (F4E2M1) has 1 sign bit, 2 exponent bits, and 1 mantissa bit.
 * The values are pre-calculated based on the user's provided Rust implementation logic,
 * which appears to be a common variant of the format. Storing this in constant
 * memory provides fast, uniform access for all threads in a warp.
 *
 * Index (4-bit value) -> float value
 */
__constant__ float d_mxfp4_to_float_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    // Note: The negative zero from the spec is treated as 0.0f here.
    // The spec's NaN/Inf values are replaced by +/-6.0f to match the apparent
    // logic from the sample code.
};

/**
 * @brief Converts an 8-bit F8E8M0 scale value to a 32-bit float.
 *
 * F8E8M0 is an 8-bit floating-point format with 8 exponent bits and 0 mantissa bits.
 * The conversion formula is 2^(value - bias), where the bias is 127.
 * The value 0xFF (255) is treated as NaN, as seen in the reference code.
 *
 * @param scale_val The 8-bit F8E8M0 scale value.
 * @return The corresponding 32-bit float value.
 */
__device__ __forceinline__ float f8e8m0_to_float(unsigned char scale_val) {
    unsigned char b = (scale_val >= 0xFF) ? 0xFD : scale_val;
    return __uint_as_float((unsigned int)b << 23);
}


//----------------------------------------------------------------------------
// MXFP4 Dequantization Kernel
//----------------------------------------------------------------------------

/**
 * @brief Dequantizes MXFP4 weights to a specified output type (fp32, fp16, or bf16).
 *
 * This kernel uses a grid-stride loop pattern where each thread is responsible
 * for dequantizing one output element. This ensures that all elements are processed
 * regardless of the grid size.
 *
 * The core logic for each thread is:
 * 1. Calculate its global output index `idx`.
 * 2. Determine the corresponding row and column in the weight matrix.
 * 3. Identify the weight group and find the associated scale index.
 * 4. Fetch the F8E8M0 scale and convert it to float.
 * 5. Fetch the packed byte containing two MXFP4 weights.
 * 6. Extract the correct 4-bit nibble for the current element.
 * 7. Convert the 4-bit weight to float using the lookup table.
 * 8. Multiply the weight by the scale.
 * 9. Cast to the target output type and store the result.
 *
 * @author
 *   Guoqing Bao
 *   Part of the project: https://github.com/guoqingbao/vllm.rs/
 * @tparam T The output data type (float, __half, or __nv_bfloat16).
 * @param out Pointer to the output tensor on the GPU.
 * @param w_q Pointer to the quantized MXFP4 input weights on the GPU.
 * @param scales Pointer to the F8E8M0 scales on the GPU.
 * @param total_elements The total number of output elements.
 * @param weights_per_row The number of dequantized weights in a single row (unpacked).
 * @param packed_row_len The length of a packed row in bytes.
 * @param groups_per_row The number of groups in a single row.
 * @param group_size The number of weights per group.
 */
template <typename T>
__device__ __forceinline__ void dequantize_mxfp4_kernel(
    T* __restrict__ out,
    const unsigned char* __restrict__ w_q,
    const unsigned char* __restrict__ scales,
    const long long total_elements,
    const int weights_per_row,
    const int packed_row_len,
    const int groups_per_row,
    const int group_size)
{
    // Using a grid-stride loop to process all elements, making the kernel
    // robust to different grid sizes.
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += gridDim.x * blockDim.x)
    {
        // 1. Calculate row and column for the current output element
        const int row = idx / weights_per_row;
        const int col = idx % weights_per_row;

        // 2. Calculate the index for the scale factor
        const int group_idx_in_row = col / group_size;
        const int scale_idx = row * groups_per_row + group_idx_in_row;

        // 3. Fetch and convert the scale
        const float scale_val = f8e8m0_to_float(scales[scale_idx]);

        // 4. Locate and fetch the packed byte containing the weight
        // Each byte contains two 4-bit weights.
        const int byte_idx_in_row = col / 2;
        const unsigned char packed_byte = w_q[row * packed_row_len + byte_idx_in_row];

        // 5. Extract the 4-bit nibble
        // If col is even, we take the lower 4 bits. If odd, the upper 4 bits.
        const unsigned char fp4_bits = (col % 2 == 0)
            ? (packed_byte & 0x0F)
            : ((packed_byte >> 4) & 0x0F);

        // 6. Convert the 4-bit weight to float using the constant memory LUT
        const float weight_val = d_mxfp4_to_float_lut[fp4_bits];

        // 7. Perform dequantization and store the result
        // The static_cast handles conversion to the target type T.
        out[idx] = static_cast<T>(weight_val * scale_val);
    }
}

extern "C" __global__ void dequantize_mxfp4_f16(
    __half* __restrict__ out,
    const unsigned char* __restrict__ w_q,
    const unsigned char* __restrict__ scales,
    const long long total_elements,
    const int weights_per_row,
    const int packed_row_len,
    const int groups_per_row,
    const int group_size) 
{
    dequantize_mxfp4_kernel<__half>(
        out, w_q, scales,
        total_elements, weights_per_row,
        packed_row_len, groups_per_row, group_size);
}

extern "C" __global__ void dequantize_mxfp4_bf16(
    __nv_bfloat16* __restrict__ out,
    const unsigned char* __restrict__ w_q,
    const unsigned char* __restrict__ scales,
    const long long total_elements,
    const int weights_per_row,
    const int packed_row_len,
    const int groups_per_row,
    const int group_size) {
    dequantize_mxfp4_kernel<__nv_bfloat16>(
        out, w_q, scales,
        total_elements, weights_per_row,
        packed_row_len, groups_per_row, group_size);
}

extern "C" __global__ void dequantize_mxfp4_f32(
        float* __restrict__ out,
    const unsigned char* __restrict__ w_q,
    const unsigned char* __restrict__ scales,
    const long long total_elements,
    const int weights_per_row,
    const int packed_row_len,
    const int groups_per_row,
    const int group_size) {
        dequantize_mxfp4_kernel<float>(
        out, w_q, scales,
        total_elements, weights_per_row,
        packed_row_len, groups_per_row, group_size);
}
