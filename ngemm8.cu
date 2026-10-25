// E17 NG8 (pjs, own code, 22 Oct 2026; scratchpad e17 DESIGN.md): the gate|up GEMM + SwiGLU of NG7 (ngemm7x.cu, ours)
// with fewer L2->SMEM bytes per FLOP. Under the always-binding 600 W cap, NG7's limiter is energy per FLOP, and the
// largest movable part of it is the operand stream L2 -> SMEM (cp.async.cg, 36-40 pJ/B, e6 ANALYSIS 3.2; about 35 %
// of the model dynamic energy of NG7's served calls). Two changes, neither of which touches an element's arithmetic:
//   1. 192 x 192 body tiles (8 warps of 48 x 96 warp tiles) instead of 128 x 256: the same 48 KB per BK 64 k-tile and
//      stage, the same 96 KB of SMEM (2 stages), the same 31.25 mB / FLOP of ldmatrix traffic, but 12.5 % more FLOPs per
//      byte streamed (96 against 85.3 FLOP pairs per element), so -11 % L2 -> SMEM bytes, cp.async / barrier / k-loop
//      instructions per FLOP in the body. A body tile holds 96 gate and the matching 96 up columns (96 outputs).
//   2. One tail band per launch, as NG7 (rows [mb, M)): 128x256 rows (B7, 1-2 rows), s96x256 (S7c) or the tall
//      t160x128 / t224x128 bands (rgh's NG9T band geometry, credit rgh), chosen per (bucket, R) by the model energy
//      (scratchpad e17 ana/geom.py); and an L2-aware CTA order: the grid walks 24 column groups of 384 outputs, each
//      group's body tiles (row tile fastest) followed by its band tiles, so a band reads its weight panel while the
//      body tiles of the same columns hold it in L2 (NG7 runs every band tile after every body tile: about 15 MB more
//      DRAM reads per band launch in the e14 trace, scratchpad e17 ana/dram_attr.py).
//
// Arithmetic (identical to NG7 / NG9B / the cuBLASLt pins of the class; nothing here changes it): every gate and up
// value is one fp32 chain of mma.sync.m16n8k16 (bf16 x bf16 -> fp32) over the 160 k16 blocks of K = 2560 in ascending
// order, started from +0, rounded once to bf16 (RN); out = bf16(float(lut[bits(g)]) * float(u)) (RN), lut = torch's
// own F.silu of every bf16 value (ng.check_ng7 / check_ng8). The primitives, the T7 template, the mainloop ml7, the
// epilogue's swiglu_pair and stmatrix staging are NG7's source unchanged; only the tile instances, the final
// SMEM -> global copy loop for a 192-column tile (12 chunks per row, which does not divide 256 threads) and the
// blockIdx -> tile map are new. Tile geometry and CTA order never change an element's chain (e6 ANALYSIS 5.0; e11 SG
// byte sweep over five geometries, 14,592 runs, 0 differences).
//
// Kernel signature = NG7's (A, B, C, M, N, K, lda, ldc, chunk, mb, ws, flags); ws = the SiLU table; mb = rows in
// 192 x 192 body tiles (a multiple of 192); N = 2H with H a multiple of 384; K = 2560 (the template's; A and W rows are
// read at stride K). Grid = (H / 384) x (4 x mb / 192 + SC x ceil((M - mb) / BM_band)), SC = 384 / (band BN / 2).
// Build (reproducible): nvcc 13.4 -O3 -arch=sm_120a -cubin -o ngemm8.cubin ngemm8.cu
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

#define DEVI __device__ __forceinline__
typedef __nv_bfloat16 bf16;

// ------------------------------------------------------------------------------------------------ NG7 primitives
// (ngemm7x.cu, unchanged)
DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
// rows past M: ignore-src form (zeros written, the source is not read; its address may lie past the tensor)
DEVI void cp16z(uint32_t dst, const void* src, bool ok) {
    asm volatile("{\n .reg .pred p;\n setp.eq.u32 p, %2, 0;\n cp.async.cg.shared.global [%0], [%1], 16, p;\n}\n"
                 ::"r"(dst), "l"(src), "r"((uint32_t)ok));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
DEVI void cp_wait0() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
DEVI void ldsm4(uint32_t a, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
DEVI void stsm4(uint32_t a, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(r0), "r"(r1), "r"(r2),
                 "r"(r3) : "memory");
}
DEVI void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pack_bf16(float lo, float hi) {   // RN, lo in bits 0-15
    uint32_t r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
DEVI float bf_lo(uint32_t p) { return __uint_as_float(p << 16); }
DEVI float bf_hi(uint32_t p) { return __uint_as_float(p & 0xffff0000u); }

// ------------------------------------------------------------------------------------------------ tile configs
// NG7's T7 (unchanged): 8 warps; BK = 64 (one 128-byte smem row per tile row and stage, 16-byte chunk c stored at
// c ^ (row & 7)); 2 stages.
template <int BM_, int BN_, int WM_, int WN_>
struct T7 {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_;
    static_assert(WM * WN == 8, "8 warps");
    static constexpr int WTM = BM / WM, WTN = BN / WN, MT = WTM / 16, NT = WTN / 8;
    static constexpr int AI = BM / 32, BI = BN / 32;                 // 16-byte chunks per thread per operand and stage
    static constexpr int A_ST = BM * 128, B_ST = BN * 128, STAGE = A_ST + B_ST;
    static constexpr int PITCH = (BN + 8) * 2;                       // plain staging row (bytes): conflict-free stsm
    static constexpr int PITCH_SW = (BN / 2 + 8) * 2;                // SwiGLU staging row (BN / 2 outputs)
    static constexpr int SMEM = 2 * STAGE > BM * PITCH ? 2 * STAGE : BM * PITCH;
    static_assert(MT >= 1 && NT % 2 == 0 && AI >= 1, "tile");
};
using B7 = T7<128, 256, 2, 4>;      // NG7 big tile: 64x64 warp tiles (here a band of 1-2 rows of 128x256 tiles)
using S7c = T7<96, 256, 2, 4>;      // NG7 s96 band: 48x64 warp tiles
using T7t160 = T7<160, 128, 2, 4>;  // NG7 t160 band: 80x32 (rgh's NG9T geometry)
using T7t224 = T7<224, 128, 2, 4>;  // NG7 t224 band: 112x32 (rgh's NG9T geometry)
using Q8 = T7<192, 192, 4, 2>;      // E17 body tile: 48x96 warp tiles (MT 3, NT 12: 6 gate + 6 up n8 tiles per warp)
static_assert(Q8::SMEM == B7::SMEM, "the body tile keeps NG7's 96 KB");
static_assert((Q8::NT / 2) % 2 == 0, "gate / up n8 tile pairs must not straddle (bpair_off)");

// ------------------------------------------------------------------------------------------------ mainloop
// NG7's ml7 (unchanged): acc = A[m0:m0+BM, kb:kb+64*KT] @ W[rows, same]^T over ascending k. MODE 1 (SwiGLU, n0 =
// output column o0): smem B rows [0, BN/2) = gate rows o0 .. o0+BN/2-1, rows [BN/2, BN) = the same up rows (W rows
// H + o0 ..); warp wn reads gate rows wn*WTN/2 + [0, WTN/2) as its n8 tiles j < NT/2 and the matching up rows as tiles
// j >= NT/2, so acc[i][j] (j < NT/2) is a gate and acc[i][j + NT/2] the matching up column.
template <class T, int MODE>
DEVI uint32_t boff_of(int ks, int lane, int wn, uint32_t sm) {
    const int r = (MODE == 0 ? wn * T::WTN : wn * (T::WTN / 2)) + ((lane >> 4) << 3) + (lane & 7);
    return sm + T::A_ST + r * 128 + (((2 * ks + ((lane >> 3) & 1)) ^ (lane & 7)) << 4);
}
template <class T, int MODE>
__host__ __device__ constexpr int bpair_off(int j) {   // byte offset of n8 tile pair (j, j+1) from the warp's B base
    return (MODE == 0 || j < T::NT / 2) ? j * 1024 : (T::BN / 2) * 128 + (j - T::NT / 2) * 1024;
}
template <class T, int KC, int MODE>
DEVI void ml7(const bf16* __restrict__ A, const bf16* __restrict__ W, int M, int m0, int n0, int H, int kb, int KT,
              uint32_t sm, float (&acc)[T::MT][T::NT][4]) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    const int r0 = t >> 3, c = t & 7;
    const uint32_t dst = sm + r0 * 128 + ((c ^ (r0 & 7)) << 4);
    const bf16* gA = A + (size_t)(m0 + r0) * KC + kb + c * 8;
    uint32_t okm = 0;
#pragma unroll
    for (int i = 0; i < T::AI; ++i) okm |= (m0 + r0 + 32 * i < M) ? (1u << i) : 0u;
    const bf16* gB0 = W + (size_t)(n0 + r0) * KC + kb + c * 8;      // MODE 1: gate rows o0 + r0 + 32 i
    const bf16* gB1 = gB0 + (size_t)H * KC;                           // MODE 1: the up rows (+H)
    auto load = [&](int st, int kt) {
        const uint32_t d = dst + st * T::STAGE;
        const int ko = kt * 64;
#pragma unroll
        for (int i = 0; i < T::AI; ++i) cp16z(d + i * 4096, gA + i * 32 * KC + ko, (okm >> i) & 1u);
#pragma unroll
        for (int i = 0; i < T::BI; ++i) {
            const bf16* p;
            if constexpr (MODE == 0) p = gB0 + i * 32 * KC;
            else p = (i < T::BI / 2) ? gB0 + i * 32 * KC : gB1 + (i - T::BI / 2) * 32 * KC;
            cp16(d + T::A_ST + i * 4096, p + ko);
        }
        cp_commit();
    };
    uint32_t aoff[4], boff[4];
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
        aoff[ks] = sm + (wm * T::WTM + (lane & 15)) * 128 + (((2 * ks + (lane >> 4)) ^ (lane & 7)) << 4);
        boff[ks] = boff_of<T, MODE>(ks, lane, wn, sm);
    }
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
    uint32_t af[2][T::MT][4];
    uint32_t bfr[2][T::NT][2];
    auto frags = [&](int buf, uint32_t so, int ks) {
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
            ldsm4(aoff[ks] + so + i * 2048, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < T::NT; j += 2)
            ldsm4(boff[ks] + so + bpair_off<T, MODE>(j), bfr[buf][j][0], bfr[buf][j][1], bfr[buf][j + 1][0],
                  bfr[buf][j + 1][1]);
    };
    load(0, 0);
    cp_wait0();
    __syncthreads();
    frags(0, 0, 0);
#pragma unroll 1
    for (int kt = 0; kt < KT; ++kt) {
        const uint32_t so = (kt & 1) * T::STAGE, sn = so ^ T::STAGE;
        const bool more = kt + 1 < KT;
        if (more) load((kt & 1) ^ 1, kt + 1);
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            if (ks < 3) {
                frags((ks + 1) & 1, so, ks + 1);
            } else {
                if (more) cp_wait0();
                __syncthreads();
                if (more) frags(0, sn, 0);
            }
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int j = 0; j < T::NT; ++j) mma16816(acc[i][j], af[ks & 1][i], bfr[ks & 1][j][0], bfr[ks & 1][j][1]);
        }
    }
}

// ------------------------------------------------------------------------------------------------ SwiGLU epilogue
// NG7's swiglu_pair (unchanged): out bf16 pair (columns c, c+1) from the gate pair (a0, a1) and up pair (b0, b1)
// accumulators: g and u rounded to bf16 (RN), s = lut[bits(g)] (torch's F.silu of that bf16 value),
// out = bf16(float(s) * float(u)) (RN)
DEVI uint32_t swiglu_pair(float a0, float a1, float b0, float b1, const uint16_t* __restrict__ lut) {
    const uint32_t gp = pack_bf16(a0, a1), up = pack_bf16(b0, b1);
    const float u0 = bf_lo(up), u1 = bf_hi(up);
    const float s0 = __uint_as_float(uint32_t(__ldg(lut + (gp & 0xffffu))) << 16);
    const float s1 = __uint_as_float(uint32_t(__ldg(lut + (gp >> 16))) << 16);
    return pack_bf16(__fmul_rn(s0, u0), __fmul_rn(s1, u1));
}

// NG7's store7_swiglu: outputs of the tile = BM rows x BN/2 columns starting at output column o0; staging pitch
// PITCH_SW. E17: the final SMEM -> global copy keeps NG7's loop when the 16-byte chunks per row divide the 256 threads
// (every NG7 tile) and walks the BM x CPR chunks linearly otherwise (Q8: 12 chunks per row, 192 x 12 = 9 x 256).
// The staged bytes are the outputs; the copy moves them unchanged.
template <class T>
DEVI void store7_swiglu(const float (&acc)[T::MT][T::NT][4], bf16* __restrict__ O, int M, int ldo, int m0, int o0,
                        uint32_t sm, const uint16_t* __restrict__ lut) {
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    constexpr int NH = T::NT / 2;                     // gate n8 tiles per warp
    constexpr int P = T::PITCH_SW;
    __syncthreads();
    const uint32_t base = sm + (wm * T::WTM + ((lane >> 3) & 1) * 8 + (lane & 7)) * P + (wn * (T::WTN / 2) +
                                                                                          (lane >> 4) * 8) * 2;
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < NH; j += 2) {
            const uint32_t r0 = swiglu_pair(acc[i][j][0], acc[i][j][1], acc[i][j + NH][0], acc[i][j + NH][1], lut);
            const uint32_t r1 = swiglu_pair(acc[i][j][2], acc[i][j][3], acc[i][j + NH][2], acc[i][j + NH][3], lut);
            const uint32_t r2 = swiglu_pair(acc[i][j + 1][0], acc[i][j + 1][1], acc[i][j + 1 + NH][0],
                                            acc[i][j + 1 + NH][1], lut);
            const uint32_t r3 = swiglu_pair(acc[i][j + 1][2], acc[i][j + 1][3], acc[i][j + 1 + NH][2],
                                            acc[i][j + 1 + NH][3], lut);
            stsm4(base + i * 16 * P + j * 16, r0, r1, r2, r3);
        }
    __syncthreads();
    constexpr int CPR = T::BN / 16;                   // 16-byte chunks per output row
    if constexpr (256 % CPR == 0) {
        constexpr int RPI = 256 / CPR;                // rows per pass
        const int cc = t % CPR, rr = t / CPR;
#pragma unroll
        for (int q = 0; q < T::BM / RPI; ++q) {
            const int r = rr + q * RPI;
            uint4 v;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "r"(sm + r * P + cc * 16));
            if (m0 + r < M) *reinterpret_cast<uint4*>(O + (size_t)(m0 + r) * ldo + o0 + cc * 8) = v;
        }
    } else {
        static_assert((T::BM * CPR) % 256 == 0, "whole passes");
#pragma unroll
        for (int q = 0; q < T::BM * CPR / 256; ++q) {
            const int idx = t + q * 256, r = idx / CPR, cc = idx % CPR;
            uint4 v;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                         : "r"(sm + r * P + cc * 16));
            if (m0 + r < M) *reinterpret_cast<uint4*>(O + (size_t)(m0 + r) * ldo + o0 + cc * 8) = v;
        }
    }
}

// ------------------------------------------------------------------------------------------------ kernels
#define ARGS                                                                                                       \
    const bf16 *__restrict__ A, const bf16 *__restrict__ B, bf16 *__restrict__ Cm, int M, int N, int K, int lda,   \
        int ldc, int chunk, int mb, void *__restrict__ ws, int *__restrict__ flags

constexpr int GO = 384;   // outputs per column group (lcm of the tiles' 96 / 128 / 64 output columns)

// The CTA -> tile map (shared with the host test tests/ng8_map_host.cu, which compiles this function from this file):
// rows [0, mb) in TQ (192 x 192) body tiles, rows [mb, M) in TS band tiles (mts rows of them; mts = 0 when mb = M).
// CTA order (L2-aware): column group g (outputs [g * GO, (g + 1) * GO)) = its QC x mq body tiles (row tile fastest),
// then its SC x mts band tiles (row fastest); groups in order. grid = (H / GO) x per.
struct Map8 {
    int band, m0, o0;   // band 0: a TQ tile, 1: a TS tile; first row; first output column
};
template <class TQ, class TS>
__host__ __device__ __forceinline__ int ng8_per(int M, int mb) {
    return (GO / (TQ::BN / 2)) * (mb / TQ::BM) + (GO / (TS::BN / 2)) * ((M - mb + TS::BM - 1) / TS::BM);
}
template <class TQ, class TS>
__host__ __device__ __forceinline__ Map8 ng8_map(int b, int M, int mb) {
    static_assert(GO % (TQ::BN / 2) == 0 && GO % (TS::BN / 2) == 0, "column groups");
    constexpr int QC = GO / (TQ::BN / 2), SC = GO / (TS::BN / 2);
    const int mq = mb / TQ::BM, mts = (M - mb + TS::BM - 1) / TS::BM;
    const int per = QC * mq + SC * mts;
    const int g = b / per;
    int r = b - g * per;
    if (r < QC * mq) return Map8{0, (r % mq) * TQ::BM, (g * QC + r / mq) * (TQ::BN / 2)};
    r -= QC * mq;
    return Map8{1, mb + (r % mts) * TS::BM, (g * SC + r / mts) * (TS::BN / 2)};
}

// SwiGLU: W = [Wg; Wu] (N = 2H rows); out [M, H] (ldc = its row stride); ws = the 64K-entry SiLU table (uint16, index
// = the bf16 bits of g). Every output tile is covered exactly once (tests/ng8_map_host.cu over every M <= 4096).
template <class TQ, class TS, int KC>
DEVI void nsw8_body(ARGS) {
    extern __shared__ __align__(1024) unsigned char smem_raw[];
    const uint32_t sm = smem_u32(smem_raw);
    const int H = N / 2;
    const uint16_t* lut = reinterpret_cast<const uint16_t*>(ws);
    const Map8 t = ng8_map<TQ, TS>(blockIdx.x, M, mb);
    if (t.band == 0) {
        float acc[TQ::MT][TQ::NT][4];
        ml7<TQ, KC, 1>(A, B, M, t.m0, t.o0, H, 0, KC / 64, sm, acc);
        store7_swiglu<TQ>(acc, Cm, M, ldc, t.m0, t.o0, sm, lut);
    } else {
        float acc[TS::MT][TS::NT][4];
        ml7<TS, KC, 1>(A, B, M, t.m0, t.o0, H, 0, KC / 64, sm, acc);
        store7_swiglu<TS>(acc, Cm, M, ldc, t.m0, t.o0, sm, lut);
    }
}

#define NSW8(NAME, TQ, TS, KC)                                                                                      \
    extern "C" __global__ void __launch_bounds__(256, 1) NAME(ARGS) {                                               \
        nsw8_body<TQ, TS, KC>(A, B, Cm, M, N, K, lda, ldc, chunk, mb, ws, flags);                                   \
    }

// gate|up + SwiGLU (18432 x 2560 -> 9216): 192 x 192 body + one band type each (e17 ana/geom.py picks per (L, R))
NSW8(nsw8_q192_b128x256_k2560, Q8, B7, 2560)       // band: 0-2 rows of 128x256 (128 outputs per tile)
NSW8(nsw8_q192_s96x256_k2560, Q8, S7c, 2560)       // band: one row of 96x256
NSW8(nsw8_q192_t160x128_k2560, Q8, T7t160, 2560)   // band: one row of 160x128 (64 outputs per tile)
NSW8(nsw8_q192_t224x128_k2560, Q8, T7t224, 2560)   // band: one row of 224x128

// ------------------------------------------------------------------------------------------------ table check
// NG7's swiglu7_elem_lut in this module (the epilogue arithmetic of the nsw8 kernels, elementwise): ng.check_ng8 runs
// it over every bf16 g x 8 u values at start-up against torch F.silu(g) * u and the engine's silu_mul
extern "C" __global__ void swiglu8_elem_lut(const uint16_t* g, const uint16_t* u, uint16_t* o, const uint16_t* lut,
                                            int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gf = __uint_as_float(uint32_t(g[i]) << 16), uf = __uint_as_float(uint32_t(u[i]) << 16);
    o[i] = uint16_t(swiglu_pair(gf, 0.f, uf, 0.f, lut) & 0xffffu);
}
