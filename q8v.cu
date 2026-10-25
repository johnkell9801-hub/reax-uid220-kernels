// Q8-V (pjs e35, ours; S1ENG_Q8V) for rgh's ROT builds (j205 / j206): exact SCHEDULE variants of rgh's Q8 int8 GEMM
// kernels q8h_* / q8g_* / q8gr_* (q8.cu, rgh native rewrite 10-24 + ROT) and of pjs's Q8-WT q8h twins (q8wt.cu): C1 = the
// int8 main loop and the protected bf16 pass share one cp.async ring; C1T (q8g / q8gr) = C1 + rgh's SwiGLU epilogue
// reading bf16(silu(g)) from a 65,536-entry table built with rgh's own swiglu1 (q8v_silu_tab). Every output byte equals
// the kernel it replaces (checked at start-up on the real weights, q8v.py). Everything up to the "==== E35" marker below
// is rgh's j206 q8.cu text (its namespace part) unchanged but for pjs's Q8-WT Loader TB edit (q8wt.cu's 3 lines, default
// TB = false = rgh's loads); the variants follow it.
// ---- rgh's j206 q8.cu (namespace part) + the Q8-WT Loader TB edit follows ----
// Q8 (rgh, native rewrite 10-24): W8A8 int8 projections for sm_120 (RTX PRO 6000).
//   out[r, n] = sa[r] * sw[n] * (Aq[r, :] . Bq[n, :]) + bias[n] + AH[r, :] . BH[n, :]
// Aq [M, K] / Bq [N, K] int8 (per-row activation scales sa, per-output-column weight scales sw, an fp32 bias from the
// calibrated channel centres), AH [M, KH] / BH [N, KH] bf16 = the protected (outlier) channels, exact bf16 products.
// Mainloop = NG9B's (ngemm9b.cu: cp.async ring, XOR-swizzled smem + ldmatrix, double-buffered fragments, one barrier
// per k-tile, serpentine mma order) with the int8 operands addressed as 2-byte units: a 64-unit k-tile = 128 int8
// columns, each 16-unit k-step one mma.sync m16n8k32 s8 (A / B / C fragments have the m16n8k16 bf16 register layout
// byte for byte, so the ldmatrix addressing and the accumulator map are unchanged). The int32 sums are exact; after
// the int8 loop they are dequantised in registers and the protected channels run the same mainloop as bf16 m16n8k16
// into those fp32 accumulators.
// Kernels (all: M rows, rows >= M never loaded / stored; N % BN == 0; K % (2 BK) == 0; KH % BK == 0):
//   q8h_<big>_<small>: bf16 out [M, N] (ldo), rows [0, mb) big tiles (CTAs 0 .. Tb-1, m fastest), the rest small
//   q8w_<big>_<small>: N = 2H stacked [Wg; Wu], out [M, H] = SwiGLU(g, u) (gate / up rows interleaved per warp column)
//   q8p_<big>_<small>: grouped split-K, blockIdx.y = group p over int8 columns [p G, (p + 1) G) of Aq / Bq (K = G):
//                      fp32 part[p][M][N] = sa[p M + r] * sw[p N + n] * acc + (p == 0 ? bias[n] : 0) (+ AH . BH, p 0)
//   q8r_sum: out[M, N] bf16 = sum_p part[p] (p ascending, fp32), N % 4 == 0
//   q8x_prep: activation prepare, grid (M, P): x[r, lo[c]] -> v = (x - mu[c]) * is[c]; group p of G columns: amax,
//             sa[p M + r] = amax / 127, xq[r, c] = rn(v * 127 / amax) (int8); hi: xh[r, j] = x[r, hi[j]] (group 0 CTAs)
#include <cuda_bf16.h>
#include <stdint.h>
#include <type_traits>

#define DEVI __device__ __forceinline__

namespace q8 {

using U = uint16_t;   // the 2-byte addressing unit of both operand kinds

DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
DEVI void cp_async16(uint32_t dst, const void* src, bool pred) {
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
DEVI void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEVI void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
DEVI void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void mma_s8(int* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI void mma_s8z(int* d, const uint32_t* a, uint32_t b0, uint32_t b1) {   // d = a . b (C = 0)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
DEVI void mma_bf(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <class T>
DEVI void mma(T* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    if constexpr (std::is_same<T, int>::value) mma_s8(d, a, b0, b1);
    else mma_bf(d, a, b0, b1);
}
DEVI void named_sync(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
DEVI uint32_t pack_bf2(float a, float b) {
    __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
DEVI float silu_f32(float x) { return __fdiv_rn(x, 1.0f + expf(-x)); }
DEVI __nv_bfloat16 swiglu1(__nv_bfloat16 g, __nv_bfloat16 u) {
    float s = __bfloat162float(__float2bfloat16_rn(silu_f32(__bfloat162float(g))));
    return __float2bfloat16_rn(s * __bfloat162float(u));
}

// ROT (rgh 10-24): randomised 128-point Hadamard rotation of a fine group, v -> H_128 (d o v) / sqrt(128) with d_i = -1
// where bit i of ROT_NEG is set: the same R = diag(d) H / sqrt(128) as the rotated weights of q8_gptq_rot.pt (the
// 1 / sqrt(128) is folded into the stored activation scale)
__device__ __constant__ uint32_t ROT_NEG[4] = {0x09bab9abu, 0xfc955699u, 0xfc9121c0u, 0xafc1071cu};
constexpr float ROT_RS = 0.08838834764831845f;
DEVI float rot_sign(int i, float v) { return ((ROT_NEG[i >> 5] >> (i & 31)) & 1u) ? -v : v; }
// butterflies over the index bits held in one thread (N consecutive elements) ...
template <int N>
DEVI void fwht_reg(float (&v)[N]) {
#pragma unroll
    for (int st = 1; st < N; st <<= 1)
#pragma unroll
        for (int e = 0; e < N; ++e)
            if ((e & st) == 0) {
                const float x = v[e], y = v[e + st];
                v[e] = x + y;
                v[e + st] = x - y;
            }
}
// ... then over the higher index bits held by lanes (lane masks 1 .. MMAX)
template <int N, int MMAX>
DEVI void fwht_lanes(float (&v)[N], int lane) {
#pragma unroll
    for (int m = 1; m <= MMAX; m <<= 1)
#pragma unroll
        for (int e = 0; e < N; ++e) {
            const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
            v[e] = (lane & m) ? o - v[e] : v[e] + o;
        }
}

template <int BK>
DEVI int swz_f(int r) {
    if constexpr (BK == 32) return (r >> 1) & 3;
    else return r & 7;
}

// EPI: 2 = plain (bf16 stmatrix epilogue or fp32 partials), 1 = SwiGLU (interleaved B rows); BK in 2-byte units
template <int BM_, int BN_, int BK_, int WARPS_M, int WARPS_N, int STAGES_, int EPI_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WARPS_M, WN = WARPS_N, EPI = EPI_;
    static constexpr int CW = WARPS_M * WARPS_N;
    static constexpr int THREADS = CW * 32;
    static constexpr int WTM = BM / WARPS_M, WTN = BN / WARPS_N;
    static constexpr int MT = WTM / 16, NT = WTN / 8, KS = BK / 16, CH = BK / 8;
    static constexpr int A_STAGE = BM * BK, B_STAGE = BN * BK;
    static constexpr int SMEM_PIPE = STAGES * (A_STAGE + B_STAGE) * 2;
    static constexpr int CPAD = 8;
    static constexpr int SMEM_EPI = BM * (BN + CPAD) * 2;
    static constexpr int SMEM = SMEM_PIPE > SMEM_EPI ? SMEM_PIPE : SMEM_EPI;
    static_assert(MT >= 1 && NT % 2 == 0 && KS % 2 == 0 && STAGES >= 2, "tile");
};

// per-thread chunk loader (NG9B's): chunk q < NA = A rows r0 + q RS, q >= NA = B rows r0 + (q - NA) RS; offsets in units
template <class C, bool TB = false>   // pjs e33 Q8-WT: TB = the tile-contiguous int8 B operand (q8wt.cu's Loader TB)
struct Loader {
    static constexpr int RS = C::THREADS / C::CH;
    static constexpr int NA = (C::BM + RS - 1) / RS;
    static constexpr int NB = (C::BN + RS - 1) / RS;
    static constexpr int NQ = NA + NB;
    const U* pa;
    const U* pb;
    uint32_t offa[NA], offb[NB];
    uint32_t dst0;
    uint32_t amask;
    int r0;
    // K, lda, ldb in units; n1 >= 0: SwiGLU interleave (gate rows from n0, up rows from n1)
    DEVI void init(const U* A, const U* B, int m0, int n0, int n1, int M, int lda, int ldb) {
        const int tid = threadIdx.x;
        r0 = tid / C::CH;
        const int c = tid % C::CH;
        dst0 = (r0 * C::BK + ((c ^ swz_f<C::BK>(r0)) * 8)) * 2;
        pa = A;
        pb = B;
        amask = 0;
#pragma unroll
        for (int q = 0; q < NA; ++q) {
            const int r = r0 + q * RS;
            const int gm = m0 + r;
            const bool ok = gm < M;
            offa[q] = (uint32_t)((ok ? gm : 0) * lda + c * 8);
            amask |= (ok ? 1u : 0u) << q;
        }
#pragma unroll
        for (int q = 0; q < NB; ++q) {
            const int r = r0 + q * RS;
            int wr;
            if (C::EPI == 1 && n1 >= 0) {
                const int wn = r / C::WTN, h = (r % C::WTN) >= C::WTN / 2, cc = r % (C::WTN / 2);
                wr = (h ? n1 : n0) + wn * (C::WTN / 2) + cc;
            } else {
                wr = n0 + r;
            }
            if constexpr (TB) offb[q] = (uint32_t)((wr >> 5) * (ldb / 64) * 2048 + (wr & 31) * 64 + c * 8);   // Q8-WT
            else offb[q] = (uint32_t)(wr * ldb + c * 8);
        }
    }
    DEVI void issue_all(uint32_t sA, uint32_t sB, int k0) {
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            if (q < NA) {
                if (C::BM % RS == 0 || r0 + q * RS < C::BM)
                    cp_async16(sA + dst0 + q * RS * C::BK * 2, pa + offa[q] + k0, (amask >> q) & 1);
            } else {
                const int qb = q - NA;
                if (C::BN % RS == 0 || r0 + qb * RS < C::BN)
                    cp_async16(sB + dst0 + qb * RS * C::BK * 2, pb + offb[qb] + (TB ? k0 * 32 : k0), true);   // Q8-WT: k-tile kt = k0 / 64 at kt * 4096 B
            }
        }
    }
};

template <class C>
struct Frag {
    uint32_t a_off[C::KS], b_off[C::KS];
    uint32_t af[2][C::MT][4];
    uint32_t bf[2][C::NT][2];
    DEVI void init(int wm, int wn, int lane) {
        constexpr int BK = C::BK;
        const int a_r = wm * C::WTM + (lane % 16);
        const int a_f = swz_f<BK>(lane % 16);
        const int a_kc = lane / 16;
        const int b_r = wn * C::WTN + (lane / 16) * 8 + (lane % 8);
        const int b_f = swz_f<BK>((lane / 16) * 8 + (lane % 8));
        const int b_kc = (lane / 8) % 2;
#pragma unroll
        for (int ks = 0; ks < C::KS; ++ks) {
            a_off[ks] = (a_r * BK + (((ks * 2 + a_kc) ^ a_f) * 8)) * 2;
            b_off[ks] = (b_r * BK + (((ks * 2 + b_kc) ^ b_f) * 8)) * 2;
        }
    }
    DEVI void load(int buf, uint32_t sA, uint32_t sB, int ks) {
        const uint32_t a = sA + a_off[ks];
        const uint32_t b = sB + b_off[ks];
#pragma unroll
        for (int i = 0; i < C::MT; ++i)
            ldsm_x4(a + i * 16 * C::BK * 2, af[buf][i][0], af[buf][i][1], af[buf][i][2], af[buf][i][3]);
#pragma unroll
        for (int j = 0; j < C::NT; j += 2)
            ldsm_x4(b + j * 8 * C::BK * 2, bf[buf][j][0], bf[buf][j][1], bf[buf][j + 1][0], bf[buf][j + 1][1]);
    }
};

// acc (+)= A[m0:m0+BM, :Ku] . B-tile[:, :Ku] (Ku units); T = int (int8 data, m16n8k32 s8) or float (bf16, m16n8k16)
template <class C, class T, bool ZERO>
DEVI void mainloop(const U* __restrict__ A, const U* __restrict__ B, int M, int Ku, int lda, int ldb, int m0, int n0,
                   int n1, U* smem, T (&acc)[C::MT][C::NT][4]) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = Ku / BK;
    auto sa = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
    if constexpr (ZERO) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][j][q] = T(0);
    }
    Loader<C> ld;
    ld.init(A, B, m0, n0, n1, M, lda, ldb);
#pragma unroll
    for (int t = 0; t < S - 1; ++t) {
        if (t < KT) ld.issue_all(sa(t), sb(t), t * BK);
        cp_commit();
    }
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<S - 2>();
    __syncthreads();
    fr.load(0, sa(0), sb(0), 0);
    int s = 0;
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) {
                const int tt = kt + S - 1;
                if (tt < KT) ld.issue_all(sa(tt % S), sb(tt % S), tt * BK);
            }
            if (ks < KS - 1 || kt + 1 < KT) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sa(ns), sb(ns), ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int ii = 0; ii < MT; ++ii) {
                    const int i = (j & 1) ? MT - 1 - ii : ii;
                    mma<T>(acc[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
        }
        if (++s == S) s = 0;
    }
    cp_wait<0>();
    __syncthreads();
}

// the global column of accumulator (j, e) of this thread: plain tiles n0 + wn WTN + j 8 + 2 t4 + e; SwiGLU tiles
// (o0 = the tile's first output column, H = N / 2): gate (j < NT / 2) o0 + .., up H + o0 + ..
template <class C>
DEVI int col_of(int j, int e, int nbase, int H) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wn = warp % C::WN, t4 = lane % 4;
    if constexpr (C::EPI == 1) {
        constexpr int NH = C::NT / 2;
        const int jj = j < NH ? j : j - NH;
        return (j < NH ? 0 : H) + nbase + wn * (C::WTN / 2) + jj * 8 + t4 * 2 + e;
    } else {
        return nbase + wn * C::WTN + j * 8 + t4 * 2 + e;
    }
}
template <class C>
DEVI int row_of(int i, int h, int m0) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN;
    return m0 + wm * C::WTM + i * 16 + lane / 4 + h * 8;
}

// int32 sums -> fp32: sa[r] sw[n] acc + bias[n] (bias may be null)
template <class C>
DEVI void dequant(const int (&ia)[C::MT][C::NT][4], float (&fa)[C::MT][C::NT][4], const float* __restrict__ sa,
                  const float* __restrict__ sw, const float* __restrict__ bias, int M, int m0, int nbase, int H) {
    float rs[C::MT][2];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row_of<C>(i, h, m0);
            rs[i][h] = r < M ? sa[r] : 0.f;
        }
#pragma unroll
    for (int j = 0; j < C::NT; ++j) {
        const int c0 = col_of<C>(j, 0, nbase, H);
        const float2 w = *reinterpret_cast<const float2*>(sw + c0);
        const float2 b = bias ? *reinterpret_cast<const float2*>(bias + c0) : make_float2(0.f, 0.f);
#pragma unroll
        for (int i = 0; i < C::MT; ++i)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float s = rs[i][q >> 1] * ((q & 1) ? w.y : w.x);
                fa[i][j][q] = fmaf((float)ia[i][j][q], s, (q & 1) ? b.y : b.x);
            }
    }
}

// plain bf16 epilogue: stmatrix.x4 -> padded smem tile -> 16-byte stores of rows < M (NG9B's store_tile)
template <class C>
DEVI void store_tile(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Cm, int M, int ldc, int m0, int n0,
                     U* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    constexpr int LD = C::BN + C::CPAD;
    const int mi = lane / 8, rr = lane % 8;
    const uint32_t base = smem_u32(smem) +
        (uint32_t)(((wm * C::WTM + (mi & 1) * 8 + rr) * LD + wn * C::WTN + (mi >> 1) * 8) * 2);
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NT; j += 2) {
            const uint32_t a = base + (uint32_t)((i * 16 * LD + j * 8) * 2);
            asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(a),
                         "r"(pack_bf2(v[i][j][0], v[i][j][1])), "r"(pack_bf2(v[i][j][2], v[i][j][3])),
                         "r"(pack_bf2(v[i][j + 1][0], v[i][j + 1][1])), "r"(pack_bf2(v[i][j + 1][2], v[i][j + 1][3]))
                         : "memory");
        }
    named_sync(1, C::THREADS);
    constexpr int CPR = C::BN / 8;
    const __nv_bfloat16* sm = reinterpret_cast<const __nv_bfloat16*>(smem);
    for (int i = threadIdx.x; i < C::BM * CPR; i += C::THREADS) {
        int r = i / CPR, c = i % CPR;
        int gm = m0 + r;
        if (gm < M)
            *reinterpret_cast<uint4*>(Cm + (size_t)gm * ldc + n0 + c * 8) = *reinterpret_cast<const uint4*>(sm + r * LD + c * 8);
    }
}

// SwiGLU register epilogue (NG9B's store_swiglu): n-tile j < NT/2 = gate columns, j + NT/2 the matching up columns
template <class C>
DEVI void store_swiglu(const float (&v)[C::MT][C::NT][4], __nv_bfloat16* __restrict__ Out, int M, int ldo, int m0,
                       int o0) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int g = lane / 4, t4 = lane % 4;
    constexpr int NH = C::NT / 2;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = m0 + wm * C::WTM + i * 16 + g + h * 8;
            if (r >= M) continue;
            __nv_bfloat16* row = Out + (size_t)r * ldo + o0 + wn * (C::WTN / 2) + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const __nv_bfloat162 gg = __floats2bfloat162_rn(v[i][j][2 * h], v[i][j][2 * h + 1]);
                const __nv_bfloat162 uu = __floats2bfloat162_rn(v[i][j + NH][2 * h], v[i][j + NH][2 * h + 1]);
                __nv_bfloat162 o;
                o.x = swiglu1(gg.x, uu.x);
                o.y = swiglu1(gg.y, uu.y);
                *reinterpret_cast<__nv_bfloat162*>(row + j * 8) = o;
            }
        }
}

// fp32 partial epilogue: float2 stores of rows < M
template <class C>
DEVI void store_part(const float (&v)[C::MT][C::NT][4], float* __restrict__ P, int M, int N, int m0, int n0) {
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = row_of<C>(i, h, m0);
            if (r >= M) continue;
            float* row = P + (size_t)r * N;
#pragma unroll
            for (int j = 0; j < C::NT; ++j)
                *reinterpret_cast<float2*>(row + col_of<C>(j, 0, n0, 0)) = make_float2(v[i][j][2 * h], v[i][j][2 * h + 1]);
        }
}

struct Args {
    const int8_t* A;        // [M, lda] int8
    const int8_t* B;        // [N, ldb] int8
    const __nv_bfloat16* AH;   // [M, KH] bf16 (KH = 0: none)
    const __nv_bfloat16* BH;   // [N, KH]
    const float* sa;        // [P][M]
    const float* sw;        // [P][N]
    const float* bias;      // [N] or null
    void* out;              // bf16 [M, ldo] (q8h / q8w) or fp32 [P][M][N] (q8p)
    int M, N, K, KH, lda, ldb, ldo, mb;
};

// one tile: EPI 2 + PART false = bf16 plain, EPI 1 = SwiGLU, PART = fp32 partial of group blockIdx.y
template <class C, bool PART>
DEVI void tile(const Args& a, int m0, int nt, U* smem) {
    const int p = PART ? blockIdx.y : 0;
    const int H = C::EPI == 1 ? a.N / 2 : 0;
    const int nbase = C::EPI == 1 ? nt * (C::BN / 2) : nt * C::BN;
    const int n1 = C::EPI == 1 ? H + nbase : -1;
    int ia[C::MT][C::NT][4];
    const U* A = reinterpret_cast<const U*>(a.A + (size_t)p * a.K);
    const U* B = reinterpret_cast<const U*>(a.B + (size_t)p * a.K);
    mainloop<C, int, true>(A, B, a.M, a.K / 2, a.lda / 2, a.ldb / 2, m0, nbase, n1, smem, ia);
    float fa[C::MT][C::NT][4];
    dequant<C>(ia, fa, a.sa + (size_t)p * a.M, a.sw + (size_t)p * a.N, p == 0 ? a.bias : nullptr, a.M, m0, nbase, H);
    if (a.KH > 0 && p == 0)
        mainloop<C, float, false>(reinterpret_cast<const U*>(a.AH), reinterpret_cast<const U*>(a.BH), a.M, a.KH, a.KH,
                                  a.KH, m0, nbase, n1, smem, fa);
    if constexpr (PART) {
        store_part<C>(fa, reinterpret_cast<float*>(a.out) + (size_t)p * a.M * a.N, a.M, a.N, m0, nbase);
    } else if constexpr (C::EPI == 1) {
        store_swiglu<C>(fa, reinterpret_cast<__nv_bfloat16*>(a.out), a.M, a.ldo, m0, nbase);
    } else {
        store_tile<C>(fa, reinterpret_cast<__nv_bfloat16*>(a.out), a.M, a.ldo, m0, nbase, smem);
    }
}

// rows [0, mb) big tiles (CTAs 0 .. Tb-1, m fastest), rows [mb, M) small tiles
template <class CB, class CS, bool PART>
DEVI void hetero(const Args& a, U* smem) {
    static_assert(CB::THREADS == CS::THREADS, "threads");
    const int ncols = CB::EPI == 1 ? a.N / 2 : a.N;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (ncols / (CB::EPI == 1 ? CB::BN / 2 : CB::BN));
    int b = blockIdx.x;
    if (b < tb) {
        tile<CB, PART>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile<CS, PART>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

// FINE (DeepSeek-style fine-grained scaling): the int8 k-tile of 128 columns = one quantisation group g with its own
// per-row activation scale sa[g M + r] and per-column weight scale sw[g N + n] (wfull: sw[n] for every group); the
// int32 sums of each k-tile are dequantised into the fp32 accumulators at its last k-step:
// fa += (sa sw) * acc_kt (scales loaded at the tile's first k-step, used three k-steps later)
template <class C>
DEVI void mainloop_fine(const U* __restrict__ A, const U* __restrict__ B, int M, int N, int Ku, int lda, int ldb, int m0,
                        int n0, U* smem, float (&fa)[C::MT][C::NT][4], const float* __restrict__ sa,
                        const float* __restrict__ sw, int wfull) {
    constexpr int KS = C::KS, BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    static_assert(BK == 64, "one k-tile = one 128-column group");
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT = Ku / BK;
    auto sa_ = [&](int s) { return sA0 + s * C::A_STAGE * 2; };
    auto sb_ = [&](int s) { return sB0 + s * C::B_STAGE * 2; };
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) fa[i][j][q] = 0.f;
    const int r0 = row_of<C>(0, 0, m0);    // rows r0 + 16 i + 8 h
    const int c0 = col_of<C>(0, 0, n0, 0);  // columns c0 + 8 j (+ 1)
    float wa[NT][2];
    if (wfull) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const float2 w = *reinterpret_cast<const float2*>(sw + c0 + 8 * j);
            wa[j][0] = w.x;
            wa[j][1] = w.y;
        }
    }
    Loader<C> ld;
    ld.init(A, B, m0, n0, -1, M, lda, ldb);
#pragma unroll
    for (int t = 0; t < S - 1; ++t) {
        if (t < KT) ld.issue_all(sa_(t), sb_(t), t * BK);
        cp_commit();
    }
    Frag<C> fr;
    fr.init(warp / C::WN, warp % C::WN, lane);
    cp_wait<S - 2>();
    __syncthreads();
    fr.load(0, sa_(0), sb_(0), 0);
    int s = 0;
    int ia[MT][NT][4];
    float ra[MT][2];
    for (int kt = 0; kt < KT; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == 0) {   // this group's scales (consumed at the last k-step)
                const float* sr = sa + (size_t)kt * M;
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = r0 + 16 * i + 8 * h;
                        ra[i][h] = r < M ? sr[r] : 0.f;
                    }
                if (!wfull) {
                    const float* sc = sw + (size_t)kt * N + c0;
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        const float2 w = *reinterpret_cast<const float2*>(sc + 8 * j);
                        wa[j][0] = w.x;
                        wa[j][1] = w.y;
                    }
                }
            }
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) {
                const int tt = kt + S - 1;
                if (tt < KT) ld.issue_all(sa_(tt % S), sb_(tt % S), tt * BK);
            }
            if (ks < KS - 1 || kt + 1 < KT) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sa_(ns), sb_(ns), ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int ii = 0; ii < MT; ++ii) {
                    const int i = (j & 1) ? MT - 1 - ii : ii;
                    if (ks == 0) mma_s8z(ia[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                    else mma_s8(ia[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
            if (ks == KS - 1) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    const float w0 = wa[j][0], w1 = wa[j][1];
#pragma unroll
                    for (int i = 0; i < MT; ++i)
#pragma unroll
                        for (int q = 0; q < 4; ++q)
                            fa[i][j][q] = fmaf((float)ia[i][j][q], ra[i][q >> 1] * ((q & 1) ? w1 : w0), fa[i][j][q]);
                }
            }
        }
        if (++s == S) s = 0;
    }
    cp_wait<0>();
    __syncthreads();
}

struct ArgsF {
    const int8_t* A;     // [M, lda] int8 (all groups)
    const int8_t* B;     // [N, ldb] int8
    const float* sa;     // [G][M]
    const float* sw;     // [G][N] (wfull: [N])
    const float* bias;   // [N] or null (added by split 0)
    float* out;          // fp32 partials [S][M][N]
    int M, N, Ks, lda, ldb, mb, wfull, pad;   // Ks = int8 columns per split (a multiple of 128)
};

// one fine split-K tile: split blockIdx.y covers int8 columns [y Ks, (y + 1) Ks) = groups y Ks / 128 ..
template <class C>
DEVI void tile_f(const ArgsF& a, int m0, int nt, U* smem) {
    const int y = blockIdx.y;
    const int n0 = nt * C::BN;
    const int g0 = y * (a.Ks / 128);
    float fa[C::MT][C::NT][4];
    mainloop_fine<C>(reinterpret_cast<const U*>(a.A + (size_t)y * a.Ks), reinterpret_cast<const U*>(a.B + (size_t)y * a.Ks),
                     a.M, a.N, a.Ks / 2, a.lda / 2, a.ldb / 2, m0, n0, smem, fa, a.sa + (size_t)g0 * a.M,
                     a.wfull ? a.sw : a.sw + (size_t)g0 * a.N, a.wfull);
    if (y == 0 && a.bias != nullptr) {
#pragma unroll
        for (int j = 0; j < C::NT; ++j) {
            const float2 b = *reinterpret_cast<const float2*>(a.bias + col_of<C>(j, 0, n0, 0));
#pragma unroll
            for (int i = 0; i < C::MT; ++i)
#pragma unroll
                for (int q = 0; q < 4; ++q) fa[i][j][q] += (q & 1) ? b.y : b.x;
        }
    }
    store_part<C>(fa, a.out + (size_t)y * a.M * a.N, a.M, a.N, m0, n0);
}
template <class CB, class CS>
DEVI void hetero_f(const ArgsF& a, U* smem) {
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (a.N / CB::BN);
    int b = blockIdx.x;
    if (b < tb) {
        tile_f<CB>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_f<CS>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

// gate|up + SwiGLU with an int8 hidden output (the down projection's fine input): the tile's 128 output columns are
// one group g = o0 / 128: v = (swiglu - mu[c]) * is[c] (null: identity), row amax over the 128 columns (quad shuffles,
// then the 4 warp columns through smem), hq[r, c] = rn(v 127 / amax), hs[g M + r] = amax / 127
struct ArgsG {
    const int8_t* A;
    const int8_t* B;
    const __nv_bfloat16* AH;
    const __nv_bfloat16* BH;
    const float* sa;      // [M]
    const float* sw;      // [N]
    const float* bias;    // [N] or null
    int8_t* hq;           // [M, N / 2]
    float* hs;            // [(N / 2) / 128][M]
    const float* mu;      // [N / 2] or null
    const float* is;      // [N / 2] or null
    int M, N, K, KH, lda, ldb, mb, pad;
};
// ROT epilogue: the tile's 128 SwiGLU outputs per row (centred) go through smem; one warp per row rotates them (4
// columns per lane), takes the row amax and writes the int8 hidden (4 bytes per lane) and hs = amax / 127 / sqrt(128)
template <class C>
DEVI void epi_g_rot(const ArgsG& a, const float (&fa)[C::MT][C::NT][4], int m0, int o0, int H, U* smem) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2, LDS = 132;
    float* S = reinterpret_cast<float*>(smem);   // [BM][LDS] (smem free after the loops)
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
#pragma unroll
            for (int j = 0; j < NH; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const __nv_bfloat16 g = __float2bfloat16_rn(fa[i][j][2 * h + e]);
                    const __nv_bfloat16 u = __float2bfloat16_rn(fa[i][j + NH][2 * h + e]);
                    float x = __bfloat162float(swiglu1(g, u));
                    const int cl = wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
                    if (a.mu != nullptr) x = (x - a.mu[o0 + cl]) * a.is[o0 + cl];
                    S[rl * LDS + cl] = x;
                }
        }
    __syncthreads();
    const int g = o0 / 128;
    for (int rl = warp; rl < C::BM; rl += C::CW) {
        const float4 y4 = *reinterpret_cast<const float4*>(S + rl * LDS + lane * 4);
        float y[4] = {y4.x, y4.y, y4.z, y4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) y[e] = rot_sign(lane * 4 + e, y[e]);
        fwht_reg<4>(y);
        fwht_lanes<4, 16>(y, lane);
        float m = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        m = fmaxf(m, 1e-12f);
        const int r = m0 + rl;
        if (r >= a.M) continue;
        const float inv = 127.f / m;
        if (lane == 0) a.hs[(size_t)g * a.M + r] = m / 127.f * ROT_RS;
        uint32_t q4 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) q4 |= (uint32_t)(__float2int_rn(y[e] * inv) & 0xff) << (8 * e);
        *reinterpret_cast<uint32_t*>(a.hq + (size_t)r * H + o0 + lane * 4) = q4;
    }
}
template <class C, bool ROT>
DEVI void tile_g(const ArgsG& a, int m0, int nt, U* smem) {
    static_assert(C::EPI == 1 && C::BN == 256, "one 128-column group per tile");
    const int H = a.N / 2;
    const int o0 = nt * (C::BN / 2);
    int ia[C::MT][C::NT][4];
    mainloop<C, int, true>(reinterpret_cast<const U*>(a.A), reinterpret_cast<const U*>(a.B), a.M, a.K / 2, a.lda / 2,
                           a.ldb / 2, m0, o0, H + o0, smem, ia);
    float fa[C::MT][C::NT][4];
    dequant<C>(ia, fa, a.sa, a.sw, a.bias, a.M, m0, o0, H);
    if (a.KH > 0)
        mainloop<C, float, false>(reinterpret_cast<const U*>(a.AH), reinterpret_cast<const U*>(a.BH), a.M, a.KH, a.KH,
                                  a.KH, m0, o0, H + o0, smem, fa);
    if constexpr (ROT) {
        epi_g_rot<C>(a, fa, m0, o0, H, smem);
        return;
    }
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2;
    float* red = reinterpret_cast<float*>(smem);   // [BM][WN] row maxima of the warp columns (smem free after the loops)
    float v[C::MT][2][NH][2];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float m = 0.f;
#pragma unroll
            for (int j = 0; j < NH; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const __nv_bfloat16 g = __float2bfloat16_rn(fa[i][j][2 * h + e]);
                    const __nv_bfloat16 u = __float2bfloat16_rn(fa[i][j + NH][2 * h + e]);
                    float x = __bfloat162float(swiglu1(g, u));
                    const int c = o0 + wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
                    if (a.mu != nullptr) x = (x - a.mu[c]) * a.is[c];
                    v[i][h][j][e] = x;
                    m = fmaxf(m, fabsf(x));
                }
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            if (t4 == 0) red[(wm * C::WTM + i * 16 + lane / 4 + h * 8) * C::WN + wn] = m;
        }
    __syncthreads();
    const int g = o0 / 128;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
            const int r = m0 + rl;
            float m = 1e-12f;
#pragma unroll
            for (int w = 0; w < C::WN; ++w) m = fmaxf(m, red[rl * C::WN + w]);
            const float inv = 127.f / m;
            if (r >= a.M) continue;
            if (wn == 0 && t4 == 0) a.hs[(size_t)g * a.M + r] = m / 127.f;
            int8_t* row = a.hq + (size_t)r * H + o0 + wn * (C::WTN / 2) + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const uint32_t q0 = (uint32_t)(__float2int_rn(v[i][h][j][0] * inv) & 0xff);
                const uint32_t q1 = (uint32_t)(__float2int_rn(v[i][h][j][1] * inv) & 0xff);
                *reinterpret_cast<uint16_t*>(row + j * 8) = (uint16_t)(q0 | (q1 << 8));
            }
        }
}
template <class CB, class CS, bool ROT>
DEVI void hetero_g(const ArgsG& a, U* smem) {
    const int H = a.N / 2;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (H / (CB::BN / 2));
    int b = blockIdx.x;
    if (b < tb) {
        tile_g<CB, ROT>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_g<CS, ROT>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}


// ==== E35 (pjs, ours): exact pipeline variants of rgh's Q8 GEMM kernels (q8.cu) / pjs's Q8-WT twins (q8wt.cu) ========
// Every variant below keeps, for every output element, rgh's arithmetic in rgh's order:
//   * int8 products summed in int32 (exact in any order; the same k-tiles, k-steps and mma instructions);
//   * the same fp32 dequant expression fmaf((float)i32, sa * sw, acc) per element / per 128-column group, groups in
//     ascending order from 0.0f (q8f), or fmaf((float)i32, sa * sw, bias) once per tile (q8h / q8g);
//   * the protected bf16 pass as the same m16n8k16 mma sequence (k-tiles and k-steps ascending) onto that fp32 value;
//   * rgh's epilogues (bf16 store, SwiGLU -> int8 + group scale, fp32 partial + bias) unchanged.
// Only the SCHEDULE changes:
//   C1 (q8h / q8g): the int8 main loop and the protected bf16 pass share ONE cp.async ring: the first bf16 k-tiles are
//       issued in the last int8 iterations, so the protected pass starts on resident data (rgh's code ends the int8
//       loop with cp.async.wait_all + barrier and restarts the pipeline from an empty ring for the bf16 pass).

template <class C>
struct KB {   // one k-tile of the shared ring: barrier, look-ahead issue, fragment double-buffering, mma (rgh's order)
    template <class T, class Issue>
    static DEVI void run(T (&acc)[C::MT][C::NT][4], Frag<C>& fr, int& s, bool has_next, uint32_t sA0, uint32_t sB0,
                         Issue&& issue) {
        constexpr int KS = C::KS, S = C::STAGES, MT = C::MT, NT = C::NT;
#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            if (ks == KS - 1) {
                cp_wait<S - 2>();
                __syncthreads();
            }
            if (ks == 0) issue();
            if (ks < KS - 1 || has_next) {
                const int ns = ks < KS - 1 ? s : (s + 1 == S ? 0 : s + 1);
                fr.load((ks + 1) & 1, sA0 + ns * C::A_STAGE * 2, sB0 + ns * C::B_STAGE * 2, ks < KS - 1 ? ks + 1 : 0);
            }
            const int cb = ks & 1;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int ii = 0; ii < MT; ++ii) {
                    const int i = (j & 1) ? MT - 1 - ii : ii;
                    mma<T>(acc[i][j], fr.af[cb][i], fr.bf[cb][j][0], fr.bf[cb][j][1]);
                }
            if (ks == 0) cp_commit();
        }
        if (++s == S) s = 0;
    }
};

// C1: int8 loop over Ku units (A / B, TB = Q8-WT tiled B), dequant (dq(ia, fa)), protected bf16 pass over KHu units
// (AH / BH row-major, ld KHu), all in one ring of STAGES slots. Requires Ku / BK >= STAGES - 1.
template <class C, bool TB, class DQ>
DEVI void mainloop_cat(const U* __restrict__ A, const U* __restrict__ B, const U* __restrict__ AH,
                       const U* __restrict__ BH, int M, int Ku, int KHu, int lda, int ldb, int m0, int n0, int n1,
                       U* smem, float (&fa)[C::MT][C::NT][4], DQ&& dq) {
    constexpr int BK = C::BK, S = C::STAGES, MT = C::MT, NT = C::NT;
    const uint32_t sA0 = smem_u32(smem);
    const uint32_t sB0 = sA0 + S * C::A_STAGE * 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int KT8 = Ku / BK, KTH = KHu / BK, KTT = KT8 + KTH;
    auto sa = [&](int t) { return sA0 + (t % S) * C::A_STAGE * 2; };
    auto sb = [&](int t) { return sB0 + (t % S) * C::B_STAGE * 2; };
    int ia[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) ia[i][j][q] = 0;
    Frag<C> fr;
    int s = 0;
    const int KTa = KT8 - (S - 1);   // int8 tiles whose look-ahead tile is int8
    {
        Loader<C, TB> ld;
        ld.init(A, B, m0, n0, n1, M, lda, ldb);
#pragma unroll
        for (int t = 0; t < S - 1; ++t) {
            ld.issue_all(sa(t), sb(t), t * BK);
            cp_commit();
        }
        fr.init(warp / C::WN, warp % C::WN, lane);
        cp_wait<S - 2>();
        __syncthreads();
        fr.load(0, sa(0), sb(0), 0);
        for (int kt = 0; kt < KTa; ++kt)
            KB<C>::run(ia, fr, s, true, sA0, sB0, [&]() {
                const int tt = kt + S - 1;
                ld.issue_all(sa(tt), sb(tt), tt * BK);
            });
    }
    Loader<C, false> lh;
    lh.init(AH, BH, m0, n0, n1, M, KHu, KHu);
    for (int kt = KTa; kt < KT8; ++kt)
        KB<C>::run(ia, fr, s, kt + 1 < KTT, sA0, sB0, [&]() {
            const int th = kt + S - 1 - KT8;
            if (th < KTH) lh.issue_all(sa(KT8 + th), sb(KT8 + th), th * BK);
        });
    dq(ia, fa);
    for (int kh = 0; kh < KTH; ++kh)
        KB<C>::run(fa, fr, s, KT8 + kh + 1 < KTT, sA0, sB0, [&]() {
            const int th = kh + S - 1;
            if (th < KTH) lh.issue_all(sa(KT8 + th), sb(KT8 + th), th * BK);
        });
    cp_wait<0>();
    __syncthreads();
}

// C1 q8h tile (EPI 2, bf16 out; TB = Q8-WT weights)
template <class C, bool TB>
DEVI void tile_c(const Args& a, int m0, int nt, U* smem) {
    static_assert(C::EPI == 2, "plain");
    const int nbase = nt * C::BN;
    float fa[C::MT][C::NT][4];
    mainloop_cat<C, TB>(reinterpret_cast<const U*>(a.A), reinterpret_cast<const U*>(a.B),
                        reinterpret_cast<const U*>(a.AH), reinterpret_cast<const U*>(a.BH), a.M, a.K / 2, a.KH,
                        a.lda / 2, a.ldb / 2, m0, nbase, -1, smem, fa,
                        [&](const int (&ia)[C::MT][C::NT][4], float (&f)[C::MT][C::NT][4]) {
                            dequant<C>(ia, f, a.sa, a.sw, a.bias, a.M, m0, nbase, 0);
                        });
    store_tile<C>(fa, reinterpret_cast<__nv_bfloat16*>(a.out), a.M, a.ldo, m0, nbase, smem);
}
template <class CB, class CS, bool TB>
DEVI void hetero_c(const Args& a, U* smem) {
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (a.N / CB::BN);
    int b = blockIdx.x;
    if (b < tb) {
        tile_c<CB, TB>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_c<CS, TB>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

// rgh's tile_g epilogue (SwiGLU -> centred int8 hidden + per-128-column group scale), verbatim
template <class C>
DEVI void epi_g(const ArgsG& a, float (&fa)[C::MT][C::NT][4], int m0, int o0, U* smem) {
    const int H = a.N / 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2;
    float* red = reinterpret_cast<float*>(smem);
    float v[C::MT][2][NH][2];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float m = 0.f;
#pragma unroll
            for (int j = 0; j < NH; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const __nv_bfloat16 g = __float2bfloat16_rn(fa[i][j][2 * h + e]);
                    const __nv_bfloat16 u = __float2bfloat16_rn(fa[i][j + NH][2 * h + e]);
                    float x = __bfloat162float(swiglu1(g, u));
                    const int c = o0 + wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
                    if (a.mu != nullptr) x = (x - a.mu[c]) * a.is[c];
                    v[i][h][j][e] = x;
                    m = fmaxf(m, fabsf(x));
                }
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            if (t4 == 0) red[(wm * C::WTM + i * 16 + lane / 4 + h * 8) * C::WN + wn] = m;
        }
    __syncthreads();
    const int g = o0 / 128;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
            const int r = m0 + rl;
            float m = 1e-12f;
#pragma unroll
            for (int w = 0; w < C::WN; ++w) m = fmaxf(m, red[rl * C::WN + w]);
            const float inv = 127.f / m;
            if (r >= a.M) continue;
            if (wn == 0 && t4 == 0) a.hs[(size_t)g * a.M + r] = m / 127.f;
            int8_t* row = a.hq + (size_t)r * H + o0 + wn * (C::WTN / 2) + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const uint32_t q0 = (uint32_t)(__float2int_rn(v[i][h][j][0] * inv) & 0xff);
                const uint32_t q1 = (uint32_t)(__float2int_rn(v[i][h][j][1] * inv) & 0xff);
                *reinterpret_cast<uint16_t*>(row + j * 8) = (uint16_t)(q0 | (q1 << 8));
            }
        }
}
// C1 q8g tile
template <class C, bool TB>
DEVI void tile_gc(const ArgsG& a, int m0, int nt, U* smem) {
    static_assert(C::EPI == 1 && C::BN == 256, "one 128-column group per tile");
    const int H = a.N / 2;
    const int o0 = nt * (C::BN / 2);
    float fa[C::MT][C::NT][4];
    mainloop_cat<C, TB>(reinterpret_cast<const U*>(a.A), reinterpret_cast<const U*>(a.B),
                        reinterpret_cast<const U*>(a.AH), reinterpret_cast<const U*>(a.BH), a.M, a.K / 2, a.KH,
                        a.lda / 2, a.ldb / 2, m0, o0, H + o0, smem, fa,
                        [&](const int (&ia)[C::MT][C::NT][4], float (&f)[C::MT][C::NT][4]) {
                            dequant<C>(ia, f, a.sa, a.sw, a.bias, a.M, m0, o0, H);
                        });
    epi_g<C>(a, fa, m0, o0, smem);
}
template <class CB, class CS, bool TB>
DEVI void hetero_gc(const ArgsG& a, U* smem) {
    const int H = a.N / 2;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (H / (CB::BN / 2));
    int b = blockIdx.x;
    if (b < tb) {
        tile_gc<CB, TB>(a, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_gc<CS, TB>(a, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}


// C1T (q8g): C1 + rgh's SwiGLU epilogue with silu(g) READ from a 65,536-entry table instead of computed: tab[bits(g)] =
// bf16(silu_f32(g)) for every bf16 bit pattern g, built at start-up by q8v_silu_tab below with rgh's own swiglu1
// (swiglu1(g, 1) = bf16(float(bf16(silu(g))) * 1) = bf16(silu(g)) exactly), so swiglu1(g, u) = bf16(float(tab[g]) * float(u))
// is the same value for every input; the rest of rgh's epilogue (centring, row amax, int8 rounding) is unchanged.
template <class C>
DEVI void epi_gt(const ArgsG& a, float (&fa)[C::MT][C::NT][4], int m0, int o0, U* smem, const uint16_t* __restrict__ tab) {
    const int H = a.N / 2;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2;
    float* red = reinterpret_cast<float*>(smem);
    float v[C::MT][2][NH][2];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float m = 0.f;
#pragma unroll
            for (int j = 0; j < NH; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const __nv_bfloat16 g = __float2bfloat16_rn(fa[i][j][2 * h + e]);
                    const __nv_bfloat16 u = __float2bfloat16_rn(fa[i][j + NH][2 * h + e]);
                    const uint16_t sb = __ldg(tab + __bfloat16_as_ushort(g));
                    float x = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__ushort_as_bfloat16(sb)) *
                                                                   __bfloat162float(u)));
                    const int c = o0 + wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
                    if (a.mu != nullptr) x = (x - a.mu[c]) * a.is[c];
                    v[i][h][j][e] = x;
                    m = fmaxf(m, fabsf(x));
                }
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            if (t4 == 0) red[(wm * C::WTM + i * 16 + lane / 4 + h * 8) * C::WN + wn] = m;
        }
    __syncthreads();
    const int g = o0 / 128;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
            const int r = m0 + rl;
            float m = 1e-12f;
#pragma unroll
            for (int w = 0; w < C::WN; ++w) m = fmaxf(m, red[rl * C::WN + w]);
            const float inv = 127.f / m;
            if (r >= a.M) continue;
            if (wn == 0 && t4 == 0) a.hs[(size_t)g * a.M + r] = m / 127.f;
            int8_t* row = a.hq + (size_t)r * H + o0 + wn * (C::WTN / 2) + t4 * 2;
#pragma unroll
            for (int j = 0; j < NH; ++j) {
                const uint32_t q0 = (uint32_t)(__float2int_rn(v[i][h][j][0] * inv) & 0xff);
                const uint32_t q1 = (uint32_t)(__float2int_rn(v[i][h][j][1] * inv) & 0xff);
                *reinterpret_cast<uint16_t*>(row + j * 8) = (uint16_t)(q0 | (q1 << 8));
            }
        }
}
struct ArgsGT {   // ArgsG + the silu table
    ArgsG g;
    const uint16_t* tab;
};
template <class C, bool TB>
DEVI void tile_gct(const ArgsGT& t, int m0, int nt, U* smem) {
    const ArgsG& a = t.g;
    static_assert(C::EPI == 1 && C::BN == 256, "one 128-column group per tile");
    const int H = a.N / 2;
    const int o0 = nt * (C::BN / 2);
    float fa[C::MT][C::NT][4];
    mainloop_cat<C, TB>(reinterpret_cast<const U*>(a.A), reinterpret_cast<const U*>(a.B),
                        reinterpret_cast<const U*>(a.AH), reinterpret_cast<const U*>(a.BH), a.M, a.K / 2, a.KH,
                        a.lda / 2, a.ldb / 2, m0, o0, H + o0, smem, fa,
                        [&](const int (&ia)[C::MT][C::NT][4], float (&f)[C::MT][C::NT][4]) {
                            dequant<C>(ia, f, a.sa, a.sw, a.bias, a.M, m0, o0, H);
                        });
    epi_gt<C>(a, fa, m0, o0, smem, t.tab);
}
template <class CB, class CS, bool TB>
DEVI void hetero_gct(const ArgsGT& t, U* smem) {
    const ArgsG& a = t.g;
    const int H = a.N / 2;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (H / (CB::BN / 2));
    int b = blockIdx.x;
    if (b < tb) {
        tile_gct<CB, TB>(t, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_gct<CS, TB>(t, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}

// ---- ROT (rgh 10-24, j205 / j206): C1 / C1T for q8gr (gate|up + SwiGLU -> rotated int8 hidden) ----
// C1T's ROT epilogue = rgh's epi_g_rot with swiglu1(g, u) -> bf16(float(tab[g]) * float(u)) (the same value, see C1T)
template <class C>
DEVI void epi_g_rot_t(const ArgsG& a, const float (&fa)[C::MT][C::NT][4], int m0, int o0, int H, U* smem,
                       const uint16_t* __restrict__ tab) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int wm = warp / C::WN, wn = warp % C::WN;
    const int t4 = lane % 4;
    constexpr int NH = C::NT / 2, LDS = 132;
    float* S = reinterpret_cast<float*>(smem);   // [BM][LDS] (smem free after the loops)
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rl = wm * C::WTM + i * 16 + lane / 4 + h * 8;
#pragma unroll
            for (int j = 0; j < NH; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const __nv_bfloat16 g = __float2bfloat16_rn(fa[i][j][2 * h + e]);
                    const __nv_bfloat16 u = __float2bfloat16_rn(fa[i][j + NH][2 * h + e]);
                    const uint16_t sb = __ldg(tab + __bfloat16_as_ushort(g));
                    float x = __bfloat162float(__float2bfloat16_rn(__bfloat162float(__ushort_as_bfloat16(sb)) *
                                                                   __bfloat162float(u)));
                    const int cl = wn * (C::WTN / 2) + j * 8 + t4 * 2 + e;
                    if (a.mu != nullptr) x = (x - a.mu[o0 + cl]) * a.is[o0 + cl];
                    S[rl * LDS + cl] = x;
                }
        }
    __syncthreads();
    const int g = o0 / 128;
    for (int rl = warp; rl < C::BM; rl += C::CW) {
        const float4 y4 = *reinterpret_cast<const float4*>(S + rl * LDS + lane * 4);
        float y[4] = {y4.x, y4.y, y4.z, y4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) y[e] = rot_sign(lane * 4 + e, y[e]);
        fwht_reg<4>(y);
        fwht_lanes<4, 16>(y, lane);
        float m = fmaxf(fmaxf(fabsf(y[0]), fabsf(y[1])), fmaxf(fabsf(y[2]), fabsf(y[3])));
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        m = fmaxf(m, 1e-12f);
        const int r = m0 + rl;
        if (r >= a.M) continue;
        const float inv = 127.f / m;
        if (lane == 0) a.hs[(size_t)g * a.M + r] = m / 127.f * ROT_RS;
        uint32_t q4 = 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) q4 |= (uint32_t)(__float2int_rn(y[e] * inv) & 0xff) << (8 * e);
        *reinterpret_cast<uint32_t*>(a.hq + (size_t)r * H + o0 + lane * 4) = q4;
    }
}
template <class C, bool TB, bool TAB>
DEVI void tile_grc(const ArgsGT& t, int m0, int nt, U* smem) {
    const ArgsG& a = t.g;
    static_assert(C::EPI == 1 && C::BN == 256, "one 128-column group per tile");
    const int H = a.N / 2;
    const int o0 = nt * (C::BN / 2);
    float fa[C::MT][C::NT][4];
    mainloop_cat<C, TB>(reinterpret_cast<const U*>(a.A), reinterpret_cast<const U*>(a.B),
                        reinterpret_cast<const U*>(a.AH), reinterpret_cast<const U*>(a.BH), a.M, a.K / 2, a.KH,
                        a.lda / 2, a.ldb / 2, m0, o0, H + o0, smem, fa,
                        [&](const int (&ia)[C::MT][C::NT][4], float (&f)[C::MT][C::NT][4]) {
                            dequant<C>(ia, f, a.sa, a.sw, a.bias, a.M, m0, o0, H);
                        });
    if constexpr (TAB) epi_g_rot_t<C>(a, fa, m0, o0, H, smem, t.tab);
    else epi_g_rot<C>(a, fa, m0, o0, H, smem);
}
template <class CB, class CS, bool TB, bool TAB>
DEVI void hetero_grc(const ArgsGT& t, U* smem) {
    const ArgsG& a = t.g;
    const int H = a.N / 2;
    const int mtb = a.mb / CB::BM;
    const int tb = mtb * (H / (CB::BN / 2));
    int b = blockIdx.x;
    if (b < tb) {
        tile_grc<CB, TB, TAB>(t, (b % mtb) * CB::BM, b / mtb, smem);
    } else {
        b -= tb;
        const int mts = (a.M - a.mb + CS::BM - 1) / CS::BM;
        tile_grc<CS, TB, TAB>(t, a.mb + (b % mts) * CS::BM, b / mts, smem);
    }
}
}  // namespace q8

// the silu table: tab[p] = bits of rgh's swiglu1(bf16 with bits p, 1.0) = bf16(silu_f32(g)), p = 0 .. 65535
extern "C" __global__ void __launch_bounds__(256) q8v_silu_tab(uint16_t* __restrict__ tab) {
    const int p = blockIdx.x * 256 + threadIdx.x;
    if (p >= 65536) return;
    const __nv_bfloat16 g = __ushort_as_bfloat16((unsigned short)p);
    tab[p] = __bfloat16_as_ushort(q8::swiglu1(g, __float2bfloat16_rn(1.0f)));
}
namespace q8 {
}  // namespace q8

#define V_KERNEL(NAME, ARGS, FN)                                                                                   \
    extern "C" __global__ void __launch_bounds__(256, 1) NAME(q8::ARGS a) {                                        \
        extern __shared__ __align__(128) unsigned char smem_raw[];                                                  \
        q8::FN(a, reinterpret_cast<q8::U*>(smem_raw));                                                             \
    }

// (BM, BN, BK units, warps M, warps N, stages, epilogue) = rgh's configurations
using P_B128x256 = q8::Cfg<128, 256, 64, 2, 4, 2, 2>;
using P_S64x256 = q8::Cfg<64, 256, 64, 2, 4, 2, 2>;
using P_S32x256 = q8::Cfg<32, 256, 64, 1, 8, 2, 2>;
using W_B128x256 = q8::Cfg<128, 256, 64, 2, 4, 2, 1>;
using W_S64x256 = q8::Cfg<64, 256, 64, 2, 4, 2, 1>;
using W_S32x256 = q8::Cfg<32, 256, 64, 1, 8, 2, 1>;

#define COMMA ,
// C1 (q8h): rgh's raw weights / Q8-WT tiled weights (the e33g ship path)
V_KERNEL(q8h_b128x256_s64x256_c1, Args, hetero_c<P_B128x256 COMMA P_S64x256 COMMA false>)
V_KERNEL(q8h_b128x256_s32x256_c1, Args, hetero_c<P_B128x256 COMMA P_S32x256 COMMA false>)
V_KERNEL(q8h_b128x256_s64x256_c1wt, Args, hetero_c<P_B128x256 COMMA P_S64x256 COMMA true>)
V_KERNEL(q8h_b128x256_s32x256_c1wt, Args, hetero_c<P_B128x256 COMMA P_S32x256 COMMA true>)
// C1 (q8g)
V_KERNEL(q8g_b128x256_s64x256_c1, ArgsG, hetero_gc<W_B128x256 COMMA W_S64x256 COMMA false>)
V_KERNEL(q8g_b128x256_s32x256_c1, ArgsG, hetero_gc<W_B128x256 COMMA W_S32x256 COMMA false>)
V_KERNEL(q8g_b128x256_s64x256_c1wt, ArgsG, hetero_gc<W_B128x256 COMMA W_S64x256 COMMA true>)
V_KERNEL(q8g_b128x256_s32x256_c1wt, ArgsG, hetero_gc<W_B128x256 COMMA W_S32x256 COMMA true>)

// C1T (q8g)
V_KERNEL(q8g_b128x256_s64x256_c1t, ArgsGT, hetero_gct<W_B128x256 COMMA W_S64x256 COMMA false>)
V_KERNEL(q8g_b128x256_s32x256_c1t, ArgsGT, hetero_gct<W_B128x256 COMMA W_S32x256 COMMA false>)
// C1 / C1T (q8gr, ROT); both take ArgsGT (C1 ignores the table)
V_KERNEL(q8gr_b128x256_s64x256_c1, ArgsGT, hetero_grc<W_B128x256 COMMA W_S64x256 COMMA false COMMA false>)
V_KERNEL(q8gr_b128x256_s32x256_c1, ArgsGT, hetero_grc<W_B128x256 COMMA W_S32x256 COMMA false COMMA false>)
V_KERNEL(q8gr_b128x256_s64x256_c1t, ArgsGT, hetero_grc<W_B128x256 COMMA W_S64x256 COMMA false COMMA true>)
V_KERNEL(q8gr_b128x256_s32x256_c1t, ArgsGT, hetero_grc<W_B128x256 COMMA W_S32x256 COMMA false COMMA true>)
