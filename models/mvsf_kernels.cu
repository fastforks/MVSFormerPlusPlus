// Fused MVSFormer++ cost volume (exp 011, max path). Batch 1, fp32 math (as upstream's autocast-disabled block).
//   feat  [V, C, H, W] channels-last (memory [V, H, W, C]), bf16 or fp32; view 0 = reference
//   rel   [S, 12] fp32: src-from-ref 3x4 (rows of R | t), as homo_warping_3D_with_mask
//   dv    [D, HW] fp32 depth hypotheses
// entropy:   ent [S, HW]     = entropy(softmax_d(sum_g groupcorr_g(d)))           (StageNet vis-net input)
// aggregate: out [G, D, HW]  = sum_s vis_s * groupcorr(s, d) / (sum_s vis_s + 1e-6) (StageNet volume_mean)
// groupcorr_g = mean over the C/G channels of group g of ref * bilinear(src) (grid_sample zeros, align_corners=True).
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>

template <typename T> __device__ __forceinline__ float tof(T v);
template <> __device__ __forceinline__ float tof<float>(float v) { return v; }
template <> __device__ __forceinline__ float tof<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T, int C>
__device__ __forceinline__ void load_vec(const T* __restrict__ p, float* out) {
#pragma unroll
    for (int c = 0; c < C; c++) out[c] = tof<T>(p[c]);
}

// projected sample position (grid_sample unnormalized, align_corners=True) of ref pixel (x, y) at depth z
__device__ __forceinline__ void project(const float* __restrict__ r, float x, float y, float z, int H, int W,
                                        float& ix, float& iy) {
    float px = (r[0] * x + r[1] * y + r[2]) * z + r[3];
    float py = (r[4] * x + r[5] * y + r[6]) * z + r[7];
    float pz = (r[8] * x + r[9] * y + r[10]) * z + r[11];
    float gx = (px / (pz + 1e-6f)) / ((W - 1) / 2.f) - 1.f;
    float gy = (py / (pz + 1e-6f)) / ((H - 1) / 2.f) - 1.f;
    ix = (gx + 1.f) / 2.f * (W - 1);
    iy = (gy + 1.f) / 2.f * (H - 1);
}

// out[g] = sum_k w_k * sum_{c in g} ref[c] * src_k[c]   (bilinear corners, zero padding)
template <typename T, int C, int G>
__device__ __forceinline__ void corr(const T* __restrict__ src, const float* ref, float ix, float iy, int H, int W,
                                     float* out) {
#pragma unroll
    for (int g = 0; g < G; g++) out[g] = 0.f;
    // All four bilinear corners lie outside the image unless -1 < ix < W and -1 < iy < H (zero padding), so such
    // samples contribute nothing. Rejecting them here (also NaN / inf) keeps the float -> int conversion below in
    // range: an out-of-range (int)floorf() is undefined and produced out-of-bounds addresses for far-off projections.
    if (!(ix > -1.f && ix < (float)W && iy > -1.f && iy < (float)H)) return;
    float fx = floorf(ix), fy = floorf(iy);
    int x0 = (int)fx, y0 = (int)fy;
    float ax = ix - fx, ay = iy - fy;
    float wts[4] = {(1.f - ax) * (1.f - ay), ax * (1.f - ay), (1.f - ax) * ay, ax * ay};
#pragma unroll
    for (int k = 0; k < 4; k++) {
        int xx = x0 + (k & 1), yy = y0 + (k >> 1);
        if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        const T* p = src + ((size_t)yy * W + xx) * C;
        float acc[G];
#pragma unroll
        for (int g = 0; g < G; g++) acc[g] = 0.f;
#pragma unroll
        for (int c = 0; c < C; c++) acc[c / (C / G)] = fmaf(ref[c], tof<T>(p[c]), acc[c / (C / G)]);
#pragma unroll
        for (int g = 0; g < G; g++) out[g] = fmaf(wts[k], acc[g], out[g]);
    }
}

template <typename T, int C, int G, int D>
__global__ void entropy_k(const T* __restrict__ feat, const float* __restrict__ rel, const float* __restrict__ dv,
                          float* __restrict__ ent, int H, int W) {
    int HW = H * W;
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (pix >= HW) return;
    float ref[C];
    load_vec<T, C>(feat + (size_t)pix * C, ref);
    const T* src = feat + (size_t)(s + 1) * HW * C;
    float r[12];
#pragma unroll
    for (int i = 0; i < 12; i++) r[i] = rel[s * 12 + i];
    float x = pix % W, y = pix / W;
    float sim[D];
    float m = -INFINITY;
#pragma unroll
    for (int d = 0; d < D; d++) {
        float ix, iy, ip[G];
        project(r, x, y, dv[(size_t)d * HW + pix], H, W, ix, iy);
        corr<T, C, G>(src, ref, ix, iy, H, W, ip);
        float sv = 0.f;
#pragma unroll
        for (int g = 0; g < G; g++) sv += ip[g] / (C / G);
        sim[d] = sv;
        m = fmaxf(m, sv);
    }
    float sum = 0.f;
#pragma unroll
    for (int d = 0; d < D; d++) {
        sim[d] = expf(sim[d] - m);
        sum += sim[d];
    }
    float e = 0.f;
#pragma unroll
    for (int d = 0; d < D; d++) {
        float p = sim[d] / sum;
        e -= p * logf(p + 1e-7f);
    }
    ent[(size_t)s * HW + pix] = e;
}

template <typename T, int C, int G>
__global__ void aggregate_k(const T* __restrict__ feat, const float* __restrict__ rel, const float* __restrict__ dv,
                            const float* __restrict__ vis, float* __restrict__ out, int S, int D, int H, int W) {
    int HW = H * W;
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y;
    if (pix >= HW) return;
    float ref[C];
    load_vec<T, C>(feat + (size_t)pix * C, ref);
    float x = pix % W, y = pix / W;
    float z = dv[(size_t)d * HW + pix];
    float acc[G], wsum = 0.f;
#pragma unroll
    for (int g = 0; g < G; g++) acc[g] = 0.f;
    for (int s = 0; s < S; s++) {
        float r[12];
#pragma unroll
        for (int i = 0; i < 12; i++) r[i] = rel[s * 12 + i];
        float ix, iy, ip[G];
        project(r, x, y, z, H, W, ix, iy);
        corr<T, C, G>(feat + (size_t)(s + 1) * HW * C, ref, ix, iy, H, W, ip);
        float v = vis[(size_t)s * HW + pix];
#pragma unroll
        for (int g = 0; g < G; g++) acc[g] = fmaf(ip[g] / (C / G), v, acc[g]);
        wsum += v;
    }
    float inv = 1.f / (wsum + 1e-6f);
#pragma unroll
    for (int g = 0; g < G; g++) out[((size_t)g * D + d) * HW + pix] = acc[g] * inv;
}

#define DISPATCH_T(feat, ...)                                                        \
    if (feat.scalar_type() == at::kBFloat16) {                                       \
        using T = __nv_bfloat16;                                                     \
        __VA_ARGS__                                                                  \
    } else {                                                                         \
        TORCH_CHECK(feat.scalar_type() == at::kFloat, "feat must be bf16 or fp32");  \
        using T = float;                                                             \
        __VA_ARGS__                                                                  \
    }

static void check(const torch::Tensor& feat, const torch::Tensor& rel, const torch::Tensor& dv, int64_t G) {
    TORCH_CHECK(feat.is_cuda() && feat.dim() == 4 && feat.is_contiguous(at::MemoryFormat::ChannelsLast),
                "feat: [V,C,H,W] channels-last");
    TORCH_CHECK(rel.scalar_type() == at::kFloat && rel.is_contiguous() && rel.size(1) == 12, "rel [S,12] fp32");
    TORCH_CHECK(dv.scalar_type() == at::kFloat && dv.is_contiguous(), "dv [D,HW] fp32");
    TORCH_CHECK(G == 8, "G == 8 only");
}

torch::Tensor entropy(torch::Tensor feat, torch::Tensor rel, torch::Tensor dv, int64_t G) {
    check(feat, rel, dv, G);
    int V = feat.size(0), C = feat.size(1), H = feat.size(2), W = feat.size(3), D = dv.size(0), S = V - 1;
    auto ent = torch::empty({S, 1, H, W}, feat.options().dtype(at::kFloat));
    dim3 grid((H * W + 127) / 128, S);
    auto st = at::cuda::getCurrentCUDAStream();
#define ENT(CC, DD)                                                                                      \
    if (C == CC && D == DD) {                                                                            \
        DISPATCH_T(feat, entropy_k<T, CC, 8, DD><<<grid, 128, 0, st>>>(                                  \
            (const T*)feat.data_ptr(), rel.data_ptr<float>(), dv.data_ptr<float>(), ent.data_ptr<float>(), H, W);) \
        return ent;                                                                                      \
    }
    ENT(64, 32) ENT(32, 16) ENT(16, 8) ENT(8, 4)
    TORCH_CHECK(false, "unsupported (C, D)");
}

torch::Tensor aggregate(torch::Tensor feat, torch::Tensor rel, torch::Tensor dv, torch::Tensor vis, int64_t G) {
    check(feat, rel, dv, G);
    int V = feat.size(0), C = feat.size(1), H = feat.size(2), W = feat.size(3), D = dv.size(0), S = V - 1;
    TORCH_CHECK(vis.scalar_type() == at::kFloat && vis.is_contiguous(), "vis [S,HW] fp32");
    auto out = torch::empty({1, G, D, H, W}, feat.options().dtype(at::kFloat));
    dim3 grid((H * W + 127) / 128, D);
    auto st = at::cuda::getCurrentCUDAStream();
#define AGG(CC)                                                                                          \
    if (C == CC) {                                                                                       \
        DISPATCH_T(feat, aggregate_k<T, CC, 8><<<grid, 128, 0, st>>>(                                    \
            (const T*)feat.data_ptr(), rel.data_ptr<float>(), dv.data_ptr<float>(), vis.data_ptr<float>(), \
            out.data_ptr<float>(), S, D, H, W);)                                                         \
        return out;                                                                                      \
    }
    AGG(64) AGG(32) AGG(16) AGG(8)
    TORCH_CHECK(false, "unsupported C");
}


// ---------------------------------------------------------------- small-channel direct conv (FPN encoder, max path)
// x [N,CIN,H,W] channels-last (bf16 or fp32), w [K][K][CIN][COUT] fp32 (BN folded), b [COUT]; leaky ReLU 0.1; y bf16 CL.
template <typename TI, int CIN, int COUT, int K, int S>
__global__ void __launch_bounds__(256) conv_small_k(const TI* __restrict__ x, const float* __restrict__ w,
                                                    const float* __restrict__ b, __nv_bfloat16* __restrict__ y, int H,
                                                    int W, int Ho, int Wo, int pad) {
    constexpr int T = 16, TIN = (T - 1) * S + K;
    extern __shared__ float sm[];
    float* sw = sm;                        // K*K*CIN*COUT
    float* st = sm + K * K * CIN * COUT;   // TIN*TIN*CIN
    int n = blockIdx.z, tid = threadIdx.y * T + threadIdx.x;
    for (int i = tid; i < K * K * CIN * COUT; i += T * T) sw[i] = w[i];
    int oy0 = blockIdx.y * T, ox0 = blockIdx.x * T;
    int iy0 = oy0 * S - pad, ix0 = ox0 * S - pad;
    const TI* xn = x + (size_t)n * H * W * CIN;
    for (int i = tid; i < TIN * TIN; i += T * T) {
        int yy = iy0 + i / TIN, xx = ix0 + i % TIN;
        bool in = yy >= 0 && yy < H && xx >= 0 && xx < W;
#pragma unroll
        for (int c = 0; c < CIN; c++) st[i * CIN + c] = in ? tof<TI>(xn[((size_t)yy * W + xx) * CIN + c]) : 0.f;
    }
    __syncthreads();
    int oy = oy0 + threadIdx.y, ox = ox0 + threadIdx.x;
    if (oy >= Ho || ox >= Wo) return;
    float acc[COUT];
#pragma unroll
    for (int o = 0; o < COUT; o++) acc[o] = b[o];
    for (int ky = 0; ky < K; ky++) {
#pragma unroll
        for (int kx = 0; kx < K; kx++) {
            const float* v = st + ((threadIdx.y * S + ky) * TIN + threadIdx.x * S + kx) * CIN;
            const float* wk = sw + (ky * K + kx) * CIN * COUT;
#pragma unroll
            for (int c = 0; c < CIN; c++) {
                float vc = v[c];
#pragma unroll
                for (int o = 0; o < COUT; o++) acc[o] = fmaf(wk[c * COUT + o], vc, acc[o]);
            }
        }
    }
    __nv_bfloat16* yp = y + (((size_t)n * Ho + oy) * Wo + ox) * COUT;
#pragma unroll
    for (int o = 0; o < COUT; o++) yp[o] = __float2bfloat16(acc[o] > 0.f ? acc[o] : 0.1f * acc[o]);
}

torch::Tensor conv_small(torch::Tensor x, torch::Tensor w, torch::Tensor b, int64_t stride, int64_t pad) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 4 && x.is_contiguous(at::MemoryFormat::ChannelsLast), "x CL");
    TORCH_CHECK(w.scalar_type() == at::kFloat && w.is_contiguous() && w.dim() == 4, "w [K,K,CIN,COUT] fp32");
    int N = x.size(0), C = x.size(1), H = x.size(2), W = x.size(3), K = w.size(0), CO = w.size(3), S = stride;
    int Ho = (H + 2 * pad - K) / S + 1, Wo = (W + 2 * pad - K) / S + 1;
    auto y = torch::empty({N, CO, Ho, Wo}, x.options().dtype(at::kBFloat16).memory_format(at::MemoryFormat::ChannelsLast));
    dim3 grid((Wo + 15) / 16, (Ho + 15) / 16, N), block(16, 16);
    auto st = at::cuda::getCurrentCUDAStream();
#define CS(CI_, CO_, K_, S_)                                                                                     \
    if (C == CI_ && CO == CO_ && K == K_ && S == S_) {                                                           \
        constexpr int TIN = 15 * S_ + K_;                                                                        \
        size_t smem = (K_ * K_ * CI_ * CO_ + TIN * TIN * CI_) * sizeof(float);                                   \
        DISPATCH_T(x, {                                                                                          \
            auto kern = conv_small_k<T, CI_, CO_, K_, S_>;                                                       \
            cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);                  \
            kern<<<grid, block, smem, st>>>((const T*)x.data_ptr(), w.data_ptr<float>(), b.data_ptr<float>(),    \
                                            (__nv_bfloat16*)y.data_ptr(), H, W, Ho, Wo, (int)pad);               \
        })                                                                                                       \
        return y;                                                                                                \
    }
    CS(3, 8, 7, 1) CS(8, 8, 5, 1) CS(8, 16, 5, 2) CS(16, 16, 3, 1)
    TORCH_CHECK(false, "conv_small: unsupported shape");
}

// ---------------------------------------------------------------- FPN decoder last level, fused (max path)
// out = swish(b3 + sum_k inb(p+k) [ up(U_k)(p+k) + Wc_k c(p+k) + bk_k ]),  k over the 3x3 taps, where
//   U = W3_k t (1x1 at half res, [N, 9*CO, h, w]), up = bilinear x2 align_corners=True, Wc_k = W3_k Wi, bk_k = W3_k bi:
//   = swish(conv3x3_W3(up(t) + conv1x1_Wi(c)) + b3) of upstream, without the [N,64,H,W] intermediates.
// prm = Wc [9][CI][CO] | bk [9][CO] | b3 [CO] (fp32)
template <int CO, int CI>
__global__ void fpn_last_k(const __nv_bfloat16* __restrict__ U, const __nv_bfloat16* __restrict__ c,
                           const float* __restrict__ prm, __nv_bfloat16* __restrict__ y, int H, int W, int h, int w) {
    __shared__ float sp[9 * CI * CO + 9 * CO + CO];
    for (int i = threadIdx.x; i < 9 * CI * CO + 9 * CO + CO; i += blockDim.x) sp[i] = prm[i];
    __syncthreads();
    int n = blockIdx.y;
    int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H * W) return;
    int py = pix / W, px = pix % W;
    const float* Wc = sp;
    const float* bk = sp + 9 * CI * CO;
    const float* b3 = bk + 9 * CO;
    float sy_scale = (float)(h - 1) / (float)(H - 1), sx_scale = (float)(w - 1) / (float)(W - 1);
    const __nv_bfloat16* Un = U + (size_t)n * h * w * 9 * CO;
    const __nv_bfloat16* cn = c + (size_t)n * H * W * CI;
    float acc[CO];
#pragma unroll
    for (int o = 0; o < CO; o++) acc[o] = b3[o];
#pragma unroll
    for (int k = 0; k < 9; k++) {
        int ny = py + k / 3 - 1, nx = px + k % 3 - 1;
        if (ny < 0 || nx < 0 || ny >= H || nx >= W) continue;
        float sy = sy_scale * ny, sx = sx_scale * nx;
        int y0 = (int)sy, x0 = (int)sx;
        int y1 = y0 + (y0 < h - 1), x1 = x0 + (x0 < w - 1);
        float ly = sy - y0, lx = sx - x0;
        float w00 = (1.f - ly) * (1.f - lx), w01 = (1.f - ly) * lx, w10 = ly * (1.f - lx), w11 = ly * lx;
        const __nv_bfloat16* u00 = Un + ((size_t)y0 * w + x0) * 9 * CO + k * CO;
        const __nv_bfloat16* u01 = Un + ((size_t)y0 * w + x1) * 9 * CO + k * CO;
        const __nv_bfloat16* u10 = Un + ((size_t)y1 * w + x0) * 9 * CO + k * CO;
        const __nv_bfloat16* u11 = Un + ((size_t)y1 * w + x1) * 9 * CO + k * CO;
        const __nv_bfloat16* cp = cn + ((size_t)ny * W + nx) * CI;
        float cv[CI];
#pragma unroll
        for (int i = 0; i < CI; i++) cv[i] = __bfloat162float(cp[i]);
#pragma unroll
        for (int o = 0; o < CO; o++) {
            float v = w00 * __bfloat162float(u00[o]) + w01 * __bfloat162float(u01[o]) + w10 * __bfloat162float(u10[o]) +
                      w11 * __bfloat162float(u11[o]) + bk[k * CO + o];
#pragma unroll
            for (int i = 0; i < CI; i++) v = fmaf(Wc[(k * CI + i) * CO + o], cv[i], v);
            acc[o] += v;
        }
    }
    __nv_bfloat16* yp = y + ((size_t)n * H * W + pix) * CO;
#pragma unroll
    for (int o = 0; o < CO; o++) yp[o] = __float2bfloat16(acc[o] / (1.f + __expf(-acc[o])));
}

torch::Tensor fpn_last(torch::Tensor U, torch::Tensor c, torch::Tensor prm) {
    TORCH_CHECK(U.scalar_type() == at::kBFloat16 && U.is_contiguous(at::MemoryFormat::ChannelsLast), "U bf16 CL");
    TORCH_CHECK(c.scalar_type() == at::kBFloat16 && c.is_contiguous(at::MemoryFormat::ChannelsLast), "c bf16 CL");
    int N = c.size(0), CI = c.size(1), H = c.size(2), W = c.size(3), CO = U.size(1) / 9, h = U.size(2), w = U.size(3);
    TORCH_CHECK(H == 2 * h && W == 2 * w, "x2");
    auto y = torch::empty({N, CO, H, W}, c.options().memory_format(at::MemoryFormat::ChannelsLast));
    dim3 grid((H * W + 255) / 256, N);
    auto st = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(CO == 8 && CI == 8, "fpn_last: CO = CI = 8 only");
    fpn_last_k<8, 8><<<grid, 256, 0, st>>>((const __nv_bfloat16*)U.data_ptr(), (const __nv_bfloat16*)c.data_ptr(),
                                           prm.data_ptr<float>(), (__nv_bfloat16*)y.data_ptr(), H, W, h, w);
    return y;
}

// ---------------------------------------------------------------- flash attention, head_dim 16 (stage-1 cost transformer)
// q, k, v: element (b, n, h, d) at ptr[b*sb + n*sn + h*16 + d] fp16 (views of the packed [B, N, 3, H, 16] qkv); o [B, N, H, 16].
// FlashAttention-2 schedule: CTA = 4 warps x 32 query rows, 64-key K/V tiles double-buffered with cp.async; S = Q K^T in fp32
// (one m16n8k16 per 8 keys), online softmax in fp32 (base 2), P V per 64-key block with fp16 (PV16) or fp32 accumulation.
// smem rows padded to 24 halves (48 B): conflict-free ldmatrix without swizzling.
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait0() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void mma_f16(uint32_t* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f32(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_h2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float2 unpack_h2(uint32_t u) { return __half22float2(*reinterpret_cast<__half2*>(&u)); }
__device__ __forceinline__ float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

template <bool PV16>
__global__ void __launch_bounds__(128) flash16_k(const __half* __restrict__ q, const __half* __restrict__ k,
                                                 const __half* __restrict__ v, __half* __restrict__ o, int N, int H,
                                                 long long qsb, int qsn, long long ksb, int ksn, long long vsb, int vsn,
                                                 long long osb, int osn, float scale_log2) {
    constexpr int RM = 2, BR = 64 * RM, BC = 64, LD = 24;
    __shared__ __align__(128) __half sQ[BR * LD];
    __shared__ __align__(128) __half sK[2][BC * LD];
    __shared__ __align__(128) __half sV[2][BC * LD];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int bh = blockIdx.y, b = bh / H, h = bh - b * H;
    const int q0 = blockIdx.x * BR;
    const __half* qb = q + b * qsb + h * 16;
    const __half* kb = k + b * ksb + h * 16;
    const __half* vb = v + b * vsb + h * 16;
#pragma unroll
    for (int i = 0; i < BR * 2 / 128; ++i) {
        int c = tid + i * 128, row = c >> 1, ch = c & 1, n = q0 + row;
        cp_async16(smem_u32(&sQ[row * LD + ch * 8]), qb + (long long)min(n, N - 1) * qsn + ch * 8, n < N);
    }
    auto load_kv = [&](int buf, int k0) {
        int row = tid >> 1, ch = tid & 1, n = k0 + row;  // 64 rows x 2 chunks = 128 threads
        int nn = min(n, N - 1);
        cp_async16(smem_u32(&sK[buf][row * LD + ch * 8]), kb + (long long)nn * ksn + ch * 8, n < N);
        cp_async16(smem_u32(&sV[buf][row * LD + ch * 8]), vb + (long long)nn * vsn + ch * 8, n < N);
    };
    load_kv(0, 0);
    cp_async_commit();
    float m_r[RM][2], l_r[RM][2], oacc[RM][2][4];
#pragma unroll
    for (int mi = 0; mi < RM; ++mi) {
        m_r[mi][0] = m_r[mi][1] = -INFINITY;
        l_r[mi][0] = l_r[mi][1] = 0.f;
#pragma unroll
        for (int u = 0; u < 2; ++u) oacc[mi][u][0] = oacc[mi][u][1] = oacc[mi][u][2] = oacc[mi][u][3] = 0.f;
    }
    uint32_t qf[RM][4];
    const int nblk = (N + BC - 1) / BC;
    for (int j = 0; j < nblk; ++j) {
        cp_async_wait0();
        __syncthreads();
        if (j == 0) {
#pragma unroll
            for (int mi = 0; mi < RM; ++mi) {
                int row = warp * 16 * RM + mi * 16 + (lane & 15);
                ldsm_x4(smem_u32(&sQ[row * LD + (lane >> 4) * 8]), qf[mi][0], qf[mi][1], qf[mi][2], qf[mi][3]);
            }
        }
        if (j + 1 < nblk) load_kv((j + 1) & 1, (j + 1) * BC);
        cp_async_commit();
        const __half* K_ = sK[j & 1];
        const __half* V_ = sV[j & 1];
        float s[RM][8][4];
#pragma unroll
        for (int mi = 0; mi < RM; ++mi)
#pragma unroll
            for (int t = 0; t < 8; ++t) s[mi][t][0] = s[mi][t][1] = s[mi][t][2] = s[mi][t][3] = 0.f;
#pragma unroll
        for (int t = 0; t < 8; t += 2) {
            uint32_t b0, b1, b2, b3;
            int row = (t + (lane >> 4)) * 8 + (lane & 7);
            ldsm_x4(smem_u32(&K_[row * LD + ((lane >> 3) & 1) * 8]), b0, b1, b2, b3);
#pragma unroll
            for (int mi = 0; mi < RM; ++mi) {
                mma_f32(s[mi][t], qf[mi], b0, b1);
                mma_f32(s[mi][t + 1], qf[mi], b2, b3);
            }
        }
        if ((j + 1) * BC > N) {
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                int key = j * BC + t * 8 + (lane & 3) * 2;
#pragma unroll
                for (int mi = 0; mi < RM; ++mi) {
                    if (key >= N) s[mi][t][0] = s[mi][t][2] = -INFINITY;
                    if (key + 1 >= N) s[mi][t][1] = s[mi][t][3] = -INFINITY;
                }
            }
        }
        uint32_t pf[RM][4][4];
#pragma unroll
        for (int mi = 0; mi < RM; ++mi) {
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                mx0 = fmaxf(mx0, fmaxf(s[mi][t][0], s[mi][t][1]));
                mx1 = fmaxf(mx1, fmaxf(s[mi][t][2], s[mi][t][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
            float mn0 = fmaxf(m_r[mi][0], mx0 * scale_log2), mn1 = fmaxf(m_r[mi][1], mx1 * scale_log2);
            float c0 = fast_exp2(m_r[mi][0] - mn0), c1 = fast_exp2(m_r[mi][1] - mn1);
            m_r[mi][0] = mn0;
            m_r[mi][1] = mn1;
            float sum0 = 0.f, sum1 = 0.f;
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                float p0 = fast_exp2(fmaf(s[mi][t][0], scale_log2, -mn0));
                float p1 = fast_exp2(fmaf(s[mi][t][1], scale_log2, -mn0));
                float p2 = fast_exp2(fmaf(s[mi][t][2], scale_log2, -mn1));
                float p3 = fast_exp2(fmaf(s[mi][t][3], scale_log2, -mn1));
                sum0 += p0 + p1;
                sum1 += p2 + p3;
                pf[mi][t >> 1][(t & 1) * 2 + 0] = pack_h2(p0, p1);
                pf[mi][t >> 1][(t & 1) * 2 + 1] = pack_h2(p2, p3);
            }
            l_r[mi][0] = l_r[mi][0] * c0 + sum0;
            l_r[mi][1] = l_r[mi][1] * c1 + sum1;
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                oacc[mi][u][0] *= c0; oacc[mi][u][1] *= c0;
                oacc[mi][u][2] *= c1; oacc[mi][u][3] *= c1;
            }
        }
        if (PV16) {
            uint32_t ot[RM][2][2];
#pragma unroll
            for (int mi = 0; mi < RM; ++mi) ot[mi][0][0] = ot[mi][0][1] = ot[mi][1][0] = ot[mi][1][1] = 0u;
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) {
                uint32_t b0, b1, b2, b3;
                int row = kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ldsm_x4_t(smem_u32(&V_[row * LD + (lane >> 4) * 8]), b0, b1, b2, b3);
#pragma unroll
                for (int mi = 0; mi < RM; ++mi) {
                    mma_f16(ot[mi][0], pf[mi][kc], b0, b1);
                    mma_f16(ot[mi][1], pf[mi][kc], b2, b3);
                }
            }
#pragma unroll
            for (int mi = 0; mi < RM; ++mi)
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    float2 a = unpack_h2(ot[mi][u][0]), c = unpack_h2(ot[mi][u][1]);
                    oacc[mi][u][0] += a.x; oacc[mi][u][1] += a.y; oacc[mi][u][2] += c.x; oacc[mi][u][3] += c.y;
                }
        } else {
#pragma unroll
            for (int kc = 0; kc < 4; ++kc) {
                uint32_t b0, b1, b2, b3;
                int row = kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                ldsm_x4_t(smem_u32(&V_[row * LD + (lane >> 4) * 8]), b0, b1, b2, b3);
#pragma unroll
                for (int mi = 0; mi < RM; ++mi) {
                    mma_f32(oacc[mi][0], pf[mi][kc], b0, b1);
                    mma_f32(oacc[mi][1], pf[mi][kc], b2, b3);
                }
            }
        }
    }
    __half* ob = o + b * osb + h * 16;
#pragma unroll
    for (int mi = 0; mi < RM; ++mi) {
        float l0 = l_r[mi][0], l1 = l_r[mi][1];
        l0 += __shfl_xor_sync(0xffffffff, l0, 1);
        l0 += __shfl_xor_sync(0xffffffff, l0, 2);
        l1 += __shfl_xor_sync(0xffffffff, l1, 1);
        l1 += __shfl_xor_sync(0xffffffff, l1, 2);
        float i0 = 1.f / l0, i1 = 1.f / l1;
        int r0 = q0 + warp * 16 * RM + mi * 16 + (lane >> 2);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            int col = u * 8 + (lane & 3) * 2;
            if (r0 < N) *reinterpret_cast<uint32_t*>(ob + (long long)r0 * osn + col) = pack_h2(oacc[mi][u][0] * i0, oacc[mi][u][1] * i0);
            if (r0 + 8 < N) *reinterpret_cast<uint32_t*>(ob + (long long)(r0 + 8) * osn + col) = pack_h2(oacc[mi][u][2] * i1, oacc[mi][u][3] * i1);
        }
    }
}

// qkv [B, N, 3, H, 16] fp16 contiguous -> o [B, N, H*16] fp16
torch::Tensor flash16(torch::Tensor qkv, double scale, bool pv16) {
    TORCH_CHECK(qkv.scalar_type() == at::kHalf && qkv.is_contiguous() && qkv.dim() == 5 && qkv.size(4) == 16, "qkv [B,N,3,H,16] fp16");
    int B = qkv.size(0), N = qkv.size(1), H = qkv.size(3);
    auto o = torch::empty({B, N, H * 16}, qkv.options());
    const __half* base = (const __half*)qkv.data_ptr();
    long long sb = (long long)N * 3 * H * 16;
    int sn = 3 * H * 16;
    dim3 grid((N + 127) / 128, B * H);
    auto st = at::cuda::getCurrentCUDAStream();
    float sl2 = (float)(scale * 1.4426950408889634);
    if (pv16)
        flash16_k<true><<<grid, 128, 0, st>>>(base, base + H * 16, base + 2 * H * 16, (__half*)o.data_ptr(), N, H, sb, sn, sb, sn,
                                              sb, sn, (long long)N * H * 16, H * 16, sl2);
    else
        flash16_k<false><<<grid, 128, 0, st>>>(base, base + H * 16, base + 2 * H * 16, (__half*)o.data_ptr(), N, H, sb, sn, sb,
                                               sn, sb, sn, (long long)N * H * 16, H * 16, sl2);
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("entropy", &entropy);
    m.def("aggregate", &aggregate);
    m.def("conv_small", &conv_small);
    m.def("fpn_last", &fpn_last);
    m.def("flash16", &flash16);
}
