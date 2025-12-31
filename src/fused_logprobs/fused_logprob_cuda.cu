#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#if defined(__CUDA_ARCH__) || defined(__CUDACC__)
#include <cuda_bf16.h>
#endif

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

static inline __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

static inline __device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    return v;
}

static inline __device__ float block_reduce_sum(float v) {
    __shared__ float shared[32]; // up to 1024 threads => 32 warps
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

static inline __device__ float block_reduce_max(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    v = warp_reduce_max(v);
    if (lane == 0) shared[wid] = v;
    __syncthreads();

    v = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -INFINITY;
    if (wid == 0) v = warp_reduce_max(v);
    return v;
}

template <typename scalar_t>
__device__ inline float to_float(scalar_t x) {
    return (float)x;
}
template <typename scalar_t>
__device__ inline scalar_t from_float(float x) {
    return (scalar_t)x;
}

// at::Half <-> float using CUDA __half intrinsics (no operator casting)
template <>
__device__ inline float to_float<at::Half>(at::Half x) {
    const __half* hx = reinterpret_cast<const __half*>(&x);
    return __half2float(*hx);
}
template <>
__device__ inline at::Half from_float<at::Half>(float x) {
    __half h = __float2half_rn(x);
    return *reinterpret_cast<at::Half*>(&h);
}

// at::BFloat16 <-> float using CUDA __nv_bfloat16 intrinsics
template <>
__device__ inline float to_float<at::BFloat16>(at::BFloat16 x) {
    const __nv_bfloat16* bx = reinterpret_cast<const __nv_bfloat16*>(&x);
    return __bfloat162float(*bx);
}
template <>
__device__ inline at::BFloat16 from_float<at::BFloat16>(float x) {
    __nv_bfloat16 b = __float2bfloat16_rn(x);
    return *reinterpret_cast<at::BFloat16*>(&b);
}

template <typename scalar_t>
__global__ void fused_logprob_fwd_kernel(
    const scalar_t* __restrict__ logits,
    const int32_t* __restrict__ targets,   // <-- int32
    float* __restrict__ out_logp,
    int N, int V
) {
    int row = (int)blockIdx.x;
    if (row >= N) return;

    // Shared memory for broadcasting reduced max value
    __shared__ float shared_m;

    // max over V
    float tmax = -INFINITY;
    for (int j = (int)threadIdx.x; j < V; j += (int)blockDim.x) {
        float x = to_float<scalar_t>(logits[row * V + j]);
        tmax = fmaxf(tmax, x);
    }
    float m = block_reduce_max(tmax);
    __syncthreads();

    // Broadcast m to all threads
    if (threadIdx.x == 0) {
        shared_m = m;
    }
    __syncthreads();
    m = shared_m;

    // sum exp
    float tsum = 0.0f;
    for (int j = (int)threadIdx.x; j < V; j += (int)blockDim.x) {
        float x = to_float<scalar_t>(logits[row * V + j]);
        tsum += __expf(x - m);
    }
    float s = block_reduce_sum(tsum);
    __syncthreads();

    if (threadIdx.x == 0) {
        float lse = logf(s) + m;
        int32_t t = targets[row];
        if (t < 0 || t >= V) {
            out_logp[row] = -INFINITY;
        } else {
            float xt = to_float<scalar_t>(logits[row * V + (int)t]);
            out_logp[row] = xt - lse;
        }
    }
}

template <typename scalar_t>
__global__ void fused_logprob_bwd_kernel(
    const scalar_t* __restrict__ logits,
    const int32_t* __restrict__ targets,   // <-- int32
    const float* __restrict__ grad_logp,
    scalar_t* __restrict__ grad_logits,
    int N, int V
) {
    int row = (int)blockIdx.x;
    if (row >= N) return;

    // Shared memory for broadcasting reduced values
    __shared__ float shared_m;
    __shared__ float shared_inv_s;
    __shared__ int32_t shared_t;
    __shared__ float shared_g;

    // max
    float tmax = -INFINITY;
    for (int j = (int)threadIdx.x; j < V; j += (int)blockDim.x) {
        float x = to_float<scalar_t>(logits[row * V + j]);
        tmax = fmaxf(tmax, x);
    }
    float m = block_reduce_max(tmax);
    __syncthreads();

    // Broadcast m to all threads
    if (threadIdx.x == 0) {
        shared_m = m;
    }
    __syncthreads();
    m = shared_m;

    // sum exp
    float tsum = 0.0f;
    for (int j = (int)threadIdx.x; j < V; j += (int)blockDim.x) {
        float x = to_float<scalar_t>(logits[row * V + j]);
        tsum += __expf(x - m);
    }
    float s = block_reduce_sum(tsum);
    __syncthreads();

    // Broadcast inv_s, target, and grad_logp to all threads
    if (threadIdx.x == 0) {
        shared_inv_s = 1.0f / s;
        shared_t = targets[row];
        shared_g = grad_logp[row];
    }
    __syncthreads();
    
    float inv_s = shared_inv_s;
    int32_t t = shared_t;
    float g = shared_g;

    for (int j = (int)threadIdx.x; j < V; j += (int)blockDim.x) {
        float x = to_float<scalar_t>(logits[row * V + j]);
        float p = __expf(x - m) * inv_s;         // softmax
        float grad = -g * p;
        if ((int)j == t) grad += g;
        grad_logits[row * V + j] = from_float<scalar_t>(grad);
    }
}

std::vector<torch::Tensor> fused_logprob_forward_cuda(torch::Tensor logits, torch::Tensor targets) {
    CHECK_CUDA(logits);
    CHECK_CUDA(targets);

    TORCH_CHECK(logits.dim() == 3, "logits must be [B,T,V]");
    TORCH_CHECK(targets.dim() == 2, "targets must be [B,T]");
    TORCH_CHECK(targets.scalar_type() == torch::kInt32, "targets must be int32 (internal)");

    auto B = logits.size(0);
    auto T = logits.size(1);
    auto V = logits.size(2);

    int64_t N64 = B * T;
    TORCH_CHECK(N64 <= INT_MAX, "B*T too large");
    int N = (int)N64;
    int Vi = (int)V;

    auto logits2  = logits.contiguous().view({N, V});
    auto targets32 = targets.to(torch::kInt32).contiguous();
    auto targets1  = targets32.view({N});

    auto out_logp = torch::empty({N},
        torch::TensorOptions().device(logits.device()).dtype(torch::kFloat32));

    const int threads = 256;
    const dim3 blocks(N);

    auto st = logits.scalar_type();
    if (st == torch::kFloat32) {
        fused_logprob_fwd_kernel<float><<<blocks, threads>>>(
            (const float*)logits2.data_ptr<float>(),
            (const int32_t*)targets1.data_ptr<int32_t>(),
            (float*)out_logp.data_ptr<float>(),
            N, Vi
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (st == torch::kFloat16) {
        fused_logprob_fwd_kernel<at::Half><<<blocks, threads>>>(
            (const at::Half*)logits2.data_ptr<at::Half>(),
            (const int32_t*)targets1.data_ptr<int32_t>(),
            (float*)out_logp.data_ptr<float>(),
            N, Vi
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (st == torch::kBFloat16) {
        fused_logprob_fwd_kernel<at::BFloat16><<<blocks, threads>>>(
            (const at::BFloat16*)logits2.data_ptr<at::BFloat16>(),
            (const int32_t*)targets1.data_ptr<int32_t>(),
            (float*)out_logp.data_ptr<float>(),
            N, Vi
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        TORCH_CHECK(false, "Unsupported logits dtype (expected fp16/bf16/fp32)");
    }

    return { out_logp.view({B, T}) };
}

torch::Tensor fused_logprob_backward_cuda(torch::Tensor logits, torch::Tensor targets, torch::Tensor grad_logp) {
    CHECK_CUDA(logits);
    CHECK_CUDA(targets);
    CHECK_CUDA(grad_logp);

    TORCH_CHECK(logits.dim() == 3, "logits must be [B,T,V]");
    TORCH_CHECK(targets.dim() == 2, "targets must be [B,T]");
    TORCH_CHECK(grad_logp.dim() == 2, "grad_logp must be [B,T]");
    TORCH_CHECK(targets.scalar_type() == torch::kInt32, "targets must be int32 (internal)");
    TORCH_CHECK(grad_logp.scalar_type() == torch::kFloat32, "grad_logp must be float32 (matches forward output)");

    auto B = logits.size(0);
    auto T = logits.size(1);
    auto V = logits.size(2);

    int64_t N64 = B * T;
    TORCH_CHECK(N64 <= INT_MAX, "B*T too large");
    int N = (int)N64;
    int Vi = (int)V;

    auto logits2  = logits.contiguous().view({N, V});
    auto targets32 = targets.to(torch::kInt32).contiguous();
    auto targets1  = targets32.view({N});
    auto g1       = grad_logp.contiguous().view({N});

    auto grad_logits = torch::empty_like(logits2);

    const int threads = 256;
    const dim3 blocks(N);

    auto st = logits.scalar_type();
    if (st == torch::kFloat32) {
        fused_logprob_bwd_kernel<float><<<blocks, threads>>>(
            (const float*)logits2.data_ptr<float>(),
            (const int32_t*)targets1.data_ptr<int32_t>(),
            (const float*)g1.data_ptr<float>(),
            (float*)grad_logits.data_ptr<float>(),
            N, Vi
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (st == torch::kFloat16) {
        fused_logprob_bwd_kernel<at::Half><<<blocks, threads>>>(
            (const at::Half*)logits2.data_ptr<at::Half>(),
            (const int32_t*)targets1.data_ptr<int32_t>(),
            (const float*)g1.data_ptr<float>(),
            (at::Half*)grad_logits.data_ptr<at::Half>(),
            N, Vi
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (st == torch::kBFloat16) {
        fused_logprob_bwd_kernel<at::BFloat16><<<blocks, threads>>>(
            (const at::BFloat16*)logits2.data_ptr<at::BFloat16>(),
            (const int32_t*)targets1.data_ptr<int32_t>(),
            (const float*)g1.data_ptr<float>(),
            (at::BFloat16*)grad_logits.data_ptr<at::BFloat16>(),
            N, Vi
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        TORCH_CHECK(false, "Unsupported logits dtype (expected fp16/bf16/fp32)");
    }

    return grad_logits.view({B, T, V});
}
