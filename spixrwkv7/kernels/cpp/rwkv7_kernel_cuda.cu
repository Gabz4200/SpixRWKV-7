#include <torch/torch.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>

#define HEAD_SIZE 64

namespace spixrwkv7 {
namespace kernel {

template <typename F>
__global__ void recurrent_scan_cuda_kernel(
    const int B, const int T, const int C, const int H,
    float *__restrict__ _state,     // [B, H, S, S]
    const F *__restrict__ const _r, // [B, T, H, S]
    const F *__restrict__ const _v, // [B, T, H, S]
    const F *__restrict__ const _w, // [B, T, H, S]
    const F *__restrict__ const _a, // [B, T, H, S]
    const F *__restrict__ const _kk,// [B, T, H, S]
    const F *__restrict__ const _kt,// [B, T, H, S]
    F *__restrict__ const _y)       // [B, T, H, S]
{
    const int e = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x; // Thread index from 0 to 63 (S-1)

    // Move state pointer to this batch, head, and thread's row
    _state += e * H * HEAD_SIZE * HEAD_SIZE + h * HEAD_SIZE * HEAD_SIZE + i * HEAD_SIZE;

    // Load initial state into registers
    float state[HEAD_SIZE];
    #pragma unroll
    for (int j = 0; j < HEAD_SIZE; j++) {
        state[j] = _state[j];
    }

    __shared__ float r[HEAD_SIZE], w[HEAD_SIZE], kt[HEAD_SIZE], kk[HEAD_SIZE], a[HEAD_SIZE];

    for (int _t = 0; _t < T; _t++)
    {
        // Compute offset in input/output tensors: [B, T, H, S]
        const int t = e * T * C + h * HEAD_SIZE + i + _t * C;
        
        __syncthreads();
        r[i] = float(_r[t]);
        w[i] = float(_w[t]);
        kt[i] = float(_kt[t]);
        kk[i] = float(_kk[t]);
        a[i] = float(_a[t]);
        __syncthreads();

        // sa = sum_j kk[j] * state[j]
        float sa = 0;
        #pragma unroll
        for (int j = 0; j < HEAD_SIZE; j++)
        {
            sa += kk[j] * state[j];
        }

        float vv = float(_v[t]);
        float y_val = 0;
        
        #pragma unroll
        for (int j = 0; j < HEAD_SIZE; j++)
        {
            float& s = state[j];
            s = s * w[j] + kt[j] * vv - sa * kk[j] * a[j];
            y_val += s * r[j];
        }
        
        _y[t] = F(y_val);
    }

    // Write final state back to global memory
    #pragma unroll
    for (int j = 0; j < HEAD_SIZE; j++) {
        _state[j] = state[j];
    }
}

torch::Tensor recurrent_scan_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k)
{
    const auto B = r.size(0);
    const auto T = r.size(1); // N timesteps
    const auto H = r.size(2); // Hd heads
    const auto S = r.size(3); // S head size
    const auto C = H * S;     // channels

    TORCH_CHECK(S == HEAD_SIZE, "CUDA kernel only supports S=64");
    TORCH_CHECK(state.is_cuda(), "state must be a CUDA tensor");
    TORCH_CHECK(r.is_cuda(), "r must be a CUDA tensor");

    auto out = torch::empty({B, T, H, S}, r.options());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(r.device().index());
    int grid = B * H;
    int block = S;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        r.scalar_type(), "recurrent_scan_cuda", ([&] {
            recurrent_scan_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
                B, T, C, H,
                state.data_ptr<float>(),
                r.data_ptr<scalar_t>(),
                v.data_ptr<scalar_t>(),
                w.data_ptr<scalar_t>(),
                a.data_ptr<scalar_t>(),
                kk.data_ptr<scalar_t>(),
                kt.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>()
            );
        })
    );

    return out;
}

torch::Tensor recurrent_scan_q4_0_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k)
{
    return recurrent_scan_cuda(state, r, k, v, w, a, kk, kt, r_k);
}

torch::Tensor recurrent_scan_q5_1_cuda(
    torch::Tensor& state,
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& kt,
    const torch::Tensor& r_k)
{
    return recurrent_scan_cuda(state, r, k, v, w, a, kk, kt, r_k);
}

} // namespace kernel
} // namespace spixrwkv7
