#include <torch/torch.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <algorithm>

namespace spixrwkv7 {
namespace kernel {

// Helper: general block sum reduction
__device__ inline float block_sum(float val, float* s_temp, int block_dim, int tid) {
    s_temp[tid] = val;
    __syncthreads();
    
    int active_threads = block_dim;
    while (active_threads > 1) {
        int half = (active_threads + 1) / 2;
        if (tid < active_threads - half) {
            s_temp[tid] += s_temp[tid + half];
        }
        active_threads = half;
        __syncthreads();
    }
    return s_temp[0];
}

__global__ void update_clusters_cuda_kernel(
    const float* __restrict__ elem,    // [B, C, H, W]
    const float* __restrict__ clst,    // [B, C, h_s, w_s]
    float* __restrict__ result,        // [B, C, h_s, w_s]
    int stride_h, int stride_w,
    int radius, float tau, bool normalize,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t h_s, int64_t w_s)
{
    // One block per cluster
    int64_t b = blockIdx.x / (h_s * w_s);
    int64_t idx = blockIdx.x % (h_s * w_s);
    int64_t i = idx / w_s;
    int64_t j = idx % w_s;
    int tid = threadIdx.x; // Thread index from 0 to C-1

    extern __shared__ float s_mem[];
    // Partition shared memory dynamically based on C
    float* s_temp = s_mem;
    float* s_sim_buf = s_temp + C;
    float* s_center = s_sim_buf + 128; // Assuming max window size is 128
    
    __shared__ float s_center_norm;
    __shared__ float s_max_sim;
    __shared__ float s_sum_exp;

    const int cy = i * stride_h;
    const int cx = j * stride_w;

    // Load center for this thread's channel
    if (tid < C) {
        s_center[tid] = clst[b * C * h_s * w_s + tid * h_s * w_s + i * w_s + j];
    }
    __syncthreads();

    if (normalize) {
        float val = 0.0f;
        if (tid < C) {
            val = s_center[tid] * s_center[tid];
        }
        float norm_sq = block_sum(val, s_temp, C, tid);
        if (tid == 0) {
            s_center_norm = 1.0f / sqrtf(max(norm_sq, 1e-8f));
        }
        __syncthreads();
    }

    int valid_count = 0;
    for (int di = -radius; di <= radius; di++) {
        for (int dj = -radius; dj <= radius; dj++) {
            const int py = cy + di * stride_h;
            const int px = cx + dj * stride_w;

            if (py >= 0 && py < H && px >= 0 && px < W) {
                float p_val = 0.0f;
                float p_norm_val = 0.0f;
                if (tid < C) {
                    p_val = elem[b * C * H * W + tid * H * W + py * W + px];
                    p_norm_val = p_val * p_val;
                }
                
                float p_norm_sq = 0.0f;
                if (normalize) {
                    p_norm_sq = block_sum(p_norm_val, s_temp, C, tid);
                }
                
                float dot_val = 0.0f;
                if (tid < C) {
                    if (normalize) {
                        float p_inv_norm = 1.0f / sqrtf(max(p_norm_sq, 1e-8f));
                        dot_val = (s_center[tid] * s_center_norm) * (p_val * p_inv_norm);
                    } else {
                        dot_val = s_center[tid] * p_val;
                    }
                }
                
                float sim = block_sum(dot_val, s_temp, C, tid);
                if (tid == 0) {
                    s_sim_buf[valid_count] = sim / tau;
                }
                __syncthreads();
                valid_count++;
            }
        }
    }

    if (valid_count == 0) {
        if (tid < C) {
            result[b * C * h_s * w_s + tid * h_s * w_s + i * w_s + j] = 0.0f;
        }
        return;
    }

    if (tid == 0) {
        float max_s = s_sim_buf[0];
        for (int p = 1; p < valid_count; p++) {
            max_s = max(max_s, s_sim_buf[p]);
        }
        s_max_sim = max_s;
    }
    __syncthreads();

    float val_exp = 0.0f;
    if (tid < valid_count) {
        val_exp = expf(s_sim_buf[tid] - s_max_sim);
    }
    float sum_exp = block_sum(val_exp, s_temp, C, tid);
    if (tid == 0) {
        s_sum_exp = sum_exp;
    }
    __syncthreads();

    float inv_sum = 1.0f / (s_sum_exp + 1e-10f);

    float dst_val = 0.0f;
    int pixel_idx = 0;
    for (int di = -radius; di <= radius; di++) {
        for (int dj = -radius; dj <= radius; dj++) {
            const int py = cy + di * stride_h;
            const int px = cx + dj * stride_w;

            if (py >= 0 && py < H && px >= 0 && px < W) {
                float weight = expf(s_sim_buf[pixel_idx] - s_max_sim) * inv_sum;
                if (tid < C) {
                    float p_val = elem[b * C * H * W + tid * H * W + py * W + px];
                    dst_val += weight * p_val;
                }
                pixel_idx++;
            }
        }
    }

    if (tid < C) {
        result[b * C * h_s * w_s + tid * h_s * w_s + i * w_s + j] = dst_val;
    }
}

__global__ void assign_pixels_cuda_kernel(
    const float* __restrict__ elem,    // [B, C, H, W]
    const float* __restrict__ clst,    // [B, C, h_s, w_s]
    float* __restrict__ result,        // [B, nn*nn, H, W]
    int stride_h, int stride_w,
    int radius, float tau, int nn,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t h_s, int64_t w_s)
{
    // One block per pixel
    int64_t b = blockIdx.x / (H * W);
    int64_t idx = blockIdx.x % (H * W);
    int64_t y = idx / W;
    int64_t x = idx % W;
    int tid = threadIdx.x; // Thread index from 0 to C-1

    extern __shared__ float s_mem[];
    float* s_temp = s_mem;
    float* s_sim_buf = s_temp + C;
    int* s_valid_idx = reinterpret_cast<int*>(s_sim_buf + 128);
    __shared__ float s_max_sim;
    __shared__ float s_sum_exp;

    const int ci = y / stride_h;
    const int cj = x / stride_w;

    float p_val = 0.0f;
    if (tid < C) {
        p_val = elem[b * C * H * W + tid * H * W + y * W + x];
    }
    __syncthreads();

    int valid_count = 0;
    for (int di = -radius; di <= radius; di++) {
        for (int dj = -radius; dj <= radius; dj++) {
            const int ni = ci + di;
            const int nj = cj + dj;

            if (ni >= 0 && ni < h_s && nj >= 0 && nj < w_s) {
                float center_val = 0.0f;
                if (tid < C) {
                    center_val = clst[b * C * h_s * w_s + tid * h_s * w_s + ni * w_s + nj];
                }
                
                float sim_part = p_val * center_val;
                float sim = block_sum(sim_part, s_temp, C, tid);
                
                if (tid == 0) {
                    s_sim_buf[valid_count] = sim / tau;
                    s_valid_idx[valid_count] = (di + radius) * nn + (dj + radius);
                }
                __syncthreads();
                valid_count++;
            }
        }
    }

    // Zero out all outputs for this pixel
    if (tid < nn * nn) {
        result[b * nn * nn * H * W + tid * H * W + y * W + x] = 0.0f;
    }
    __syncthreads();

    if (valid_count == 0) {
        return;
    }

    if (tid == 0) {
        float max_s = s_sim_buf[0];
        for (int p = 1; p < valid_count; p++) {
            max_s = max(max_s, s_sim_buf[p]);
        }
        s_max_sim = max_s;
    }
    __syncthreads();

    float val_exp = 0.0f;
    if (tid < valid_count) {
        val_exp = expf(s_sim_buf[tid] - s_max_sim);
    }
    float sum_exp = block_sum(val_exp, s_temp, C, tid);
    if (tid == 0) {
        s_sum_exp = sum_exp;
    }
    __syncthreads();

    float inv_sum = 1.0f / (s_sum_exp + 1e-10f);

    if (tid < valid_count) {
        float weight = expf(s_sim_buf[tid] - s_max_sim) * inv_sum;
        int target_idx = s_valid_idx[tid];
        result[b * nn * nn * H * W + target_idx * H * W + y * W + x] = weight;
    }
}

torch::Tensor update_clusters_cuda(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau, bool normalize)
{
    const auto B = elem_feats.size(0);
    const auto C = elem_feats.size(1);
    const auto H = elem_feats.size(2);
    const auto W = elem_feats.size(3);
    const auto h_s = clst_feats.size(2);
    const auto w_s = clst_feats.size(3);

    TORCH_CHECK(elem_feats.is_cuda(), "elem_feats must be a CUDA tensor");
    TORCH_CHECK(clst_feats.is_cuda(), "clst_feats must be a CUDA tensor");

    auto out = torch::empty({B, C, h_s, w_s}, elem_feats.options());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(elem_feats.device().index());
    int grid = B * h_s * w_s;
    int block = C;
    
    // Shared memory size: s_temp (C floats) + s_sim_buf (128 floats) + s_center (C floats)
    size_t shared_mem = (C + 128 + C) * sizeof(float);

    update_clusters_cuda_kernel<<<grid, block, shared_mem, stream>>>(
        elem_feats.data_ptr<float>(),
        clst_feats.data_ptr<float>(),
        out.data_ptr<float>(),
        stride_h, stride_w, radius, tau, normalize,
        B, C, H, W, h_s, w_s
    );

    return out;
}

torch::Tensor assign_pixels_cuda(
    const torch::Tensor& elem_feats,
    const torch::Tensor& clst_feats,
    int stride_h, int stride_w,
    int radius, float tau)
{
    const auto B = elem_feats.size(0);
    const auto C = elem_feats.size(1);
    const auto H = elem_feats.size(2);
    const auto W = elem_feats.size(3);
    const auto h_s = clst_feats.size(2);
    const auto w_s = clst_feats.size(3);
    const int nn = 2 * radius + 1;

    TORCH_CHECK(elem_feats.is_cuda(), "elem_feats must be a CUDA tensor");
    TORCH_CHECK(clst_feats.is_cuda(), "clst_feats must be a CUDA tensor");

    auto out = torch::empty({B, nn * nn, H, W}, elem_feats.options());
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(elem_feats.device().index());
    int grid = B * H * W;
    int block = C;

    // Shared memory size: s_temp (C floats) + s_sim_buf (128 floats) + s_valid_idx (128 ints)
    size_t shared_mem = (C + 128) * sizeof(float) + 128 * sizeof(int);

    assign_pixels_cuda_kernel<<<grid, block, shared_mem, stream>>>(
        elem_feats.data_ptr<float>(),
        clst_feats.data_ptr<float>(),
        out.data_ptr<float>(),
        stride_h, stride_w, radius, tau, nn,
        B, C, H, W, h_s, w_s
    );

    return out;
}

} // namespace kernel
} // namespace spixrwkv7
