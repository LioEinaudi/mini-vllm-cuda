/*
RMSNorm v0 naive CUDA kernel

Input:
  x:      [num_tokens, hidden_size], CUDA tensor, float16 or float32
  weight: [hidden_size], CUDA tensor, same dtype as x
  eps:    numerical stability constant

Output:
  y:      [num_tokens, hidden_size], same dtype and shape as x

Math:
  rms = sqrt(mean(x_i^2) + eps)
  y_i = x_i / rms * weight_i

Thread/block mapping:
  - One CUDA block handles one token row.
  - Each thread processes columns col = tid, tid + blockDim.x, ...
  - Threads accumulate partial sum(x^2) in fp32.
  - Dynamic shared memory reduces thread partial sums into one row sum.
  - Threads make a second pass over the row and write normalized output.

Memory access pattern:
  For contiguous x and y, threads in a warp read and write neighboring columns,
  so global memory access is coalesced in the main row traversal. weight[col] is
  also read contiguously by the block.

Current limitations:
  This v0 favors readability over peak speed. It uses a full shared-memory tree
  reduction and reads x twice. Future versions can use warp-level reduction,
  vectorized loads, or half2 paths.
*/

#include "common.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

template <typename scalar_t>
__global__ void rmsnorm_v0_naive_kernel(
  const scalar_t *x ,
  const scalar_t *weight ,
  scalar_t *y ,
  int hidden_size , 
  float eps
){
  extern __shared__ float shared[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int row_offset = row * hidden_size;

  float thread_sum = 0.0f;

  for (int col = tid; col < hidden_size; col += blockDim.x ) {
    float v = static_cast<float> (x[row_offset + col]);
    thread_sum += v * v;
  }

  shared[tid] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2 ) {
    if ( tid < stride )  { 
    shared[tid] += shared[tid + stride];
  }
    __syncthreads();
  }
  
  float inv_rms = rsqrtf(shared[0] / static_cast <float> (hidden_size) + eps ) ;

  for (int col = tid; col < hidden_size; col +=blockDim.x ) {
    float v =static_cast <float>(x[col+row_offset]) ;
    float w = static_cast<float>(weight[col]);
    float out = v * inv_rms * w;
    y[col + row_offset] = static_cast<scalar_t>(out); 
  }
}

__device__ __forceinline__ float warp_reduce_sum(float value)
{
  unsigned int mask = 0xffffffffu;
  value += __shfl_down_sync(mask, value, 16);
  value += __shfl_down_sync(mask, value, 8);
  value += __shfl_down_sync(mask, value, 4);
  value += __shfl_down_sync(mask, value, 2);
  value += __shfl_down_sync(mask, value, 1);
  return value;
}

/*
RMSNorm v1 warp-reduce CUDA kernel

This kernel keeps the same public behavior and row/block mapping as v0:
one CUDA block handles one token row and accumulation stays in fp32.

The optimization is only in the row reduction:
  - each warp reduces its thread-local partial sums with __shfl_down_sync
  - each warp writes one warp sum to shared memory
  - warp 0 reduces those per-warp sums into the final row sum

Compared with v0, this reduces shared-memory reduction traffic and the number
of block-wide synchronizations. It still reads x twice, so memory dependency
and long scoreboard stalls can remain important for larger rows.
*/
template<typename scalar_t>
    __global__ void rmsnorm_v1_kernel(
        const scalar_t *x,
        const scalar_t *weight,
        scalar_t *y,
        int hidden_size,
        float eps)
{
  extern __shared__ float shared[];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  int row_offset = row * hidden_size;

  float thread_sum = 0.0f;

  for (int col = tid; col < hidden_size; col += blockDim.x)
  {
    float v = static_cast<float>(x[row_offset + col]);
    thread_sum += v * v;
  }

  int lane_id = tid & 31;
  int warp_id = tid >> 5;
  int num_warps = blockDim.x / 32;
  float warp_sum = warp_reduce_sum(thread_sum);
  if ( lane_id == 0 ) {
    shared[warp_id] = warp_sum;
  }
  __syncthreads();
  float block_sum = 0.0f;
  if ( warp_id == 0 ) {
    block_sum = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
    block_sum = warp_reduce_sum(block_sum);
    if ( lane_id == 0 ) {
      shared[0] = rsqrtf(block_sum / static_cast<float>(hidden_size) + eps);

    }
  }
  __syncthreads();
  float inv_rms = shared[0];

  for (int col = tid; col < hidden_size; col += blockDim.x)
  {
    float v = static_cast<float>(x[col + row_offset]);
    float w = static_cast<float>(weight[col]);
    float out = v * inv_rms * w;
    y[col + row_offset] = static_cast<scalar_t>(out);
  }
}

torch::Tensor rmsnorm_launch(torch::Tensor x, torch::Tensor weight, double eps, bool use_v1) {
  MVC_CHECK_CUDA(x);
  MVC_CHECK_CUDA(weight);
  TORCH_CHECK (x.dim() == 2, "x must have shape [num_tokens, hidden_size]");
  TORCH_CHECK(weight.dim() == 1, "weight must have shape [hidden_size]");
  TORCH_CHECK(last_dim(x) == weight.numel(), "weight must match x last dimension");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight must have the same dtype");
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat16,
      "rmsnorm supports only float32 and float16");

  const int64_t num_tokens = x.size(0);
  const int64_t hidden_size = x.size(1);
  TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
  TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");

  auto x_contig = x.contiguous();
  auto weight_contig = weight.contiguous();
  auto y = at::empty_like(x_contig);

  constexpr int threads = 256;
  dim3 block(threads); 
  const dim3 grid = (static_cast<unsigned int >(num_tokens));
  size_t shared_bytes = threads * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_contig.scalar_type(),"rmsnorm_kernel",[&]{
    if (use_v1) {
      rmsnorm_v1_kernel<scalar_t> <<< grid , block,shared_bytes,stream>>>(
        x_contig.data_ptr <scalar_t>() ,
        weight_contig.data_ptr < scalar_t>() ,
        y.data_ptr < scalar_t>() ,
        static_cast< int> ( hidden_size ) ,
        static_cast < float >(eps)
      );
    } else {
      rmsnorm_v0_naive_kernel<scalar_t> <<< grid , block,shared_bytes,stream>>>(
        x_contig.data_ptr <scalar_t>() ,
        weight_contig.data_ptr < scalar_t>() ,
        y.data_ptr < scalar_t>() ,
        static_cast< int> ( hidden_size ) ,
        static_cast < float >(eps)
      );
    }
  });
  
  C10_CUDA_KERNEL_LAUNCH_CHECK(); 
  return y; 
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  return rmsnorm_launch(x, weight, eps, false);
}

torch::Tensor rmsnorm_v1_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
  return rmsnorm_launch(x, weight, eps, true);
}
