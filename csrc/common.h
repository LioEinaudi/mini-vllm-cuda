#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/util/Exception.h>
#include <torch/types.h>

#define MVC_CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define MVC_CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define MVC_CHECK_DTYPE(x, dtype) TORCH_CHECK((x).scalar_type() == dtype, #x " has unexpected dtype")

inline int64_t last_dim(const at::Tensor& x) {
  TORCH_CHECK(x.dim() > 0, "tensor must have at least one dimension");
  return x.size(x.dim() - 1);
}
