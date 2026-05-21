#include <torch/extension.h>

torch::Tensor rmsnorm_cuda(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor rmsnorm_v1_cuda(torch::Tensor x, torch::Tensor weight, double eps);
std::vector<torch::Tensor> rope_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin);
torch::Tensor decode_attention_cuda(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache, int64_t seq_len);
torch::Tensor int8_gemv_cuda(torch::Tensor x, torch::Tensor w_int8, torch::Tensor scales);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rmsnorm", &rmsnorm_cuda, "RMSNorm v0 forward (CUDA)");
  m.def("rmsnorm_v1", &rmsnorm_v1_cuda, "RMSNorm v1 warp-reduce forward (CUDA)");
  m.def("rope", &rope_cuda, "RoPE forward for q and k (CUDA placeholder)");
  m.def("decode_attention", &decode_attention_cuda, "Decode attention over continuous KV cache (CUDA placeholder)");
  m.def("int8_gemv", &int8_gemv_cuda, "INT8 weight-only GEMV (CUDA placeholder)");
}
