#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <curand_kernel.h>
#include "src/ops_gpu.cuh"

template <typename Dtype>
class QDQ {
private:
  uint64_t bucket_size;
  at::Tensor levels;
  unsigned int seed;
  curandState* states;
public:
  QDQ(unsigned int bucket_size, at::Tensor levels, unsigned int seed = 1234);
  void qdqGPU(at::Tensor in_vector, at::Tensor norm, at::Tensor out_vector, at::Tensor rand_vector);
};

template class QDQ<float>;
