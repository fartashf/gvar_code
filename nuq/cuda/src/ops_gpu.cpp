#include "src/ops_gpu.h"
#include "src/utils.hpp"

template <typename Dtype>
QDQ<Dtype>::QDQ(unsigned int bucket_size, at::Tensor levels, unsigned int seed)
    : bucket_size(bucket_size), levels(levels), seed(seed){
  // states = GPUInit_curand(bucket_size, seed, at::cuda::getCurrentCUDAStream());
}

template <typename Dtype>
void QDQ<Dtype>::qdqGPU(at::Tensor in_vector, at::Tensor norm, at::Tensor out_vector, at::Tensor rand_vector) {
  int N = in_vector.numel();
  if (N != out_vector.numel())  //  || N > bucket_size)
    throw std::invalid_argument(Formatter()
                                << "Size mismatch A.numel(): " << in_vector.numel()
                                << ", B.numel(): " << out_vector.numel());

  // out_vector.resize_({bucket_size});
  int num_levels = levels.numel();

  qdqGPUKernel(
          in_vector.data<Dtype>(),
          norm.data<Dtype>(),
          out_vector.data<Dtype>(),
          N,
          levels.data<Dtype>(), num_levels,
          rand_vector.data<long>(),
          at::cuda::getCurrentCUDAStream());
}

// template void QDQ::qdqGPU<float>(at::Tensor in_vector, float norm, at::Tensor out_vector, at::Tensor levels);

