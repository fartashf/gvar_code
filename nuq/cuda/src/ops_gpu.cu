
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>


// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
constexpr int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N)
{
  AT_ASSERTM(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

__global__ void _initCURand(unsigned int seed, curandState* states)
{
  unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  /* we have to initialize the state */
  curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
              index, /* the sequence number should be different for each core (unless you want all
                             cores to get the same sequence of numbers for some reason - use thread id! */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &states[index]);
}

//for QSGD: we should pass norm2 of the bucket and levels_uni,
// for NUQSGD: we should pass norm2 of the bucket and levels_exp,
// for QSGD-inf: we should pass maximum value of the bucket and levels_uni,
constexpr float EPS = 1e-7;

__global__ void _qdq(const float *in_vector, const float *norm, float
    *out_vector, const int n, const float *levels, const int num_levels, long
    *rand_vector)
{
    CUDA_KERNEL_LOOP(i, n) {
        int j = 0;
        float level_up, diff;
        while (j+1 < num_levels) 
        { 
            level_up =  levels[j+1];
            if (in_vector[i]/(norm[i]+EPS)<=level_up)
            {
                diff = level_up - levels[j];	
                if (in_vector[i]/(norm[i]+EPS)+diff*(rand_vector[i]%1000001 / 1000000.0)>level_up)
                {
                    j = j+1;
                }
                break;
            }
            j = j+1;			
        }
        out_vector[i] = norm[i]*levels[j];	        
    }
}


curandState* GPUInit_curand(int n, unsigned int seed, cudaStream_t stream)
{
    curandState* states;

    cudaMalloc(&states, GET_BLOCKS(n) * CUDA_NUM_THREADS * sizeof(curandState));

    _initCURand<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>>(seed, states);

    return states;    
}


void qdqGPUKernel(float *in_vector, float *norm, float *out_vector, int n,
        float *levels, int num_levels, long * rand_vector, cudaStream_t
        stream)
{
    _qdq<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>>(in_vector, norm,
            out_vector, n, levels, num_levels, rand_vector);
    // cudaStreamSynchronize(stream);
    
}
