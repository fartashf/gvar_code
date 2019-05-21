
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

#define maxThreadsPerBlock 1024

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

__global__ void _qdq(const float *in_vector, const float norm, float *out_vector, const int n, const float *levels, const int num_levels, curandState* states)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;    

    if (index >= n)
    return;

    curandState local_state;
    local_state = states[index];

    for (int i = index; i < n; i += stride)
    {	
        int j = 0;
        float level_up, diff;
        while (j+1 < num_levels) 
        { 
            level_up =  levels[j+1];
            if (in_vector[i]/norm<=level_up)
            {
                diff = level_up - levels[j];	
                if (in_vector[i]/norm+diff*(curand(&local_state)%1000001 / 1000000.0)>level_up)
                {
                    j = j+1;
                }
                break;
            }
            j = j+1;			
        }
        out_vector[i] = norm*levels[j];	        
    }
    states[index] = local_state;          
}


curandState* GPUInit_curand(int n, unsigned int seed, cudaStream_t stream)
{
    curandState* states;

    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    cudaMalloc(&states, blocksPerGrid * maxThreadsPerBlock * sizeof(curandState));

    _initCURand<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(seed, states);

    return states;    
}


void qdqGPUKernel(float *in_vector, float norm, float *out_vector, int n,
        float *levels, int num_levels, curandState* states, cudaStream_t
        stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _qdq<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(in_vector, norm, out_vector, n, levels, num_levels, states);
    cudaStreamSynchronize(stream);
    
}
