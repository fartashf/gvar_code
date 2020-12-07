curandState* GPUInit_curand(int n, unsigned int seed, cudaStream_t stream);
void qdqGPUKernel(float *in_vector, float *norm, float *out_vector, int n,
        float *levels, int num_levels, long *rand_vector, cudaStream_t
        stream);