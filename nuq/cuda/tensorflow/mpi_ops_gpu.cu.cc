

#define EIGEN_USE_GPU

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "mpi_cuda.h"


#define maxThreadsPerBlock 1024
// #define bucketsize 512
// #define num_levels 4

__global__ void _scaleAndAdd(int n, float scale1, float *x, float scale2, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = scale1 * x[i] + scale2 * y[i];              

}

__global__ void _scale(int n, float scaler, float *x)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = scaler * x[i];              

}

__global__ void _add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];              

}

/*__global__ void _findMaxAndMin2(float *array, float *max, float *min, int *mutex, int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    unsigned int offset = 0;

    __shared__ float cache1[maxThreadsPerBlock];//  maxThreadsPerBlock is fixed to be 1024. 
    __shared__ float cache2[maxThreadsPerBlock];


    float temp1 = 0;
    float temp2 = 0;
    while(index + offset < n){

        temp1 = fmaxf(temp1, array[index + offset]);// defined in <math.h> float  fmaxf( float x, float y ); returns the max of 2 floats. 
        temp2 = fminf(temp2, array[index + offset]);

        offset += stride;
    }

    cache1[threadIdx.x] = temp1;
    cache2[threadIdx.x] = temp2;

    __syncthreads();

    // reduction
    unsigned int i = blockDim.x/2;
    while(i != 0){
        if(threadIdx.x < i){

            cache1[threadIdx.x] = fmaxf(cache1[threadIdx.x], cache1[threadIdx.x + i]);
            cache2[threadIdx.x] = fminf(cache2[threadIdx.x], cache2[threadIdx.x + i]);
        }

        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0){
        while(atomicCAS(mutex,0,1) != 0);  //lock
        *max = fmaxf(*max, cache1[0]);
        *min = fminf(*min, cache2[0]);
        atomicExch(mutex, 0);  //unlock
    }
}

__global__ void _findMaxAndMin(float *array, float *maxandmin, int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    __shared__ float cache1[maxThreadsPerBlock];
    __shared__ float cache2[maxThreadsPerBlock];

    for(int j = index; j < n; j += stride)
    {

        int my_bucket = j / bucketsize;
        int index_in_bucket = j % bucketsize;
        int offset = (my_bucket&1) ? bucketsize : 0;

        // reduction
        unsigned int i = bucketsize / 2;
        while(i != 0)
        {
            if(index_in_bucket < i)
            {

                if(i == bucketsize / 2) //get data in cache in first loop
                {
                    cache1[index_in_bucket + offset] = fmaxf(array[j], array[j + i]);
                    cache2[index_in_bucket + offset] = fminf(array[j], array[j + i]);                 
                }
                else
                {
                    cache1[index_in_bucket + offset] = fmaxf(cache1[index_in_bucket + offset], cache1[index_in_bucket + offset + i]);
                    cache2[index_in_bucket + offset] = fminf(cache2[index_in_bucket + offset], cache2[index_in_bucket + offset + i]);  
                }

            }
            __syncthreads();
            i /= 2;
        }

        

        if(threadIdx.x == 0)
        {
            maxandmin[2 * my_bucket] = cache1[0];
            maxandmin[2 * my_bucket + 1] = cache2[0];
        }
        else if(threadIdx.x == bucketsize)//threadIdx can be 0 t0 1023. 
        {
            maxandmin[2 * my_bucket] = cache1[bucketsize];
            maxandmin[2 * my_bucket + 1] = cache2[bucketsize];
        }
    }

}*/

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


/*__global__ void _quantizeValue(unsigned char *x, const float *y, const float *maxandmin, const int n, curandState* states)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    curandState local_state;
    local_state = states[index];


    for (int i = index; i < n; i += stride)
    {
        int my_bucket = i / bucketsize;
        float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) / 15.0;
        float d = (y[i] - maxandmin[my_bucket * 2 + 1]) / unit + (curand(&local_state)%1000001 / 1000000.0); 
        x[i] = (unsigned char) floor(d);
    }
    states[index] = local_state;       
}




__global__ void _dequantizeValue(unsigned char *recv, float *maxandmin, float *x, const int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
    {
        int my_bucket = i / bucketsize;
        float unit = (maxandmin[my_bucket * 2] - maxandmin[my_bucket * 2 + 1]) / 15.0;
        x[i] = maxandmin[my_bucket * 2 + 1] + recv[i] * unit;  
    }          
}*/

__global__ void _copyValue(float* x, const float* y, const int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        x[i] = y[i];
}


__global__ void _findNorm2(float *array, float *norm, int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    const int bucketsize = 512;
    __shared__ float cachenorm1[bucketsize];
    __shared__ float cachenorm2[bucketsize];
    
    for(int j = index; j < n; j += stride)
    {

        int my_bucket = j / bucketsize;
        int index_in_bucket = j % bucketsize;
	if(my_bucket==0)
	{
	cachenorm1[index_in_bucket] = array[j];	
	}
	else if(my_bucket==1)
	{
        cachenorm2[index_in_bucket] = array[j];	
	}
       
        __syncthreads();
    }
     norm[0] = normf(bucketsize,cachenorm1);
     norm[1] = normf(bucketsize,cachenorm2);
}

/*__global__ void _QSGDquantizeValue(signed char *x, const float *y, const float *norm, const int n, curandState* states)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    curandState local_state;
    local_state = states[index];


    for (int i = index; i < n; i += stride)
    {
        int my_bucket = i / bucketsize;
	float unit = 2*norm[my_bucket] / 15.0;
        float d = (y[i] + norm[my_bucket]) / unit -8.0 + (curand(&local_state)%1000001 / 1000000.0);
        x[i] = (signed char) floor(d);
    }
    states[index] = local_state;       
}

__global__ void _QSGDdequantizeValue(signed char *recv, const float *norm, float *x, const int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride)
    {
        int my_bucket = i / bucketsize;
	float unit = 2*norm[my_bucket] / 15.0;
        x[i] = -norm[my_bucket] + (recv[i]+8.0) * unit;
    }          
}*/

__global__ void _NUQSGDquantizeValue(signed char *x, const float *y, const float *norm, const int n, curandState* states)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;

    curandState local_state;
    local_state = states[index];
    const int bucketsize = 512;
    const int num_levels = 16;
/*    float levels[num_levels] = {-1.0, -0.9, -0.81, -0.7290000000000001, -0.6561, -0.5904900000000001, -0.531441, -0.4782969000000001, -0.4304672100000001, -0.3874204890000001, -0.3486784401000001, -0.31381059609000006, -0.2824295364810001, -0.2541865828329001, -0.2287679245496101, -0.20589113209464907, -0.18530201888518416, -0.16677181699666577, -0.15009463529699918, -0.13508517176729928, -0.12157665459056935, -0.10941898913151242, -0.09847709021836118, -0.08862938119652507, -0.07976644307687256, -0.0717897987691853, -0.06461081889226677, -0.058149737003040096, -0.05233476330273609, -0.047101286972462485, -0.04239115827521624, -0.038152042447694615, 0.038152042447694615, 0.04239115827521624, 0.047101286972462485, 0.05233476330273609, 0.058149737003040096, 0.06461081889226677, 0.0717897987691853, 0.07976644307687256, 0.08862938119652507, 0.09847709021836118, 0.10941898913151242, 0.12157665459056935, 0.13508517176729928, 0.15009463529699918, 0.16677181699666577, 0.18530201888518416, 0.20589113209464907, 0.2287679245496101, 0.2541865828329001, 0.2824295364810001, 0.31381059609000006, 0.3486784401000001, 0.3874204890000001, 0.4304672100000001, 0.4782969000000001, 0.531441, 0.5904900000000001, 0.6561, 0.7290000000000001, 0.81, 0.9, 1.0};*/
    float multiplier = 0.5;
    	float levels[num_levels];
    	float value = 1.0;
    	for (int j = 0; j < num_levels/2; j++) {
    		levels[j] = -value;
    		levels[num_levels-1-j] = value;
    		value = value * multiplier;
    }
    for (int i = index; i < n; i += stride)
    {	
        int my_bucket = i / bucketsize;
	int j = 0;
	float level_up, diff;
	while (j+1 < num_levels) 
	{ 
		level_up =  levels[j+1];
		if (y[i]/norm[my_bucket]<=level_up)
		{
			diff = level_up - levels[j];	
			if (y[i]/norm[my_bucket]+diff*(curand(&local_state)%1000001 / 1000000.0)>level_up)
			{
				j = j+1;
			}
			break;
		}
		j = j+1;			
	}
	x[i] = (signed char) j;
    }
    states[index] = local_state;       
}

__global__ void _NUQSGDdequantizeValue(signed char *recv, const float *norm, float *x, const int n)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    const int bucketsize = 512;
    const int num_levels = 16;
    /*float levels[num_levels] = {-1.0, -0.9, -0.81, -0.7290000000000001, -0.6561, -0.5904900000000001, -0.531441, -0.4782969000000001, -0.4304672100000001, -0.3874204890000001, -0.3486784401000001, -0.31381059609000006, -0.2824295364810001, -0.2541865828329001, -0.2287679245496101, -0.20589113209464907, -0.18530201888518416, -0.16677181699666577, -0.15009463529699918, -0.13508517176729928, -0.12157665459056935, -0.10941898913151242, -0.09847709021836118, -0.08862938119652507, -0.07976644307687256, -0.0717897987691853, -0.06461081889226677, -0.058149737003040096, -0.05233476330273609, -0.047101286972462485, -0.04239115827521624, -0.038152042447694615, 0.038152042447694615, 0.04239115827521624, 0.047101286972462485, 0.05233476330273609, 0.058149737003040096, 0.06461081889226677, 0.0717897987691853, 0.07976644307687256, 0.08862938119652507, 0.09847709021836118, 0.10941898913151242, 0.12157665459056935, 0.13508517176729928, 0.15009463529699918, 0.16677181699666577, 0.18530201888518416, 0.20589113209464907, 0.2287679245496101, 0.2541865828329001, 0.2824295364810001, 0.31381059609000006, 0.3486784401000001, 0.3874204890000001, 0.4304672100000001, 0.4782969000000001, 0.531441, 0.5904900000000001, 0.6561, 0.7290000000000001, 0.81, 0.9, 1.0};*/
   float multiplier = 0.5;
   float levels[num_levels];
   float value = 1.0;
   for (int j = 0; j < num_levels/2; j++) {
    		levels[j] = -value;
    		levels[num_levels-1-j] = value;
    		value = value * multiplier;
    }
    for (int i = index; i < n; i += stride)
    {
    	
        int my_bucket = i / bucketsize;
	x[i] = norm[my_bucket]*levels[recv[i]];
    }          
}

void GPUScaleAndAdd(int n, float scale1, float *x, float scale2, float *y, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _scaleAndAdd<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(n, scale1, x, scale2, y);
    cudaStreamSynchronize(stream);	    
}


void GPUScale(int n, float scaler, float *x, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _scale<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(n, scaler, x);
    cudaStreamSynchronize(stream);	    
}

void GPUAdd(int n, float *x, float *y, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _add<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(n, x, y);
    cudaStreamSynchronize(stream);	    
}


/*void GPUFindMaxAndMin(float *array, float *maxandmin, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _findMaxAndMin<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(array, maxandmin, n);
    cudaStreamSynchronize(stream); 
}

void GPUFindMaxAndMin2(float *array, float *max, float *min, int *mutex, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _findMaxAndMin2<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(array, max, min, mutex, n);
    cudaStreamSynchronize(stream);
    
}*/

curandState* GPUInit_curand(int n, unsigned int seed, cudaStream_t stream)
{
    curandState* states;

    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    cudaMalloc(&states, blocksPerGrid * maxThreadsPerBlock * sizeof(curandState));

    _initCURand<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(seed, states);

    return states;    
}

/*void GPUQuantizeValue(unsigned char *x, float *y, float *maxandmin, int n, curandState* states, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _quantizeValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, y, maxandmin, n, states);
    cudaStreamSynchronize(stream);
    
}

void GPUDequantizeValue(unsigned char *recv, float *maxandmin, float *x, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _dequantizeValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(recv, maxandmin, x, n);
    cudaStreamSynchronize(stream);
    
}*/


void GPUCopyValue(float* x, float* y, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _copyValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, y, n);
    cudaStreamSynchronize(stream);
    
} 


void GPUfindNorm2(float *array, float *norm, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _findNorm2<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(array, norm, n);
    cudaStreamSynchronize(stream);
    
}

/*void GPUQSGDquantizeValue(signed char *x, float *y, float *norm, int n, curandState* states, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _QSGDquantizeValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, y, norm, n, states);
    cudaStreamSynchronize(stream);
    
}

void GPUQSGDdequantizeValue(signed char *recv, float *norm, float *x, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _QSGDdequantizeValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(recv, norm, x, n);
    cudaStreamSynchronize(stream);
    
}*/

void GPUNUQSGDquantizeValue(signed char *x, float *y, float *norm, int n, curandState* states, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _NUQSGDquantizeValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(x, y, norm, n, states);
    cudaStreamSynchronize(stream);
    
}

void GPUNUQSGDdequantizeValue(signed char *recv, float *norm, float *x, int n, cudaStream_t stream)
{
    int blocksPerGrid = (int) ceil(1.0 * n / maxThreadsPerBlock);
    _NUQSGDdequantizeValue<<<blocksPerGrid, maxThreadsPerBlock, 0, stream>>>(recv, norm, x, n);
    cudaStreamSynchronize(stream);
    
}

