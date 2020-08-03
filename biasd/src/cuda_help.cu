#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define WARP_SIZE 32

int device_count(void) {
	int count;
	cudaGetDeviceCount(&count);
	return count;
}

int cuda_errors(int device){
	cudaSetDevice(device);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("CUDA Error : %s\n", cudaGetErrorString(err));
		return 1;
	}
	return 0;
}


// #if __CUDA_ARCH__ < 600
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
__device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull =(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,__double_as_longlong(val + __longlong_as_double(assumed)));

	// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __longlong_as_double(old);
}

#endif

#if CUDA_VERSION < 9000
	__device__ double __shfl_xor_sync(unsigned mask, double var, int laneMask, int width=warpSize){
		return __shfl_xor(mask,var,laneMask);
	}
#endif


/*
The following is from:
https://github.com/jordanbonilla/Cuda-Example
*/


/* Use all GPU Streaming Multiprocessors to add elements in parallel.
	 Requies that the number of elements is a multiple of #SMs * 1024
	 since the algorithm processes elements in chunks of this size.
	 This is taken care of in "cuda_parallel_sum which pads zeros. */
__global__ void
cuda_parallel_sum(double *in, int num_elements, double *sum) {
		//Holds intermediates in shared memory reduction
		__syncthreads();
		__shared__ double buffer[WARP_SIZE];
		int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
		int lane = threadIdx.x % WARP_SIZE;
		double temp;
		while(globalIdx < num_elements) {
			// All threads in a block of 1024 take an element
				temp = in[globalIdx];
				// All warps in this block (32) compute the sum of all
				// threads in their warp
				for(int delta = WARP_SIZE/2; delta > 0; delta /= 2) {
						 temp+= __shfl_xor_sync(0xffffffff,temp, delta);
				}
				// Write all 32 of these partial sums to shared memory
				if(lane == 0) {
						buffer[threadIdx.x / WARP_SIZE] = temp;
				}
				__syncthreads();
				// Add the remaining 32 partial sums using a single warp
				if(threadIdx.x < WARP_SIZE) {
						temp = buffer[threadIdx.x];
						for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2) {
								temp += __shfl_xor_sync(0xffffffff,temp, delta);
						}
				}
				// Add this block's sum to the total sum
				if(threadIdx.x == 0) {
						atomicAdd(sum, temp);
				}
				// Jump ahead 1024 * #SMs to the next region of numbers to sum
				globalIdx += blockDim.x * gridDim.x;
				__syncthreads();
		}
}

double parallel_sum(void * a_d, int N, int num_SMs) {
	// a_d is a pointer to an array already on the GPU
	// a_d must already be padded to a multiple of 1024

	// Result
	double result = 0.0;
	double * result_d;
	cudaMalloc( (void**) &result_d, sizeof(double) );
	cudaMemcpy( result_d, &result, sizeof(double), cudaMemcpyHostToDevice );

	// Call kernel to get sum
	cuda_parallel_sum<<<num_SMs , 1024 >>>((double*)a_d, N, result_d);

	cudaMemcpy( &result, result_d, sizeof(double), cudaMemcpyDeviceToHost );
	cudaFree(result_d);

	return result;
}


int get_num_SM(int device) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	int num_SMs = prop.multiProcessorCount;
	return num_SMs;
}


int get_padding(int device, int N) {
	int num_SMs = get_num_SM(device);
	int batch_size = num_SMs * 1024;
	int padding = (batch_size - (N % batch_size)) % batch_size;
	return padding;
}
