#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "biasd_cuda_integrate.h"
#include "biasd_cuda.h"
#include "cuda_help.h"

// ##########################################################
// ##########################################################

__global__ void kernel_loglikelihood(int N, double * d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * ll) {

	int idx = threadIdx.x + blockIdx.x*blockDim.x;

	if (idx < N) {
		double out, intval[2] = {0.,0.};
		theta t = {.ep1 = ep1, .ep2 = ep2, .sig1 = sigma1, .sig2 = sigma2, .k1 = k1, .k2 = k2, .tau = tau};

		// Calculate the state contributions
		out = k2/(k1+k2) / sigma1 * exp(-k1*tau - .5 * pow((d[idx]-ep1)/sigma1,2.)); // state 1
		out += k1/(k1+k2) / sigma2 * exp(-k2*tau - .5 * pow((d[idx]-ep2)/sigma2,2.)); // state 2

		// Perform the blurring integral
		adaptive_integrate(0.,1.,intval,epsilon,d[idx],&t);

		out += 2.*k1*k2/(k1+k2)*tau *intval[0]; // the blurring contribution
		out = log(out) - .5 * log(2.* M_PI); // Add prefactor
		ll[idx] = out; // transfer out result
	}
}

void load_data(int device, int N, double * d, double ** d_d, double ** ll_d){
	cudaSetDevice(device);
	int padding = get_padding(device,N);

	cudaMalloc((void**)d_d,N*sizeof(double));
	cudaMalloc((void**)ll_d,(N+padding)*sizeof(double)); // Make this bigger to pad to 1024

	cudaMemcpy(*d_d,d,N*sizeof(double),cudaMemcpyHostToDevice);


	double * ll;
	ll = (double *) malloc((N+padding)*sizeof(double));
	for (int i =0; i < N+padding;i++){
		ll[i] = 0.0;
	}
	cudaMemcpy(*ll_d,ll,(N+padding)*sizeof(double),cudaMemcpyHostToDevice);
	free(ll);

	//printf("load data\n");
	//printf("%p:%p: %f %f %f\n",d_d,*d_d,d[0],d[1],d[2]);
	//printf("loaded data\n");
	//double q = test_data(10,d_d);
	//q = test_data(10,ll_d);
}

void free_data(double **d_d, double **ll_d){
	//printf("free data\n");
	//double q = test_data(10,d_d);
	cudaFree(*d_d);
	cudaFree(*ll_d);
	//q = test_data(10,d_d);
}

void log_likelihood(int device, int N, double **d_d, double **ll_d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * ll) {

	//printf("log likelihood\n");
	//double q = test_data(N,d_d);
	//q = test_data(N,ll_d);

	// Sanity checks from the model
	if ((ep1 < ep2) && (sigma1 > 0.) && (sigma2 > 0.) && (k1 > 0.) && (k2 > 0.) && (tau > 0.) && (epsilon > 0.)) {

		// Initialize CUDA things
		int threads = 256;//deviceProp.maxThreadsPerBlock/8;
		int blocks = (N+threads-1)/threads;
		// Evaluate integrand at f -> store in ll.
		kernel_loglikelihood<<<blocks,threads>>>(N,(double*)*d_d,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon,(double*)*ll_d);
		cudaMemcpy(ll,*ll_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	} else {
		int i;
		for (i=0;i<N;i++){ ll[i] = -INFINITY;}
	}
}

double sum_log_likelihood(int device, int N, double **d_d, double **ll_d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon) {

	//printf("sum log likelihood\n");
	//double q = test_data(N,d_d);
	//q = test_data(N,ll_d);

	double sum = 0.;

	if ((ep1 < ep2) && (sigma1 > 0.) && (sigma2 > 0.) && (k1 > 0.) && (k2 > 0.) && (tau > 0.) && (epsilon > 0.)) {
		int padding = get_padding(device,N);
		int nSM = get_num_SM(device);

		int threads = 256;//deviceProp.maxThreadsPerBlock/8;
		int blocks = (N+threads-1)/threads;

		kernel_loglikelihood<<<blocks,threads>>>(N,(double*)*d_d,ep1,ep2,sigma1,sigma2,k1,k2,tau,epsilon,(double*)*ll_d);
		sum = parallel_sum(*ll_d,N+padding,nSM);
	} else {
		sum = -INFINITY;
	}
	return sum;
}

double test_data(int N, double **d_d){

	double * q;
	q = (double *) malloc(N*sizeof(double));
	cudaMemcpy(q,*d_d,N*sizeof(double),cudaMemcpyDeviceToHost);

	double sum = 0;
	for (int i=0;i<N;i++){
		sum += q[i];
	}

	printf("%p:%p: %f %f %f\n",d_d,*d_d,q[0],q[1],q[2]);

	free(q);
	return sum;
}

int device_count() {
	int count;
	cudaGetDeviceCount(&count);
	return count;
}

int py_cuda_errors(int device){
        for (int i=0;i<=device;i++){
                cudaSetDevice(i);
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                        return 0;
                }
        }
        return 1;
}
