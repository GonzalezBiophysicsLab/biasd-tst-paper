#ifndef BIASD_2S_GK_H_
#define BIASD_2S_GK_H_

typedef struct {
	double ep1, ep2, sig1, sig2, k1, k2, tau;
} theta;

// BIASD related functions
__device__ double integrand(double f, double d, theta *t);

// Adaptive Gauss-10, Kronrod-21 quadrature
__device__ void qg10k21(double a, double b, double *out, double d, theta * t);
__device__ void adaptive_integrate(double a0, double b0, double *out, double epsilon, double d, theta *t);

#endif
