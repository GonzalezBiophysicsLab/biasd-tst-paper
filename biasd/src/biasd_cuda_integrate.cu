#include <stdio.h>
#include <cuda.h>
#include "biasd_cuda_integrate.h"

/*
	Parameters ripped from Quadpack dqk21.f on 3/20/2017:
	c gauss quadrature weights and kronron quadrature abscissae and weights
	c as evaluated with 80 decimal digit arithmetic by l. w. fullerton,
	c bell labs, nov. 1981.
*/

__constant__ static double wg[5] = {
	0.066671344308688137593568809893332,
	0.149451349150580593145776339657697,
	0.219086362515982043995534934228163,
	0.269266719309996355091226921569469,
	0.295524224714752870173892994651338
};
__constant__ static double wgk[11] = {
	0.011694638867371874278064396062192,
	0.032558162307964727478818972459390,
	0.054755896574351996031381300244580,
	0.075039674810919952767043140916190,
	0.093125454583697605535065465083366,
	0.109387158802297641899210590325805,
	0.123491976262065851077958109831074,
	0.134709217311473325928054001771707,
	0.142775938577060080797094273138717,
	0.147739104901338491374841515972068,
	0.149445554002916905664936468389821
};
__constant__ static double xgk[11] = {
	0.995657163025808080735527280689003,
	0.973906528517171720077964012084452,
	0.930157491355708226001207180059508,
	0.865063366688984510732096688423493,
	0.780817726586416897063717578345042,
	0.679409568299024406234327365114874,
	0.562757134668604683339000099272694,
	0.433395394129247190799265943165784,
	0.294392862701460198131126603103866,
	0.148874338981631210884826001129720,
	0.000000000000000000000000000000000
};


// ##########################################################
// ##########################################################

__device__ double integrand(double f, double d, theta *t) {
	double y = 2.*(*t).tau*sqrt((*t).k1*(*t).k2*f*(1.-f));
	double var = (*t).sig1 * (*t).sig1 * f + (*t).sig2 * (*t).sig2 * (1.-f);

	return exp(-((*t).k1*f+(*t).k2*(1.-f))*(*t).tau)/ sqrt(var)
		* exp(-.5*pow((d - ((*t).ep1 * f + (*t).ep2 * (1.-f))),2.)/var)
		* (cyl_bessel_i0(y) + ((*t).k2*f+(*t).k1*(1.-f))*(*t).tau
			* cyl_bessel_i1(y)/y); // Full Expression
}

__device__ void qg10k21(double a, double b, double *out, double d, theta * t) {
	// Do quadrature with the Gauss-10, Kronrod-21 rule
	// translated from Fortran from Quadpack's dqk21.f...

	double center = 0.5*(b+a);
	double halflength = 0.5*(b-a);

	double result10 = 0.;
	double result21 = 0.;

	int i;
	// Pre-calculate function values
	double fval1[10],fval2[10];
	for (i=0;i<10;i++){
		fval1[i] = integrand(center+halflength*xgk[i],d,t);
		fval2[i] = integrand(center-halflength*xgk[i],d,t);
	}

	for (i=0;i<5;i++){
		// Evaluate Gauss-10
		result10 += wg[i]*(fval1[2*i+1] + fval2[2*i+1]);

		//Evaluate Kronrod-21
		result21 += wgk[2*i]*(fval1[2*i] + fval2[2*i]);
		result21 += wgk[2*i+1]*(fval1[2*i+1] + fval2[2*i+1]);
	}

	// Add 0 point to Kronrod-21
	double fc = integrand(center,d,t);
	result21 += wgk[10]*fc;

	// Scale results to the interval
	result10 *= fabs(halflength);
	result21 *= fabs(halflength);

	// Error calculation
	double avgg = result21/fabs(b-a), errors = 0.;
	for (i=0;i<5;i++){
		errors += wgk[2*i]*(fabs(fval1[2*i]-avgg)+fabs(fval2[2*i]-avgg));
		errors += wgk[2*i+1]*(fabs(fval1[2*i+1]-avgg)+fabs(fval2[2*i+1]-avgg));
	}
	errors += wgk[10]*(fabs(fc-avgg));
	errors *= min(1.,pow(200.*fabs(result21-result10)/errors,1.5));

	// Output results
	out[0] += result21;
	out[1] += errors;
}

__device__ void adaptive_integrate(double a0, double b0, double *out, double epsilon, double d, theta *t){

	// a is the lowerbound of the integral
	// b is the upperbound of the integral
	// out[0] is the value of the integral
	// out[1] is the error of the integral
	// epsilon is the error limit we must reach in each sub-interval

	double current_a = a0, current_b = b0, temp_out[2];

	// This loops from the bottom to the top, subdividing and trying again on the lower half, then moving up once converged to epsilon
	while (current_a < b0){
		// Evaluate quadrature on current interval
		temp_out[0] = 0.;
		temp_out[1] = 0.;
		// Perform the quadrature
		qg10k21(current_a,current_b,temp_out,d,t);
		// Check convergence on this interval
		if (temp_out[1] < epsilon ){
			// keep the results
			out[0] += temp_out[0];
			out[1] += temp_out[1];
			// step the interval
			current_a = current_b;
			current_b = b0;
		} else { // if the interval failed to converge
			// sub-divide the interval
			current_b -= 0.5*(current_b-current_a);
		}
	}
}
