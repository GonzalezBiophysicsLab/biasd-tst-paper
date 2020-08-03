#ifndef BIASD_LL_H_
#define BIASD_LL_H_

typedef struct integral_params { double d; double ep1; double ep2; double sig1; double sig2; double k1; double k2; double tau;} ip;

// #############################################################################
// #############################################################################


double bessel_i0(double x);
double bessel_i1(double x);
double integrand(double f,ip * p);
void adaptive_integrate(double a0, double b0, double *out, double epsilon, ip * p);
void qg10k21(double a, double b, double *out, ip * p);
void log_likelihood(int N, double * d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon, double * out);
double sum_log_likelihood(int N, double *d, double ep1, double ep2, double sigma1, double sigma2, double k1, double k2, double tau, double epsilon);

#endif
