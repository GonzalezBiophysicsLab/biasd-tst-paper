"""
.. module:: laplace
	:synopsis: Contains function to calculate the laplace approximation to the BIASD posterior probability distribution.

"""

import numpy as np
from scipy.optimize import minimize
from biasd.likelihood import log_posterior
from biasd.distributions import parameter_collection,normal,convert_distribution

def calc_hessian(fxn,x,eps = np.sqrt(np.finfo(np.float64).eps)):
	"""
	Calculate the Hessian using the finite difference approximation.

	Finite difference formulas given in Abramowitz & Stegun

		- Eqn. 25.3.24 (on-diagonal)
		- Eqn. 25.3.27 (off-diagonal)

	Input:
		* `fxn` is a function that can be evaluated at x
		* `x` is a 1D `np.ndarray`

	Returns:
		* an NxN `np.ndarray`, where N is the size of `x`

	"""

	# Notes:
	# xij is the position to evaluate the function at
	# yij is the function evaluated at xij
	#### if i or j = 0, it's the starting postion
	#### 1 or m1 are x + 1.*eps and x - 1.*eps, respetively


	h = np.zeros((x.size,x.size))
	y00 = fxn(x)

	for i in range(x.size):
		for j in range(x.size):
			#Off-diagonals below the diagonal are the same as those above.
			if j < i:
				h[i,j] = h[j,i]
			else:
				#On-diagonals
				if i == j:
					x10 = x.copy()
					xm10 = x.copy()
					x20 = x.copy()
					xm20 = x.copy()

					x10[i] += eps
					xm10[i] -= eps
					x20[i] += 2*eps
					xm20[i] -= 2*eps

					y10 = fxn(x10)
					ym10 = fxn(xm10)
					y20 = fxn(x20)
					ym20 = fxn(xm20)

					h[i,j] = eps**(-2.)/12. * (-y20 + 16.* y10 - 30.*y00 +16.*ym10 - ym20)

				#Off-diagonals above the diagonal
				elif j > i:
					x10 = x.copy()
					xm10 = x.copy()
					x01 = x.copy()
					x0m1 = x.copy()
					x11 = x.copy()
					xm1m1 = x.copy()

					x10[i] += eps
					xm10[i] -= eps
					x01[j] += eps
					x0m1[j] -= eps
					x11[i] += eps
					x11[j] += eps
					xm1m1[i] -= eps
					xm1m1[j] -= eps

					y10 = fxn(x10)
					ym10 = fxn(xm10)
					y01 = fxn(x01)
					y0m1 = fxn(x0m1)
					y11 = fxn(x11)
					ym1m1 = fxn(xm1m1)

					h[i,j] = -1./(2.*eps**2.) * (y10 + ym10 + y01 + y0m1 - 2.*y00 - y11 - ym1m1)
	return h

class _laplace_posterior:
	"""
	Holds the results of a Laplace approximation of the posterior probability distribution from BIASD
	"""
	def __init__(self,mean,covar,prior=None):
		self.mu = mean
		self.covar = covar
		self.posterior = parameter_collection(*[normal(m,s) for m,s in zip(self.mu,self.covar.diagonal()**.5)])

	def transform(self,prior):
		self.posterior.e1 = convert_distribution(self.posterior.e1,prior.e1.name)
		self.posterior.e2 = convert_distribution(self.posterior.e2,prior.e2.name)
		self.posterior.sigma = convert_distribution(self.posterior.sigma,prior.sigma.name)
		self.posterior.k1 = convert_distribution(self.posterior.k1,prior.k1.name)
		self.posterior.k2 = convert_distribution(self.posterior.k2,prior.k2.name)

	def samples(self,n):
		return np.random.multivariate_normal(self.mu,self.covar,n)

def _min_fxn(theta,data,prior,tau,device):
	return -1.*log_posterior(theta,data,prior,tau,device)
def _minimizer(inputt):
	data,prior,tau,x0,meth,device = inputt
	mind =  minimize(_min_fxn,x0,method=meth,args=(data,prior,tau,device), tol = np.sqrt(np.finfo(np.float).eps))
	return mind

def find_map(data,prior,tau,meth='nelder-mead',xx=None,nrestarts=2,threads=1,device=0):
	'''
	Use numerical minimization to find the maximum a posteriori estimate of a BIASD log-posterior distribution.

	Inputs:
		* `data` is a 1D `np.ndarray` of the time series
		* `prior` is a `biasd.distributions.parameter_collection` that contains the prior the BIASD Bayesian inference
		* `tau` is the measurement period

	Optional:
		* `meth` is the minimizer used to find the minimum of the negative posterior (i.e., the maximum). Defaults to simplex.
		* `xx` will initialize the minimizer at this theta position. Defaults to mean of the priors.
		* `nrestarts` is the number of times to try to find the minimum. Restarts initialize at a random variate chosen from prior distributions
		* `threads` is the number of threads to run the restarts in parallel with (if > 1)

	Returns:
		* the minimizer dictionary
	'''

	#If no xx, start at the mean of the priors
	if not isinstance(xx,np.ndarray):
		xx = [prior.mean()]
	else:
		xx = [xx]
	xx.extend([prior.rvs(1).flatten() for _ in range(nrestarts-1)])


	for i in np.arange(len(xx)):
		if xx[i][0] > xx[i][1]:
			#print xx[i]
			temp = xx[i][0].copy()
			xx[i][0] = xx[i][1].copy()
			xx[i][1] = temp

	if threads > 1:
		import multiprocessing as mp
		p = mp.Pool(threads)
		ylist = p.map(_minimizer,[[data,prior,tau,xx[i],meth,device] for i in range(nrestarts)])
		p.close()
	else:
		ylist =   list(map(_minimizer,[[data,prior,tau,xx[i],meth,device] for i in range(nrestarts)]))

	#Select the best MAP estimate
	ymin = np.inf
	for i in ylist:
		if i['success']:
			if i['fun'] < ymin:
				ymin = i['fun']
				y = i
	#If no MAP estimates, return None
	if ymin == np.inf:
		y = None
	return y

def laplace_approximation(data,prior,tau,nrestarts=2,verbose=False,threads=1,device=0):
	'''
	Perform the Laplace approximation on the BIASD posterior probability distribution of this trace.

	Inputs:
		* `data` is a 1D `np.ndarray` of the time series
		* `prior` is a `biasd.distributions.parameter_collection` that contains the prior the BIASD Bayesian inference
		* `tau` is the measurement period

	Optional:
		* `nrestarts` is the number of times to try to find the MAP in `find_map`.
		* `verbose` is a boolean that determines whether the evaluation time for each Hessian and minimization is printed.
		* `threads` is the number of threads to run the restarts in parallel with (if > 1) - don't do this with the GPU...

	Returns:
		* a `biasd.laplace._laplace_posterior` object, which has a `.mu` with the means, a `.covar` with the covariances, and a `.posterior` which is a marginalized `biasd.distributions.parameter_collection` of normal distributions.
	'''

	#Calculate the best MAP estimate
	import time
	t0 = time.time()
	mind = find_map(data,prior,tau,nrestarts=nrestarts,threads=threads,device=device)
	t1 = time.time()
	if verbose:
		print(t1-t0)

	if not mind is None:
		#Calculate the Hessian at MAP estimate
		if mind['success']:
			mu = mind['x']
			feps = np.sqrt(np.finfo(np.float).eps)
			feps *= 8. ## The interval is typically too small
			t0 = time.time()
			hessian = calc_hessian(lambda theta: log_posterior(theta,data,prior,tau), mu,eps=feps)
			print(hessian)
			t1 = time.time()
			if verbose:
				print(t1-t0)

			#Ensure that the hessian is positive semi-definite by checking that all eigenvalues are positive
			#If not, expand the value of machine error in the hessian calculation and try again
			try:
				#Check eigenvalues, use pseudoinverse if ill-conditioned
				var = -np.linalg.inv(hessian)

				#Ensuring Hessian(variance) is stable
				new_feps = feps*2.
				new_hess = calc_hessian(lambda theta: log_posterior(theta,data,prior,tau), mu,eps= new_feps)
				new_var = -np.linalg.inv(new_hess)
				it = 0

				while np.any(np.abs(new_var-var)/var > 1e-2):
					new_feps *= 2
					var = new_var.copy()
					new_hess = calc_hessian(lambda theta: log_posterior(theta,data,prior,tau), mu,eps= new_feps)
					new_var = -np.linalg.inv(new_hess)
					it +=1
					# 2^26 times feps = 1. Arbitrary upper-limit, increase if necessary (probably not for BIASD)
					if it > 25:
						raise ValueError('Whelp, you hit the end there. bud')
				# print('Hessian iterations')
				# print(np.log2(new_feps/feps), it)

				#Ensure symmetry of covariance matrix if witin machine error
				if np.allclose(var,var.T):
					n = var.shape[0]
					var = np.tri(n,n,-1)*var+(np.tri(n,n)*var).T
					return _laplace_posterior(mu,var)

			#If this didn't work, return None
			except np.linalg.LinAlgError:
				raise ValueError("Wasn't able to calculate the Hessian")

	raise ValueError("No MAP estimate")
	return _laplace_posterior(None,None)
