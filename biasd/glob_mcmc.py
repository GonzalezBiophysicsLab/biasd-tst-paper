'''
.. module:: glob_mcmc
	:synopsis: Integrates emcee's MCMC into BIASD for global analysis

'''

import numpy as _np
import emcee
from time import time as _time

def setup(data, T, priors, E_priors, tau, nwalkers, initialize='rvs', threads=1,device=0):
	"""
	Prepare the MCMC sampler

	Input:
		* `data` is a `np.ndarray` of the time series
		* `priors` is a `biasd.distributions.parameter_collection` of the priors
		* `tau` is the measurement period each data point
		* `nwalkers` is the number of walkers in the MCMC ensemble. The more the better
		* `initialze` =
			- 'rvs' will initialize the walkers at a random spot chosen from the priors
			- 'mean' will initialize the walkers tightly clustered around the mean of the priors.
			- an (`nwalkers`,5) `np.ndarray` of whatever spots you want to initialize the walkers at.
		* `threads` is the number of threads to use for evaluating the log-posterior of the walkers. Be careful when using the CUDA log-likelihood function, because you'll probably be bottle-necked there.

	Results:
		* An `emcee` sampler object. Please see the `emcee` documentation for more information.
	"""

	#from biasd.likelihood import log_global_posterior, load_cuda_glob
	import biasd.likelihood as bl

	ndim = 7
	u = [0,1,2]

	if isinstance(initialize,_np.ndarray) and initialize.shape == (nwalkers,7):
		initial_positions = initialize

	elif initialize == 'rvs':

		H1 = E_priors[0].rvs(nwalkers).flatten()
		S1 = (H1 - 71400.)/300.
		H2 = E_priors[2].rvs(nwalkers).flatten()
		S2 = (H2 - 71400.)/300.

		initial_positions = _np.array([_np.concatenate((priors.rvs(1).flatten()[u], _np.array([H1[i], S1[i], H2[i], S2[i]]))) for i in range(nwalkers)])
	elif initialize == 'mean':

		H1 = E_priors[0].mean()
		S1 = (H1 - 71400.)/300.
		H2 = E_priors[2].mean()
		S2 = (H2 - 71400.)/300.

		initial_positions = _np.array([_np.concatenate((_np.array([p.mean() for p in priors]).flatten()[u], _np.array([H1,S1,H2,S2])), axis = 0) for _ in range(nwalkers)])

	else:
		raise AttributeError('Could not initialize the walkers. Try calling with initialize=\'rvs\'')

	# Slap-dash hackery to make sure the first E_fret is the lower one
	for i in range(initial_positions.shape[0]):
		if initial_positions[i,0] > initial_positions[i,1]:
			temp = initial_positions[i,0]
			initial_positions[i,0] = initial_positions[i,1]
			initial_positions[i,1] = temp

	if bl.ll_version == "CUDA (Global)":
		bl.load_cuda_glob(data)

	sampler = emcee.EnsembleSampler(nwalkers, ndim, bl.log_global_posterior, args=[data,T,priors,E_priors,tau,device],threads=threads)

	return sampler,initial_positions

def burn_in(sampler,positions,nsteps=100,timer = True):
	"""
	Burn-in will run some MCMC, getting new positions, and then reset the sampler so that nothing has been sampled.

	Input:
		* `sampler` is an `emcee` sampler
		* `positions` is the starting walker positions (maybe provided by `biasd.mcmc.setup`?)
		* `nsteps` is the integer number of MCMC steps to take
		* `timer` is a boolean for displaying the timing statistics

	Results:
		* `sampler` is now a cleared `emcee` sampler where no steps have been made
		* `positions` is an array of the final walkers positions for use when starting a more randomized sampling

	"""
	sampler = run(sampler,positions,nsteps,timer)
	positions = _np.copy(sampler.chain[:,-1,:])
	sampler.reset()
	return sampler,positions

def run(sampler,positions,nsteps,timer=True):
	"""
	Acquire some MCMC samples, and keep them in the sampler

	Input:
		* `sampler` is an `emcee` sampler
		* `positions` is the starting walker positions (maybe provided by `biasd.mcmc.setup`?)
		* `nsteps` is the integer number of MCMC steps to take
		* `timer` is a boolean for displaying the timing statistics

	Results:
		* `sampler` is the updated `emcee` sampler

	"""

	t0 = _time()
	sampler.run_mcmc(positions,nsteps)
	t1 = _time()
	if timer:
		print("Steps: ", sampler.chain.shape[1])
		print("Total Time:",(t1-t0))
		print("Time/Sample:",(t1-t0)/sampler.flatchain.shape[0]/sampler.args[0].size)
	return sampler

def continue_run(sampler,nsteps,timer=True):
	"""
	Similar to `biasd.mcmc.run`, but you do not need to specify the initial positions, because they will be the last sampled positions in `sampler`
	"""
	positions = sampler.chain[:,-1,:]
	sampler = run(sampler,positions,nsteps,timer=timer)
	return sampler


def chain_statistics(sampler,verbose=True):
	"""
	Calculate the acceptance fraction and autocorrelation times of the samples in `sampler`
	"""
	# Chain statistics
	if verbose:
		print("Mean acceptance fraction:", _np.mean(sampler.acceptance_fraction))
		print("Autocorrelation time:", sampler.get_autocorr_time())
	maxauto = _np.int(sampler.get_autocorr_time().max())+1
	return maxauto

def get_samples(sampler,nwalkers,uncorrelated=True,culled=False):
	"""
	Get the samples from `sampler`

	Input:
		* `sampler` is an `emcee` sampler with samples in it
		* `uncorrelated` is a boolean for whether to provide all the samples, or every n'th sample, where n is the larges autocorrelation time of the dimensions.
		* `culled` is a boolean, where any sample with a log-probability less than 0 is removed. This is necessary because sometimes a few chains get very stuck, and their samples (not being representative of the posterior) mess up subsequent plots.

	Returns:
		An (N,7) `np.ndarray` of samples from the sampler
	"""

	index = _np.argmax(nwalkers == _np.array(sampler.lnprobability.T.shape))
	#index corresponds to nsteps. lnprobability is a 2-D array of nsteps and nwalkers.
	#The dimension which corresponds to nwalkers in the transpose corresponds to nsteps
	#(i.e. not walkers) in the original. Hey, it works!

	if uncorrelated:
		maxauto = chain_statistics(sampler,verbose=False)
	else:
		maxauto = 1
	if culled:
		cut = sampler.lnprobability.mean(index) < 0.
	else:
		cut = sampler.lnprobability.mean(index) < -_np.inf
	samples = sampler.chain[~cut,::maxauto,:].reshape((-1,7))
	return samples

def plot_corner(samples):
	"""
	Use the python package called corner <https://github.com/dfm/corner.py> to make some very nice corner plots (joints and marginalized) of posterior in the 5-dimensions used by the two-state BIASD posterior.

	Input:
		* `samples` is a (N,7) `np.ndarray`
	Returns:
		* `fig` which is the handle to the figure containing the corner plot
	"""

	import corner
	labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\sigma_1$', r'$k_1$', r'$k_2$', r'$\varepsilon_1$', r'$\varepsilon_2$', r'$\sigma_1$', r'$kp_1$', r'$kp_2$',r'$q$']
	fig = corner.corner(samples, labels=labels, quantiles=[.025,.50,.975],levels=(1-_np.exp(-0.5),))
	return fig

def create_posterior_collection(samples,priors):
	"""
	Take the MCMC samples, marginalize them, and then calculate the first and second moments. Use these to moment-match to the types of distributions specified for each dimension in the priors. For instance, if the prior for :math:`\\epsilon_1` was beta distributed, this will moment-match the posterior to as a beta distribution.

	Input:
		* `samples` is a (N,5) `np.ndarray`
		* `priors` is a `biasd.distributions.parameter_collection` that provides the distribution-forms to moment-match to
	Returns:
		* A `biasd.distributions.parameter_collection` containing the marginalized, moment-matched posteriors
	"""

	from biasd.distributions import parameter_collection
	#Moment-match, marginalized posteriors
	first = samples.mean(0)
	second = _np.var(samples,axis=0)+first**2

	e1 = priors.e1.new(_np.around(priors.e1._moment2param_fxn(first[0], second[0]),4))
	e2 = priors.e2.new(_np.around(priors.e2._moment2param_fxn(first[1], second[1]),4))
	sigma = priors.sigma.new(_np.around(priors.sigma._moment2param_fxn(first[2], second[2]),4))
	k1 = priors.k1.new(_np.around(priors.k1._moment2param_fxn(first[3], second[3]),4))
	k2 = priors.k2.new(_np.around(priors.k2._moment2param_fxn(first[4], second[4]),4))

	return parameter_collection(e1,e2,sigma,k1,k2)

class mcmc_result(object):
	"""
	Holds the results of a MCMC sampler of the posterior probability distribution from BIASD
	Input:
		* `mcmc_input` is either an `emcee.sampler.Sampler` or child, or a list of `[acor, chain, lnprobability, iterations, naccepted, nwalkers, dim]`
	"""
	def __init__(self, mcmc_input):
		try:
			if 'lnprobfn' in mcmc_input.__dict__:
				if 'acor' not in mcmc_input.__dict__:
					mcmc_input.get_autocorr_time()
				self.acor = mcmc_input.acor
				self.chain = mcmc_input.chain
				self.lnprobability = mcmc_input.lnprobability
				self.iterations = mcmc_input.iterations
				self.naccepted = mcmc_input.naccepted
				self.nwalkers = mcmc_input.k
				self.dim = mcmc_input.dim
				return
		except:
			pass
		try:
			self.acor, self.chain, self.lnprobability, self.iterations, self.naccepted,self.nwalkers,self.dim = mcmc_input
			return
		except:
			pass
		raise Exception("Couldn't initialize mcmc_result")
