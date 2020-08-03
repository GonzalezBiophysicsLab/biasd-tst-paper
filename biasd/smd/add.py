"""
Add a BIASD result to an HDF5 SMD file. This includes functions for `prior collections`, `posterior collections`, `Laplace posteriors`, `baseline corrections`, and `MCMC results`.

Input:
	* `parent` is group of the HDF5 object where the analysis data will be added as a new group
	* `the next argument` is the data object that will be added
	* `label` is a string that the new group will be called. Defaults to descriptive label.

"""

from .smd_hdf5 import _addhash
import numpy as _np

def parameter_collection(parent,collection,label='BIASD parameter collection'):
	"""
	Add a `parameter_collection` from `biasd.distributions` to HDF5 group
	"""

	l = ['e1','e2','sigma','k1','k2']
	names = [d.name for d in [collection.e1,collection.e2,collection.sigma,collection.k1,collection.k2]]
	params = [d.parameters.tolist() for d in [collection.e1,collection.e2,collection.sigma,collection.k1,collection.k2]]

	group = parent.create_group(label)
	_addhash(group)
	group.attrs['description'] = 'BIASD parameter collection'
	for i in range(5):
		group.attrs[l[i]+' distribution type'] = names[i]
		group.attrs[l[i]+' parameters'] = _np.array(params[i])

def laplace_posterior(parent,laplace_posterior,label='result'):
	"""
	Add a `_laplace_posterior` from `biasd.laplace` to HDF5 group
	"""

	group = parent.create_group(label)
	_addhash(group)
	group.attrs['description'] = 'BIASD Laplace posterior'
	group.attrs["mu"] = laplace_posterior.mu
	group.attrs["covar"] = laplace_posterior.covar

def baseline(parent,params,label='result'):
	"""
	Add a `_params` result from `biasd.utils.baseline` to HDF5 group
	"""
	# pi,mu,var,r,baseline,R2,ll,iter

	group = parent.create_croup(label)
	_addhash(group)
	group.attrs['description'] = 'White-noise baseline correction parameters'
	group.attrs['pi'] = params.pi
	group.attrs['mu'] = params.mu
	group.attrs['var'] = params.var
	group.create_dataset('r', data = params.r)
	group.create_dataset('baseline', data = params.baseline)
	group.attrs['R2'] = params.R2
	group.attrs['log likelihood'] = params.ll
	group.attrs['iterations'] = params.iter


def mcmc(parent,mcmc_result,label='result'):
	"""
	Add a `mcmc_result` from `biasd.mcmc` to a HDF5 group
	"""
	# acor ,chain,lnprobability,iterations,naccepted,nwalkers,dim

	group = parent.create_group(label)
	_addhash(group)
	group.attrs['description'] = 'BIASD MCMC result'
	group.attrs['autocorrelation times'] = mcmc_result.acor
	group.create_dataset('sampler chain', data = mcmc_result.chain)
	group.create_dataset('log probability', data = mcmc_result.lnprobability)
	group.attrs['iterations'] = mcmc_result.iterations
	group.attrs['number accepted'] = mcmc_result.naccepted
	group.attrs['number of walkers'] = mcmc_result.nwalkers
	group.attrs['number of data dimensions'] = mcmc_result.dim

def kmeans(parent,kmeans_result,label='result'):
	"""
	Add a `_results_kmeans` from `biasd.distributions` to a HDF5 Group
	"""

	group = parent.create_group(label)
	_addhash(group)
	group.attrs['description'] = 'K-means result'
	group.attrs['nstates'] = kmeans_result.nstates
	group.attrs['pi'] = kmeans_result.pi
	group.create_dataset('responsibilities', data = kmeans_result.r)
	group.attrs['mu'] = kmeans_result.mu
	group.attrs['var'] = kmeans_result.var

def trajectories(f,x,y,x_label='Time',y_label='Signal'):
	'''
	Simple add for homogeneous-length signal versus time trajectories to an HDF5 file for use in BIASD.
	Input:
		* `f` is for an already open h5py `File` -- create one with smd.new(filename), and open it with smd.open(filename)
		* `x` is a (T) numpy `ndarray`
		* `y` is a (N,T) numpy `ndarray`
	where `N` is the number of molecules, and `T` is the number of datapoints

	Trajectories are added as groups called \`trajectory i\`, where `i` is an integer. Also, adds the number of trajectories in the file as an attribute to the `File`. This can be accessed as `f.attrs['number of trajectories]`. Also, adds the number of datapoints in these trajectories as `f.attrs['number of datapoints]`.
	'''

	# Check to see how many trajectories already exist
	if not list(f.attrs.keys()).count('number of trajectories'):
		f.attrs['number of trajectories'] = 0
	nstart = f.attrs['number of trajectories']

	# Shape the incoming data into NxT
	if y.ndim == 1:
		y = y[None,:]
	n,t = y.shape

	if x is None:
		x = np.arange(t)

	# Add the incoming data trajectory by trajectory
	for i in range(nstart,nstart+n):
		group = f.create_group('trajectory %05d'%(i))
		_addhash(group)

		g = group.create_group('data')
		g.attrs['number of datapoints'] = t
		_addhash(g)

		dset = g.create_dataset(x_label,data = x)
		_addhash(dset)
		dset = g.create_dataset(y_label,data = y[i-nstart])
		_addhash(dset)

	f.attrs['number of trajectories'] = len(list(f.keys()))
