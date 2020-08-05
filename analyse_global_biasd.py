import numpy as np
import biasd as b
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py as h

fname = "experimental_data/RC_fMet_l1l9_100ms.hdf5"
f = h.File(fname, 'r')
d = []
for i in f.keys():
	temp = f[i][()]
	temp[np.where(temp > 2)] = 2.
	temp[np.where(temp < -1)] = -1.
	d.append(temp)
	print(i,' : ',temp.size)


e1 = b.distributions.normal(0.3545, 0.0001)
e2 = b.distributions.normal(0.5840, 0.0001)
sigma = b.distributions.uniform(0.0001, 0.15)
H1_prior = b.distributions.uniform(-500000., 500000.)
S1_prior = b.distributions.uniform(-1000., 1000.)
H2_prior = b.distributions.uniform(-500000., 500000.)
S2_prior = b.distributions.uniform(-1000., 1000.)

# The following two priors are not used but included for the completeness of the
# parameter collection used in normal BIASD

k1 = b.distributions.uniform(0., 6.)
k2 = b.distributions.uniform(0., 6.)

priors = b.distributions.parameter_collection(e1, e2, sigma, k1, k2)
E_priors = [H1_prior, S1_prior, H2_prior, S2_prior]

T = np.array([25,28,31,34,37]) + 273.

tau = 0.1
try:
	b.likelihood.use_cuda_glob_ll()
except:
	pass
print(b.likelihood.ll_version)
nwalkers = 1000

sampler, initial_positions = b.glob_mcmc.setup(d, T, priors, E_priors, tau, nwalkers, threads = 1)

for i,result in enumerate(tqdm(sampler.sample(initial_positions,iterations=3000))): pass

lx = sampler.lnprobability[-1,:].argsort()
better_positions = sampler.chain[lx,-1][-int(nwalkers/2):]

oname0 = 'RC_fMet_l1l9_100ms_glob_init.dat'
np.savetxt(oname0, better_positions, delimiter = ',')


sampler = b.glob_mcmc.setup(d, T, priors, E_priors, tau,int(nwalkers/2), threads = 1)[0]

better_positions =  np.loadtxt('RC_fMet_l1l9_100ms_glob_init.dat', delimiter = ',')
for i,result in enumerate(tqdm(sampler.sample(better_positions,iterations=1000))): pass
burned_positions = sampler.chain[:,-1].copy()
sampler.reset()
for i,result in enumerate(tqdm(sampler.sample(burned_positions,iterations=5000))): pass

uncorrelated_samples = b.glob_mcmc.get_samples(sampler,int(nwalkers/2),uncorrelated=True)

maxim = np.argmax(sampler.lnprobability[-1,:])

zs = sampler.chain[maxim, -1, :]
print(zs)
sds =  np.std(uncorrelated_samples, axis = 0)

oname = 'RC_fMet_l1l9_100ms_glob_modes.dat'
x = np.array([zs, sds])
np.savetxt(oname, x, delimiter = ',')

oname1 = 'RC_fMet_l1l9_100ms_glob_mcmc_samples.dat'
np.savetxt(oname1, uncorrelated_samples, delimiter = ',')
