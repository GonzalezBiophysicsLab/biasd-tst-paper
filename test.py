import numpy as np
import biasd as b
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize

fname = "/home/korak/ec2+lys_rt_33ms_combined_biasd.dat"
data = np.loadtxt(fname, delimiter = ',')
data[np.where(data > 2)] = 2.
data[np.where(data < -1)] = -1.
d = data[:200]

e1 = b.distributions.normal(0.15, 0.2)
e2 = b.distributions.normal(0.75, 0.2)
sigma = b.distributions.gamma(10., 10./0.08)
k1 = b.distributions.gamma(1./0.5, 0.3/0.5)
k2 = b.distributions.gamma(1./0.5, 0.3/0.5)
k1_prime = b.distributions.gamma(1/0.5, 0.3/0.5)
k2_prime = b.distributions.gamma(1./0.5, 0.3/0.5)

priors1 = b.distributions.parameter_collection(e1, e2, sigma, k1, k2)
priors2 = b.distributions.parameter_collection(e1, e2, sigma, k1_prime, k2_prime)

priors = [priors1, priors2]

pop_prior = b.distributions.dirichlet((80.,20.,1.))

tau = 0.033

posterior = b.mix_laplace.laplace_approximation(data, priors,pop_prior, tau,nrestarts = 10)

print posterior.mu

nwalkers = 200

sampler, initial_positions = b.mix_mcmc.setup(data, priors, pop_prior, tau, nwalkers, threads = 1)
sampler, burned_positions = b.mix_mcmc.burn_in(sampler,initial_positions,nsteps=100)
for i,result in enumerate(tqdm(sampler.sample(burned_positions,iterations=1000))): pass

plt.plot(sampler.lnprobability.T, alpha = 0.1)
plt.savefig('walker.png')
#print sampler.lnprobability

c = np.where(np.isfinite(sampler.lnprobability[:,-1].flatten()))

log_weights = sampler.lnprobability[c,-1].flatten() - np.max(sampler.lnprobability[c,-1])

weights = np.exp(log_weights)

xs = sampler.chain[c, -1].copy().reshape(len(c[0]), 7)

x_bar = np.nansum(weights[:, None]*xs, axis = 0)/np.nansum(weights)

print 'Mean0:'
print x_bar

maxim = np.argmax(sampler.lnprobability[c,-1])

#np.savetxt('data.dat', 'Mean')
#np.savetxt('data.dat', x_bar, delimiter = ' ')



finite_zs = sampler.chain[c, -1, :].reshape(len(c[0]), 7)

temp = sampler.lnprobability[c,-1].copy().flatten()

#np.savetxt('data.dat', 'Max')
#np.savetxt('data.dat', temp[maxim], delimiter = ' ')

oname = 'data0.dat'
output = open(oname, 'w+')
np.savetxt(output, x_bar, delimiter = ',')
np.savetxt(output, finite_zs[maxim, :], delimiter = ',')
output.close()
print 'Max0:'
print finite_zs[maxim]

lx = sampler.lnprobability[:, -1].argsort()
pl = sampler.chain[lx, -1][-nwalkers/2:]

sampler, initial_positions = b.mix_mcmc.setup(data, priors, pop_prior, tau, nwalkers/2, threads = 1)
sampler, burned_positions = b.mix_mcmc.burn_in(sampler,initial_positions,nsteps=100)
for i,result in enumerate(tqdm(sampler.sample(burned_positions,iterations=1000))): pass

c = np.where(np.isfinite(sampler.lnprobability[:,-1].flatten()))
log_weights = sampler.lnprobability[c,-1].flatten() - np.max(sampler.lnprobability[c,-1])

weights = np.exp(log_weights)

xs = sampler.chain[c, -1].copy().reshape(len(c[0]), 7)

x_bar = np.nansum(weights[:, None]*xs, axis = 0)/np.nansum(weights)

print 'Mean1:'
print x_bar

maxim = np.argmax(sampler.lnprobability[c,-1])

finite_zs = sampler.chain[c, -1, :].reshape(len(c[0]), 7)

temp = sampler.lnprobability[c,-1].copy().flatten()

oname = 'data1.dat'
output = open(oname, 'w+')
np.savetxt(output, x_bar, delimiter = ',')
np.savetxt(output, finite_zs[maxim, :], delimiter = ',')
output.close()
print 'Max1:'
print finite_zs[maxim]
'''
if np.isfinite(temp[maxim]):
	zs = finite_zs[maxim, :]
	
	p = []
	for i in range(10):
		p.append(b.distributions.normal(zs[i], 1e-6))

	#pop_prior_prime = b.distributions.dirichlet(np.array([zs[10]*100,zs[11]*100, 100 - zs[10:11].sum()*100]))


	pop_prior_prime = b.distributions.beta(zs[10]*100, 100 - zs[10]*100)
	priors1_prime = b.distributions.parameter_collection(p[0], p[1], p[2], p[3], p[4])
	priors2_prime = b.distributions.parameter_collection(p[5], p[6], p[7], p[8], p[9])

	priors_prime = [priors1_prime, priors2_prime]


	#def _min_fxn(theta, data, prior, pop_prior, tau):
	#	return -1.*b.likelihood.bi_mix_log_posterior(theta, data, prior, pop_prior, tau)


	#mind =  minimize(_min_fxn,zs,method='Nelder-Mead',args=(data,priors,pop_prior_prime,tau))

	#print mind

else:
	print 'no can do, bro'

posterior_prime = b.mix_laplace.laplace_approximation(data, priors_prime,pop_prior_prime, tau)

np.savetxt('laplace.dat', posterior_prime.mu)

print posterior_prime.mu

sampler.reset()
new = np.array([zs + np.random.randn(12)*1e-6 for _ in range(nwalkers)])

sampler, burned_positions = b.mix_mcmc.burn_in(sampler, new, nsteps = 1000)

for i,result in enumerate(tqdm(sampler.sample(burned_positions,iterations=1000))): pass

maxim = np.argmax(sampler.lnprobability[:,-1])

zs = sampler.chain[maxim, 1, :]
print zs

plt.plot(sampler.lnprobability.T, alpha = 0.1)
plt.show()
uncorrelated_samples = b.mix_mcmc.get_samples(sampler,uncorrelated=True)

fig = b.mix_mcmc.plot_corner(uncorrelated_samples)
fig.savefig('test1_mcmc_corner.png')
'''
