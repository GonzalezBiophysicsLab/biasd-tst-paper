import numpy as np
import numba as nb

import ctypes
from sys import platform
import os


################################################################################
######### C Libraries
################################################################################
path = os.path.dirname(__file__)
if platform == 'darwin':
	_sopath = path+'/sm_ssa-mac'
elif platform == 'linux' or platform == 'linux2':
	_sopath = path + '/sm_ssa-linux'
else:
	_sopath = path + '/sm_ssa'
print _sopath+'.so'
_lib = np.ctypeslib.load_library(_sopath, '.')

_lib.sm_ssa.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.double), np.ctypeslib.ndpointer(dtype = np.int32), np.ctypeslib.ndpointer(dtype = np.double), np.ctypeslib.ndpointer(dtype=np.int32)]
_lib.sm_ssa.restype  = ctypes.c_void_p

_lib.render_trace.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.double),  np.ctypeslib.ndpointer(dtype = np.double), np.ctypeslib.ndpointer(dtype = np.int32), np.ctypeslib.ndpointer(dtype = np.double), np.ctypeslib.ndpointer(dtype=np.double), np.ctypeslib.ndpointer(dtype=np.double)]
_lib.render_trace.restype  = ctypes.c_void_p

################################################################################
######### Python Wrappers
################################################################################

def ssa_dwells(rates,tlength):
	'''
	Input:
		* `rates` is a KxK np.ndarray of the rate constants for transitions between states i and j. It should have zeros on the diagonals
		* `tlength` is a float/double that specifies the total time of the trajectory
	Output
		* a 2xN np.ndarray of states indices (0) and the corresponding dwell times (1)
	'''
	np.random.seed()

	q = rates.copy()
	for i in range(q.shape[0]):
		q[i,i] = - q[i].sum()
	pst = steady_state(q)

	p = np.random.rand()
	initialstate = np.searchsorted(pst.cumsum(),p)

	nstates = rates.shape[0]
	rates = rates.flatten()
	n = np.max((int(np.floor(tlength*rates.max())),1))*2

	states = np.zeros(n, dtype=np.int32)
	dwells = np.zeros(n, dtype=np.double)
	cut = np.array(0,dtype=np.int32)
	_lib.sm_ssa(n, (tlength), nstates, initialstate, rates, states, dwells,cut)
	states = states[:cut]
	dwells = dwells[:cut]
	if np.size(states) == 0:
		states = [initialstate]
		dwells = [tlength]

	return np.array((states,dwells))

def render_trajectory(trajectory,steps,dt,emission):
	'''
	Input:
		* `trajectory` is the 2xM output from the `ssa_dwells` function
		* `steps` is the integer number of discrete timepoints to render
		* `dt` is a float specifying the period of each timepoint
		* `emission` is a np.ndarray of length K with the emission means of each state
	Output:
		* a np.ndarray of shape 2x`steps` containing the time points (0) and the signal values (1) of the rendered trajectory
	'''

	states = trajectory[0].astype(np.int32)
	dwells = trajectory[1]
	times = dwells.cumsum().astype(np.double)
	times = np.append(0,times)
	timesteps = states.shape[0]
	steps = int(steps)

	nstates = emission.size
	emissions = emission.astype(np.double)
	x = np.arange(steps,dtype=np.double)*dt + dt
	y  = np.zeros_like(x)
	_lib.render_trace(steps, timesteps, nstates, x, y, states, times, dwells, emissions)
	return np.array((x,y))

def steady_state(q):
	'''
	Calculates the steady state probabilities the states from a Q matrix
	'''
	from scipy.linalg import expm
	tinf = 1000./np.abs(q).min()
	return expm(q*tinf)[0]


def simulate(rates,emissions,noise,nframes,dt):
	'''
	Input:
		* `rates` is a KxK np.ndarray of the rate constants for transitions between states i and j. It should have zeros on the diagonals
		* `emissions` is a np.ndarray of length K with the emission means of each state
		* `noise` is a float with the standard deviation of the normal distribution used to add noise to the signal
		* `nframes` is an integer number of datapoints in the signal versus time trajectory
		* `dt` is the time period of each datapoint
	Output:
		* `trajectory` is a 2xN np.ndarray of states indices (0) and the corresponding dwell times (1) of the state trajectory
		* `signal` is a 2x`nframes` np.ndarray containing the time points (0) and the signal values (1) of the rendered signal trajectory
	'''

	np.random.seed()
	trajectory = ssa_dwells(rates,nframes*dt)
	signal = render_trajectory(trajectory,nframes,dt,emissions)
	signal[1] += np.random.normal(size=signal.shape[1])*noise
	return trajectory,signal

def test(noise):
	'''
	Tries to simulate a trajectory and make a plot
	'''

	import matplotlib.pyplot as plt

	rates = np.array(([0,0.176],[0.115,0.]))
	emissions = np.array((0.36,0.58))
	noise = noise # SNR = 20
	nframes = 500
	dt = .1 # 500 msec

	trajectory,signal = simulate(rates,emissions,noise,nframes,dt)
	'''
	q = rates.copy()
	p = np.zeros(q.shape[0])
	for i in range(q.shape[0]):
		q[i,i] = -q[i].sum()
		p[i] = (trajectory[1][trajectory[0] == i]).sum()
	p /= p.sum()
	
	print 'Steady State:',steady_state(q)
	print 'Simulation  :',p
	'''
	return signal
	'''
	stop = 1000
	plt.plot(signal[0,:stop],signal[1,:stop],lw=1)
	plt.xlim(0,signal[0,stop])
	plt.xlabel('Time',fontsize=12)
	plt.ylabel('Signal',fontsize=12)
	plt.title('Blurred SSA Trajectory',fontsize=16)
	plt.show()
	'''
