import simulate_trajectories as sim
import numpy as np

for i in xrange(10):
	data = []

	for j in xrange(550):
		data.append(sim.test(0.01*(1 + i))[1])

	data = np.array(data)

	#print data.shape

	oname = 'l1l9_sim_25c_biasd_{}_100ms_fret.dat'.format(i+1)
	print oname
	np.savetxt(oname, data, delimiter = ',')
