from .smd_hdf5 import new,save,_addhash
import numpy as np

def convert(input_filename):
	## MAKE A TERRIBLE HYBRID DATASET>>>>

	## Load .mat SMD
	from scipy import io
	d = io.matlab.loadmat(input_filename)

	## Make new HDF5 dataset
	output_filename =  '.'.join(input_filename.split('.')[:-1]) + '.hdf5'
	dataset = new(output_filename,force=True)

	## Add new attributes
	from time import asctime
	dataset.attrs.create('Conversion','Converted to HDF5 from Matlab on %s'%(asctime()))
	nmol = d['data'].shape[1]
	dataset.attrs.create('number of trajectories',nmol)

	## Transfer old attributes
	na = len(d['attr'][0,0])
	for i in range(na):
		dataset.attrs.create(d['attr'].dtype.names[i],str(d['attr'][0,0][i][0]))
	try:
		dataset.attrs.create('type',str(d['type'][0]))
	except:
		pass

	## Transfer ID
	#dataset.create_dataset('id',data=str(d['id'][0]))
	dataset.attrs.create('SMD hash ID',str(d['id'][0]))

	## Transfer Types
	#### FROM iSMS
	try:
		column_labels = [aa[0] for aa in d['columns'][0]]
		dataset.create_group('types')
		dataset['types'].create_group('values')
		for i in range(len(column_labels)):
			dataset['types/values'].create_dataset('col%d'%(i),data=str(column_labels[i]))
	except:
		column_labels = None

	## Transfer Data
	for i in range(nmol):
		print('Converting ',i)
		## Collect Data
		dd = d['data'][0,i]
		names = list(dd.dtype.names)

		## Add the incoming data trajectory by trajectory
		group = dataset.create_group('trajectory %06d'%(i))
		group.attrs.create('SMD hash ID',str(dd[names.index('id')]))
		group.create_dataset('id',data=str(dd[names.index('id')]))

		## Add attributes
		ats = dd[names.index('attr')][0,0]
		anames = list(ats.dtype.names)
		for i in range(len(ats)):
			if type(ats[i]) is list:
				ats[i] = ats[i][0]
			group.attrs.create(anames[i],str(ats[i]))

		## Add trajectory data
		g = group.create_group('data')
		_addhash(g)

		g.create_dataset('time',data=dd[names.index('index')])

		qd = dd[names.index('values')]
		## Specific to iSMS
		if not column_labels is None:
			for i in range(len(column_labels)):
				g.create_dataset(column_labels[i],data=qd[:,i])

	## Save the changes, and close the HDF5 file
	save(dataset)
