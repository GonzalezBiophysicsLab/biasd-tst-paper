#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module allows you to use the single-molecule dataset (SMD) format, and integrate BIASD results into them. The data is stored in as HDF5, and as such can be arbitrarily changed to fit your needs. Generally speaking, the structure is

	SMD File ``(HDF5 File)``:
		- attrs ``(HDF5 Attributes)``
			- time created ``(Str)``
			- SMD hash ID ``(Str)``
			- number of trajectories ``(Int)``
			- attribute 1 (denotes generic attribute) ``(Anything)``
			- attribute 2 (denotes generic attribute) ``(Anything)``
			- ...
		- trajectory 0 ``(HDF5 Group)``
			- attrs ``(HDF5 Attributes)``
				- time created
				- SMD hash ID
				- attribute 1
				- attribute 2
				- ...
			- data ``(HDF5 Group)``
				- time ``(HDF5 Dataset)``
					- value ``(numpy ndarray of values)``
				- signal 1 ``(HDF5 Dataset)``
					- value ``(numpy ndarray of values)``
				- ...
			- BIASD analysis 1 ``(HDF5 Group), eg ensemble MCMC results``
				- attrs ``(HDF5 Attributes)``
					- time created
					- SMD hash ID
					- attribute 1
					- attribute 2
					- ...
				- analysis dataset 1 ``(HDF5 Dataset), eg MCMC samples``
					- ...
		- trajectory 1 ``(HDF5 Dataset)``
			- ...
"""
try:
	from .smd_hdf5 import new, load, save, loadtxt
	from . import add
	from . import read
	from ..gui.smd_loader import launch as viewer
	from .matlab_to_hdf5 import convert as convert_matlab
except:
	print("SMD failed to load - Make sure h5py is installed")
