#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import likelihood
from . import laplace
from . import distributions


try:
	from . import mcmc

except:
	print("Could not import MCMC - install emcee and corner?")

try:
	from . import glob_mcmc
except:
	print("Could not import global MCMC")

from . import smd
try:
	from . import gui
except:
	print('Could not import gui - install PyQt5?')

__version__ = "0.2.2"
