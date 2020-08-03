# BIASD
Bayesian Inference for the Analysis of Sub-temporal-resolution Data
Version 0.2.2

An updated version of BIASD capable of running global analyses across multiple datasets

## Original Manuscript
Increasing the time resolution of single-molecule experiments with Bayesian inference.
Colin D Kinz-Thompson, Ruben L Gonzalez Jr.,
bioRxiv 099648; doi: https://doi.org/10.1101/099648
https://www.biorxiv.org/content/early/2017/05/26/099648

## Updates
* Updated to Python 3.
* Compatible with emcee 3.0 (backward-compatible with emcee 2.0).
* Input data now persistent on GPU. Decreases I/O times and makes the CUDA version faster.
* Capable of running global analysis across datasets using parameters that can be recast to rate constants.
  Implemented here to use transition state theory (TST) to analyse datasets collected across a range of 
  temperatures, yielding estimates of activation enthalpies and entropies for the kinetics of the analysed process
  (currently only available using MCMC sampling mode). 

## Notes
See the [documentation](http://biasd.readthedocs.io/) or open `Documentation.html` for more information.
