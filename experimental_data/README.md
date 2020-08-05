# Temperature Dependent smFRET data
Combined smFRET traces for different ribosomal complexes at 5 temperature points. Input data
for global analysis using BIASD

## Manuscript
*add MS*

## Notes
* Compressed HDF5 files using the gzip (deflate) filter
* The HDF5 files are named following the scheme RC_(tRNA)_ (FRET signal )_ (acquisition time).hdf5
* Each HDF5 file contains 5 datasets which are named by the corresponding temperature (in K)
* Each dataset is a numpy array containing the combined smFRET traces of all the individual
	molecules of the particular RC at the specific temperature.
