Reference Files
===============
The Ramp Fitting step uses two reference files: GAIN and READNOISE. The gain values
are used to temporarily convert the pixel values from units of DN to
electrons, and convert the results of ramp fitting back to DN.
The read noise values are used as part of the noise estimate for
each pixel. Both are necessary for proper computation of noise estimates.


CRDS Selection Criteria
-----------------------
The readnoise reference file is selected by instrument, detector and, where
necessary, subarray.

The gain reference file is selected based on instrument, detector and,
where necessary, subarray.


Gain Reference File Format
--------------------------
The gain reference file is a FITS file with a single IMAGE extension,
with ``EXTNAME=SCI``, which contains a 2-D floating-point array of gain values
(in e/DN) per pixel. The REFTYPE value is ``GAIN``.


Read Noise Reference File Format
--------------------------------
The read noise reference file is a FITS file with a single IMAGE extension,
with ``EXTNAME=SCI``, which contains a 2-D floating-point array of read noise values
per pixel. The units of the read noise should be electrons and should be the
CDS (Correlated Double Sampling) read noise, i.e. the effective noise between
any pair of non-destructive detector reads. The REFTYPE value is
``READNOISE``.


