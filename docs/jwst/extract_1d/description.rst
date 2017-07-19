
Description
===========
The extract_1d step extracts a 1-d signal from a 2-d or 3-d dataset and
writes a spectrum to a product.  This works on fixed-slit data (NIRSpec
data through any one or more of the fixed slits, MIRI LRS data through
the slit or in the slitless region, and NIRISS slitless data) as well as
IFU data and NIRSpec MOS (micro-shutter array) data.

For IFU data, the extraction options differ depending on
whether the target is a point source or an extended source.  For a point
source, the spectrum will be extracted using circular aperture photometry,
optionally including background subtraction using a circular annulus.
For an extended source, rectangular aperture photometry will be used, with
no background subtraction.  The photometry makes use of astropy photutils.
The region of overlap between an aperture and a pixel can be calculated by
one of three different methods:  "exact", limited only by finite precision
arithmetic; "center", i.e. the full value in a pixel will be included if its
center is within the aperture; or "subsample", which means pixels will be
subsampled N x N, and the "center" option will be used for each sub-pixel.


Input
=====
Level 2-b countrate data, or level-3 data.  The format should be a
CubeModel, an IFUCubeModel, an ImageModel, or a MultiSlitModel, with at
least SCI and RELSENS extensions for each slit.  The SCI extensions should
have keyword SLTNAME to specify which slit was extracted, though if there
is only one slit (e.g. full-frame data), the slit name can be taken from
the JSON reference file instead.

Output
======
The output will be in MultiSpecModel format; for each input slit there will
be an output table extension with the name EXTRACT1D.  This extension will
have columns WAVELENGTH, FLUX, ERROR, DQ, NET, NERROR, BACKGROUND, and BERROR.
WAVELENGTH is the value calculated using the WCS.  NET is the count rate
minus background, in counts/pixel of spectral width, summed along the
direction perpendicular to the dispersion.  Currently only a simple
summation is done, with no weighting.  A more sophisticated algorithm will
be introduced in future builds.  BACKGROUND is the measured background,
scaled to the extraction width used for the NET.  BACKGROUND will be
zero if the reference file did not specify that background should be
determined.  FLUX will be computed from NET if there is a RELSENS table
for the input slit; otherwise, FLUX will be zero.
ERROR, DQ, NERROR, and BERROR are not populated with useful values yet.
