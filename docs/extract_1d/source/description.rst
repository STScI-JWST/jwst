Description
===========
The extract_1d step extracts a 1-d signal from a 2-d dataset and writes a
spectrum to a product.  Currently this only works on Fixed-Slit data.  This
includes NIRSpec data through any one or more of the fixed slits, MIRI LRS
data through the slit or in the slitless region, and NIRISS slitless data.

Input
=====
Level 2-b countrate data.  The format should be a CubeModel, an
ImageModel, or a MultiSlitModel, with SCI, ERR, DQ, and possibly RELSENS
extensions for each slit.  The SCI extensions should have keyword SLTNAME
to specify which slit was extracted, though if there is only one slit
(e.g. full-frame data), the slit name can be taken from the JSON
reference file instead.

Output
======
The output will be in MultiSpecModel format; for each SLIT there will be
a table extension with the name EXTRACT1D.  This extension will have
columns giving the number of the column or row in the input file, the
calculated wavelength, background, and countrate in counts/pixel of
spectral width, summed along the direction perpendicular to the dispersion.
Currently only a simple summation is done, with no weighting.  A more
sophisticated algorithm will be introduced in future builds.
