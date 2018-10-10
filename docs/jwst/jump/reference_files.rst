Reference File Types
=====================

The Jump step uses two reference files: GAIN and READNOISE. The gain values
are used to temporarily convert the pixel values from units of DN to
electrons. The read noise values are used as part of the noise estimate for
each pixel. Both are necessary for proper computation of noise estimates.

.. include:: ../includes/standard_keywords.rst

.. include:: gain_selection.rst

.. include:: readnoise_selection.rst


Reference File Formats
----------------------

GAIN Reference Files
^^^^^^^^^^^^^^^^^^^^

The gain reference file is a FITS file with a single IMAGE extension,
with ``EXTNAME=SCI``, which contains a 2-D floating-point array of gain values
(in e/DN) per pixel. The REFTYPE value is ``GAIN``.


READNOISE Reference Files
^^^^^^^^^^^^^^^^^^^^^^^^^
The read noise reference file is a FITS file with a single IMAGE extension,
with ``EXTNAME=SCI``, which contains a 2-D floating-point array of read noise
values per pixel. The units of the read noise should be DN and should be the
CDS (Correlated Double Sampling) read noise, i.e. the effective noise between
any pair of non-destructive detector reads. The REFTYPE value is ``READNOISE``.

