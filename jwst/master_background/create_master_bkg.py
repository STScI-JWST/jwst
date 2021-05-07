import logging

import numpy as np

from .. import datamodels

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def create_background(wavelength, surf_bright):
    """Create a 1-D spectrum table as a MultiSpecModel.

    This is the syntax for accessing the data in the columns:
    wavelength = output_model.spec[0].spec_table['wavelength']
    background = output_model.spec[0].spec_table['surf_bright']

    Parameters
    ----------
    wavelength : 1-D ndarray
        Array of wavelengths, in micrometers.

    surf_bright : 1-D ndarray
        Array of background surface brightness values.

    Returns
    -------
    output_model : `~jwst.datamodels.MultiSpecModel`, or None
        A data model containing the 1-D background spectrum.  This can be
        written to disk by calling:

        output_model.save(<filename>)
    """

    wl_shape = wavelength.shape
    sb_shape = surf_bright.shape
    bad = False
    if len(wl_shape) > 1:
        bad = True
        log.error("The wavelength array has shape {}; expected "
                  "a 1-D array".format(wl_shape))
    if len(sb_shape) > 1:
        bad = True
        log.error("The background surf_bright array has shape {}; expected "
                  "a 1-D array".format(sb_shape))
    if bad:
        return None

    if wl_shape[0] != sb_shape[0]:
        log.error("wavelength array has length {}, "
                  "but background surf_bright array has length {}."
                  .format(wl_shape[0], sb_shape[0]))
        log.error("The arrays must be the same size.")
        return None

    # Create arrays for columns that we won't need.
    dummy = np.zeros(wl_shape[0], dtype=np.float64)
    dq = np.zeros(wl_shape[0], dtype=np.int32)
    npixels = np.ones(wl_shape[0], dtype=np.float64)

    output_model = datamodels.MultiSpecModel()

    spec_dtype = datamodels.SpecModel().spec_table.dtype

    otab = np.array(list(zip(wavelength, dummy, dummy, dummy, dummy, dummy,
                             surf_bright, dummy, dummy, dummy, dummy,
                             dq, dummy, dummy, dummy, dummy, dummy, npixels)),
                    dtype=spec_dtype)

    spec = datamodels.SpecModel(spec_table=otab)
    output_model.spec.append(spec)

    return output_model
