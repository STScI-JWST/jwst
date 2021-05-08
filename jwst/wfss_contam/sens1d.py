import numpy as np

from jwst.photom.photom import find_row

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def get_photom_data(phot_model, filter, pupil, order):
    """
    Retrieves wavelength and response data from photom ref file
    for the filter+pupil (grism) mode in use.

    Parameters
    ----------
    phot_model : `jwst.datamodels.NrcWfssPhotomModel` or `jwst.datamodels.NisWfssPhotomModel`
        Photom ref file data model
    filter : str
        Filter value
    pupil : str
        Pupil value
    order : int
        Spectral order number

    Returns
    -------
    ref_waves : float array
        Array of wavelengths from the ref file
    relresps : float array
        Array of response (flux calibration) values from the ref file
    """

    # Get the appropriate row of data from the reference table
    phot_table = phot_model.phot_table
    fields_to_match = {'filter': filter, 'pupil': pupil, 'order': order}
    row = find_row(phot_table, fields_to_match)
    tabdata = phot_table[row]

    # Scalar conversion factor
    scalar_conversion = tabdata['photmjsr']   # unit is MJy / sr

    # Get the length of the relative response arrays in this row
    nelem = tabdata['nelem']

    # Load the wavelength and relative response arrays
    ref_waves = tabdata['wavelength'][:nelem]
    relresps = scalar_conversion * tabdata['relresponse'][:nelem]

    # Make sure waves and relresps are in increasing wavelength order
    if not np.all(np.diff(ref_waves) > 0):
        index = np.argsort(ref_waves)
        ref_waves = ref_waves[index].copy()
        relresps = relresps[index].copy()

    # Convert wavelengths from meters to microns, if necessary
    microns_100 = 1.e-4         # 100 microns, in meters
    if ref_waves.max() > 0. and ref_waves.max() < microns_100:
        ref_waves *= 1.e+6

    return ref_waves, relresps


def create_1d_sens(data_waves, ref_waves, relresps):
    """
    Create a 1D array of photometric conversion values based on
    wavelengths per pixel and response as a function of wavelength.

    Parameters
    ----------
    data_waves : float array
        Input data wavelength values
    ref_waves : float array
        1D wavelength vector on which relative response values are sampled
    relresps : float array
        1D photometric response values, as a function of wavelength

    Returns
    -------
    sens_1d : float array
        1D array of computed photometric conversion values
    no_cal : int array
        1D mask indicating where no conversion is available
    """

    # Interpolate the photometric response values onto the
    # 1D wavelength grid of the data
    sens_1d = np.interp(data_waves, ref_waves, relresps, left=np.NaN, right=np.NaN)

    no_cal = np.isnan(sens_1d)
    sens_1d[no_cal] = 0.

    return sens_1d, no_cal
