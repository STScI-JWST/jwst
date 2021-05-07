import abc

from typing import Tuple, Union, Type
from scipy.interpolate import interp2d, interp1d
from astropy.io import fits
from stdatamodels import DataModel

from ..assign_wcs.util import compute_scale
from ..datamodels import MultiSlitModel
import numpy as np


class ApCorrBase(abc.ABC):
    """Base class for aperture correction classes.

    Parameters
    ----------
    input_model : `~/jwst.datamodels.DataModel`
        Input data model used to determine matching parameters.
    apcorr_table : `~/astropy.io.fits.FITS_rec`
        Aperture correction table data from APCORR reference file.
    location : tuple, Optional
        Reference location (RA, DEC) used to calculate the pixel scale used in converting values in arcec to pixels.
        Default is None, however, if the input reference data contains size/radius data in units of arcsecs, a location
        is required.
    apcorr_row_units : str, Optional
        Units for the aperture correction data "row" values (assuming the aperture correction is a 2D array).
        If not given, units will be determined from the input apcorr_table ColDefs if possible.
    slit_name : str, Optional
        For MultiSlitModels, the name of the slit being processed.

    Raises
    ------
    ValueError :
        If apcorr_row_units are not supplied and units are undefined in apcorr_table ColDefs
    ValueError :
        If the input apcorr_table cannot be reduced to a single row based on match criteria from input_model.

    """
    match_pars = {
        'MIRI': {
            'LRS': {'subarray': ['name']}
        },
        'NIRSPEC': {
            'MSASPEC': {'instrument': ['filter', 'grating']},
            'FIXEDSLIT': {'instrument': ['filter', 'grating']},  # Slit is also required; passed in as init arg
            'BRIGHTOBJ': {'instrument': ['filter', 'grating']}
        },
        'NIRCAM': {
            'WFSS': {'instrument': ['filter', 'pupil']},
            'NRC_GRISM': {'instrument': ['filter', 'pupil']}
        },
        'NIRISS': {
            'WFSS': {'instrument': ['filter', 'pupil']}
        }
    }

    size_key = None

    def __init__(self, input_model: DataModel, apcorr_table: fits.FITS_rec, sizeunit: str,
                 location: Tuple[float, float, float] = None, slit_name: str = None, **match_kwargs):
        self.correction = None

        self.model = input_model
        self._reference_table = apcorr_table
        self.location = location
        self.apcorr_sizeunits = sizeunit
        self.slit_name = slit_name

        self.match_keys = self._get_match_keys()
        self.match_pars = self._get_match_pars()
        self.match_pars.update(match_kwargs)
        self.reference = self._reduce_reftable()
        self._convert_size_units()
        self.apcorr_func = self.approximate()

    def _convert_size_units(self):
        """If the SIZE or Radius column is in units of arcseconds, convert to pixels."""
        if self.apcorr_sizeunits.startswith('arcsec'):
            # compute_scale returns scale in degrees
            if self.location is not None:
                if isinstance(self.model, MultiSlitModel):
                    idx = [slit.name for slit in self.model.slits].index(self.slit_name)
                    scale_degrees = compute_scale(
                        self.model.slits[idx].meta.wcs,
                        self.location,
                        disp_axis=self.model.slits[idx].meta.wcsinfo.dispersion_direction)
                    scale_arcsec = scale_degrees * 3600.00
                    self.reference[self.size_key] /= scale_arcsec
                else:
                    scale_degrees = compute_scale(
                        self.model.meta.wcs,
                        self.location,
                        disp_axis=self.model.meta.wcsinfo.dispersion_direction)
                    scale_arcsec = scale_degrees * 3600.00
                    self.reference[self.size_key] /= scale_arcsec
            else:
                raise ValueError(
                    'If the size column for the input APCORR reference file is in units with arcseconds, a location '
                    '(RA, DEC, wavelength) must be provided in order to compute a pixel scale to convert arcseconds to '
                    'pixels.'
                )

    def _get_match_keys(self) -> dict:
        """Get column keys needed for reducing the reference table based on input."""
        instrument = self.model.meta.instrument.name.upper()
        exptype = self.model.meta.exposure.type.upper()

        relevant_pars = self.match_pars[instrument]

        for key in relevant_pars.keys():
            if key in exptype:
                return relevant_pars[key]

    def _get_match_pars(self) -> dict:
        """Get meta parameters required for reference table row-selection."""
        match_pars = {}

        for node, keys in self.match_keys.items():
            meta_node = self.model.meta[node]

            for key in keys:
                match_pars[key if key != 'name' else node] = getattr(meta_node, key)

        return match_pars

    def _reduce_reftable(self) -> fits.FITS_record:
        """Reduce full reference table to a single matched row."""
        table = self._reference_table.copy()

        for key, value in self.match_pars.items():

            if isinstance(value, str):  # Not all files will have the same format as input model metadata values.
                table = table[table[key].upper() == value.upper()]
            else:
                table = table[table[key] == value]

        if len(table) != 1:
            raise ValueError('Could not resolve APCORR reference for input.')
        return table[0]

    @abc.abstractmethod
    def approximate(self):
        """Generate an approximate aperture correction function based on input data."""
        pass

    def apply(self, spec_table: fits.FITS_rec):
        """Apply interpolated aperture correction value to source-related extraction results in-place.

        Parameters
        ----------
        spec_table : `~fits.FITS_rec`
            Table of aperture corrections values from apcorr reference file.

        """
        cols_to_correct = ('flux', 'flux_error', 'flux_var_poisson', 'flux_var_rnoise',
                           'flux_var_flat', 'surf_bright', 'sb_error', 'sb_var_poisson',
                           'sb_var_rnoise', 'sb_var_flat')

        for row in spec_table:
            correction = self.apcorr_func(row['npixels'], row['wavelength'])

            for col in cols_to_correct:
                row[col] *= correction


class ApCorrPhase(ApCorrBase):
    """Produce and apply aperture correction for input data with pixel phase.

    Parameters
    ----------
    pixphase : float, Default = 0.5
        Pixel phase of the input data.

    *args, **kwargs :
        See ApCorrBase for more.

    """
    size_key = 'size'

    def __init__(self, *args, pixphase: float = 0.5, **kwargs):
        self.phase = pixphase  # In the future we'll attempt to measure the pixel phase from inputs.

        super().__init__(*args, **kwargs)

    def approximate(self):
        """Generate an approximate function for interpolating apcorr values to input wavelength and size."""
        def _approx_func(wavelength: float, size: float, pixel_phase: float) -> interp2d:
            """Create a 'custom' approximation function that approximates the aperture correction in two stages based on
            input data.

            Parameters
            ----------
            wavelength : float
                Input wavelength
            size : float
                Input size (In extract.py this would be `n_pixels`)
            pixel_phase : float
                Input pixel phase

            Returns
            -------
            Aperture correction approximation function that takes wavelength, size, and pixel_phase as inputs.

            """
            # apcorr column data has shape (pixphase, wavelength, size)
            # Reduce apcorr dimensionality by interpolating in the pixphase dimension first, then size & wavelength
            apcorr_pixphase_func = interp1d(self.reference['pixphase'], self.reference['apcorr'])
            size_wl_func = interp1d(self.reference['wavelength'], self.reference['size'])

            apcorr_pixphase = apcorr_pixphase_func(pixel_phase)
            size_wl = size_wl_func(wavelength)

            pixphase_size_func = interp2d(self.reference['wavelength'], size_wl, apcorr_pixphase)

            return pixphase_size_func(wavelength, size)

        return _approx_func

    def measure_phase(self):  # Future method in determining pixel phase
        pass

    def apply(self, spec_table: fits.FITS_rec):
        """Apply interpolated aperture correction value to source-related extraction results in-place.

        Parameters
        ----------
        spec_table : `~fits.FITS_rec`
            Table of aperture corrections values from apcorr reference file.

        """
        cols_to_correct = ('flux', 'flux_error', 'flux_var_poisson', 'flux_var_rnoise',
                           'flux_var_flat', 'surf_bright', 'sb_error', 'sb_var_poisson',
                           'sb_var_rnoise', 'sb_var_flat')

        for row in spec_table:
            try:
                correction = self.apcorr_func(row['wavelength'], row['npixels'], self.phase)
            except ValueError:
                correction = None  # Some input wavelengths might not be supported (especially at the ends of the range)

            for col in cols_to_correct:
                if correction:
                    row[col] *= correction


class ApCorrRadial(ApCorrBase):
    """Aperture correction class used with spectral data produced from an extraction aperture radius."""

    def __init__(self, input_model: DataModel, apcorr_table,
                 location: Tuple[float, float, float] = None):

        self.correction = None
        self.model = input_model
        self.location = location
        self.reference = apcorr_table.apcorr_table
        self.apcorr_sizeunits = self.reference.radius_units
        self._convert_size_units()

    def approximate(self):
        # Base class needs this. For ApCorrRadial this is done in find_apcorr_func
        pass

    def _convert_size_units(self):
        """If the SIZE or Radius column is in units of arcseconds, convert to pixels."""
        if self.apcorr_sizeunits.startswith('arcsec'):
            # compute_scale returns scale in degrees
            if self.location is not None:
                scale_degrees = compute_scale(
                    self.model.meta.wcs,
                    self.location,
                    disp_axis=self.model.meta.wcsinfo.dispersion_direction)
                scale_arcsec = scale_degrees * 3600.00
                self.reference.radius /= scale_arcsec
            else:
                raise ValueError(
                    'If the size column for the input APCORR reference file is in units with arcseconds, a location '
                    '(RA, DEC, wavelength) must be provided in order to compute a pixel scale to convert arcseconds to '
                    'pixels.'
                )

    def apply(self, spec_table: fits.FITS_rec):
        """Apply interpolated aperture correction value to source-related extraction results in-place.

        Parameters
        ----------
        spec_table : `~fits.FITS_rec`
            Table of aperture corrections values from apcorr reference file.

        """
        cols_to_correct = ('flux', 'flux_error', 'flux_var_poisson', 'flux_var_rnoise',
                           'flux_var_flat', 'surf_bright', 'sb_error', 'sb_var_poisson',
                           'sb_var_rnoise', 'sb_var_flat')

        for i, row in enumerate(spec_table):
            correction = self.apcorr_correction[i]
            for col in cols_to_correct:
                row[col] *= correction

    def match_wavelengths(self, wavelength_ifu):
        # given the ifu wavelength value - redefine the apcor func and radius to this wavelength
        # apcor reference data
        self.wavelength = self.reference.wavelength.flatten()
        self.size = self.reference.radius
        self.apcorr = self.reference.apcorr

        dim = self.apcorr.shape[0]
        size_match = np.zeros((dim, wavelength_ifu.shape[0]))
        apcorr_match = np.zeros((dim, wavelength_ifu.shape[0]))
        self.apcorr_correction = []  # set up here defined in find_apcor_func
        # loop over each radius dependent plane and interpolate to ifu wavelength
        for i in range(dim):
            radi = self.size[i, :]
            frad = interp1d(self.wavelength, radi, bounds_error=False, fill_value="extrapolate")
            radius_match = frad(wavelength_ifu)
            size_match[i, :] = radius_match

            appi = self.apcorr[i, :]
            fap = interp1d(self.wavelength, appi, bounds_error=False, fill_value="extrapolate")
            ap_match = fap(wavelength_ifu)
            apcorr_match[i, :] = ap_match

        self.apcorr = apcorr_match
        self.size = size_match

    def find_apcorr_func(self, iwave, radius_ifu):
        # at ifu wavelength plane (iwave), the extraction radius is radius_ifu
        # pull out the radius values (self.size)  to use in the apcor ref file for this iwave
        # self.size and self.apcorr have already been interpolated in wavelength to match the
        # the ifu wavelength range.

        radius_apcor = self.size[:, iwave]
        temparray = self.apcorr[:, iwave]
        fap = interp1d(radius_apcor, temparray, fill_value="extrapolate")
        correction = fap(radius_ifu)
        self.apcorr_correction.append(correction)
        return


class ApCorr(ApCorrBase):
    """'Default' Aperture correction class for use with most spectroscopic modes."""
    size_key = 'size'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def approximate(self):
        """Generate an approximate function for interpolating apcorr values to input wavelength and size."""
        wavelength = self.reference['wavelength'][:self.reference['nelem_wl']]
        size = self.reference['size'][:self.reference['nelem_size']]
        apcorr = self.reference['apcorr'][:self.reference['nelem_wl'], :self.reference['nelem_size']]

        return interp2d(size, wavelength, apcorr)


def select_apcorr(input_model: DataModel) -> Union[Type[ApCorr], Type[ApCorrPhase], Type[ApCorrRadial]]:
    """Select appropriate Aperture correction class based on input DataModel.

    Parameters
    ----------
    input_model : `~/jwst.datamodels.DataModel`
        Input data on which the aperture correction is to be applied.

    Returns
    -------
    Aperture correction class.

    """
    if input_model.meta.instrument.name == 'MIRI':
        if 'MRS' in input_model.meta.exposure.type:
            return ApCorrRadial
        else:
            return ApCorr

    if input_model.meta.instrument.name == 'NIRCAM':
        return ApCorr

    if input_model.meta.instrument.name == 'NIRISS':
        return ApCorr

    if input_model.meta.instrument.name == 'NIRSPEC':
        if input_model.meta.exposure.type.upper() == 'NRS_IFU':
            return ApCorrRadial
        else:
            return ApCorrPhase
