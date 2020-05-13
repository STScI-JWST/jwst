import abc
import numpy as np

from typing import Tuple, Union, Type
from scipy.interpolate import interp2d
from astropy.io import fits

from ..assign_wcs.util import compute_scale
from ..datamodels import DataModel


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

    Raises
    ------
    ValueError :
        If apcorr_row_units are not supplied and units are undifined in apcorr_table ColDefs
    ValueError :
        If the input apcorr_table cannot be reduced to a single row based on match criteria from input_model.

    """
    match_pars = {
        'MIRI': {
            'LRS': {'subarray': ['name']},
            'MRS': {'instrument': []},  # Only one row is available for this mode; no selection criteria
        },
        'NIRSPEC': {
            'IFU': {'instrument': ['filter', 'grating']},
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

    def __init__(self, input_model: DataModel, apcorr_table: fits.FITS_rec, location: Tuple[float, float] = None,
                 apcorr_row_units: str = None, **match_kwargs):
        self.correction = None

        self.model = input_model
        self._reference_table = apcorr_table
        self.location = location
        self.apcorr_row_units = apcorr_row_units

        self.match_keys = self._get_match_keys()
        self.match_pars = self._get_match_pars()
        self.match_pars.update(match_kwargs)

        self.reference = self._reduce_reftable()
        self.apcorr_func = self.approximate()

    def _get_row_units(self, row_key: str):
        """Attempt to read units of the apcorr data row dimension.

        Parameters
        ----------
        row_key : str
            Column name that corresponds to the row-dimension of the input apcorr data.

        Raises
        ------
        ValuError :
            If apcorr_row_units are not given during init and units are not defined for the input FITS_rec ColDefs.

        """
        if self.apcorr_row_units is None:
            self.apcorr_row_units = self._reference_table[row_key].units

        if not self.apcorr_row_units:
            raise ValueError(f'Could not determine units for {row_key} from input reference table.')

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
        pass

    def apply(self, spec_table: fits.FITS_rec):
        """Apply interpolated aperture correction value to source-related extraction results in-place.

        Parameters
        ----------
        spec_table : `~fits.FITS_rec`
            Table of aperture corrections values from apcorr reference file.

        """
        cols_to_correct = ('flux', 'surf_bright', 'error', 'sb_error')

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
    def __init__(self, *args, pixphase: float = 0.5, **kwargs):
        self.phase = pixphase  # In the future we'll attempt to measure the pixel phase from inputs.

        super().__init__(*args, **kwargs)

        self._get_row_units('size')

        if self.apcorr_row_units.startswith('arcsec'):
            if self.location is not None:
                self.reference['size'] /= compute_scale(self.model.wcs, self.location)
            else:
                raise ValueError(
                    'If the size column for the input APCORR reference file are in units with arcseconds, a location '
                    '(RA, DEC) must be provided in order to compute a pixel scale to convert arcseconds to pixels.'
                )

    def approximate(self):
        """Generate an approximate function for interpolating apcorr values to input wavelength and size."""
        wavelength = self.reference['wavelength'][:self.reference['nelem_wl']]
        phase_idx = np.where(self.reference['pixphase'] == self.phase)[0][0]

        size = self.reference['size'][phase_idx][:self.reference['nelem_size']]

        apcorr = (
            self.reference['apcorr'][phase_idx][:self.reference['nelem_size'], :self.reference['nelem_wl']]
        )

        return interp2d(wavelength, size, apcorr)

    def measure_phase(self):  # Future method in determining pixel phase
        pass


class ApCorrRadial(ApCorrBase):
    """Aperture correction class used with spectral data produced from an extraction aperture radius."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._get_row_units('radius')

        if self.apcorr_row_units.startswith('arcsec'):
            if self.location is not None:
                self.reference['radius'] /= compute_scale(self.model.wcs, self.location)
            else:
                raise ValueError(
                    'If the radius column for the input APCORR reference file are in units with arcseconds, a location'
                    ' (RA, DEC) must be provided in order to compute a pixel scale to convert arcseconds to pixels.'
                )

    def approximate(self):
        """Generate an approximate function for interpolating apcorr values to input wavelength and radius."""
        wavelength = self.reference['wavelength'][:self.reference['nelem_wl']]
        size = self.reference['radius'][:self.reference['nelem_wl']]
        apcorr = self.reference['apcorr'][:self.reference['nelem_wl']]

        return interp2d(wavelength, size, apcorr)


class ApCorr(ApCorrBase):
    """'Default' Aperture correction class for use with most spectroscopic modes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._get_row_units('size')

        if self.apcorr_row_units.startswith('arcsec'):
            if self.location is not None:
                self.reference['size'] /= compute_scale(self.model.wcs, self.location)
            else:
                raise ValueError(
                    'If the size column for the input APCORR reference file are in units with arcseconds, a location '
                    '(RA, DEC) must be provided in order to compute a pixel scale to convert arcseconds to pixels.'
                )

    def approximate(self):
        """Generate an approximate function for interpolating apcorr values to input wavelength and size."""
        wavelength = self.reference['wavelength'][:self.reference['nelem_wl']]
        size = self.reference['size'][:self.reference['nelem_size']]
        apcorr = self.reference['apcorr'][:self.reference['nelem_size'], :self.reference['nelem_wl']]

        return interp2d(wavelength, size, apcorr)


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
        return ApCorrPhase
