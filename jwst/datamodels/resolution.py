from __future__ import absolute_import, unicode_literals, division, print_function

from . import model_base

__all__ = ['ResolutionModel', 'MiriResolutionModel']


class ResolutionModel(model_base.DataModel):
    """
    A data model for Spectral Resolution  parameters reference tables.
    """
    schema_url = "resolution.schema.yaml"

    def __init__(self, init=None, resolution_table=None, **kwargs):
        super(ResolutionModel, self).__init__(init=init, **kwargs)

        if resolution_table is not None:
            self.resolution_table = ifucubepars_table


class MiriResolutionModel(ResolutionModel):
    """
    A data model for MIRI Resolution reference files.
    
    Parameters
    ----------
    intit: any
       Any of the initializers supported by '~jwst.datamodels.DataModel'

    Resolving  Power table
      A table containing resolving power of the MRS. THe table consist of 11 cols and 12 rows
      each row corresponds to a band
      The columns give the name of band, central wavelength, and polynomial coefficeints (a,b,c)
      needed to obtain the limits and average value of the spectral resolution

    PSF FWHM Alpha Table
      5 columns
      Column 1 gives the cutoff wavelength where the polynomials describing alpha FWHM change
      Columns 2 and 3 give the polynomial cofficients (a,b) describing alpha FWHM for wavelengths
         shorter than cuttoff.
      Columns 4 and 5 give the polynomial coefficients (a,b) describing alpha FWHM for wavelengths
         longer than the cutoff

    PSF FWHM Beta Table
      5 columns
      Column 1 gives the cutoff wavelength where the polynomials describing alpha FWHM change
      Columns 2 and 3 give the polynomial cofficients (a,b) describing beta FWHM for wavelengths
         shorter than cuttoff.
      Columns 4 and 5 give the polynomial coefficients (a,b) describing beta FWHM for wavelengths
         longer than the cutoff

    """
    schema_url = "miri_resolution.schema.yaml"

    def __init__(self, init=None, resolving_power_table=None, psf_fwhm_alpha_table = None,
                 psf_fwhm_beta_table=None, **kwargs):
        super(MiriResolutionModel, self).__init__(init=init, **kwargs)

        if resolving_power_table is not None:
            self.resolving_power_table = resolving_power_table

        if psf_fwhm_alpha_table is not None:
            self.psf_fwhm_alpha_table = psf_fwhm_alpha_table

        if psf_fwhm_beta_table is not None:
            self.psf_fwhm_beta_table = psf_fwhm_beta_table


