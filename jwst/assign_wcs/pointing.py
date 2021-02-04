import numpy as np
from astropy import units as u
from astropy import coordinates as coords
from astropy.modeling import models as astmodels
from astropy.modeling.models import Shift, Scale, RotationSequence3D
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product
from gwcs import utils as gwutils
from gwcs.geometry import SphericalToCartesian, CartesianToSpherical
from gwcs import coordinate_frames as cf
from gwcs import wcs

from stdatamodels import DataModel


__all__ = ["compute_roll_ref", "frame_from_model", "fitswcs_transform_from_model"]


def v23tosky(input_model, wrap_v2_at=180, wrap_lon_at=360):
    v2_ref = input_model.meta.wcsinfo.v2_ref / 3600
    v3_ref = input_model.meta.wcsinfo.v3_ref / 3600
    roll_ref = input_model.meta.wcsinfo.roll_ref
    ra_ref = input_model.meta.wcsinfo.ra_ref
    dec_ref = input_model.meta.wcsinfo.dec_ref

    angles = np.array([v2_ref, -v3_ref, roll_ref, dec_ref, -ra_ref])
    axes = "zyxyz"
    rot = RotationSequence3D(angles, axes_order=axes)

    # The sky rotation expects values in deg.
    # This should be removed when models work with quantities.
    m = ((Scale(1/3600) & Scale(1/3600)) | SphericalToCartesian(wrap_lon_at=wrap_v2_at)
         | rot | CartesianToSpherical(wrap_lon_at=wrap_lon_at))
    m.name = 'v23tosky'
    return m


def compute_roll_ref(v2_ref, v3_ref, roll_ref, ra_ref, dec_ref, new_v2_ref, new_v3_ref):
    """
    Computes the position of V3 (measured N to E) at the center af an aperture.

    Parameters
    ----------
    v2_ref, v3_ref : float
        Reference point in the V2, V3 frame [in arcsec] (FITS keywords V2_REF and V3_REF)
    roll_ref : float
        Position angle of V3 at V2_REF, V3_REF, [in deg]
        When ROLL_REF == PA_V3, then (V2_REF, V3_REF) = (0, 0)
    ra_ref, dec_ref : float
        RA and DEC corresponding to V2_REF and V3_REF, [in deg]
    new_v2_ref, new_v3_ref : float
        The new position in V2, V3 where the position of V3 is computed, [in arcsec]
        The center of the aperture in V2,V3

    Returns
    -------
    new_roll : float
        The value of ROLL_REF (in deg)

    """
    v2 = np.deg2rad(new_v2_ref / 3600)
    v3 = np.deg2rad(new_v3_ref / 3600)

    v2_ref = v2_ref / 3600
    v3_ref = v3_ref / 3600

    if np.isclose(v2_ref, 0, atol=1e-13) and np.isclose(v3_ref, 0, atol=1e-13):
        angles = [roll_ref, dec_ref, ra_ref]
        axes = "xyz"
    else:
        angles = [v2_ref, -v3_ref, roll_ref, dec_ref, -ra_ref]
        axes = "zyxyz"

    matrices = [rotation_matrix(a, ax) for a, ax in zip(angles, axes)]
    m = matrix_product(*matrices[::-1])
    return _roll_angle_from_matrix(m, v2, v3)


def _roll_angle_from_matrix(matrix, v2, v3):
    X = -(matrix[2, 0] * np.cos(v2) + matrix[2, 1] * np.sin(v2)) * \
        np.sin(v3) + matrix[2, 2] * np.cos(v3)
    Y = (matrix[0, 0] * matrix[1, 2] - matrix[1, 0] * matrix[0, 2]) * np.cos(v2) + \
        (matrix[0, 1] * matrix[1, 2] - matrix[1, 1] * matrix[0, 2]) * np.sin(v2)
    new_roll = np.rad2deg(np.arctan2(Y, X))
    if new_roll < 0:
        new_roll += 360
    return new_roll


def wcsinfo_from_model(input_model):
    """
    Create a dict {wcs_keyword: array_of_values} pairs from a data model.

    Parameters
    ----------
    input_model : `~stdatamodels.DataModel`
        The input data model

    """
    defaults = {'CRPIX': 0, 'CRVAL': 0, 'CDELT': 1., 'CTYPE': "", 'CUNIT': u.Unit("")}
    wcsinfo = {}
    wcsaxes = input_model.meta.wcsinfo.wcsaxes
    wcsinfo['WCSAXES'] = wcsaxes
    for key in ['CRPIX', 'CRVAL', 'CDELT', 'CTYPE', 'CUNIT']:
        val = []
        for ax in range(1, wcsaxes + 1):
            k = (key + "{0}".format(ax)).lower()
            v = getattr(input_model.meta.wcsinfo, k, defaults[key])
            val.append(v)
        wcsinfo[key] = np.array(val)

    pc = np.zeros((wcsaxes, wcsaxes))
    for i in range(1, wcsaxes + 1):
        for j in range(1, wcsaxes + 1):
            pc[i - 1, j - 1] = getattr(input_model.meta.wcsinfo, 'pc{0}_{1}'.format(i, j), 1)
    wcsinfo['PC'] = pc
    wcsinfo['RADESYS'] = input_model.meta.coordinates.reference_frame
    wcsinfo['has_cd'] = False
    return wcsinfo


def fitswcs_transform_from_model(wcsinfo, wavetable=None):
    """
    Create a WCS object using from datamodel.meta.wcsinfo.
    Transforms assume 0-based coordinates.

    Parameters
    ----------
    wcsinfo : dict-like
        ``~jwst.meta.wcsinfo`` structure.

    Return
    ------
    transform : `~astropy.modeling.core.Model`
        WCS forward transform - from pixel to world coordinates.

    """
    spatial_axes, spectral_axes, unknown = gwutils.get_axes(wcsinfo)
    #sp_axis = spectral_axes[0]

    transform = gwutils.make_fitswcs_transform(wcsinfo)
    if spectral_axes:
        sp_axis = spectral_axes[0]
        if wavetable is None :
            # Subtract one from CRPIX which is 1-based.
            spectral_transform = astmodels.Shift(-(wcsinfo['CRPIX'][sp_axis] - 1)) | \
                astmodels.Scale(wcsinfo['CDELT'][sp_axis]) | \
                astmodels.Shift(wcsinfo['CRVAL'][sp_axis])
        else :
            # Wave dimension is an array that needs to be converted to a table
            waves = wavetable['wavelength'].flatten()
            spectral_transform = astmodels.Tabular1D(lookup_table=waves)

        transform = transform & spectral_transform

    return transform


def frame_from_fits(ff):
    raise NotImplementedError


def frame_from_model(wcsinfo):
    """
    Initialize a coordinate frame based on values in model.meta.wcsinfo.

    Parameters
    ----------
    wcsinfo : `~stdatamodels.DataModel` or dict
        Either one of the JWST data moels or a dict with model.meta.wcsinfo.

    Returns
    -------
    frame : `~coordinate_frames.CoordinateFrame`

    """
    if isinstance(wcsinfo, DataModel):
        wcsinfo = wcsinfo_from_model(wcsinfo)

    wcsaxes = wcsinfo['WCSAXES']
    celestial_axes, spectral_axes, other = gwutils.get_axes(wcsinfo)
    cunit = wcsinfo['CUNIT']
    frames = []
    if celestial_axes:
        ref_frame = coords.ICRS()
        celestial = cf.CelestialFrame(name='sky', axes_order=tuple(celestial_axes),
                                    reference_frame=ref_frame, unit=cunit[celestial_axes],
                                    axes_names=('RA', 'DEC'))
        frames.append(celestial)
    if spectral_axes:
        spec = cf.SpectralFrame(name='spectral', axes_order=tuple(spectral_axes),
                                unit=cunit[spectral_axes],
                                axes_names=('wavelength',))
        frames.append(spec)
    if other:
        # Make sure these are strings and not np.str_ objects.
        axes_names = tuple([str(name) for name in wcsinfo['CTYPE'][other]])
        name = "_".join(wcsinfo['CTYPE'][other])
        spatial = cf.Frame2D(name=name, axes_order=tuple(other), unit=cunit[other],
                             axes_names=axes_names)
        frames.append(spatial)
    if wcsaxes == 2:
        return frames[0]
    elif wcsaxes == 3:
        world = cf.CompositeFrame(frames, name='world')
        return world
    else:
        raise ValueError("WCSAXES can be 2 or 3, got {0}".format(wcsaxes))


def create_fitswcs(inp, input_frame=None):
    if isinstance(inp, DataModel):
        wcsinfo = wcsinfo_from_model(inp)
        wavetable = None
        spatial_axes, spectral_axes, unknown = gwutils.get_axes(wcsinfo)
        if spectral_axes:
            sp_axis = spectral_axes[0]
            if wcsinfo['CTYPE'][sp_axis] == 'WAVE-TAB':
                wavetable = inp.wavetable
        transform = fitswcs_transform_from_model(wcsinfo, wavetable=wavetable)
        output_frame = frame_from_model(wcsinfo)
    else:
        raise TypeError("Input is expected to be a DataModel instance or a FITS file.")

    if input_frame is None:
        wcsaxes = wcsinfo['WCSAXES']
        if wcsaxes == 2:
            input_frame = cf.Frame2D(name="detector")
        elif wcsaxes == 3:
            input_frame = cf.CoordinateFrame(name="detector", naxes=3,
                axes_order=(0, 1, 2), unit=(u.pix, u.pix, u.pix),
                axes_type=["SPATIAL", "SPATIAL", "SPECTRAL"],
                axes_names=('x', 'y', 'z'), axis_physical_types=None)
        else:
            raise TypeError(f"WCSAXES is expected to be 2 or 3, instead it is {wcsaxes}")
    pipeline = [(input_frame, transform),
               (output_frame, None)]

    wcsobj = wcs.WCS(pipeline)
    return wcsobj


def va_corr_model(datamodel, **kwargs):
    """
    Create transformation that accounts for differential velocity aberration (scale).

    Parameters
    ----------

    datamodel : `~stdatamodels.DataModel`, None
        A full data model. If ``datamodel`` is `None`, all optional parameters
        must be provided (``v2_ref``, ``v3_ref``, and ``va_scale`` - see below).

    Other Parameters
    ----------------

    v2_ref : float
        Can be provided *only when* ``datamodel`` is `None`.

    v3_ref : float
        Can be provided *only when* ``datamodel`` is `None`.

    va_scale : float
        Ratio of the apparent plate scale to the true plate scale.
        Can be provided *only when* ``datamodel`` is `None`.

    Returns
    -------
    va_corr : astropy.modeling.CompoundModel, None
        A 2D compound model that corrects DVA. If input model does not contain
        enough information to compute correction then `None` will be returned.

    """
    if datamodel is None:
        try:
            v2_ref = float(kwargs['v2_ref'])
            v3_ref = float(kwargs['v3_ref'])
            va_scale = float(kwargs['va_scale'])
        except KeyError:
            raise KeyError(
                "'v2_ref', 'v3_ref', and 'va_scale' are all required when "
                "'datamodel' is set to None."
            )
        except ValueError:
            raise TypeError(
                "'v2_ref', 'v3_ref', and 'va_scale' must be numbers."
            )

    else:
        if kwargs:
            raise ValueError(
                "'v2_ref', 'v3_ref', and 'va_scale' cannot be provided when "
                "'datamodel' is not None."
            )
        try:
            va_scale = float(datamodel.meta.velocity_aberration.scale_factor)
            v2_ref = float(datamodel.meta.wcsinfo.v2_ref)
            v3_ref = float(datamodel.meta.wcsinfo.v3_ref)
        except (AttributeError, TypeError):
            return None

    v2_shift = (1 - va_scale) * v2_ref
    v3_shift = (1 - va_scale) * v3_ref

    #  NOTE: it is assumed that v2, v3 angles are small enough so that
    #  for expected scale factors the issue of angle wrapping (180 degrees)
    #  can be neglected.
    va_corr = (
        (Scale(va_scale, name='va_scale_v2') & Scale(va_scale, name='va_scale_v3')) |
        (Shift(v2_shift, name='va_v2_shift') & Shift(v3_shift, name='va_v3_shift'))
    )

    # INFO: Code below does not assume small angle approximation and performs
    # complete computations (Euler rotations and tangent plane projection).
    # This code may be useful if VA code will be used with instruments further
    # away from the optical axis (velocity is assumed parallel to the optical
    # axis).
    #
    # from astropy.modeling.models import Identity
    #
    # s2c = SphericalToCartesian(name='s2c', wrap_lon_at=180)
    # c2s = CartesianToSpherical(name='c2s', wrap_lon_at=180)
    #
    # unit_conv = Scale(1.0 / 3600.0, name='arcsec_to_deg_1D')
    # unit_conv = unit_conv & unit_conv
    # unit_conv.name = 'arcsec_to_deg_2D'
    #
    # va_scale_corr = (Identity(1, name='I_1D') &
    #                  Scale(va_scale, name='va_scale_1Dy') &
    #                  Scale(va_scale, name='va_scale_1Dz'))
    # va_scale_corr.name = 'va_scale_2D'
    #
    # rot = RotationSequence3D(
    #     [v2_ref / 3600.0, -v3_ref / 3600.0],
    #     'zy',
    #     name='det_to_optic_axis'
    # )
    #
    # va_corr = (
    #     unit_conv | s2c | rot | va_scale_corr |
    #     rot.inverse | c2s | unit_conv.inverse
    # )

    va_corr.name = 'VA_Correction'

    return va_corr
