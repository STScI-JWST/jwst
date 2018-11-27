"""Set Telescope Pointing from quaternions"""
import logging
from math import (cos, sin)
import os.path
import sqlite3

import numpy as np

from namedlist import namedlist

from ..datamodels import Level1bModel
from ..lib.engdb_tools import (
    ENGDB_BASE_URL,
    ENGDB_Service,
)


# Setup logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Default transformation matricies
FGS12SIFOV_DEFAULT = np.array(
    [[0.9999994955442, 0.0000000000000, 0.0010044457459],
     [0.0000011174826, 0.9999993811310, -0.0011125359826],
     [-0.0010044451243, 0.0011125365439, 0.9999988766756]]
)

J2FGS_MATRIX_DEFAULT = np.array(
    [[0.999997425983907, 0, -0.002268926080840],
     [0., 1., 0.],
     [0.002268926080840, 0., 0.999997425983907]]
)

SIFOV2V_DEFAULT = np.array(
    [[0.99999742598, 0., 0.00226892608],
     [0., 1., 0.],
     [-0.00226892608, 0., 0.99999742598]]
)

# JWST Exposures that are Fine Guidance exposures that actually
# define the pointing.
FGS_GUIDE_EXP_TYPES = [
    'fgs_acq1',
    'fgs_acq2',
    'fgs_fineguide',
    'fgs_id-image',
    'fgs_id-stack',
    'fgs_track',
]

# Degree, radian, angle transformations
R2D = 180./np.pi
D2R = np.pi/180.
A2R = D2R/3600.
R2A = 3600.*R2D

# SIAF container
SIAF = namedlist(
    'SIAF',
    ['v2ref', 'v3ref', 'v3idlyang', 'vparity'],
    default=None
)

# Pointing container
Pointing = namedlist(
    'Pointing',
    ['q', 'j2fgs_matrix', 'fsmcorr', 'obstime'],
    default=None
)

# Transforms
Transforms = namedlist(
    'Transforms',
    [
        'm_eci2j',            # ECI to J-Frame
        'm_j2fgs1',           # J-Frame to FGS1
        'm_sifov_fsm_delta',  # FSM correction
        'm_fgs12sifov',       # FGS1 to SIFOV
        'm_eci2sifov',        # ECI to SIFOV
        'm_sifov2v',          # SIFOV to V1
        'm_eci2v',            # ECI to V
        'm_v2siaf',           # V to SIAF
        'm_eci2siaf'          # ECI to SIAF
    ],
    default=None
)

# WCS reference container
WCSRef = namedlist(
    'WCSRef',
    ['ra', 'dec', 'pa'],
    default=None
)


def add_wcs(filename, default_pa_v3=0., siaf_path=None, strict_time=False, **transform_kwargs):
    """Add WCS information to a FITS file

    Telescope orientation is attempted to be obtained from
    the engineering database. Failing that, a default pointing
    is used based on proposal target.

    The FITS file is updated in-place.

    Parameters
    ----------
    filename: str
        The path to a data file

    default_pa_v3: float
        The V3 position angle to use if the pointing information
        is not found.

    siaf_path: str or file-like object
        The path to the SIAF database.

    strict_time: bool
        If true, pointing must be within the observation time.
        Otherwise, nearest values are allowed.

    transform_kwargs: dict
        Keyword arguments used by matrix calculation routines
    """
    logger.info('Updating WCS info for file {}'.format(filename))
    model = Level1bModel(filename)
    update_wcs(
        model,
        default_pa_v3=default_pa_v3,
        siaf_path=siaf_path,
        strict_time=strict_time,
        **transform_kwargs
    )
    try:
        _add_axis_3(model)
    except Exception:
        pass
    model.meta.model_type = None
    model.save(filename)
    model.close()
    logger.info('...update completed')


def update_wcs(model, default_pa_v3=0., siaf_path=None, strict_time=False, **transform_kwargs):
    """Update WCS pointing information

    Given a `jwst.datamodels.DataModel`, determine the simple WCS parameters
    from the SIAF keywords in the model and the engineering parameters
    that contain information about the telescope pointing.

    It presumes all the accessed keywords are present (see first block).

    Parameters
    ----------
    model : `~jwst.datamodels.DataModel`
        The model to update.

    default_pa_v3 : float
        If pointing information cannot be retrieved,
        use this as the V3 position angle.

    siaf_path : str
        The path to the SIAF file, i.e. ``XML_DATA`` env variable.

    strict_time: bool
        If True, pointing must be within the observation time.
        Otherwise, nearest values are allowed.

    transform_kwargs: dict
        Keyword arguments used by matrix calculation routines.
    """

    # If the type of exposure is not FGS, then attempt to get pointing
    # from telemetry.
    try:
        exp_type = model.meta.exposure.type.lower()
    except AttributeError:
        exp_type = None
    if exp_type in FGS_GUIDE_EXP_TYPES:
        update_wcs_from_fgs_guiding(
            model, default_pa_v3=default_pa_v3
        )
    else:
        update_wcs_from_telem(
            model, default_pa_v3=default_pa_v3, siaf_path=siaf_path, strict_time=strict_time, **transform_kwargs
        )


def update_wcs_from_fgs_guiding(model, default_pa_v3=0.0, default_vparity=1):
    """ Update WCS pointing from header information

    For Fine Guidance guiding observations, nearly everything
    in the `wcsinfo` meta information is already populated,
    except for the PC matrix. This function updates the PC
    matrix based on the rest of the `wcsinfo`.

    Parameters
    ----------
    model : `~jwst.datamodels.DataModel`
        The model to update.

    default_pa_v3 : float
        If pointing information cannot be retrieved,
        use this as the V3 position angle.

    default_vparity: int
        The default `VIdlParity` to use and should
        be either "1" or "-1". "1" is the
        default since FGS guiding will be using the
        OSS aperture.
    """

    logger.info('Updating WCS for Fine Guidance.')

    # Get position angle
    try:
        pa = model.meta.wcsinfo.pa_v3
    except AttributeError:
        logger.warning(
            'Keyword `PA_V3` not found. Using {} as default value'.format(
                default_pa_v3
            )
        )
        pa = default_pa_v3
    pa_rad = pa * D2R

    # Get VIdlParity
    try:
        vparity = model.meta.wcsinfo.vparity
    except AttributeError:
        logger.warning(
            'Keyword "VPARITY" not found. Using {} as default value'.format(
                default_vparity
            )
        )
        vparity = default_vparity

    (model.meta.wcsinfo.pc1_1,
     model.meta.wcsinfo.pc1_2,
     model.meta.wcsinfo.pc2_1,
     model.meta.wcsinfo.pc2_2) = calc_rotation_matrix(pa_rad, vparity=vparity)


def update_wcs_from_telem(model, default_pa_v3=0., siaf_path=None, strict_time=False, **transform_kwargs):
    """Update WCS pointing information

    Given a `jwst.datamodels.DataModel`, determine the simple WCS parameters
    from the SIAF keywords in the model and the engineering parameters
    that contain information about the telescope pointing.

    It presumes all the accessed keywords are present (see first block).

    Parameters
    ----------
    model : `~jwst.datamodels.DataModel`
        The model to update.

    default_pa_v3 : float
        If pointing information cannot be retrieved,
        use this as the V3 position angle.

    siaf_path : str
        The path to the SIAF file, i.e. ``XML_DATA`` env variable.

    strict_time: bool
        If true, pointing must be within the observation time.
        Otherwise, nearest values are allowed.

    transform_kwargs: dict
        Keyword arguments used by matrix calculation routines.
    """

    logger.info('Updating wcs from telemetry.')

    # Get the SIAF and observation parameters
    obsstart = model.meta.exposure.start_time
    obsend = model.meta.exposure.end_time
    siaf = SIAF(
        v2ref=model.meta.wcsinfo.v2_ref,
        v3ref=model.meta.wcsinfo.v3_ref,
        v3idlyang=model.meta.wcsinfo.v3yangle,
        vparity=model.meta.wcsinfo.vparity
    )

    # Setup default WCS info if actual pointing and calculations fail.
    wcsinfo = WCSRef(
        model.meta.target.ra,
        model.meta.target.dec,
        default_pa_v3
    )
    vinfo = wcsinfo

    # Get the pointing information
    try:
        pointing = get_pointing(obsstart, obsend, strict_time=strict_time)
    except ValueError as exception:
        logger.warning(
            'Cannot retrieve telescope pointing.'
            ' Default pointing parameters will be used.'
            '\nException is {}'.format(exception)
        )
    else:
        # compute relevant WCS information
        logger.info('Successful read of engineering quaternions:')
        logger.info('\tPointing = {}'.format(pointing))
        try:
            wcsinfo, vinfo = calc_wcs(pointing, siaf, **transform_kwargs)
        except Exception as e:
            logger.warning(
                'WCS calculation has failed and will be skipped.'
                'Default pointing parameters will be used.'
                '\nException is {}'.format(e)
            )

    logger.info('Aperture WCS info: {}'.format(wcsinfo))
    logger.info('V1 WCS info: {}'.format(vinfo))

    # Update V1 pointing
    model.meta.pointing.ra_v1 = vinfo.ra
    model.meta.pointing.dec_v1 = vinfo.dec
    model.meta.pointing.pa_v3 = vinfo.pa

    # Update Aperture pointing
    model.meta.aperture.position_angle = wcsinfo.pa
    model.meta.wcsinfo.crval1 = wcsinfo.ra
    model.meta.wcsinfo.crval2 = wcsinfo.dec
    model.meta.wcsinfo.ra_ref = wcsinfo.ra
    model.meta.wcsinfo.dec_ref = wcsinfo.dec
    model.meta.wcsinfo.roll_ref = compute_local_roll(
        vinfo.pa, wcsinfo.ra, wcsinfo.dec, siaf.v2ref, siaf.v3ref
    )
    (model.meta.wcsinfo.pc1_1,
     model.meta.wcsinfo.pc1_2,
     model.meta.wcsinfo.pc2_1,
     model.meta.wcsinfo.pc2_2) = calc_rotation_matrix(
         wcsinfo.pa * D2R, vparity=siaf.vparity
     )

    wcsaxes = model.meta.wcsinfo.wcsaxes
    if wcsaxes is None:
        wcsaxes = 2
    model.meta.wcsinfo.wcsaxes = max(2, wcsaxes)

    # Calculate S_REGION with the footprint
    # information
    try:
        update_s_region(model, prd_db_filepath=siaf_path)
    except Exception as e:
        logger.warning(
            'Calculation of S_REGION failed and will be skipped.'
            '\nException is {}'.format(e)
        )


def update_s_region(model, prd_db_filepath=None):
    """Update ``S_REGION`` sky footprint information.

    The ``S_REGION`` keyword is intended to store the spatial footprint of
    an observation using the VO standard STCS representation.

    Parameters
    ----------
    model : `~jwst.datamodels.DataModel`
        The model to update in-place.
    prd_db_filepath : str
        The filepath to the SIAF file (PRD database).
        If None, attempt to get the path from the ``XML_DATA`` environment variable.
    """
    if prd_db_filepath is None:
        try:
            prd_db_filepath = os.path.join(os.environ['XML_DATA'], "prd.db")
        except KeyError:
            logger.info("Unknown path to PRD DB file: {0}".format(prd_db_filepath))
            return
    if not os.path.exists(prd_db_filepath):
        logger.info("Invalid path to PRD DB file: {0}".format(prd_db_filepath))
        return
    aperture_name = model.meta.aperture.name
    useafter = model.meta.observation.date
    vertices = _get_vertices_idl(
        aperture_name, useafter, prd_db_filepath=prd_db_filepath
    )
    vertices = list(vertices.values())[0]
    xvert = vertices[:4]
    yvert = vertices[4:]
    logger.info(
        "Vertices for aperture {0}: {1}".format(aperture_name, vertices)
    )
    # Execute IdealToV2V3, followed by V23ToSky
    from ..transforms.models import IdealToV2V3, V23ToSky
    v2_ref_deg = model.meta.wcsinfo.v2_ref / 3600 # in deg
    v3_ref_deg = model.meta.wcsinfo.v3_ref / 3600 # in deg
    roll_ref = model.meta.wcsinfo.roll_ref
    ra_ref = model.meta.wcsinfo.ra_ref
    dec_ref = model.meta.wcsinfo.dec_ref
    vparity = model.meta.wcsinfo.vparity
    v3yangle = model.meta.wcsinfo.v3yangle

    # V2_ref and v3_ref should be in arcsec
    idltov23 = IdealToV2V3(
        v3yangle,
        model.meta.wcsinfo.v2_ref, model.meta.wcsinfo.v2_ref,
        vparity
    )
    v2, v3 = idltov23(xvert, yvert)  # in arcsec

    # Convert to deg
    v2 = v2 / 3600 # in deg
    v3 = v3 / 3600 # in deg
    angles = [-v2_ref_deg, v3_ref_deg, -roll_ref, -dec_ref, ra_ref]
    axes = "zyxyz"
    v23tosky = V23ToSky(angles, axes_order=axes)
    ra_vert, dec_vert = v23tosky(v2, v3)
    negative_ind = ra_vert < 0
    ra_vert[negative_ind] = ra_vert[negative_ind] + 360
    # Do not do any sorting, use the vertices in the SIAF order.
    footprint = np.array([ra_vert, dec_vert]).T
    s_region = (
        "POLYGON ICRS "
        " {0} {1}"
        " {2} {3}"
        " {4} {5}"
        " {6} {7}".format(*footprint.flatten()))
    model.meta.wcsinfo.s_region = s_region


def calc_wcs(pointing, siaf, **transform_kwargs):
    """Transform from the given SIAF information and Pointing
    the aperture and V1 wcs

    Parameters
    ----------
    siaf: SIAF
        The SIAF transformation. See ref:`Notes` for further details

    pointing: Pointing
        The telescope pointing. See ref:`Notes` for further details

    transform_kwargs: dict
        Keyword arguments used by matrix calculation routines

    Returns
    -------
    (wcsinfo, vinfo): (WCSRef, WCSRef)
        A 2-tuple is returned with the WCS pointing for
        the aperture and the V1 axis

    Notes
    -----

    The SIAF information is as follows:

    v2ref (arcsec), v3ref (arcsec), v3idlyang (deg), vidlparity
    (+1 or -1), are the relevant siaf parameters. The assumed
    units are shown in parentheses.

    It is assumed that the siaf ref position is the corresponding WCS
    reference position.

    The `Pointing` information is as follows:

    Parameter q is the SA_ZATTEST<n> engineering parameters where
    n ranges from 1 to 4.

    Parameter j2fgs_matrix is the transformation matrix specified by
    engineering parameters SA_ZRFGS2J<n><m> where both n and m range
    from 1 to 3. This is to be provided as a 1d list using this order:
    11, 21, 31, 12, 22, 32, 13, 23, 33

    Parameter fsmcorr are two values provided as a list consisting of:
    [SA_ZADUCMDX, SA_ZADUCMDY]
    """

    # Calculate transforms
    tforms = calc_transforms(pointing, siaf, **transform_kwargs)

    # Calculate the V1 WCS information
    vinfo = calc_v1_wcs(tforms.m_eci2v)

    # Calculate the Aperture WCS
    wcsinfo = calc_aperture_wcs(tforms.m_eci2siaf)

    # That's all folks
    return (wcsinfo, vinfo)


def calc_transforms(pointing, siaf, fsmcorr_version='latest'):
    """Calculate transforms from pointing to SIAF

    Given the spacecraft pointing parameters and the
    aperture-specific SIAF, calculate all the transforms
    necessary to produce WCS information.

    Parameters
    ----------
    pointing: Pointing
        Observatory pointing information

    siaf: SIAF
        Aperture information

    fsmcorr_version: str
        The version of the FSM correction calculation to use.
        See :ref:`calc_sifov_fsm_delta_matrix`

    Returns
    -------
    transforms: Transforms
        The list of coordinate matrix transformations

    Notes
    -----
    The matrix transform pipeline to convert from ECI J2000 observatory
    qauternion pointing to aperture ra/dec/roll information
    is given by the following formula. Each term is a 3x3 matrix:

        M_eci_to_siaf =           # The complete transformation
            M_v1_to_siaf      *   # V1 to SIAF
            M_sifov_to_v1     *   # Science Instruments Aperture to V1
            M_sifov_fsm_delta *   # Fine Steering Mirror correction
            M_fgs1_to_sifov   *   # FGS1 to Science Instruments Aperture
            M_j_to_fgs1       *   # J-Frame to FGS1
            M_eci_to_j        *   # ECI to J-Frame
    """

    tforms = Transforms()

    # Determine the ECI to J-frame matrix
    tforms.m_eci2j = calc_eci2j_matrix(pointing.q)

    # Calculate the J-frame to FGS! ICS matrix
    tforms.m_j2fgs1 = calc_j2fgs1_matrix(pointing.j2fgs_matrix)

    # Calculate the FSM corrections to the SI_FOV frame
    tforms.m_sifov_fsm_delta = calc_sifov_fsm_delta_matrix(
        pointing.fsmcorr, fsmcorr_version=fsmcorr_version
    )

    # Calculate the FGS1 ICS to SI-FOV matrix
    tforms.m_fgs12sifov = calc_fgs1_to_sifov_mastrix()

    # Calculate ECI to SI FOV
    tforms.m_eci2sifov = np.dot(
        tforms.m_sifov_fsm_delta,
        np.dot(
            tforms.m_fgs12sifov,
            np.dot(
                tforms.m_j2fgs1,
                tforms.m_eci2j
            )
        )
    )

    # Calculate SI FOV to V1 matrix
    tforms.m_sifov2v = calc_sifov2v_matrix()

    # Calculate the complete transform to the V1 reference
    tforms.m_eci2v = np.dot(
        tforms.m_sifov2v,
        tforms.m_eci2sifov
    )

    # Calculate the SIAF transform matrix
    tforms.m_v2siaf = calc_v2siaf_matrix(siaf)

    # Calculate the full ECI to SIAF transform matrix
    tforms.m_eci2siaf = np.dot(
        tforms.m_v2siaf,
        tforms.m_eci2v
    )

    return tforms


def calc_v1_wcs(m_eci2v):
    """Calculate the V1 wcs information

    Parameters
    ----------
    m_eci2v: np.array((3, 3))
        The ECI to V transformation matrix

    Returns
    -------
    vinfo: WCSRef
        The V1 wcs pointing
    """
    vinfo = WCSRef()

    # V1 RA/Dec is the first row of the transform
    vinfo.ra, vinfo.dec = vector_to_ra_dec(m_eci2v[0])

    # V3 is the third row of the transformation
    v3info = WCSRef()
    v3info.ra, v3info.dec = vector_to_ra_dec(m_eci2v[2])

    # Calculate the V3 position angle
    vinfo.pa = calc_position_angle(vinfo, v3info)

    # Convert to degrees
    vinfo = WCSRef(
        ra=vinfo.ra * R2D,
        dec=vinfo.dec * R2D,
        pa=vinfo.pa * R2D
    )

    return vinfo


def calc_aperture_wcs(m_eci2siaf):
    """Calculate the aperture WCS

    Parameters
    ----------
    m_eci2siaf: np.array((3, 3))
        The ECI to SIAF transformation matrix

    Returns
    -------
    wcsinfo: WCSRef
        The aperturn wcs information
    """

    # Calculate point on sky.
    # Note, the SIAF referenct point is hardcoded to
    # (0, 0). The calculation is left here in case
    # this is not desired.
    wcsinfo = WCSRef()
    siaf_x = 0. * A2R
    siaf_y = 0. * A2R
    refpos = np.array(
        [siaf_x,
         siaf_y,
         np.sqrt(1.-siaf_x * siaf_x - siaf_y * siaf_y)]
    )
    msky = np.dot(m_eci2siaf.transpose(), refpos)
    wcsinfo.ra, wcsinfo.dec = vector_to_ra_dec(msky)

    # Calculate the position angle
    vysiaf = np.array([0., 1., 0.])
    myeci = np.dot(m_eci2siaf.transpose(), vysiaf)

    # The Y axis of the aperture is given by
    vy_ra, vy_dec = vector_to_ra_dec(myeci)

    # The VyPA @ xref,yref is given by
    y = cos(vy_dec) * sin(vy_ra-wcsinfo.ra)
    x = sin(vy_dec) * cos(wcsinfo.dec) - \
        cos(vy_dec) * sin(wcsinfo.dec) * cos((vy_ra - wcsinfo.ra))
    wcsinfo.pa = np.arctan2(y, x)

    # Convert all WCS to degrees
    wcsinfo = WCSRef(
        ra=wcsinfo.ra * R2D,
        dec=wcsinfo.dec * R2D,
        pa=wcsinfo.pa * R2D
    )

    return wcsinfo


def calc_eci2j_matrix(q):
    """Calculate ECI to J-frame matrix from quaternions

    Parameters
    ----------
    q: np.array(q1, q2, q3, q4)
        Array of quaternions from the engineering database

    Returns
    -------
    transform: np.array((3, 3))
        The transform matrix representing the transformation
        from observatory orientation to J-Frame
    """
    q1, q2, q3, q4 = q
    transform = np.array(
        [[1. - 2.*q2*q2 - 2.*q3*q3,
          2.*(q1*q2 + q3*q4),
          2.*(q3*q1 - q2*q4)],
         [2.*(q1*q2 - q3*q4),
          1. - 2.*q3*q3 - 2.*q1*q1,
          2.*(q2*q3 + q1*q4)],
         [2.*(q3*q1 + q2*q4),
          2.*(q2*q3 - q1*q4),
          1. - 2.*q1*q1 - 2.*q2*q2]]
    )

    return transform


def calc_j2fgs1_matrix(j2fgs_matrix):
    """Calculate the J-frame to FGS1 transformation

    Parameters
    ----------
    j2fgs_matrix: n.array((9,))
        Matrix parameters from the engineering database.
        If all zeros, a predefined matrix is used.

    Returns
    -------
    transform: np.array((3, 3))
        The transformation matrix
    """
    if np.isclose(j2fgs_matrix, 0.).all():
        logger.warning(
            'J-Frame to FGS1 engineering parameters are all zero.'
            '\nUsing default matrix'
        )
        m_partial = np.asarray(
            [
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]
            ]
        )
        transform = np.dot(
            m_partial,
            J2FGS_MATRIX_DEFAULT
        )

    else:
        logger.info(
            'Using J-Frame to FGS1 engineering parameters'
            'for the J-Frame to FGS1 transformation.'
        )
        transform = np.array(j2fgs_matrix).reshape((3, 3)).transpose()

    return transform


def calc_sifov_fsm_delta_matrix(fsmcorr, fsmcorr_version='latest'):
    """Calculate Fine Steering Mirror correction matrix

    Parameters
    ----------
    fsmcorr: np.array((2,))
        The FSM correction parameters:
            0: SA_ZADUCMDX
            1: SA_ZADUCMDY

    fsmcorr_version: str
        The version of the FSM correction calculation to use.
        Versions available:
            latest: The state-of-art. Currently `v2`
            v2: Update 201708 to use actual spherical calculations
            v1: Original linear approximation

    Returns
    -------
    transform: np.array((3, 3))
        The transformation matrix
    """
    version = fsmcorr_version.lower()
    logger.debug('Using version {}'.format(version))

    x = fsmcorr[0]  # SA_ZADUCMDX
    y = fsmcorr[1]  # SA_ZADUCMDY

    # `V1`: Linear approximation calcuation
    if version == 'v1':
        transform = np.array(
            [
                [1.,       x/22.01, y/21.68],
                [-x/22.01, 1.,      0.],
                [-y/21.68, 0.,      1.]
            ]
        )

    # Default or `V2`: Direct spherical calculation
    # Note: With the "0.0" in the lower middle Y transform
    else:
        if version not in ('latest', 'v2'):
            logger.warning(
                'Unknown version "{}" specified.'
                ' Using the latest (spherical) calculation.'
            )
        m_x_partial = np.array(
            [
                [1., 0.,      0.],
                [0., cos(x),  sin(x)],
                [0., -sin(x), cos(x)]
            ]
        )
        m_y_partial = np.array(
            [
                [cos(y), 0., -sin(y)],
                [0.,     1., 0.],
                [sin(y), 0., cos(y)]
            ]
        )
        transform = np.dot(m_x_partial, m_y_partial)

    return transform


def calc_fgs1_to_sifov_mastrix():
    """
    Calculate the FGS! to SI-FOV matrix

    Currently, this is a defined matrix
    """
    m_partial = np.array(
        [[0, 0, 1],
         [1, 0, 0],
         [0, 1, 0]]
    )

    transform = np.dot(m_partial, FGS12SIFOV_DEFAULT)
    return transform


def calc_sifov2v_matrix():
    """Calculate the SI-FOV to V-Frame matrix

    This is currently defined as the inverse Euler rotation
    about an angle of 7.8 arcmin. Here returns the pre-calculate
    matrix.
    """
    return SIFOV2V_DEFAULT


def calc_v2siaf_matrix(siaf):
    """Calculate the SIAF transformation matrix

    Parameters
    ----------
    siaf: SIAF
        The SIAF parameters

    Returns
    -------
    transform: np.array((3, 3))
        The V1 to SIAF transformation matrix
    """
    v2, v3, v3idlyang, vparity = siaf
    v2 *= A2R
    v3 *= A2R
    v3idlyang *= D2R

    mat = np.array(
        [[cos(v3)*cos(v2),
          cos(v3)*sin(v2),
          sin(v3)],
         [-cos(v3idlyang)*sin(v2)+sin(v3idlyang)*sin(v3)*cos(v2),
          cos(v3idlyang)*cos(v2)+sin(v3idlyang)*sin(v3)*sin(v2),
          -sin(v3idlyang)*cos(v3)],
         [-sin(v3idlyang)*sin(v2)-cos(v3idlyang)*sin(v3)*cos(v2),
          sin(v3idlyang)*cos(v2)-cos(v3idlyang)*sin(v3)*sin(v2),
          cos(v3idlyang)*cos(v3)]])
    pmat = np.array([[0., vparity, 0.],
                     [0., 0., 1.],
                     [1., 0., 0.]])

    transform = np.dot(pmat, mat)
    return transform


def calc_position_angle(v1, v3):
    """Calculate V3 position angle @V1

    Parameters
    ----------
    v1: WCSRef
        The V1 wcs parameters

    v3: WCSRef
        The V3 wcs parameters

    Returns
    -------
    v3_pa: float
      The V3 position angle, in radians
    """
    y = cos(v3.dec) * sin(v3.ra-v1.ra)
    x = sin(v3.dec) * cos(v1.dec) - \
        cos(v3.dec) * sin(v1.dec) * cos((v3.ra - v1.ra))
    v3_pa = np.arctan2(y, x)

    return v3_pa


def get_pointing(obsstart, obsend, strict_time=False, reduce_func=None):
    """
    Get telescope pointing engineering data.

    Parameters
    ----------
    obsstart, obsend: float
        MJD observation start/end times

    strict_time: bool
        If true, pointing must be within the observation time.
        Otherwise, nearest values are allowed.

    reduce_func: func or None
        Reduction function to use on values.
        If None, the full list of `Pointing`s
        is returned.

    Returns
    -------
    pointing: Pointing or [Pointing(, ...)]
        The engineering pointing parameters.
        If the `result_type` is `all`, a list
        of pointings will be returned

    Raises
    ------
    ValueError
        Cannot retrieve engineering information

    Notes
    -----
    For the moment, the first found values will be used.
    This will need be re-examined when more information is
    available.
    """
    logger.info(
        'Determining pointing between observations times (mjd):'
        '\n\tobsstart = {}'
        '\n\tobsend = {}'.format(obsstart, obsend)
    )
    logger.info(
        'Querying engineering DB: {}'.format(ENGDB_BASE_URL)
    )
    try:
        engdb = ENGDB_Service()
    except Exception as exception:
        raise ValueError(
            'Cannot open engineering DB connection'
            '\nException: {}'.format(
                exception
            )
        )
    params = {
        'SA_ZATTEST1':  None,
        'SA_ZATTEST2':  None,
        'SA_ZATTEST3':  None,
        'SA_ZATTEST4':  None,
        'SA_ZRFGS2J11': None,
        'SA_ZRFGS2J21': None,
        'SA_ZRFGS2J31': None,
        'SA_ZRFGS2J12': None,
        'SA_ZRFGS2J22': None,
        'SA_ZRFGS2J32': None,
        'SA_ZRFGS2J13': None,
        'SA_ZRFGS2J23': None,
        'SA_ZRFGS2J33': None,
        'SA_ZADUCMDX':  None,
        'SA_ZADUCMDY':  None,
    }

    # First try go retrieve values without the database bracket values.
    for param in params:
        try:
            params[param] = engdb.get_values(
                param, obsstart, obsend,
                time_format='mjd', include_obstime=True
            )
        except Exception as exception:
            raise ValueError(
                'Cannot retrive {} from engineering.'
                '\nFailure was {}'.format(
                    param,
                    exception
                )
            )

    # For any parameters that did not have values, re-retrieve with
    # the bracket values.
    if not strict_time:
        for param in params:
            if not len(params[param]):
                params[param] = engdb.get_values(
                    param, obsstart, obsend,
                    time_format='mjd', include_obstime=True, include_bracket_values=True
                )

    # Find the first set of non-zero values
    results = []
    for idx in range(len(params['SA_ZATTEST1'])):
        values = [
            params[param][idx].value
            for param in params
        ]
        if any(values):
            pointing = Pointing()

            # The tagged obstime will come from the SA_ZATTEST1 mneunonic
            pointing.obstime = params['SA_ZATTEST1'][idx].obstime

            # Fill out the matricies
            pointing.q = np.array([
                params['SA_ZATTEST1'][idx].value,
                params['SA_ZATTEST2'][idx].value,
                params['SA_ZATTEST3'][idx].value,
                params['SA_ZATTEST4'][idx].value,
            ])

            pointing.j2fgs_matrix = np.array([
                params['SA_ZRFGS2J11'][idx].value,
                params['SA_ZRFGS2J21'][idx].value,
                params['SA_ZRFGS2J31'][idx].value,
                params['SA_ZRFGS2J12'][idx].value,
                params['SA_ZRFGS2J22'][idx].value,
                params['SA_ZRFGS2J32'][idx].value,
                params['SA_ZRFGS2J13'][idx].value,
                params['SA_ZRFGS2J23'][idx].value,
                params['SA_ZRFGS2J33'][idx].value,
            ])

            pointing.fsmcorr = np.array([
                params['SA_ZADUCMDX'][idx].value,
                params['SA_ZADUCMDY'][idx].value,

            ])

            results.append(pointing)

            # Short circuit if all we're looking for is the first.
            break

    if not len(results):
        raise ValueError(
                'No non-zero quanternion found '
                'in the DB between MJD {} and {}'.format(obsstart, obsend)
            )

    return results[0]


def vector_to_ra_dec(v):
    """Returns tuple of spherical angles from unit direction Vector

    Parameters
    ----------
    v: [v0, v1, v2]

    Returns
    -------
    ra, dec: float, float
        The spherical angles, in radians
    """
    ra = np.arctan2(v[1], v[0])
    dec = np.arcsin(v[2])
    if ra < 0.:
        ra += 2. * np.pi
    return(ra, dec)


def compute_local_roll(pa_v3, ra_ref, dec_ref, v2_ref, v3_ref):
    """
    Computes the position angle of V3 (measured N to E)
    at the center af an aperture.

    Parameters
    ----------
    pa_v3 : float
        Position angle of V3 at (V2, V3) = (0, 0) [in deg]
    v2_ref, v3_ref : float
        Reference point in the V2, V3 frame [in arcsec]
    ra_ref, dec_ref : float
        RA and DEC corresponding to V2_REF and V3_REF, [in deg]

    Returns
    -------
    new_roll : float
        The value of ROLL_REF (in deg)

    """
    v2 = np.deg2rad(v2_ref / 3600)
    v3 = np.deg2rad(v3_ref / 3600)
    ra_ref = np.deg2rad(ra_ref)
    dec_ref = np.deg2rad(dec_ref)
    pa_v3 = np.deg2rad(pa_v3)

    M = np.array(
        [[cos(ra_ref) * cos(dec_ref),
          -sin(ra_ref) * cos(pa_v3) + cos(ra_ref) * sin(dec_ref) * sin(pa_v3),
          -sin(ra_ref) * sin(pa_v3) - cos(ra_ref) * sin(dec_ref) * cos(pa_v3)],
         [sin(ra_ref) * cos(dec_ref),
          cos(ra_ref) * cos(pa_v3) + sin(ra_ref) * sin(dec_ref) * sin(pa_v3),
          cos(ra_ref) * sin(pa_v3) - sin(ra_ref) * sin(dec_ref) * cos(pa_v3)],
         [sin(dec_ref),
          -cos(dec_ref) * sin(pa_v3),
          cos(dec_ref) * cos(pa_v3)]]
    )

    return _roll_angle_from_matrix(M, v2, v3)


def _roll_angle_from_matrix(matrix, v2, v3):
    X = -(matrix[2, 0] * np.cos(v2) + matrix[2, 1] * np.sin(v2)) * np.sin(v3) + matrix[2, 2] * np.cos(v3)
    Y = (matrix[0, 0] *  matrix[1, 2] - matrix[1, 0] * matrix[0, 2]) * np.cos(v2) + \
      (matrix[0, 1] * matrix[1, 2] - matrix[1, 1] * matrix[0, 2]) * np.sin(v2)
    new_roll = np.rad2deg(np.arctan2(Y, X))
    if new_roll < 0:
        new_roll += 360
    return new_roll


def _get_vertices_idl(aperture_name, useafter, prd_db_filepath=None):
    prd_db_filepath = "file:{0}?mode=ro".format(prd_db_filepath)
    logger.info("Using SIAF database from {}".format(prd_db_filepath))
    logger.info("Getting aperture vertices for aperture "
             "{0} with USEAFTER {1}".format(aperture_name, useafter))
    aperture = (aperture_name, useafter)

    RESULT = {}
    try:
        PRD_DB = sqlite3.connect(prd_db_filepath, uri=True)

        cursor = PRD_DB.cursor()
        cursor.execute("SELECT Apername, XIdlVert1, XIdlVert2, XIdlVert3, XIdlVert4, "
                       "YIdlVert1, YIdlVert2, YIdlVert3, YIdlVert4 "
                       "FROM Aperture WHERE Apername = ? and UseAfterDate <= ? ORDER BY UseAfterDate LIMIT 1", aperture)
        for row in cursor:
            RESULT[row[0]] = tuple(row[1:9])
        PRD_DB.commit()
    except sqlite3.Error as err:
        print("Error" + err.args[0])
        raise
    finally:
        if PRD_DB:
            PRD_DB.close()
    logger.info("loaded {0} table rows from {1}".format(len(RESULT), prd_db_filepath))
    return RESULT


def _add_axis_3(model):
    """
    Adds CTYPE3 and CUNIT3 for spectral observations.
    This is temporary, may be moved to SDP proccessing later.
    """
    wcsaxes = model.meta.wcsinfo.wcsaxes
    if wcsaxes is not None and wcsaxes == 3:
        model.meta.wcsinfo.ctype3 = "WAVE"
        model.meta.wcsinfo.cunit3 = "um"


def calc_rotation_matrix(angle, vparity=1):
    """ Calculate the rotation matrix

    Parameters
    ----------
    angle: float in radians
        The angle to create the matrix

    vparity: int
        The x-axis parity, usually taken from
        the JWST SIAF parameter VIdlParity.
        Value should be "1" or "-1".

    Returns
    -------
    matrix: [pc1_1, pc1_2, pc2_1, pc2_2]
        The rotation matrix

    Notes
    -----
    The rotation is

       ----------------
       | pc1_1  pc2_1 |
       | pc1_2  pc2_2 |
       ----------------

    where:
        pc1_1 = vparity * cos(angle)
        pc1_2 = sin(angle)
        pc2_1 = -1 * vparity * sin(angle)
        pc2_2 = cos(angle)
    """

    pc1_1 = vparity * cos(angle)
    pc1_2 = sin(angle)
    pc2_1 = vparity * -sin(angle)
    pc2_2 = cos(angle)

    return [pc1_1, pc1_2, pc2_1, pc2_2]
