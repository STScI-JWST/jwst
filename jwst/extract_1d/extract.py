import abc
import logging
import copy
import json
import math
import numpy as np

from typing import Union, Tuple, NamedTuple, List
from astropy.modeling import polynomial
from gwcs import WCS
from stdatamodels import DataModel
from stdatamodels.ndmodel import MetaNode

from .. import datamodels
from ..datamodels import dqflags, SlitModel, SpecModel
from ..datamodels.apcorr import (
    MirLrsApcorrModel, MirMrsApcorrModel, NrcWfssApcorrModel, NrsFsApcorrModel,
    NrsMosApcorrModel, NrsIfuApcorrModel, NisWfssApcorrModel
)

from ..assign_wcs import niriss         # for specifying spectral order number
from ..assign_wcs.util import wcs_bbox_from_shape
from ..lib import pipe_utils
from ..lib.wcs_utils import get_wavelengths
from . import extract1d
from . import ifu
from . import spec_wcs
from .apply_apcorr import select_apcorr

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# WFSS_EXPTYPES = ['NIS_WFSS', 'NRC_WFSS', 'NRC_GRISM', 'NRC_TSGRISM']
WFSS_EXPTYPES = ['NIS_WFSS', 'NRC_WFSS', 'NRC_GRISM']
"""Exposure types to be regarded as wide-field slitless spectroscopy."""

# These values are used to indicate whether the input extract1d reference file
# (if any) is JSON, IMAGE or ASDF (added for IFU data)
FILE_TYPE_JSON = "JSON"
FILE_TYPE_IMAGE = "IMAGE"
FILE_TYPE_ASDF = "ASDF"
FILE_TYPE_OTHER = "N/A"

# This is to prevent calling offset_from_offset multiple times for multi-integration data.
OFFSET_NOT_ASSIGNED_YET = "not assigned yet"

ANY = "ANY"
"""Wildcard for slit name.

Extended summary
----------------
For full-frame input data, keyword SLTNAME may not be populated, so the
slit name will be set to this string to indicate that the first slit in
the reference file should be used.
A slit name in the extract1d reference file can also be ANY, in which case the
reference information for that slit will be regarded as matching any slit
name from the input data.
"""

ANY_ORDER = 1000
"""Wildcard for spectral order number in a reference image.

Extended summary
----------------
If the extract1d reference file contains images, keyword SPORDER gives the order
number of the spectrum that would be extracted using a given image in
the reference file.
"""

HORIZONTAL = 1
VERTICAL = 2
"""Dispersion direction, predominantly horizontal or vertical."""

# This is intended to be larger than any possible distance (in pixels) between the target and any point in the image;
# used by locn_from_wcs().
HUGE_DIST = 1.e20

# These values are assigned in get_extract_parameters, using key "match".
# If there was an aperture in the reference file for which the "id" key matched, that's (at least) a partial match.
# If "spectral_order" also matched, that's an exact match.
NO_MATCH = "no match"
PARTIAL = "partial match"
EXACT = "exact match"


class Aperture(NamedTuple):  # When python 3.6 is no longer supported, consider converting to DataClass
    xstart: Union[int, float]
    xstop: Union[int, float]
    ystart: Union[int, float]
    ystop: Union[int, float]


class Extract1dError(Exception):
    pass


class InvalidSpectralOrderNumberError(Extract1dError):
    """The spectral order number was invalid or off the detector."""
    pass


def open_extract1d_ref(refname: str, exptype: str) -> dict:
    """Open the extract1d reference file.

    Parameters
    ----------
    refname : str
        The name of the extract1d reference file.  This file is expected to be
        a JSON file or ASDF file  giving extraction information, or a file
        containing one or more images that are to be used as masks that
        define the extraction region and optionally background regions.

    Returns
    -------
    ref_dict : dict
        If the extract1d reference file is in JSON format, ref_dict will be the
        dictionary returned by json.load(), except that the file type
        ('JSON') will also be included with key 'ref_file_type'.
        If the extract1d reference file is in asdf format, the ref_dict will
        be a dictionary  containing two keys: ref_dict['ref_file_type'] = 'ASDF'
        and ref_dict['ref_model'].
        If the reference file is an image, ref_dict will be a
        dictionary with two keys:  ref_dict['ref_file_type'] = 'IMAGE'
        and ref_dict['ref_model'].  The latter will be the open file
        handle for the jwst.datamodels object for the extract1d file.
    """

    if refname == "N/A":
        ref_dict = None
    elif exptype in ['MIR_MRS', 'NRS_IFU']:
        # read in asdf file
        extract_model = datamodels.Extract1dIFUModel(refname)
        ref_dict = {}
        ref_dict['ref_file_type'] = FILE_TYPE_ASDF
        ref_dict['ref_model'] = extract_model

    else:
        # Try reading the file as JSON.
        fd = open(refname)
        try:
            ref_dict = json.load(fd)
            ref_dict['ref_file_type'] = FILE_TYPE_JSON
            fd.close()
        except UnicodeDecodeError:
            fd.close()
            # Try opening the file as a reference image.
            try:
                fd = datamodels.MultiExtract1dImageModel(refname)
                ref_dict = {'ref_file_type': FILE_TYPE_IMAGE, 'ref_model': fd}
            except OSError:
                log.info("The extract1d reference file should be JSON or FITS.")
                log.error(f"Don't know how to read {refname}.")
                raise

    return ref_dict


def open_apcorr_ref(refname: str, exptype: str) -> DataModel:
    """Determine the appropriate DataModel class to use when opening the input APCORR reference file.

    Parameters
    ----------
    refname : str
        Path of the APCORR reference file

    exptype : str
        EXPTYPE of the input to the extract_1d step.

    Returns
    -------
    Opened APCORR DataModel.

    Notes
    -----
    This function should be removed after the DATAMODL keyword is required for the APCORR reference file.

    """
    apcorr_model_map = {
        'MIR_LRS-FIXEDSLIT': MirLrsApcorrModel,
        'MIR_LRS-SLITLESS': MirLrsApcorrModel,
        'MIR_MRS': MirMrsApcorrModel,
        'NRC_GRISM': NrcWfssApcorrModel,
        'NRC_WFSS': NrcWfssApcorrModel,
        'NIS_WFSS': NisWfssApcorrModel,
        'NRS_BRIGHTOBJ': NrsFsApcorrModel,
        'NRS_FIXEDSLIT': NrsFsApcorrModel,
        'NRS_IFU': NrsIfuApcorrModel,
        'NRS_MSASPEC': NrsMosApcorrModel
    }

    apcorr_model = apcorr_model_map[exptype]
    return apcorr_model(refname)


def get_extract_parameters(
        ref_dict: Union[dict, None],
        input_model: DataModel,
        slitname: str,
        sp_order: int,
        meta: MetaNode,
        smoothing_length: Union[int, None],
        bkg_fit: str,
        bkg_order: Union[int, None],
        use_source_posn: Union[bool, None]
) -> dict:
    """Get extract1d reference file values.

    Parameters
    ----------
    ref_dict : dict or None
        For an extract1d reference file in JSON format, `ref_dict` will be the entire
        contents of the file.  For an EXTRACT1D reference image, `ref_dict` will have
        just two entries, 'ref_file_type' (a string) and 'ref_model', a
        JWST data model for a collection of images.  If there is no
        extract1d reference file, `ref_dict` will be None.

    input_model : data model
        This can be either the input science file or one SlitModel out of
        a list of slits.

    slitname : str
        The name of the slit, or "ANY", or for NIRISS SOSS data this will
        be the subarray name.

    sp_order : int
        The spectral order number.

    meta : metadata for the actual input model, i.e. not just for the
        current slit.
        Not currently used.

    smoothing_length : int or None
        Width of a boxcar function for smoothing the background regions.
        If None, the smoothing length will be gotten from `ref_dict`, or
        it will be set to 0 (no background smoothing) if this key is
        not found in `ref_dict`.
        If `smoothing_length` is not None, that means that the user
        explicitly specified the value, so that value will be used.
        This argument is only used if background regions have been
        specified.

    bkg_fit : str
        The type of fit to apply to background values in each
        column (or row, if the dispersion is vertical). The default
        `poly` results in a polynomial fit of order `bkg_order`. Other
        options are `mean` and `median`. If `mean` or `median` is selected,
        `bkg_order` is ignored.

    bkg_order : int or None
        Polynomial order for fitting to each column (or row, if the
        dispersion is vertical) of background.  If None, the polynomial
        order will be gotten from `ref_dict`, or it will be set to 0 if
        not found in `ref_dict`.
        A value of 0 means that a simple average of the background
        regions, column by column (or row by row), will be used.
        If `bkg_order` is not None, that means that the user explicitly
        specified the value, so that value will be used.
        This argument must be positive or zero, and it is only used if
        background regions have been specified.

    use_source_posn : bool or None
        If True, the target and background positions specified in `ref_dict`
        (or a default target position) will be shifted to account for
        the actual source location in the data.
        If None, the value specified in `ref_dict` will be used, or it will
        be set to True if not found in `ref_dict`.

    Returns
    -------
    extract_params : dict
        Information copied out of `ref_dict`.  The items will be selected
        based on `slitname` and `sp_order`.  Default values will be
        assigned if `ref_dict` is None.  For a reference image, the key
        'ref_image' gives the (open) image model.
    """

    extract_params = {'match': NO_MATCH}  # initial value
    if ref_dict is None:
        # There is no extract1d reference file; use "reasonable" default values.
        extract_params['ref_file_type'] = FILE_TYPE_OTHER
        extract_params['match'] = EXACT
        shape = input_model.data.shape
        extract_params['spectral_order'] = sp_order
        extract_params['xstart'] = 0  # first pixel in X
        extract_params['xstop'] = shape[-1] - 1  # last pixel in X
        extract_params['ystart'] = 0  # first pixel in Y
        extract_params['ystop'] = shape[-2] - 1  # last pixel in Y
        extract_params['extract_width'] = None
        extract_params['src_coeff'] = None
        extract_params['bkg_coeff'] = None  # no background sub.
        extract_params['smoothing_length'] = 0  # because no background sub.
        extract_params['bkg_fit'] = None  # because no background sub.
        extract_params['bkg_order'] = 0  # because no background sub.
        extract_params['subtract_background'] = False

        if use_source_posn is None:
            extract_params['use_source_posn'] = False
        else:
            extract_params['use_source_posn'] = use_source_posn

        extract_params['position_correction'] = 0
        extract_params['independent_var'] = 'pixel'
        # Note that extract_params['dispaxis'] is not assigned.  This will be done later, possibly slit by slit.

    elif ref_dict['ref_file_type'] == FILE_TYPE_JSON:
        extract_params['ref_file_type'] = ref_dict['ref_file_type']

        for aper in ref_dict['apertures']:
            if (
                    'id' in aper
                    and aper['id'] != "dummy"
                    and (aper['id'] == slitname or aper['id'] == "ANY" or slitname == "ANY")
            ):
                extract_params['match'] = PARTIAL
                # region_type is retained for backward compatibility; it is
                # not required to be present.
                # spectral_order is a secondary selection criterion.  The
                # default is the expected value, so if the key is not present
                # in the JSON file, the current aperture will be selected.
                # If the current aperture in the JSON file has
                # "spectral_order": "ANY", that aperture will be selected.
                region_type = aper.get("region_type", "target")

                if region_type != "target":
                    continue

                spectral_order = aper.get("spectral_order", sp_order)

                if spectral_order == sp_order or spectral_order == ANY:
                    extract_params['match'] = EXACT
                    extract_params['spectral_order'] = sp_order
                    # Note: extract_params['dispaxis'] is not assigned. This is done later, possibly slit by slit.

                    if meta.target.source_type == "EXTENDED":
                        log.info("Target is extended, so the entire region will be extracted.")
                        shape = input_model.data.shape
                        extract_params['src_coeff'] = None
                        extract_params['bkg_coeff'] = None
                        extract_params['bkg_fit'] = None
                        extract_params['bkg_order'] = 0
                        extract_params['subtract_background'] = False
                        extract_params['use_source_posn'] = False
                        extract_params['xstart'] = 0
                        extract_params['xstop'] = shape[-1] - 1
                        extract_params['ystart'] = 0
                        extract_params['ystop'] = shape[-2] - 1
                        extract_params['extract_width'] = None
                        extract_params['independent_var'] = 'pixel'
                        extract_params['position_correction'] = 0  # default value
                    else:
                        extract_params['src_coeff'] = aper.get('src_coeff')
                        extract_params['bkg_coeff'] = aper.get('bkg_coeff')
                        if extract_params['bkg_coeff'] is not None:
                            extract_params['subtract_background'] = True
                            extract_params['bkg_fit'] = aper.get('bkg_fit', 'poly')
                        else:
                            extract_params['bkg_fit'] = None
                            extract_params['subtract_background'] = False
                        extract_params['independent_var'] = aper.get('independent_var', 'pixel').lower()

                        if bkg_order is None:
                            extract_params['bkg_order'] = aper.get('bkg_order', 0)
                        else:
                            # If the user supplied a value, use that value.
                            extract_params['bkg_order'] = bkg_order

                        if use_source_posn is None:
                            extract_params['use_source_posn'] = aper.get('use_source_posn', False)
                        else:
                            # If the user supplied a value, use that value.
                            extract_params['use_source_posn'] = use_source_posn

                        extract_params['xstart'] = aper.get('xstart')
                        extract_params['xstop'] = aper.get('xstop')
                        extract_params['ystart'] = aper.get('ystart')
                        extract_params['ystop'] = aper.get('ystop')
                        extract_params['extract_width'] = aper.get('extract_width')
                        extract_params['position_correction'] = 0  # default value

                    if smoothing_length is None:
                        extract_params['smoothing_length'] = aper.get('smoothing_length', 0)
                    else:
                        # If the user supplied a value, use that value.
                        extract_params['smoothing_length'] = smoothing_length

                    break

    elif ref_dict['ref_file_type'] == FILE_TYPE_IMAGE:
        # Note that we will use the supplied image-format extract1d reference file,
        # without regard for the distinction between point source and
        # extended source.
        extract_params['ref_file_type'] = ref_dict['ref_file_type']
        foundit = False
        im = None

        for im in ref_dict['ref_model'].images:
            if im.name == slitname or im.name == ANY or slitname == ANY:
                extract_params['match'] = PARTIAL

                if im.spectral_order == sp_order or im.spectral_order >= ANY_ORDER:
                    extract_params['match'] = EXACT
                    extract_params['spectral_order'] = sp_order
                    foundit = True
                    break

        if foundit:
            extract_params['ref_image'] = im
            # Note that extract_params['dispaxis'] is not assigned.  This will be done later, possibly slit by slit.
            if smoothing_length is None:
                extract_params['smoothing_length'] = im.smoothing_length
            else:
                # The user-supplied value takes precedence.
                extract_params['smoothing_length'] = smoothing_length

            if use_source_posn is None:
                extract_params['use_source_posn'] = False
            else:
                extract_params['use_source_posn'] = use_source_posn

            extract_params['position_correction'] = 0

    else:
        log.error(f"Reference file type {ref_dict['ref_file_type']} not recognized")

    return extract_params


def log_initial_parameters(extract_params: dict):
    """Log some of the initial extraction parameters.

    Parameters
    ----------
    extract_params : dict
        Information read from the reference file.
    """
    if "xstart" not in extract_params:
        return

    log.debug("Initial parameters:")
    log.debug(f"dispaxis = {extract_params['dispaxis']}")
    log.debug(f"spectral order = {extract_params['spectral_order']}")
    log.debug(f"initial xstart = {extract_params['xstart']}")
    log.debug(f"initial xstop = {extract_params['xstop']}")
    log.debug(f"initial ystart = {extract_params['ystart']}")
    log.debug(f"initial ystop = {extract_params['ystop']}")
    log.debug(f"extract_width = {extract_params['extract_width']}")
    log.debug(f"initial src_coeff = {extract_params['src_coeff']}")
    log.debug(f"initial bkg_coeff = {extract_params['bkg_coeff']}")
    log.debug(f"bkg_fit = {extract_params['bkg_fit']}")
    log.debug(f"bkg_order = {extract_params['bkg_order']}")
    log.debug(f"smoothing_length = {extract_params['smoothing_length']}")
    log.debug(f"independent_var = {extract_params['independent_var']}")
    log.debug(f"use_source_posn = {extract_params['use_source_posn']}")


def get_aperture(
        im_shape: tuple, wcs: WCS, verbose: bool, extract_params: dict
) -> Union[Aperture, dict]:
    """Get the extraction limits xstart, xstop, ystart, ystop.

    Parameters
    ----------
    im_shape : tuple
        The shape (2-D) of the input data.  This will be for the current
        integration, if the input contains more than one integration.

    wcs : a WCS object, or None
        The wcs (if any) for the input data or slit.

    verbose : bool
        If True, log messages.

    extract_params : dict
        Parameters read from the reference file.

    Returns
    -------
    ap_ref : Aperture NamedTuple or an empty dict
        Keys are 'xstart', 'xstop', 'ystart', and 'ystop'.
    """
    if extract_params['ref_file_type'] == FILE_TYPE_IMAGE:
        return {}

    ap_ref = aperture_from_ref(extract_params, im_shape)
    ap_ref, truncated = update_from_shape(ap_ref, im_shape)

    if truncated and verbose:
        log.warning("Extraction limits extended outside image borders; limits have been truncated.")

    if wcs is not None:
        ap_wcs = aperture_from_wcs(wcs, verbose)
    else:
        ap_wcs = None

    # If the xstart, etc., values were not specified for the dispersion direction, the extraction region should be
    # centered within the WCS bounding box (domain).
    ap_ref = update_from_wcs(ap_ref, ap_wcs, extract_params["extract_width"], extract_params["dispaxis"], verbose)
    ap_ref = update_from_width(ap_ref, extract_params["extract_width"], extract_params["dispaxis"])

    return ap_ref


def aperture_from_ref(extract_params: dict, im_shape: Tuple[int]) -> Aperture:
    """Get extraction region from reference file or image shape.

    Parameters
    ----------
    extract_params : dict
        Parameters read from the reference file.

    im_shape : tuple of int
        The last two elements are the height and width of the input image
        (slit).

    Returns
    -------
    ap_ref : namedtuple
        Keys are 'xstart', 'xstop', 'ystart', and 'ystop'.
    """
    nx = im_shape[-1]
    ny = im_shape[-2]
    _xstart = 0
    _xstop = nx - 1
    _ystart = 0
    _ystop = ny - 1

    xstart = extract_params.get('xstart', 0)
    xstop = extract_params.get('xstop', nx - 1)  # limits are inclusive
    ystart = extract_params.get('ystart', 0)
    ystop = extract_params.get('ystop', ny - 1)

    ap_ref = Aperture(
        xstart=xstart if xstart is not None else _xstart,
        xstop=xstop if xstop is not None else _xstop,
        ystart=ystart if ystart is not None else _ystart,
        ystop=ystop if ystop is not None else _ystop
    )

    return ap_ref


def update_from_width(
        ap_ref: Aperture, extract_width: Union[int, None], direction: int
) -> Aperture:
    """Update XD extraction limits based on extract_width.

    If extract_width was specified, that value should override
    ystop - ystart (or xstop - xstart, depending on dispersion direction).

    Parameters
    ----------
    ap_ref : Aperture NamedTuple
        Contains xstart, xstop, ystart, ystop.  These are the initial
        values as read from the extract1d reference file, except that they may
        have been truncated at the image borders.

    extract_width : int or None
        The number of pixels in the cross-dispersion direction to add
        together to make a 1-D spectrum from a 2-D image.

    direction : int
        HORIZONTAL (1) if the dispersion direction is predominantly
        horizontal.  VERTICAL (2) if the dispersion direction is
        predominantly vertical.

    Returns
    -------
    ap_width : namedtuple
        Keys are 'xstart', 'xstop', 'ystart', and 'ystop'.
    """
    if extract_width is None:
        return ap_ref

    if direction == HORIZONTAL:
        temp_width = ap_ref.ystop - ap_ref.ystart + 1
    else:
        temp_width = ap_ref.xstop - ap_ref.xstart + 1

    if extract_width == temp_width:
        return ap_ref  # OK as is

    # An integral value corresponds to the center of a pixel.
    # If the extraction limits were not specified via polynomial coefficients, assign_polynomial_limits will create
    # polynomial functions using values from an Aperture, and these lower and upper limits will be expanded by 0.5 to
    # give polynomials (constant functions) for the lower and upper edges of the bounding pixels.

    width = float(extract_width)

    if direction == HORIZONTAL:
        lower = float(ap_ref.ystart)
        upper = float(ap_ref.ystop)
        lower = (lower + upper) / 2. - (width - 1.) / 2.
        upper = lower + (width - 1.)
        ap_width = Aperture(xstart=ap_ref.xstart, xstop=ap_ref.xstop, ystart=lower, ystop=upper)
    else:
        lower = float(ap_ref.xstart)
        upper = float(ap_ref.xstop)
        lower = (lower + upper) / 2. - (width - 1.) / 2.
        upper = lower + (width - 1.)
        ap_width = Aperture(xstart=lower, xstop=upper, ystart=ap_ref.ystart, ystop=ap_ref.ystop)

    return ap_width


def update_from_shape(
        ap: Aperture, im_shape: Tuple[int]
) -> Tuple[Aperture, bool]:
    """Truncate extraction region based on input image shape.

    Parameters
    ----------
    ap : namedtuple
        Extraction region.

    im_shape : tuple of int
        The last two elements are the height and width of the input image.

    Returns
    -------
    ap_shape : namedtuple
        Keys are 'xstart', 'xstop', 'ystart', and 'ystop'.

    truncated : bool
         True if any value was truncated at an image edge.
    """
    nx = im_shape[-1]
    ny = im_shape[-2]

    xstart = ap.xstart
    xstop = ap.xstop
    ystart = ap.ystart
    ystop = ap.ystop

    truncated = False

    if ap.xstart < 0:
        xstart = 0
        truncated = True

    if ap.xstop >= nx:
        xstop = nx - 1  # limits are inclusive
        truncated = True

    if ap.ystart < 0:
        ystart = 0
        truncated = True

    if ap.ystop >= ny:
        ystop = ny - 1
        truncated = True

    ap_shape = Aperture(xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)

    return ap_shape, truncated


def aperture_from_wcs(wcs: WCS, verbose: bool) -> Union[NamedTuple, None]:
    """Get the limits over which the WCS is defined.

    Parameters
    ----------
    wcs : WCS
        The world coordinate system interface.

    verbose : bool
        If True, log messages.

    Returns
    -------
    ap_wcs : namedtuple or None
        Keys are 'xstart', 'xstop', 'ystart', and 'ystop'.  These are the
        limits copied directly from wcs.bounding_box.
    """
    got_bounding_box = False

    try:
        bounding_box = wcs.bounding_box
        got_bounding_box = True
    except AttributeError:
        if verbose:
            log.info("wcs.bounding_box not found; using wcs.domain instead.")

        bounding_box = (
            (wcs.domain[0]['lower'], wcs.domain[0]['upper']),
            (wcs.domain[1]['lower'], wcs.domain[1]['upper'])
        )

    if got_bounding_box and bounding_box is None:
        if verbose:
            log.warning("wcs.bounding_box is None")
        return None

    # bounding_box should be a tuple of tuples, each of the latter consisting of (lower, upper) limits.
    if len(bounding_box) < 2:
        if verbose:
            log.warning("wcs.bounding_box has the wrong shape")
        return None

    # These limits are float, and they are inclusive.
    xstart = bounding_box[0][0]
    xstop = bounding_box[0][1]
    ystart = bounding_box[1][0]
    ystop = bounding_box[1][1]

    ap_wcs = Aperture(xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)

    return ap_wcs


def update_from_wcs(
        ap_ref: Aperture,
        ap_wcs: Union[Aperture, None],
        extract_width: int,
        direction: int,
        verbose: bool
) -> Aperture:
    """Limit the extraction region to the WCS bounding box.

    Parameters
    ----------
    ap_ref : namedtuple
        Contains xstart, xstop, ystart, ystop.  These are the values of
        the extraction region as specified by the reference file or the
        image size.

    ap_wcs : namedtuple or None
        These are the bounding box limits.

    extract_width : int
        The number of pixels in the cross-dispersion direction to add
        together to make a 1-D spectrum from a 2-D image.

    direction : int
        HORIZONTAL (1) if the dispersion direction is predominantly
        horizontal.  VERTICAL (2) if the dispersion direction is
        predominantly vertical.

    verbose : bool
        If True, log messages.

    Returns
    -------
    ap : namedtuple
        Keys are 'xstart', 'xstop', 'ystart', and 'ystop'.
    """
    if ap_wcs is None:
        return ap_ref

    # If the wcs limits don't pass the sanity test, ignore the bounding box.
    if not sanity_check_limits(ap_ref, ap_wcs, verbose):
        return ap_ref

    # ap_wcs has the limits over which the WCS transformation is defined; take those as the outer limits over which we
    # will extract.
    xstart = compare_start(ap_ref.xstart, ap_wcs.xstart)
    ystart = compare_start(ap_ref.ystart, ap_wcs.ystart)
    xstop = compare_stop(ap_ref.xstop, ap_wcs.xstop)
    ystop = compare_stop(ap_ref.ystop, ap_wcs.ystop)

    if extract_width is not None:
        if direction == HORIZONTAL:
            width = ystop - ystart + 1
        else:
            width = xstop - xstart + 1

        if width < extract_width:
            if verbose:
                log.warning(f"extract_width was truncated from {extract_width} to {width}")

    ap = Aperture(xstart=xstart, xstop=xstop, ystart=ystart, ystop=ystop)

    return ap


def sanity_check_limits(
        ap_ref: Aperture, ap_wcs: Aperture, verbose: bool
) -> bool:
    """Sanity check.

    Parameters
    ----------
    ap_ref : namedtuple
        Contains xstart, xstop, ystart, ystop.  These are the values of
        the extraction region as specified by the reference file or the
        image size.

    ap_wcs : namedtuple
        These are the bounding box limits.

    verbose : bool
        If True, log messages.

    Returns
    -------
    flag : boolean
        True if ap_ref and ap_wcs do overlap, i.e. if the sanity test passes.
    """
    if (
            ap_wcs.xstart >= ap_ref.xstop
            or ap_wcs.xstop <= ap_ref.xstart
            or ap_wcs.ystart >= ap_ref.ystop
            or ap_wcs.ystop <= ap_ref.ystart
    ):
        if verbose:
            log.warning(
                f"The WCS bounding box is outside the aperture:\n\t"
                f"aperture: {ap_ref.xstart}, {ap_ref.xstop}, {ap_ref.ystart}, {ap_ref.ystop}\n\t"
                f"wcs: {ap_wcs.xstart}, {ap_wcs.xstop}, {ap_wcs.ystart}, {ap_wcs.ystop}\n"
                f"so the wcs bounding box will be ignored."
            )

        flag = False
    else:
        flag = True

    return flag


def compare_start(
        aperture_start: Union[int, float], wcs_bb_lower_lim: Union[int, float]
) -> Union[int, float]:
    """Compare the start limit from the aperture with the WCS lower limit.

    Extended summary
    ----------------
    The more restrictive (i.e. larger) limit is the one upon which the
    output value will be based.  If the WCS limit is larger, the value will
    be increased to an integer, on the assumption that WCS lower limits
    correspond to the lower edge of the bounding pixel.  If this value
    will actually be used for an extraction limit (i.e. if the limits were
    not already specified by polynomial coefficients), then
    `assign_polynomial_limits` will create a polynomial function using this
    value, except that it will be decreased by 0.5 to correspond to the
    lower edge of the bounding pixels.

    Parameters
    ----------
    aperture_start : int or float
        xstart or ystart, as specified by the extract1d reference file or the image
        size.

    wcs_bb_lower_lim : int or float
        The lower limit from the WCS bounding box.

    Returns
    -------
    value : int or float
        The start limit, possibly constrained by the WCS start limit.
    """
    if aperture_start >= wcs_bb_lower_lim:  # ref is inside WCS limit
        return aperture_start
    else:  # outside (below) WCS limit
        return math.ceil(wcs_bb_lower_lim)


def compare_stop(
        aperture_stop: Union[int, float], wcs_bb_upper_lim: Union[int, float]
) -> Union[int, float]:
    """Compare the stop limit from the aperture with the WCS upper limit.

    The more restrictive (i.e. smaller) limit is the one upon which the
    output value will be based.  If the WCS limit is smaller, the value
    will be truncated to an integer, on the assumption that WCS upper
    limits correspond to the upper edge of the bounding pixel.  If this
    value will actually be used for an extraction limit (i.e. if the
    limits were not already specified by polynomial coefficients), then
    `assign_polynomial_limits` will create a polynomial function using this
    value, except that it will be increased by 0.5 to correspond to the
    upper edge of the bounding pixels.

    Parameters
    ----------
    aperture_stop : int or float
        xstop or ystop, as specified by the extract1d reference file or the image
        size.

    wcs_bb_upper_lim : int or float
        The upper limit from the WCS bounding box.

    Returns
    -------
    value : int or float
        The stop limit, possibly constrained by the WCS stop limit.
    """
    if aperture_stop <= wcs_bb_upper_lim:  # ref is inside WCS limit
        return aperture_stop
    else:  # outside (above) WCS limit
        return math.floor(wcs_bb_upper_lim)


def create_poly(coeff: List[float]) -> Union[polynomial.Polynomial1D, None]:
    """Create a polynomial model from coefficients.

    Parameters
    ----------
    coeff : list of float
        The coefficients of the polynomial, constant term first, highest
        order term last.

    Returns
    -------
    `astropy.modeling.polynomial.Polynomial1D` object, or None if `coeff`
        is empty.
    """
    n = len(coeff)

    if n < 1:
        return None

    coeff_dict = {f'c{i}': coeff[i] for i in range(n)}

    return polynomial.Polynomial1D(degree=n - 1, **coeff_dict)


class ExtractBase(abc.ABC):
    """Base class for 1-D extraction info and methods.

    Attributes
    ----------
    exp_type : str
        Exposure type.

    ref_image : data model, or None
        The reference image model.

    spectral_order : int
        Spectral order number.

    dispaxis : int
        Dispersion direction:  1 is horizontal, 2 is vertical.

    xstart : int or None
        First pixel (zero indexed) in extraction region.

    xstop : int or None
        Last pixel (zero indexed) in extraction region.

    ystart : int or None
        First pixel (zero indexed) in extraction region.

    ystop : int or None
        Last pixel (zero indexed) in extraction region.

    extract_width : int or None
        Height (in the cross-dispersion direction) of the extraction
        region.

    independent_var : str
        The polynomial functions for computing the upper and lower
        boundaries of extraction and background regions can be functions
        of pixel number or wavelength (in microns).  These options are
        distinguished by `independent_var`.

    position_correction : float
        If not zero, this will be added to the extraction region limits
        for the cross-dispersion direction, both target and background.

    src_coeff : list of lists of float, or None
        These are coefficients of polynomial functions that define the
        cross-dispersion limits of one or more source extraction regions
        (yes, there can be more than one extraction region).  Note that
        these are float values, and the limits can include fractions of
        pixels.
        If specified, this takes priority over `ystart`, `ystop`, and
        `extract_width`, though `xstart` and `xstop` will still be used
        for the limits in the dispersion direction.  (Interchange "x" and
        "y" if the dispersion is in the vertical direction.)

        For example, the value could be:

-           [[1, 2], [3, 4, 5], [6], [7, 8]]

        which means:

-           [1, 2] coefficients for the lower limit of the first region
-           [3, 4, 5] coefficients for the upper limit of the first region
-           [6] coefficient(s) for the lower limit of the second region
-           [7, 8] coefficients for the upper limit of the second region

        The coefficients are listed with the constant term first, highest
        order term last.  For example, [3, 4, 5] means 3 + 4 * x + 5 * x^2.

    bkg_coeff : list of lists of float, or None
        This has the same format as `src_coeff`, but the polynomials
        define one or more background regions.

    p_src : list of astropy.modeling.polynomial.Polynomial1D
        These are Astropy polynomial functions defining the limits in the
        cross-dispersion direction of the source extraction region(s).
        If `src_coeff` was specified, `p_src` will be created directly
        from `src_coeff`; otherwise, a constant function based on
        `ystart`, `ystop`, `extract_width` will be assigned to `p_src`.

    p_bkg : list of astropy.modeling.polynomial.Polynomial1D, or None
        These are Astropy polynomial functions defining the limits in the
        cross-dispersion direction of the background extraction regions.
        This list will be populated from `bkg_coeff`, if that was specified.

    wcs : WCS object
        For computing the right ascension, declination, and wavelength at
        one or more pixels.

    smoothing_length : int
        Width of a boxcar function for smoothing the background regions.
        This argument must be an odd positive number or zero, and it is
        only used if background regions have been specified.

    bkg_fit : str
        Type of background fitting to perform in each column (or row, if
        the dispersion is vertical). Allowed values are `poly` (default),
        `mean`, and `median`.

    bkg_order : int
        Polynomial order for fitting to each column (or row, if the
        dispersion is vertical) of background. Only used if `bkg_fit` is
        `poly`.
        This argument must be positive or zero, and it is only used if
        background regions have been specified.


    subtract_background : bool or None
        A flag that indicates whether the background should be subtracted.
        If None, the value in the extract_1d reference file will be used.
        If not None, this parameter overrides the value in the
        extract_1d reference file.

    use_source_posn : bool or None
        If True, the target and background positions specified in the
        reference file (or the default position, if there is no
        reference file) will be shifted to account for the actual
        source position in the data.
    """

    def __init__(
            self,
            input_model: DataModel,
            slit: Union[DataModel, None] = None,
            ref_image: Union[DataModel, None] = None,
            dispaxis: int = HORIZONTAL,
            spectral_order: int = 1,
            xstart: int = None,
            xstop: int = None,
            ystart: int = None,
            ystop: int = None,
            extract_width: int = None,
            src_coeff: Union[List[List[float]], None] = None,
            bkg_coeff: Union[List[List[float]], None] = None,
            independent_var: str = "pixel",
            smoothing_length: int = 0,
            bkg_fit: str = "poly",
            bkg_order: int = 0,
            position_correction: float = 0.,
            subtract_background: Union[bool, None] = None,
            use_source_posn: Union[bool, None] = None,
            match: Union[str, None] = None,
            ref_file_type: Union[str, None] = None
    ):
        """
        Parameters
        ----------
        input_model : data model
            The input science data.

        slit : an input slit, or None if not used
            For MultiSlit, `slit` is one slit from
            a list of slits in the input.  For other types of data, `slit`
            will not be used.

        ref_image : data model, or None
            The reference image.

        dispaxis : int
            Dispersion direction:  1 is horizontal, 2 is vertical.

        spectral_order : int
            Spectral order number.

        xstart : int
            First pixel (zero indexed) in extraction region.

        xstop : int
            Last pixel (zero indexed) in extraction region.

        ystart : int
            First pixel (zero indexed) in extraction region.

        ystop : int
            Last pixel (zero indexed) in extraction region.

        extract_width : int
            Height (in the cross-dispersion direction) of the extraction
            region.

        src_coeff : list of lists of float, or None
            These are coefficients of polynomial functions that define
            the cross-dispersion limits of one or more source extraction
            regions.

        bkg_coeff : list of lists of float, or None
            This has the same format as `src_coeff`, but the polynomials
            define one or more background regions.

        independent_var : str
            This can be either "pixel" or "wavelength" to specify the
            independent variable for polynomial functions.

        smoothing_length : int
            Width of a boxcar function for smoothing the background
            regions.

        bkg_fit : str
            Type of fitting to apply to background values in each column
            (or row, if the dispersion is vertical).

        bkg_order : int
            Polynomial order for fitting to each column (or row, if the
            dispersion is vertical) of background.

        position_correction : float
            If not zero, this will be added to the extraction region limits
            for the cross-dispersion direction, both target and background.

        subtract_background : bool or None
            A flag which indicates whether the background should be subtracted.
            If None, the value in the extract_1d reference file will be used.
            If not None, this parameter overrides the value in the
            extract_1d reference file.

        use_source_posn : bool or None
            If True, the target and background positions specified in the
            reference file (or the default position, if there is no
            reference file) will be shifted to account for the actual
            source position in the data.
        """
        self.exp_type = input_model.meta.exposure.type

        self.dispaxis = dispaxis
        self.spectral_order = spectral_order
        self.ref_image = ref_image
        self.xstart = xstart
        self.xstop = xstop
        self.ystart = ystart
        self.ystop = ystop
        self.match = match
        self.ref_file_type = ref_file_type

        # xstart, xstop, ystart, or ystop may be overridden with src_coeff, they may be limited by the input image size
        # or by the WCS bounding box, or they may be modified if extract_width was specified (because extract_width
        # takes precedence).
        # If these values are specified, the limits in the cross-dispersion direction should be integers, but they may
        # later be replaced with fractional values, depending on extract_width, in order to center the extraction window
        # in the originally specified xstart to xstop (or ystart to ystop).
        if self.dispaxis == VERTICAL:
            if not isinstance(self.xstart, int) and self.xstart is not None:
                self.xstart = round(self.xstart)
                log.warning(f"xstart {xstart} should have been an integer; rounding to {self.xstart}")

            if not isinstance(self.xstop, int) and self.xstop is not None:
                self.xstop = round(self.xstop)
                log.warning(f"xstop {xstop} should have been an integer; rounding to {self.xstop}")

        if self.dispaxis == HORIZONTAL:
            if not isinstance(self.ystart, int) and self.ystart is not None:
                self.ystart = round(self.ystart)
                log.warning(f"ystart {ystart} should have been an integer; rounding to {self.ystart}")

            if not isinstance(self.ystop, int) and self.ystop is not None:
                self.ystop = round(self.ystop)
                log.warning(f"ystop {ystop} should have been an integer; rounding to {self.ystop}")

        if extract_width is None:
            self.extract_width = None
        else:
            self.extract_width = round(extract_width)

        self.independent_var = independent_var.lower()

        # Coefficients for source (i.e. target) and background limits and corresponding polynomial functions.
        self.src_coeff = copy.deepcopy(src_coeff)
        self.bkg_coeff = copy.deepcopy(bkg_coeff)

        # These functions will be assigned by assign_polynomial_limits.
        # The "p" in the attribute name indicates a polynomial function.
        self.p_src = None
        self.p_bkg = None

        self.smoothing_length = smoothing_length if smoothing_length is not None else 0

        if 0 < self.smoothing_length == self.smoothing_length // 2 * 2:
            log.warning(f"smoothing_length was even ({self.smoothing_length}), so incremented by 1")
            self.smoothing_length += 1  # must be odd

        self.bkg_fit = bkg_fit
        self.bkg_order = bkg_order
        self.use_source_posn = use_source_posn
        self.position_correction = position_correction
        self.subtract_background = subtract_background

        self.wcs = None  # initial value

        if input_model.meta.exposure.type == "NIS_SOSS":
            if hasattr(input_model.meta, 'wcs'):
                try:
                    self.wcs = niriss.niriss_soss_set_input(input_model, self.spectral_order)
                except ValueError:
                    raise InvalidSpectralOrderNumberError(f"Spectral order {self.spectral_order} is not valid")
        elif slit is None:
            if hasattr(input_model.meta, 'wcs'):
                self.wcs = input_model.meta.wcs
        elif hasattr(slit, 'meta') and hasattr(slit.meta, 'wcs'):
            self.wcs = slit.meta.wcs

        if self.wcs is None:
            log.warning("WCS function not found in input.")

    def update_extraction_limits(self, ap):
        pass

    def assign_polynomial_limits(self, verbose):
        pass

    @staticmethod
    def get_target_coordinates(
            input_model: DataModel, slit: Union[DataModel, None]
    ) -> Tuple[Union[float, None], Union[float, None]]:
        """Get the right ascension and declination of the target.

        For MultiSlitModel (or similar) data, each slit has the source
        right ascension and declination as attributes, and this can vary
        from one slit to another (e.g. for NIRSpec MOS, or for WFSS).  In
        this case, we want the celestial coordinates from the slit object.
        For other models, however, the celestial coordinates of the source
        are in input_model.meta.target.

        Parameters
        ----------
        input_model : data model
            The input science data model.

        slit : SlitModel or None
            One slit from a MultiSlitModel (or similar), or None if
            there are no slits.

        Returns
        -------
        targ_ra : float or None
            The right ascension of the target, or None

        targ_dec : float or None
            The declination of the target, or None
        """
        targ_ra = None
        targ_dec = None

        if slit is not None:
            # If we've been passed a slit object, get the RA/Dec
            # from the slit source attributes
            targ_ra = getattr(slit, 'source_ra', None)
            targ_dec = getattr(slit, 'source_dec', None)
        elif isinstance(input_model, datamodels.SlitModel):
            # If the input model is a single SlitModel, again
            # get the coords from the slit source attributes
            targ_ra = getattr(input_model, 'source_ra', None)
            targ_dec = getattr(input_model, 'source_dec', None)

        if targ_ra is None or targ_dec is None:
            # Otherwise get it from the generic target coords
            targ_ra = input_model.meta.target.ra
            targ_dec = input_model.meta.target.dec

        # Issue a warning if none of the methods succeeded
        if targ_ra is None or targ_dec is None:
            log.warning("Target RA and Dec could not be determined")
            targ_ra = targ_dec = None

        return targ_ra, targ_dec

    def offset_from_offset(
            self, input_model: DataModel, slit: DataModel, verbose: bool
    ) -> Tuple[float, Union[float, None]]:
        """Get position offset from the target coordinates.

        Parameters
        ----------
        input_model : data model
            The input science data.

        slit : SlitModel or None
            One slit from a MultiSlitModel (or similar), or None if
            there are no slits.

        verbose : bool
            If True, log messages.

        Returns
        -------
        offset : float
            The offset of the exposure from the nominal position, due to
            source positioning.  This is the component of the offset
            perpendicular to the dispersion direction.  A positive value
            means that the spectrum is at a larger pixel number than the
            nominal location.

        locn : float or None
            The pixel coordinate of the target in the cross-dispersion
            direction, at the middle of the spectrum in the dispersion
            direction.
        """
        targ_ra, targ_dec = self.get_target_coordinates(input_model, slit)

        if targ_ra is None or targ_dec is None:
            return 0., None

        # Use the WCS function to find the cross-dispersion (XD) location that is closest to the target coordinates.
        # This is the "actual" location of the spectrum, so the extraction region should be centered here.
        locn_info = self.locn_from_wcs(input_model, slit, targ_ra, targ_dec, verbose)

        if locn_info is None:
            middle = middle_wl = locn = locn_info
        else:
            middle, middle_wl, locn = locn_info

        if middle is not None and verbose:
            log.debug(f"Spectrum location from WCS used column/row {middle}")

        # Find the nominal extraction location, i.e. the XD location specified in the reference file prior to adding any
        # position offset.
        # The difference is the position offset.
        offset = 0.

        if middle is not None and locn is not None:
            nominal_location = self.nominal_locn(middle, middle_wl)

            if verbose:
                log.debug(f"Target spectrum is at {locn:.2f} in the cross-dispersion direction")

            if nominal_location is not None:
                if verbose:
                    log.debug(f"and the nominal XD location of the spectrum is {nominal_location:.2f}")

                offset = locn - nominal_location
            else:
                if verbose:
                    log.debug("but couldn't determine the nominal XD location.")

        if np.isnan(offset):
            if verbose:
                log.warning("Source position offset is NaN; setting it to 0")
            offset = 0.

        self.position_correction = offset

        return offset, locn

    def locn_from_wcs(
            self,
            input_model: DataModel,
            slit: Union[DataModel, None],
            targ_ra: Union[float, None],
            targ_dec: Union[float, None],
            verbose: bool
    ) -> Union[Tuple[int, float, float], None]:
        """Get the location of the spectrum, based on the WCS.

        Parameters
        ----------
        input_model : data model
            The input science model.

        slit : one slit from a MultiSlitModel (or similar), or None
            The WCS and target coordinates will be gotten from `slit`
            unless `slit` is None, and in that case they will be gotten
            from `input_model`.

        targ_ra : float or None
            The right ascension of the target, or None

        targ_dec : float or None
            The declination of the target, or None

        verbose : bool
            If True, log messages.

        Returns
        -------
        tuple (middle, middle_wl, locn) or None
        middle : int
            Pixel coordinate in the dispersion direction within the 2-D
            cutout (or the entire input image) at the middle of the WCS
            bounding box.  This is the point at which to determine the
            nominal extraction location, in case it varies along the
            spectrum.  The offset will then be the difference between
            `locn` (below) and the nominal location.

        middle_wl : float
            The wavelength at pixel `middle`.

        locn : float
            Pixel coordinate in the cross-dispersion direction within the
            2-D cutout (or the entire input image) that has right ascension
            and declination coordinates corresponding to the target location.
            The spectral extraction region should be centered here.

        None will be returned if there was not sufficient information
        available, e.g. if the wavelength attribute or wcs function is not
        defined.
        """
        # WFSS data are not currently supported by this function
        if input_model.meta.exposure.type in WFSS_EXPTYPES:
            log.warning("Can't use target coordinates to get location of spectrum "
                        f"for exp type {input_model.meta.exposure.type}")
            return

        bb = self.wcs.bounding_box  # ((x0, x1), (y0, y1))

        if bb is None:
            if slit is None:
                shape = input_model.data.shape
            else:
                shape = slit.data.shape

            bb = wcs_bbox_from_shape(shape)

        if self.dispaxis == HORIZONTAL:
            # Width (height) in the cross-dispersion direction, from the start of the 2-D cutout (or of the full image)
            # to the upper limit of the bounding box.
            # This may be smaller than the full width of the image, but it's all we need to consider.
            xd_width = int(round(bb[1][1]))  # must be an int
            middle = int((bb[0][0] + bb[0][1]) / 2.)  # Middle of the bounding_box in the dispersion direction.
            x = np.empty(xd_width, dtype=np.float64)
            x[:] = float(middle)
            y = np.arange(xd_width, dtype=np.float64)
            lower = bb[1][0]
            upper = bb[1][1]
        else:  # dispaxis = VERTICAL
            xd_width = int(round(bb[0][1]))  # Cross-dispersion total width of bounding box; must be an int
            middle = int((bb[1][0] + bb[1][1]) / 2.)  # Mid-point of width along dispersion direction
            x = np.arange(xd_width, dtype=np.float64)  # 1-D vector of cross-dispersion (x) pixel indices
            y = np.empty(xd_width, dtype=np.float64)  # 1-D vector all set to middle y index
            y[:] = float(middle)

            # lower and upper range in cross-dispersion direction
            lower = bb[0][0]
            upper = bb[0][1]

        # We need stuff[2], a 1-D array of wavelengths crossing the spectrum near its middle.
        fwd_transform = self.wcs(x, y)
        middle_wl = np.nanmean(fwd_transform[2])

        try:
            x_y = self.wcs.backward_transform(targ_ra, targ_dec, middle_wl)
        except NotImplementedError:
            log.warning("Inverse wcs is not implemented, so can't use target coordinates to get location of spectrum.")
            return

        # locn is the XD location of the spectrum:
        if self.dispaxis == HORIZONTAL:
            locn = x_y[1]
        else:
            locn = x_y[0]

        if locn < lower or locn > upper and targ_ra > 340.:
            # Try this as a temporary workaround.
            x_y = self.wcs.backward_transform(targ_ra - 360., targ_dec, middle_wl)

            if self.dispaxis == HORIZONTAL:
                temp_locn = x_y[1]
            else:
                temp_locn = x_y[0]

            if lower <= temp_locn <= upper:
                # Subtracting 360 from the right ascension worked!
                locn = temp_locn

                if verbose:
                    log.warning(f"targ_ra changed from {targ_ra} to {targ_ra - 360.}")

        # If the target is at the edge of the image or at the edge of the non-NaN area, we can't use the WCS to find the
        # location of the target spectrum.
        if locn < lower or locn > upper:
            if verbose:
                log.warning(f"WCS implies the target is at {locn:.2f}, which is outside the bounding box,")
                log.warning("so we can't get spectrum location using the WCS")
            locn = None

        return middle, middle_wl, locn

    @abc.abstractmethod
    def nominal_locn(self, middle, middle_wl):
        # Implemented in the subclasses.
        pass


class ExtractModel(ExtractBase):
    """The extraction region was specified in a JSON file."""

    def __init__(self, verbose, *base_args, **base_kwargs):
        """Create a polynomial model from coefficients.

        Extended summary
        ----------------
        If InvalidSpectralOrderNumberError is raised, processing of the
        current slit or spectral order should be skipped.

        Parameters
        ----------
        verbose : bool
            If True, log messages.

        *base_args, **base_kwargs :
            see ExtractBase parameters for more information.

        """
        self.verbosity = verbose

        super().__init__(*base_args, **base_kwargs)

        # The independent variable for functions for the lower and upper limits of target and background regions can be
        # either 'pixel' or 'wavelength'.
        if (
                self.independent_var != "wavelength"
                and self.independent_var != "pixel"
                and self.independent_var != "pixels"
        ):
            log.error(f"independent_var = {self.independent_var}'; specify 'wavelength' or 'pixel'")
            raise RuntimeError("Invalid value for independent_var")

        # Do sanity checks between requested background subtraction and the
        # existence of background region specifications
        if self.subtract_background is not None:
            if self.subtract_background:
                # If background subtraction was requested, but no background region(s)
                # were specified, turn it off
                if self.bkg_coeff is None:
                    self.subtract_background = False
                    if self.verbosity:
                        log.info("Skipping background subtraction because background regions are not defined.")
            else:
                # If background subtraction was NOT requested, even though background region(s)
                # were specified, blank out the bkg region info
                if self.bkg_coeff is not None:
                    self.bkg_coeff = None
                    if self.verbosity:
                        log.info("Background subtraction was specified in the reference file,")
                        log.info("but has been overridden by the step parameter.")

    def nominal_locn(self, middle: int, middle_wl: float) -> Union[float, None]:
        """Find the nominal cross-dispersion location of the target spectrum.

        This version is for the case that the reference file is a JSON file,
        or that there is no reference file.

        Parameters
        ----------
        middle: int
            The zero-indexed pixel number of the point in the dispersion
            direction at which `locn_from_wcs` determined the actual
            location (in the cross-dispersion direction) of the target
            spectrum.  This is used for evaluating the polynomial
            functions if the independent variable is pixel.

        middle_wl: float
            The wavelength at pixel `middle`.  This is only used if the
            independent variable for polynomial functions is wavelength.

        Returns
        -------
        location: float or None
            The nominal cross-dispersion location (i.e. unmodified by
            position offset) of the target spectrum.

        """
        if self.src_coeff is None:
            if self.dispaxis == HORIZONTAL:
                location = float(self.ystart + self.ystop) / 2.
            else:
                location = float(self.xstart + self.xstop) / 2.
        else:
            if self.independent_var.startswith("wavelength"):
                x = float(middle_wl)
            else:
                x = float(middle)

            # Create the polynomial functions.
            # We'll do this again later, after adding the position offset to the coefficients, but we need to evaluate
            # them at x now in order to get the nominal location of the spectrum.
            self.assign_polynomial_limits(verbose=False)

            n_srclim = len(self.p_src)
            sum_data = 0.
            sum_weights = 0.

            for i in range(n_srclim):
                lower = self.p_src[i][0](x)
                upper = self.p_src[i][1](x)
                weight = (upper - lower)
                sum_data += weight * (lower + upper) / 2.
                sum_weights += weight

            if sum_weights == 0.:
                location = None
            else:
                location = sum_data / sum_weights

        return location

    def add_position_correction(self, verbose: bool, shape: tuple):
        """Add the position offset to the extraction location (in-place).

        Extended summary
        ----------------
        If source extraction coefficients src_coeff were specified, this
        method will add the source position correction to the first coefficient
        of every coefficient list; otherwise, the source offset will be added
        to xstart & xstop or to ystart & ystop.
        If background extraction coefficients bkg_coeff were specified,
        this method will add the source offset to the first coefficients.
        Note that background coefficients are handled independently of
        src_coeff.

        Parameters
        ----------
        verbose : bool
            If True, messages can be logged.

        shape : tuple
            The shape of the data array (may be just the last two axes).
            This is used for truncating a shifted limit at the image edge.
        """

        if self.position_correction == 0.:
            return

        if self.dispaxis == HORIZONTAL:
            direction = "y"
            self.ystart += self.position_correction
            self.ystop += self.position_correction
            # These values must not be negative.
            self.ystart = max(self.ystart, 0)
            self.ystop = max(self.ystop, 0)
            self.ystart = min(self.ystart, shape[-2] - 1)
            self.ystop = min(self.ystop, shape[-2] - 1)  # inclusive limit
        else:
            direction = "x"
            self.xstart += self.position_correction
            self.xstop += self.position_correction
            # These values must not be negative.
            self.xstart = max(self.xstart, 0)
            self.xstop = max(self.xstop, 0)
            self.xstart = min(self.xstart, shape[-1] - 1)
            self.xstop = min(self.xstop, shape[-1] - 1)  # inclusive limit

        if self.src_coeff is None and verbose:
            log.info(f"Applying position offset of {self.position_correction:.2f} to {direction}start and {direction}stop")

        if self.src_coeff is not None or self.bkg_coeff is not None:
            if verbose:
                log.info(f"Applying position offset of {self.position_correction:.2f} to polynomial coefficients")

        if self.src_coeff is not None:
            self._apply_position_corr(self.src_coeff)

        if self.bkg_coeff is not None:
            self._apply_position_corr(self.bkg_coeff)

    def _apply_position_corr(self, coeffs):
        for i in range(len(coeffs)):
            coeff_list = coeffs[i]
            coeff_list[0] += self.position_correction
            coeffs[i] = copy.copy(coeff_list)

    def update_extraction_limits(self, ap: Aperture):
        """Update start and stop limits.

        Extended summary
        ----------------
        Copy the values of xstart, etc., to the attributes.  Note, however,
        that if src_coeff was specified, that will override the values
        given by xstart, etc.
        The limits in the dispersion direction will be rounded to integer.

        Parameters
        ----------
        ap : namedtuple

        """
        self.xstart = ap.xstart
        self.xstop = ap.xstop
        self.ystart = ap.ystart
        self.ystop = ap.ystop

        if self.dispaxis == HORIZONTAL:
            self.xstart = int(round(self.xstart))
            self.xstop = int(round(self.xstop))
        else:  # vertical
            self.ystart = int(round(self.ystart))
            self.ystop = int(round(self.ystop))

    def log_extraction_parameters(self):
        """Log the updated extraction parameters."""
        log.debug("Updated parameters:")
        log.debug(f"position_correction = {self.position_correction}")

        if self.src_coeff is not None:
            log.debug(f"src_coeff = {self.src_coeff}")

            # Since src_coeff was specified, that will be used instead of xstart & xstop (or ystart & ystop).
            if self.dispaxis == HORIZONTAL:
                # Only print xstart/xstop, because ystart/ystop are not used
                log.debug(f"xstart = {self.xstart}")
                log.debug(f"xstop = {self.xstop}")
            else:
                # Only print ystart/ystop, because xstart/xstop are not used
                log.debug(f"ystart = {self.ystart}")
                log.debug(f"ystop = {self.ystop}")

        if self.bkg_coeff is not None:
            log.debug(f"bkg_coeff = {self.bkg_coeff}")

    def assign_polynomial_limits(self, verbose: bool):
        """Create polynomial functions for extraction limits.

        Extended summary
        ----------------
        self.src_coeff and self.bkg_coeff contain lists of polynomial
        coefficients.  These will be used to create corresponding lists of
        polynomial functions, self.p_src and self.p_bkg.  Note, however,
        that the structures of those two lists are not the same.
        The coefficients lists have this form:

-           [[1, 2], [3, 4, 5], [6], [7, 8]]

        which means:

-           [1, 2] coefficients for the lower limit of the first region
-           [3, 4, 5] coefficients for the upper limit of the first region
-           [6] coefficient(s) for the lower limit of the second region
-           [7, 8] coefficients for the upper limit of the second region

        The lists of coefficients must always be in pairs, for the lower
        and upper limits respectively, but they're not explicitly in an
        additional layer of two-element lists,

-           i.e., not like this:  [[[1, 2], [3, 4, 5]], [[6], [7, 8]]]

        That seemed unnecessarily messy and harder for the user to specify.
        For the lists of polynomial functions, on the other hand, that
        additional layer of list is used:

-           [[fcn_lower1, fcn_upper1], [fcn_lower2, fcn_upper2]]

        where:

-           fcn_lower1 is 1 + 2 * x
-           fcn_upper1 is 3 + 4 * x + 5 * x**2
-           fcn_lower2 is 6
-           fcn_upper2 is 7 + 8 * x

        Parameters
        ----------
        verbose : bool
            If True, messages can be logged.

        """
        if self.src_coeff is None:
            # Create constant functions.

            if self.dispaxis == HORIZONTAL:
                lower = float(self.ystart) - 0.5
                upper = float(self.ystop) + 0.5
            else:
                lower = float(self.xstart) - 0.5
                upper = float(self.xstop) + 0.5

            if verbose:
                log.debug(f"Converting extraction limits to [[{lower}], [{upper}]]")

            self.p_src = [[create_poly([lower]), create_poly([upper])]]
        else:
            # The source extraction can include more than one region.
            n_src_coeff = len(self.src_coeff)
            if n_src_coeff // 2 * 2 != n_src_coeff:
                raise RuntimeError("src_coeff must contain alternating lists of lower and upper limits.")

            self.p_src = self._poly_per_region(self.src_coeff)

        if self.bkg_coeff is not None:
            n_bkg_coeff = len(self.bkg_coeff)
            if n_bkg_coeff // 2 * 2 != n_bkg_coeff:
                raise RuntimeError("bkg_coeff must contain alternating lists of lower and upper limits.")

            self.p_bkg = self._poly_per_region(self.bkg_coeff)

    def _poly_per_region(self, coeffs):
        result = []
        expect_lower = True  # toggled in loop

        for coeff_list in coeffs:
            if expect_lower:
                lower = create_poly(coeff_list)
            else:
                upper = create_poly(coeff_list)
                result.append([lower, upper])

            expect_lower = not expect_lower

        return result

    def extract(
            self,
            data: np.ndarray,
            var_poisson: np.ndarray,
            var_rnoise: np.ndarray,
            var_flat: np.ndarray,
            wl_array: Union[np.ndarray, None],
            verbose: bool
    ) -> Tuple[
        float, float, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray
    ]:
        """Do the extraction.

        Extended summary
        ----------------
        This version is for the case that the reference file is a JSON
        file, or that there is no reference file.

        Parameters
        ----------
        data : ndarray, 2-D
            Data array from which the spectrum will be extracted.

        var_poisson: ndarray, 2-D
            Poisson noise variance array to be extracted following data extraction method.

        var_rnoise: ndarray, 2-D
            Read noise variance array to be extracted following data extraction method.

        var_flat: ndarray, 2-D
            Flat noise variance array to be extracted following data extraction method.

        wl_array : ndarray, 2-D, or None
            Wavelengths corresponding to `data`, or None if no WAVELENGTH
            extension was found in the input file.

        verbose : bool
            If True, log messages.

        Returns
        -------
        ra, dec : float
            ra and dec are the right ascension and declination respectively
            at the nominal center of the slit.

        wavelength : ndarray, 1-D, float64
            The wavelength in micrometers at each pixel.

        temp_flux : ndarray, 1-D
            The sum of the data values in the extraction region minus the
            sum of the data values in the background regions (scaled by the
            ratio of the numbers of pixels), for each pixel.
            The data values are in units of surface brightness, so this
            value isn't really the flux, it's an intermediate value.
            Dividing by `npixels` (to compute the average) will give the
            array for the `surf_bright` (surface brightness) output column,
            and multiplying by the solid angle of a pixel will give the
            flux for a point source.

        f_var_poisson : ndarray, 1-D
            The extracted poisson variance values to go along with the
            temp_flux array.

        f_var_rnoise : ndarray, 1-D
            The extracted read noise variance values to go along with the
            temp_flux array.

        f_var_flat : ndarray, 1-D
            The extracted flat field variance values to go along with the
            temp_flux array.

        background : ndarray, 1-D, float64
            The background count rate that was subtracted from the sum of
            the source data values to get `temp_flux`.

        b_var_poisson : ndarray, 1-D
            The extracted poisson variance values to go along with the
            background array.

        b_var_rnoise : ndarray, 1-D
            The extracted read noise variance values to go along with the
            background array.

        b_var_flat : ndarray, 1-D
            The extracted flat field variance values to go along with the
            background array.

        npixels : ndarray, 1-D, float64
            The number of pixels that were added together to get `temp_flux`.

        dq : ndarray, 1-D, uint32
            The data quality array.

        """
        # If the wavelength attribute exists and is populated, use it in preference to the wavelengths returned by the
        # wcs function.
        # But since we're now calling get_wavelengths from lib.wcs_utils, wl_array should be populated, and we should be
        # able to remove some of this code.
        got_wavelength = False if wl_array is None or len(wl_array) == 0 else True  # may be reset later

        # The default value is 0, so all 0 values means that the wavelength attribute was not populated.
        if not got_wavelength or (wl_array.min() == 0. and wl_array.max() == 0.):
            got_wavelength = False

        if got_wavelength:
            # We need a 1-D array of wavelengths, one element for each output table row.
            # These are slice limits.
            sx0 = int(round(self.xstart))
            sx1 = int(round(self.xstop)) + 1
            sy0 = int(round(self.ystart))
            sy1 = int(round(self.ystop)) + 1

            # Convert non-positive values to NaN, to easily ignore them.
            wl = wl_array.copy()  # Don't modify wl_array
            nan_flag = np.isnan(wl)

            # To avoid a warning about invalid value encountered in less_equal.
            wl[nan_flag] = -1000.
            wl = np.where(wl <= 0., np.nan, wl)

            if self.dispaxis == HORIZONTAL:
                wavelength = np.nanmean(wl[sy0:sy1, sx0:sx1], axis=0)
            else:
                wavelength = np.nanmean(wl[sy0:sy1, sx0:sx1], axis=1)

        # Now call the wcs function to compute the celestial coordinates.
        # Also use the returned wavelengths if we weren't able to get them from the wavelength attribute.

        # Used for computing the celestial coordinates.
        if self.dispaxis == HORIZONTAL:
            slice0 = int(round(self.xstart))
            slice1 = int(round(self.xstop)) + 1
            x_array = np.arange(slice0, slice1, dtype=np.float64)
            y_array = np.empty(x_array.shape, dtype=np.float64)
            y_array.fill((self.ystart + self.ystop) / 2.)
        else:
            slice0 = int(round(self.ystart))
            slice1 = int(round(self.ystop)) + 1
            y_array = np.arange(slice0, slice1, dtype=np.float64)
            x_array = np.empty(y_array.shape, dtype=np.float64)
            x_array.fill((self.xstart + self.xstop) / 2.)

        if self.wcs is not None:
            coords = self.wcs(x_array, y_array)
            ra = coords[0]
            dec = coords[1]
            wcs_wl = coords[2]

            # We need one right ascension and one declination, representing the direction of pointing.
            mask = np.isnan(wcs_wl)
            not_nan = np.logical_not(mask)

            if np.any(not_nan):
                ra2 = ra[not_nan]
                min_ra = ra2.min()
                max_ra = ra2.max()
                ra = (min_ra + max_ra) / 2.
                dec2 = dec[not_nan]
                min_dec = dec2.min()
                max_dec = dec2.max()
                dec = (min_dec + max_dec) / 2.
            else:
                if verbose:
                    log.warning("All wavelength values are NaN; assigning dummy value -999 to RA and Dec.")
                ra = -999.
                dec = -999.
        else:
            ra, dec, wcs_wl = None

        if not got_wavelength:
            wavelength = wcs_wl  # from wcs, or None

        if self.dispaxis == HORIZONTAL:
            image = data
        else:
            image = np.transpose(data, (1, 0))
            var_poisson = np.transpose(var_poisson, (1, 0))
            var_rnoise = np.transpose(var_rnoise, (1, 0))
            var_flat = np.transpose(var_flat, (1, 0))

        if wavelength is None:
            if verbose:
                log.warning("Wavelengths could not be determined.")

            if slice0 <= 0:
                wavelength = np.arange(1, slice1 - slice0 + 1, dtype=np.float64)
            else:
                wavelength = np.arange(slice0, slice1, dtype=np.float64)

        temp_wl = wavelength.copy()
        nan_mask = np.isnan(wavelength)
        n_nan = nan_mask.sum(dtype=np.intp)

        if n_nan > 0:
            if verbose:
                log.debug(f"{n_nan} NaNs in wavelength array")

            temp_wl = np.nan_to_num(temp_wl, nan=0.01)  # NaNs in the wavelength array cause problems; replace them.

        disp_range = [slice0, slice1]  # Range (slice) of pixel numbers in the dispersion direction.

        temp_flux, f_var_poisson, f_var_rnoise, f_var_flat, \
            background, b_var_poisson, b_var_rnoise, b_var_flat, \
            npixels = extract1d.extract1d(
                image,
                var_poisson,
                var_rnoise,
                var_flat,
                temp_wl,
                disp_range,
                self.p_src,
                self.p_bkg,
                self.independent_var,
                self.smoothing_length,
                self.bkg_fit,
                self.bkg_order,
                weights=None
            )

        del temp_wl

        dq = np.zeros(temp_flux.shape, dtype=np.uint32)

        if n_nan > 0:
            wavelength, dq, nan_slc = nans_at_endpoints(wavelength, dq, verbose)
            temp_flux = temp_flux[nan_slc]
            background = background[nan_slc]
            npixels = npixels[nan_slc]
            f_var_poisson = f_var_poisson[nan_slc]
            f_var_rnoise = f_var_rnoise[nan_slc]
            f_var_flat = f_var_flat[nan_slc]
            b_var_poisson = b_var_poisson[nan_slc]
            b_var_rnoise = b_var_rnoise[nan_slc]
            b_var_flat = b_var_flat[nan_slc]

        return (ra, dec, wavelength,
                temp_flux, f_var_poisson, f_var_rnoise, f_var_flat,
                background, b_var_poisson, b_var_rnoise, b_var_flat, npixels, dq)


class ImageExtractModel(ExtractBase):
    """This uses an image that specifies the extraction region.

    Extended summary
    ----------------
    One of the requirements for this step is that for an extended target,
    the entire aperture is supposed to be extracted (with no background
    subtraction).  It doesn't make any sense to use an image reference file
    to extract the entire aperture; a trivially simple JSON reference file
    would do.  Therefore, we assume that if the user specified a reference
    file in image format, the user actually wanted that reference file
    to be used, so we will ignore the requirement and extract as specified
    by the reference image.
    """

    def __init__(self, verbose, *base_args, **base_kwargs):
        """Extract using a reference image to define the extraction and
           background regions.

        Parameters
        ----------
        verbose : bool
            If True, log messages.

        *base_args, **base_kwargs :
            See ExtractionBase for more.

        """
        self.verbosity = verbose

        super().__init__(*base_args, **base_kwargs)

    def nominal_locn(
            self, middle: float, middle_wl: float
    ) -> Union[float, None]:
        """Find the nominal cross-dispersion location of the target spectrum.

        This version is for the case that the reference file is an image.

        Parameters
        ----------
        middle: int
            The zero-indexed pixel number of the point in the dispersion
            direction at which `locn_from_wcs` determined the actual
            location (in the cross-dispersion direction) of the target
            spectrum.

        middle_wl: float
            The wavelength at pixel `middle`.  This is not used in this
            version.

        Returns
        -------
        location: float or None
            The nominal cross-dispersion location (i.e. unmodified by
            source position offset) of the target spectrum.
            The value will be None if `middle` is outside the reference
            image or if the reference image does not specify any pixels
            to extract at `middle`.

        """
        shape = self.ref_image.data.shape
        middle_line = None
        location = None

        if self.dispaxis == HORIZONTAL:
            if 0 <= middle < shape[1]:
                middle_line = self.ref_image.data[:, middle]
        else:
            if 0 <= middle < shape[0]:
                middle_line = self.ref_image.data[middle, :]

        if middle_line is None:
            log.warning(
                f"Can't determine nominal location of spectrum because middle = {middle} is off the image."
            )

            return

        mask_target = np.where(middle_line > 0., 1., 0.)
        x = np.arange(len(middle_line), dtype=np.float64)

        numerator = (x * mask_target).sum()
        denominator = mask_target.sum()

        if denominator > 0.:
            location = numerator / denominator

        return location

    def add_position_correction(self, verbose: bool):
        """Shift the reference image (in-place).

        Parameters
        ----------
        verbose : bool
            If True, messages can be logged.

        """
        if self.position_correction == 0:
            return

        if verbose:
            log.info(f"Applying source offset of {self.position_correction:.2f}")

        # Shift the image in the cross-dispersion direction.
        ref = self.ref_image.data.copy()
        shift = self.position_correction
        ishift = round(shift)

        if ishift != shift:
            if verbose:
                log.info(f"Rounding source offset of {shift} to {ishift}")

        if self.dispaxis == HORIZONTAL:
            if abs(ishift) >= ref.shape[0]:
                if verbose:
                    log.warning(f"Nod offset {ishift} is too large, skipping ...")

                return

            self.ref_image.data[:, :] = 0.

            if ishift > 0:
                self.ref_image.data[ishift:, :] = ref[:-ishift, :]
            else:
                ishift = -ishift
                self.ref_image.data[:-ishift, :] = ref[ishift:, :]
        else:
            if abs(ishift) >= ref.shape[1]:
                if verbose:
                    log.warning(f"Nod offset {ishift} is too large, skipping ...")

                return

            self.ref_image.data[:, :] = 0.

            if ishift > 0:
                self.ref_image.data[:, ishift:] = ref[:, :-ishift]
            else:
                ishift = -ishift
                self.ref_image.data[:, :-ishift] = ref[:, ishift:]

    def log_extraction_parameters(self):
        """Log the updated extraction parameters."""
        log.debug("Using a reference image that defines extraction regions.")
        log.debug(f"dispaxis = {self.dispaxis}")
        log.debug(f"spectral order = {self.spectral_order}")
        log.debug(f"smoothing_length = {self.smoothing_length}")
        log.debug(f"position_correction = {self.position_correction}")

    def extract(self, data,
                var_poisson: np.ndarray,
                var_rnoise: np.ndarray,
                var_flat: np.ndarray,
                wl_array: np.ndarray,
                verbose: bool)\
            -> \
            Tuple[float, float, np.ndarray,
                  np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                  np.ndarray, np.ndarray]:
        """
        Do the actual extraction, for the case that the extract1d reference file
        is an image.

        Parameters
        ----------
        data : ndarray, 2-D
            Science data array.

        var_poisson : ndarray, 2-D
            Poisson noise variance array to be extracted following data extraction method.

        var_rnoise : ndarray, 2-D
            Read noise variance array to be extracted following data extraction method.

        var_flat : ndarray, 2-D
            Flat noise variance array to be extracted following data extraction method.

        wl_array : ndarray, 2-D, or None
            Wavelengths corresponding to `data`, or None if no WAVELENGTH
            extension was found in the input file.

        verbose : bool
            If True, log messages.

        Returns
        -------
        ra, dec : float
            ra and dec are the right ascension and declination respectively
            at the nominal center of the slit.

        wavelength : ndarray, 1-D
            The wavelength in micrometers at each pixel.

        temp_flux : ndarray, 1-D
            The sum of the data values in the extraction region minus the
            sum of the data values in the background regions (scaled by the
            ratio of the numbers of pixels), for each pixel.
            The data values are in units of surface brightness, so this
            value isn't really the flux, it's an intermediate value.
            Multiply `temp_flux` by the solid angle of a pixel to get the
            flux for a point source (column "flux").  Divide `temp_flux` by
            `npixels` (to compute the average) to get the array for the
            "surf_bright" (surface brightness) output column.

        f_var_poisson : ndarray, 1-D
            The extracted poisson variance values to go along with the
            temp_flux array.

        f_var_rnoise : ndarray, 1-D
            The extracted read noise variance values to go along with the
            temp_flux array.

        f_var_flat : ndarray, 1-D
            The extracted flat field variance values to go along with the
            temp_flux array.

        background : ndarray, 1-D
            The background count rate that was subtracted from the sum of
            the source data values to get `temp_flux`.

        b_var_poisson : ndarray, 1-D
            The extracted poisson variance values to go along with the
            background array.

        b_var_rnoise : ndarray, 1-D
            The extracted read noise variance values to go along with the
            background array.

        b_var_flat : ndarray, 1-D
            The extracted flat field variance values to go along with the
            background array.

        npixels : ndarray, 1-D, float64
            The number of pixels that were added together to get `temp_flux`.

        dq : ndarray, 1-D, uint32
        """
        shape = data.shape
        ref = self.match_shape(shape)  # Truncate or expand reference image to match the science data.

        # This is the axis along which to add up the data.
        if self.dispaxis == HORIZONTAL:
            axis = 0
        else:
            axis = 1

        # The values of these arrays will be just 0 or 1.
        # If ref did not define any background pixels, however, mask_bkg will be None.
        (mask_target, mask_bkg) = self.separate_target_and_background(ref)

        # This is the number of pixels in the cross-dispersion direction, in the target extraction region.
        n_target = mask_target.sum(axis=axis, dtype=float)

        # Extract the data.
        gross = (data * mask_target).sum(axis=axis, dtype=float)
        f_var_poisson = (var_poisson * mask_target).sum(axis=axis, dtype=float)
        f_var_rnoise = (var_rnoise * mask_target).sum(axis=axis, dtype=float)
        f_var_flat = (var_flat * mask_target).sum(axis=axis, dtype=float)

        # Compute the number of pixels that were added together to get gross.
        temp = np.ones_like(data)
        npixels = (temp * mask_target).sum(axis=axis, dtype=float)

        if self.subtract_background is not None:
            if not self.subtract_background:
                if verbose and mask_bkg is not None:
                    log.info("Background subtraction was turned off - skipping it.")
                mask_bkg = None
            else:
                if verbose and mask_bkg is None:
                    log.info("Skipping background subtraction because background regions are not defined.")

        # Extract the background.
        if mask_bkg is not None:
            n_bkg = mask_bkg.sum(axis=axis, dtype=float)
            n_bkg = np.where(n_bkg == 0., -1., n_bkg)  # -1 is used as a flag, and also to avoid dividing by zero.

            background = (data * mask_bkg).sum(axis=axis, dtype=float)
            b_var_poisson = (var_poisson * mask_bkg).sum(axis=axis, dtype=float)
            b_var_rnoise = (var_rnoise * mask_bkg).sum(axis=axis, dtype=float)
            b_var_flat = (var_flat * mask_bkg).sum(axis=axis, dtype=float)

            scalefactor = n_target / n_bkg
            scalefactor = np.where(n_bkg > 0., scalefactor, 0.)

            background *= scalefactor

            if self.smoothing_length > 1:
                background = extract1d.bxcar(background, self.smoothing_length)  # Boxcar smoothing.
                background = np.where(n_bkg > 0., background, 0.)
                b_var_poisson = extract1d.bxcar(b_var_poisson, self.smoothing_length)
                b_var_poisson = np.where(n_bkg > 0., b_var_poisson, 0.)
                b_var_rnoise = extract1d.bxcar(b_var_rnoise, self.smoothing_length)
                b_var_rnoise = np.where(n_bkg > 0., b_var_rnoise, 0.)
                b_var_flat = extract1d.bxcar(b_var_flat, self.smoothing_length)
                b_var_flat = np.where(n_bkg > 0., b_var_flat, 0.)

            temp_flux = gross - background
        else:
            background = np.zeros_like(gross)
            b_var_poisson = np.zeros_like(gross)
            b_var_rnoise = np.zeros_like(gross)
            b_var_flat = np.zeros_like(gross)
            temp_flux = gross.copy()

        del gross

        # Since we're now calling get_wavelengths from lib.wcs_utils, wl_array should be populated, and we should be
        # able to remove some of this code.
        if wl_array is None or len(wl_array) == 0:
            got_wavelength = False
        else:
            got_wavelength = True  # may be reset below

        # If wl_array has all 0 values, interpret that to mean that the wavelength attribute was not populated.
        if not got_wavelength or wl_array.min() == 0. and wl_array.max() == 0.:
            got_wavelength = False

        # Used for computing the celestial coordinates and the 1-D array of wavelengths.
        flag = (mask_target > 0.)
        grid = np.indices(shape)
        masked_grid = flag.astype(float) * grid[axis]
        g_sum = masked_grid.sum(axis=axis)
        f_sum = flag.sum(axis=axis, dtype=float)
        f_sum_zero = np.where(f_sum <= 0.)
        f_sum[f_sum_zero] = 1.  # to avoid dividing by zero

        spectral_trace = g_sum / f_sum

        del f_sum, g_sum, masked_grid, grid, flag

        # We want x_array and y_array to be 1-D arrays, with the X values initially running from 0 at the left edge of
        # the input cutout to the right edge, and the Y values being near the middle of the spectral extraction region.
        # So the locations (x_array[i], y_array[i]) should be the spectral trace.
        # Near the left and right edges, there might not be any non-zero values in mask_target, so a slice will be
        # extracted from both x_array and y_array in order to exclude pixels that are not within the extraction region.
        if self.dispaxis == HORIZONTAL:
            x_array = np.arange(shape[1], dtype=float)
            y_array = spectral_trace
        else:
            x_array = spectral_trace
            y_array = np.arange(shape[0], dtype=float)

        # Trim off the ends, if there's no data there.
        # Save trim_slc.
        mask = np.where(n_target > 0.)

        if len(mask[0]) > 0:
            trim_slc = slice(mask[0][0], mask[0][-1] + 1)
            temp_flux = temp_flux[trim_slc]
            background = background[trim_slc]
            f_var_poisson = f_var_poisson[trim_slc]
            f_var_rnoise = f_var_rnoise[trim_slc]
            f_var_flat = f_var_flat[trim_slc]
            b_var_poisson = b_var_poisson[trim_slc]
            b_var_rnoise = b_var_rnoise[trim_slc]
            b_var_flat = b_var_flat[trim_slc]
            npixels = npixels[trim_slc]
            x_array = x_array[trim_slc]
            y_array = y_array[trim_slc]

        if got_wavelength:
            indx = np.around(x_array).astype(int)
            indy = np.around(y_array).astype(int)
            indx = np.where(indx < 0, 0, indx)
            indx = np.where(indx >= shape[1], shape[1] - 1, indx)
            indy = np.where(indy < 0, 0, indy)
            indy = np.where(indy >= shape[0], shape[0] - 1, indy)
            wavelength = wl_array[indy, indx]

        ra = dec = wcs_wl = None

        if self.wcs is not None:
            stuff = self.wcs(x_array, y_array)
            ra = stuff[0]
            dec = stuff[1]
            wcs_wl = stuff[2]

            # We need one right ascension and one declination, representing the direction of pointing.
            middle = ra.shape[0] // 2  # ra and dec have same shape
            mask = np.isnan(ra)
            not_nan = np.logical_not(mask)

            if not_nan[middle]:
                if verbose:
                    log.debug("Using midpoint of spectral trace for RA and Dec.")

                ra = ra[middle]
            else:
                if np.any(not_nan):
                    if verbose:
                        log.warning(
                            "Midpoint of coordinate array is NaN; using the average of non-NaN min and max values."
                        )

                    ra = (np.nanmin(ra) + np.nanmax(ra)) / 2.
                else:
                    if verbose:
                        log.warning("All right ascension values are NaN; assigning dummy value -999.")

                    ra = -999.

            mask = np.isnan(dec)
            not_nan = np.logical_not(mask)

            if not_nan[middle]:
                dec = dec[middle]
            else:
                if np.any(not_nan):
                    dec = (np.nanmin(dec) + np.nanmax(dec)) / 2.
                else:
                    if verbose:
                        log.warning("All declination values are NaN; assigning dummy value -999.")

                    dec = -999.

        if not got_wavelength:
            wavelength = wcs_wl  # from wcs, or None

        if wavelength is None:
            if self.dispaxis == HORIZONTAL:
                wavelength = np.arange(shape[1], dtype=float)
            else:
                wavelength = np.arange(shape[0], dtype=float)

            wavelength = wavelength[trim_slc]

        dq = np.zeros(temp_flux.shape, dtype=np.uint32)
        nan_mask = np.isnan(wavelength)
        n_nan = nan_mask.sum(dtype=np.intp)

        if n_nan > 0:
            if verbose:
                log.warning(f"{n_nan} NaNs in wavelength array")

            wavelength, dq, nan_slc = nans_at_endpoints(wavelength, dq, verbose)
            temp_flux = temp_flux[nan_slc]
            background = background[nan_slc]
            npixels = npixels[nan_slc]
            f_var_poisson = f_var_poisson[nan_slc]
            f_var_rnoise = f_var_rnoise[nan_slc]
            f_var_flat = f_var_flat[nan_slc]
            b_var_poisson = b_var_poisson[nan_slc]
            b_var_rnoise = b_var_rnoise[nan_slc]
            b_var_flat = b_var_flat[nan_slc]
        return (ra, dec, wavelength,
                temp_flux, f_var_poisson, f_var_rnoise, f_var_flat,
                background, b_var_poisson, b_var_rnoise, b_var_flat,
                npixels, dq)

    def match_shape(self, shape: tuple) -> np.ndarray:
        """Truncate or expand reference image to match the science data.

        Extended summary
        ----------------
        The science data may be 2-D or 3-D, but the reference image only
        needs to be 2-D.

        Parameters
        ----------
        shape : tuple
            The shape of the science data.

        Returns
        -------
        ndarray, 2-D
            This is either the reference image (the data array, not the
            complete data model), or an array of the same type, either
            larger or smaller than the actual reference image, but matching
            the science data array both in size and location on the
            detector.

        """
        ref = self.ref_image.data
        ref_shape = ref.shape

        if shape == ref_shape:
            return ref

        # This is the shape of the last two axes of the science data.
        buf = np.zeros((shape[-2], shape[-1]), dtype=ref.dtype)
        y_max = min(shape[-2], ref_shape[0])
        x_max = min(shape[-1], ref_shape[1])
        slc0 = slice(0, y_max)
        slc1 = slice(0, x_max)
        buf[slc0, slc1] = ref[slc0, slc1].copy()

        return buf

    @staticmethod
    def separate_target_and_background(
            ref
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
        """Create masks for target and background.

        Parameters
        ----------
        ref : ndarray, 2-D
            This is the reference image as returned by `match_shape`,
            i.e. it might be a subset of the original reference image.

        Returns
        -------
        mask_target : ndarray, 2-D
            This is an array of the same type and shape as the science
            image, but with values of only 0 or 1.  A value of 1 indicates
            that the corresponding pixel in the science data array should
            be included when adding up values to make the 1-D spectrum,
            and a value of 0 means that it should not be included.

        mask_bkg : ndarray, 2-D, or None.
            This is like `mask_target` but for background regions.
            A negative value in the reference image flags a pixel that
            should be included in the background region(s).  If there is
            no pixel in the reference image with a negative value,
            `mask_bkg` will be set to None.

        """
        mask_target = np.where(ref > 0., 1., 0.)
        mask_bkg = None

        if np.any(ref < 0.):
            mask_bkg = np.where(ref < 0., 1., 0.)

        return mask_target, mask_bkg


def run_extract1d(
        input_model: DataModel,
        extract_ref_name: str,
        apcorr_ref_name: Union[str, None],
        smoothing_length: Union[int, None],
        bkg_fit: str,
        bkg_order: Union[int, None],
        bkg_sigma_clip: float,
        log_increment: int,
        subtract_background: Union[bool, None],
        use_source_posn: Union[bool, None],
        was_source_model: bool = False,
) -> DataModel:
    """Extract 1-D spectra.

    This just reads the reference files (if any) and calls do_extract1d.

    Parameters
    ----------
    input_model : data model
        The input science model.

    extract_ref_name : str
        The name of the extract1d reference file, or "N/A".

    smoothing_length : int or None
        Width of a boxcar function for smoothing the background regions.

    bkg_fit : str
        Type of fitting to apply to background values in each column
        (or row, if the dispersion is vertical).

    bkg_order : int or None
        Polynomial order for fitting to each column (or row, if the
        dispersion is vertical) of background. Only used if `bkg_fit`
        is `poly`.

    bkg_sigma_clip : float
        Sigma clipping value to use on background to remove noise/outliers

    log_increment : int
        if `log_increment` is greater than 0 and the input data are
        multi-integration, a message will be written to the log every
        `log_increment` integrations.

    subtract_background : bool or None
        User supplied flag indicating whether the background should be
        subtracted.
        If None, the value in the extract_1d reference file will be used.
        If not None, this parameter overrides the value in the
        extract_1d reference file.

    use_source_posn : bool or None
        If True, the target and background positions specified in the
        reference file (or the default position, if there is no reference
        file) will be shifted to account for source position offset.

    was_source_model : bool
        True if and only if `input_model` is actually one SlitModel
        obtained by iterating over a SourceModelContainer.  The default
        is False.

    apcorr_ref_name : str
        Name of the APCORR reference file. Default is None

    Returns
    -------
    output_model : data model
        A new MultiSpecModel containing the extracted spectra.

    """
    # Read and interpret the extract1d reference file.
    try:
        ref_dict = open_extract1d_ref(extract_ref_name, input_model.meta.exposure.type)
    except AttributeError:  # Input is a ModelContainer of some type
        ref_dict = open_extract1d_ref(extract_ref_name, input_model[0].meta.exposure.type)

    apcorr_ref_model = None

    if apcorr_ref_name is not None and apcorr_ref_name != 'N/A':
        try:
            apcorr_ref_model = open_apcorr_ref(apcorr_ref_name, input_model.meta.exposure.type)
        except AttributeError:  # SourceModelContainers don't have exposure nodes
            apcorr_ref_model = open_apcorr_ref(apcorr_ref_name, input_model[0].meta.exposure.type)

    # This item is a flag to let us know that do_extract1d was called from run_extract1d; that is, we don't expect this
    # key to be present in ref_dict if do_extract1d was called directly.
    # If this key is not in ref_dict, or if it is but it's True, then we'll set S_EXTR1D to 'COMPLETE'.
    if ref_dict is not None:
        ref_dict['need_to_set_to_complete'] = False

    output_model = do_extract1d(
        input_model,
        ref_dict,
        apcorr_ref_model,
        smoothing_length,
        bkg_fit,
        bkg_order,
        bkg_sigma_clip,
        log_increment,
        subtract_background,
        use_source_posn,
        was_source_model,
    )

    if apcorr_ref_model is not None:
        apcorr_ref_model.close()

    # Remove target.source_type from the output model, so that it
    # doesn't force creation of an empty SCI extension in the output
    # x1d product just to hold this keyword.
    output_model.meta.target.source_type = None

    return output_model


def ref_dict_sanity_check(ref_dict: Union[dict, None]) -> Union[dict, None]:
    """Check for required entries.

    Parameters
    ----------
    ref_dict : dict or None
        The contents of the extract1d reference file.

    Returns
    -------
    ref_dict : dict or None

    """
    if ref_dict is None:
        return ref_dict

    if 'ref_file_type' not in ref_dict:  # We can make an educated guess as to what this must be.
        if 'ref_model' in ref_dict:
            log.info("Assuming extract1d reference file type is image")
            ref_dict['ref_file_type'] = FILE_TYPE_IMAGE
        else:
            log.info("Assuming extract1d reference file type is JSON")
            ref_dict['ref_file_type'] = FILE_TYPE_JSON

            if 'apertures' not in ref_dict:
                raise RuntimeError("Key 'apertures' must be present in the extract1d reference file")

            for aper in ref_dict['apertures']:
                if 'id' not in aper:
                    log.warning(f"Key 'id' not found in aperture {aper} in extract1d reference file")

    return ref_dict


def do_extract1d(
        input_model: DataModel,
        extract_ref_dict: Union[dict, None],
        apcorr_ref_model=None,
        smoothing_length: Union[int, None] = None,
        bkg_fit: str = "poly",
        bkg_order: Union[int, None] = None,
        bkg_sigma_clip: float = 0,
        log_increment: int = 50,
        subtract_background: Union[int, None] = None,
        use_source_posn: Union[bool, None] = None,
        was_source_model: bool = False
) -> DataModel:
    """Extract 1-D spectra.

    In the pipeline, this function would be called by run_extract1d.
    This exists as a separate function to allow a user to call this step
    in a Python script, passing in a dictionary of parameters in order to
    bypass reading reference files.

    Parameters
    ----------
    input_model : data model
        The input science model.

    extract_ref_dict : dict, or None
        The contents of the extract1d reference file, or None in order to use
        default values.  If `ref_dict` is not None, use key 'ref_file_type'
        to specify whether the parameters are those that could be read
        from a JSON-format reference file
        (i.e. ref_dict['ref_file_type'] = "JSON")
        from a asdf-format reference file
        (i.e. ref_dict['ref_file_type'] = "ASDF")
        or parameters relevant for a reference image
        (i.e. ref_dict['ref_file_type'] = "IMAGE").

    smoothing_length : int or None
        Width of a boxcar function for smoothing the background regions.

    bkg_fit : str
        Type of fitting to perform on background values in each column
        (or row, if the dispersion is vertical).

    bkg_order : int or None
        Polynomial order for fitting to each column (or row, if the
        dispersion is vertical) of background.

    bkg_sigma_clip : float
        Sigma clipping value to use on background to remove noise/outliers

    log_increment : int
        if `log_increment` is greater than 0 and the input data are
        multi-integration, a message will be written to the log every
        `log_increment` integrations.

    subtract_background : bool or None
        User supplied flag indicating whether the background should be
        subtracted.
        If None, the value in the extract_1d reference file will be used.
        If not None, this parameter overrides the value in the
        extract_1d reference file.

    use_source_posn : bool or None
        If True, the target and background positions specified in the
        reference file (or the default position, if there is no reference
        file) will be shifted to account for source position offset.

    was_source_model : bool
        True if and only if `input_model` is actually one SlitModel
        obtained by iterating over a SourceModelContainer.  The default
        is False.

    apcorr_ref_model : `~fits.FITS_rec`, datamodel or  None
        Table of aperture correction values from the APCORR reference file.

    Returns
    -------
    output_model : data model
        A new MultiSpecModel containing the extracted spectra.
    """

    extract_ref_dict = ref_dict_sanity_check(extract_ref_dict)

    if isinstance(input_model, datamodels.SourceModelContainer):
        # log.debug('Input is a SourceModelContainer')
        was_source_model = True

    # Temporarily set "input" to either the first model in a container, or the individual input model, for convenience
    # of retrieving meta attributes in subsequent statements
    if was_source_model:  # input_model is SourceContainer with a single SlitModel
        if isinstance(input_model, datamodels.SlitModel):
            input_temp = input_model
        else:
            # log.debug(f'do_extract1d: len(input_model)={len(input_model)}')
            input_temp = input_model[0]
    else:
        input_temp = input_model

    # Setup the output model
    output_model = datamodels.MultiSpecModel()

    if hasattr(input_temp, "int_times"):
        output_model.int_times = input_temp.int_times.copy()

    output_model.update(input_temp, only='PRIMARY')

    spec_dtype = datamodels.SpecModel().spec_table.dtype  # This data type is used for creating an output table.

    # This will be relevant if we're asked to extract a spectrum and the spectral order is zero.
    # That's only OK if the disperser is a prism.
    prism_mode = is_prism(input_temp)
    instrument = input_temp.meta.instrument.name
    exp_type = input_temp.meta.exposure.type

    if instrument is not None:
        instrument = instrument.upper()

    # We need a flag to indicate whether the photom step has been run.
    # If it hasn't, we'll copy the count rate to the flux column.
    try:
        s_photom = input_temp.meta.cal_step.photom
    except AttributeError:
        s_photom = None

    if s_photom is not None and s_photom.upper() == 'COMPLETE':
        photom_has_been_run = True
        flux_units = 'Jy'
        f_var_units = 'Jy^2'
        sb_units = 'MJy/sr'
        sb_var_units = '(MJy/sr)^2'
    else:
        photom_has_been_run = False
        flux_units = 'DN/s'
        f_var_units = '(DN/s)^2'
        sb_units = 'DN/s'
        sb_var_units = '(DN/s)^2'
        log.warning("The photom step has not been run.")

    # use_source_posn doesn't apply to WFSS, so turn it off if it's currently on
    if use_source_posn:
        if exp_type in WFSS_EXPTYPES:
            use_source_posn = False
            log.warning(
                f"Correcting for source position is not supported for exp_type = "
                f"{input_temp.meta.exposure.type}, so use_source_posn will be set to False",
            )

    # Turn use_source_posn on for types that should use it by default
    if use_source_posn is None:
        if exp_type in ['MIR_LRS-FIXEDSLIT', 'MIR_MRS', 'NRS_FIXEDSLIT', 'NRS_IFU', 'NRS_MSASPEC']:
            use_source_posn = True
            log.info(
                f"Turning on source position correction for exp_type = {exp_type}"
            )

    # Handle inputs that contain one or more slit models
    if was_source_model or isinstance(input_model, datamodels.MultiSlitModel):
        if was_source_model:   # SourceContainer has a single list of SlitModels
            if isinstance(input_model, datamodels.SlitModel):
                # If input is a single SlitModel, as opposed to a list of SlitModels,
                # put it into a list so that it's iterable later on
                slits = [input_model]
            else:
                slits = input_model

            # The subsequent work on data uses the individual SlitModels, but there are many places where meta
            # attributes are retreived from input_model, so set this to allow that to work.
            if not isinstance(input_model, datamodels.SlitModel):
                input_model = input_model[0]

        elif isinstance(input_model, datamodels.MultiSlitModel):  # A simple MultiSlitModel, not in a container
            slits = input_model.slits

        # Save original use_source_posn value, because it can get
        # toggled within the following loop over slits
        save_use_source_posn = use_source_posn

        for slit in slits:  # Loop over the slits in the input model
            log.info(f'Working on slit {slit.name}')
            slitname = slit.name
            prev_offset = OFFSET_NOT_ASSIGNED_YET
            use_source_posn = save_use_source_posn  # restore original value

            if np.size(slit.data) <= 0:
                log.info(f'No data for slit {slit.name}, skipping ...')
                continue

            sp_order = get_spectral_order(slit)
            if sp_order == 0 and not prism_mode:
                log.info("Spectral order 0 is a direct image, skipping ...")
                continue

            # Turn off use_source_posn if the source is not POINT
            source_type = slit.source_type
            if source_type != 'POINT':
                use_source_posn = False
                log.info(f"Setting use_source_posn to False for source type {source_type}")

            # Turn off use_source_posn if working on non-primary NRS fixed slits
            if exp_type == 'NRS_FIXEDSLIT' and slitname != slit.meta.instrument.fixed_slit:
                use_source_posn = False
                log.info("Can only compute source location for primary NIRSpec slit, ")
                log.info("so setting use_source_posn to False")

            extract_params = get_extract_parameters(
                extract_ref_dict,
                slit,
                slit.name,
                sp_order,
                input_model.meta,
                smoothing_length,
                bkg_fit,
                bkg_order,
                use_source_posn
            )

            if subtract_background is not None:
                extract_params['subtract_background'] = subtract_background

            if extract_params['match'] == NO_MATCH:
                log.critical('Missing extraction parameters.')
                raise ValueError('Missing extraction parameters.')
            elif extract_params['match'] == PARTIAL:
                log.info(f'Spectral order {sp_order} not found, skipping ...')
                continue

            extract_params['dispaxis'] = slit.meta.wcsinfo.dispersion_direction

            if extract_params['dispaxis'] is None:
                log.warning("The dispersion direction information is missing, so skipping ...")
                continue

            if photom_has_been_run:
                pixel_solid_angle = slit.meta.photometry.pixelarea_steradians

                if pixel_solid_angle is None:
                    pixel_solid_angle = 1.
                    log.warning("Pixel area (solid angle) is not populated; the flux will not be correct.")
            else:
                pixel_solid_angle = 1.  # not needed

            try:
                ra, dec, wavelength, temp_flux, f_var_poisson,\
                    f_var_rnoise, f_var_flat, background, b_var_poisson,\
                    b_var_rnoise, b_var_flat, npixels, dq, prev_offset = extract_one_slit(
                        input_model,
                        slit,
                        -1,  # Integration number is not relevant in this case
                        prev_offset,
                        True,
                        extract_params
                    )
            except InvalidSpectralOrderNumberError as e:
                log.info(f'{str(e)}, skipping ...')
                continue

            # Convert the sum to an average, for surface brightness.
            npixels_temp = np.where(npixels > 0., npixels, 1.)
            surf_bright = temp_flux / npixels_temp  # may be reset below
            sb_var_poisson = f_var_poisson / npixels_temp
            sb_var_rnoise = f_var_rnoise / npixels_temp
            sb_var_flat = f_var_flat / npixels_temp
            background /= npixels_temp
            b_var_poisson = b_var_poisson / npixels_temp
            b_var_rnoise = b_var_rnoise / npixels_temp
            b_var_flat = b_var_flat / npixels_temp

            del npixels_temp

            # Convert to flux density.
            # The input units will normally be MJy / sr, but for NIRSpec and NIRISS SOSS point-source spectra the units
            # will be MJy.
            input_units_are_megajanskys = (
                photom_has_been_run
                and source_type == 'POINT'
                and (instrument == 'NIRSPEC' or exp_type == 'NIS_SOSS')

            )

            if photom_has_been_run:
                # for NIRSpec data and NIRISS SOSS, point source
                if input_units_are_megajanskys:
                    flux = temp_flux * 1.e6  # MJy --> Jy
                    f_var_poisson *= 1.e12  # MJy**2 --> Jy**2
                    f_var_rnoise *= 1.e12  # MJy**2 --> Jy**2
                    f_var_flat *= 1.e12  # MJy**2 --> Jy**2
                    surf_bright[:] = 0.
                    sb_var_poisson = 0.
                    sb_var_rnoise = 0.
                    sb_var_flat = 0.
                    background[:] /= pixel_solid_angle  # MJy / sr
                    b_var_poisson /= pixel_solid_angle
                    b_var_rnoise /= pixel_solid_angle
                    b_var_flat /= pixel_solid_angle

                else:
                    flux = temp_flux * pixel_solid_angle * 1.e6  # MJy / steradian --> Jy
                    f_var_poisson *= (pixel_solid_angle**2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
                    f_var_rnoise *= (pixel_solid_angle**2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
                    f_var_flat *= (pixel_solid_angle**2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
                    # surf_bright was assigned above
            else:
                flux = temp_flux  # count rate

            del temp_flux

            error = np.sqrt(f_var_poisson + f_var_rnoise + f_var_flat)
            sb_error = np.sqrt(sb_var_poisson + sb_var_rnoise + sb_var_flat)
            berror = np.sqrt(b_var_poisson + b_var_rnoise + b_var_flat)

            otab = np.array(
                list(
                    zip(wavelength, flux, error, f_var_poisson,f_var_rnoise, f_var_flat,
                        surf_bright, sb_error, sb_var_poisson, sb_var_rnoise, sb_var_flat,
                        dq, background, berror, b_var_poisson, b_var_rnoise, b_var_flat, npixels)
                ),
                dtype=spec_dtype
            )

            spec = datamodels.SpecModel(spec_table=otab)
            spec.meta.wcs = spec_wcs.create_spectral_wcs(ra, dec, wavelength)
            spec.spec_table.columns['wavelength'].unit = 'um'
            spec.spec_table.columns['flux'].unit = flux_units
            spec.spec_table.columns['flux_error'].unit = flux_units
            spec.spec_table.columns['flux_var_poisson'].unit = f_var_units
            spec.spec_table.columns['flux_var_rnoise'].unit = f_var_units
            spec.spec_table.columns['flux_var_flat'].unit = f_var_units
            spec.spec_table.columns['surf_bright'].unit = sb_units
            spec.spec_table.columns['sb_error'].unit = sb_units
            spec.spec_table.columns['sb_var_poisson'].unit = sb_var_units
            spec.spec_table.columns['sb_var_rnoise'].unit = sb_var_units
            spec.spec_table.columns['sb_var_flat'].unit = sb_var_units
            spec.spec_table.columns['background'].unit = sb_units
            spec.spec_table.columns['bkgd_error'].unit = sb_units
            spec.spec_table.columns['bkgd_var_poisson'].unit = sb_var_units
            spec.spec_table.columns['bkgd_var_rnoise'].unit = sb_var_units
            spec.spec_table.columns['bkgd_var_flat'].unit = sb_var_units
            spec.slit_ra = ra
            spec.slit_dec = dec
            spec.spectral_order = sp_order
            spec.dispersion_direction = extract_params['dispaxis']
            copy_keyword_info(slit, slit.name, spec)

            if source_type is not None and source_type.upper() == 'POINT' and apcorr_ref_model is not None:
                log.info('Applying Aperture correction.')
                # NIRSpec needs to use a wavelength in the middle of the range rather then the beginning of the range
                # for calculating the pixel scale since some wavelengths at the edges of the range won't map to the sky
                if instrument == 'NIRSPEC':
                    wl = np.median(wavelength)
                else:
                    wl = wavelength.min()

                match_kwargs = {'location': (ra, dec, wl)}
                if exp_type in ['NRS_FIXEDSLIT', 'NRS_BRIGHTOBJ']:
                    match_kwargs['slit'] = slitname

                apcorr = select_apcorr(input_model)(
                    input_model, apcorr_ref_model.apcorr_table, apcorr_ref_model.sizeunit,
                    slit_name=slitname, **match_kwargs
                )
                apcorr.apply(spec.spec_table)

            output_model.spec.append(spec)
    else:
        # These default values for slitname are not really slit names, and slitname may be assigned a better value
        # below, in the sections for input_model being an ImageModel or a SlitModel.
        slitname = input_model.meta.exposure.type

        if slitname is None:
            slitname = ANY

        if slitname == 'NIS_SOSS':
            slitname = input_model.meta.subarray.name

        # Loop over these spectral order numbers.
        if input_model.meta.exposure.type == "NIS_SOSS":
            # This list of spectral order numbers may need to be assigned differently for other exposure types.
            spectral_order_list = [1, 2, 3]
        else:
            spectral_order_list = ["not set yet"]  # For this case, we'll call get_spectral_order to get the order.

        if photom_has_been_run:
            pixel_solid_angle = input_model.meta.photometry.pixelarea_steradians

            if pixel_solid_angle is None:
                log.warning("Pixel area (solid angle) is not populated; the flux will not be correct.")
                pixel_solid_angle = 1.
        else:
            pixel_solid_angle = 1.  # not needed

        if isinstance(input_model, datamodels.SlitModel):
            source_type = input_model.source_type
            if source_type is None:
                source_type = input_model.meta.target.source_type
        else:
            source_type = input_model.meta.target.source_type

        input_units_are_megajanskys = (
            photom_has_been_run
            and source_type == 'POINT'
            and (instrument == 'NIRSPEC' or exp_type == 'NIS_SOSS')
        )

        # Turn off use_source_posn if the source is not POINT
        if source_type != 'POINT':
            use_source_posn = False
            log.info(f"Setting use_source_posn to False for source type {source_type}")

        if isinstance(input_model, datamodels.ImageModel):
            if hasattr(input_model, "name"):
                slitname = input_model.name

            prev_offset = OFFSET_NOT_ASSIGNED_YET

            for sp_order in spectral_order_list:
                if sp_order == "not set yet":
                    sp_order = get_spectral_order(input_model)

                if sp_order == 0 and not prism_mode:
                    log.info("Spectral order 0 is a direct image, skipping ...")
                    continue

                extract_params = get_extract_parameters(
                    extract_ref_dict,
                    input_model,
                    slitname,
                    sp_order,
                    input_model.meta,
                    smoothing_length,
                    bkg_fit,
                    bkg_order,
                    use_source_posn
                )

                if subtract_background is not None:
                    extract_params['subtract_background'] = subtract_background

                if extract_params['match'] == EXACT:
                    slit = None
                    extract_params['dispaxis'] = input_model.meta.wcsinfo.dispersion_direction

                    if extract_params['dispaxis'] is None:
                        log.warning("The dispersion direction information is missing, so skipping ...")
                        continue

                    try:
                        ra, dec, wavelength, temp_flux, f_var_poisson, \
                            f_var_rnoise, f_var_flat, background, b_var_poisson, \
                            b_var_rnoise, b_var_flat, npixels, dq, prev_offset = extract_one_slit(
                                input_model,
                                slit,
                                -1,
                                prev_offset,
                                True,
                                extract_params
                            )
                    except InvalidSpectralOrderNumberError as e:
                        log.info(f'{str(e)}, skipping ...')
                        continue

                elif extract_params['match'] == PARTIAL:
                    log.info(f'Spectral order {sp_order} not found, skipping ...')
                    continue
                else:
                    log.critical('Missing extraction parameters.')
                    raise ValueError('Missing extraction parameters.')

                # Convert the sum to an average, for surface brightness.
                npixels_temp = np.where(npixels > 0., npixels, 1.)
                surf_bright = temp_flux / npixels_temp
                sb_var_poisson = f_var_poisson / npixels_temp
                sb_var_rnoise = f_var_rnoise / npixels_temp
                sb_var_flat = f_var_flat / npixels_temp
                background /= npixels_temp
                b_var_poisson = b_var_poisson / npixels_temp
                b_var_rnoise = b_var_rnoise / npixels_temp
                b_var_flat = b_var_flat / npixels_temp

                del npixels_temp

                # Convert to flux density.
                if photom_has_been_run:
                    # for NIRSpec data and NIRISS SOSS, point source
                    if input_units_are_megajanskys:
                        flux = temp_flux * 1.e6  # MJy --> Jy
                        f_var_poisson *= 1.e12  # MJy**2 --> Jy**2
                        f_var_rnoise *= 1.e12  # MJy**2 --> Jy**2
                        f_var_flat *= 1.e12  # MJy**2 --> Jy**2
                        surf_bright[:] = 0.
                        sb_var_poisson = 0.
                        sb_var_rnoise = 0.
                        sb_var_flat = 0.
                        background[:] /= pixel_solid_angle  # MJy / sr
                        b_var_poisson /= pixel_solid_angle
                        b_var_rnoise /= pixel_solid_angle
                        b_var_flat /= pixel_solid_angle
                    else:
                        flux = temp_flux * pixel_solid_angle * 1.e6  # MJy / steradian --> Jy
                        f_var_poisson *= (pixel_solid_angle ** 2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
                        f_var_rnoise *= (pixel_solid_angle ** 2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
                        f_var_flat *= (pixel_solid_angle ** 2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
                else:
                    flux = temp_flux  # count rate

                del temp_flux

                error = np.sqrt(f_var_poisson + f_var_rnoise + f_var_flat)
                sb_error = np.sqrt(sb_var_poisson + sb_var_rnoise + sb_var_flat)
                berror = np.sqrt(b_var_poisson + b_var_rnoise + b_var_flat)

                otab = np.array(
                    list(
                        zip(wavelength, flux, error, f_var_poisson, f_var_rnoise, f_var_flat,
                            surf_bright, sb_error, sb_var_poisson, sb_var_rnoise, sb_var_flat,
                            dq, background, berror, b_var_poisson, b_var_rnoise, b_var_flat, npixels)
                    ),
                    dtype=spec_dtype
                )

                spec = datamodels.SpecModel(spec_table=otab)
                spec.meta.wcs = spec_wcs.create_spectral_wcs(ra, dec, wavelength)
                spec.spec_table.columns['wavelength'].unit = 'um'
                spec.spec_table.columns['flux'].unit = flux_units
                spec.spec_table.columns['flux_error'].unit = flux_units
                spec.spec_table.columns['flux_var_poisson'].unit = f_var_units
                spec.spec_table.columns['flux_var_rnoise'].unit = f_var_units
                spec.spec_table.columns['flux_var_flat'].unit = f_var_units
                spec.spec_table.columns['surf_bright'].unit = sb_units
                spec.spec_table.columns['sb_error'].unit = sb_units
                spec.spec_table.columns['sb_var_poisson'].unit = sb_var_units
                spec.spec_table.columns['sb_var_rnoise'].unit = sb_var_units
                spec.spec_table.columns['sb_var_flat'].unit = sb_var_units
                spec.spec_table.columns['background'].unit = sb_units
                spec.spec_table.columns['bkgd_error'].unit = sb_units
                spec.spec_table.columns['bkgd_var_poisson'].unit = sb_var_units
                spec.spec_table.columns['bkgd_var_rnoise'].unit = sb_var_units
                spec.spec_table.columns['bkgd_var_flat'].unit = sb_var_units
                spec.slit_ra = ra
                spec.slit_dec = dec
                spec.spectral_order = sp_order
                spec.dispersion_direction = extract_params['dispaxis']

                # The first argument of copy_keyword_info was originally intended to be a SlitModel object, but
                # ImageModel now has the attributes we will look for in this function.
                copy_keyword_info(input_model, slitname, spec)

                if source_type is not None and source_type.upper() == 'POINT' and apcorr_ref_model is not None:
                    log.info('Applying Aperture correction.')

                    if instrument == 'NIRSPEC':
                        wl = np.median(wavelength)
                    else:
                        wl = wavelength.min()

                    apcorr = select_apcorr(input_model)(
                        input_model,
                        apcorr_ref_model.apcorr_table,
                        apcorr_ref_model.sizeunit,
                        location=(ra, dec, wl)
                    )
                    apcorr.apply(spec.spec_table)

                output_model.spec.append(spec)

        elif isinstance(input_model, (datamodels.CubeModel, datamodels.SlitModel)):
            # SOSS uses this branch in both calspec2 and caltso3, where in both cases
            # the input is a single CubeModel, because caltso3 loops over exposures

            # This branch will be invoked for inputs that are a CubeModel, which typically includes
            # NIRSpec BrightObj (fixed slit) and NIRISS SOSS modes, as well as inputs that are a
            # single SlitModel, which typically includes data from a single resampled/combined slit
            # instance from level-3 processing of NIRSpec fixed slits and MOS modes.

            # Replace the default value for slitname with a more accurate value, if possible.
            slit = None

            # For NRS_BRIGHTOBJ, the slit name comes from the slit model info
            if input_model.meta.exposure.type == 'NRS_BRIGHTOBJ' and hasattr(input_model, "name"):
                slitname = input_model.name

            # For NRS_FIXEDSLIT, the slit name comes from the FXD_SLIT keyword in the model meta
            if input_model.meta.exposure.type == 'NRS_FIXEDSLIT':
                slitname = input_model.meta.instrument.fixed_slit

            # Get the pixel solid angle size for use later in converting fluxes
            if photom_has_been_run:
                pixel_solid_angle = input_model.meta.photometry.pixelarea_steradians
                if pixel_solid_angle is None:
                    log.warning("Pixel area (solid angle) is not populated; the flux will not be correct.")
                    pixel_solid_angle = 1.
            else:
                pixel_solid_angle = 1.  # not needed

            # Loop over all spectral orders available for extraction
            prev_offset = OFFSET_NOT_ASSIGNED_YET
            for sp_order in spectral_order_list:
                if sp_order == "not set yet":
                    sp_order = get_spectral_order(input_model)

                    if sp_order == 0 and not prism_mode:
                        log.info("Spectral order 0 is a direct image, skipping ...")
                        continue

                log.info(f'Processing spectral order {sp_order}')

                extract_params = get_extract_parameters(
                    extract_ref_dict,
                    input_model,
                    slitname,
                    sp_order,
                    input_model.meta,
                    smoothing_length,
                    bkg_fit,
                    bkg_order,
                    use_source_posn
                )

                if subtract_background is not None:
                    extract_params['subtract_background'] = subtract_background

                if extract_params['match'] == NO_MATCH:
                    log.critical('Missing extraction parameters.')
                    raise ValueError('Missing extraction parameters.')
                elif extract_params['match'] == PARTIAL:
                    log.warning(f'Spectral order {sp_order} not found, skipping ...')
                    continue

                extract_params['dispaxis'] = input_model.meta.wcsinfo.dispersion_direction

                if extract_params['dispaxis'] is None:
                    log.warning("The dispersion direction information is missing, so skipping ...")
                    continue

                # Loop over each integration in the input model
                verbose = True  # for just the first integration
                shape = input_model.data.shape

                if len(shape) == 3 and shape[0] == 1 or len(shape) == 2:
                    integrations = [-1]
                else:
                    log.info(f"Beginning loop over {shape[0]} integrations ...")
                    integrations = range(shape[0])

                for integ in integrations:
                    try:
                        ra, dec, wavelength, temp_flux, f_var_poisson, \
                            f_var_rnoise, f_var_flat, background, b_var_poisson, \
                            b_var_rnoise, b_var_flat, npixels, dq, prev_offset = extract_one_slit(
                                input_model,
                                slit,
                                integ,
                                prev_offset,
                                verbose,
                                extract_params
                            )
                    except InvalidSpectralOrderNumberError as e:
                        log.info(f'{str(e)}, skipping ...')
                        break

                    # Convert the sum to an average, for surface brightness.
                    npixels_temp = np.where(npixels > 0., npixels, 1.)
                    surf_bright = temp_flux / npixels_temp
                    sb_var_poisson = f_var_poisson / npixels_temp
                    sb_var_rnoise = f_var_rnoise / npixels_temp
                    sb_var_flat = f_var_flat / npixels_temp
                    background /= npixels_temp
                    b_var_poisson = b_var_poisson / npixels_temp
                    b_var_rnoise = b_var_rnoise / npixels_temp
                    b_var_flat = b_var_flat / npixels_temp

                    del npixels_temp

                    # Convert to flux density.
                    if photom_has_been_run:
                        # for NIRSpec data and NIRISS SOSS, point source
                        if input_units_are_megajanskys:
                            flux = temp_flux * 1.e6  # MJy --> Jy
                            f_var_poisson *= 1.e12  # MJy**2 --> Jy**2
                            f_var_rnoise *= 1.e12  # MJy**2 --> Jy**2
                            f_var_flat *= 1.e12  # MJy**2 --> Jy**2
                            surf_bright[:] = 0.
                            sb_var_poisson = 0.
                            sb_var_rnoise = 0.
                            sb_var_flat = 0.
                            background[:] /= pixel_solid_angle  # MJy / sr
                            b_var_poisson /= pixel_solid_angle
                            b_var_rnoise /= pixel_solid_angle
                            b_var_flat /= pixel_solid_angle
                        else:
                            flux = temp_flux * pixel_solid_angle * 1.e6  # MJy / steradian --> Jy
                            f_var_poisson *= (pixel_solid_angle ** 2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
                            f_var_rnoise *= (pixel_solid_angle ** 2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
                            f_var_flat *= (pixel_solid_angle ** 2 * 1.e12)  # (MJy / sr)**2 --> Jy**2
                    else:
                        flux = temp_flux  # count rate

                    del temp_flux

                    error = np.sqrt(f_var_poisson + f_var_rnoise + f_var_flat)
                    sb_error = np.sqrt(sb_var_poisson + sb_var_rnoise + sb_var_flat)
                    berror = np.sqrt(b_var_poisson + b_var_rnoise + b_var_flat)

                    otab = np.array(
                        list(
                            zip(wavelength, flux, error, f_var_poisson, f_var_rnoise, f_var_flat,
                                surf_bright, sb_error, sb_var_poisson, sb_var_rnoise, sb_var_flat,
                                dq, background, berror, b_var_poisson, b_var_rnoise, b_var_flat, npixels)
                        ),
                        dtype=spec_dtype
                    )

                    spec = datamodels.SpecModel(spec_table=otab)
                    spec.meta.wcs = spec_wcs.create_spectral_wcs(ra, dec, wavelength)
                    spec.spec_table.columns['wavelength'].unit = 'um'
                    spec.spec_table.columns['flux'].unit = flux_units
                    spec.spec_table.columns['flux_error'].unit = flux_units
                    spec.spec_table.columns['flux_var_poisson'].unit = f_var_units
                    spec.spec_table.columns['flux_var_rnoise'].unit = f_var_units
                    spec.spec_table.columns['flux_var_flat'].unit = f_var_units
                    spec.spec_table.columns['surf_bright'].unit = sb_units
                    spec.spec_table.columns['sb_error'].unit = sb_units
                    spec.spec_table.columns['sb_var_poisson'].unit = sb_var_units
                    spec.spec_table.columns['sb_var_rnoise'].unit = sb_var_units
                    spec.spec_table.columns['sb_var_flat'].unit = sb_var_units
                    spec.spec_table.columns['background'].unit = sb_units
                    spec.spec_table.columns['bkgd_error'].unit = sb_units
                    spec.spec_table.columns['bkgd_var_poisson'].unit = sb_var_units
                    spec.spec_table.columns['bkgd_var_rnoise'].unit = sb_var_units
                    spec.spec_table.columns['bkgd_var_flat'].unit = sb_var_units
                    spec.slit_ra = ra
                    spec.slit_dec = dec
                    spec.spectral_order = sp_order
                    spec.dispersion_direction = extract_params['dispaxis']
                    copy_keyword_info(input_model, slitname, spec)

                    if source_type is not None and source_type.upper() == 'POINT' and apcorr_ref_model is not None:
                        log.info('Applying Aperture correction.')

                        if instrument == 'NIRSPEC':
                            wl = np.median(wavelength)
                        else:
                            wl = wavelength.min()

                        match_kwargs = {'location': (ra, dec, wl)}
                        if exp_type in ['NRS_FIXEDSLIT', 'NRS_BRIGHTOBJ']:
                            match_kwargs['slit'] = slitname

                        apcorr = select_apcorr(input_model)(
                            input_model, apcorr_ref_model.apcorr_table, apcorr_ref_model.sizeunit,
                            slit_name=slitname, **match_kwargs
                        )
                        apcorr.apply(spec.spec_table)

                    output_model.spec.append(spec)

                    if log_increment > 0 and (integ + 1) % log_increment == 0:
                        if integ == -1:
                            pass
                        elif integ == 0:
                            if input_model.data.shape[0] == 1:
                                log.info("1 integration done")
                            else:
                                log.info("... 1 integration done")
                        elif integ == input_model.data.shape[0] - 1:
                            log.info(f"All {input_model.data.shape[0]} integrations done")
                        else:
                            log.info(f"... {integ + 1} integrations done")
                        progress_msg_printed = True
                    else:
                        progress_msg_printed = False

                    verbose = False

                if not progress_msg_printed:
                    if input_model.data.shape[0] == 1:
                        log.info("1 integration done")
                    else:
                        log.info(f"All {input_model.data.shape[0]} integrations done")

        elif isinstance(input_model, datamodels.IFUCubeModel):
            try:
                source_type = input_model.meta.target.source_type
            except AttributeError:
                source_type = "UNKNOWN"

            if source_type is None:
                source_type = "UNKNOWN"

            output_model = ifu.ifu_extract1d(
                input_model, extract_ref_dict, source_type, subtract_background,
                bkg_sigma_clip, apcorr_ref_model
            )

        else:
            log.error("The input file is not supported for this step.")
            raise RuntimeError("Can't extract a spectrum from this file.")

    # Copy the integration time information from the INT_TIMES table to keywords in the output file.
    if pipe_utils.is_tso(input_model):
        populate_time_keywords(input_model, output_model)
    else:
        log.debug("Not copying from the INT_TIMES table because this is not a TSO exposure.")
        if hasattr(output_model, "int_times"):
            del output_model.int_times

    output_model.meta.wcs = None  # See output_model.spec[i].meta.wcs instead.

    # If the extract1d reference file is an image, explicitly close it.
    if extract_ref_dict is not None and 'ref_model' in extract_ref_dict:
        extract_ref_dict['ref_model'].close()

    if (extract_ref_dict is None
            or 'need_to_set_to_complete' not in extract_ref_dict
            or extract_ref_dict['need_to_set_to_complete']):
        output_model.meta.cal_step.extract_1d = 'COMPLETE'

    return output_model


def populate_time_keywords(
        input_model: DataModel, output_model: DataModel
) -> Union[DataModel, None]:
    """Copy the integration times keywords to header keywords.

    Parameters
    ----------
    input_model : data model
        The input science model.

    output_model : data model
        The output science model.  This may be modified in-place.

    """
    nints = input_model.meta.exposure.nints
    int_start = input_model.meta.exposure.integration_start

    if int_start is None:
        log.warning("INTSTART not found; assuming a value of 1.")
        int_start = 1

    int_start -= 1  # zero indexed
    int_end = input_model.meta.exposure.integration_end

    if int_end is None:
        log.warning(f"INTEND not found; assuming a value of {nints}.")
        int_end = nints

    int_end -= 1  # zero indexed

    if nints > 1:
        num_integrations = int_end - int_start + 1
    else:
        num_integrations = 1

    if hasattr(input_model, 'int_times') and input_model.int_times is not None:
        nrows = len(input_model.int_times)
    else:
        nrows = 0

    if nrows < 1:
        log.warning("There is no INT_TIMES table in the input file.")
        return

    # If we have a single plane (e.g. ImageModel or MultiSlitModel), we will only populate the keywords if the
    # corresponding uncal file had one integration.
    # If the data were or might have been segmented, we use the first and last integration numbers to determine whether
    # the data were in fact averaged over integrations, and if so, we should not populate the int_times-related header
    # keywords.
    skip = False  # initial value

    if isinstance(input_model, (datamodels.MultiSlitModel, datamodels.ImageModel)):
        if num_integrations > 1:
            log.warning("Not using INT_TIMES table because the data have been averaged over integrations.")
            skip = True
    elif isinstance(input_model, (datamodels.CubeModel, datamodels.SlitModel)):
        shape = input_model.data.shape

        if len(shape) == 2 and num_integrations > 1:
            log.warning("Not using INT_TIMES table because the data have been averaged over integrations.")
            skip = True
        elif len(shape) != 3 or shape[0] > nrows:
            # Later, we'll check that the integration_number column actually
            # has a row corresponding to every integration in the input.
            log.warning(
                "Not using INT_TIMES table because the data shape is not consistent with the number of table rows."
            )
            skip = True
    elif isinstance(input_model, datamodels.IFUCubeModel):
        log.warning("The INT_TIMES table will be ignored for IFU data.")
        skip = True

    if skip:
        return

    int_num = input_model.int_times['integration_number']
    start_time_mjd = input_model.int_times['int_start_MJD_UTC']
    mid_time_mjd = input_model.int_times['int_mid_MJD_UTC']
    end_time_mjd = input_model.int_times['int_end_MJD_UTC']
    start_tdb = input_model.int_times['int_start_BJD_TDB']
    mid_tdb = input_model.int_times['int_mid_BJD_TDB']
    end_tdb = input_model.int_times['int_end_BJD_TDB']

    data_range = (int_start, int_end)  # Inclusive range of integration numbers in the input data, zero indexed.

    # Inclusive range of integration numbers in the INT_TIMES table, zero indexed.
    table_range = (int_num[0] - 1, int_num[-1] - 1)
    offset = data_range[0] - table_range[0]

    if data_range[0] < table_range[0] or data_range[1] > table_range[1]:
        log.warning("Not using the INT_TIMES table because it does not include rows for all integrations in the data.")
        return

    log.debug("TSO data, so copying times from the INT_TIMES table.")

    if hasattr(input_model, 'data'):
        shape = input_model.data.shape

        if len(shape) == 2:
            num_integ = 1
        else:  # len(shape) == 3
            num_integ = shape[0]
    else:  # e.g. MultiSlit data
        num_integ = 1

    # This assumes that the spec attribute of output_model has already been created, and spectra have been appended.
    n_output_spec = len(output_model.spec)

    # num_j is the number of spectra per integration, e.g. the number of fixed-slit spectra, MSA spectra, or different
    # spectral orders; num_integ is the number of integrations.
    # The total number of output spectra is n_output_spec = num_integ * num_j
    num_j = n_output_spec // num_integ

    if n_output_spec != num_j * num_integ:  # sanity check
        log.warning(
            f"populate_time_keywords:  Don't understand n_output_spec = {n_output_spec}, num_j = {num_j}, num_integ = "
            f"{num_integ}"
        )
    else:
        log.debug(
            f"Number of output spectra = {n_output_spec}; number of spectra for each integration = {num_j}; "
            f"number of integrations = {num_integ}"
        )

    n = 0  # Counter for spectra in output_model.

    for j in range(num_j):  # for each spectrum or order
        for k in range(num_integ):  # for each integration
            row = k + offset
            spec = output_model.spec[n]  # n is incremented below
            spec.int_num = int_num[row]
            spec.time_scale = "UTC"
            spec.start_time_mjd = start_time_mjd[row]
            spec.mid_time_mjd = mid_time_mjd[row]
            spec.end_time_mjd = end_time_mjd[row]
            spec.start_tdb = start_tdb[row]
            spec.mid_tdb = mid_tdb[row]
            spec.end_tdb = end_tdb[row]
            n += 1


def get_spectral_order(slit: DataModel) -> int:
    """Get the spectral order number.

    Parameters
    ----------
    slit : SlitModel object
        One slit from an input MultiSlitModel or similar.

    Returns
    -------
    int
        Spectral order number for `slit`.  If no information about spectral
        order is available in `wcsinfo`, a default value of 1 will be
        returned.

    """
    if hasattr(slit.meta, 'wcsinfo'):
        sp_order = slit.meta.wcsinfo.spectral_order

        if sp_order is None:
            log.warning("spectral_order is None; using 1")
            sp_order = 1
    else:
        log.warning("slit.meta doesn't have attribute wcsinfo; setting spectral order to 1")
        sp_order = 1

    return sp_order


def is_prism(input_model: DataModel) -> bool:
    """Determine whether the current observing mode used a prism.

    Extended summary
    ----------------
    The reason for this test is so we can skip spectral extraction if the
    spectral order is zero and the exposure was not made using a prism.
    In this context, therefore, a grism is not considered to be a prism.

    Parameters
    ----------
    input_model : data model
        The input science model.

    Returns
    -------
    bool
        True if the exposure used a prism; False otherwise.

    """
    instrument = input_model.meta.instrument.name

    if instrument is None:
        return False

    instrument_filter = input_model.meta.instrument.filter

    if instrument_filter is None:
        instrument_filter = "NONE"
    else:
        instrument_filter = instrument_filter.upper()

    grating = input_model.meta.instrument.grating

    if grating is None:
        grating = "NONE"
    else:
        grating = grating.upper()

    prism_mode = False

    if (
            instrument == "MIRI"
            and instrument_filter.find("P750L") >= 0
            or instrument == "NIRSPEC"
            and grating.find("PRISM") >= 0
    ):
        prism_mode = True

    return prism_mode


def copy_keyword_info(
        slit: SlitModel, slitname: Union[str, None], spec: SpecModel
):
    """Copy metadata from the input to the output spectrum.

    Parameters
    ----------
    slit : A SlitModel object
        Metadata will be copied from the input `slit` to output `spec`.

    slitname : str or None
        The name of the slit.

    spec : One element of MultiSpecModel.spec
        Metadata attributes will be updated in-place.

    """
    if slitname is not None and slitname != "ANY":
        spec.name = slitname

    if hasattr(slit, "slitlet_id"):
        spec.slitlet_id = slit.slitlet_id

    if hasattr(slit, "source_id"):
        spec.source_id = slit.source_id

    if hasattr(slit, "source_name") and slit.source_name is not None:
        spec.source_name = slit.source_name

    if hasattr(slit, "source_alias") and slit.source_alias is not None:
        spec.source_alias = slit.source_alias

    if hasattr(slit, "source_type") and slit.source_type is not None:
        spec.source_type = slit.source_type

    if hasattr(slit, "stellarity") and slit.stellarity is not None:
        spec.stellarity = slit.stellarity

    if hasattr(slit, "source_xpos"):
        spec.source_xpos = slit.source_xpos

    if hasattr(slit, "source_ypos"):
        spec.source_ypos = slit.source_ypos

    if hasattr(slit, "source_ra"):
        spec.source_ra = slit.source_ra

    if hasattr(slit, "source_dec"):
        spec.source_dec = slit.source_dec

    if hasattr(slit, "shutter_state"):
        spec.shutter_state = slit.shutter_state


def extract_one_slit(
        input_model: DataModel,
        slit: SlitModel,
        integ: int,
        prev_offset: Union[float, str],
        verbose: bool,
        extract_params: dict
) -> Tuple[float, float, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, float]:
    """Extract data for one slit, or spectral order, or plane.

    Parameters
    ----------
    input_model : data model
        The input science model.

    slit : one slit from a MultiSlitModel (or similar), or None
        If slit is None, the data array is input_model.data; otherwise,
        the data array is slit.data.
        In the former case, if `integ` is zero or larger, the spectrum
        will be extracted from the 2-D slice input_model.data[integ].

    integ : int
        For the case that input_model is a SlitModel or a CubeModel,
        `integ` is the integration number.  If the integration number is
        not relevant (i.e. the data array is 2-D), `integ` should be -1.

    prev_offset : float or str
        When extracting from multi-integration data, the source position
        offset only needs to be determined once. `prev_offset` is either the
        previously computed offset or a value (a string) indicating that
        the offset hasn't been computed yet.  In the latter case, method
        `offset_from_offset` will be called to determine the offset.

    verbose : boolean
        If True, log more info (extraction parameters, for example).

    extract_params : dict
        Parameters read from the extract1d reference file.

    Returns
    -------
    ra, dec : float
        ra and dec are the right ascension and declination respectively
        at the nominal center of the slit.

    wavelength : ndarray, 1-D, float64
        The wavelength in micrometers at each pixel.

    temp_flux : ndarray, 1-D, float64
        The sum of the data values in the extraction region minus the sum
        of the data values in the background regions (scaled by the ratio
        of the numbers of pixels), for each pixel.
        The data values are in units of surface brightness, so this value
        isn't really the flux, it's an intermediate value.  Multiply
        `temp_flux` by the solid angle of a pixel to get the flux for a
        point source (column "flux").  Divide `temp_flux` by `npixels` (to
        compute the average) to get the array for the "surf_bright"
        (surface brightness) output column.

    f_var_poisson : ndarray, 1-D
        The extracted poisson variance values to go along with the
        temp_flux array.

    f_var_rnoise : ndarray, 1-D
        The extracted read noise variance values to go along with the
        temp_flux array.

    f_var_flat : ndarray, 1-D
        The extracted flat field variance values to go along with the
        temp_flux array.

    background : ndarray, 1-D
        The background count rate that was subtracted from the sum of
        the source data values to get `temp_flux`.

    b_var_poisson : ndarray, 1-D
        The extracted poisson variance values to go along with the
        background array.

    b_var_rnoise : ndarray, 1-D
        The extracted read noise variance values to go along with the
        background array.

    b_var_flat : ndarray, 1-D
        The extracted flat field variance values to go along with the
        background array.

    npixels : ndarray, 1-D, float64
        The number of pixels that were added together to get `temp_flux`.

    dq : ndarray, 1-D, uint32
        The data quality array.

    offset : float
       The source position offset in the cross-dispersion direction, either
       computed by calling `offset_from_offset` in this function, or
       copied from the input `prev_offset`.

    """

    if verbose:
        log_initial_parameters(extract_params)

    try:
        exp_type = input_model.meta.exposure.type
    except AttributeError:
        exp_type = slit.meta.exposure.type

    if integ > -1:
        data = input_model.data[integ]
        var_poisson = input_model.var_poisson[integ]
        var_rnoise = input_model.var_rnoise[integ]
        var_flat = input_model.var_flat[integ]
        input_dq = input_model.dq[integ]
    elif slit is None:
        data = input_model.data
        var_poisson = input_model.var_poisson
        var_rnoise = input_model.var_rnoise
        var_flat = input_model.var_flat
        input_dq = input_model.dq
    else:
        data = slit.data
        var_poisson = slit.var_poisson
        var_rnoise = slit.var_rnoise
        var_flat = slit.var_flat
        input_dq = slit.dq

    if input_dq.size == 0:
        input_dq = None

    wl_array = get_wavelengths(input_model if slit is None else slit, exp_type, extract_params['spectral_order'])
    data = replace_bad_values(data, input_dq, wl_array)

    if extract_params['ref_file_type'] == FILE_TYPE_IMAGE:  # The reference file is an image.
        extract_model = ImageExtractModel(input_model=input_model, slit=slit, verbose=verbose, **extract_params)
        ap = None
    else:
        # If there is an extract1d reference file (there doesn't have to be), it's in JSON format.
        extract_model = ExtractModel(input_model=input_model, slit=slit, verbose=verbose, **extract_params)
        ap = get_aperture(data.shape, extract_model.wcs, verbose, extract_params)
        extract_model.update_extraction_limits(ap)

    if extract_model.use_source_posn:
        if prev_offset == OFFSET_NOT_ASSIGNED_YET:  # Only call this method for the first integration.
            offset, locn = extract_model.offset_from_offset(input_model, slit, verbose)

            if verbose:
                if offset is not None and locn is not None:
                    log.debug(f"Computed source offset={offset:.2f}, source location={locn:.2f}")

            if not extract_model.use_source_posn:
                offset = 0.
        else:
            offset = prev_offset
    else:
        offset = 0.

    extract_model.position_correction = offset

    # Add the source position offset to the polynomial coefficients, or shift the reference image
    # (depending on the type of reference file).
    extract_model.add_position_correction(verbose, data.shape)

    if verbose:
        extract_model.log_extraction_parameters()

    extract_model.assign_polynomial_limits(verbose)

    # Log the extraction limits being used; verbose flag is used to
    # turn off logging for loops over multiple integrations
    if verbose:
        log.info("Using extraction limits: ")
        if extract_model.src_coeff is not None:
            # Because src_coeff was specified, that will be used instead of xstart/xstop (or ystart/ystop).
            if extract_model.dispaxis == HORIZONTAL:
                # Only print xstart/xstop, because ystart/ystop are not used
                log.info(f"xstart={extract_model.xstart}, "
                         f"xstop={extract_model.xstop}, and src_coeff")
            else:
                # Only print ystart/ystop, because xstart/xstop are not used
                log.info(f"ystart={extract_model.ystart}, "
                         f"ystop={extract_model.ystop}, and src_coeff")
        else:
            # No src_coeff, so print all xstart/xstop and ystart/ystop values
            log.info(f"xstart={extract_model.xstart}, xstop={extract_model.xstop}, "
                     f"ystart={extract_model.ystart}, ystop={extract_model.ystop}")

        if extract_params['subtract_background']:
            log.info("with background subtraction")

    ra, dec, wavelength, temp_flux, f_var_poisson,\
        f_var_rnoise, f_var_flat, background, b_var_poisson,\
        b_var_rnoise, b_var_flat, npixels, dq = extract_model.extract(data, var_poisson, var_rnoise,
                                                                      var_flat, wl_array, verbose)

    return (ra, dec, wavelength, temp_flux, f_var_poisson, f_var_rnoise, f_var_flat,
            background, b_var_poisson, b_var_rnoise, b_var_flat, npixels, dq, offset)


def replace_bad_values(
        data: np.ndarray,
        input_dq: Union[np.ndarray, None],
        wl_array: np.ndarray
) -> np.ndarray:
    """Replace values flagged with DO_NOT_USE or that have NaN wavelengths.

    Parameters
    ----------
    data : ndarray
        The science data array.

    input_dq : ndarray or None
        If not None, this will be checked for flag value DO_NOT_USE.  The
        science data will be set to NaN for every pixel that is flagged
        with DO_NOT_USE in `input_dq`.

    wl_array : ndarray, 2-D
        Wavelengths corresponding to `data`.  For any element of this
        array that is NaN, the corresponding element in `data` will be
        set to NaN.

    Returns
    -------
    ndarray
        A possibly modified copy of `data`.  If no change was made, this
        will be a view rather than a copy.  Values that are set to NaN
        should not be included when doing the 1-D spectral extraction.

    """
    mask = np.isnan(wl_array)

    if input_dq is not None:
        bad_mask = np.bitwise_and(input_dq, dqflags.pixel['DO_NOT_USE']) > 0
        mask = np.logical_or(mask, bad_mask)

    if np.any(mask):
        mod_data = data.copy()
        mod_data[mask] = np.nan
        return mod_data

    return data


def nans_at_endpoints(
        wavelength: np.ndarray,
        dq: np.ndarray,
        verbose: bool
) -> Tuple[np.ndarray, np.ndarray, slice]:
    """Flag NaNs in the wavelength array.

    Extended summary
    ----------------
    Both input arrays should be 1-D and have the same shape.
    If NaNs are present at endpoints of `wavelength`, the arrays will be
    trimmed to remove the NaNs.  NaNs at interior elements of `wavelength`
    will be left in place, but they will be flagged with DO_NOT_USE in the
    `dq` array.

    Parameters
    ----------
    wavelength : ndarray
        Array of wavelengths, possibly containing NaNs.

    dq : ndarray
        Data quality array.

    verbose : bool
        If True and the arrays were trimmed, log a message.

    Returns
    -------
    wavelength, dq : ndarray
        The returned `dq` array may have NaNs flagged with DO_NOT_USE,
        and both arrays may have been trimmed at either or both ends.

    slc : slice
        The slice to be applied to other output arrays to match the modified
        shape of the wavelength array.
    """
    # The input arrays will not be modified in-place.
    new_wl = wavelength.copy()
    new_dq = dq.copy()
    nelem = wavelength.shape[0]
    slc = slice(nelem)

    nan_mask = np.isnan(wavelength)
    new_dq[nan_mask] = np.bitwise_or(new_dq[nan_mask], dqflags.pixel['DO_NOT_USE'])
    not_nan = np.logical_not(nan_mask)
    flag = np.where(not_nan)

    if len(flag[0]) > 0:
        n_trimmed = flag[0][0] + nelem - (flag[0][-1] + 1)

        if n_trimmed > 0:
            if verbose:
                log.info(f"Output arrays have been trimmed by {n_trimmed} elements")

            slc = slice(flag[0][0], flag[0][-1] + 1)
            new_wl = new_wl[slc]
            new_dq = new_dq[slc]
    else:
        new_dq |= dqflags.pixel['DO_NOT_USE']

    return new_wl, new_dq, slc
