"""V1 Calculation based on time and engineering database info
"""
from collections import defaultdict
import logging

from astropy.table import Table

from . import set_telescope_pointing as stp
from .. import datamodels as dm

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ['v1_calculate_from_models', 'v1_calculate_over_time', 'simplify_table']


def v1_calculate_from_models(sources, **calc_wcs_from_time_kwargs):
    """Calculate V1 over the time period for the given models

    Parameters
    ----------
    sources : [File-like or jwst.datamodels.Datamodel[...]]

    calc_wcs_from_time_kwargs : dict
        Keyword arguments to pass to `calc_wcs_from_time`

    Returns
    -------
    v1_table : astropy.table.Table
        Table of V1 pointing
    """
    # Initialize structures.
    v1_dict = defaultdict(list)
    siaf = stp.SIAF(v2_ref=0., v3_ref=0., v3yangle=0., vparity=1.)

    # Calculate V1 for all sources.
    for source in sources:
        with dm.open(source) as model:
            obsstart = model.meta.exposure.start_time
            obsend = model.meta.exposure.end_time
            guide_star_wcs = stp.WCSRef(
                model.meta.guidestar.gs_ra,
                model.meta.guidestar.gs_dec,
                model.meta.guidestar.gs_pa
            )

            obstimes, _, vinfos = stp.calc_wcs_over_time(
                obsstart, obsend, siaf=siaf, guide_star_wcs=guide_star_wcs, **calc_wcs_from_time_kwargs
            )
        sources = [source] * len(obstimes)
        v1_dict['source'] += sources
        v1_dict['obstime'] += obstimes
        v1_dict['v1'] += vinfos

    # Format and return.
    v1_table = Table(v1_dict)
    return v1_table


def v1_calculate_over_time(obsstart, obsend, **calc_wcs_from_time_kwargs):
    """Calculate V1 over the given time period

    Parameters
    ----------
    obsstart, obsend : float
        The MJD start and end time to search for pointings.

    calc_wcs_from_time_kwargs : dict
        Keyword arguments to pass to `calc_wcs_from_time`

    Returns
    -------
    v1_table : astropy.table.Table
        Table of V1 pointing
    """
    # Initialize structures.
    siaf = stp.SIAF(v2_ref=0., v3_ref=0., v3yangle=0., vparity=1.)

    # Calculate V1 for all sources.
    obstimes, _, vinfos = stp.calc_wcs_over_time(
        obsstart, obsend, siaf=siaf, **calc_wcs_from_time_kwargs
    )
    v1_dict = dict()
    v1_dict['source'] = ['time range'] * len(obstimes)
    v1_dict['obstime'] = obstimes
    v1_dict['v1'] = vinfos

    # Format and return.
    v1_table = Table(v1_dict)
    return v1_table


def simplify_table(v1_table):
    """Convert pure object-based table to ASCII/Human-friendly

    The tables as produced by the `v1_calculate` functions use native objects.
    For instance, the "obstime" column contains `astopy.time.Time` objects and
    "v1" is the `jwst.lib.set_telescope_pointing.WCSREF` object

    This routine converts such objects to strings or Python-native builtin objects.

    Parameters
    ----------
    v1_table: astropy.table.Table
        V1 table as produced by `v1_calculate` functions.

    Returns
    -------
    formatted: astropy.table.Table
        Reformatted table.
    """
    if v1_table['source'].dtype == object:
        source_formatted = [v.meta.filename for v in v1_table['source']]
    else:
        source_formatted = v1_table['source']
    obstime_formatted = v1_table['obstime'].isot
    ras, decs, pa_v3s = list(map(list, zip(*v1_table['v1'])))

    formatted = Table(
        [source_formatted, obstime_formatted, ras, decs, pa_v3s],
        names=('source', 'obstime', 'ra', 'dec', 'pa_v3')
    )
    return formatted
