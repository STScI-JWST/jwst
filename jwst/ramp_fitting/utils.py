#! /usr/bin/env python
#
# utils.py: utility functions
from __future__ import division, print_function

from __future__ import division
import numpy as np
import logging

from .. import datamodels
from ..datamodels import dqflags
from ..lib import reffile_utils

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class OptRes(object):
    """
    Object to hold optional results for all good pixels for
    y-intercept, slope, uncertainty for y-intercept, uncertainty for
    slope, inverse variance, first frame (for pedestal image), and
    cosmic ray magnitude.
    """

    def __init__(self, n_int, imshape, max_seg, nreads):
        """
        Short Summary
        -------------
        Initialize the optional attributes. These are 4D arrays for the
        segment-specific values of the y-intercept, the slope, the uncertainty
        associated with both, the weights, the approximate cosmic ray
        magnitudes, and the inverse variance.  These are 3D arrays for the
        integration-specific first frame and pedestal values.

        Parameters
        ----------
        n_int: int
            number of integrations in data set

        imshape: (int, int) tuple
            shape of 2D image

        max_seg: int
            maximum number of segments fit

        nreads: int
            number of reads in an integration
        """

        self.yint_seg = np.zeros((n_int,)+(max_seg,)+imshape,dtype=np.float32)
        self.slope_seg = np.zeros((n_int,)+(max_seg,)+imshape,dtype=np.float32)
        self.sigyint_seg = np.zeros((n_int,)+(max_seg,)+imshape,dtype=np.float32)
        self.sigslope_seg = np.zeros((n_int,)+(max_seg,)+imshape,dtype=np.float32)
        self.inv_var_seg = np.zeros((n_int,)+(max_seg,)+imshape,dtype=np.float32)
        self.firstf_int = np.zeros((n_int,) + imshape, dtype=np.float32)
        self.ped_int = np.zeros((n_int,) + imshape, dtype=np.float32)
        self.cr_mag_seg = np.zeros((n_int,) + (nreads,) + imshape, dtype=np.int16)
        self.var_p_seg = np.zeros((n_int,)+(max_seg,)+imshape,dtype=np.float32)
        self.var_r_seg = np.zeros((n_int,)+(max_seg,)+imshape,dtype=np.float32)


    def init_2d(self, npix, max_seg):
        """
        Short Summary
        -------------
        Initialize the 2D segment-specific attributes for the current data
        section.

        Parameters
        ----------
        npix: integer
            number of pixels in section of 2D array

        max_seg: integer
            maximum number of segments that will be fit within an
            integration, calculated over all pixels and all integrations

        Returns
        -------
        None

        """
        self.interc_2d = np.zeros((max_seg, npix), dtype=np.float32)
        self.slope_2d = np.zeros((max_seg, npix), dtype=np.float32)
        self.siginterc_2d = np.zeros((max_seg, npix), dtype=np.float32)
        self.sigslope_2d = np.zeros((max_seg, npix), dtype=np.float32)
        self.inv_var_2d = np.zeros((max_seg, npix), dtype=np.float32)
        self.firstf_2d = np.zeros((max_seg, npix), dtype=np.float32)


    def reshape_res(self, num_int, rlo, rhi, sect_shape, ff_sect):
        """
        Short Summary
        -------------
        Loop over the segments and copy the reshaped 2D segment-specific
        results for the current data section to the 4D output arrays.

        Parameters
        ----------
        num_int: int
            integration number

        rlo: int
            first column of section

        rhi: int
            last column of section

        sect_sect: (int,int) tuple
            shape of section image

        ff_sect: (int,int) tuple
            shape of first frame image

        Returns
        -------
        """

        for ii_seg in range(0, self.yint_seg.shape[1]):
            self.yint_seg[num_int, ii_seg, rlo:rhi, :] = \
                           self.interc_2d[ii_seg, :].reshape(sect_shape)
            self.slope_seg[num_int, ii_seg, rlo:rhi, :] = \
                            self.slope_2d[ii_seg, :].reshape(sect_shape)
            self.sigyint_seg[num_int, ii_seg, rlo:rhi, :] = \
                              self.siginterc_2d[ii_seg, :].reshape(sect_shape)
            self.sigslope_seg[num_int, ii_seg, rlo:rhi, :] = \
                               self.sigslope_2d[ii_seg, :].reshape(sect_shape)
            self.inv_var_seg[num_int, ii_seg, rlo:rhi, :] = \
                              self.inv_var_2d[ii_seg, :].reshape(sect_shape)
            self.firstf_int[num_int, rlo:rhi, :] = ff_sect


    def append_arr(self, num_seg, g_pix, intercept, slope, sig_intercept,
                    sig_slope, inv_var):
        """
        Short Summary
        -------------
        Add the fitting results for the current segment to the 2d arrays.

        Parameters
        ----------
        num_seg: int, 1D array
            counter for segment number within the section

        g_pix: int, 1D array
            pixels having fitting results in current section

        intercept: float, 1D array
            intercepts for pixels in current segment and section

        slope: float, 1D array
            slopes for pixels in current segment and section

        sig_intercept: float, 1D array
            uncertainties of intercepts for pixels in current segment
            and section

        sig_slope: float, 1D array
            uncertainties of slopes for pixels in current segment and
            section

        inv_var: float, 1D array
            reciprocals of variances for fits of pixels in current
            segment and section

        Returns
        -------
        None

        """
        self.interc_2d[num_seg[g_pix], g_pix] = intercept[g_pix]
        self.slope_2d[num_seg[g_pix], g_pix] = slope[g_pix]
        self.siginterc_2d[num_seg[g_pix], g_pix] = sig_intercept[g_pix]
        self.sigslope_2d[num_seg[g_pix], g_pix] = sig_slope[g_pix]
        self.inv_var_2d[num_seg[g_pix], g_pix] = inv_var[g_pix]


    def shrink_crmag(self, n_int, dq_cube, imshape, nreads):
        """
        Extended Summary
        ----------------
        Compress the 4D cosmic ray magnitude array for the current
        integration, removing all reads whose cr magnitude is 0 for
        pixels having at least one read with a non-zero magnitude. For
        every integration, the depth of the array is equal to the
        maximum number of cosmic rays flagged in all pixels in all
        integrations.  This routine currently involves a loop over all
        pixels having at least 1 read flagged; if this algorithm takes
        too long for datasets having an overabundance of cosmic rays,
        this routine will require further optimization.

        Parameters
        ----------
        n_int: int
            number of integrations in dataset

        dq_cube: 4D float array
            input data quality array

        imshape: (int, int) tuple
            shape of a single input image

        nreads: int
            number of reads in an integration

        Returns
        ----------
        None

        """
        # Loop over data integrations to find max num of crs flagged per pixel
        # (this could exceed the maximum number of segments fit)
        max_cr = 0
        for ii_int in range(0, n_int):
            dq_int = dq_cube[ii_int, :, :, :]
            dq_cr = np.bitwise_and(dqflags.group['JUMP_DET'], dq_int)
            max_cr_int = (dq_cr > 0.).sum(axis=0).max()
            max_cr = max(max_cr, max_cr_int)

        # Allocate compressed array based on max number of crs
        cr_com = np.zeros((n_int,) + (max_cr,) + imshape, dtype=np.int16)

        # Loop over integrations and reads: for those pix having a cr, add
        #    the magnitude to the compressed array
        for ii_int in range(0, n_int):
            cr_mag_int = self.cr_mag_seg[ii_int, :, :, :]
            cr_int_has_cr = np.where(cr_mag_int.sum(axis=0) != 0)

            # Initialize number of crs for each image pixel for this integration
            end_cr = np.zeros(imshape, dtype=np.int16)

            for k_rd in range(nreads):
                # loop over pixels having a CR
                for nn in range(len(cr_int_has_cr[0])):
                    y, x = cr_int_has_cr[0][nn], cr_int_has_cr[1][nn]

                    if (cr_mag_int[k_rd, y, x] > 0.):
                        cr_com[ii_int, end_cr[y, x], y, x] = cr_mag_int[k_rd, y, x]
                        end_cr[y, x] += 1

        max_num_crs = end_cr.max()
        self.cr_mag_seg = cr_com [:,:max_num_crs,:,:]


    def output_optional(self, model, effintim):
        """
        Short Summary
        -------------
        These results are the cosmic ray magnitudes in the
        segment-specific results for the count rates, y-intercept,
        uncertainty in the slope, uncertainty in the y-intercept,
        pedestal image, fitting weights, and the uncertainties in
        the slope due to poisson noise only and read noise only, and
        the integration-specific results for the pedestal image.  The
        slopes are divided by the effective integration time here to
        yield the count rates.

        Parameters
        ----------
        model: instance of Data Model
            DM object for input

        effintim: float
            effective integration time for a single read

        Returns
        -------
        rfo_model: Data Model object
        """

        rfo_model = \
        datamodels.RampFitOutputModel(\
            slope=self.slope_seg.astype(np.float32) / effintim,
            sigslope=self.sigslope_seg.astype(np.float32),
            var_p=self.var_p_seg.astype(np.float32),
            var_r=self.var_r_seg.astype(np.float32),
            yint=self.yint_seg.astype(np.float32),
            sigyint=self.sigyint_seg.astype(np.float32),
            pedestal=self.ped_int.astype(np.float32),
            weights=self.weights.astype(np.float32),
            crmag=self.cr_mag_seg)

        rfo_model.meta.filename = model.meta.filename

        return rfo_model


    def print_full(self):
        """
        Short Summary
        -------------
        Diagnostic function for printing optional output arrays; most
        useful for tiny datasets

        Parameters
        ----------

        Returns
        -------
        None
        """
        print('Will now print all optional output arrays - ')
        print(' yint_seg: ')
        print((self.yint_seg))
        print('  ')
        print(' slope_seg: ')
        print(self.slope_seg)
        print('  ')
        print(' sigyint_seg: ')
        print(self.sigyint_seg)
        print('  ')
        print(' sigslope_seg: ')
        print(self.sigslope_seg)
        print('  ')
        print(' inv_var_2d: ')
        print((self.inv_var_2d))
        print('  ')
        print(' firstf_int: ')
        print((self.firstf_int))
        print('  ')
        print(' ped_int: ')
        print((self.ped_int))
        print('  ')
        print(' cr_mag_seg: ')
        print((self.cr_mag_seg))


def alloc_int(n_int, imshape):
    """
    Short Summary
    -------------
    Allocate arrays for integration-specific results.

    Parameters
    ----------
    n_int: int
        number of integrations

    imshape: (int, int) tuple
        shape of a single image

    Returns
    -------
    slope_int: float, 3D array
        Cube of integration-specific slopes

    err_int: float, 3D array
        Cube of integration-specific uncertainties

    dq_int: int, 3D array
        Cube of integration-specific group data quality values

    m_by_var_int: float, 3D array
        Cube of integration-specific values of the slope divided by
        variance

    inv_var_int: float, 3D array
        Cube of integration-specific inverse variance values

    var_p_int: float, 3D array
        Cube of integration-specific values for the slope variance due to 
        Poisson noise only

    var_r_int: float, 3D array
        Cube of integration-specific values for the slope variance due to 
        read noise only
    """

    slope_int = np.zeros((n_int,) + imshape, dtype=np.float64)
    err_int = np.zeros((n_int,) + imshape, dtype=np.float64)
    dq_int = np.zeros((n_int,) + imshape, dtype=np.uint32)
    m_by_var_int = np.zeros((n_int,) + imshape, dtype=np.float64)
    inv_var_int = np.zeros((n_int,) + imshape, dtype=np.float64)

    var_p_int = np.zeros((n_int,) + imshape, dtype=np.float64)
    var_r_int = np.zeros((n_int,) + imshape, dtype=np.float64)

    return slope_int, err_int, dq_int, m_by_var_int, inv_var_int, \
           var_p_int, var_r_int


def calc_vars_int(slope_int, readnoise_2d, gdq_4d, ii_int, var_p_int, \
                   var_r_int):
    """
    Short Summary
    -------------
    Calculate the integration-specific variance arrays for the given
    integration.

    Parameters
    ----------
    slope_int: float, 3D array
        cube of integration-specific slopes

    readnoise_2d: float, 2D array; multiplied by gain
        readnoise subarray

    gdq_4d: 4-D ndarray
        The group data quality array.

    ii_int: int
        integration number

    var_p_int: float, 3D array
        Cube of integration-specific values for the slope variance due to 
        Poisson noise only

    var_r_int: float, 3D array
        Cube of integration-specific values for the slope variance due to 
        read noise only

    Returns
    -------
    var_poisson: float, 2D array
        Slice (corresponding to the given integration) of the slope
        variance array due to Poisson noise only

    var_readnoise: float, 2D array
        Slice (corresponding to the given integration) of the slope
        variance array due to read noise only
    """
    (nint, nreads, asize2, asize1) = gdq_4d.shape
    npix = asize1 * asize2
    imshape = (asize2, asize1)

    # create int-specific chunks of input arrays
    gdq_2d = gdq_4d[ii_int,:,:,:].reshape(( nreads, npix ))
    slope_1d = slope_int[ii_int,:,:].reshape( npix )
    rn_1d = readnoise_2d.reshape( npix )

    gdq_2d_nan = gdq_2d.copy()  # group dq with SATS will be replaced by nans
    gdq_2d_nan = gdq_2d_nan.astype(np.float32)

    wh_sat = np.where(np.bitwise_and(gdq_2d, dqflags.group['SATURATED']))
    if (len( wh_sat[0]) > 0):
        gdq_2d_nan[ wh_sat ] = np.nan  # set all SAT reads to nan

    # get lengths of semiramps for all pix [number_of_semiramps, number_of_pix]
    segments = gdq_2d.copy()* 0.

    # counter of semiramp for each pixel
    sr_index = np.zeros( npix, dtype=np.uint8 )
    pix_not_done = np.ones( npix, dtype=np.bool)  # initialize to True

    i_read = 0
    # loop over reads for all pixels
    while (i_read < nreads and np.any(pix_not_done)): 
        gdq_1d = gdq_2d_nan[ i_read, :]

        wh_good = np.where( gdq_1d == 0) # good reads
        # if this read is good, increment that pixel's SR's length
        if ( len( wh_good[0] > 0 )):
            segments[ sr_index[ wh_good], wh_good ] += 1

        # Locate any CRs that appear before the first SAT read...
        wh_cr = np.where( gdq_2d_nan[ i_read, :] == dqflags.group['JUMP_DET'])

        # ... but not on final read:
        if (len(wh_cr[0] > 0 ) and (i_read < nreads-1) ):
            sr_index[ wh_cr[0] ] += 1
            segments[ sr_index[wh_cr], wh_cr ] += 1

        wh_nan = np.where( np.isnan(gdq_2d_nan[ i_read, :]))
        if ( len( wh_nan[0] > 0 )):
            pix_not_done[ wh_nan[0]] = False

        i_read += 1

    segments = segments.astype(np.uint8)

    ngroups_pix = segments.sum(0) # total number of groups per pixel
    nsegs_pix = (segments != 0).sum(0) #  number of segments per pixel
    sum_wgt_p = ngroups_pix - nsegs_pix # weight sum for poisson

    # calculate variance due to poisson noise only
    sum_p = nsegs_pix * slope_1d

    var_poisson = sum_p / (sum_wgt_p * nsegs_pix)
    var_poisson[ np.isnan( var_poisson )] = 0.
    var_poisson[ np.isinf( var_poisson )] = 0.
    var_poisson = var_poisson.reshape( imshape )

    # calculate variance due to readnoise only
    var_r = rn_1d**2 * 12./(segments**3. - segments)
    var_r[ np.isinf(var_r)] = 0.
    var_r[ np.isnan(var_r)] = 0.

    sum_r = (segments*var_r[:]/((nsegs_pix)**3.)).sum(axis=0)
    sum_r[ np.isnan( sum_r )] = 0.
    sum_r[ np.isinf( sum_r )] = 0.

    var_readnoise = sum_r/nsegs_pix
    var_readnoise[ np.isnan( var_readnoise )] = 0.
    var_readnoise[ np.isinf( var_readnoise )] = 0.
    var_readnoise = var_readnoise.reshape( imshape )

    return (var_poisson, var_readnoise)


def calc_slope_int(slope_int, m_by_var_int, inv_var_int, num_int):
    """
    Short Summary
    -------------
    Calculate the integration-specific slope array for the given
    integration

    Parameters
    ----------
    slope_int: float, 3D array
        cube of integration-specific slopes

    m_by_var_int: float, 3D array
        cube of integration-specific slopes divided by variance

    inv_var_int: float, 3D array
        cube of integration-specific inverse variances

    num_int: int
        integration number

    Returns
    -------
    slope_slice: float, 2D array
        slope image for given integration
    """
    slope_slice = slope_int[num_int, :, :].copy()
    m_slice = m_by_var_int[num_int, :, :]
    v_slice = inv_var_int[num_int, :, :]
    wh_v = (v_slice != 0.0)
    slope_slice[wh_v] = m_slice[wh_v] / v_slice[wh_v]

    return slope_slice


def calc_pedestal(num_int, slope_int, firstf_int, dq_cube, nframes, groupgap,
                  dropframes1):
    """
    Short Summary
    -------------
    The pedestal is calculated by extrapolating the final slope for
    each pixel from its value at the first sample in the integration to
    an exposure time of zero; this calculation accounts for the values of
    nframes and groupgap.  Any pixel that is saturated on the 1st
    read is given a pedestal value of 0.

    Parameters
    ----------
    num_int: int
        integration number

    slope_int: float, 3D array
        cube of integration-specific slopes

    firstf_int: float, 2D array
        integration-specific first frame array

    dq_cube: int, 4D array
        hypercube of DQ array

    nframes: int
        number of frames averaged per group; from the NFRAMES keyword. Does
        not contain the groupgap.

    groupgap: int
        number of frames dropped between groups, from the GROUPGAP keyword.

    dropframes1: int
        number of frames dropped at the beginning of every integration.
        Currently this is not stored as an available keyword, so the value
        is hardcoded to be 0.

    Returns
    -------
    ped: float, 2D array
        pedestal image
    """
    ff_all = firstf_int[num_int, :, :].astype(np.float32)
    dq_first = dq_cube[num_int, 0, :, :]
    ped = ff_all - slope_int[num_int, : :] * \
             (((nframes + 1.)/2. + dropframes1)/(nframes+groupgap))

    ped[dq_first == dqflags.group['SATURATED']] = 0

    return ped


def output_integ(model, slope_int, dq_int, effintim, var_p_int, var_r_int):

    """
    Short Summary
    -------------
    Construct the output integration-specific results

    Parameters
    ----------
    model: instance of Data Model
       DM object for input

    slope_int: float, 3D array
       Data cube of weighted slopes for each integration

    dq_int: int, 3D array
       Data cube of DQ arrays for each integration

    effintim: float
       Effective integration time per integration

    var_p_int: float, 3D array
        Cube of integration-specific values for the slope variance due to
        Poisson noise only

    var_r_int: float, 3D array
        Cube of integration-specific values for the slope variance due to
        read noise only

    Returns
    -------
    cubemod: Data Model object

    """
    cubemod = datamodels.CubeModel()
    cubemod.data = slope_int / effintim
    cubemod.err = np.sqrt(var_p_int + var_r_int)
    cubemod.dq = dq_int
    cubemod.var_p_int = var_p_int
    cubemod.var_r_int = var_r_int

    cubemod.update(model) # keys from input needed for photom step

    return cubemod


def gls_output_optional(model, intercept_int, intercept_err_int,
                        pedestal_int,
                        ampl_int, ampl_err_int):
    """Construct the optional results for the GLS algorithm.

    Short Summary
    -------------
    Construct the GLS-specific optional output data.
    These results are the Y-intercepts, uncertainties in the intercepts,
    pedestal (first group extrapolated back to zero time),
    cosmic ray magnitudes, and uncertainties in the CR magnitudes.

    Parameters
    ----------
    model: instance of Data Model
        Data model object for input; this is used only for the file name.

    intercept_int: 3-D ndarray, float32, shape (n_int, ny, nx)
        Y-intercept for each integration, at each pixel.

    intercept_err_int: 3-D ndarray, float32, shape (n_int, ny, nx)
        Uncertainties for Y-intercept for each integration, at each pixel.

    pedestal_int: 3-D ndarray, float32, shape (n_int, ny, nx)
        The pedestal, for each integration and each pixel.

    ampl_int: 4-D ndarray, float32, shape (n_int, ny, nx, max_num_cr)
        Cosmic-ray amplitudes for each integration, at each pixel, and for
        each CR hit in the ramp.  max_num_cr will be the maximum number of
        CRs within the ramp for any pixel, or it will be one if there were
        no CRs at all.

    ampl_err_int: 4-D ndarray, float32, shape (n_int, ny, nx, max_num_cr)
        Uncertainties for cosmic-ray amplitudes for each integration, at
        each pixel, and for each CR in the ramp.

    Returns
    -------
    gls_ramp_model: GLS_RampFitModel object
        GLS-specific ramp fit data for the exposure.
    """

    gls_ramp_model = datamodels.GLS_RampFitModel()

    gls_ramp_model.yint = intercept_int
    gls_ramp_model.sigyint = intercept_err_int
    gls_ramp_model.pedestal = pedestal_int
    gls_ramp_model.crmag = ampl_int
    gls_ramp_model.sigcrmag = ampl_err_int

    return gls_ramp_model


def gls_pedestal(first_group, slope_int, s_mask,
                 frame_time, nframes_used):

    """Calculate the pedestal for the GLS case.

    The pedestal is the first group, but extrapolated back to zero time
    using the slope obtained by the fit to the whole ramp.  The time of
    the first group is the frame time multiplied by (M + 1) / 2, where M
    is the number of frames per group, not including the number (if any)
    of skipped frames.

    The input arrays and output pedestal are slices of the full arrays.
    They are just the relevant data for the current integration (assuming
    that this function is called within a loop over integrations), and
    they may include only a subset of image lines.  For example, this
    function might be called with slope_int argument given as:
    `slope_int[num_int, rlo:rhi, :]`.  The input and output parameters
    are in electrons.

    Parameters
    ----------
    first_group: float32, 2-D array
        A slice of the first group in the ramp.

    slope_int: float32, 2-D array
        The slope obtained by GLS ramp fitting.  This is a slice for the
        current integration and a subset of the image lines.

    s_mask: bool, 2-D array
        True for ramps that were saturated in the first group.

    frame_time: float
        The time to read one frame, in seconds.

    nframes_used: int
        Number of frames that were averaged together to make a group.
        Exludes the groupgap.

    Returns
    -------
    pedestal: float32, 2-D array
        This is a slice of the full pedestal array, and it's for the
        current integration.
    """

    M = float(nframes_used)
    pedestal = first_group - slope_int * frame_time * (M + 1.) / 2.
    if s_mask.any():
        pedestal[s_mask] = 0.

    return pedestal


def shift_z(a, off):
    """
    Short Summary
    -------------
    Shift input 3D array by requested offset in z-direction, padding
    shifted array (of the same size) by leading or trailing zeros as
    needed.

    Parameters
    ----------
    a: float, 3D array
        input array

    off: int
        offset in z-direction

    Returns
    -------
    b: float, 3D array
        shifted array

    """
    # set initial and final indices along z-direction for original and
    #    shifted 3D arrays
    ai_z = int((abs(off) + off) / 2)
    af_z = a.shape[0] + int((-abs(off) + off) / 2)

    bi_z = a.shape[0] - af_z
    bf_z = a.shape[0] - ai_z

    b = a * 0
    b[bi_z:bf_z, :, :] = a[ai_z:af_z, :, :]

    return b


def get_efftim_ped(model):
    """
    Short Summary
    -------------
    Calculate the effective integration time for a single read, and return the
    number of frames per group, and the number of frames dropped between groups.

    Parameters
    ----------
    model: instance of Data Model
        DM object for input

    Returns
    -------
    effintim: float
        effective integration time for a single read

    nframes: int
        number of frames averaged per group; from the NFRAMES keyword.

    groupgap: int
        number of frames dropped between groups; from the GROUPGAP keyword.

    dropframes1: int
        number of frames dropped at the beginning of every integration. 
        Currently this is not stored as an available keyword, so the value
        returned is 0.
    """
    groupgap = model.meta.exposure.groupgap
    nframes = model.meta.exposure.nframes
    frame_time = model.meta.exposure.frame_time
    dropframes1 = 0

    try:
        effintim = (nframes + groupgap) * frame_time
    except ValueError:
        log.error('Can not retrieve values needed to calculate integ. time')

    log.debug('Calculating effective integration time for a single group using:')
    log.debug(' groupgap: %s' % (groupgap))
    log.debug(' nframes: %s' % (nframes))
    log.debug(' frame_time: %s' % (frame_time))
    log.debug(' dropframes1: %s' % (dropframes1))
    log.info('Effective integration time per group: %s' % (effintim))

    return effintim, nframes, groupgap, dropframes1


def get_dataset_info(model):
    """
    Short Summary
    -------------
    Extract values for the number of reads, the number of pixels,
    dataset shapes, the number of integrations, the instrument name,
    the frame time, and the observation time

    Parameters
    ----------
    model: instance of Data Model
       DM object for input

    Returns
    -------
    nreads: int
       number of reads in input dataset

    npix: int
       number of pixels in 2D array

    imshape: (int, int) tuple
       shape of 2D image

    cubeshape: (int, int, int) tuple
       shape of input dataset

    n_int: int
       number of integrations

    instrume: string
       instrument

    frame_time: float
       integration time from TGROUP

    ngroups: int
        number of groups per integration
    """

    instrume = model.meta.instrument.name
    frame_time = model.meta.exposure.frame_time
    ngroups = model.meta.exposure.ngroups

    n_int = model.data.shape[0]
    nreads = model.data.shape[1]
    asize2 = model.data.shape[2]
    asize1 = model.data.shape[3]

    npix = asize2 * asize1  # number of pixels in 2D array
    imshape = (asize2, asize1)
    cubeshape = (nreads,) + imshape

    return nreads, npix, imshape, cubeshape, n_int, instrume, frame_time, ngroups


def get_more_info(model):
    """Get information used by GLS algorithm.

    Parameters
    ----------
    model: instance of Data Model
        DM object for input

    Returns
    -------
    group_time: float
        Time increment between groups, in seconds.

    nframes_used: int
        Number of frames that were averaged together to make a group,
        i.e. excluding skipped frames.

    saturated_flag: int
        Group data quality flag that indicates a saturated pixel.

    jump_flag: int
        Group data quality flag that indicates a cosmic ray hit.
    """

    group_time = model.meta.exposure.group_time
    nframes_used = model.meta.exposure.nframes
    saturated_flag = dqflags.group['SATURATED']
    jump_flag = dqflags.group['JUMP_DET']

    return (group_time, nframes_used, saturated_flag, jump_flag)


def get_max_num_cr(gdq_cube, jump_flag):
    """
    Short Summary
    -------------
    Find the maximum number of cosmic-ray hits in any one pixel.

    Parameters
    ----------
    gdq_cube: 3-D ndarray
        The group data quality array.

    jump_flag: int
        The data quality flag indicating a cosmic-ray hit.

    Returns
    -------
    max_num_cr: int
        The maximum number of cosmic-ray hits for any pixel.
    """

    cr_flagged = np.empty(gdq_cube.shape, dtype=np.uint8)
    cr_flagged[:] = np.where(np.bitwise_and(gdq_cube, jump_flag), 1, 0)
    max_num_cr = cr_flagged.sum(axis=0, dtype=np.int32).max()
    del cr_flagged

    return max_num_cr


def reset_bad_gain(pdq, gain):
    """
    Short Summary
    -------------
    For pixels in the gain array that are either non-positive or NaN, reset the
    the corresponding pixels in the pixel DQ array to NO_GAIN_VALUE and
    DO_NOT_USE so that they will be ignored.

    Parameters
    ----------
    pdq: int, 2D array
        pixel dq array of input model

    gain: float32, 2D array
        grain array from reference file

    Returns
    -------
    pdq: int, 2D array
        pixleldq array of input model, reset to NO_GAIN_VALUE and DO_NOT_USE
        for pixels in the gain array that are either non-positive or NaN.
    """
    wh_g = np.where( gain <= 0.)
    if len(wh_g[0] > 0):
        pdq[wh_g] = np.bitwise_or( pdq[wh_g], dqflags.pixel['NO_GAIN_VALUE'] )
        pdq[wh_g] = np.bitwise_or( pdq[wh_g], dqflags.pixel['DO_NOT_USE'] )

    wh_g = np.where( np.isnan( gain ))
    if len(wh_g[0] > 0):
        pdq[wh_g] = np.bitwise_or( pdq[wh_g], dqflags.pixel['NO_GAIN_VALUE'] )
        pdq[wh_g] = np.bitwise_or( pdq[wh_g], dqflags.pixel['DO_NOT_USE'] )

    return pdq


def get_ref_subs(model, readnoise_model, gain_model):
    """
    Short Summary
    -------------
    Get readnoise array for calculation of variance of noiseless ramps, and
    the gain array in case optimal weighting is to be done.

    Parameters
    ----------
    model: data model
        input data model, assumed to be of type RampModel

    readnoise_model: instance of data Model
        readnoise for all pixels

    gain_model: instance of gain Model
        gain for all pixels

    Returns
    -------
    readnoise_2d: float, 2D array; multiplied by gain
        readnoise subarray

    gain_2d: float, 2D array
        gain subarray
    """
    if reffile_utils.ref_matches_sci(model, gain_model):
        gain_2d = gain_model.data
    else:
        log.info('Extracting gain subarray to match science data')
        gain_2d = reffile_utils.get_subarray_data(model, gain_model)

    if reffile_utils.ref_matches_sci(model, readnoise_model):
        readnoise_2d = readnoise_model.data
    else:
        log.info('Extracting readnoise subarray to match science data')
        readnoise_2d = reffile_utils.get_subarray_data(model, readnoise_model)

    readnoise_2d *= gain_2d # convert read noise to correct units

    return readnoise_2d, gain_2d
