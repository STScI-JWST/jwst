#! /usr/bin/env python
#
# utils.py: utility functions
import logging
import multiprocessing
import numpy as np
import warnings

from ..datamodels import dqflags  # TODO here for flags

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Replace zero or negative variances with this:
LARGE_VARIANCE = 1.e8

# TODO Change dependencies when moved to STCAL
DO_NOT_USE = dqflags.group['DO_NOT_USE']
JUMP_DET = dqflags.group['JUMP_DET']
SATURATED = dqflags.group['SATURATED']
UNRELIABLE_SLOPE = dqflags.pixel['UNRELIABLE_SLOPE']


class OptRes:
    """
    Object to hold optional results for all good pixels for
    y-intercept, slope, uncertainty for y-intercept, uncertainty for
    slope, inverse variance, first frame (for pedestal image), and
    cosmic ray magnitude.
    """

    def __init__(self, n_int, imshape, max_seg, nreads, save_opt):
        """
        Initialize the optional attributes. These are 4D arrays for the
        segment-specific values of the y-intercept, the slope, the uncertainty
        associated with both, the weights, the approximate cosmic ray
        magnitudes, and the inverse variance.  These are 3D arrays for the
        integration-specific first frame and pedestal values.

        Parameters
        ----------
        n_int : int
            number of integrations in data set

        imshape : (int, int) tuple
            shape of 2D image

        max_seg : int
            maximum number of segments fit

        nreads : int
            number of reads in an integration

        save_opt : boolean
           save optional fitting results
        """
        self.slope_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
        if save_opt:
            self.yint_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.sigyint_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.sigslope_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.inv_var_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.firstf_int = np.zeros((n_int,) + imshape, dtype=np.float32)
            self.ped_int = np.zeros((n_int,) + imshape, dtype=np.float32)
            self.cr_mag_seg = np.zeros((n_int,) + (nreads,) + imshape, dtype=np.float32)
            self.var_p_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)
            self.var_r_seg = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32)

    def init_2d(self, npix, max_seg, save_opt):
        """
        Initialize the 2D segment-specific attributes for the current data
        section.

        Parameters
        ----------
        npix : integer
            number of pixels in section of 2D array

        max_seg : integer
            maximum number of segments that will be fit within an
            integration, calculated over all pixels and all integrations

        save_opt : boolean
            save optional fitting results

        Returns
        -------
        None
        """
        self.slope_2d = np.zeros((max_seg, npix), dtype=np.float32)
        if save_opt:
            self.interc_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.siginterc_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.sigslope_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.inv_var_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.firstf_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.var_s_2d = np.zeros((max_seg, npix), dtype=np.float32)
            self.var_r_2d = np.zeros((max_seg, npix), dtype=np.float32)

    def reshape_res(self, num_int, rlo, rhi, sect_shape, ff_sect, save_opt):
        """
        Loop over the segments and copy the reshaped 2D segment-specific
        results for the current data section to the 4D output arrays.

        Parameters
        ----------
        num_int : int
            integration number

        rlo : int
            first column of section

        rhi : int
            last column of section

        sect_sect : (int,int) tuple
            shape of section image

        ff_sect : 2D array
            first frame data

        save_opt : boolean
            save optional fitting results

        Returns
        -------
        """
        for ii_seg in range(0, self.slope_seg.shape[1]):
            self.slope_seg[num_int, ii_seg, rlo:rhi, :] = \
                self.slope_2d[ii_seg, :].reshape(sect_shape)

            if save_opt:
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
                   sig_slope, inv_var, save_opt):
        """
        Add the fitting results for the current segment to the 2d arrays.

        Parameters
        ----------
        num_seg : int, 1D array
            counter for segment number within the section

        g_pix : int, 1D array
            pixels having fitting results in current section

        intercept : float, 1D array
            intercepts for pixels in current segment and section

        slope : float, 1D array
            slopes for pixels in current segment and section

        sig_intercept : float, 1D array
            uncertainties of intercepts for pixels in current segment
            and section

        sig_slope : float, 1D array
            uncertainties of slopes for pixels in current segment and
            section

        inv_var : float, 1D array
            reciprocals of variances for fits of pixels in current
            segment and section

        save_opt : boolean
            save optional fitting results

        Returns
        -------
        None
        """
        self.slope_2d[num_seg[g_pix], g_pix] = slope[g_pix]

        if save_opt:
            self.interc_2d[num_seg[g_pix], g_pix] = intercept[g_pix]
            self.siginterc_2d[num_seg[g_pix], g_pix] = sig_intercept[g_pix]
            self.sigslope_2d[num_seg[g_pix], g_pix] = sig_slope[g_pix]
            self.inv_var_2d[num_seg[g_pix], g_pix] = inv_var[g_pix]

    def shrink_crmag(self, n_int, dq_cube, imshape, nreads):
        """
        Compress the 4D cosmic ray magnitude array for the current
        integration, removing all groups whose cr magnitude is 0 for
        pixels having at least one group with a non-zero magnitude. For
        every integration, the depth of the array is equal to the
        maximum number of cosmic rays flagged in all pixels in all
        integrations.  This routine currently involves a loop over all
        pixels having at least 1 group flagged; if this algorithm takes
        too long for datasets having an overabundance of cosmic rays,
        this routine will require further optimization.

        Parameters
        ----------
        n_int : int
            number of integrations in dataset

        dq_cube : 4D float array
            input data quality array

        imshape : (int, int) tuple
            shape of a single input image

        nreads : int
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
        cr_com = np.zeros((n_int,) + (max_cr,) + imshape, dtype=np.float32)

        # Loop over integrations and groups: for those pix having a cr, add
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
        if max_num_crs == 0:
            max_num_crs = 1
            self.cr_mag_seg = np.zeros(shape=(n_int, 1, imshape[0], imshape[1]))
        else:
            self.cr_mag_seg = cr_com[:, :max_num_crs, :, :]

    def output_optional(self, effintim):
        """
        These results are the cosmic ray magnitudes in the
        segment-specific results for the count rates, y-intercept,
        uncertainty in the slope, uncertainty in the y-intercept,
        pedestal image, fitting weights, and the uncertainties in
        the slope due to poisson noise only and read noise only, and
        the integration-specific results for the pedestal image.  The
        slopes are divided by the effective integration time here to
        yield the count rates. Any variance values that are a large fraction
        of the default value LARGE_VARIANCE correspond to non-existent segments,
        so will be set to 0 here before output.

        Parameters
        ----------
        effintim : float
            effective integration time for a single group

        Returns
        -------
        opt_info: tuple
            The tuple of computed optional results arrays for fitting.
        """
        self.var_p_seg[self.var_p_seg > 0.4 * LARGE_VARIANCE] = 0.
        self.var_r_seg[self.var_r_seg > 0.4 * LARGE_VARIANCE] = 0.

        # Suppress, then re-enable, arithmetic warnings
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
        # Tiny 'weights' values correspond to non-existent segments, so set to 0.
        self.weights[1. / self.weights > 0.4 * LARGE_VARIANCE] = 0.
        warnings.resetwarnings()

        slope = self.slope_seg.astype(np.float32) / effintim
        sigslope = self.sigslope_seg.astype(np.float32)
        var_poisson = self.var_p_seg.astype(np.float32)
        var_rnoise = self.var_r_seg.astype(np.float32)
        yint = self.yint_seg.astype(np.float32)
        sigyint = self.sigyint_seg.astype(np.float32)
        pedestal = self.ped_int.astype(np.float32)
        weights = self.weights.astype(np.float32)
        crmag = self.cr_mag_seg

        opt_info = (slope, sigslope, var_poisson, var_rnoise,
                    yint, sigyint, pedestal, weights, crmag)

        return opt_info

    def print_full(self):  # pragma: no cover
        """
        Diagnostic function for printing optional output arrays; most
        useful for tiny datasets

        Parameters
        ----------
        None

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


def alloc_arrays_1(n_int, imshape):
    """
    Allocate arrays for integration-specific results and segment-specific
    results and variances.

    Parameters
    ----------
    n_int : int
        number of integrations

    imshape : (int, int) tuple
        shape of a single image

    Returns
    -------
    dq_int : int, 3D array
        Cube of integration-specific group data quality values

    median_diffs_2d : float, 2D array
        Estimated median slopes

    num_seg_per_int : integer, 3D array
        Cube of numbers of segments for all integrations and pixels

    sat_0th_group_int : uint8, 3D array
        Integration-specific slice whose value for a pixel is 1 if the initial
        group of the ramp is saturated
    """
    dq_int = np.zeros((n_int,) + imshape, dtype=np.uint32)
    num_seg_per_int = np.zeros((n_int,) + imshape, dtype=np.uint8)

    # for estimated median slopes
    median_diffs_2d = np.zeros(imshape, dtype=np.float32)
    sat_0th_group_int = np.zeros((n_int,) + imshape, dtype=np.uint8)

    return (dq_int, median_diffs_2d, num_seg_per_int, sat_0th_group_int)


def alloc_arrays_2(n_int, imshape, max_seg):
    """
    Allocate arrays for integration-specific results and segment-specific
    results and variances.

    Parameters
    ----------
    n_int : int
        number of integrations

    imshape : (int, int) tuple
        shape of a single image

    max_seg : int
        maximum number of segments fit

    Returns
    -------
    var_p3 : float, 3D array
        Cube of integration-specific values for the slope variance due to
        Poisson noise only

    var_r3 : float, 3D array
        Cube of integration-specific values for the slope variance due to
        readnoise only

    var_p4 : float, 4D array
        Hypercube of segment- and integration-specific values for the slope
        variance due to Poisson noise only

    var_r4 : float, 4D array
        Hypercube of segment- and integration-specific values for the slope
        variance due to read noise only

    var_both4 : float, 4D array
        Hypercube of segment- and integration-specific values for the slope
        variance due to combined Poisson noise and read noise

    var_both3 : float, 3D array
        Cube of segment- and integration-specific values for the slope
        variance due to combined Poisson noise and read noise

    inv_var_both4 : float, 4D array
        Hypercube of reciprocals of segment- and integration-specific values for
        the slope variance due to combined Poisson noise and read noise

    s_inv_var_p3 : float, 3D array
        Cube of reciprocals of segment- and integration-specific values for
        the slope variance due to Poisson noise only, summed over segments

    s_inv_var_r3 : float, 3D array
        Cube of reciprocals of segment- and integration-specific values for
        the slope variance due to read noise only, summed over segments

    s_inv_var_both3 : float, 3D array
        Cube of reciprocals of segment- and integration-specific values for
        the slope variance due to combined Poisson noise and read noise, summed
        over segments

    segs_4 : integer, 4D array
        Hypercube of lengths of segments for all integrations and pixels
    """
    # Initialize variances so that non-existing ramps and segments will have
    #   negligible contributions
    # Integration-specific:
    var_p3 = np.zeros((n_int,) + imshape, dtype=np.float32) + LARGE_VARIANCE
    var_r3 = var_p3.copy()
    var_both3 = var_p3.copy()
    s_inv_var_p3 = np.zeros_like(var_p3)
    s_inv_var_r3 = np.zeros_like(var_p3)
    s_inv_var_both3 = np.zeros_like(var_p3)

    # Segment-specific:
    var_p4 = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.float32) + LARGE_VARIANCE
    var_r4 = var_p4.copy()
    var_both4 = var_p4.copy()
    inv_var_both4 = np.zeros_like(var_p4)

    # number of segments
    segs_4 = np.zeros((n_int,) + (max_seg,) + imshape, dtype=np.uint8)

    return (var_p3, var_r3, var_p4, var_r4, var_both4, var_both3,
            inv_var_both4, s_inv_var_p3, s_inv_var_r3,
            s_inv_var_both3, segs_4)


def calc_slope_vars(rn_sect, gain_sect, gdq_sect, group_time, max_seg):
    """
    Calculate the segment-specific variance arrays for the given
    integration.

    Parameters
    ----------
    rn_sect : float, 2D array
        read noise values for all pixels in data section

    gain_sect : float, 2D array
        gain values for all pixels in data section

    gdq_sect : int, 3D array
        data quality flags for pixels in section

    group_time : float
        Time increment between groups, in seconds.

    max_seg : int
        maximum number of segments fit

    Returns
    -------
    den_r3 : float, 3D array
        for a given integration, the reciprocal of the denominator of the
        segment-specific variance of the segment's slope due to read noise

    den_p3 : float, 3D array
        for a given integration, the reciprocal of the denominator of the
        segment-specific variance of the segment's slope due to Poisson noise

    num_r3 : float, 3D array
        numerator of the segment-specific variance of the segment's slope
        due to read noise

    segs_beg_3 : integer, 3D array
        lengths of segments for all pixels in the given data section and
        integration
    """
    (nreads, asize2, asize1) = gdq_sect.shape
    npix = asize1 * asize2
    imshape = (asize2, asize1)

    # Create integration-specific sections of input arrays for determination
    #   of the variances.
    gdq_2d = gdq_sect[:, :, :].reshape((nreads, npix))
    gain_1d = gain_sect.reshape(npix)
    gdq_2d_nan = gdq_2d.copy()  # group dq with SATS will be replaced by nans
    gdq_2d_nan = gdq_2d_nan.astype(np.float32)

    wh_sat = np.where(np.bitwise_and(gdq_2d, dqflags.group['SATURATED']))
    if len(wh_sat[0]) > 0:
        gdq_2d_nan[wh_sat] = np.nan  # set all SAT groups to nan

    del wh_sat

    # Get lengths of semiramps for all pix [number_of_semiramps, number_of_pix]
    segs = np.zeros_like(gdq_2d)

    # Counter of semiramp for each pixel
    sr_index = np.zeros(npix, dtype=np.uint8)
    pix_not_done = np.ones(npix, dtype=bool)  # initialize to True

    i_read = 0
    # Loop over reads for all pixels to get segments (segments per pixel)
    while (i_read < nreads and np.any(pix_not_done)):
        gdq_1d = gdq_2d_nan[i_read, :]
        wh_good = np.where(gdq_1d == 0)  # good groups

        # if this group is good, increment those pixels' segments' lengths
        if len(wh_good[0]) > 0:
            segs[sr_index[wh_good], wh_good] += 1
        del wh_good

        # Locate any CRs that appear before the first SAT group...
        wh_cr = np.where(gdq_2d_nan[i_read, :].astype(np.int32) & dqflags.group['JUMP_DET'] > 0)

        # ... but not on final read:
        if (len(wh_cr[0]) > 0 and (i_read < nreads - 1)):
            sr_index[wh_cr[0]] += 1
            segs[sr_index[wh_cr], wh_cr] += 1

        del wh_cr

        # If current group is a NaN, this pixel is done (pix_not_done is False)
        wh_nan = np.where(np.isnan(gdq_2d_nan[i_read, :]))
        if len(wh_nan[0]) > 0:
            pix_not_done[wh_nan[0]] = False

        del wh_nan

        i_read += 1

    segs = segs.astype(np.uint8)
    segs_beg = segs[:max_seg, :]  # the leading nonzero lengths

    # Create reshaped version [ segs, y, x ] to simplify computation
    segs_beg_3 = segs_beg.reshape(max_seg, imshape[0], imshape[1])
    segs_beg_3 = remove_bad_singles(segs_beg_3)

    # Create a version 1 less for later calculations for the variance due to
    #   Poisson, with a floor=1 to handle single-group segments
    wh_pos_3 = np.where(segs_beg_3 > 1)
    segs_beg_3_m1 = segs_beg_3.copy()
    segs_beg_3_m1[wh_pos_3] -= 1
    segs_beg_3_m1[segs_beg_3_m1 < 1] = 1

    # For a segment, the variance due to Poisson noise
    #   = slope/(tgroup * gain * (ngroups-1)),
    #   where slope is the estimated median slope, tgroup is the group time,
    #   and ngroups is the number of groups in the segment.
    #   Here the denominator of this quantity will be computed, which will be
    #   later multiplied by the estimated median slope.

    # Suppress, then re-enable, harmless arithmetic warnings, as NaN will be
    #   checked for and handled later
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    den_p3 = 1. / (group_time * gain_1d.reshape(imshape) * segs_beg_3_m1)
    warnings.resetwarnings()

    # For a segment, the variance due to readnoise noise
    # = 12 * readnoise**2 /(ngroups_seg**3. - ngroups_seg)/( tgroup **2.)
    num_r3 = 12. * (rn_sect / group_time)**2.  # always >0

    # Reshape for every group, every pixel in section
    num_r3 = np.dstack([num_r3] * max_seg)
    num_r3 = np.transpose(num_r3, (2, 0, 1))

    # Denominator den_r3 = 1./(segs_beg_3 **3.-segs_beg_3). The minimum number
    #   of allowed groups is 2, which will apply if there is actually only 1
    #   group; in this case den_r3 = 1/6. This covers the case in which there is
    #   only one good group at the beginning of the integration, so it will be
    #   be compared to the plane of (near) zeros resulting from the reset. For
    #   longer segments, this value is overwritten below.
    den_r3 = num_r3.copy() * 0. + 1. / 6
    wh_seg_pos = np.where(segs_beg_3 > 1)

    # Suppress, then, re-enable harmless arithmetic warnings, as NaN will be
    #   checked for and handled later
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    den_r3[wh_seg_pos] = 1. / (segs_beg_3[wh_seg_pos] ** 3. -
                               segs_beg_3[wh_seg_pos])  # overwrite where segs>1
    warnings.resetwarnings()

    return (den_r3, den_p3, num_r3, segs_beg_3)


def calc_pedestal(num_int, slope_int, firstf_int, dq_first, nframes, groupgap,
                  dropframes1):
    """
    The pedestal is calculated by extrapolating the final slope for each pixel
    from its value at the first sample in the integration to an exposure time
    of zero; this calculation accounts for the values of nframes and groupgap.
    Any pixel that is saturated on the 1st group is given a pedestal value of 0.

    Parameters
    ----------
    num_int : int
        integration number

    slope_int : float, 3D array
        cube of integration-specific slopes

    firstf_int : float, 3D array
        integration-specific first frame array

    dq_first : int, 2D array
        DQ of the initial group for all ramps in the given integration

    nframes : int
        number of frames averaged per group; from the NFRAMES keyword. Does
        not contain the groupgap.

    groupgap : int
        number of frames dropped between groups, from the GROUPGAP keyword.

    dropframes1 : int
        number of frames dropped at the beginning of every integration, from
        the DRPFRMS1 keyword.

    Returns
    -------
    ped : float, 2D array
        pedestal image
    """
    ff_all = firstf_int[num_int, :, :].astype(np.float32)
    ped = ff_all - slope_int[num_int, ::] * \
        (((nframes + 1.) / 2. + dropframes1) / (nframes + groupgap))

    ped[np.bitwise_and(dq_first, dqflags.group['SATURATED']
                       ) == dqflags.group['SATURATED']] = 0
    ped[np.isnan(ped)] = 0.

    return ped


def output_integ(slope_int, dq_int, effintim, var_p3, var_r3, var_both3,
                 int_times):
    """
    For the OLS algorithm, construct the output integration-specific results.
    Any variance values that are a large fraction of the default value
    LARGE_VARIANCE correspond to non-existent segments, so will be set to 0
    here before output.

    Parameters
    ----------
    model : instance of Data Model
       DM object for input

    slope_int : float, 3D array
       Data cube of weighted slopes for each integration

    dq_int : int, 3D array
       Data cube of DQ arrays for each integration

    effintim : float
       Effective integration time per integration

    var_p3 : float, 3D array
        Cube of integration-specific values for the slope variance due to
        Poisson noise only

    var_r3 : float, 3D array
        Cube of integration-specific values for the slope variance due to
        read noise only

    var_both3 : float, 3D array
        Cube of integration-specific values for the slope variance due to
        read noise and Poisson noise

    int_times : bintable, or None
        The INT_TIMES table, if it exists in the input, else None

    Returns
    -------
    integ_info: tuple
        The tuple of computed integration:t fitting arrays.

    """
    # Suppress harmless arithmetic warnings for now
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

    var_p3[var_p3 > 0.4 * LARGE_VARIANCE] = 0.
    var_r3[var_r3 > 0.4 * LARGE_VARIANCE] = 0.
    var_both3[var_both3 > 0.4 * LARGE_VARIANCE] = 0.

    data = slope_int / effintim
    err = np.sqrt(var_both3)
    dq = dq_int
    var_poisson = var_p3
    var_rnoise = var_r3
    int_times = int_times
    integ_info = (data, dq, var_poisson, var_rnoise, int_times, err)

    # Reset the warnings filter to its original state
    warnings.resetwarnings()

    return integ_info


'''
# BEGIN remove GLS
def gls_output_integ(model, slope_int, slope_err_int, dq_int):
    """
    For the GLS algorithm, construct the output integration-specific results.
    Parameters
    ----------
    model : instance of Data Model
        DM object for input
    slope_int : float, 3D array
        Data cube of weighted slopes for each integration
    slope_err_int : float, 3D array
        Data cube of slope errors for each integration
    dq_int : int, 3D array
        Data cube of DQ arrays for each integration
    Returns
    -------
    cubemod : Data Model object
    """
    # Suppress harmless arithmetic warnings for now
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

    cubemod = datamodels.CubeModel()
    cubemod.data = slope_int
    cubemod.err = slope_err_int
    cubemod.dq = dq_int

    # Reset the warnings filter to its original state
    warnings.resetwarnings()

    cubemod.update(model)  # keys from input needed for photom step

    return cubemod


def gls_output_optional(model, intercept_int, intercept_err_int,
                        pedestal_int,
                        ampl_int, ampl_err_int):  # pragma: no cover
    """Construct the optional results for the GLS algorithm.

    Extended Summary
    ----------------
    Construct the GLS-specific optional output data.
    These results are the Y-intercepts, uncertainties in the intercepts,
    pedestal (first group extrapolated back to zero time),
    cosmic ray magnitudes, and uncertainties in the CR magnitudes.

    Parameters
    ----------
    model : instance of Data Model
        Data model object for input; this is used only for the file name.

    intercept_int : 3-D ndarray, float32, shape (n_int, ny, nx)
        Y-intercept for each integration, at each pixel.

    intercept_err_int : 3-D ndarray, float32, shape (n_int, ny, nx)
        Uncertainties for Y-intercept for each integration, at each pixel.

    pedestal_int : 3-D ndarray, float32, shape (n_int, ny, nx)
        The pedestal, for each integration and each pixel.

    ampl_int : 4-D ndarray, float32, shape (n_int, ny, nx, max_num_cr)
        Cosmic-ray amplitudes for each integration, at each pixel, and for
        each CR hit in the ramp.  max_num_cr will be the maximum number of
        CRs within the ramp for any pixel, or it will be one if there were
        no CRs at all.

    ampl_err_int : 4-D ndarray, float32, shape (n_int, ny, nx, max_num_cr)
        Uncertainties for cosmic-ray amplitudes for each integration, at
        each pixel, and for each CR in the ramp.

    Returns
    -------
    gls_ramp_model : GLS_RampFitModel object
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
                 frame_time, nframes_used):  # pragma: no cover
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
    first_group : float32, 2-D array
        A slice of the first group in the ramp.

    slope_int : float32, 2-D array
        The slope obtained by GLS ramp fitting.  This is a slice for the
        current integration and a subset of the image lines.

    s_mask : bool, 2-D array
        True for ramps that were saturated in the first group.

    frame_time : float
        The time to read one frame, in seconds.

    nframes_used : int
        Number of frames that were averaged together to make a group.
        Exludes the groupgap.

    Returns
    -------
    pedestal : float32, 2-D array
        This is a slice of the full pedestal array, and it's for the
        current integration.
    """

    M = float(nframes_used)
    pedestal = first_group - slope_int * frame_time * (M + 1.) / 2.
    if s_mask.any():
        pedestal[s_mask] = 0.

    return pedestal
# END remove GLS
'''


def shift_z(a, off):
    """
    Shift input 3D array by requested offset in z-direction, padding shifted
    array (of the same size) by leading or trailing zeros as needed.

    Parameters
    ----------
    a : float, 3D array
        input array

    off : int
        offset in z-direction

    Returns
    -------
    b : float, 3D array
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
    Calculate the effective integration time for a single group, and return the
    number of frames per group, and the number of frames dropped between groups.

    Parameters
    ----------
    model : instance of Data Model
        DM object for input

    Returns
    -------
    effintim : float
        effective integration time for a single group

    nframes : int
        number of frames averaged per group; from the NFRAMES keyword.

    groupgap : int
        number of frames dropped between groups; from the GROUPGAP keyword.

    dropframes1 : int
        number of frames dropped at the beginning of every integration; from
        the DRPFRMS1 keyword, or 0 if the keyword is missing
    """
    groupgap = model.meta.exposure.groupgap
    nframes = model.meta.exposure.nframes
    frame_time = model.meta.exposure.frame_time
    dropframes1 = model.meta.exposure.drop_frames1

    if (dropframes1 is None):    # set to default if missing
        dropframes1 = 0
        log.debug('Missing keyword DRPFRMS1, so setting to default value of 0')

    try:
        effintim = (nframes + groupgap) * frame_time
    except TypeError:
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
    Extract values for the number of groups, the number of pixels, dataset
    shapes, the number of integrations, the instrument name, the frame time,
    and the observation time.

    Parameters
    ----------
    model : instance of Data Model
       DM object for input

    Returns
    -------
    nreads : int
       number of reads in input dataset

    npix : int
       number of pixels in 2D array

    imshape : (int, int) tuple
       shape of 2D image

    cubeshape : (int, int, int) tuple
       shape of input dataset

    n_int : int
       number of integrations

    instrume : string
       instrument

    frame_time : float
       integration time from TGROUP

    ngroups : int
        number of groups per integration

    group_time : float
        Time increment between groups, in seconds.
    """
    instrume = model.meta.instrument.name
    frame_time = model.meta.exposure.frame_time
    ngroups = model.meta.exposure.ngroups
    group_time = model.meta.exposure.group_time

    n_int = model.data.shape[0]
    nreads = model.data.shape[1]
    asize2 = model.data.shape[2]
    asize1 = model.data.shape[3]

    # If nreads and ngroups are not the same, override the value of ngroups
    #   with nreads, which is more likely to be correct, since it's based on
    #   the image shape.
    if nreads != ngroups:
        log.warning('The value from the key NGROUPS does not (but should) match')
        log.warning('  the value of nreads from the data; will use value of')
        log.warning('  nreads: %s' % (nreads))
        ngroups = nreads

    npix = asize2 * asize1  # number of pixels in 2D array
    imshape = (asize2, asize1)
    cubeshape = (nreads,) + imshape

    return nreads, npix, imshape, cubeshape, n_int, instrume, frame_time, \
        ngroups, group_time


def get_more_info(model):  # pragma: no cover
    """Get information used by GLS algorithm.

    Parameters
    ----------
    model : instance of Data Model
        DM object for input

    Returns
    -------
    group_time : float
        Time increment between groups, in seconds.

    nframes_used : int
        Number of frames that were averaged together to make a group,
        i.e. excluding skipped frames.

    saturated_flag : int
        Group data quality flag that indicates a saturated pixel.

    jump_flag : int
        Group data quality flag that indicates a cosmic ray hit.
    """

    group_time = model.meta.exposure.group_time
    nframes_used = model.meta.exposure.nframes
    saturated_flag = dqflags.group['SATURATED']
    jump_flag = dqflags.group['JUMP_DET']

    return (group_time, nframes_used, saturated_flag, jump_flag)


def get_max_num_cr(gdq_cube, jump_flag):  # pragma: no cover
    """
    Find the maximum number of cosmic-ray hits in any one pixel.

    Parameters
    ----------
    gdq_cube : 3-D ndarray
        The group data quality array.

    jump_flag : int
        The data quality flag indicating a cosmic-ray hit.

    Returns
    -------
    max_num_cr : int
        The maximum number of cosmic-ray hits for any pixel.
    """
    cr_flagged = np.empty(gdq_cube.shape, dtype=np.uint8)
    cr_flagged[:] = np.where(np.bitwise_and(gdq_cube, jump_flag), 1, 0)
    max_num_cr = cr_flagged.sum(axis=0, dtype=np.int32).max()
    del cr_flagged

    return max_num_cr


def reset_bad_gain(pdq, gain):
    """
    For pixels in the gain array that are either non-positive or NaN, reset the
    the corresponding pixels in the pixel DQ array to NO_GAIN_VALUE and
    DO_NOT_USE so that they will be ignored.

    Parameters
    ----------
    pdq : int, 2D array
        pixel dq array of input model

    gain : float32, 2D array
        gain array from reference file

    Returns
    -------
    pdq : int, 2D array
        pixleldq array of input model, reset to NO_GAIN_VALUE and DO_NOT_USE
        for pixels in the gain array that are either non-positive or NaN.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value.*", RuntimeWarning)
        wh_g = np.where(gain <= 0.)
    if len(wh_g[0]) > 0:
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], dqflags.pixel['NO_GAIN_VALUE'])
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], dqflags.pixel['DO_NOT_USE'])

    wh_g = np.where(np.isnan(gain))
    if len(wh_g[0]) > 0:
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], dqflags.pixel['NO_GAIN_VALUE'])
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], dqflags.pixel['DO_NOT_USE'])

    return pdq


def remove_bad_singles(segs_beg_3):
    """
    For the current integration and data section, remove all segments having only
    a single group if there are other segments in the ramp.  This method allows
    for the possibility that a ramp can have multiple (necessarily consecutive)
    1-group segments, which in principle could occur if there are consecutive
    cosmic rays.

    Parameters
    ----------
    segs_beg_3 : integer, 3D array
        lengths of all segments for all ramps in the given data section and
        integration; some of these ramps may contain segments having a single
        group, and another segment.

    Returns
    -------
    segs_beg_3 : integer, 3D array
        lengths of all segments for all ramps in the given data section and
        integration; segments having a single group, and another segment
        will be removed.
    """
    max_seg = segs_beg_3.shape[0]

    # get initial number of ramps having single-group segments
    tot_num_single_grp_ramps = len(np.where((segs_beg_3 == 1) &
                                            (segs_beg_3.sum(axis=0) > 1))[0])

    while(tot_num_single_grp_ramps > 0):
        # until there are no more single-group segments
        for ii_0 in range(max_seg):
            slice_0 = segs_beg_3[ii_0, :, :]

            for ii_1 in range(max_seg):  # correctly includes EARLIER segments
                if (ii_0 == ii_1):  # don't compare with itself
                    continue

                slice_1 = segs_beg_3[ii_1, :, :]

                # Find ramps of a single-group segment and another segment
                # either earlier or later
                wh_y, wh_x = np.where((slice_0 == 1) & (slice_1 > 0))

                if (len(wh_y) == 0):
                    # Are none, so go to next pair of segments to check
                    continue

                # Remove the 1-group segment
                segs_beg_3[ii_0:-1, wh_y, wh_x] = segs_beg_3[ii_0 + 1:, wh_y, wh_x]

                # Zero the last segment entry for the ramp, which would otherwise
                # remain non-zero due to the shift
                segs_beg_3[-1, wh_y, wh_x] = 0

                del wh_y, wh_x

                tot_num_single_grp_ramps = len(np.where((segs_beg_3 == 1) &
                                                        (segs_beg_3.sum(axis=0) > 1))[0])

    return segs_beg_3


def fix_sat_ramps(sat_0th_group_int, var_p3, var_both3, slope_int, dq_int):
    """
    For ramps within an integration that are saturated on the initial group,
    reset the integration-specific variances and slope so they will have no
    contribution.

    Parameters
    ----------
    sat_0th_group_int : uint8, 3D array
        Integration-specific slice whose value for a pixel is 1 if the initial
        group of the ramp is saturated

    var_p3 : float, 3D array
        Cube of integration-specific values for the slope variance due to
        Poisson noise only; some ramps may be saturated in the initial group

    var_both3 : float, 3D array
        Cube of segment- and integration-specific values for the slope
        variance due to combined Poisson noise and read noise; some ramps may
        be saturated in the initial group

    slope_int : float, 3D array
        Cube of integration-specific slopes. Some ramps may be saturated in the
        initial group.

    dq_int : uint32, 3D array
        Cube of integration-specific DQ flags.

    Returns
    -------
    var_p3 : float, 3D array
        Cube of integration-specific values for the slope variance due to
        Poisson noise only; for ramps that are saturated in the initial group,
        this variance has been reset to a huge value to minimize the ramps
        contribution.

    var_both3 : float, 3D array
        Cube of segment- and integration-specific values for the slope
        variance due to combined Poisson noise and read noise; for ramps that
        are saturated in the initial group, this variance has been reset to a
        huge value to minimize the ramps contribution.

    slope_int : float, 3D array
        Cube of integration-specific slopes; for ramps that are saturated in
        the initial group, this variance has been reset to a huge value to
        minimize the ramps contribution.

    dq_int : uint32, 3D array
        Cube of integration-specific DQ flags. For ramps that are saturated in
        the initial group, the flag 'DO_NOT_USE' is added.
    """
    var_p3[sat_0th_group_int > 0] = LARGE_VARIANCE
    var_both3[sat_0th_group_int > 0] = LARGE_VARIANCE
    slope_int[sat_0th_group_int > 0] = 0.
    dq_int[sat_0th_group_int > 0] = np.bitwise_or(
        dq_int[sat_0th_group_int > 0], dqflags.pixel['DO_NOT_USE'])

    return var_p3, var_both3, slope_int, dq_int


def do_all_sat(pixeldq, groupdq, imshape, n_int, save_opt):
    """
    For an input exposure where all groups in all integrations are saturated,
    the DQ in the primary and integration-specific output products are updated,
    and the other arrays in all output products are populated with zeros.

    Parameters
    ----------
    model : instance of Data Model
       DM object for input

    imshape : (int, int) tuple
       shape of 2D image

    n_int : int
       number of integrations

    save_opt : boolean
       save optional fitting results

    Returns
    -------
    image_info: tuple
        The tuple of computed ramp fitting arrays.

    integ_info: tuple
        The tuple of computed integration fitting arrays.

    opt_info: tuple
        The tuple of computed optional results arrays for fitting.
    """
    # Create model for the primary output. Flag all pixels in the pixiel DQ
    #   extension as SATURATED and DO_NOT_USE.
    pixeldq = np.bitwise_or(pixeldq, dqflags.group['SATURATED'])
    pixeldq = np.bitwise_or(pixeldq, dqflags.group['DO_NOT_USE'])

    data = np.zeros(imshape, dtype=np.float32)
    dq = pixeldq
    var_poisson = np.zeros(imshape, dtype=np.float32)
    var_rnoise = np.zeros(imshape, dtype=np.float32)
    err = np.zeros(imshape, dtype=np.float32)
    image_info = (data, dq, var_poisson, var_rnoise, err)

    # Create model for the integration-specific output. The 3D group DQ created
    #   is based on the 4D group DQ of the model, and all pixels in all
    #   integrations will be flagged here as DO_NOT_USE (they are already flagged
    #   as SATURATED). The INT_TIMES extension will be left as None.
    if n_int > 1:
        m_sh = groupdq.shape  # (integ, grps/integ, y, x )
        groupdq_3d = np.zeros((m_sh[0], m_sh[2], m_sh[3]), dtype=np.uint32)

        for ii in range(n_int):  # add SAT flag to existing groupdq in each slice
            groupdq_3d[ii, :, :] = np.bitwise_or.reduce(groupdq[ii, :, :, :],
                                                        axis=0)

        groupdq_3d = np.bitwise_or(groupdq_3d, dqflags.group['DO_NOT_USE'])

        data = np.zeros((n_int,) + imshape, dtype=np.float32)
        dq = groupdq_3d
        var_poisson = np.zeros((n_int,) + imshape, dtype=np.float32)
        var_rnoise = np.zeros((n_int,) + imshape, dtype=np.float32)
        int_times = None
        err = np.zeros((n_int,) + imshape, dtype=np.float32)

        integ_info = (data, dq, var_poisson, var_rnoise, int_times, err)
    else:
        integ_info = None

    # Create model for the optional output
    if save_opt:
        new_arr = np.zeros((n_int,) + (1,) + imshape, dtype=np.float32)

        slope = new_arr
        sigslope = new_arr
        var_poisson = new_arr
        var_rnoise = new_arr
        yint = new_arr
        sigyint = new_arr
        pedestal = np.zeros((n_int,) + imshape, dtype=np.float32)
        weights = new_arr
        crmag = new_arr

        opt_info = (slope, sigslope, var_poisson, var_rnoise,
                    yint, sigyint, pedestal, weights, crmag)

    else:
        opt_info = None

    log.info('All groups of all integrations are saturated.')

    return image_info, integ_info, opt_info


def log_stats(c_rates):
    """
    Optionally log statistics of detected cosmic rays

    Parameters
    ----------
    c_rates : float, 2D array
       weighted count rate

    Returns
    -------
    None
    """
    wh_c_0 = np.where(c_rates == 0.)  # insuff data or no signal

    log.debug('The number of pixels having insufficient data')
    log.debug('due to excessive CRs or saturation %d:', len(wh_c_0[0]))
    log.debug('Count rates - min, mean, max, std: %f, %f, %f, %f'
              % (c_rates.min(), c_rates.mean(), c_rates.max(), c_rates.std()))


def compute_slices(max_cores):
    """
    Computes the number of slices to be created for multiprocessing.

    Parameters
    ----------
    max_cores : string
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all'. This is the fraction of cores to use for multi-proc. The
        total number of cores includes the SMT cores (Hyper Threading for Intel).

    Returns
    -------
    number_slices : int
        The number of slices for multiprocessing.
    """
    if max_cores == 'none':
        number_slices = 1
    else:
        num_cores = multiprocessing.cpu_count()
        log.debug(f'Found {num_cores} possible cores to use for ramp fitting')
        if max_cores == 'quarter':
            number_slices = num_cores // 4 or 1
        elif max_cores == 'half':
            number_slices = num_cores // 2 or 1
        elif max_cores == 'all':
            number_slices = num_cores
        else:
            number_slices = 1
    return number_slices


def dq_compress_final(dq_int, n_int):
    """
    Combine the integration-specific dq arrays (which have already been
    compressed and combined with the PIXELDQ array) to create the dq array
    of the primary output product.

    Parameters
    ----------
    dq_int : uint16, 3D array
        cube of combined dq arrays for all data sections in a single integration

    n_int : int
        total number of integrations in data set

    Returns
    -------
    f_dq : uint16, 2D array
        combination of all integration's pixeldq arrays
    """
    f_dq = dq_int[0, :, :]

    for jj in range(1, n_int):
        f_dq = np.bitwise_or(f_dq, dq_int[jj, :, :])

    return f_dq


def dq_compress_sect(gdq_sect, pixeldq_sect):
    """
    Get ramp locations where the data has been flagged as saturated in the 4D
    GROUPDQ array for the current data section, find the corresponding image
    locations, and set the SATURATED flag in those locations in the PIXELDQ
    array. Similarly, get the ramp locations where the data has been flagged as
    a jump detection in the 4D GROUPDQ array, find the corresponding image
    locations, and set the COSMIC_BEFORE flag in those locations in the PIXELDQ
    array. These modifications to the section of the PIXELDQ array are not used
    to flag groups for any computations; they are used only in the integration-
    specific output.

    Parameters
    ----------
    gdq_sect : int (uint8), 3D array
        cube of GROUPDQ array for a data section

    pixeldq_sect : int, 2D array
        dq array of data section of input model

    Returns
    -------
    pixeldq_sect : int, 2D array
        dq array of data section updated with saturated and jump-detected flags

    """
    sat_loc_r = np.bitwise_and(gdq_sect, SATURATED)
    sat_loc_im = np.where(sat_loc_r.sum(axis=0) > 0)
    pixeldq_sect[sat_loc_im] = np.bitwise_or(pixeldq_sect[sat_loc_im], SATURATED)

    cr_loc_r = np.bitwise_and(gdq_sect, JUMP_DET)
    cr_loc_im = np.where(cr_loc_r.sum(axis=0) > 0)
    pixeldq_sect[cr_loc_im] = np.bitwise_or(pixeldq_sect[cr_loc_im], JUMP_DET)

    return pixeldq_sect
