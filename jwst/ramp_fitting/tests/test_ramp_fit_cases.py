import numpy as np
import numpy.testing as npt

from jwst.ramp_fitting.ramp_fit import ramp_fit
# from stcal.ramp_fitting.ramp_fit import ramp_fit

from jwst.datamodels import RampModel
from jwst.datamodels import GainModel, ReadnoiseModel

#
# The first 12 tests are for a single ramp in a single integration. The ramps
#  have a variety of GROUPDQ vectors, with 1 or more segments in each ramp.  The
#  comparison of the PRIMARY output results are partly to verify the slopes and
#  variances of the combination of the segments in a ramp within the single
#  integration.  The comparison of the OPTIONAL output results are to verify the
#  results for each of the individual segments in a ramp.  Within each test is a
#  description of classification ('CASE') within the code of all of the segments
#  for the pixel based on the ramp's GROUPDQ, and the resulting segments and
#  their SCI values (these are mostly for my reference).
#


def test_pix_0():
    """
    CASE A: segment has >2 groups, at end of ramp.
    SCI seg is [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.](A)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    # [data, dq, err, var_p, var_r]
    p_true = [1.0117551, 0, 0.0921951, 0.0020202, 0.00647973]

    # Set truth values for OPTIONAL results:
    # [slope, sigslope, var_poisson, var_rnoise, yint, sigyint, ped, weights]
    o_true = [1.0117551, 4.874572, 0.0020202, 0.00647973,
              15.911023, 27.789335, 4.882449, 13841.038]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_1():
    """
    CASE H: the segment has a good 1st group and a bad 2nd group, so is a
      single group. If there are no later and longer segments in the ramp,
      this group's data will be used in the 'fit'. If there are later and
      longer segments, this group's data will not be used.
    CASE F: segment has 2 good groups not at array end.
    SCI segs are: seg0[15] (H, ignored), seg1[35, 54] (F)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 4, 4, 0, 2, 2, 2, 2, 2, 2])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.8999999, 6, 1.046670, 0.02636364, 1.0691562]

    # Set truth values for OPTIONAL results:
    o_true = [1.9, 56.870003, 0.02636364, 1.0691562, -3., 56.870003,
              -3.999998, 0.83321977]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_2():
    """
    CASE B: segment has >2 groups, not at end of ramp.
    CASE F: (twice) segment has 2 good groups not at array end.
    SCI segs are: seg0[15,25,35](B), seg1[54,55](F), seg2[65,75](F)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 0, 0, 4, 0, 4, 0, 4, 2, 2])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [0.84761256, 6, 0.42986465, 0.00659091, 0.1781927]

    # Set truth values for OPTIONAL results for all segments
    o_true = [[1.0000001, 0.1, 1.],                 # slopes for 3 segments
              [28.435, 56.870003, 56.870003],        # sigslope
              [0.01318182, 0.02636364, 0.02636364, ],  # var_poisson
              [0.26728904, 1.0691562, 1.0691562],   # var_rnoise
              [14.999998, 51., 15.],                # yint
              [36.709427, 56.870003, 56.870003],     # sigyint
              [6.5238733],                          # pedestal
              [12.712313, 0.83321977, 0.83321977],   # weights
              ]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_3():
    """
    CASE B: segment has >2 groups, not at end of ramp.
    CASE E: segment has 2 good groups, at end of ramp.
    SCI segs are: seg0[15,25,35,54,55,65,75,94](B), seg1[95,105](E)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 4, 0])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.0746869, 4, 0.12186482, 0.00227273, 0.01257831]

    # Set truth values for OPTIONAL results:
    o_true = [[1.0757396, 1.],
              [6.450687, 56.870003],
              [0.0025974, 0.01818182],
              [0.01272805, 1.0691562],
              [14.504965, 15.],
              [27.842508, 56.870003],
              [4.253134],
              [4.2576841e+03, 8.458062e-01],
              ]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_4():
    """
    CASE G: segment is the good 1st group of the entire ramp, and no later
      groups are good.
    SCI seg is seg0[15](G)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 1055., 1065., 1075., 2594., 2595., 2605.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 2, 2, 2, 2, 2, 2, 2, 2, 2])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.5, 2, 1.047105, 0.02727273, 1.0691562]

    # Set truth values for OPTIONAL results:
    o_true = [1.5, 0., 0.02727273, 1.0691562, 0., 0., 0., 0.8318386]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_5():
    """
    CASE B: segment has >2 groups, not at end of ramp.
    CASE A: segment has >2 groups, at end of ramp.
    SCI segs are: seg0[15, 25, 35, 54](B), seg1[ 2055, 2065, 2075, 2094, 2095,
       2105](A)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 2055., 2065., 2075., 2094., 2095., 2105.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 0, 0, 0, 4, 0, 0, 0, 0, 0])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.076075, 4, 0.16134359, 0.00227273, 0.02375903]

    # Set truth values for OPTIONAL results:
    o_true = [[1.2799551, 1.0144024],
              [18.312422, 9.920552],
              [0.00606061, 0.00363636],
              [0.10691562, 0.03054732],
              [13.537246, 2015.0737],
              [35.301933, 67.10882],
              [4.2391253],
              [78.34764, 855.78046]
              ]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_6():
    """
    CASE F: segment has 2 good groups not at array end
    CASE A: segment has >2 groups, at end of ramp.
    SCI segs are: seg0[15,25](F), seg1[54, 55, 65, 375, 394, 395, 405](A)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 375., 394., 395., 405.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 0, 4, 4, 0, 0, 0, 0, 0, 0])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [6.092052, 4, 0.14613187, 0.0025974, 0.01875712]

    # Set truth values for OPTIONAL results:
    o_true = [[1., 6.195652],
              [56.870003, 8.8936615],
              [0.01818182, 0.0030303],
              [1.0691562, 0.01909207],
              [15., -143.2391],
              [56.870003, 58.76999],
              [-45.92052],
              [8.4580624e-01, 2.0433204e+03]
              ]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_7():
    """
    CASE B: segment has >2 groups, not at end of ramp.
    SCI seg is seg0[15,25,35,54,55,65,75,94](B)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 195., 205.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 4, 4])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.0757396, 4, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.0757396, 6.450687, 0.0025974, 0.01272805, 14.504951,
              27.842508, 4.2426033, 4257.684]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_8():
    """
    CASE H: the segment has a good 1st group and a bad 2nd group.
    CASE B: segment has >2 groups, not at end of ramp.
    SCI segs are: seg0[15](H, ignored), seg1[25, 35, 54, 55, 65, 75](B)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 4, 0, 0, 0, 0, 0, 2, 2, 2])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.0101178, 6, 0.1848883, 0.00363636, 0.03054732]

    # Set truth values for OPTIONAL results:
    o_true = [1.0101178, 12.385354, 0.00363636, 0.03054732, 16.508228,
              40.81897, 4.898822, 855.78046]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_9():
    """
    CASE F: segment has 2 good groups not at array end.
    CASE B: segment has >2 groups, not at end of ramp.
    CASE E: segment has 2 good groups, at end of ramp.
    SCI seg are: seg0[15,25](F), seg1[54, 55, 65, 75, 94](B), seg2[95, 105](E)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 0, 4, 4, 0, 0, 0, 0, 4, 0])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [0.9999994, 4, 0.22721863, 0.0030303, 0.048598]

    # Set truth values for OPTIONAL results:
    o_true = [[1., 0.9999994, 1.],
              [56.870003, 13.036095, 56.870003],
              [0.01818182, 0.00454545, 0.01818182],
              [1.0691562, 0.05345781, 1.0691562],
              [15., 20.119896, 15.],
              [56.870003, 68.618195, 56.870003],
              [5.000005],
              [0.84580624, 297.23172, 0.84580624]
              ]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_10():
    """
    CASE F: segment has 2 good groups not at array end.
    CASE B: segment has >2 groups, not at end of ramp.
    CASE A: segment has >2 groups, at end of ramp.
    SCI segs are: seg0[15,25](F), seg1[35,54,55](B), seg2[65,75,94,95,105](A)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 0, 4, 0, 0, 4, 0, 0, 0, 0])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1., 4, 0.21298744, 0.0025974, 0.04276625]

    # Set truth values for OPTIONAL results:
    o_true = [[1., 1.0000014, 0.99999964],
              [56.870003, 28.434996, 13.036095],
              [0.01818182, 0.00909091, 0.00454545],
              [1.0691562, 0.26728904, 0.05345781],
              [15., 17.999956, 15.000029],
              [56.870003, 88.40799, 93.73906],
              [5.],
              [0.84580624, 13.091425, 297.23172]
              ]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_11():
    """
    CASE F: segment has 2 good groups not at array end.
    SCI seg is: seg0[15,25](F)
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [15., 25., 35., 54., 55., 65., 75., 94., 95., 105.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 0, 2, 2, 2, 2, 2, 2, 2, 2])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1., 2, 1.042755, 0.01818182, 1.0691562]

    # Set truth values for OPTIONAL results:
    o_true = [1., 56.870003, 0.01818182, 1.0691562, 15., 56.870003, 5.,
              0.84580624]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_pix_12():
    """
    CASE NGROUPS=2: the segment has a good 1st group and a saturated 2nd group,
      so is a single group. Group 1's data will be used in the 'fit'.
    """

    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ngroups = 2
    nints = 1
    ncols = 2
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array([15., 59025.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([0, 2])
    ramp_model.data[0, :, 0, 1] = np.array([61000., 61000.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 1] = np.array([2, 2])

    # call ramp_fit
    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results for pixel 1:
    # slope, dq, err, var_p, var_r
    # slope = group1 / deltatime = 15 / 10 = 1.5
    # dq = 2 (saturation) because group2 is saturated, but DO_NOT_USE is *not* set
    p_true = [1.5, 2, 1.047105, 0.027273, 1.069156]

    # Set truth values for OPTIONAL results:
    # slope, sig_slope, var_p, var_r, yint, sig_yint, pedestal, weights
    # slope = group1 / deltatime = 15 / 10 = 1.5
    # sig_slope, yint, sig_yint, and pedestal are all zero, because only 1 good group
    o_true = [1.5, 0., 0.027273, 1.069156, 0., 0., 0., 0.831839]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)

    # Set truth values for PRIMARY results for pixel 2:
    # slope, dq, err, var_p, var_r
    # slope = zero, because no good data
    # dq = 3 (saturation + do_not_use) because both groups are saturated
    p_true = [0., 3, 0., 0., 0.]

    # Set truth values for OPTIONAL results:
    # slope, sig_slope, var_p, var_r, yint, sig_yint, pedestal, weights
    # all values zero, because no good data
    o_true = [0., 0., 0., 0., 0., 0., 0., 0.]

    assert_pri(p_true, new_mod, 1)
    assert_opt(o_true, opt_mod, 1)


# -------------- start of MIRI tests: all have only a single segment-----
def test_miri_0():
    """
    MIRI data with ramp's 0th and final groups are flagged as DO_NOT_USE
    SCI seg is: [8888., 25., 35., 54., 55., 65., 75., 94., 95., 888.]
    GROUPDQ is: [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [8888., 25., 35., 54., 55., 65., 75., 94., 95., 888.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.025854, 0, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.025854, 6.450687, 0.0025974, 0.01272805, 26.439266, 27.842508,
              14.74146, 4257.684]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_miri_1():
    """
    MIRI data with ramp's 0th and final groups flagged as DO_NOT_USE; 0th group
    is also as a cosmic ray
    SCI seg is: [7777., 125., 135., 154., 165., 175., 185., 204., 205., 777.]
    GROUPDQ is: [5, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    """
    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [7777., 125., 135., 154., 165., 175., 185., 204., 205., 777.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.1996487, 0, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.1996487, 6.450687, 0.0025974, 0.01272805, 126.110214,
              27.842508, 113.00351, 4257.684]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_miri_2():
    """
    MIRI data with ramp's 0th and final groups flagged as both DO_NOT_USE
    and as CR.
    SCI seg is: [4444., 25., 35., 54., 55., 65., 75., 94., 95., 444.]
    GROUPDQ is: [5, 0, 0, 0, 0, 0, 0, 0, 0, 5]
    """
    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [4444., 25., 35., 54., 55., 65., 75., 94., 95., 444.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 5])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.025854, 0, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.025854, 6.450687, 0.0025974, 0.01272805, 26.439266, 27.842508,
              14.74146, 4257.684]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def test_miri_3():
    """
    MIRI data with ramp's 0th and final groups flagged as DO_NOT_USE, and final
    group also flagged as CR.
    SCI seg is: [6666., 25., 35., 54., 55., 65., 75., 94., 95., 666.]
    GROUPDQ is: [1, 0, 0, 0, 0, 0, 0, 0, 0, 5]
    """
    ngroups, nints, nrows, ncols, deltatime, gain, readnoise = set_scalars()
    ramp_model, rnoise_model, gain_model, pixdq, groupdq, err = create_mod_arrays(
        ngroups, nints, nrows, ncols, deltatime, gain, readnoise)

    # Populate pixel-specific SCI and GROUPDQ arrays
    ramp_model.data[0, :, 0, 0] = np.array(
        [6666., 25., 35., 54., 55., 65., 75., 94., 95., 666.], dtype=np.float32)
    ramp_model.groupdq[0, :, 0, 0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 5])

    new_mod, int_mod, opt_mod, gls_opt_mod = ramp_fit(
        ramp_model, 1024 * 300., True, rnoise_model, gain_model, 'OLS', 'optimal', 'none')

    # Set truth values for PRIMARY results:
    p_true = [1.025854, 0, 0.12379601, 0.0025974, 0.01272805]

    # Set truth values for OPTIONAL results:
    o_true = [1.025854, 6.450687, 0.0025974, 0.01272805, 26.439266,
              27.842508, 14.74146, 4257.684]

    assert_pri(p_true, new_mod, 0)
    assert_opt(o_true, opt_mod, 0)


def assert_pri(p_true, new_info, pix):
    """
    Compare true and fit values of primary output for extensions
    SCI, DQ, ERR, VAR_POISSSON, VAR_RNOISE.
    """

    # image_info = (data, dq, var_poisson, var_rnoise, err)
    data, dq, var_poisson, var_rnoise, err = new_info

    npt.assert_allclose(data[0, pix], p_true[0], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(dq[0, pix], p_true[1], atol=1E-1)
    npt.assert_allclose(err[0, pix], p_true[2], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(var_poisson[0, pix], p_true[3], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(var_rnoise[0, pix], p_true[4], atol=2E-5, rtol=2e-5)

    return None


def assert_opt(o_true, opt_info, pix):
    """
    Compare true and fit values of optional output for extensions SLOPE,
    SIGSLOPE, VAR_POISSSON, VAR_RNOISE, YINT, SIGYINT, PEDESTAL, and WEIGHTS.
    Selecting the particular (and only) ramp in the optional output, which is
    [0,:,0,0]
    """
    (slope, sigslope, var_poisson, var_rnoise,
        yint, sigyint, pedestal, weights, crmag) = opt_info

    opt_slope = slope[0, :, 0, pix]
    opt_sigslope = sigslope[0, :, 0, pix]
    opt_var_poisson = var_poisson[0, :, 0, pix]
    opt_var_rnoise = var_rnoise[0, :, 0, pix]
    opt_yint = yint[0, :, 0, pix]
    opt_sigyint = sigyint[0, :, 0, pix]
    opt_pedestal = pedestal[:, 0, pix]
    opt_weights = weights[0, :, 0, pix]

    npt.assert_allclose(opt_slope, o_true[0], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_sigslope, o_true[1], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_var_poisson, o_true[2], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_var_rnoise, o_true[3], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_yint, o_true[4], atol=2E-2)
    npt.assert_allclose(opt_sigyint, o_true[5], atol=2E-5, rtol=2e-5)
    npt.assert_allclose(opt_pedestal, o_true[6], atol=2E-5, rtol=3e-5)
    npt.assert_allclose(opt_weights, o_true[7], atol=2E-5, rtol=2e-5)

    return None


def set_scalars():
    """
    Set needed scalars for the size of the dataset, and other values needed for
    the fit.
    """

    ngroups = 10
    nints = 1
    nrows = 1
    ncols = 1
    deltatime = 10.0
    gain = 5.5
    readnoise = 10.34

    return ngroups, nints, nrows, ncols, deltatime, gain, readnoise


def create_mod_arrays(ngroups, nints, nrows, ncols, deltatime, gain, readnoise):
    """
    For an input datacube (arbitrarily chosen to be MIRI), create arrays having
    the specified dimensions for the pixel DQ, the group DQ, and the
    ERR extensions, and create datamodels for the ramp, readnoise, and gain.
    """

    gain = np.ones(shape=(nrows, ncols), dtype=np.float32) * gain
    err = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    read_noise = np.full((nrows, ncols), readnoise, dtype=np.float32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)

    # Create and populate ramp model
    ramp_model = RampModel(data=data, err=err, pixeldq=pixdq, groupdq=gdq)
    ramp_model.meta.instrument.name = 'MIRI'
    ramp_model.meta.instrument.detector = 'MIRIMAGE'
    ramp_model.meta.instrument.filter = 'F480M'
    ramp_model.meta.observation.date = '2015-10-13'
    ramp_model.meta.exposure.type = 'MIR_IMAGE'
    ramp_model.meta.exposure.group_time = deltatime

    ramp_model.meta.subarray.name = 'FULL'
    ramp_model.meta.subarray.xstart = 1
    ramp_model.meta.subarray.ystart = 1
    ramp_model.meta.subarray.xsize = ncols
    ramp_model.meta.subarray.ysize = nrows

    ramp_model.meta.exposure.frame_time = deltatime
    ramp_model.meta.exposure.ngroups = ngroups
    ramp_model.meta.exposure.group_time = deltatime
    ramp_model.meta.exposure.nframes = 1
    ramp_model.meta.exposure.groupgap = 0
    ramp_model.meta.exposure.drop_frames1 = 0

    return ramp_model, read_noise, gain, pixdq, gdq, err

    # Create and populate gain model
    gain_model = GainModel(data=gain)
    gain_model.meta.instrument.name = 'MIRI'
    gain_model.meta.subarray.xstart = 1
    gain_model.meta.subarray.ystart = 1
    gain_model.meta.subarray.xsize = ncols
    gain_model.meta.subarray.ysize = nrows

    # Create and populate readnoise model
    rnoise_model = ReadnoiseModel(data=read_noise)
    rnoise_model.meta.instrument.name = 'MIRI'
    rnoise_model.meta.subarray.xstart = 1
    rnoise_model.meta.subarray.ystart = 1
    rnoise_model.meta.subarray.xsize = ncols
    rnoise_model.meta.subarray.ysize = nrows

    return ramp_model, rnoise_model, gain_model, pixdq, gdq, err
