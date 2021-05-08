import pytest
import numpy as np

from jwst.ramp_fitting.ramp_fit import ramp_fit
from jwst.ramp_fitting.ols_fit import calc_num_seg
# from stcal.ramp_fitting.ramp_fit import ramp_fit
# from stcal.ramp_fitting.ols_fit import calc_num_seg

from jwst.datamodels import dqflags
from jwst.datamodels import RampModel

DO_NOT_USE = dqflags.group['DO_NOT_USE']
JUMP_DET = dqflags.group['JUMP_DET']
SATURATED = dqflags.group['SATURATED']

DELIM = "-" * 70

# single group intergrations fail in the GLS fitting
# so, keep the two method test separate and mark GLS test as
# expected to fail.  Needs fixing, but the fix is not clear
# to me. [KDG - 19 Dec 2018]


def test_one_group_small_buffer_fit_ols():
    model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=1, gain=1, readnoise=10)
    model1.data[0, 0, 50, 50] = 10.0

    slopes, cube, optional, gls_dummy = ramp_fit(
        model1, 512, True, rnoise, gain, 'OLS', 'optimal', 'none')

    data = slopes[0]
    np.testing.assert_allclose(data[50, 50], 10.0, 1e-6)


def test_drop_frames1_not_set():
    model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=1, gain=1, readnoise=10)
    model1.data[0, 0, 50, 50] = 10.0
    model1.meta.exposure.drop_frames1 = None

    slopes, cube, optional, gls_dummy = ramp_fit(
        model1, 512, True, rnoise, gain, 'OLS', 'optimal', 'none')

    data = slopes[0]
    np.testing.assert_allclose(data[50, 50], 10.0, 1e-6)


def test_mixed_crs_and_donotuse():

    gdq = np.zeros((3, 10, 3, 3), dtype=np.uint32)

    # pix with only first and last group flagged DO_NOT_USE;
    # results in 1 segment (flags at ends of ramp do not break the ramp)
    gdq[0, 0, 0, 0] = DO_NOT_USE
    gdq[0, -1, 0, 0] = DO_NOT_USE

    # pix with first and last group flagged DO_NOT_USE and 1 CR in middle
    # results in 2 segments
    gdq[0, 0, 1, 1] = DO_NOT_USE
    gdq[0, -1, 1, 1] = DO_NOT_USE
    gdq[0, 3, 1, 1] = JUMP_DET

    # max segments should be 2
    max_seg, max_cr = calc_num_seg(gdq, 3)
    assert(max_seg == 2)

    # pix with only 1 middle group flagged DO_NOT_USE;
    # results in 2 segments
    gdq[1, 2, 0, 0] = DO_NOT_USE

    # pix with middle group flagged as CR and DO_NOT_USE;
    # results in 2 segments
    gdq[2, 2, 0, 0] = DO_NOT_USE + JUMP_DET

    # pix with DO_NOT_USE and CR in different middle groups;
    # results in 3 segments
    gdq[2, 2, 1, 1] = DO_NOT_USE
    gdq[2, 4, 1, 1] = JUMP_DET

    # max segments should now be 3
    max_seg, max_cr = calc_num_seg(gdq, 3)
    assert(max_seg == 3)


@pytest.mark.skip(reason="GLS code does not [yet] handle single group integrations.")
def test_one_group_small_buffer_fit_gls():
    model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=1, gain=1, readnoise=10)
    model1.data[0, 0, 50, 50] = 10.0

    slopes, cube, optional, gls_dummy = ramp_fit(
        model1, 512, True, rnoise, gain, 'GLS', 'optimal', 'none')

    data = slopes[0]
    np.testing.assert_allclose(data[50, 50], 10.0, 1e-6)


def test_one_group_two_ints_fit_ols():
    model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=1, gain=1, readnoise=10, nints=2)
    model1.data[0, 0, 50, 50] = 10.0
    model1.data[1, 0, 50, 50] = 12.0

    slopes, cube, optional, gls_dummy = ramp_fit(
        model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

    data = slopes[0]
    np.testing.assert_allclose(data[50, 50], 11.0, 1e-6)


@pytest.mark.skip(reason="GLS does not correctly combine the slopes for integrations into the exposure slope.")
def test_gls_vs_ols_two_ints_ols():
    """
    A test to see if GLS is correctly combining integrations. The combination should only use the read noise variance.
    The current version of GLS does not work correctly.
    """
    model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=11, gain=5, readnoise=1, nints=2)
    ramp = np.asarray([x * 100 for x in range(11)])
    model1.data[0, :, 50, 50] = ramp
    model1.data[1, :, 50, 50] = ramp * 2

    slopes = ramp_fit(model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')
    np.testing.assert_allclose(slopes[0].data[50, 50], 150.0, 1e-6)

    slopes_gls = ramp_fit(model1, 1024 * 30000., True, rnoise, gain, 'GLS', 'optimal', 'none')
    np.testing.assert_allclose(slopes_gls[0].data[50, 50], 150.0, 1e-6)


@pytest.mark.skip(reason="Jenkins environment does not correctly handle multi-processing.")
def test_multiprocessing():
    nints, ngroups, nrows = 3, 25, 100
    ncols = nrows  # make sure these are the same, so the loops below work

    model1, gdq, rnoise, pixdq, err, gain = setup_inputs(
        ngroups=ngroups, gain=1, readnoise=10, nints=nints, nrows=nrows, ncols=ncols)

    delta_plane1 = np.zeros((nrows, ncols), dtype=np.float64)
    delta_plane2 = np.zeros((nrows, ncols), dtype=np.float64)
    delta_vec = np.asarray([x / 50.0 for x in range(nrows)])
    for i in range(ncols):
        delta_plane1[i, :] = delta_vec * i
        delta_plane2[:, i] = delta_vec * i

    model1.data[:, :, :, :] = 0
    for j in range(ngroups - 1):
        model1.data[0, j + 1, :, :] = model1.data[0, j, :, :] + delta_plane1 + delta_plane2
        model1.data[1, j + 1, :, :] = model1.data[1, j, :, :] + delta_plane1 + delta_plane2
        model1.data[2, j + 1, :, :] = model1.data[2, j, :, :] + delta_plane1 + delta_plane2
    model1.data = np.round(model1.data + np.random.normal(0, 5, (nints, ngroups, ncols, nrows)))

    # TODO change this to be parametrized once GLS gets working.
    algo = "OLS"
    # algo = "GLS"
    slopes, int_model, opt_model, gls_opt_model = ramp_fit(
        model1, 1024 * 30000., False, rnoise, gain, algo, 'optimal', 'none')

    slopes_multi, int_model_multi, opt_model_multi, gls_opt_model_multi = ramp_fit(
        model1, 1024 * 30000., False, rnoise, gain, algo, 'optimal', 'half')

    np.testing.assert_allclose(slopes.data, slopes_multi.data, rtol=1e-5)


@pytest.mark.skip(reason="Jenkins environment does not correctly handle multi-processing.")
def test_multiprocessing2():
    nints, ngroups, nrows = 1, 25, 100
    ncols = nrows  # make sure these are the same, so the loops below work
    model1, gdq, rnoise, pixdq, err, gain = setup_inputs(
        ngroups=ngroups, gain=1, readnoise=10, nints=nints, nrows=nrows, ncols=ncols)

    delta_plane1 = np.zeros((nrows, ncols), dtype=np.float64)
    delta_plane2 = np.zeros((nrows, ncols), dtype=np.float64)
    delta_vec = np.asarray([x / 50.0 for x in range(nrows)])
    for i in range(ncols):
        delta_plane1[i, :] = delta_vec * i
        delta_plane2[:, i] = delta_vec * i

    model1.data[:, :, :, :] = 0
    for j in range(ngroups - 1):
        model1.data[0, j + 1, :, :] = model1.data[0, j, :, :] + delta_plane1 + delta_plane2
    model1.data = np.round(model1.data + np.random.normal(0, 5, (nints, ngroups, ncols, nrows)))

    # TODO change this to be parametrized once GLS gets working.
    algo = "OLS"
    # algo = "GLS"
    slopes, int_model, opt_model, gls_opt_model = ramp_fit(
        model1, 1024 * 30000., True, rnoise, gain, algo, 'optimal', 'none')

    slopes_multi, int_model_multi, opt_model_multi, gls_opt_model_multi = ramp_fit(
        model1, 1024 * 30000., True, rnoise, gain, algo, 'optimal', 'half')

    np.testing.assert_allclose(slopes.data, slopes_multi.data, rtol=1e-5)


@pytest.mark.xfail(reason="GLS code does not [yet] handle single group integrations.")
def test_one_group_two_ints_fit_gls():
    model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=1, gain=1, readnoise=10, nints=2)
    model1.data[0, 0, 50, 50] = 10.0
    model1.data[1, 0, 50, 50] = 12.0

    slopes = ramp_fit(model1, 1024 * 30000., True, rnoise, gain, 'GLS', 'optimal', 'none')

    np.testing.assert_allclose(slopes[0].data[50, 50], 11.0, 1e-6)

# tests that apply to both 'ols' and 'gls' are in the TestMethods class so
# that both can use the parameterized 'method'


# @pytest.mark.parametrize("method", ['OLS', 'GLS'])  # don't do GLS to see if it causes hang
@pytest.mark.parametrize("method", ['OLS'])  # don't do GLS to see if it causes hang
class TestMethods:

    def test_nocrs_noflux(self, method):
        # all pixel values are zero. So slope should be zero
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=5)

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 60000, False, rnoise, gain, method, 'optimal', 'none')

        data = slopes[0]
        assert(0 == np.max(data))
        assert(0 == np.min(data))

    def test_nocrs_noflux_firstrows_are_nan(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=5)
        model1.data[0, :, 0:12, :] = np.nan

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 60000, False, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        assert(0 == np.max(data))
        assert(0 == np.min(data))

    @pytest.mark.xfail(reason="Fails, without frame_time it doesn't work")
    def test_error_when_frame_time_not_set(self, method):
        # all pixel values are zero. So slope should be zero
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=5)
        model1.meta.exposure.frame_time = None

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 64000, False, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes.data
        assert(0 == np.max(data))
        assert(0 == np.min(data))

    def test_five_groups_two_ints_Poisson_noise_only(self, method):
        grouptime = 3.0
        ingain = 2000
        inreadnoise = 7
        ngroups = 5
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(
            ngroups=ngroups, gain=ingain, readnoise=inreadnoise, deltatime=grouptime, nints=2)

        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 33.0
        model1.data[0, 4, 50, 50] = 60.0
        model1.data[1, 0, 50, 50] = 10.0
        model1.data[1, 1, 50, 50] = 15.0
        model1.data[1, 2, 50, 50] = 25.0
        model1.data[1, 3, 50, 50] = 33.0
        model1.data[1, 4, 50, 50] = 160.0

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        out_slope = slopes[0][50, 50]
        deltaDN1 = 50
        deltaDN2 = 150
        np.testing.assert_allclose(out_slope, (deltaDN1 + deltaDN2) / 2.0, 75.0, 1e-6)

    def test_ngroups_doesnot_match_cube_size(self, method):
        # all pixel values are zero. So slope should be zero
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=5)
        model1.meta.exposure.ngroups = 11

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 64000, False, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        assert(0 == np.max(data))
        assert(0 == np.min(data))

    def test_bad_gain_values(self, method):
        # all pixel values are zero. So slope should be zero
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=5)
        model1.meta.exposure.ngroups = 11
        gain.data[10, 10] = -10
        gain.data[20, 20] = np.nan

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 64000, False, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        dq = slopes[1]
        assert(0 == np.max(data))
        assert(0 == np.min(data))
        assert dq[10, 10] == 524288 + 1
        assert dq[20, 20] == 524288 + 1

    def test_simple_ramp(self, method):
        # Here given a 10 group ramp with an exact slope of 20/group. The output slope should be 20.
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=10, deltatime=3)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 30.0
        model1.data[0, 2, 50, 50] = 50.0
        model1.data[0, 3, 50, 50] = 70.0
        model1.data[0, 4, 50, 50] = 90.0
        model1.data[0, 5, 50, 50] = 110.0
        model1.data[0, 6, 50, 50] = 130.0
        model1.data[0, 7, 50, 50] = 150.0
        model1.data[0, 8, 50, 50] = 170.0
        model1.data[0, 9, 50, 50] = 190.0

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 64000, True, rnoise, gain, 'OLS', 'optimal', 'none')

        # take the ratio of the slopes to get the relative error
        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], (20.0 / 3), 1e-6)

    def test_read_noise_only_fit(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=5, readnoise=50)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 33.0
        model1.data[0, 4, 50, 50] = 60.0

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        xvalues = np.arange(5) * 1.0
        yvalues = np.array([10, 15, 25, 33, 60])
        coeff = np.polyfit(xvalues, yvalues, 1)

        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], coeff[0], 1e-6)

    def test_photon_noise_only_fit(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=5, gain=1000, readnoise=1)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 33.0
        model1.data[0, 4, 50, 50] = 60.0
        cds_slope = (model1.data[0, 4, 50, 50] - model1.data[0, 0, 50, 50]) / 4.0

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], cds_slope, 1e-2)

    def test_photon_noise_only_bad_last_frame(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=5, gain=1000, readnoise=1)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 33.0
        model1.data[0, 4, 50, 50] = 60.0
        model1.groupdq[0, 4, :, :] = DO_NOT_USE
        cds_slope = (model1.data[0, 3, 50, 50] - model1.data[0, 0, 50, 50]) / 3.0

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], cds_slope, 1e-2)

    @pytest.mark.xfail(reason="Fails, bad last frame yields only one good one. \
        This should not every happen. When ngroups==2 the last frame doesn't get flagged.")
    def test_photon_noise_only_bad_last_frame_two_groups(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=2, gain=1000, readnoise=1)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.groupdq[0, 1, :, :] = DO_NOT_USE
        cds_slope = (model1.data[0, 1, 50, 50] - model1.data[0, 0, 50, 50]) / 1.0

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes.data
        np.testing.assert_allclose(data[50, 50], cds_slope, 1e-6)

    def test_photon_noise_with_unweighted_fit(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=5, gain=1000, readnoise=1)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 33.0
        model1.data[0, 4, 50, 50] = 60.0

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'unweighted', 'none')

        # cds_slope = (model1.data[0,4,500,500] - model1.data[0,0,500,500])/ 4.0
        xvalues = np.arange(5) * 1.0
        yvalues = np.array([10, 15, 25, 33, 60])
        coeff = np.polyfit(xvalues, yvalues, 1)

        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], coeff[0], 1e-6)

    def test_two_groups_fit(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=2, gain=1, readnoise=10)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 0, 50, 51] = 20.0
        model1.data[0, 1, 50, 51] = 60.0
        model1.data[0, 0, 50, 52] = 200.0
        model1.data[0, 1, 50, 52] = 600.0
        model1.meta.exposure.drop_frames1 = 0
        # 2nd group is saturated
        model1.groupdq[0, 1, 50, 51] = SATURATED
        # 1st group is saturated
        model1.groupdq[0, 0, 50, 52] = SATURATED
        model1.groupdq[0, 1, 50, 52] = SATURATED  # should not be set this way
        cds_slope = (model1.data[0, 1, 50, 50] - model1.data[0, 0, 50, 50])

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], cds_slope, 1e-6)

        # expect SATURATED
        dq = slopes[1]
        assert dq[50, 51] == SATURATED
        # expect SATURATED and DO_NOT_USE, because 1st group is Saturated
        assert dq[50, 52] == SATURATED + DO_NOT_USE

    def test_four_groups_oneCR_orphangroupatend_fit(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=4, gain=1, readnoise=10)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 20.0
        model1.data[0, 3, 50, 50] = 145.0
        model1.groupdq[0, 3, 50, 50] = JUMP_DET
        cds_slope = (model1.data[0, 1, 50, 50] - model1.data[0, 0, 50, 50])

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], cds_slope, 1e-6)

    # @pytest.mark.skip(reason="not using now")
    def test_four_groups_two_CRs_at_end(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=4, gain=1, readnoise=10)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 145.0
        model1.groupdq[0, 2, 50, 50] = JUMP_DET
        model1.groupdq[0, 3, 50, 50] = JUMP_DET
        cds_slope = (model1.data[0, 1, 50, 50] - model1.data[0, 0, 50, 50])

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], cds_slope, 1e-6)

    def test_four_groups_four_CRs(self, method):
        #
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=4, gain=1, readnoise=10)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 145.0
        model1.groupdq[0, 0, 50, 50] = JUMP_DET
        model1.groupdq[0, 1, 50, 50] = JUMP_DET
        model1.groupdq[0, 2, 50, 50] = JUMP_DET
        model1.groupdq[0, 3, 50, 50] = JUMP_DET

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], 0, 1e-6)

    def test_four_groups_three_CRs_at_end(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=4, gain=1, readnoise=10)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 145.0
        model1.groupdq[0, 1, 50, 50] = JUMP_DET
        model1.groupdq[0, 2, 50, 50] = JUMP_DET
        model1.groupdq[0, 3, 50, 50] = JUMP_DET

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        expected_slope = 10.0
        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], expected_slope, 1e-6)

    def test_four_groups_CR_causes_orphan_1st_group(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=4, gain=.01, readnoise=10000)
        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 125.0
        model1.data[0, 2, 50, 50] = 145.0
        model1.data[0, 3, 50, 50] = 165.0
        model1.groupdq[0, 1, 50, 50] = JUMP_DET

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        expected_slope = 20.0
        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], expected_slope, 1e-6)

    def test_one_group_fit(self, method):
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(ngroups=1, gain=1, readnoise=10)
        model1.data[0, 0, 50, 50] = 10.0

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], 10.0, 1e-6)

    def test_two_groups_unc(self, method):
        grouptime = 3.0
        deltaDN = 5
        ingain = 2
        inreadnoise = 10
        ngroups = 2
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(
            ngroups=ngroups, gain=ingain, readnoise=inreadnoise, deltatime=grouptime)

        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 10.0 + deltaDN

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        data, dq, var_poisson, var_rnoise, err = slopes
        # delta_electrons = deltaDN * ingain
        single_sample_readnoise = inreadnoise / np.sqrt(2)
        np.testing.assert_allclose(
            var_poisson[50, 50],
            ((deltaDN / ingain) / grouptime**2),
            1e-6)
        np.testing.assert_allclose(
            var_rnoise[50, 50],
            (inreadnoise**2 / grouptime**2),
            1e-6)
        np.testing.assert_allclose(
            var_rnoise[50, 50],
            (12 * single_sample_readnoise**2 / (ngroups * (ngroups**2 - 1) * grouptime**2)),
            1e-6)
        np.testing.assert_allclose(
            err[50, 50],
            (np.sqrt((deltaDN / ingain) / grouptime**2 + (inreadnoise**2 / grouptime**2))),
            1e-6)

    def test_five_groups_unc(self, method):
        grouptime = 3.0
        # deltaDN = 5
        ingain = 2
        inreadnoise = 7
        ngroups = 5
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(
            ngroups=ngroups, gain=ingain, readnoise=inreadnoise, deltatime=grouptime)

        model1.data[0, 0, 50, 50] = 10.0
        model1.data[0, 1, 50, 50] = 15.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 33.0
        model1.data[0, 4, 50, 50] = 60.0

        slopes, cube, optional, gls_dummy = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        # out_slope=slopes[0].data[500, 500]
        median_slope = np.median(np.diff(model1.data[0, :, 50, 50])) / grouptime
        # deltaDN = 50
        delta_time = (ngroups - 1) * grouptime
        # delta_electrons = median_slope * ingain *delta_time
        single_sample_readnoise = np.float64(inreadnoise / np.sqrt(2))

        data, dq, var_poisson, var_rnoise, err = slopes

        np.testing.assert_allclose(
            var_poisson[50, 50],
            ((median_slope) / (ingain * delta_time)),
            1e-6)
        np.testing.assert_allclose(
            var_rnoise[50, 50],
            (12 * single_sample_readnoise**2 / (ngroups * (ngroups**2 - 1) * grouptime**2)),
            1e-6)
        np.testing.assert_allclose(
            err[50, 50],
            np.sqrt(var_poisson[50, 50] + var_rnoise[50, 50]),
            1e-6)

    def test_oneCR_10_groups_combination(self, method):
        grouptime = 3.0
        # deltaDN = 5
        ingain = 200  # use large gain to show that Poisson noise doesn't affect the recombination
        inreadnoise = 7
        ngroups = 10

        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(
            ngroups=ngroups, gain=ingain, readnoise=inreadnoise, deltatime=grouptime)

        # two segments perfect fit, second segment has twice the slope
        model1.data[0, 0, 50, 50] = 15.0
        model1.data[0, 1, 50, 50] = 20.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 30.0
        model1.data[0, 4, 50, 50] = 35.0
        model1.data[0, 5, 50, 50] = 140.0
        model1.data[0, 6, 50, 50] = 150.0
        model1.data[0, 7, 50, 50] = 160.0
        model1.data[0, 8, 50, 50] = 170.0
        model1.data[0, 9, 50, 50] = 180.0
        model1.groupdq[0, 5, 50, 50] = JUMP_DET

        slopes, int_info, opt_info, gls_opt_model = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        segment_groups = 5
        single_sample_readnoise = np.float64(inreadnoise / np.sqrt(2))
        # check that the segment variance is as expected

        ovar_rnoise = opt_info[3]
        np.testing.assert_allclose(
            ovar_rnoise[0, 0, 50, 50],
            (12.0 * single_sample_readnoise**2 /
                (segment_groups * (segment_groups**2 - 1) * grouptime**2)),
            rtol=1e-6)

        # check the combined slope is the average of the two segments since they have the same number of groups
        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], 2.5, rtol=1e-5)

        # check that the slopes of the two segments are correct
        oslope = opt_info[0]
        np.testing.assert_allclose(oslope[0, 0, 50, 50], 5 / 3.0, rtol=1e-5)
        np.testing.assert_allclose(oslope[0, 1, 50, 50], 10 / 3.0, rtol=1e-5)

    def test_oneCR_10_groups_combination_noisy2ndSegment(self, method):
        grouptime = 3.0
        # deltaDN = 5
        ingain = 200  # use large gain to show that Poisson noise doesn't affect the recombination
        inreadnoise = 7
        ngroups = 10
        model1, gdq, rnoise, pixdq, err, gain = setup_inputs(
            ngroups=ngroups, gain=ingain, readnoise=inreadnoise, deltatime=grouptime)

        # two segments perfect fit, second segment has twice the slope
        model1.data[0, 0, 50, 50] = 15.0
        model1.data[0, 1, 50, 50] = 20.0
        model1.data[0, 2, 50, 50] = 25.0
        model1.data[0, 3, 50, 50] = 30.0
        model1.data[0, 4, 50, 50] = 35.0
        model1.data[0, 5, 50, 50] = 135.0
        model1.data[0, 6, 50, 50] = 155.0
        model1.data[0, 7, 50, 50] = 160.0
        model1.data[0, 8, 50, 50] = 168.0
        model1.data[0, 9, 50, 50] = 180.0
        model1.groupdq[0, 5, 50, 50] = JUMP_DET

        slopes, int_info, opt_info, gls_opt_model = ramp_fit(
            model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

        oslope = opt_info[0]
        avg_slope = (oslope[0, 0, 50, 50] + oslope[0, 1, 50, 50]) / 2.0

        # even with noiser second segment, final slope should be just the
        # average since they have the same number of groups
        data = slopes[0]
        np.testing.assert_allclose(data[50, 50], avg_slope, rtol=1e-5)


def test_twenty_groups_two_segments():
    """
    Test to verify weighting of multiple segments in combination:
    a) gdq all 0 ; b) 1 CR (2 segments) c) 1 CR then SAT (2 segments)
    """
    (ngroups, nints, nrows, ncols, deltatime) = (20, 1, 1, 3, 6.)
    model1, gdq, rnoise, pixdq, err, gain = setup_small_cube(
        ngroups, nints, nrows, ncols, deltatime)

    # a) ramp having gdq all 0
    model1.data[0, :, 0, 0] = np.arange(ngroups) * 10. + 30.

    # b) ramp having 1 CR at group 15; 2 segments
    model1.data[0, :, 0, 1] = np.arange(ngroups) * 10. + 50.
    gdq[0, 15, 0, 1] = JUMP_DET
    model1.data[0, 15:, 0, 1] += 1000.

    # c) ramp having 1 CR at group 2; SAT starting in group 15
    model1.data[0, :, 0, 2] = np.arange(ngroups) * 10. + 70.
    gdq[0, 2, 0, 2] = JUMP_DET
    model1.data[0, 2:, 0, 2] += 2000.
    gdq[0, 15:, 0, 2] = SATURATED
    model1.data[0, 15:, 0, 2] = 25000.

    new_mod, int_model, opt_info, gls_opt_model = ramp_fit(
        model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

    # Check some PRI & OPT output arrays
    data = new_mod[0]
    np.testing.assert_allclose(data, 10. / deltatime, rtol=1E-4)

    (oslope, sigslope, var_poisson, var_rnoise,
        oyint, sigyint, opedestal, weights, crmag) = opt_info

    wh_data = oslope != 0.  # only test existing segments
    np.testing.assert_allclose(oslope[wh_data], 10. / deltatime, rtol=1E-4)
    np.testing.assert_allclose(oyint[0, 0, 0, :], model1.data[0, 0, 0, :], rtol=1E-5)

    np.testing.assert_allclose(
        opedestal[0, 0, :],
        model1.data[0, 0, 0, :] - 10.,
        rtol=1E-5)


def test_miri_all_sat():
    """
    Test of all groups in all integrations being saturated; all output arrays
    (particularly variances) should be 0.
    """
    (ngroups, nints, nrows, ncols, deltatime) = (3, 2, 2, 2, 6.)
    model1, gdq, rnoise, pixdq, err, gain = setup_small_cube(
        ngroups, nints, nrows, ncols, deltatime)

    model1.groupdq[:, :, :, :] = SATURATED

    image_info, integ_info, opt_info, gls_opt_model = ramp_fit(
        model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

    # Check PRI output arrays
    data, dq, var_poisson, var_rnoise, err = image_info
    np.testing.assert_allclose(data, 0.0, atol=1E-6)
    np.testing.assert_allclose(err, 0.0, atol=1E-6)
    np.testing.assert_allclose(var_poisson, 0.0, atol=1E-6)
    np.testing.assert_allclose(var_rnoise, 0.0, atol=1E-6)

    # Check INT output arrays
    data, dq, var_poisson, var_rnoise, int_times, err = integ_info
    np.testing.assert_allclose(data, 0.0, atol=1E-6)
    np.testing.assert_allclose(err, 0.0, atol=1E-6)
    np.testing.assert_allclose(var_poisson, 0.0, atol=1E-6)
    np.testing.assert_allclose(var_rnoise, 0.0, atol=1E-6)

    # Check OPT output arrays
    (slope, sigslope, var_poisson, var_rnoise,
        yint, sigyint, pedestal, weights, crmag) = opt_info

    np.testing.assert_allclose(slope, 0.0, atol=1E-6)
    np.testing.assert_allclose(var_poisson, 0.0, atol=1E-6)
    np.testing.assert_allclose(var_rnoise, 0.0, atol=1E-6)
    np.testing.assert_allclose(sigslope, 0.0, atol=1E-6)
    np.testing.assert_allclose(yint, 0.0, atol=1E-6)
    np.testing.assert_allclose(sigyint, 0.0, atol=1E-6)
    np.testing.assert_allclose(pedestal, 0.0, atol=1E-6)
    np.testing.assert_allclose(weights, 0.0, atol=1E-6)


def test_miri_first_last():
    """
    This is a test of whether ramp fitting correctly handles having all 0th
    group dq flagged as DO_NOT_USE, and all final group dq flagged as
    DO_NOT_USE for MIRI data.  For 1 pixel ([1,1]) the 1st (again, 0-based)
    group is flagged as a CR.  For such a ramp, the code removes the CR-flag
    from such a CR-affected 1st group; so if it initially was 4 it is reset
    to 0 ("good"), in which case it's as if that CR was not even there.
    """
    # (ngroups, nints, nrows, ncols, deltatime) = (10, 1, 2, 2, 3.)
    nints, ngroups, nrows, ncols = 1, 10, 2, 2
    deltatime = 3.
    model1, gdq, rnoise, pixdq, err, gain = setup_small_cube(
        ngroups, nints, nrows, ncols, deltatime)

    # Make smooth ramps having outlier SCI values in the 0th and final groups
    #   to reveal if they are included in the fit (they shouldn't be, as those
    #   groups are flagged as DO_NOT_USE)
    model1.data[0, :, 0, 0] = np.array(
        [-200., 30., 40., 50., 60., 70., 80., 90., 100., -500.], dtype=np.float32)
    model1.data[0, :, 0, 1] = np.array(
        [-300., 80., 90., 100., 110., 120., 130., 140., 150., -600.], dtype=np.float32)
    model1.data[0, :, 1, 0] = np.array(
        [-200., 40., 50., 60., 70., 80., 90., 100., 110., 900.], dtype=np.float32)
    model1.data[0, :, 1, 1] = np.array(
        [-600., 140., 150., 160., 170., 180., 190., 200., 210., -400.], dtype=np.float32)

    # For all pixels, set gdq for 0th and final groups to DO_NOT_USE
    model1.groupdq[:, 0, :, :] = DO_NOT_USE
    model1.groupdq[:, -1, :, :] = DO_NOT_USE

    # Put CR in 1st (0-based) group
    model1.groupdq[0, 1, 1, 1] = JUMP_DET

    image_info, int_model, opt_model, gls_opt_model = ramp_fit(
        model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

    data = image_info[0]
    np.testing.assert_allclose(data, 10. / 3., rtol=1E-5)


def test_miri_no_good_pixel():
    """
    With no good data, MIRI will remove all groups where all pixels are bad.
    If all groups are bad, NoneType is returned for all return values from
    ramp_fit.  This test is to force this return of NoneType.
    """
    nints, ngroups, nrows, ncols = 1, 2, 2, 2
    deltatime = 3.
    model1, gdq, rnoise, pixdq, err, gain = setup_small_cube(
        ngroups, nints, nrows, ncols, deltatime)

    # Dummy non-zero data to make sure if processing occurs a non-NoneType gets
    # returned.  Processing should not occur and a NoneType should be returned.
    model1.data[0, :, 0, 0] = np.array([-200., -500.], dtype=np.float32)
    model1.data[0, :, 0, 1] = np.array([-300., -600.], dtype=np.float32)
    model1.data[0, :, 1, 0] = np.array([-200., 900.], dtype=np.float32)
    model1.data[0, :, 1, 1] = np.array([-600., -400.], dtype=np.float32)

    # Set all groups to DO_NOT_USE
    model1.groupdq[:, :, :, :] = DO_NOT_USE

    image_info, int_model, opt_model, gls_opt_model = ramp_fit(
        model1, 1024 * 30000., True, rnoise, gain, 'OLS', 'optimal', 'none')

    assert image_info is None


def setup_small_cube(ngroups=10, nints=1, nrows=2, ncols=2, deltatime=10.,
                     gain=1., readnoise=10.):
    """
    Create input MIRI datacube having the specified dimensions
    """
    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
    err = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float64)
    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float64)
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.int32)
    rnoise = np.full((nrows, ncols), readnoise, dtype=np.float64)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)
    model1 = RampModel(data=data, err=err, pixeldq=pixdq, groupdq=gdq)

    model1.meta.instrument.name = 'MIRI'
    model1.meta.instrument.detector = 'MIRIMAGE'
    model1.meta.instrument.filter = 'F480M'
    model1.meta.observation.date = '2015-10-13'
    model1.meta.exposure.type = 'MIR_IMAGE'
    model1.meta.exposure.group_time = deltatime
    model1.meta.subarray.name = 'FULL'

    model1.meta.subarray.xstart = 1
    model1.meta.subarray.ystart = 1
    model1.meta.subarray.xsize = ncols
    model1.meta.subarray.ysize = nrows
    model1.meta.exposure.drop_frames1 = 0
    model1.meta.exposure.frame_time = deltatime
    model1.meta.exposure.ngroups = ngroups
    model1.meta.exposure.group_time = deltatime
    model1.meta.exposure.nframes = 1
    model1.meta.exposure.groupgap = 0

    return model1, gdq, rnoise, pixdq, err, gain


# Need test for multi-ints near zero with positive and negative slopes
def setup_inputs(ngroups=10, readnoise=10, nints=1,
                 nrows=103, ncols=102, nframes=1, grouptime=1.0, gain=1, deltatime=1):

    data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
    gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint8)
    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
    rnoise = np.full((nrows, ncols), readnoise, dtype=np.float32)
    int_times = np.zeros((nints,))

    model1 = RampModel(data=data, err=err, pixeldq=pixdq, groupdq=gdq, int_times=int_times)
    model1.meta.instrument.name = 'MIRI'
    model1.meta.instrument.detector = 'MIRIMAGE'
    model1.meta.instrument.filter = 'F480M'
    model1.meta.observation.date = '2015-10-13'
    model1.meta.exposure.type = 'MIR_IMAGE'
    model1.meta.exposure.group_time = deltatime
    model1.meta.subarray.name = 'FULL'
    model1.meta.subarray.xstart = 1
    model1.meta.subarray.ystart = 1
    model1.meta.subarray.xsize = ncols
    model1.meta.subarray.ysize = nrows
    model1.meta.exposure.frame_time = deltatime
    model1.meta.exposure.ngroups = ngroups
    model1.meta.exposure.group_time = deltatime
    model1.meta.exposure.nframes = 1
    model1.meta.exposure.groupgap = 0
    model1.meta.exposure.drop_frames1 = 0

    return model1, gdq, rnoise, pixdq, err, gain
