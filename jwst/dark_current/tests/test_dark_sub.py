"""
Unit tests for dark current correction
"""

import pytest
import numpy as np

from jwst.dark_current.dark_sub import average_dark_frames
from jwst.datamodels import (
    RampModel, 
    DarkModel, 
    MIRIRampModel, 
    DarkMIRIModel, 
)
from jwst.dark_current.dark_sub import do_correction as darkcorr


# Dictionary of NIRCam readout patterns
READPATTERNS = dict(
    DEEP8 = dict(ngroups=20, nframes=8, nskip=12),
    DEEP2 = dict(ngroups=20, nframes=2, nskip=18),
    MEDIUM8 = dict(ngroups=10, nframes=8, nskip=2),
    MEDIUM2 = dict(ngroups=10, nframes=2, nskip=8),
    SHALLOW4 = dict(ngroups=10, nframes=4, nskip=1),
    SHALLOW2 = dict(ngroups=10, nframes=2, nskip=3),
    BRIGHT2 = dict(ngroups=10, nframes=2, nskip=0),
    BRIGHT1 = dict(ngroups=10, nframes=1, nskip=1),
    RAPID = dict(ngroups=10, nframes=1, nskip=0),
    )

TFRAME = 10.73677


def test_frame_averaging(setup_nrc_cube):
    '''Check that if nframes>1 or groupgap>0, then the pipeline reconstructs
       the dark reference file to match the frame averaging and groupgap
       settings of the exposure.'''

    # Values to build the fake data arrays.  Rows/Cols are smaller than the
    # normal 2048x2048 to save memory and time
    ngroups = 3
    nrows = 512
    ncols = 512

    # Loop over the NIRCam readout patterns:
    for readpatt in READPATTERNS:

        # Get the configuration for the readout pattern
        nframes = READPATTERNS[readpatt]['nframes']
        groupgap = READPATTERNS[readpatt]['nskip']

        # Create data and dark model
        data, dark = setup_nrc_cube(readpatt, ngroups, nrows, ncols)

        # Add ramp values to dark model data array
        dark.data[:, 500, 500] = np.arange(0, 100)
        dark.err[:, 500, 500] = np.arange(100, 200)

        # Run the pipeline's averaging function
        avg_dark = average_dark_frames(dark, ngroups, nframes, groupgap)

        # Group input groups into collections of frames which will be averaged
        total_frames = (nframes * ngroups) + (groupgap * (ngroups-1))

        # Get starting/ending indexes of the input groups to be averaged
        gstrt_ind = np.arange(0, total_frames, nframes + groupgap)
        gend_ind = gstrt_ind + nframes

        # Prepare arrays to hold results of averaging
        manual_avg = np.zeros((ngroups))
        manual_errs = np.zeros((ngroups))

        # Manually average the input data to compare with pipeline output
        for newgp, gstart, gend in zip(range(ngroups), gstrt_ind, gend_ind):

            # Average the data frames
            newframe = np.mean(dark.data[gstart:gend, 500, 500])
            manual_avg[newgp] = newframe

            # ERR arrays will be quadratic sum of error values
            manual_errs[newgp] = np.sqrt(np.sum(dark.err[gstart:gend, 500, 500]**2)) / (gend - gstart)

        # Check that pipeline output matches manual averaging results
        assert np.all(manual_avg == avg_dark.data[:, 500, 500])
        assert np.all(manual_errs == avg_dark.err[:, 500, 500])

        # Check that meta data was properly updated
        assert avg_dark.meta.exposure.nframes == nframes
        assert avg_dark.meta.exposure.ngroups == ngroups
        assert avg_dark.meta.exposure.groupgap == groupgap



def test_more_sci_frames():
    '''Check that data is unchanged if there are more frames in the science
    data is than in the dark reference file'''

    # size of integration
    nints = 1
    ngroups = 30
    xsize = 1032
    ysize = 1024

    # create raw input data for step
    dm_ramp = make_rampmodel(nints, ngroups, ysize, xsize)
    dm_ramp.meta.exposure.nframes = 1
    dm_ramp.meta.exposure.groupgap = 0

    # populate data array of science cube
    for i in range(0, ngroups-1):
        dm_ramp.data[0, i, :, :] = i

    refgroups = 20
    # create dark reference file model with fewer frames than science data
    dark = make_darkmodel(refgroups, ysize, xsize)

    # populate data array of reference file
    for i in range(0, refgroups - 1):
        dark.data[0, i, :, :] = i * 0.1

    # apply correction
    outfile = darkcorr(dm_ramp, dark)

    # check that no correction/subtraction was applied; input file = output file
    diff = dm_ramp.data[:, :, :, :] - outfile.data[:, :, :, :]

    # test that the science data are not changed

    np.testing.assert_array_equal(np.full((nints, ngroups, ysize, xsize), 0.0, dtype=float),
                                  diff, err_msg='no changes should be seen in array ')


def test_sub_by_frame():
    '''Check that if NFRAMES=1 and GROUPGAP=0 for the science data, the dark reference data are
    directly subtracted frame by frame'''

    # size of integration
    nints = 1
    ngroups = 30
    xsize = 1032
    ysize = 1024

    # create raw input data for step
    dm_ramp = make_rampmodel(nints, ngroups, ysize, xsize)
    dm_ramp.meta.exposure.nframes = 1
    dm_ramp.meta.exposure.groupgap = 0

    # populate data array of science cube
    for i in range(0, ngroups-1):
        dm_ramp.data[0, i, :, :] = i

    refgroups = 50
    # create dark reference file model with fewer frames than science data
    dark = make_darkmodel(refgroups, ysize, xsize)

    # populate data array of reference file
    for i in range(0, refgroups - 1):
        dark.data[0, i, :, :] = i * 0.1

    # apply correction
    outfile = darkcorr(dm_ramp, dark)

    # remove the single dimension at start of file (1, 30, 1032, 1024) so comparison in assert works
    outdata = np.squeeze(outfile.data)

    # check that the dark file is subtracted frame by frame from the science data
    diff = dm_ramp.data[0, :, :, :] - dark.data[0, :ngroups, :, :]

    # test that the output data file is equal to the difference found when subtracting ref file from sci file

    np.testing.assert_array_equal(outdata, diff, err_msg='dark file should be subtracted from sci file ')


def test_nan():
    '''Verify that when a dark has NaNs, these are correctly assumed as zero and the PIXELDQ is set properly'''

    # size of integration
    nints = 1
    ngroups = 10
    xsize = 1032
    ysize = 1024

    # create raw input data for step
    dm_ramp = make_rampmodel(nints, ngroups, ysize, xsize)
    dm_ramp.meta.exposure.nframes = 1
    dm_ramp.meta.exposure.groupgap = 0

    # populate data array of science cube
    for i in range(0, ngroups-1):
        dm_ramp.data[0, i, :, :] = i

    refgroups = 15
    # create dark reference file model with fewer frames than science data
    dark = make_darkmodel(refgroups, ysize, xsize)

    # populate data array of reference file
    for i in range(0, refgroups - 1):
        dark.data[0, i, :, :] = i * 0.1

    # set NaN in dark file
    dark.data[0, 5, 500, 500] = np.nan

    # apply correction
    outfile = darkcorr(dm_ramp, dark)

    print(outfile.pixeldq[500, 500])
    print(outfile.groupdq[0, 5, 500, 500])

    # test that the NaN dark reference pixel was set to 0 (nothing subtracted)
    assert outfile.data[0, 5, 500, 500] == 5.0

    # test that the output dq file is flagged (with what)


def test_dq_combine():
    '''Verify that the DQ array of the dark is correctly combined with the PIXELDQ array of the science data.'''

    # size of integration
    nints = 1
    ngroups = 5
    xsize = 1032
    ysize = 1024

    # create raw input data for step
    dm_ramp = make_rampmodel(nints, ngroups, ysize, xsize)
    dm_ramp.meta.exposure.nframes = 1
    dm_ramp.meta.exposure.groupgap = 0

    # populate data array of science cube
    for i in range(0, ngroups-1):
        dm_ramp.data[0, i, :, :] = i

    refgroups = 15
    # create dark reference file model with fewer frames than science data
    dark = make_darkmodel(refgroups, ysize, xsize)

    # populate dq flags of sci pixeldq and reference dq
    dm_ramp.pixeldq[500, 500] = 4
    dm_ramp.pixeldq[500, 501] = 2

    dark.dq[0, 0, 500, 500] = 1
    dark.dq[0, 0, 500, 501] = 1

    # run correction step
    outfile = darkcorr(dm_ramp, dark)

    # check that dq flags were correctly added
    assert outfile.pixeldq[500, 500] == 5
    assert outfile.pixeldq[500, 501] == 3


def test_2_int():
    '''Verify the dark correction is done by integration for MIRI observations'''

    # size of integration
    nints = 2
    ngroups = 10
    xsize = 1032
    ysize = 1024

    # create raw input data for step
    dm_ramp = make_rampmodel(nints, ngroups, ysize, xsize)
    dm_ramp.meta.exposure.nframes = 1
    dm_ramp.meta.exposure.groupgap = 0

    # populate data array of science cube
    for i in range(0, ngroups-1):
        dm_ramp.data[:, i, :, :] = i

    refgroups = 15
    # create dark reference file model with fewer frames than science data
    dark = make_darkmodel(refgroups, ysize, xsize)

    # populate data array of reference file
    for i in range(0, refgroups - 1):
        dark.data[0, i, :, :] = i * 0.1
        dark.data[1, i, :, :] = i * 0.2

    # run correction
    outfile = darkcorr(dm_ramp, dark)

    # perform subtractions manually

    # check that the dark file is subtracted frame by frame from the science data
    diff = dm_ramp.data[0, :, :, :] - dark.data[0, :ngroups, :, :]
    diff_int2 = dm_ramp.data[1, :, :, :] - dark.data[1, :ngroups, :, :]

    # test that the output data file is equal to the difference found when subtracting ref file from sci file

    np.testing.assert_array_equal(outfile.data[0, :, :, :], diff,
                                  err_msg='dark file should be subtracted from sci file ')
    np.testing.assert_array_equal(outfile.data[1, :, :, :], diff_int2,
                                  err_msg='dark file should be subtracted from sci file ')


def test_dark_skipped():
    '''Verify that when the dark is not applied, the data is correctly flagged as such.'''

    # size of integration
    nints = 1
    ngroups = 30
    xsize = 1032
    ysize = 1024

    # create raw input data for step
    dm_ramp = make_rampmodel(nints, ngroups, ysize, xsize)
    dm_ramp.meta.exposure.nframes = 1
    dm_ramp.meta.exposure.groupgap = 0

    # populate data array of science cube
    for i in range(0, ngroups-1):
        dm_ramp.data[0, i, :, :] = i

    refgroups = 20
    # create dark reference file model with fewer frames than science data
    dark = make_darkmodel(refgroups, ysize, xsize)

    # populate data array of reference file
    for i in range(0, refgroups - 1):
        dark.data[0, i, :, :] = i * 0.1

    # apply correction
    outfile = darkcorr(dm_ramp, dark)

    # get dark correction status from header
    darkstatus = outfile.meta.cal_step.dark_sub
    print('Dark status', darkstatus)

    assert darkstatus == 'SKIPPED'


def test_frame_avg():
    '''Check that if NFRAMES>1 or GROUPGAP>0, the frame-averaged dark data are
    subtracted group-by-group from science data groups and the ERR arrays are not modified'''

    # size of integration
    nints = 1
    ngroups = 5
    xsize = 1032
    ysize = 1024

    # create raw input data for step
    dm_ramp = make_rampmodel(nints, ngroups, ysize, xsize)
    dm_ramp.meta.exposure.nframes = 4
    dm_ramp.meta.exposure.groupgap = 0

    # populate data array of science cube
    for i in range(0, ngroups-1):
        dm_ramp.data[:, i, :, :] = i + 1

    refgroups = 20
    # create dark reference file model
    dark = make_darkmodel(refgroups, ysize, xsize)

    # populate data array of reference file
    for i in range(0, refgroups - 1):
        dark.data[0, i, :, :] = i * 0.1

    # apply correction
    outfile = darkcorr(dm_ramp, dark)

    # dark frames should be averaged in groups of 4 frames
    # this will result in average values of 0.15, 0.55, .095. 1.35

    assert outfile.data[0, 0, 500, 500] == pytest.approx(0.85)
    assert outfile.data[0, 1, 500, 500] == pytest.approx(1.45)
    assert outfile.data[0, 2, 500, 500] == pytest.approx(2.05)
    assert outfile.data[0, 3, 500, 500] == pytest.approx(2.65)

    # check that the error array is not modified.
    np.testing.assert_array_equal(outfile.err[:, :], 0,
                                  err_msg='error array should remain 0 ')


def make_rampmodel(nints, ngroups, ysize, xsize):
    '''Make MIRI Ramp model for testing'''
    # create the data and groupdq arrays
    csize = (nints, ngroups, ysize, xsize)
    data = np.full(csize, 1.0)
    pixeldq = np.zeros((ysize, xsize), dtype=int)
    groupdq = np.zeros(csize, dtype=int)
    err = np.zeros((ysize, xsize), dtype=int)

    # create a JWST datamodel for MIRI data
    dm_ramp = MIRIRampModel(data=data, pixeldq=pixeldq, groupdq=groupdq, err=err)

    dm_ramp.meta.instrument.name = 'MIRI'
    dm_ramp.meta.observation.date = '2018-01-01'
    dm_ramp.meta.observation.time = '00:00:00'
    dm_ramp.meta.subarray.xstart = 1
    dm_ramp.meta.subarray.xsize = xsize
    dm_ramp.meta.subarray.ystart = 1
    dm_ramp.meta.subarray.ysize = ysize
    dm_ramp.meta.description = 'Fake data.'


    return dm_ramp


def make_darkmodel(ngroups, ysize, xsize):
    '''Make MIRI dark model for testing'''
    # create the data and groupdq arrays
    nints = 2
    csize = (nints, ngroups, ysize, xsize)
    data = np.full(csize, 1.0)
    dq = np.zeros((nints, 1, ysize, xsize), dtype=int)

    # create a JWST datamodel for MIRI data
    dark = DarkMIRIModel(data=data, dq=dq)

    dark.meta.instrument.name = 'MIRI'
    dark.meta.date = '2018-01-01'
    dark.meta.time = '00:00:00'
    dark.meta.subarray.xstart = 1
    dark.meta.subarray.xsize = xsize
    dark.meta.subarray.ystart = 1
    dark.meta.subarray.ysize = ysize
    dark.meta.exposure.nframes = 1
    dark.meta.exposure.groupgap = 0
    dark.meta.description = 'Fake data.'
    dark.meta.reftype = 'DarkModel'
    dark.meta.author = 'Alicia'
    dark.meta.pedigree = 'Dummy'
    dark.meta.useafter = '2015-10-01T00:00:00'    

    return dark


@pytest.fixture(scope='function')
def setup_nrc_cube():
    '''Set up fake NIRCam data to test.'''

    def _cube(readpatt, ngroups, nrows, ncols):

        nints = 1
        groupgap = READPATTERNS[readpatt.upper()]['nskip']
        nframes = READPATTERNS[readpatt.upper()]['nframes']

        data_model = RampModel((nints, ngroups, nrows, ncols))
        data_model.meta.subarray.xstart = 1
        data_model.meta.subarray.ystart = 1
        data_model.meta.subarray.xsize = ncols
        data_model.meta.subarray.ysize = nrows
        data_model.meta.exposure.ngroups = ngroups
        data_model.meta.exposure.groupgap = groupgap
        data_model.meta.exposure.nframes = nframes
        data_model.meta.exposure.frame_time = TFRAME
        data_model.meta.exposure.group_time = (nframes + groupgap) * TFRAME
        data_model.meta.instrument.name = 'NIRCAM'
        data_model.meta.instrument.detector = 'NRCA1'
        data_model.meta.observation.date = '2017-10-01'
        data_model.meta.observation.time = '00:00:00'

        dark_model = DarkModel((100, 2048, 2048))
        dark_model.meta.subarray.xstart = 1
        dark_model.meta.subarray.ystart = 1
        dark_model.meta.subarray.xsize = 2048
        dark_model.meta.subarray.ysize = 2048
        dark_model.meta.exposure.ngroups = 100
        dark_model.meta.exposure.groupgap = 0
        dark_model.meta.exposure.nframes = 1
        dark_model.meta.instrument.name = 'NIRCAM'
        dark_model.meta.description = 'Fake data.'
        dark_model.meta.telescope = 'JWST'
        dark_model.meta.reftype = 'DarkModel'
        dark_model.meta.author = 'Alicia'
        dark_model.meta.pedigree = 'Dummy'
        dark_model.meta.useafter = '2015-10-01T00:00:00'

        return data_model, dark_model

    return _cube