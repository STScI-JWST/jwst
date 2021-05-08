import pytest
import numpy as np

from jwst.ramp_fitting.ramp_fit_step import RampFitStep

from jwst.datamodels import RampModel
from jwst.datamodels import GainModel, ReadnoiseModel


@pytest.fixture(scope="module")
def generate_miri_reffiles(tmpdir_factory):
    gainfile = str(tmpdir_factory.mktemp("data").join("gain.fits"))
    readnoisefile = str(tmpdir_factory.mktemp("data").join('readnoise.fits'))

    ingain = 6
    xsize = 103
    ysize = 102
    gain = np.ones(shape=(ysize, xsize), dtype=np.float64) * ingain
    gain_model = GainModel(data=gain)
    gain_model.meta.instrument.name = "MIRI"
    gain_model.meta.subarray.name = "FULL"
    gain_model.meta.subarray.xstart = 1
    gain_model.meta.subarray.ystart = 1
    gain_model.meta.subarray.xsize = xsize
    gain_model.meta.subarray.ysize = ysize
    gain_model.save(gainfile)

    inreadnoise = 5
    rnoise = np.ones(shape=(ysize, xsize), dtype=np.float64) * inreadnoise
    readnoise_model = ReadnoiseModel(data=rnoise)
    readnoise_model.meta.instrument.name = "MIRI"
    readnoise_model.meta.subarray.xstart = 1
    readnoise_model.meta.subarray.ystart = 1
    readnoise_model.meta.subarray.xsize = xsize
    readnoise_model.meta.subarray.ysize = ysize
    readnoise_model.save(readnoisefile)

    return gainfile, readnoisefile


@pytest.fixture
def setup_inputs():

    def _setup(ngroups=10, readnoise=10, nints=1, nrows=1024, ncols=1032,
               nframes=1, grouptime=1.0, gain=1, deltatime=1):
        times = np.array(list(range(ngroups)), dtype=np.float64) * deltatime
        gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
        err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float64)
        data = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.float64)
        pixdq = np.zeros(shape=(nrows, ncols), dtype=np.uint32)
        read_noise = np.full((nrows, ncols), readnoise, dtype=np.float64)
        gdq = np.zeros(shape=(nints, ngroups, nrows, ncols), dtype=np.uint32)
        int_times = np.zeros((nints,))

        rampmodel = RampModel(
            data=data, err=err, pixeldq=pixdq, groupdq=gdq, times=times, int_times=int_times)

        rampmodel.meta.instrument.name = 'MIRI'
        rampmodel.meta.instrument.detector = 'MIRIMAGE'
        rampmodel.meta.instrument.filter = 'F480M'
        rampmodel.meta.observation.date = '2015-10-13'
        rampmodel.meta.exposure.type = 'MIR_IMAGE'
        rampmodel.meta.exposure.group_time = deltatime
        rampmodel.meta.subarray.name = 'FULL'
        rampmodel.meta.subarray.xstart = 1
        rampmodel.meta.subarray.ystart = 1
        rampmodel.meta.subarray.xsize = ncols
        rampmodel.meta.subarray.ysize = nrows
        rampmodel.meta.exposure.frame_time = deltatime
        rampmodel.meta.exposure.ngroups = ngroups
        rampmodel.meta.exposure.group_time = deltatime
        rampmodel.meta.exposure.nframes = 1
        rampmodel.meta.exposure.groupgap = 0

        gain = GainModel(data=gain)
        gain.meta.instrument.name = 'MIRI'
        gain.meta.subarray.xstart = 1
        gain.meta.subarray.ystart = 1
        gain.meta.subarray.xsize = ncols
        gain.meta.subarray.ysize = nrows

        rnmodel = ReadnoiseModel(data=read_noise)
        rnmodel.meta.instrument.name = 'MIRI'
        rnmodel.meta.subarray.xstart = 1
        rnmodel.meta.subarray.ystart = 1
        rnmodel.meta.subarray.xsize = ncols
        rnmodel.meta.subarray.ysize = nrows

        return rampmodel, gdq, rnmodel, pixdq, err, gain

    return _setup


def setup_subarray_inputs(
        nints=1, ngroups=10, nrows=1032, ncols=1024,
        subxstart=1, subxsize=1024, subystart=1, subysize=1032,
        nframes=1, grouptime=1.0, deltatime=1,
        readnoise=10, gain=1):

    data = np.zeros(shape=(nints, ngroups, subysize, subxsize), dtype=np.float32)
    err = np.ones(shape=(nints, ngroups, nrows, ncols), dtype=np.float32)
    pixdq = np.zeros(shape=(subysize, subxsize), dtype=np.uint32)
    gdq = np.zeros(shape=(nints, ngroups, subysize, subxsize), dtype=np.uint8)
    gain = np.ones(shape=(nrows, ncols), dtype=np.float64) * gain
    read_noise = np.full((nrows, ncols), readnoise, dtype=np.float32)
    times = np.array(list(range(ngroups)), dtype=np.float64) * deltatime

    model1 = RampModel(data=data, err=err, pixeldq=pixdq, groupdq=gdq, times=times)
    model1.meta.instrument.name = 'MIRI'
    model1.meta.instrument.detector = 'MIRIMAGE'
    model1.meta.instrument.filter = 'F480M'
    model1.meta.observation.date = '2015-10-13'
    model1.meta.exposure.type = 'MIR_IMAGE'
    model1.meta.exposure.group_time = deltatime
    model1.meta.subarray.name = 'FULL'
    model1.meta.subarray.xstart = subxstart
    model1.meta.subarray.ystart = subystart
    model1.meta.subarray.xsize = subxsize
    model1.meta.subarray.ysize = subysize
    model1.meta.exposure.frame_time = deltatime
    model1.meta.exposure.ngroups = ngroups
    model1.meta.exposure.group_time = deltatime
    model1.meta.exposure.nframes = 1
    model1.meta.exposure.groupgap = 0

    gain = GainModel(data=gain)
    gain.meta.instrument.name = 'MIRI'
    gain.meta.subarray.xstart = 1
    gain.meta.subarray.ystart = 1
    gain.meta.subarray.xsize = 1024
    gain.meta.subarray.ysize = 1032

    rnModel = ReadnoiseModel(data=read_noise)
    rnModel.meta.instrument.name = 'MIRI'
    rnModel.meta.subarray.xstart = 1
    rnModel.meta.subarray.ystart = 1
    rnModel.meta.subarray.xsize = 1024
    rnModel.meta.subarray.ysize = 1032

    return model1, gdq, rnModel, pixdq, err, gain


def test_ramp_fit_step(generate_miri_reffiles, setup_inputs):
    """
    Create a simple input to instantiate RampFitStep and execute a call to test
    the step class and class method.
    """
    override_gain, override_readnoise = generate_miri_reffiles
    ingain, inreadnoise = 6, 7
    grouptime = 3.0
    nints, ngroups, nrows, ncols = 1, 5, 2, 2
    model, gdq, rnModel, pixdq, err, gain = setup_inputs(
        ngroups=ngroups, readnoise=inreadnoise, nints=nints, nrows=nrows,
        ncols=ncols, gain=ingain, deltatime=grouptime)

    # Add basic ramps to each pixel
    pix = [(0, 0), (0, 1), (1, 0), (1, 1)]
    base_ramp = np.array([k + 1 for k in range(ngroups)])
    ans_slopes = np.zeros(shape=(2,2))
    for k, p in enumerate(pix):
        ramp = base_ramp * (k + 1)  # A simple linear ramp
        x, y = p
        model.data[0, :, x, y] = ramp
        ans_slopes[x, y] = ramp[0] / grouptime

    # Call ramp fit through the step class
    slopes, cube_model = RampFitStep.call(
        model, override_gain=override_gain, override_readnoise=override_readnoise,
        maximum_cores="none")

    assert slopes is not None
    assert cube_model is not None

    # Test to make sure the ramps are as expected and that the step is complete
    np.testing.assert_allclose(slopes.data, ans_slopes, rtol=1e-5)
    assert slopes.meta.cal_step.ramp_fit == "COMPLETE"


def test_subarray_5groups(tmpdir_factory):
    # all pixel values are zero. So slope should be zero
    gainfile = str(tmpdir_factory.mktemp("data").join("gain.fits"))
    readnoisefile = str(tmpdir_factory.mktemp("data").join('readnoise.fits'))

    model1, gdq, rnModel, pixdq, err, gain = setup_subarray_inputs(
        ngroups=5, subxstart=10, subystart=20, subxsize=5, subysize=15, readnoise=50)
    gain.save(gainfile)
    rnModel.save(readnoisefile)

    model1.meta.exposure.ngroups = 11
    model1.data[0, 0, 12, 1] = 10.0
    model1.data[0, 1, 12, 1] = 15.0
    model1.data[0, 2, 12, 1] = 25.0
    model1.data[0, 3, 12, 1] = 33.0
    model1.data[0, 4, 12, 1] = 60.0

    # Call ramp fit through the step class
    slopes, cube_model = RampFitStep.call(
        model1, override_gain=gainfile, override_readnoise=readnoisefile,
        maximum_cores="none")

    assert slopes is not None
    assert cube_model is not None

    xvalues = np.arange(5) * 1.0
    yvalues = np.array([10, 15, 25, 33, 60])
    coeff = np.polyfit(xvalues, yvalues, 1)

    np.testing.assert_allclose(slopes.data[12, 1], coeff[0], 1e-6)


def test_int_times1(generate_miri_reffiles, setup_inputs):
    # Test whether int_times table gets copied to output when it should
    override_gain, override_readnoise = generate_miri_reffiles
    ingain, inreadnoise = 6, 7
    grouptime = 3.0
    nints, ngroups, nrows, ncols = 5, 3, 2, 2
    model, gdq, rnModel, pixdq, err, gain = setup_inputs(
        ngroups=ngroups, readnoise=inreadnoise, nints=nints, nrows=nrows,
        ncols=ncols, gain=ingain, deltatime=grouptime)

    # Set TSOVISIT false, in which case the int_times table should come back with zero length
    model.meta.visit.tsovisit = False

    # Call ramp fit through the step class
    slopes, cube_model = RampFitStep.call(
        model, override_gain=override_gain, override_readnoise=override_readnoise,
        maximum_cores="none")

    assert slopes is not None
    assert cube_model is not None

    assert(len(cube_model.int_times) == 0)


def test_int_times2(generate_miri_reffiles, setup_inputs):
    # Test whether int_times table gets copied to output when it should
    override_gain, override_readnoise = generate_miri_reffiles
    ingain, inreadnoise = 6, 7
    grouptime = 3.0
    nints, ngroups, nrows, ncols = 5, 3, 2, 2
    model, gdq, rnModel, pixdq, err, gain = setup_inputs(
        ngroups=ngroups, readnoise=inreadnoise, nints=nints, nrows=nrows,
        ncols=ncols, gain=ingain, deltatime=grouptime)

    # Set TSOVISIT false, in which case the int_times table should come back with zero length
    model.meta.visit.tsovisit = True

    # Call ramp fit through the step class
    slopes, cube_model = RampFitStep.call(
        model, override_gain=override_gain, override_readnoise=override_readnoise,
        maximum_cores="none")

    assert slopes is not None
    assert cube_model is not None

    assert(len(cube_model.int_times) == nints)
