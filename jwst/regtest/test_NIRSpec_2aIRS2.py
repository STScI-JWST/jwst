import os
import pytest

from glob import glob
from astropy.io.fits.diff import FITSDiff

from jwst.pipeline import Detector1Pipeline
from jwst.pipeline.collect_pipeline_cfgs import collect_pipeline_cfgs
from jwst.stpipe import Step


@pytest.mark.bigdata
def test_NIRSpec_det1(rtdata, fitsdiff_default_kwargs, _jail):
    """
        Check to make sure that the input and outout files exist and that they are not the
        same file.
    """

    rtdata.get_data("nirspec/test_detector1Pipeline/jw0010010_11010_nrs1_chimera_uncal.fits")
    Detector1Pipeline.call(rtdata.input, save_results=True)
    rtdata.output = "jw0010010_11010_nrs1_chimera_rate.fits"

    rtdata.get_truth("truth/nirspec/test_detector1Pipeline/jw0010010_11010_nrs1_chimera_rate.fits")
    assert rtdata.output != rtdata.truth

    diff = FITSDiff(rtdata.output, rtdata.truth, **fitsdiff_default_kwargs)
    assert diff.identical, diff.report

@pytest.fixture(scope="module")
def run_pipeline(rtdata_module, jail):
    """Run calwebb_Detector1 pipeline on NIRSpec uncal data. The config file is
        needed so that the intermediate results are saved."""
    rtdata = rtdata_module
    rtdata.get_data("nirspec/test_detector1Pipeline/jw0010010_11010_nrs1_chimera_rate.fits")

    #collect_pipeline_cfgs('config')
    #config_file = os.path.join('config', 'calwebb_image2.cfg')
    #Detector1Pipeline.call(rtdata.input, config_file=config_file,
    #                    save_results=True)

    #Detector1Pipeline.step_defs['jump'].rejection_threshold=150
    det1_pipe = Detector1Pipeline()
    det1_pipe.jump.rejection_threshold = 150.0
    det1_pipe.run(rtdata.input)
    #Detector1Pipeline.call(rtdata.input, save_results=True)

    return rtdata
@pytest.mark.bigdata
def test_NIRSpec_det1_completion(run_pipeline):
    """ Check that two files are generated by the pipeline."""
    files = glob('*_rate.fits')
    # There should be 1 output
    assert len(files) == 1


@pytest.mark.bigdata
@pytest.mark.parametrize("output", ['jw0010010_11010_nrs1_chimera_rate.fits'],ids=['rate'])
def test_NIRSpec_det1Pipeline(run_pipeline, fitsdiff_default_kwargs, output):
    """
    Regression test of calwebb_Detector1 pipeline performed on NIRSpec data.
    """
    rtdata = run_pipeline
    rtdata.output = output
    rtdata.get_truth(os.path.join("truth/nirspec/test_detector1Pipeline", output))


    diff = FITSDiff(rtdata.output, rtdata.truth, **fitsdiff_default_kwargs)
    assert diff.identical, diff.report()
