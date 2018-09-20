import pytest
from jwst.pipeline.collect_pipeline_cfgs import collect_pipeline_cfgs
from jwst.stpipe import Step

from ..resources import MIRITest

pytestmark = [
    pytest.mark.usefixtures('_jail'),
    pytest.mark.skipif(not pytest.config.getoption('bigdata'),
                       reason='requires --bigdata')
]

class TestSpec2Pipeline(MIRITest):
    ref_loc = ['test_spec2pipeline']

    def test_mirilrs2pipeline1(self):
        """
        Regression test of calwebb_spec2 pipeline performed on
        MIRI LRS slitless data.
        """
        input_file = self.get_data('test_spec2pipeline',
                                   'jw80600012001_02101_00003_mirimage_rateints.fits')

        collect_pipeline_cfgs()
        args = [
            'calwebb_tso_spec2.cfg',
            input_file,
        ]
        Step.from_cmdline(args)

        outputs = [('jw80600012001_02101_00003_mirimage_calints.fits',
                    'jw80600012001_02101_00003_mirimage_calints_ref.fits',
                    ['primary', 'sci', 'err', 'dq', 'relsens']
                    ),
                    ('jw80600012001_02101_00003_mirimage_x1dints.fits',
                     'jw80600012001_02101_00003_mirimage_x1dints_ref.fits',
                     ['primary', ('extract1d', 1), ('extract1d', 2), ('extract1d', 3), ('extract1d', 4)]
                    )
        ]
        self.compare_outputs(outputs)
