"""Test calwebb_spec2 for NIRSpec MSA"""

from os import path
import pytest
from shutil import copy as file_copy

from .helpers import (
    SCRIPT_DATA_PATH,
    abspath,
    mk_tmp_dirs,
    require_bigdata,
    require_crds_context,
    runslow,
    update_asn_basedir,
)

from ...associations import load_asn
from ...stpipe.step import Step

DATAPATH = abspath(
    '$TEST_BIGDATA/nirspec/test_datasets/msa/simulated-3nod'
)


@require_crds_context(365)
@runslow
@require_bigdata
def test_run_msaflagging(mk_tmp_dirs, caplog):
    """Test msa flagging operation"""
    tmp_current_path, tmp_data_path, tmp_config_path = mk_tmp_dirs

    # Copy msa config files from DATAPATH to
    # current working directory
    file_copy(path.join(DATAPATH, 'jw95065006001_0_msa_twoslit.fits'), '.')

    asn_path = update_asn_basedir(
        path.join(DATAPATH, 'mos_udf_g235m_twoslit_spec2_asn.json'),
        root=path.join(DATAPATH, 'level2a_twoslit')
    )
    args = [
        path.join(SCRIPT_DATA_PATH, 'calwebb_spec2_basic.cfg'),
        asn_path,
        '--steps.msa_flagging.skip=false'
    ]

    Step.from_cmdline(args)

    assert 'Step msa_flagging running with args' in caplog.text
    assert 'Step msa_flagging done' in caplog.text

    with open(asn_path) as fp:
        asn = load_asn(fp)

    for product in asn['products']:
        prod_name = product['name'] + '_cal.fits'
        assert path.isfile(prod_name)
