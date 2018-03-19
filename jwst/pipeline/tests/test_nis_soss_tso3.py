"""Test calwebb_tso3 with NIRISS SOSS"""

from collections import defaultdict
from copy import copy
from glob import glob
from os import path
import pytest

from .helpers import (
    SCRIPT_PATH,
    SCRIPT_DATA_PATH,
    abspath,
    mk_tmp_dirs,
    require_bigdata,
    runslow,
    update_asn_basedir,
)

from ...associations import load_asn
from ...stpipe.step import (Step, remove_suffix)

DATAPATH = abspath(
    path.join('$TEST_BIGDATA', 'pipelines', 'niriss_caltso3')
)


@pytest.mark.xfail(
    reason='Issue #1800',
    run=False
)
@runslow
@require_bigdata
def test_run_full_noscale(mk_tmp_dirs):
    """Test a full run"""
    tmp_current_path, tmp_data_path, tmp_config_path = mk_tmp_dirs

    asn_path = update_asn_basedir(
        path.join(DATAPATH, 'jw87600-a3001_20170527t111213_tso3_001_asn.json'),
        root=DATAPATH
    )
    args = [
        path.join(SCRIPT_DATA_PATH, 'cfgs', 'calwebb_tso3.cfg'),
        asn_path,
        '--scale_detection=False',
        '--steps.outlier_detection.save_intermediate_results=True',
        '--output_dir=' + tmp_data_path
    ]

    Step.from_cmdline(args)

    # Now test for file existence. Get the association
    with open(asn_path) as fh:
        asn = load_asn(fh)
    acid = asn['asn_id']
    product = asn['products'][0]
    product_name = product['name']
    members_by_type = defaultdict(list)
    for member in product['members']:
        expname = path.split(member['expname'])[1]
        members_by_type[member['exptype'].lower()].append(expname)

    output_files = [
        path.split(result_path)[1]
        for result_path in
        glob(path.join(tmp_data_path, '*'))
    ]
    print('Created files ares: {}'.format(output_files))

    # Check Level3 products
    product_name_file = product_name + '_whtlt.ecsv'
    assert product_name_file in output_files
    output_files.remove(product_name_file)

    product_name_file = product_name + '_x1dints.fits'
    assert product_name_file in output_files
    output_files.remove(product_name_file)

    # Check Level2 products
    for member in members_by_type['science']:
        basename, ext = path.splitext(path.split(member)[1])
        basename, separator = remove_suffix(basename)

        name = basename + separator + acid + separator + 'crfints' + ext
        assert name in output_files
        output_files.remove(name)

        name = basename + separator + acid + separator + 'median' + ext
        assert name in output_files
        output_files.remove(name)

    # If there are files left, this is an error
    assert len(output_files) == 0
