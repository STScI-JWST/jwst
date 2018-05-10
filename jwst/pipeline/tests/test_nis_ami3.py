"""Test calwebb_ami3 with NIR"""

from collections import defaultdict
from glob import glob
from os import path

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
    path.join('$TEST_BIGDATA', 'niriss', 'test_ami_pipeline')
)


@runslow
@require_bigdata
def test_run_full(mk_tmp_dirs):
    """Test a full run"""
    tmp_current_path, tmp_data_path, tmp_config_path = mk_tmp_dirs

    asn_path = path.join(DATAPATH, 'test_lg1_correct_asn.json')
    args = [
        path.join(SCRIPT_DATA_PATH, 'cfgs', 'calwebb_ami3.cfg'),
        asn_path,
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

    output_files = glob('*')
    print('Created files ares: {}'.format(output_files))

    # Check Level3 products
    product_name_file = product_name + '_amiavg.fits'
    assert product_name_file in output_files
    output_files.remove(product_name_file)

    product_name_file = product_name + '_psf-amiavg.fits'
    assert product_name_file in output_files
    output_files.remove(product_name_file)

    product_name_file = product_name + '_aminorm.fits'
    assert product_name_file in output_files
    output_files.remove(product_name_file)

    # Check Level2 products
    for member in members_by_type['psf']:
        name, ext = path.splitext(path.split(member)[1])
        name, separator = remove_suffix(name)
        name = name + separator + acid + separator + 'ami' + ext
        assert name in output_files
        output_files.remove(name)

    for member in members_by_type['science']:
        name, ext = path.splitext(path.split(member)[1])
        name, separator = remove_suffix(name)
        name = name + separator + acid + separator + 'ami' + ext
        assert name in output_files
        output_files.remove(name)

    # If there are files left, this is an error
    assert len(output_files) == 0
