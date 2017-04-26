from __future__ import absolute_import

from glob import glob
import os
import pytest

from astropy.table import Table

from .helpers import (
    TemporaryDirectory,
    full_pool_rules,
)

from ..main import Main
from .. import load_asn

# Temporarily skip if running under Travis
pytestmark = pytest.mark.skipif(
    "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
    reason='Temporarily disable due to performance issues'
)


@pytest.yield_fixture(
    scope='module',
    params=['yaml', 'json']
)
def make_asns(request):
    asn_format = request.param
    pool, rules, pool_fname = full_pool_rules(None)
    with TemporaryDirectory() as path:
        generated = Main([
            pool_fname,
            '-p', path,
            '--save-orphans',
            '--format', asn_format
        ])
        yield generated, path, asn_format


def test_roundtrip(make_asns):
    generated, path, asn_format = make_asns
    asn_files = glob(os.path.join(path, '*.' + asn_format))
    assert len(asn_files) == len(generated.associations)

    for asn_file in asn_files:
        with open(asn_file, 'r') as asn_fp:
            asn = load_asn(asn_fp)

    orphaned_files = glob(os.path.join(path, '*.csv'))
    assert len(orphaned_files) == 1
    orphaned = Table.read(
        orphaned_files[0],
        format='ascii',
        delimiter='|'
    )
    assert len(orphaned) == len(generated.orphaned)


def test_load_asn_all(make_asns):
    generated, path, asn_format = make_asns
    asn_files = glob(os.path.join(path, '*.' + asn_format))
    assert len(asn_files) == len(generated.associations)

    for asn_file in asn_files:
        with open(asn_file, 'r') as asn_fp:
            asns = load_asn(asn_fp, registry=generated.rules, first=False)
        assert len(asns) > 1
