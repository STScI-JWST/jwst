"""
Test the utility functions
"""

import os

from astropy.table import QTable

from ...lib.catalog_utils import SkyObject
from ... import datamodels

from ..util import get_object_info, bounding_box_from_shape

from . import data

data_path = os.path.split(os.path.abspath(data.__file__))[0]


def get_file_path(filename):
    """
    Construct an absolute path.
    """
    return os.path.join(data_path, filename)


def test_bounding_box_from_shape_2d():
    model = datamodels.ImageModel((512, 2048))
    bb = bounding_box_from_shape(model.data.shape)
    assert bb == ((-0.5, 2047.5), (-0.5, 511.5))


def test_bounding_box_from_shape_3d():
    model = datamodels.CubeModel((3, 32, 2048))
    bb = bounding_box_from_shape(model.data.shape)
    assert bb == ((-0.5, 2047.5), (-0.5, 31.5))

    model = datamodels.IFUCubeModel((750, 45, 50))
    bb = bounding_box_from_shape(model.data.shape)
    assert bb == ((-0.5, 49.5), (-0.5, 44.5))


def read_catalog(catalogname):
    return get_object_info(catalogname)


def test_create_grism_objects():
    source_catalog = get_file_path('step_SourceCatalogStep_cat.ecsv')

    # create from test ascii file
    grism_objects = read_catalog(source_catalog)
    assert isinstance(grism_objects, list), "return grism objects were not a list"

    required_fields = list(SkyObject()._fields)
    go_fields = grism_objects[0]._fields
    assert all([a == b for a, b in zip(required_fields, go_fields)]), "Required fields mismatch for SkyObject and GrismObject"

    # create from QTable object
    tempcat = QTable.read(source_catalog, format='ascii.ecsv')
    grism_object_from_table = read_catalog(tempcat)
    assert isinstance(grism_object_from_table, list), "return grism objects were not a list"
