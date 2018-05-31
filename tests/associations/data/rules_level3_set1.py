"""Association Definitions: DMS Level3 product associations
"""
import os
import sys

from jwst.associations import Association

# The schema that these associations must adhere to.
_ASN_SCHEMA_LEVEL3 = 'asn_schema_jw_level3.json'
_DMS_POOLNAME_REGEX = 'jw(\d{5})_(\d{8}[Tt]\d{6})_pool'


class DMS_Level3_Base_Set1(Association):
    """Basic class for DMS Level3 associations."""


class Asn_Dither_Set1(DMS_Level3_Base_Set1):
    """Non-Association Candidate Dither Associations"""


class Asn_WFS_Set1(DMS_Level3_Base_Set1):
    """Wavefront Sensing association"""
