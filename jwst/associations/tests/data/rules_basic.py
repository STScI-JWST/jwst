"""Association Rules: Basic
"""
import os
import sys

from jwst.associations import Association
from jwst.associations.registry import RegistryMarker
from jwst.associations.lib.constraint import ConstraintTrue


@RegistryMarker.rule
class Rule_1(Association):
    """Basic rule"""

    def __init__(self, version_id=None):
        self.constraints = ConstraintTrue()
        super(Rule_1, self).__init__(version_id=version_id)
        self.data['members'] = []

    def _add(self, item):
        self.data['members'].append(item)

    def is_item_member(self, item):
        """Check if item is already a member of this association

        Parameters
        ----------
        item: dict
            The item to add.

        Returns
        -------
        is_item_member: bool
            True if item is a member.
        """
        return item in self.data['members']
