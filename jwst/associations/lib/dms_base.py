"""Association attributes common to DMS-based Rules"""

from jwst.associations.exceptions import (
    AssociationNotValidError,
)
from jwst.associations.lib.acid import ACIDMixin

# DMS file name templates
_ASN_NAME_TEMPLATE_STAMP = 'jw{program}-{acid}_{stamp}_{type}_{sequence:03d}_asn'
_ASN_NAME_TEMPLATE = 'jw{program}-{acid}_{type}_{sequence:03d}_asn'

__all__ = ['DMSBaseMixin']


class DMSBaseMixin(ACIDMixin):
    """Association attributes common to DMS-based Rules"""

    @property
    def acid(self):
        """Association ID"""
        return self.acid_from_constraints()

    @property
    def asn_name(self):
        program = self.data['program']
        version_id = self.version_id
        asn_type = self.data['asn_type']
        sequence = self.sequence

        if version_id:
            name = _ASN_NAME_TEMPLATE_STAMP.format(
                program=program,
                acid=self.acid.id,
                stamp=version_id,
                type=asn_type,
                sequence=sequence,
            )
        else:
            name = _ASN_NAME_TEMPLATE.format(
                program=program,
                acid=self.acid.id,
                type=asn_type,
                sequence=sequence,
            )
        return name.lower()

    @property
    def validity(self):
        """Keeper of the validity tests"""
        try:
            validity = self._validity
        except AttributeError:
            self._validity = {}
            validity = self._validity
        return validity

    @validity.setter
    def validity(self, item):
        """Set validity dict"""
        self._validity = item

    @property
    def current_product(self):
        return self.data['products'][-1]

    def new_product(self, product_name=None):
        """Start a new product"""
        self.product_name = product_name
        product = {
            'name': self.product_name,
            'members': []
        }
        try:
            self.data['products'].append(product)
        except KeyError:
            self.data['products'] = [product]

    @property
    def product_name(self):
        if self._product_name is None:
            product_name = self.dms_product_name()
        else:
            product_name = self._product_name
        return product_name

    @product_name.setter
    def product_name(self, value):
        self._product_name = value

    def update_validity(self, entry):
        for test in self.validity.values():
            if not test['validated']:
                test['validated'] = test['check'](entry)

    @classmethod
    def validate(cls, asn):
        super(DMSBaseMixin, cls).validate(asn)

        if isinstance(asn, DMSBaseMixin):
            result = False
            try:
                result = all(
                    test['validated']
                    for test in asn.validity.values()
                )
            except (AttributeError, KeyError):
                raise AssociationNotValidError('Validation failed')
            if not result:
                raise AssociationNotValidError(
                    'Validation failed validity tests.'
                )

        return True
