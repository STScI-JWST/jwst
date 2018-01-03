"""Association attributes common to DMS-based Rules"""
from .counter import Counter

from jwst.associations.lib.utilities import getattr_from_list
from jwst.associations.exceptions import (
    AssociationNotAConstraint,
    AssociationNotValidError,
)
from jwst.associations.lib.acid import ACIDMixin


# Default product name
PRODUCT_NAME_DEFAULT = 'undefined'

# DMS file name templates
_ASN_NAME_TEMPLATE_STAMP = 'jw{program}-{acid}_{stamp}_{type}_{sequence:03d}_asn'
_ASN_NAME_TEMPLATE = 'jw{program}-{acid}_{type}_{sequence:03d}_asn'

# Exposure EXP_TYPE to Association EXPTYPE mapping
EXPTYPE_MAP = {
    'mir_dark':      'dark',
    'mir_flatimage': 'flat',
    'mir_flatmrs':   'flat',
    'mir_tacq':      'target_acquistion',
    'nis_dark':      'dark',
    'nis_focus':     'engineering',
    'nis_lamp':      'engineering',
    'nis_tacq':      'target_acquistion',
    'nis_taconfirm': 'target_acquistion',
    'nrc_dark':      'dark',
    'nrc_flat':      'flat',
    'nrc_focus':     'engineering',
    'nrc_led':       'engineering',
    'nrc_tacq':      'target_acquistion',
    'nrc_taconfirm': 'target_acquistion',
    'nrs_autoflat':  'autoflat',
    'nrs_autowave':  'autowave',
    'nrs_confirm':   'target_acquistion',
    'nrs_dark':      'dark',
    'nrs_focus':     'engineering',
    'nrs_image':     'engineering',
    'nrs_lamp':      'engineering',
    'nrs_tacq':      'target_acquistion',
    'nrs_taconfirm': 'target_acquistion',
    'nrs_taslit':    'target_acquistion',
}

# Exposures that are always TSO
TSO_EXP_TYPES = (
    'mir_lrs-slitless',
    'nis_soss',
    'nrc_tsimage',
    'nrc_tsgrism',
    'nrs_brightobj'
)

# Exposures that get Level2b processing
IMAGE2_SCIENCE_EXP_TYPES = [
    'mir_image',
    'mir_lyot',
    'mir_4qpm',
    'nis_ami',
    'nis_image',
    'nrc_image',
    'nrc_coron',
    'nrc_tsimage',
]

IMAGE2_NONSCIENCE_EXP_TYPES = [
    'mir_coroncal',
    'mir_tacq',
    'nis_focus',
    'nis_tacq',
    'nis_taconfirm',
    'nrc_tacq',
    'nrc_taconfirm',
    'nrc_focus',
    'nrs_bota',
    'nrs_confirm',
    'nrs_focus',
    'nrs_image',
    'nrs_mimf',
    'nrs_taslit',
    'nrs_tacq',
    'nrs_taconfirm',
]

SPEC2_SCIENCE_EXP_TYPES = [
    'nrc_grism',
    'nrc_tsgrism',
    'mir_lrs-fixedslit',
    'mir_lrs-slitless',
    'mir_mrs',
    'nrs_fixedslit',
    'nrs_ifu',
    'nrs_msaspec',
    'nrs_brightobj',
    'nis_soss',
]

# Key that uniquely identfies members.
MEMBER_KEY = 'expname'

# Non-specified values found in DMS Association Pools
_EMPTY = (None, '', 'NULL', 'Null', 'null', '--', 'N', 'n', 'F', 'f')

# Degraded status information
_DEGRADED_STATUS_OK = (
    'No known degraded exposures in association.'
)
_DEGRADED_STATUS_NOTOK = (
    'One or more members have an error associated with them.'
    '\nDetails can be found in the member.exposerr attribute.'
)

__all__ = ['DMSBaseMixin']


class DMSBaseMixin(ACIDMixin):
    """Association attributes common to DMS-based Rules

    Attributes
    ----------
    from_items: [item[,...]]
        The list of items that contributed to the association.

    sequence: int
        The sequence number of the current association
    """

    # Associations of the same type are sequenced.
    _sequence = Counter(start=1)

    def __init__(self, *args, **kwargs):
        super(DMSBaseMixin, self).__init__(*args, **kwargs)

        self.from_items = []
        self.sequence = None
        if 'degraded_status' not in self.data:
            self.data['degraded_status'] = _DEGRADED_STATUS_OK
        if 'program' not in self.data:
            self.data['program'] = 'noprogram'

    @classmethod
    def create(cls, item, version_id=None):
        """Create association if item belongs

        Parameters
        ----------
        item: dict
            The item to initialize the association with.

        version_id: str or None
            Version_Id to use in the name of this association.
            If None, nothing is added.

        Returns
        -------
        (association, reprocess_list)
            2-tuple consisting of:
            - association: The association or, if the item does not
                this rule, None
            - [ProcessList[, ...]]: List of items to process again.
        """
        asn, reprocess = super(DMSBaseMixin, cls).create(item, version_id)
        if not asn:
            return None, reprocess
        asn.sequence = next(asn._sequence)
        return asn, reprocess

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
    def member_ids(self):
        """Set of all member ids in all products of this association"""
        member_ids = set(
            member[MEMBER_KEY]
            for product in self['products']
            for member in product['members']
        )
        return member_ids

    @property
    def current_product(self):
        return self.data['products'][-1]

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

    def new_product(self, product_name=PRODUCT_NAME_DEFAULT):
        """Start a new product"""
        product = {
            'name': product_name,
            'members': []
        }
        try:
            self.data['products'].append(product)
        except KeyError:
            self.data['products'] = [product]

    def update_asn(self, item=None, member=None):
        """Update association meta information

        Parameters
        ----------
        item: dict or None
            Item to use as a source. If not given, item-specific
            information will be left unchanged.

        member: dict or None
            An association member to use as source.
            If not given, member-specific information will be update
            from current association/product membership.

        Notes
        -----
        If both `item` and `member` are given,
        information in `member` will take precedence.
        """
        self.update_degraded_status()

    def update_degraded_status(self):
        """Update association degraded status"""

        if self.data['degraded_status'] == _DEGRADED_STATUS_OK:
            for product in self.data['products']:
                for member in product['members']:
                    try:
                        exposerr = member['exposerr']
                    except KeyError:
                        continue
                    else:
                        if exposerr not in _EMPTY:
                            self.data['degraded_status'] = _DEGRADED_STATUS_NOTOK
                            break

    def update_validity(self, entry):
        for test in self.validity.values():
            if not test['validated']:
                test['validated'] = test['check'](entry)

    @classmethod
    def reset_sequence(cls):
        cls._sequence = Counter(start=1)

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

    def get_exposure_type(self, item, default='science'):
        """Determine the exposure type of a pool item

        Parameters
        ----------
        item: dict
            The pool entry to determine the exposure type of

        default: str or None
            The default exposure type.
            If None, routine will raise LookupError

        Returns
        -------
        exposure_type: str
            Exposure type. Can be one of
                'science': Item contains science data
                'target_aquisition': Item contains target acquisition data.
                'autoflat': NIRSpec AUTOFLAT
                'autowave': NIRSpec AUTOWAVE
                'psf': PSF
                'imprint': MSA/IFU Imprint/Leakcal

        Raises
        ------
        LookupError
            When `default` is None and an exposure type cannot be determined
        """
        result = default

        # Look for specific attributes
        try:
            self.item_getattr(item, ['is_psf'])
        except KeyError:
            pass
        else:
            return 'psf'
        try:
            self.item_getattr(item, ['is_imprt'])
        except KeyError:
            pass
        else:
            return 'imprint'
        try:
            self.item_getattr(item, ['bkgdtarg'])
        except KeyError:
            pass
        else:
            return 'background'

        # Base type off of exposure type.
        try:
            exp_type = item['exp_type']
        except KeyError:
            raise LookupError('Exposure type cannot be determined')

        result = EXPTYPE_MAP.get(exp_type, default)

        if result is None:
            raise LookupError('Cannot determine exposure type')
        return result

    def item_getattr(self, item, attributes):
        """Return value from any of a list of attributes

        Parameters
        ----------
        item: dict
            item to retrieve from

        attributes: list
            List of attributes

        Returns
        -------
        (attribute, value)
            Returns the value and the attribute from
            which the value was taken.

        Raises
        ------
        KeyError
            None of the attributes are found in the dict.
        """
        return getattr_from_list(
            item,
            attributes,
            invalid_values=self.INVALID_VALUES
        )

    def is_member(self, new_member):
        """Check if member is already a member

        Parameters
        ----------
        new_member: dict
            The member to check for
        """
        try:
            current_members = self.current_product['members']
        except KeyError:
            return False

        for member in current_members:
            if member == new_member:
                return True
        return False

    def is_item_member(self, item):
        """Check if item is already a member of this association

        Parameters
        ----------
        item: dict
            The item to check for.

        Returns
        -------
        is_item_member: bool
            True if item is a member.
        """
        member = self.make_member(item)
        return self.is_member(member)

    def _get_target(self):
        """Get string representation of the target

        Returns
        -------
        target: str
            The Level3 Product name representation
            of the target or source ID.
        """
        target_id = format_list(self.constraints['target'].found_values)
        target = 't{0:0>3s}'.format(str(target_id))
        return target

    def _get_instrument(self):
        """Get string representation of the instrument

        Returns
        -------
        instrument: str
            The Level3 Product name representation
            of the instrument
        """
        instrument = format_list(self.constraints['instrument'].found_values)
        return instrument

    def _get_opt_element(self):
        """Get string representation of the optical elements

        Returns
        -------
        opt_elem: str
            The Level3 Product name representation
            of the optical elements.
        """
        opt_elem = ''
        join_char = ''
        try:
            value = format_list(self.constraints['opt_elem'].found_values)
        except KeyError:
            pass
        else:
            if value not in _EMPTY and value != 'clear':
                opt_elem = value
                join_char = '-'
        try:
            value = format_list(self.constraints['opt_elem2'].found_values)
        except KeyError:
            pass
        else:
            if value not in _EMPTY and value != 'clear':
                opt_elem = join_char.join(
                    [opt_elem, value]
                )
        if opt_elem == '':
            opt_elem = 'clear'
        return opt_elem

    def _get_exposure(self):
        """Get string representation of the exposure id

        Returns
        -------
        exposure: str
            The Level3 Product name representation
            of the exposure & activity id.

        Raises
        ------
        AssociationNotAConstraint
            No constraints produce this value
        """
        try:
            activity_id = format_list(
                self.constraints['activity_id'].found_values
            )
        except KeyError:
            raise AssociationNotAConstraint
        else:
            if activity_id not in _EMPTY:
                exposure = '{0:0>2s}'.format(activity_id)
        return exposure

# #########
# Utilities
# #########


def format_list(alist):
    """Format a list according to DMS naming specs"""
    return '-'.join(alist)
