from __future__ import absolute_import, unicode_literals, division, print_function

from . import model_base
from .drizproduct import DrizProductModel

__all__ = ['MultiProductModel']


class MultiProductModel(model_base.DataModel):
    """
    A data model for multi-DrizProduct images.

    This model has a special member `products` that can be used to
    deal with each DrizProduct at a time.  It behaves like a list::

       >>> multiprod_model.products.append(image_model)
       >>> multislit_model.products[0]
       <DrizProductModel>

    If `init` is a file name or an `DrizProductModel` instance, an empty
    `DrizProductModel` will be created and assigned to attribute `products[0]`,
    and the `data`, `wht`, `con`, and `relsens` attributes from the
    input file or `DrizProductModel` will be copied to the first element of
    `products`.

    Parameters
    ----------
    init : any
        Any of the initializers supported by `~jwst.datamodels.DataModel`.
    """
    schema_url = "multiproduct.schema.yaml"

    def __init__(self, init=None, **kwargs):
        if isinstance(init, DrizProductModel):
            super(MultiProductModel, self).__init__(init=None, **kwargs)
            self.update(init)
            self.products.append(self.products.item())
            self.products[0].data = init.data
            self.products[0].wht = init.wht
            self.products[0].con = init.con
            self.products[0].relsens = init.relsens
            return

        super(MultiProductModel, self).__init__(init=init, **kwargs)
