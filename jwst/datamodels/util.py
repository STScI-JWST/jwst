"""
Various utility functions and data types
"""
from __future__ import absolute_import, unicode_literals, division, print_function

import sys
from os.path import basename
import numpy as np
from astropy.extern import six
from astropy.io import fits

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def open(init=None, extensions=None, **kwargs):
    """
    Creates a DataModel from a number of different types

    Parameters
    ----------

    init : shape tuple, file path, file object, astropy.io.fits.HDUList,
           numpy array, dict, None

        - None: A default data model with no shape

        - shape tuple: Initialize with empty data of the given shape

        - file path: Initialize from the given file (FITS , JSON or ASDF)

        - readable file object: Initialize from the given file object

        - astropy.io.fits.HDUList: Initialize from the given
          `~astropy.io.fits.HDUList`

        - A numpy array: A new model with the data array initialized
          to what was passed in.

        - dict: The object model tree for the data model

    extensions : list of AsdfExtension
        A list of extensions to the ASDF to support when reading
        and writing ASDF files.

   Results
    -------

    model : DataModel instance
    """

    from . import model_base
    from . import _defined_models as defined_models # dict of model classes

    model_type = None

    # Get three special cases for opening a model out of the way
    # all three cases return a model if they match
    
    if init is None:
        return model_base.DataModel(None)
    # Send _asn.json files to ModelContainer; avoid shape "cleverness" below
    elif (isinstance(init, six.string_types) and
            basename(init).split('.')[0].split('_')[-1] == 'asn'):
        try:
            from . import container
            return container.ModelContainer(init, extensions=extensions, 
                **kwargs)
        except:
            raise TypeError(
                "init ASN not valid for ModelContainer"
                )
    elif isinstance(init, model_base.DataModel):
        # Copy the object so it knows not to close here
        return init.__class__(init)
    
    # Get the shape from the input argument where possible
    if isinstance(init, tuple):
        for item in init:
            if not isinstance(item, int):
                raise ValueError("shape must be a tuple of ints")
        shape = init
    elif isinstance(init, np.ndarray):
        shape = init.shape
    else:
        shape = ()

    # Get the list of hdus where possible
    if isinstance(init, (six.text_type, bytes)) or hasattr(init, "read"):
        hdulist = fits.open(init)
    elif isinstance(init, fits.HDUList):
        hdulist = init
    else:
        hdulist = {}
        
    # First try to get the class name from the primary header
    if hdulist:
        # Can also return None if no header keyword
        new_class = _class_from_model_type(hdulist)
    else:
        new_class = None

    # Get the class name from the shape and other header keywords
    if new_class is None:
        new_class = _class_from_shape(hdulist, shape)

    # Throw an error if these attempts were unsuccessful
    if new_class is None:
        raise TypeError("Can't determine datamodel class from argument to open")
        
    # Log a message about how the model was opened
    if isinstance(init, (six.text_type, bytes)):
        log.debug('Opening {0} as {1}'.format(basename(init), new_class))
    else:
        log.debug('Opening as {0}'.format(new_class))

    # Actually open the model
    model = new_class(init, extensions=extensions, **kwargs)
    return model


def _class_from_model_type(hdulist):
    """
    Get the model type from the primary header, lookup to get class
    """
    from . import _defined_models as defined_models

    try:
        primary = hdulist[0]
    except KeyError:
        model_type = None
    else:
        model_type = primary.header.get('DATAMODL')

    if model_type is None:
        new_class = None
    else:
        new_class = defined_models.get(model_type)

    return new_class


def _class_from_shape(hdulist, shape):
    """
    Get the class name from the shape and other header keywords
    """
    # If we do not have it, determine the shape from the science hdu
    if len(shape) == 0:
        try:
            hdu = hdulist[(fits_header_name('SCI'), 1)]
        except KeyError:
            pass
        else:
            if hasattr(hdu, 'shape'):
                shape = hdu.shape

    # Try to figure out which type to return, otherwise, just return a
    # new instance of the requested class
    if len(shape) == 0:
        from . import model_base
        new_class = model_base.DataModel
    elif len(shape) == 4:
        # It's a RampModel, MIRIRampModel, or QuadModel
        try:
            dqhdu = hdulist[fits_header_name('DQ')]
        except KeyError:
            # It's a RampModel or MIRIRampModel
            try:
                refouthdu = hdulist[fits_header_name('REFOUT')]
            except KeyError:
                # It's a RampModel
                from . import ramp
                new_class = ramp.RampModel
            else:
                # It's a MIRIRampModel
                from . import miri_ramp
                new_class = miri_ramp.MIRIRampModel
        else:
            # It's a QuadModel
            from . import quad
            new_class = quad.QuadModel
    elif len(shape) == 3:
        # It's a CubeModel
        from . import cube
        new_class = cube.CubeModel
    elif len(shape) == 2:
        try:
            hdu = hdulist[(fits_header_name('SCI'), 2)]
        except (KeyError, NameError):
            # It's an ImageModel
            from . import image
            new_class = image.ImageModel
        else:
            # It's a MultiSlitModel
            from . import multislit
            new_class = multislit.MultiSlitModel
    else:
        new_class = None
        
    return new_class


def can_broadcast(a, b):
    """
    Given two shapes, returns True if they are broadcastable.
    """
    for i in range(1, min(len(a), len(b)) + 1):
        adim = a[-i]
        bdim = b[-i]

        if not (adim == 1 or bdim == 1 or adim == bdim):
            return False

    return True


def to_camelcase(token):
    return ''.join(x.capitalize() for x in token.split('_-'))


if six.PY3:
    def fits_header_name(name):
        """
        Returns a FITS header name in the correct form for the current
        version of Python.
        """
        if isinstance(name, bytes):
            return name.decode('ascii')
        return name
else:
    def fits_header_name(name):
        """
        Returns a FITS header name in the correct form for the current
        version of Python.
        """
        if isinstance(name, unicode):
            return name.encode('ascii')
        return name


def gentle_asarray(a, dtype):
    """
    Performs an asarray that doesn't cause a copy if the byteorder is
    different.  It also ignores column name differences -- the
    resulting array will have the column names from the given dtype.
    """
    out_dtype = np.dtype(dtype)
    if isinstance(a, np.ndarray):
        in_dtype = a.dtype
        # Non-table array
        if in_dtype.fields is None and out_dtype.fields is None:
            if np.can_cast(in_dtype, out_dtype, 'equiv'):
                return a
            else:
                return np.asanyarray(a, dtype=out_dtype)
        elif in_dtype.fields is not None and out_dtype.fields is not None:
            if in_dtype == out_dtype:
                return a
            if len(in_dtype) != len(out_dtype):
                raise ValueError(
                    "Wrong number of columns.  Expected {0}, got {1}".format(
                        len(out_dtype), len(in_dtype)))
            new_dtype = []
            # Change the dtype name to match the fits record names
            # as the mismatch causes case insensitive access to fail
            if hasattr(in_dtype, 'names') and hasattr(out_dtype, 'names'):
                out_dtype.names = in_dtype.names
            for i in range(len(out_dtype.fields)):
                in_type = in_dtype[i]
                out_type = out_dtype[i]
                if in_type.subdtype is None:
                    type_str = in_type.str
                else:
                    type_str = in_type.subdtype[0].str
                if np.can_cast(in_type, out_type, 'equiv'):
                    new_dtype.append(
                        (out_dtype.names[i],
                         type_str,
                         in_type.shape))
                else:
                    return np.asanyarray(a, dtype=out_dtype)
            return a.view(dtype=np.dtype(new_dtype))
        else:
            return np.asanyarray(a, dtype=out_dtype)
    else:
        try:
            a = np.asarray(a, dtype=out_dtype)
        except:
            raise ValueError("Can't convert {0!s} to ndarray".format(type(a)))
        return a

def get_short_doc(schema):
    title = schema.get('title', None)
    description = schema.get('description', None)
    if description is None:
        description = title or ''
    else:
        if title is not None:
            description = title + '\n\n' + description
    return description.partition('\n')[0]


def ensure_ascii(s):
    if isinstance(s, six.text_type):
        s = s.encode('ascii', 'replace')
        if six.PY3:
            s = s.decode('ascii')
    return s
