How To Create Reference files in ASDF format
============================================

All WCS reference files are in `ASDF <http://asdf-standard.readthedocs.org/en/latest/>`__  format.
ASDF is a human-readable, hierarchical metadata structure, made up of basic dynamic data
types such as strings, numbers, lists and mappings. Data is saved as binary arrays. It is
primarily intended as an interchange format for delivering products from
instruments to scientists or between scientists. It's based on YAML and JSON schema and as such
provides automatic structure and metadata validation.

While it is possible to write or edit an ASDF file in a text editor, the best way to create
reference files is using the python implementation of the format
`asdf <http://asdf.readthedocs.io/en/latest/>`__ and
`astropy.modeling <http://astropy.readthedocs.org/en/latest/modeling/index.html>`__ .
There are two steps in this process:

- create a transform using the simple models and the rules to combine them
- save the transform to an ASDF file (this automatically validates it)

The rest of this document provides a brief description and examples of models in
`astropy.modeling <http://astropy.readthedocs.org/en/latest/modeling/index.html>`__
which are most relevant to WCS and examples of creating WCS reference files.

Create a transform
------------------

`astropy.modeling <http://astropy.readthedocs.org/en/latest/modeling/index.html>`__
is a framework for representing, evaluating and fitting models. All available
models can be imported from the ``models`` module.

>>> from astropy.modeling import models as astmodels

If necessary all fitters can be imported through the ``fitting`` module.

>>> from astropy.modeling import fitting

Many analytical models are already implemented and it is
easy to implement new ones. Models are initialized with their parameter values.
They are evaluated by passing the inputs directly, similar
to the way functions are called. For example,

>>> poly_x = astmodels.Polynomial2D(degree=2, c0_0=.2, c1_0=.11, c2_0=2.3, c0_1=.43, c0_2=.1, c1_1=.5)
>>> poly_x(1, 1)
    3.639999

Models have their analytical inverse defined if it exists and accessible through the ``inverse`` property.
An inverse model can also be (re)defined by assigning to the ``inverse`` property.

>>> rotation = astmodels.Rotation2D(angle=23.4)
>>> rotation.inverse
    <Rotation2D(angle=-23.4)>
>>> poly_x.inverse = astmodels.Polynomial2D(degree=3, **coeffs)

astropy.modeling also provides the means to combine models in various ways.

**Model concatenation** uses the ``&`` operator. Models are evaluated on independent
inputs and results are concatenated. The total number of inputs must be equal to the
sum of the number of inputs of all models.

>>> shift_x = astmodels.Shift(-34.2)
>>> shift_y = astmodels.Shift(-120)
>>> model = shift_x & shift_y
>>> model(1, 1)
    (-33.2, -119.0)

**Model composition** uses the ``|`` operator. The output of one model is passed
as input to the next one, so the number of outputs of one model must be equal to the number
of inputs to the next one.

>>> model = poly_x | shift_x | scale_x
>>> model = shift_x & shift_y | poly_x

Two models, ``Mapping`` and ``Identity``, are useful for axes manipulation - dropping
or creating axes, or switching the order of the inputs.

``Mapping`` takes a tuple of integers and an optional number of inputs. The tuple
represents indices into the inputs. For example, to represent a 2D Polynomial distortion
in ``x`` and ``y``, preceded by a shift in both axes:

>>> poly_y = astmodels.Polynomial2D(degree=2, c0_0=.2, c1_0=1.1, c2_0=.023, c0_1=3, c0_2=.01, c1_1=2.2)
>>> model = shift_x & shift_y | astmodels.Mapping((0, 1, 0, 1)) | poly_x & poly_y
>>> model(1, 1)
    (5872.03, 29242.892)

``Identity`` takes an integer which represents the number of inputs to be passed unchanged.
This can be useful when one of the inputs does not need more processing. As an example,
two spatial (V2V3) and one spectral (wavelength) inputs are passed to a composite model which
transforms the spatial coordinates to celestial coordinates and needs to pass the wavelength unchanged.

>>> tan = astmodels.Pix2Sky_TAN()
>>> model = tan & astmodels.Identity(1)
>>> model(0.2, 0.3, 10**-6)
   (146.30993247402023, 89.63944963170002, 1e-06)

**Arithmetic Operators** can be used to combine models. In this case each model is evaluated
with all inputs and the operator is applied to the results, e.g. ``model = m1 + m2 * m3 – m4/m5**m6``

>>> model = shift_x + shift_y
>>> model(1, 1)
    -152.2


Save a transform to an ASDF file
--------------------------------

`asdf <http://asdf.readthedocs.io/en/latest/>`__ is used to read and write reference files in
`ASDF <http://asdf-standard.readthedocs.org/en/latest/>`__ format. Once the model is create using the rules in the above section, it needs to be assigned
to the ASDF tree.

>>> from asdf import AsdfFile
>>> f = AsdfFile()
>>> f.tree['model'] = model
>>> f.write_to('reffile.asdf')

The ``write_to`` command validates the file and writes it to disk. It will catch any errors due to
inconsistent inputs/outputs or invalid parameters.

To test the file, it can be read in again using the ``AsdfFile.open()`` method:

>>> ff = AsdfFile.open('reffile.asdf')
>>> model = ff.tree['model']
>>> model(1, 1)
    -152.2


