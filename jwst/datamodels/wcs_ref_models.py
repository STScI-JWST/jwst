from __future__ import absolute_import, unicode_literals, division, print_function
import numpy as np
from astropy.modeling.core import Model
from astropy import units as u
from . import model_base

from .extension import BaseExtension
from jwst.transforms.jwextension import JWSTExtension
from gwcs.extension import GWCSExtension


jwst_extensions = [GWCSExtension(), JWSTExtension(), BaseExtension()]


__all__ = ['DistortionModel', 'DistortionMRSModel', 'SpecwcsModel', 'RegionsModel',
           'WavelengthrangeModel', 'CameraModel', 'CollimatorModel', 'OTEModel',
           'FOREModel', "FPAModel", 'IFUPostModel', 'IFUFOREModel', 'IFUSlicerModel',
           'MSAModel', 'FilteroffsetModel', 'DisperserModel']


class _SimpleModel(model_base.DataModel):
    """
    A model for a reference file of type "distortion".
    """
    schema_url = None
    reftype = None

    def __init__(self, init=None, model=None, input_units=None, output_units=None, **kwargs):

        super(_SimpleModel, self).__init__(init=init, **kwargs)
        if model is not None:
            self.model = model
        if input_units is not None:
            self.meta.input_units = input_units
        if output_units is not None:
            self.meta.output_units = output_units
        if init is None:
            try:
                self.populate_meta()
            except NotImplementedError:
                pass

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def populate_meta(self):
        """
        Subclasses can overwrite this to populate specific meta keywords.
        """
        raise NotImplementedError

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")

    def validate(self):
        assert isinstance(self.model, Model)
        assert isinstance(self.meta.input_units, (str, u.NamedUnit))
        assert isinstance(self.meta.output_units, (str, u.NamedUnit))
        assert self.meta.instrument.name in ["NIRCAM", "NIRSPEC", "MIRI", "TFI", "FGS", "NIRISS"]
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class DistortionModel(_SimpleModel):
    """
    A model for a reference file of type "distortion".
    """
    schema_url = "distortion.schema.yaml"
    reftype = "distortion"

    def validate(self):
        super(DistortionModel, self).validate()
        if self.meta.instrument.name == 'NIRCAM':
            assert self.meta.instrument.module is not None
            assert self.meta.instrument.channel is not None
            assert self.meta.instrument.p_pupil is not None


class DistortionMRSModel(model_base.DataModel):
    """
    A model for a reference file of type "distortion" for the MIRI MRS.
    """
    schema_url = "distortion_mrs.schema.yaml"
    reftype = "distortion"

    def __init__(self, init=None, x_model=None, y_model=None, alpha_model=None, beta_model=None,
                 bzero=None, bdel=None, input_units=None, output_units=None, **kwargs):

        super(DistortionMRSModel, self).__init__(init=init, **kwargs)

        if x_model is not None:
            self.x_model = x_model
        if y_model is not None:
            self.y_model = y_model
        if alpha_model is not None:
            self.alpha_model = alpha_model
        if beta_model is not None:
            self.beta_model = beta_model
        if bzero is not None:
            self.bzero = bzero
        if bdel is not None:
            self.bdel = bdel
        if input_units is not None:
            self.meta.input_units = input_units
        if output_units is not None:
            self.meta.output_units = output_units
        if init is None:
            try:
                self.populate_meta()
            except NotImplementedError:
                pass

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def populate_meta(self):
        self.meta.instrument.name = "MIRI"
        self.meta.exposure.type = "MIR_MRS"
        self.meta.input_units = u.pix
        self.meta.output_units = u.arcsec

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")

    def validate(self):
        assert isinstance(self.meta.input_units, (str, u.NamedUnit))
        assert isinstance(self.meta.output_units, (str, u.NamedUnit))
        assert self.meta.instrument.name == "MIRI"
        assert self.meta.exposure.type == "MIR_MRS"
        assert self.meta.instrument.channel in ("12", "34", "1", "2", "3", "4")
        assert self.meta.instrument.band in ("SHORT", "LONG", "MEDIUM")
        assert self.meta.instrument.detector in ("MIRIFUSHORT", "MIRIFULONG")
        assert all([isinstance(m, Model) for m in self.x_model])
        assert all([isinstance(m, Model) for m in self.y_model])
        assert all([isinstance(m, Model) for m in self.alpha_model])
        assert all([isinstance(m, Model) for m in self.beta_model])
        assert len(self.abv2v3_model.model) == 2
        assert len(self.abv2v3_model.channel_band) == 2
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class SpecwcsModel(_SimpleModel):
    """
    A model for a reference file of type "specwcs".
    """
    schema_url = "specwcs.schema.yaml"
    reftype = "specwcs"

    def validate(self):
        assert isinstance(self.meta.input_units, (str, u.NamedUnit))
        assert isinstance(self.meta.output_units, (str, u.NamedUnit))
        assert self.meta.instrument.name in ["NIRCAM", "NIRSPEC", "MIRI", "TFI", "FGS", "NIRISS"]
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class RegionsModel(model_base.DataModel):
    """
    A model for a reference file of type "regions".
    """
    schema_url = "regions.schema.yaml"
    reftype = "regions"

    def __init__(self, init=None, regions=None, **kwargs):
        super(RegionsModel, self).__init__(init=init, **kwargs)
        if regions is not None:
            self.regions = regions

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")

    def validate(self):
        assert isinstance(self.regions.copy(), np.ndarray)
        assert self.meta.instrument.name == "MIRI"
        assert self.meta.exposure.type == "MIR_MRS"
        assert self.meta.instrument.channel in ("12", "34", "1", "2", "3", "4")
        assert self.meta.instrument.band in ("SHORT", "LONG")
        assert self.meta.instrument.detector in ("MIRIFUSHORT", "MIRIFULONG")
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class WavelengthrangeModel(model_base.DataModel):
    """
    A model for a reference file of type "wavelenghrange".
    """
    schema_url = "wavelengthrange.schema.yaml"
    reftype = "wavelengthrange"

    def __init__(self, init=None, channels=None, wrange=None, order=None, wunits=None, **kwargs):

        super(WavelengthrangeModel, self).__init__(init=init, **kwargs)
        if channels is not None:
            self.channels = channels
        if wrange is not None:
            self.wavelengthrange = wrange
        if order is not None:
            self.order = order
        if wunits is not None:
            self.meta.wavelength_units = wunits

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")

    def validate(self):
        assert self.meta.instrument.name in ("MIRI", "NIRSPEC")
        assert self.meta.exposure.type in ("MIR_MRS", "NRS_AUTOFLAT", "NRS_AUTOWAVE", "NRS_BOTA",
                                           "NRS_BRIGHTOBJ", "NRS_CONFIRM", "NRS_DARK", "NRS_FIXEDSLIT",
                                           "NRS_FOCUS", "NRS_IFU", "NRS_IMAGE", "NRS_LAMP", "NRS_MIMF",
                                           "NRS_MSASPEC", "NRS_TACONFIRM", "NRS_TACQ", "NRS_TASLIT", "N/A",
                                           "ANY")
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class FPAModel(model_base.DataModel):
    """
    A model for a NIRSPEC reference file of type "fpa".
    """
    schema_url = "fpa.schema.yaml"
    reftype = "fpa"

    def __init__(self, init=None, nrs1_model=None, nrs2_model=None, **kwargs):

        super(FPAModel, self).__init__(init=init, **kwargs)
        if nrs1_model is not None:
            self.nrs1_model = nrs1_model
        if nrs2_model is not None:
            self.nrs2_model = nrs2_model
        if init is None:
            self.populate_meta()

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"
        self.meta.exposure.p_exptype = "NRS_TACQ|NRS_TASLIT|NRS_TACONFIRM|\
        NRS_CONFIRM|NRS_FIXEDSLIT|NRS_IFU|NRS_MSASPEC|NRS_IMAGE|NRS_FOCUS|\
        NRS_MIMF|NRS_BOTA|NRS_LAMP|NRS_BRIGHTOBJ|"

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")

    def validate(self):
        assert isinstance(self.nrs1_model, Model)
        assert isinstance(self.nrs2_model, Model)
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class IFUPostModel(model_base.DataModel):
    """
    A model for a NIRSPEC reference file of type "ifupost".
    """
    schema_url = "ifupost.schema.yaml"
    reftype = "ifupost"

    def __init__(self, init=None, models=None, **kwargs):

        super(IFUPostModel, self).__init__(init=init, **kwargs)
        if models is not None:
            if len(models) != 30:
                raise ValueError("Expected 30 slice models, got {0}".format(len(models)))
            else:
                for i, m in enumerate(models):
                    setattr(self, "slice_{0]".format(i), m)

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"
        self.meta.exposure.type = "NRS_IFU"

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")

    def validate(self):
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class IFUSlicerModel(model_base.DataModel):
    """
    A model for a NIRSPEC reference file of type "ifuslicer".
    """
    schema_url = "ifuslicer.schema.yaml"
    reftype = "ifuslicer"

    def __init__(self, init=None, model=None, data=None, **kwargs):

        super(IFUSlicerModel, self).__init__(init=init, **kwargs)
        if model is not None:
            self.model = model
        if data is not None:
            seld.data = data

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"
        self.meta.exposure.type = "NRS_IFU"

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")

    def validate(self):
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class MSAModel(model_base.DataModel):
    """
    A model for a NIRSPEC reference file of type "msa".
    """
    schema_url = "msa.schema.yaml"
    reftype = "msa"

    def __init__(self, init=None, models=None, data=None, **kwargs):
        super(MSAModel, self).__init__(init=init, **kwargs)
        if models is not None and data is not None:
            self.Q1 = {'model': models['Q1'], 'data': data['Q1']}
            self.Q2 = {'model': models['Q2'], 'data': data['Q2']}
            self.Q3 = {'model': models['Q3'], 'data': data['Q3']}
            self.Q4 = {'model': models['Q4'], 'data': data['Q4']}
            self.Q5 = {'model': models['Q5'], 'data': data['Q5']}

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"
        self.meta.exposure.p_exptype = "NRS_TACQ|NRS_TASLIT|NRS_TACONFIRM|\
        NRS_CONFIRM|NRS_FIXEDSLIT|NRS_IFU|NRS_MSASPEC|NRS_IMAGE|NRS_FOCUS|\
        NRS_MIMF|NRS_BOTA|NRS_LAMP|NRS_BRIGHTOBJ|"

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")

    def validate(self):
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class DisperserModel(model_base.DataModel):
    """
    A model for a NIRSPEC reference file of type "disperser".
    """
    schema_url = "disperser.schema.yaml"
    reftype = "disperser"

    def __init__(self, init=None, angle=None, gwa_tiltx=None, gwa_tilty=None,
                 kcoef=None, lcoef=None, tcoef=None, pref=None, tref=None,
                 theta_x=None, theta_y=None,theta_z=None, tilt_x=None, tilt_y=None,
                 **kwargs):
        super(DisperserModel, self).__init__(init=init, **kwargs)
        if angle is not None:
            self.angle = angle
        if gwa_tiltx is not None:
            gwa_tiltx = gwa_tiltx
        if gwa_tilty is not None:
            gwa_tilty = gwa_tilty
        if kcoef is not None:
            self.kcoef = kcoef
        if lcoef is not None:
            self.lcoef = lcoef
        if tcoef is not None:
            self.tcoef = tcoef
        if pref is not None:
            self.pref = pref
        if tref is not None:
            self.tref = tref
        if theta_x is not None:
            self.theta_x = theta_x
        if theta_y is not None:
            self.theta_y = theta_y
        if theta_z is not None:
            self.theta_z = theta_z
        if tilt_x is not None:
            self.tilt_x = tilt_x
        if tilt_y is not None:
            self.tilt_y = tilt_y

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"
        self.meta.exposure.p_exptype = "NRS_TACQ|NRS_TASLIT|NRS_TACONFIRM|\
        NRS_CONFIRM|NRS_FIXEDSLIT|NRS_IFU|NRS_MSASPEC|NRS_IMAGE|NRS_FOCUS|\
        NRS_MIMF|NRS_BOTA|NRS_LAMP|NRS_BRIGHTOBJ|"
        self.meta.instrument.p_grating = "G140M|G235M|G395M|G140H|G235H|G395H|PRISM|MIRROR|"

    def to_fits(self):
        raise NotImplementedError("FITS format is not supported for this file.")

    def on_save(self, path=None):
        self.meta.reftype = self.reftype

    def validate(self):
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class FilteroffsetModel(model_base.DataModel):
    """
    A model for a NIRSPEC reference file of type "disperser".
    """
    schema_url = "filteroffset.schema.yaml"
    reftype = "filteroffset"

    def __init__(self, init=None, filters=None, **kwargs):
        super(FilteroffsetModel, self).__init__(init, **kwargs)
        if filters is not None:
            self.filters = filters

    def populate_meta(self):
        self.meta.instrument.name = "MIRI"
        self.meta.instrument.detector = "MIRIMAGE"
        self.meta.instrument.pfilter = "F1130W|F1140C|F2300C|F2100W|F1800W|\
        F1550C|F560W|F2550WR|FND|F2550W|F1500W|F1000W|F1065C|F770W|F1280W|"

    def on_save(self, path=None):
        self.meta.reftype = self.reftype
        self.meta.telescope = self.meta.telescope

    def validate(self):
        assert self.meta.instrument.name == "MIRI"
        assert self.meta.instrument.detector == "MIRIMAGE"
        assert self.meta.description is not None
        assert self.meta.telescope is not None
        assert self.meta.reftype is not None
        assert self.meta.author is not None
        assert self.meta.pedigree is not None


class IFUFOREModel(_SimpleModel):
    """
    A model for a NIRSPEC reference file of type "ifufore".
    """
    schema_url = "ifufore.schema.yaml"
    reftype = "ifufore"

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"
        self.meta.exposure.type = "NRS_IFU"


class CameraModel(_SimpleModel):
    """
    A model for a reference file of type "camera".
    """
    schema_url = "camera.schema.yaml"
    reftype = 'camera'

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"
        self.meta.exposure.p_exptype = "NRS_TACQ|NRS_TASLIT|NRS_TACONFIRM|\
        NRS_CONFIRM|NRS_FIXEDSLIT|NRS_IFU|NRS_MSASPEC|NRS_IMAGE|NRS_FOCUS|\
        NRS_MIMF|NRS_BOTA|NRS_LAMP|NRS_BRAIGHTOBJ|"


class CollimatorModel(_SimpleModel):
    """
    A model for a reference file of type "collimator".
    """
    schema_url = "collimator.schema.yaml"
    reftype = 'collimator'

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"


class OTEModel(_SimpleModel):
    """
    A model for a reference file of type "ote".
    """
    schema_url = "ote.schema.yaml"
    reftype = 'ote'

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"


class FOREModel(_SimpleModel):
    """
    A model for a reference file of type "fore".
    """
    schema_url = "fore.schema.yaml"
    reftype = 'fore'

    def populate_meta(self):
        self.meta.instrument.name = "NIRSPEC"
        self.meta.instrument.p_detector = "NRS1|NRS2|"
        self.meta.instrument.p_filter = "CLEAR|F070LP|F100LP|F110W|F140X|F170LP|F290LP|"
