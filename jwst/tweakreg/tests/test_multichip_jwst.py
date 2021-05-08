import copy
import asdf

import numpy as np
from astropy.io import fits
from astropy import table
from astropy import wcs as fitswcs
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling import polynomial
from astropy.modeling.models import (
    Shift, AffineTransformation2D, Pix2Sky_TAN, RotateNative2Celestial,
    Identity, Mapping, Const1D, Scale
)
from astropy import units as u
from astropy import coordinates as coord

import tweakwcs
from tweakwcs.imalign import align_wcs

from jwst.datamodels import ImageModel, ModelContainer
from jwst.tweakreg import tweakreg_step

import gwcs
from gwcs import coordinate_frames as cf
from gwcs.geometry import SphericalToCartesian, CartesianToSpherical

_REF_RMSE_RA = 3e-9
_REF_RMSE_DEC = 3e-10
_RAD2ARCSEC = 3600.0 * np.rad2deg(1.0)
_ARCSEC2RAD = 1.0 / _RAD2ARCSEC


def _make_gwcs_wcs(fits_hdr):
    hdr = fits.Header.fromfile(get_pkg_data_filename(fits_hdr))
    fw = fitswcs.WCS(hdr)

    a_order = hdr['A_ORDER']
    a_coeff = {}
    for i in range(a_order + 1):
        for j in range(a_order + 1 - i):
            key = 'A_{:d}_{:d}'.format(i, j)
            if key in hdr:
                a_coeff[key] = hdr[key]

    b_order = hdr['B_ORDER']
    b_coeff = {}
    for i in range(b_order + 1):
        for j in range(b_order + 1 - i):
            key = 'B_{:d}_{:d}'.format(i, j)
            if key in hdr:
                b_coeff[key] = hdr[key]

    distortion = polynomial.SIP(
        fw.wcs.crpix,
        fw.sip.a_order,
        fw.sip.b_order,
        a_coeff,
        b_coeff
    ) + Identity(2)

    unit_conv = Scale(1.0 / 3600.0, name='arcsec_to_deg_1D')
    unit_conv = unit_conv & unit_conv
    unit_conv.name = 'arcsec_to_deg_2D'

    unit_conv_inv = Scale(3600.0, name='deg_to_arcsec_1D')
    unit_conv_inv = unit_conv_inv & unit_conv_inv
    unit_conv_inv.name = 'deg_to_arcsec_2D'

    c2s = CartesianToSpherical(name='c2s', wrap_lon_at=180)
    s2c = SphericalToCartesian(name='s2c', wrap_lon_at=180)
    c2tan = ((Mapping((0, 1, 2), name='xyz') /
              Mapping((0, 0, 0), n_inputs=3, name='xxx')) |
             Mapping((1, 2), name='xtyt'))
    c2tan.name = 'Cartesian 3D to TAN'

    tan2c = (Mapping((0, 0, 1), n_inputs=2, name='xtyt2xyz') |
             (Const1D(1, name='one') & Identity(2, name='I(2D)')))
    tan2c.name = 'TAN to cartesian 3D'

    tan2c.inverse = c2tan
    c2tan.inverse = tan2c

    aff = AffineTransformation2D(matrix=fw.wcs.cd)

    offx = Shift(-fw.wcs.crpix[0])
    offy = Shift(-fw.wcs.crpix[1])

    s = 5e-6
    scale = Scale(s) & Scale(s)

    distortion |= (offx & offy) | scale | tan2c | c2s | unit_conv_inv

    taninv = s2c | c2tan
    tan = Pix2Sky_TAN()
    n2c = RotateNative2Celestial(fw.wcs.crval[0], fw.wcs.crval[1], 180)
    wcslin = unit_conv | taninv | scale.inverse | aff | tan | n2c

    sky_frm = cf.CelestialFrame(reference_frame=coord.ICRS())
    det_frm = cf.Frame2D(name='detector')
    v2v3_frm = cf.Frame2D(
        name="v2v3",
        unit=(u.arcsec, u.arcsec),
        axes_names=('x', 'y'),
        axes_order=(0, 1)
    )
    pipeline = [(det_frm, distortion), (v2v3_frm, wcslin), (sky_frm, None)]

    gw = gwcs.WCS(input_frame=det_frm, output_frame=sky_frm,
                  forward_transform=pipeline)
    gw.crpix = fw.wcs.crpix
    gw.crval = fw.wcs.crval

    # sanity check:
    for _ in range(100):
        x = np.random.randint(1, fw.pixel_shape[0])
        y = np.random.randint(1, fw.pixel_shape[0])
        assert np.allclose(gw(x, y), fw.all_pix2world(x, y, 1),
                           rtol=0, atol=1e-11)

    return gw


def _make_reference_gwcs_wcs(fits_hdr):
    hdr = fits.Header.fromfile(get_pkg_data_filename(fits_hdr))
    fw = fitswcs.WCS(hdr)

    unit_conv = Scale(1.0 / 3600.0, name='arcsec_to_deg_1D')
    unit_conv = unit_conv & unit_conv
    unit_conv.name = 'arcsec_to_deg_2D'

    unit_conv_inv = Scale(3600.0, name='deg_to_arcsec_1D')
    unit_conv_inv = unit_conv_inv & unit_conv_inv
    unit_conv_inv.name = 'deg_to_arcsec_2D'

    c2s = CartesianToSpherical(name='c2s', wrap_lon_at=180)
    s2c = SphericalToCartesian(name='s2c', wrap_lon_at=180)
    c2tan = ((Mapping((0, 1, 2), name='xyz') /
              Mapping((0, 0, 0), n_inputs=3, name='xxx')) |
             Mapping((1, 2), name='xtyt'))
    c2tan.name = 'Cartesian 3D to TAN'

    tan2c = (Mapping((0, 0, 1), n_inputs=2, name='xtyt2xyz') |
             (Const1D(1, name='one') & Identity(2, name='I(2D)')))
    tan2c.name = 'TAN to cartesian 3D'

    tan2c.inverse = c2tan
    c2tan.inverse = tan2c

    aff = AffineTransformation2D(matrix=fw.wcs.cd)

    offx = Shift(-fw.wcs.crpix[0])
    offy = Shift(-fw.wcs.crpix[1])

    s = 5e-6
    scale = Scale(s) & Scale(s)

    det2tan = (offx & offy) | scale | tan2c | c2s | unit_conv_inv

    taninv = s2c | c2tan
    tan = Pix2Sky_TAN()
    n2c = RotateNative2Celestial(fw.wcs.crval[0], fw.wcs.crval[1], 180)
    wcslin = unit_conv | taninv | scale.inverse | aff | tan | n2c

    sky_frm = cf.CelestialFrame(reference_frame=coord.ICRS())
    det_frm = cf.Frame2D(name='detector')
    v2v3_frm = cf.Frame2D(
        name="v2v3",
        unit=(u.arcsec, u.arcsec),
        axes_names=('x', 'y'),
        axes_order=(0, 1)
    )
    pipeline = [(det_frm, det2tan), (v2v3_frm, wcslin), (sky_frm, None)]

    gw = gwcs.WCS(input_frame=det_frm, output_frame=sky_frm,
                  forward_transform=pipeline)
    gw.crpix = fw.wcs.crpix
    gw.crval = fw.wcs.crval

    return gw


def _match(x, y):
    lenx = len(x)
    leny = len(y)
    if lenx == leny:
        return (np.arange(lenx), np.arange(leny))
    elif lenx < leny:
        lenx, leny = leny, lenx
        x, y = y, x
    match = (np.arange(leny) + (0 if y.meta['name'] == 'ext1' else leny),
             np.arange(leny))
    return match


def _make_tweakreg_catalog(model, *args, **kwargs):
    return model.tweakreg_catalog


tweakreg_step.make_tweakreg_catalog = _make_tweakreg_catalog


def _align_wcs(imcats, **kwargs):
    new_kwargs = { k: v for k, v in kwargs.items() if k != 'match'}
    new_kwargs['match'] = _match
    return align_wcs(imcats, **new_kwargs)


tweakreg_step.align_wcs = _align_wcs


class _CustomImageModel(ImageModel):
    @staticmethod
    def clone(target, source, deepcopy=False, memo=None):
        if deepcopy:
            instance = copy.deepcopy(source._instance, memo=memo)
            target._asdf = asdf.AsdfFile()  # avoid crash due to unserializable SIP
            target._instance = instance
            target._iscopy = source._iscopy
        else:
            target._asdf = source._asdf
            target._instance = source._instance
            target._iscopy = True

        target._files_to_close = []
        target._shape = source._shape
        target._ctx = target
        target._no_asdf_extension = source._no_asdf_extension


def test_multichip_jwst_alignment():
    # this test is fundamentally equivalent to test_multichip_alignment_step()
    # with the following differences:
    # 1. test_multichip_alignment_step() test includes parts of the JWST
    #    pipeline step itself;
    # 2. test_multichip_alignment_step() does not have access to 'fit_info'
    #    in the meta data and so test_multichip_jwst_alignment() can test
    #    the fit more extensively.
    w1 = _make_gwcs_wcs('data/wfc3_uvis1.hdr')
    imcat1 = tweakwcs.JWSTgWCS(w1, {'v2_ref': 0, 'v3_ref': 0, 'roll_ref': 0})
    imcat1.meta['catalog'] = table.Table.read(
        get_pkg_data_filename('data/wfc3_uvis1.cat'),
        format='ascii.csv',
        delimiter=' ',
        names=['x', 'y']
    )
    imcat1.meta['catalog']['x'] += 1
    imcat1.meta['catalog']['y'] += 1
    imcat1.meta['group_id'] = 1
    imcat1.meta['name'] = 'ext1'

    w2 = _make_gwcs_wcs('data/wfc3_uvis2.hdr')
    imcat2 = tweakwcs.JWSTgWCS(w2, {'v2_ref': 0, 'v3_ref': 0, 'roll_ref': 0})
    imcat2.meta['catalog'] = table.Table.read(
        get_pkg_data_filename('data/wfc3_uvis2.cat'),
        format='ascii.csv',
        delimiter=' ',
        names=['x', 'y']
    )
    imcat2.meta['catalog']['x'] += 1
    imcat2.meta['catalog']['y'] += 1
    imcat2.meta['group_id'] = 1
    imcat2.meta['name'] = 'ext4'

    refcat = table.Table.read(
        get_pkg_data_filename('data/ref.cat'),
        format='ascii.csv', delimiter=' ',
        names=['RA', 'DEC']
    )

    align_wcs([imcat1, imcat2], refcat, match=_match, nclip=None,
              sigma=3, fitgeom='general')

    fi1 = imcat1.meta['fit_info']
    fi2 = imcat2.meta['fit_info']

    w1m = imcat1.wcs
    w2m = imcat2.wcs

    assert np.allclose(w1m(*w1.crpix), (83.206917667519, -67.73275818507248), rtol=0)
    assert np.allclose(w2m(*w2.crpix), (83.15167050722597, -67.74220306069903), rtol=0)

    assert np.allclose(fi1['<scale>'], 1.0025, rtol=0, atol=2e-8)
    assert np.allclose(fi2['<scale>'], 1.0025, rtol=0, atol=2e-8)

    assert fi1['rmse'] < 5e-5
    assert fi2['rmse'] < 5e-5

    ra1, dec1 = imcat1.wcs(imcat1.meta['catalog']['x'],
                           imcat1.meta['catalog']['y'])
    ra2, dec2 = imcat2.wcs(imcat2.meta['catalog']['x'],
                           imcat2.meta['catalog']['y'])
    ra = np.concatenate([ra1, ra2])
    dec = np.concatenate([dec1, dec2])
    rra = refcat['RA']
    rdec = refcat['DEC']
    rmse_ra = np.sqrt(np.mean((ra - rra)**2))
    rmse_dec = np.sqrt(np.mean((dec - rdec)**2))

    assert rmse_ra < _REF_RMSE_RA
    assert rmse_dec < _REF_RMSE_DEC


def test_multichip_alignment_step():
    # image 1
    w1 = _make_gwcs_wcs('data/wfc3_uvis1.hdr')

    m1 = _CustomImageModel(np.zeros((100, 100)))
    m1.meta.filename = 'ext1'
    m1.meta.observation.observation_number = '1'
    m1.meta.observation.program_number = '1'
    m1.meta.observation.visit_number = '1'
    m1.meta.observation.visit_group = '1'
    m1.meta.observation.sequence_id = '1'
    m1.meta.observation.activity_id = '1'
    m1.meta.observation.exposure_number = '1'

    m1.meta.wcsinfo.v2_ref = 0
    m1.meta.wcsinfo.v3_ref = 0
    m1.meta.wcsinfo.roll_ref = 0
    m1.meta.wcs = w1

    imcat1 = table.Table.read(
        get_pkg_data_filename('data/wfc3_uvis1.cat'),
        format='ascii.csv',
        delimiter=' ',
        names=['x', 'y']
    )
    imcat1['x'] += 1
    imcat1['y'] += 1
    m1.tweakreg_catalog = imcat1

    # image 2
    w2 = _make_gwcs_wcs('data/wfc3_uvis2.hdr')

    m2 = _CustomImageModel(np.zeros((100, 100)))
    m2.meta.filename = 'ext4'

    m2.meta.observation.observation_number = '1'
    m2.meta.observation.program_number = '1'
    m2.meta.observation.visit_number = '1'
    m2.meta.observation.visit_group = '1'
    m2.meta.observation.sequence_id = '1'
    m2.meta.observation.activity_id = '1'
    m2.meta.observation.exposure_number = '1'

    m2.meta.wcsinfo.v2_ref = 0
    m2.meta.wcsinfo.v3_ref = 0
    m2.meta.wcsinfo.roll_ref = 0
    m2.meta.wcs = w2

    imcat2 = table.Table.read(
        get_pkg_data_filename('data/wfc3_uvis2.cat'),
        format='ascii.csv',
        delimiter=' ',
        names=['x', 'y']
    )
    imcat2['x'] += 1
    imcat2['y'] += 1
    m2.tweakreg_catalog = imcat2

    # refcat
    wr = _make_reference_gwcs_wcs('data/wfc3_uvis1.hdr')
    mr = _CustomImageModel(np.zeros((100, 100)))
    mr.meta.filename = 'refcat'
    mr.meta.observation.observation_number = '0'
    mr.meta.observation.program_number = '0'
    mr.meta.observation.visit_number = '0'
    mr.meta.observation.visit_group = '0'
    mr.meta.observation.sequence_id = '0'
    mr.meta.observation.activity_id = '0'
    mr.meta.observation.exposure_number = '0'

    mr.meta.wcsinfo.v2_ref = 0
    mr.meta.wcsinfo.v3_ref = 0
    mr.meta.wcsinfo.roll_ref = 0
    mr.meta.wcs = wr

    refcat = table.Table.read(
        get_pkg_data_filename('data/ref.cat'),
        format='ascii.csv', delimiter=' ',
        names=['RA', 'DEC']
    )
    x, y = wr.world_to_pixel(refcat['RA'], refcat['DEC'])
    refcat['x'] = x
    refcat['y'] = y
    mr.tweakreg_catalog = refcat

    mc = ModelContainer([mr, m1, m2])
    mc.models_grouped

    step = tweakreg_step.TweakRegStep()
    step.fitgeometry = 'general'
    step.nclip = 0

    mr, m1, m2 = step.process(mc)

    wc1 = m1.meta.wcs
    wc2 = m2.meta.wcs

    ra1, dec1 = wc1(imcat1['x'], imcat1['y'])
    ra2, dec2 = wc2(imcat2['x'], imcat2['y'])
    ra = np.concatenate([ra1, ra2])
    dec = np.concatenate([dec1, dec2])
    rra = refcat['RA']
    rdec = refcat['DEC']
    rmse_ra = np.sqrt(np.mean((ra - rra)**2))
    rmse_dec = np.sqrt(np.mean((dec - rdec)**2))

    assert rmse_ra < _REF_RMSE_RA
    assert rmse_dec < _REF_RMSE_DEC
