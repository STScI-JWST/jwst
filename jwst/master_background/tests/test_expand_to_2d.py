"""
Test for master_background.expand_to_2d
"""
import numpy as np

from jwst import datamodels
from jwst.master_background import expand_to_2d


def test_expand_to_2d_1():
    """Test 1"""
    input = slit_data_a()               # MultiSlitModel
    m_bkg_spec = user_bkg_spec_a()      # MultiSpecModel
    bkg = expand_to_2d.expand_to_2d(input, m_bkg_spec)
    truth_a = truth_array_a()

    assert np.allclose(bkg.slits[0].data, truth_a, rtol=1.e-6)


def test_expand_to_2d_2():
    """Test 2"""
    # Same input data as in the first test.
    input = slit_data_a()               # MultiSlitModel
    # Check that expand_to_2d works if the wavelength array is reversed.
    # The flux array is also reversed, so the results should be unchanged.
    m_bkg_spec = user_bkg_spec_b()
    bkg = expand_to_2d.expand_to_2d(input, m_bkg_spec)
    # Same truth array as in the first test.
    truth_a = truth_array_a()

    assert np.allclose(bkg.slits[0].data, truth_a, rtol=1.e-6)


def test_expand_to_2d_3():
    """Test 3"""
    input = image_data_c()              # ImageModel
    m_bkg_spec = user_bkg_spec_c()      # CombinedSpecModel
    bkg = expand_to_2d.expand_to_2d(input, m_bkg_spec)
    truth_c = truth_array_c()

    assert np.allclose(bkg.data, truth_c, rtol=1.e-6)


# slit_data_a and image_data_c create `input` for the tests above.

def slit_data_a():
    """Create "science" data for testing.

    Returns
    -------
    input_model : `~jwst.datamodels.MultiSlitModel`
    """

    # Create a MultiSlitModel object.
    data_shape = (5, 9)
    data = np.zeros(data_shape, dtype=np.float32) + 10.
    dq = np.zeros(data_shape, dtype=np.uint32)
    # One row of wavelengths.
    temp_wl = np.linspace(1.3, 4.8, num=data_shape[1], endpoint=True,
                          retstep=False, dtype=np.float32)
    wavelength = np.zeros(data_shape, dtype=np.float32)
    # Shift the wavelength values from row to the next.
    dwl = 0.1
    for j in range(data_shape[0]):
        wavelength[j, :] = (temp_wl + j * dwl)
    wavelength = np.around(wavelength, 4)
    input_model = datamodels.MultiSlitModel()
    slit = datamodels.SlitModel(init=None, data=data, dq=dq,
                                wavelength=wavelength)
    input_model.slits.append(slit)

    return input_model


def image_data_c():
    """Create "science" data for testing.

    Returns
    -------
    input_model : `~jwst.datamodels.ImageModel`
    """

    data_shape = (9, 5)
    data = np.zeros(data_shape, dtype=np.float32) + 10.
    dq = np.zeros(data_shape, dtype=np.uint32)
    input_model = datamodels.ImageModel(data=data, dq=dq)

    def mock_wcs(x, y):
        """Fake wcs method."""

        # One row of wavelengths.
        temp_wl = np.linspace(1.3, 4.8, num=data_shape[1], endpoint=True,
                              retstep=False, dtype=np.float32)
        wavelength = np.zeros(data_shape, dtype=np.float32)
        # Shift the wavelength values from row to the next.
        dwl = 0.1
        for j in range(data_shape[0]):
            wavelength[j, :] = (temp_wl + j * dwl)

        if hasattr(x, 'dtype'):
            ix = np.around(x).astype(np.int64)
        else:
            ix = round(x)
        if hasattr(y, 'dtype'):
            iy = np.around(y).astype(np.int64)
        else:
            iy = round(y)

        wl = wavelength[iy, ix]
        ra = wl.copy()                          # placeholder
        dec = wl.copy()                         # placeholder
        return (ra, dec, wl)

    input_model.meta.wcs = mock_wcs

    return input_model


# Functions user_bkg_spec_a, user_bkg_spec_b, and user_bkg_spec_c create
# "user-supplied" background data for the tests above.

def user_bkg_spec_a():
    """Create "user-background" data for testing.

    Returns
    -------
    m_bkg_spec : `~jwst.datamodels.MultiSpecModel`
    """

    # This data type is used for creating a MultiSpecModel.
    spec_dtype = datamodels.SpecModel().spec_table.dtype

    # m_bkg_spec doesn't have to be a MultiSpecModel, but that's an option.
    m_bkg_spec = datamodels.MultiSpecModel()
    wavelength = np.geomspace(1.5, 4.5, num=25, endpoint=True,
                              dtype=np.float64)
    flux = np.zeros_like(wavelength)
    error = np.ones_like(wavelength)
    var_dummy = error.copy()
    surf_bright = np.linspace(13., 25., num=25, endpoint=True, retstep=False,
                              dtype=np.float64)
    sb_error = np.ones_like(wavelength)
    dq = np.zeros(wavelength.shape, dtype=np.uint32)
    background = np.ones_like(wavelength)
    berror = np.ones_like(wavelength)
    # The npixels column should no longer be used.  Set it to a large value
    # to make it more obvious in case it actually is still used.
    npixels = np.zeros_like(wavelength) + 2000.
    otab = np.array(list(zip(wavelength, flux, error, var_dummy, var_dummy,
                             var_dummy, surf_bright, sb_error, var_dummy,
                             var_dummy, var_dummy, dq, background, berror,
                             var_dummy, var_dummy, var_dummy, npixels)),
                    dtype=spec_dtype)
    spec = datamodels.SpecModel(spec_table=otab)
    m_bkg_spec.spec.append(spec)

    return m_bkg_spec


def user_bkg_spec_b():
    """Create "user-background" data for testing.

    `expand_to_2d` uses `np.interp` for interpolation, and the wavelength
    array that is passed to `np.interp` must be increasing.  `expand_to_2d`
    is supposed to handle the case that the wavelengths are decreasing.
    Create data for checking that the results are the same even if the
    wavelength array in `m_bkg_spec` is reversed so that the values are
    decreasing, and the corresponding flux array is also reversed to retain
    the original (wavelength, flux) relation.

    Returns
    -------
    m_bkg_spec : `~jwst.datamodels.MultiSpecModel`
    """

    # This data type is used for creating a MultiSpecModel.
    spec_dtype = datamodels.SpecModel().spec_table.dtype

    m_bkg_spec = datamodels.MultiSpecModel()
    wavelength = np.geomspace(1.5, 4.5, num=25, endpoint=True,
                              dtype=np.float64)[::-1]
    flux = np.zeros_like(wavelength)
    error = np.ones_like(wavelength)
    var_dummy = error.copy()
    surf_bright = np.linspace(13., 25., num=25, endpoint=True, retstep=False,
                              dtype=np.float64)[::-1]
    sb_error = np.ones_like(wavelength)
    dq = np.zeros(wavelength.shape, dtype=np.uint32)
    background = np.ones_like(wavelength)
    berror = np.ones_like(wavelength)
    npixels = np.ones_like(wavelength)
    otab = np.array(list(zip(wavelength, flux, error, var_dummy, var_dummy,
                             var_dummy, surf_bright, sb_error, var_dummy,
                             var_dummy, var_dummy, dq, background, berror,
                             var_dummy, var_dummy, var_dummy, npixels)),
                    dtype=spec_dtype)
    spec = datamodels.SpecModel(spec_table=otab)
    m_bkg_spec.spec.append(spec)

    return m_bkg_spec


def user_bkg_spec_c():
    """Create "user-background" data for testing.

    Returns
    -------
    m_bkg_spec : `~jwst.datamodels.CombinedSpecModel`
    """

    # This is the data type of an output table from combine_1d.
    spec_table_dtype = datamodels.CombinedSpecModel().spec_table.dtype

    wavelength = np.geomspace(1.5, 4.5, num=25, endpoint=True,
                              dtype=np.float64)
    flux = np.zeros_like(wavelength)
    error = np.ones_like(wavelength)
    surf_bright = np.linspace(13., 25., num=25, endpoint=True, retstep=False,
                              dtype=np.float64)
    sb_error = np.ones_like(wavelength)
    dq = np.zeros(wavelength.shape, dtype=np.uint32)
    weight = np.ones_like(wavelength)
    n_input = np.ones_like(wavelength)                  # yes, float64
    data = np.array(list(zip(wavelength, flux, error, surf_bright,
                             sb_error, dq, weight, n_input)),
                    dtype=spec_table_dtype)
    m_bkg_spec = datamodels.CombinedSpecModel(spec_table=data)

    return m_bkg_spec


def truth_array_a():
    """Create an array of comparison values, for testing.

    Returns
    -------
    truth_a : ndarray, 2-D, float64
        An array to compare with the data in the output from `expand_to_2d`.
    """

    truth_a = np.array([[0., 14.603571, 17.057364, 19.059263, 20.748844,
                         22.21304, 23.506573, 24.658548, 0.],
                        [0., 15.213889, 17.548525, 19.470137, 21.102207,
                         22.524107, 23.77871, 24.906878, 0.],
                        [13., 15.792757, 18.018988, 19.86396, 21.444334,
                         22.822334, 24.04857, 0., 0.],
                        [13.702182, 16.342768, 18.469252, 20.244974,
                         21.773643, 23.115166, 24.308529, 0., 0.],
                        [14.364901, 16.866348, 18.900745, 20.614536,
                         22.095966, 23.40005, 24.565424, 0., 0.]],
                       dtype=np.float64)

    return truth_a


def truth_array_c():
    """Create an array of comparison values, for testing.

    Returns
    -------
    truth_c : ndarray, 2-D, float64
        An array to compare with the data in the output from `expand_to_2d`.
    """

    truth_c = np.array([[0., 17.057364, 20.748844, 23.506573, 0.],
                        [0., 17.548523, 21.102207, 23.77871, 0.],
                        [13., 18.018988, 21.444334, 24.04857, 0.],
                        [13.702181, 18.469252, 21.773643, 24.308529, 0.],
                        [14.3649, 18.900745, 22.095966, 24.565424, 0.],
                        [14.9912815, 19.31606, 22.408163, 24.813755, 0.],
                        [15.5804825, 19.716778, 22.710499, 0., 0.],
                        [16.139992, 20.104378, 23.008335, 0., 0.],
                        [16.672644, 20.479303, 23.293217, 0., 0.]],
                       dtype=np.float64)

    return truth_c
