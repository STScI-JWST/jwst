"""
Test for extract_1d._extract_src_flux
"""
import math

import numpy as np

from jwst.extract_1d import extract1d


def test_extract_src_flux():

    shape = (9, 5)
    image = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    var_poisson = image.copy()
    var_rnoise = image.copy()
    var_rflat = image.copy()
    x = 2
    j = 2
    lam = 1.234                 # an arbitrary value (not actually used)
    weights = None
    bkgmodels = [None, None, None, None]
    lower = np.zeros(shape[1], dtype=np.float64) + 3.   # middle of pixel 3
    upper = np.zeros(shape[1], dtype=np.float64) + 7.   # middle of pixel 7
    srclim = [[lower, upper]]

    (total_flux, f_var_poisson, f_var_rnoise, f_var_flat,
     bkg_flux, b_var_poisson, b_var_rnoise, b_var_flat,
     tarea, twht) = extract1d._extract_src_flux(
        image, var_poisson, var_rnoise, var_rflat,
        x, j, lam, srclim, weights, bkgmodels)

    # 0.5 * 17. + 22. + 27. + 32. + 0.5 * 37.
    assert math.isclose(total_flux, 108., rel_tol=1.e-8, abs_tol=1.e-8)

    assert bkg_flux == 0.

    assert math.isclose(tarea, 4., rel_tol=1.e-8, abs_tol=1.e-8)

    image[5, 2] = np.nan

    (total_flux, f_var_poisson, f_var_rnoise, f_var_flat,
     bkg_flux, b_var_poisson, b_var_rnoise, b_var_flat,
     tarea, twht) = extract1d._extract_src_flux(
        image, var_poisson, var_rnoise, var_rflat,
        x, j, lam, srclim, weights, bkgmodels)

    # 0.5 * 17. + 22. + 32. + 0.5 * 37.
    assert math.isclose(total_flux, 81., rel_tol=1.e-8, abs_tol=1.e-8)

    assert math.isclose(tarea, 3., rel_tol=1.e-8, abs_tol=1.e-8)

    image[:, 2] = np.nan

    (total_flux, f_var_poisson, f_var_rnoise, f_var_flat,
     bkg_flux, b_var_poisson, b_var_rnoise, b_var_flat,
     tarea, twht) = extract1d._extract_src_flux(
        image, var_poisson, var_rnoise, var_rflat,
        x, j, lam, srclim, weights, bkgmodels)

    assert np.isnan(total_flux)

    assert tarea == 0.
