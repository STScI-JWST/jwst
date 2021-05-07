#
# Module for combining 1-D spectra
#

import logging

import numpy as np

from .. import datamodels
from ..extract_1d.spec_wcs import create_spectral_wcs

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class InputSpectrumModel:
    """Attributes:
        wavelength
        flux
        flux_error
        surf_bright
        sb_error
        dq
        nelem
        weight
        unit_weight
        right_ascension
        declination
        source_id
    """

    def __init__(self, ms, spec, exptime_key):
        """Create an InputSpectrumModel object.

        Parameters
        ----------
        ms : `~jwst.datamodels.DataModel`, MultiSpecModel or SpecModel
            This is used to get the integration time.

        spec : `~jwst.datamodels.DataModel`, SpecModel table
            The table containing columns "wavelength" and "flux".
            The `ms` object may contain more than one spectrum, but `spec`
            should be just one of those.

        exptime_key : str
            A string identifying which keyword to use to get the exposure
            time, which is used as a weight; or "unit_weight", which means
            to use weight = 1.
        """

        self.wavelength = spec.spec_table.field("wavelength").copy()

        self.flux = spec.spec_table.field("flux").copy()
        self.flux_error = spec.spec_table.field("flux_error").copy()
        try:
            self.surf_bright = spec.spec_table.field("surf_bright").copy()
            self.sb_error = spec.spec_table.field("sb_error").copy()
        except KeyError:
            self.surf_bright = np.zeros_like(self.flux)
            self.sb_error = np.zeros_like(self.flux)
            log.warning("There is no SURF_BRIGHT column in the input.")
        self.dq = spec.spec_table.field("dq").copy()
        self.nelem = self.wavelength.shape[0]
        self.unit_weight = False        # may be reset below
        self.right_ascension = np.zeros_like(self.wavelength)
        self.declination = np.zeros_like(self.wavelength)
        self.source_id = spec.source_id

        self.weight = np.ones_like(self.wavelength)
        if exptime_key == "integration_time":
            self.weight *= ms.meta.exposure.integration_time
        elif exptime_key == "exposure_time":
            self.weight *= ms.meta.exposure.exposure_time
        elif exptime_key == "unit_weight":
            self.unit_weight = True
        else:
            raise RuntimeError("Don't understand exptime_key = '%s'" %
                               exptime_key)

        try:
            self.right_ascension[:], self.declination[:], _ = spec.meta.wcs(0.)
        except AttributeError:
            self.right_ascension[:] = ms.meta.target.ra
            self.declination[:] = ms.meta.target.dec
            log.warning("There is no WCS in the input.")

    def close(self):
        self.wavelength = None
        self.flux = None
        self.flux_error = None
        self.surf_bright = None
        self.sb_error = None
        self.dq = None
        self.nelem = 0
        self.weight = 1.
        self.unit_weight = False
        self.right_ascension = None
        self.declination = None
        self.source_id = None


class OutputSpectrumModel:
    """Attributes:
        wavelength
        flux
        flux_error
        surf_bright
        sb_error
        dq
        weight
        count
        wcs
        normalized
    """

    def __init__(self):

        self.wavelength = None
        self.flux = None
        self.flux_error = None
        self.surf_bright = None
        self.sb_error = None
        self.dq = None
        self.weight = None
        self.count = None
        self.wcs = None
        self.normalized = False
        self.source_id = None

    def assign_wavelengths(self, input_spectra):
        """Create an array of wavelengths to use for the output spectrum.

        Take the union of all input wavelengths, then call compute_output_wl
        to bin wavelengths in groups of the number of overlapping spectra.

        Parameters
        ----------
        input_spectra : list of InputSpectrumModel objects
            List of input spectra.
        """

        (wl, n_input_spectra) = count_input(input_spectra)

        self.wavelength = compute_output_wl(wl, n_input_spectra)

        self.wcs = create_spectral_wcs(input_spectra[0].right_ascension[0],
                                       input_spectra[0].declination[0],
                                       self.wavelength)

    def accumulate_sums(self, input_spectra):
        """Compute a weighted sum of all the input spectra.

        Each pixel of each input spectrum will be added to one pixel of
        the output spectrum.  The wavelength spacing of the input and
        output should be comparable, so each input pixel will usually
        correspond to one output pixel (i.e. have the same wavelength,
        to within half a pixel).  However, for any given input spectrum,
        there can be output pixels that are incremented by more than one
        input pixel, and there can be output pixels that are not incremented.
        If there are several input spectra, such gaps will hopefully be
        filled in.

        Pixels in an input spectrum that are flagged (via the DQ array)
        as DO_NOT_USE will not be used, i.e. they will simply be ignored
        when incrementing output arrays from an input spectrum.  If an
        input pixel is flagged with a non-zero value other than DO_NOT_USE,
        the value will be propagated to the output DQ array via bitwise OR.

        Parameters
        ----------
        input_spectra : list of InputSpectrumModel objects
            List of input spectra.
        """

        # This is the data type for the output spectrum.  We'll use double
        # precision for accumulating sums for most columns, but for the DQ
        # array, use the correct output data type.
        cmb_dtype = datamodels.CombinedSpecModel().spec_table.dtype
        dq_dtype = cmb_dtype.fields["DQ"][0]

        nelem = self.wavelength.shape[0]

        self.flux = np.zeros(nelem, dtype=np.float64)
        self.flux_error = np.zeros(nelem, dtype=np.float64)
        self.surf_bright = np.zeros(nelem, dtype=np.float64)
        self.sb_error = np.zeros(nelem, dtype=np.float64)
        self.dq = np.zeros(nelem, dtype=dq_dtype)
        self.weight = np.zeros(nelem, dtype=np.float64)
        self.count = np.zeros(nelem, dtype=np.float64)

        n_nan = 0                                       # initial value
        ninputs = 0
        for in_spec in input_spectra:
            ninputs += 1
            log.info(f'Accumulating data from input spectrum {ninputs}')
            # Get the pixel numbers in the output corresponding to the
            # wavelengths of the current input spectrum.
            out_pixel = self.wcs.invert(in_spec.right_ascension,
                                        in_spec.declination,
                                        in_spec.wavelength)
            # i is a pixel number in the current input spectrum, and
            # k is the corresponding pixel number in the output spectrum.
            nan_flag = np.isnan(out_pixel)
            n_nan += nan_flag.sum()
            for i in range(len(out_pixel)):
                if in_spec.dq[i] & datamodels.dqflags.pixel['DO_NOT_USE'] > 0:
                    continue
                # Round to the nearest pixel.
                if nan_flag[i]:         # skip if the pixel number is NaN
                    continue
                k = round(float(out_pixel[i]))
                if k < 0 or k >= nelem:
                    continue
                weight = in_spec.weight[i]
                self.dq[k] |= in_spec.dq[i]
                self.flux[k] += in_spec.flux[i] * weight
                self.flux_error[k] += (in_spec.flux_error[i] * weight)**2
                self.surf_bright[k] += (in_spec.surf_bright[i] * weight)
                self.sb_error[k] += (in_spec.sb_error[i] * weight)**2
                self.weight[k] += weight
                self.count[k] += 1.
        if n_nan > 0:
            log.warning("%d output pixel numbers were NaN", n_nan)

        # Since the output wavelengths will not usually be exactly the same
        # as the input wavelengths, it's possible that there will be output
        # pixels for which there is no corresponding pixel in any of the
        # input spectra.  Check for this case.
        index = np.where(self.count > 0.)
        n_good = len(index[0])
        if nelem > n_good:
            log.warning("%d elements of output had no corresponding"
                        " input data;" % (nelem - n_good,))
            log.warning("    these elements will be omitted.")
            self.wavelength = self.wavelength[index]
            self.flux = self.flux[index]
            self.flux_error = self.flux_error[index]
            self.surf_bright = self.surf_bright[index]
            self.sb_error = self.sb_error[index]
            self.dq = self.dq[index]
            self.weight = self.weight[index]
            self.count = self.count[index]
        del index

        self.normalized = False

    def compute_combination(self):
        """Compute the combined values."""

        if not self.normalized:
            sum_weight = np.where(self.weight > 0., self.weight, 1.)
            self.surf_bright /= sum_weight
            self.flux /= sum_weight
            self.flux_error = np.sqrt(self.flux_error / sum_weight)
            self.sb_error = np.sqrt(self.sb_error / sum_weight)
            self.normalized = True

    def create_output_data(self):
        """Create the output data.

        Returns
        -------
        output_model : `~jwst.datamodels.DataModel`, CombinedSpecModel object
            A table of combined spectral data.
        """

        if not self.normalized:
            log.warning("Data have not been divided by"
                        " the sum of the weights.")

        cmb_dtype = datamodels.CombinedSpecModel().spec_table.dtype

        # Note that these arrays have to be in the right order.
        data = np.array(list(zip(self.wavelength,
                                 self.flux,
                                 self.flux_error,
                                 self.surf_bright,
                                 self.sb_error,
                                 self.dq,
                                 self.weight,
                                 self.count)), dtype=cmb_dtype)
        output_model = datamodels.CombinedSpecModel(spec_table=data)

        return output_model

    def close(self):
        self.wavelength = None
        self.flux = None
        self.flux_error = None
        self.surf_bright = None
        self.sb_error = None
        self.dq = None
        self.weight = None
        self.count = None
        self.wcs = None
        self.normalized = False
        self.source_id = None


def count_input(input_spectra):
    """Determine the number of input spectra that cover each wavelength.

    For any given input spectrum, the array of wavelengths gives the
    wavelengths at the centers of the pixels.  In this context, the
    expression that an input spectrum "covers" some particular wavelength
    (typically not one of the elements in the input spectrum's array of
    wavelengths) means that this wavelength is within the interval between
    the left edge of the first pixel and the right edge of the last pixel
    of the input spectrum.

    Parameters
    ----------
    input_spectra : list of InputSpectrumModel objects
        List of input spectra.

    Returns
    -------
    wl : ndarray
        Sorted list of all the wavelengths in all the input spectra.

    n_input_spectra : ndarray, 1-D
        For each element of `wl`, the corresponding element of
        `n_input_spectra` is the number of input spectra that cover the
        wavelength in `wl`.
    """

    # Create an array with all the input wavelengths (i.e. the union
    # of the input wavelengths).
    wl = np.hstack([in_spec.wavelength for in_spec in input_spectra])
    wl.sort()
    nwl = len(wl)

    # n_input_spectra will be the number of input spectra that cover the
    # corresponding wavelength in wl.
    n_input_spectra = np.zeros(nwl, dtype=np.int64)
    for in_spec in input_spectra:
        input_wl = in_spec.wavelength

        # Check for degenerate spectrum. Skip with log.
        if len(input_wl) < 2:
            log.warning(f'Spectrum {in_spec} is degenerate with length {len(input_wl)}')
            log.warning('Skipping...')
            continue

        # wl0 and wl1 will be about a half pixel wider on either side
        # of the wavelength range for the current input spectrum.
        if input_wl[1] > input_wl[0]:       # wavelengths are increasing
            wl0 = input_wl[0] - 0.5 * (input_wl[1] - input_wl[0])
            wl1 = input_wl[-1] + 0.5 * (input_wl[-1] - input_wl[-2])
        elif input_wl[1] < input_wl[0]:     # wavelengths are decreasing
            wl0 = input_wl[-1] - 0.5 * (input_wl[-2] - input_wl[-1])
            wl1 = input_wl[0] + 0.5 * (input_wl[0] - input_wl[1])
        else:
            log.warning(f'Spectrum {in_spec} has a monotonic wavelength solution.')
            log.warning('Skipping...')
            continue
        temp = np.where(wl >= wl0, 1, 0)
        temp = np.where(wl >= wl1, 0, temp)
        n_input_spectra += temp
        del temp

    # This shouldn't happen.
    if np.any(n_input_spectra <= 0.):
        raise RuntimeError("Problem with input wavelengths.")

    return (wl, n_input_spectra)


def compute_output_wl(wl, n_input_spectra):
    """Compute output wavelengths.

    In summary, the output wavelengths are computed by binning the
    input wavelengths in groups of the number of overlapping spectra.
    The merged and sorted input wavelengths are in `wl`.

    If the wavelength arrays of the input spectra are nearly all the
    same, the values in wl will be in clumps of N nearly identical
    values (for N input files).  We want to average the wavelengths
    in those clumps to get the output wavelengths.  However, if the
    wavelength scales of the input files differ slightly, the values
    in wl may clump in some regions but not in others.  The algorithm
    below tries to find clumps; a clump is identified by having a
    small standard deviation of all the wavelengths in a slice of wl,
    compared with neighboring slices.  It is intended that this should
    not find clumps where they're not present, i.e. in regions where
    the wavelengths of the input spectra are not well aligned with each
    other.  These regions will be gaps in the initial determination
    of output wavelengths based on clumps.  In such regions, output
    wavelengths will then be computed by just averaging groups of
    input wavelengths to fill in the gaps.

    Parameters
    ----------
    wl : 1-D array
        An array containing all the wavelengths from all the input
        spectra, sorted in increasing order.

    n_input_spectra : 1-D array
        An integer array of the same length as `wl`.  For a given
        array index k (for example), n_input_spectra[k] is the number of
        input spectra that cover wavelength wl[k].

    Returns
    -------
    wavelength :  1-D array
        Array of wavelengths for the output spectrum.
    """

    nwl = len(wl)

    # sigma is an array of the standard deviation at each element
    # of wl, over n_input_spectra elements.  A small value implies that
    # there's a clump, i.e. several elements of wl with nearly the
    # same wavelength.
    sigma = np.zeros(nwl, dtype=np.float64) + 9999.

    # mean_wl is the mean wavelength over the same slice of wl that we
    # used to compute sigma.  If sigma is small enough that it looks
    # as if there's a clump, we'll copy the mean_wl value to temp_wl
    # to be one element of the output wavelengths.
    mean_wl = np.zeros(nwl, dtype=np.float64) - 99.

    # temp_wl has the same number of elements as wl, but we expect the
    # array of output wavelengths to be significantly smaller, so
    # temp_wl is initialized to a negative value as a flag.  Positive
    # elements will be copied to the array of output wavelengths.
    temp_wl = np.zeros(nwl, dtype=np.float64) - 99.

    for k in range(nwl):
        n = n_input_spectra[k]
        if n == 1:
            sigma[k] = 0.
            mean_wl[k] = wl[k]
            temp_wl[k] = mean_wl[k]
        else:
            k1 = k + n // 2 + 1
            k0 = k1 - n
            if k0 >= 0 and k1 <= nwl:
                sigma[k] = wl[k0:k1].std()
                mean_wl[k] = wl[k0:k1].mean()
                if sigma[k] == 0.:
                    temp_wl[k] = mean_wl[k]

    cutoff = 0.8
    for k in range(nwl):
        # If sigma[k] equals 0, temp_wl has already been assigned.
        if sigma[k] > 0.:
            if k == 0:
                if sigma[k] < cutoff * sigma[1]:
                    temp_wl[k] = mean_wl[k]
            elif k == nwl - 1:
                if sigma[k] < cutoff * sigma[nwl - 2]:
                    temp_wl[k] = mean_wl[k]
            else:
                if sigma[k] < cutoff * (sigma[k - 1] + sigma[k + 1]) / 2.:
                    temp_wl[k] = mean_wl[k]

    # Fill gaps in the output wavelengths by taking averages of the
    # input wavelengths.  If there are n overlapping input spectra,
    # average a block of n elements of `wl`.
    done = False
    i = 0
    while not done:
        if i >= nwl:
            done = True
        else:
            n = n_input_spectra[i]
            middle = n // 2
            if i + middle < nwl:
                n = n_input_spectra[i + middle]
                if i + n < nwl:
                    # The nominal range is i - 1 to i + n (inclusive) for
                    # checking whether an output wavelength has already
                    # been assigned.
                    low = max(i - 1, 0)
                    high = min(i + n + 1, nwl - 1)
                    if temp_wl[low:high].max() <= 0.:
                        temp_wl[i] = wl[i:i + n].mean()
            i += n

    return temp_wl[np.where(temp_wl > 0.)].copy()


def check_exptime(exptime_key):
    """Check exptime_key for validity.

    This function checks exptime_key.  If it is valid, the corresponding
    value used by the metadata interface will be returned.  This will be
    either "integration_time" or "exposure_time".  If it is not valid,
    "unit weight" will be returned (meaning that a weight of 1 will be
    used when averaging spectra), and a warning will be logged.

    Parameters
    ----------
    exptime_key : str
        A keyword or string indicating what value (integration time or
        exposure time) should be used as a weight when combing spectra.

    Returns
    -------
    exptime_key : str
        The value will be either "integration_time", "exposure_time",
        or "unit_weight".
    """

    exptime_lwr = exptime_key.lower()
    if exptime_lwr.startswith("integration") or \
       exptime_lwr == "effinttm":
        exptime_key = "integration_time"
        log.info("Using integration time as the weight.")
    elif exptime_lwr.startswith("exposure") or \
            exptime_lwr == "effexptm":
        exptime_key = "exposure_time"
        log.info("Using exposure time as the weight.")
    elif exptime_lwr == "unit_weight" or exptime_lwr == "unit weight":
        exptime_key = "unit_weight"
        log.info("Using weight = 1.")
    else:
        log.warning("Don't understand exptime_key = '%s';"
                    " using unit weight." % exptime_key)
        log.info("The options for exptime_key are:")
        log.info("  integration_time, effinttm, exposure_time, effexptm,"
                 " unit_weight, unit weight")
        exptime_key = "unit_weight"

    return exptime_key


def combine_1d_spectra(input_model, exptime_key):
    """Combine the input spectra.

    Parameters
    ----------
    input_model : `~jwst.datamodels.DataModel`
        The input spectra.  This will likely be a ModelContainer object.

    exptime_key : str
        A string identifying which keyword to use to get the exposure time,
        which is used as a weight when combining spectra.  The value should
        be one of:  "exposure_time" (the default), "integration_time",
        or "unit_weight".

    Returns
    -------
    output_model : `~jwst.datamodels.DataModel`
        A datamodels.CombinedSpecModel object.
    """

    log.debug("Using exptime_key = {}.".format(exptime_key))

    exptime_key = check_exptime(exptime_key)

    input_spectra = {}
    output_spectra = {}
    if isinstance(input_model, datamodels.ModelContainer):
        for ms in input_model:
            for in_spec in ms.spec:
                spectral_order = in_spec.spectral_order
                if spectral_order not in input_spectra:
                    input_spectra[spectral_order] = []
                input_spectra[spectral_order].append(InputSpectrumModel(
                    ms, in_spec, exptime_key))
    else:
        for in_spec in input_model.spec:
            spectral_order = in_spec.spectral_order
            if spectral_order not in input_spectra:
                input_spectra[spectral_order] = []
            input_spectra[spectral_order].append(InputSpectrumModel(
                input_model, in_spec, exptime_key))

    for order in input_spectra:
        output_spectra[order] = OutputSpectrumModel()
        output_spectra[order].assign_wavelengths(input_spectra[order])
        output_spectra[order].accumulate_sums(input_spectra[order])
        output_spectra[order].compute_combination()

    output_model = datamodels.MultiCombinedSpecModel()

    for order in output_spectra:
        output_order = output_spectra[order].create_output_data()
        output_order.spectral_order = order
        output_order.source_id = input_spectra[order][0].source_id
        output_model.spec.append(output_order)

    # Copy one of the input headers to output.
    if isinstance(input_model, datamodels.ModelContainer):
        output_model.update(input_model[0], only="PRIMARY")
    else:
        output_model.update(input_model, only="PRIMARY")

    # Looks clunky, but need an output_spec instance to copy wcs
    output_model.meta.wcs = output_spectra[list(output_spectra)[0]].wcs
    output_model.meta.cal_step.combine_1d = 'COMPLETE'

    for order in input_spectra:
        for in_spec in input_spectra[order]:
            in_spec.close()

    for order in output_spectra:
        output_spectra[order].close()

    return output_model
