import time
import numpy as np
from multiprocessing import Pool

from scipy import sparse

from jwst import datamodels
from .disperse import dispersed_pixel

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Observation:
    """This class defines an actual observation. It is tied to a single grism image."""

    def __init__(self, direct_images, segmap_model, grism_wcs, wr_ref, filter, sens_waves,
                 sens_resp, order=1, max_split=100, SED_file=None, extrapolate_SED=False,
                 max_cpu=1, ID=0, SBE_save=None, boundaries=[], renormalize=True):

        """
        Parameters
        ----------
        direct_images : List of strings
            List of file name(s) containing direct imaging data
        segmap_model : `jwst.datamodels.ImageModel`
            Segmentation map model
        grism_wcs : gwcs object
            WCS object from grism image
        wr_ref : `jwst.datamodels.WavelengthrangeModel`
            Wavelengthrange reference file model
        filter : str
            Filter name
        sens_waves : float array
            Wavelength array from photom reference file
        sens_resp : float array
            Response (flux calibration) array from photom reference file
        order : int
            Spectral order to simulate, +1 or +2 for NIRCAM
        max_split : int
            Number of chunks to compute instead of trying everything at once.
        SED_file : str
            Name of Spectral Energy Distribution (SED) file containing datasets matching
            the ID in the segmentation file and each consisting of a [[lambda],[flux]] array.
        extrapolate_SED : bool
            Flag indicating whether to extrapolate wavelength range of SED
        max_cpu : int
            Max number of cpu's to use when multiprocessing
        ID : int
            ID of source to process. If zero, all sources processed.
        SBE_save : str
            If set to a path, file containing simulated stamps for all objects will be saved.
        boundaries : tuple
            Start/Stop coordinates of the FOV within the larger seed image.
        renormalize : bool
            Flag indicating whether to renormalize SED's
        """

        # Load all the info for this grism mode
        self.seg_wcs = segmap_model.meta.wcs
        self.grism_wcs = grism_wcs
        self.ID = ID
        self.IDs = []
        self.dir_image_names = direct_images
        self.seg = segmap_model.data
        self.filter = filter
        self.order = order
        self.SED_file = SED_file   # should always be NONE for baseline pipeline (use flat SED)
        self.SBE_save = SBE_save
        self.max_cpu = max_cpu
        self.cache = False
        self.renormalize = renormalize
        self.sens_waves = sens_waves
        self.sens_resp = sens_resp

        # Get the wavelength range for the grism image
        wavelength_range = wr_ref.get_wfss_wavelength_range(self.filter, [self.order])
        self.wmin = wavelength_range[self.order][0]
        self.wmax = wavelength_range[self.order][1]
        log.debug(f"wmin={self.wmin}, wmax={self.wmax}")

        # Set the limits of the dispersed image to be simulated
        if len(boundaries) == 0:
            log.debug("No boundaries passed.")
            self.xstart = 0
            self.xend = self.xstart + self.dims[0] - 1
            self.ystart = 0
            self.yend = self.ystart + self.dims[1] - 1
        else:
            self.xstart, self.xend, self.ystart, self.yend = boundaries
        self.dims = (self.yend - self.ystart + 1, self.xend - self.xstart + 1)
        log.debug(f"Using final size of {self.dims[1]} {self.dims[0]}")

        # Allow for SED extrapolation
        self.extrapolate_SED = extrapolate_SED
        if self.extrapolate_SED:
            log.warning("SED Extrapolation turned on.")

        # Create pixel lists for ALL sources labeled in segmentation map
        self.create_pixel_list()

    def create_pixel_list(self):
        # Create a list of pixels to be dispersed, grouped per object ID.

        if self.ID == 0:
            # When ID=0, all sources in the segmentation map are processed.
            # This creates a huge list of all x,y pixel indices that have non-zero values
            # in the seg map, sorted by those indices belonging to a particular source ID.
            self.xs = []
            self.ys = []
            all_IDs = np.array(list(set(np.ravel(self.seg))))
            all_IDs = all_IDs[all_IDs > 0]
            log.info(f"Total of {len(all_IDs)} sources to process")
            for ID in all_IDs:
                ys, xs = np.nonzero(self.seg == ID)
                if (len(xs) > 0) & (len(ys) > 0):
                    self.xs.append(xs)
                    self.ys.append(ys)
                    self.IDs = all_IDs

        else:
            # Process only the given source ID
            log.info(f"Process source {self.ID}")
            ys, xs = np.nonzero(self.seg == self.ID)
            if (len(xs) > 0) & (len(ys) > 0):
                self.xs.append(xs)
                self.ys.append(ys)
                self.IDs = [self.ID]

        # Populate lists of direct image flux values for the sources.
        self.fluxes = {}
        for dir_image_name in self.dir_image_names:

            if self.SED_file is None:
                # Default pipeline will use SED_file=None, so we need to compute
                # photometry values that used to come from HST-style header keywords.
                # Compute pivlam as average of wmin/wmax range.
                pivlam = (self.wmin + self.wmax) / 2

            log.info(f"Loading direct image {dir_image_name}")
            dimage = datamodels.open(dir_image_name).data

            if self.SED_file is None:
                # If an SED file is not used, then we use pixel fluxes from
                # the direct image.
                self.fluxes[pivlam] = []
                for i in range(len(self.IDs)):
                    # This loads lists of pixel flux values for each source
                    # from the direct image
                    self.fluxes[pivlam].append(dimage[self.ys[i], self.xs[i]])

            else:
                # Use an SED file. Need to normalize the object stamps.
                for ID in self.IDs:
                    vg = self.seg == ID
                    dnew = dimage
                    if self.renormalize:
                        sum_seg = np.sum(dimage[vg])  # But normalize by the whole flux
                        if sum_seg != 0:
                            dimage[vg] /= sum_seg
                    else:
                        log.debug("not renormlazing sources to unity")

                self.fluxes["SED"] = []
                for i in range(len(self.IDs)):
                    self.fluxes["SED"].append(dnew[self.ys[i], self.xs[i]])

    def disperse_all(self, cache=False):
        """
        Compute dispersed pixel values for all sources identified in
        the segmentation map.
        """
        if cache:
            log.debug("Object caching ON")
            self.cache = True
            self.cached_object = {}

        # Initialize the simulated dispersed image
        self.simulated_image = np.zeros(self.dims, np.float)

        # Loop over all source ID's from segmentation map
        for i in range(len(self.IDs)):
            if self.cache:
                self.cached_object[i] = {}
                self.cached_object[i]['x'] = []
                self.cached_object[i]['y'] = []
                self.cached_object[i]['f'] = []
                self.cached_object[i]['w'] = []
                self.cached_object[i]['minx'] = []
                self.cached_object[i]['maxx'] = []
                self.cached_object[i]['miny'] = []
                self.cached_object[i]['maxy'] = []

            # Disperse object "i"
            self.disperse_chunk(i)

    def disperse_chunk(self, c):
        """
        Method that computes dispersion for a single source.
        To be called after create_pixel_list().
        """

        log.info(f"Dispersing object {c+1}")
        pars = []  # initialize params for this object

        # Loop over all pixels in list for object "c"
        log.debug(f"source contains {len(self.xs[c])} pixels")
        for i in range(len(self.xs[c])):

            # Here "i" and "ID" are just indexes into the pixel list for the object
            # being processed, as opposed to the ID number of the object itself
            ID = i

            # xc, yc are the coordinates of the central pixel of the group
            # of pixels surrounding the direct image pixel index
            width = 1.0
            height = 1.0
            xc = self.xs[c][i] + 0.5 * width
            yc = self.ys[c][i] + 0.5 * height

            # "lams" is the array of wavelengths previously stored in flux list
            # and correspond to the central wavelengths of the filters used in
            # the input direct image(s). For the simple case of 1 combined direct image,
            # this contains a single value (e.g. 4.44 for F444W).

            # "fluxes" is the array of pixel values from the direct image(s).
            # For the simple case of 1 combined direct image, this contains a
            # a single value (just like "lams").
            fluxes, lams = map(np.array, zip(*[
                (self.fluxes[lm][c][i], lm) for lm in sorted(self.fluxes.keys())
                if self.fluxes[lm][c][i] != 0
            ]))

            pars_i = (xc, yc, width, height, lams, fluxes, self.order,
                      self.wmin, self.wmax, self.sens_waves, self.sens_resp,
                      self.seg_wcs, self.grism_wcs, ID, self.dims[::-1], 2,
                      self.extrapolate_SED, self.xstart, self.ystart)

            pars.append(pars_i)
            # now have full pars list for all pixels for this object

        time1 = time.time()
        if self.max_cpu > 1:
            mypool = Pool(self.max_cpu)  # Create the pool
            all_res = mypool.imap_unordered(dispersed_pixel, pars)  # Fill the pool
            mypool.close()  # Drain the pool
        else:
            all_res = []
            for i in range(len(pars)):
                all_res.append(dispersed_pixel(*pars[i]))

        # Initialize blank image for this source
        this_object = np.zeros(self.dims, np.float)

        nres = 0
        for pp in all_res:
            if pp is None:
                continue

            nres += 1
            x, y, _, w, f, *_ = pp

            # skip results that don't have pixels in the field
            if len(x) < 1:
                continue

            minx = int(min(x))
            maxx = int(max(x))
            miny = int(min(y))
            maxy = int(max(y))
            a = sparse.coo_matrix((f, (y - miny, x - minx)),
                                  shape=(maxy - miny + 1, maxx - minx + 1)).toarray()

            # Accumulate results into simulated images
            self.simulated_image[miny:maxy + 1, minx:maxx + 1] += a
            this_object[miny:maxy + 1, minx:maxx + 1] += a

            if self.cache:
                self.cached_object[c]['x'].append(x)
                self.cached_object[c]['y'].append(y)
                self.cached_object[c]['f'].append(f)
                self.cached_object[c]['w'].append(w)
                self.cached_object[c]['minx'].append(minx)
                self.cached_object[c]['maxx'].append(maxx)
                self.cached_object[c]['miny'].append(miny)
                self.cached_object[c]['maxy'].append(maxy)

        time2 = time.time()
        log.debug(f"Elapsed time {time2-time1} sec")

        return this_object

    def disperse_all_from_cache(self, trans=None):
        if not self.cache:
            return

        self.simulated_image = np.zeros(self.dims, np.float)

        for i in range(len(self.IDs)):
            this_object = self.disperse_chunk_from_cache(i, trans=trans)

        return this_object

    def disperse_chunk_from_cache(self, c, trans=None):
        """Method that handles the dispersion. To be called after create_pixel_list()"""

        if not self.cache:
            return

        time1 = time.time()

        # Initialize blank image for this object
        this_object = np.zeros(self.dims, np.float)

        if trans is not None:
            log.debug("Applying a transmission function...")

        for i in range(len(self.cached_object[c]['x'])):
            x = self.cached_object[c]['x'][i]
            y = self.cached_object[c]['y'][i]
            f = self.cached_object[c]['f'][i] * 1.
            w = self.cached_object[c]['w'][i]

            if trans is not None:
                f *= trans(w)

            minx = self.cached_object[c]['minx'][i]
            maxx = self.cached_object[c]['maxx'][i]
            miny = self.cached_object[c]['miny'][i]
            maxy = self.cached_object[c]['maxy'][i]

            a = sparse.coo_matrix((f, (y - miny, x - minx)),
                                  shape=(maxy - miny + 1, maxx - minx + 1)).toarray()

            # Accumulate the results into the simulated images
            self.simulated_image[miny:maxy + 1, minx:maxx + 1] += a
            this_object[miny:maxy + 1, minx:maxx + 1] += a

        time2 = time.time()
        log.debug(f"Elapsed time {time2-time1} sec")

        return this_object
