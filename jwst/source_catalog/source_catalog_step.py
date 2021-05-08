"""
Module for the source catalog step.
"""

import os
import warnings

from crds.core.exceptions import CrdsLookupError
import numpy as np
from photutils.utils.exceptions import NoDetectionsWarning

from .source_catalog import (ReferenceData, Background, make_kernel,
                             make_segment_img, JWSTSourceCatalog)
from .. import datamodels
from ..stpipe import Step

__all__ = ["SourceCatalogStep"]


class SourceCatalogStep(Step):
    """
    Create a final catalog of source photometry and morphologies.

    Parameters
    -----------
    input : str or `ImageModel`
        A FITS filename or an `ImageModel` of a drizzled image.
    """

    spec = """
        bkg_boxsize = integer(default=100)    # background mesh box size in pixels
        kernel_fwhm = float(default=2.0)      # Gaussian kernel FWHM in pixels
        snr_threshold = float(default=3.0)    # SNR threshold above the bkg
        npixels = integer(default=5)          # min number of pixels in source
        deblend = boolean(default=False)      # deblend sources?
        aperture_ee1 = integer(default=30)    # aperture encircled energy 1
        aperture_ee2 = integer(default=50)    # aperture encircled energy 2
        aperture_ee3 = integer(default=70)    # aperture encircled energy 3
        ci1_star_threshold = float(default=2.0)  # CI 1 star threshold
        ci2_star_threshold = float(default=1.8)  # CI 2 star threshold
        suffix = string(default='cat')        # Default suffix for output files
    """

    reference_file_types = ['apcorr', 'abvegaoffset']

    def process(self, input_model):
        with datamodels.open(input_model) as model:
            try:
                apcorr_fn = self.get_reference_file(input_model, 'apcorr')
            except CrdsLookupError:
                apcorr_fn = None
            self.log.info(f'Using APCORR reference file {apcorr_fn}')

            try:
                abvegaoffset_fn = self.get_reference_file(input_model,
                                                          'abvegaoffset')
            except CrdsLookupError:
                abvegaoffset_fn = None
            self.log.info('Using ABVEGAOFFSET reference file '
                          f'{abvegaoffset_fn}')

            aperture_ee = (self.aperture_ee1, self.aperture_ee2,
                           self.aperture_ee3)
            try:
                refdata = ReferenceData(model, aperture_ee=aperture_ee,
                                        apcorr_filename=apcorr_fn,
                                        abvegaoffset_filename=abvegaoffset_fn)
                aperture_params = refdata.aperture_params
                abvega_offset = refdata.abvega_offset
            except RuntimeError as err:
                msg = f'{err} Source catalog will not be created.'
                self.log.warning(msg)
                return

            coverage_mask = np.isnan(model.err) | (model.wht == 0)
            if coverage_mask.all():
                self.log.warning('There are no valid pixels. Source catalog '
                                 'will not be created.')
                return

            bkg = Background(model.data, box_size=self.bkg_boxsize,
                             mask=coverage_mask)
            model.data -= bkg.background

            threshold = self.snr_threshold * bkg.background_rms
            kernel = make_kernel(self.kernel_fwhm)
            with warnings.catch_warnings():
                # suppress NoDetectionsWarning from photutils
                warnings.filterwarnings('ignore',
                                        category=NoDetectionsWarning)
                segment_img = make_segment_img(model.data, threshold,
                                               npixels=self.npixels,
                                               kernel=kernel,
                                               mask=coverage_mask,
                                               deblend=self.deblend)
            if segment_img is None:
                self.log.warning('No sources were found. Source catalog '
                                 'will not be created.')
                return
            self.log.info(f'Detected {segment_img.nlabels} sources')

            ci_star_thresholds = (self.ci1_star_threshold,
                                  self.ci2_star_threshold)
            catobj = JWSTSourceCatalog(model, segment_img, kernel=kernel,
                                       kernel_fwhm=self.kernel_fwhm,
                                       aperture_params=aperture_params,
                                       abvega_offset=abvega_offset,
                                       ci_star_thresholds=ci_star_thresholds)
            catalog = catobj.catalog

            if self.save_results:
                cat_filepath = self.make_output_path(ext='.ecsv')
                catalog.write(cat_filepath, format='ascii.ecsv',
                              overwrite=True)
                model.meta.source_catalog = os.path.basename(cat_filepath)
                self.log.info(f'Wrote source catalog: {cat_filepath}')

                segm_model = datamodels.ImageModel(segment_img.data)
                segm_model.update(model, only="PRIMARY")
                segm_model.meta.wcs = model.meta.wcs
                segm_model.meta.wcsinfo = model.meta.wcsinfo
                self.save_model(segm_model, suffix='segm')
                model.meta.segmentation_map = segm_model.meta.filename
                self.log.info(f'Wrote segmentation map: {segm_model.meta.filename}')

        return catalog
