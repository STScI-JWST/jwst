"""
JWST pipeline step for image alignment.

:Authors: Mihai Cara

"""
from os import path

from astropy.table import Table
from tweakwcs.imalign import align_wcs
from tweakwcs.tpwcs import JWSTgWCS
from tweakwcs.matchutils import TPMatch

# LOCAL
from ..stpipe import Step
from .. import datamodels

from . import astrometric_utils as amutils
from .tweakreg_catalog import make_tweakreg_catalog


__all__ = ['TweakRegStep']


class TweakRegStep(Step):
    """
    TweakRegStep: Image alignment based on catalogs of sources detected in
    input images.
    """

    spec = """
        save_catalogs = boolean(default=False) # Write out catalogs?
        catalog_format = string(default='ecsv') # Catalog output file format
        kernel_fwhm = float(default=2.5) # Gaussian kernel FWHM in pixels
        snr_threshold = float(default=10.0) # SNR threshold above the bkg
        brightest = integer(default=100) # Keep top ``brightest`` objects
        peakmax = float(default=None) # Filter out objects with pixel values >= ``peakmax``
        enforce_user_order = boolean(default=False) # Align images in user specified order?
        expand_refcat = boolean(default=False) # Expand reference catalog with new sources?
        minobj = integer(default=15) # Minimum number of objects acceptable for matching
        searchrad = float(default=1.0) # The search radius in arcsec for a match
        use2dhist = boolean(default=True) # Use 2d histogram to find initial offset?
        separation = float(default=0.5) # Minimum object separation in arcsec
        tolerance = float(default=1.0) # Matching tolerance for xyxymatch in arcsec
        xoffset = float(default=0.0), # Initial guess for X offset in arcsec
        yoffset = float(default=0.0) # Initial guess for Y offset in arcsec
        fitgeometry = option('shift', 'rscale', 'general', default='general') # Fitting geometry
        nclip = integer(min=0, default=3) # Number of clipping iterations in fit
        sigma = float(min=0.0, default=3.0) # Clipping limit in sigma units
        align_to_gaia = boolean(default=False)  # Align to GAIA catalog
        gaia_catalog = option('GAIADR2', 'GAIADR1', default='GAIADR2')
        min_gaia = integer(min=0, default=5) # Min number of GAIA sources needed
        output_gaia = boolean(default=False)  # Write out GAIA catalog as a separate product
    """


    reference_file_types = []

    def process(self, input):

        try:
            images = datamodels.ModelContainer(input)
        except TypeError as e:
            e.args = ("Input to tweakreg must be a list of DataModels, an "
                      "association, or an already open ModelContainer "
                      "containing one or more DataModels.", ) + e.args[1:]
            raise e

        # Build the catalogs for input images
        for image_model in images:
            catalog = make_tweakreg_catalog(
                image_model, self.kernel_fwhm, self.snr_threshold,
                brightest=self.brightest, peakmax=self.peakmax
            )

            # filter out sources outside the image array if WCS validity
            # region is provided:
            wcs_bounds = image_model.meta.wcs.pixel_bounds
            if wcs_bounds is not None:
                ((xmin, xmax), (ymin, ymax)) = wcs_bounds
                xname = 'xcentroid' if 'xcentroid' in catalog.colnames else 'x'
                yname = 'ycentroid' if 'ycentroid' in catalog.colnames else 'y'
                x = catalog[xname]
                y = catalog[yname]
                mask = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
                catalog = catalog[mask]

            filename = image_model.meta.filename
            nsources = len(catalog)
            if nsources == 0:
                self.log.warning('No sources found in {}.'.format(filename))
            else:
                self.log.info('Detected {} sources in {}.'
                              .format(len(catalog), filename))

            if self.save_catalogs:
                catalog_filename = filename.replace(
                    '.fits', '_cat.{}'.format(self.catalog_format)
                )
                if self.catalog_format == 'ecsv':
                    fmt = 'ascii.ecsv'
                elif self.catalog_format == 'fits':
                    # NOTE: The catalog must not contain any 'None' values.
                    #       FITS will also not clobber existing files.
                    fmt = 'fits'
                else:
                    raise ValueError(
                        '\'catalog_format\' must be "ecsv" or "fits".'
                    )
                catalog.write(catalog_filename, format=fmt, overwrite=True)
                self.log.info('Wrote source catalog: {}'
                              .format(catalog_filename))
                image_model.meta.tweakreg_catalog = catalog_filename

            image_model.catalog = catalog

        # Now use the catalogs for tweakreg
        if len(images) == 0:
            raise ValueError("Input must contain at least one image model.")

        # group images by their "group id":
        grp_img = images.models_grouped

        self.log.info('')
        self.log.info("Number of image groups to be aligned: {:d}."
                      .format(len(grp_img)))
        self.log.info("Image groups:")

        if len(grp_img) == 1:
            self.log.info("* Images in GROUP 1:")
            for im in grp_img[0]:
                self.log.info("     {}".format(im.meta.filename))
            self.log.info('')

            # we need at least two exposures to perform image alignment
            self.log.warning("At least two exposures are required for image "
                             "alignment.")
            self.log.warning("Nothing to do. Skipping 'TweakRegStep'...")
            self.skip = True
            for model in images:
                model.meta.cal_step.tweakreg = "SKIPPED"
            return input

        # create a list of WCS-Catalog-Images Info and/or their Groups:
        imcats = []
        for g in grp_img:
            if len(g) == 0:
                raise AssertionError("Logical error in the pipeline code.")
            else:
                group_name = _common_name(g)
                wcsimlist = list(map(self._imodel2wcsim, g))
                self.log.info("* Images in GROUP '{}':".format(group_name))
                for im in wcsimlist:
                    im.meta['group_id'] = group_name
                    self.log.info("     {}".format(im.meta['name']))
                imcats.extend(wcsimlist)

        self.log.info('')

        # align images:
        tpmatch = TPMatch(
            searchrad=self.searchrad,
            separation=self.separation,
            use2dhist=self.use2dhist,
            tolerance=self.tolerance,
            xoffset=self.xoffset,
            yoffset=self.yoffset
        )

        try:
            align_wcs(
                imcats,
                refcat=None,
                enforce_user_order=self.enforce_user_order,
                expand_refcat=self.expand_refcat,
                minobj=self.minobj,
                match=tpmatch,
                fitgeom=self.fitgeometry,
                nclip=self.nclip,
                sigma=(self.sigma, 'rmse')
            )

        except ValueError as e:
            msg = e.args[0]
            if (msg == "Too few input images (or groups of images) with "
                "non-empty catalogs."):
                # we need at least two exposures to perform image alignment
                self.log.warning(msg)
                self.log.warning("At least two exposures are required for "
                                 "image alignment.")
                self.log.warning("Nothing to do. Skipping 'TweakRegStep'...")
                self.skip = True
                for model in images:
                    model.meta.cal_step.tweakreg = "SKIPPED"
                return images

            else:
                raise e

        if self.align_to_gaia:
            try:
                # Get catalog of GAIA sources for the field
                #
                # NOTE:  If desired, the pipeline can write out the reference catalog
                #        as a separate product with a name based on whatever convention
                #        is determined by the JWST Cal Working Group
                if self.output_gaia:
                    output_name = 'fit_{}_ref.ecsv'.format(self.gaia_catalog.lower())
                else:
                    output_name = None
                ref_cat = amutils.create_astrometric_catalog(images,
                                                             self.gaia_catalog,
                                                             output=output_name)

                # Check that there are enough GAIA sources for a reliable/valid fit
                num_ref = len(ref_cat)
                if num_ref < self.min_gaia:
                    msg = "Not enough GAIA sources for a fit: {}\n".format(num_ref)
                    msg += "Skipping alignment to {} astrometric catalog!\n".format(self.gaia_catalog)
                    raise ValueError(msg)
                # Set group_id to same value so all get fit as one observation
                # The assigned value, 987654, has been hard-coded to make it
                # easy to recognize when alignment to GAIA was being performed
                # as opposed to the group_id values used for relative alignment
                # earlier in this step.
                for imcat in imcats:
                    imcat.meta['orig_group_id'] = imcat.meta['group_id']
                    imcat.meta['group_id'] = 987654

                # Perform fit
                align_wcs(
                    imcats,
                    refcat=ref_cat,
                    enforce_user_order=False,
                    expand_refcat=False,
                    minobj=self.minobj,
                    match=tpmatch,
                    fitgeom=self.fitgeometry,
                    nclip=self.nclip,
                    sigma=(self.sigma, 'rmse')
                )
                # Reset group_id to original values
                # Also, update/create the WCS .name attribute with information on this astrometric fit
                for imcat in imcats:
                    imcat.meta['group_id'] = imcat.meta['orig_group_id']
                    # NOTE: This .name attribute needs to be defined using a convention
                    #       agreed upon by the JWST Cal Working Group.
                    #       Current value is merely a place-holder based on HST conventions
                    #       This value should also be translated to the FITS WCSNAME keyword
                    #       IF that is what gets recorded in the archive for end-user searches
                    imcat.meta.wcs.name = "FIT-LVL3-{}".format(self.gaia_catalog)

            except ValueError as e:
                msg = e.args[0]
                # Warn the user that the FIT to the astrometric catalog was not successful.
                self.log.warning(msg)


        for imcat in imcats:
            imcat.meta['image_model'].meta.cal_step.tweakreg = 'COMPLETE'
            # retrieve fit status and update wcs if fit is successful:
            fit_info = imcat.meta.get('fit_info')
            if fit_info['status'] in 'SUCCESS':
                imcat.meta['image_model'].meta.wcs = imcat.wcs

        return images

    def _imodel2wcsim(self, image_model):
        # make sure that we have a catalog:
        if hasattr(image_model, 'catalog'):
            catalog = image_model.catalog
        else:
            catalog = image_model.meta.tweakreg_catalog

        model_name = path.splitext(image_model.meta.filename)[0].strip('_- ')

        if isinstance(catalog, Table):
            if not catalog.meta.get('name', None):
                catalog.meta['name'] = model_name

        else:
            try:
                cat_name = str(catalog)
                catalog = Table.read(catalog, format='ascii.ecsv')
                catalog.meta['name'] = cat_name
            except IOError:
                self.log.error("Cannot read catalog {}".format(catalog))

        if 'xcentroid' in catalog.colnames:
            catalog.rename_column('xcentroid', 'x')
            catalog.rename_column('ycentroid', 'y')

        # create WCSImageCatalog object:
        refang = image_model.meta.wcsinfo.instance
        im = JWSTgWCS(
            wcs=image_model.meta.wcs,
            wcsinfo={'roll_ref': refang['roll_ref'],
                     'v2_ref': refang['v2_ref'],
                     'v3_ref': refang['v3_ref']},
            meta={'image_model': image_model, 'catalog': catalog,
                  'name': model_name}
        )

        return im


def _common_name(group):
    file_names = [path.splitext(im.meta.filename)[0].strip('_- ')
                  for im in group]
    fname_len = list(map(len, file_names))
    assert all(fname_len[0] == l for l in fname_len)
    cn = path.commonprefix(file_names)
    assert cn
    return cn
