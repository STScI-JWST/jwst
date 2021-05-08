.. _faq:

Frequently Asked Questions
==========================

How do I install the JWST pipeline?
-----------------------------------

The latest public release can be installed via pip.
::

	$ pip install jwst

To work with the absolute most up-to-date version of the pipeline, the `main` branch of the Github repository can be installed. 

:: 

	$ pip install git+https://github.com/spacetelescope/jwst

See the `Github README <https://github.com/spacetelescope/jwst>`_ for more detailed installation instructions.

How do I run the JWST pipeline?
-------------------------------

There are two options for running the JWST pipeline or individual pipeline steps.

1. In Python. Pipelines and steps can be imported, configured, and run in a Python session. See `here <https://jwst-pipeline.readthedocs.io/en/latest/jwst/introduction.html#running-from-within-python>`_ for more information.

2. Using the command line interface ``stpipe`` - ``strun`` is an alias for ``stpipe run``. 

Both of these options access the same underlying code, the choice of which to use is a matter of personal preference and convenience.

How do I find out which pipelines and steps exist and are available to run?
--------------------------------------------------------------------------

Running the pipeline requires you to know the exact class name of the pipeline or step that you wish to run (e.g Detector1Pipeline). Some pipelines have shorter aliases that can be used interchangeably with their full class names. The command line tool
::

   $ stpipe list

will show all available pipelines and steps and their aliases. Additionally, a list of all pipelines and aliases is available `here <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/main.html#pipelines>`_.

What files does the pipeline run on?
------------------------------------

The pipeline either takes a single ``.fits`` file, or an association (``.asn``) file which represents a group of related exposures, as input to process.


What are the naming convention for fits file suffixes (i.e 'uncal.fits' 'i2d.fits')?
------------------------------------------------------------------------------------

Separate from the JWST data `naming conventions <https://jwst-pipeline.readthedocs.io/en/latest/jwst/associations/jwst_conventions.html>`_, the suffix of each file (e.g 'uncal', 'rate') provides information about which processing steps were performed to produce that file. See `here <https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/science_products.html>`_ for a description of each possible file suffix that the pipeline can produce. 

How do I save the results from running the pipeline to a fits file?
-------------------------------------------------------------------

When running the pipeline in Python, the default behavior for pipelines/steps that have a single output is to only return the final resulting `DataModel` in memory. For example, running

:: 
	
	result = Detector1Pipeline.call('jw42424001001_01101_00001_nrca5_uncal.fits')

will not write out a ``rate.fits`` file - that file is represented by the returned ``result`` which is a ``DataModel``. 

If you want to write out the result as well, the `save_results` argument can be used.

::

	result = Detector1Pipeline.call('jw42424001001_01101_00001_nrca5_uncal.fits', save_results=True)

By default, the base filename of the input file (jw42424001001_01101_00001_nrca5) will be the base name of the output file. The name of the output file can be customized with the ``output_file`` argument. 


When running the pipeline in Python, what is the difference between the 'pipe.run' and 'pipe.call' methods?
-----------------------------------------------------------------------------------------------------------

When you create a pipeline or step instance in Python, you will notice that there seemingly multiple methods available to call the pipeline. For example,

::

	pipe = Detector1Pipeline()

	pipe.run('input_file.fits')

	pipe('input_file.fits')

	pipe.call('input_file.fits')


The first two options - `.run` and calling the instance directly - are equivalent. The `.call` method however, which is the reccomended way to run the pipeline, is slightly different and involves some additional setup internally to allow it to seamlessly work with parameter files.

When the `.call` method is called on a pipeline instance, a new instance of that pipeline is created internally. The values in the parameter file are set as attributes on this new instance, the pipeline is run with these values, and then it is disposed of and the final result is returned. This is the reccomended way to run the pipeline since it is intended to be configured via parameter files.

When 'pipe.run' or simply 'pipe()' are called, the instance you created is directly used. So, any attributes set on that pipeline will be the ones used to direct the processing. The additional setup done in `call` to set the parameter file as
attributes on the pipeline is not done, you will have to set each pipeline parameter individually as an attribute on the pipeline instance you created before running it. For example, if you wanted to use `.run` and configure and call the `tweakreg` step, that would be done like this:

:: 

	pipe3 = Image3Pipeline()

	pipe3.brightest = 50
	pipe3.kernel_fwhm = 2.302
	pipe3.minobj = 15
	pipe3.nclip = 2
	pipe3.searchrad = 1.0
	pipe3.separation = 0.5
	pipe3.sigma = 3.0
	pipe3.snr_threshold = 5


	pipe3.run('jw42424001001_01101_00001_nrca5_cal.fits')

Wheras if you used `call`, you could just modify these values in a parameter file. If you wanted to change only one or two of these parameters, it is much easier to do so with a parameter file - if you set them directly you will have to set ALL of the parameters for that step to the default value in the parameter file, then you can change the ones you desire. 

In short, `call` is the reccomended way to use the pipeline and it uses parameter files to direct processing, while `run` requires you to do all that set up yourself. 

What is a parameter file?
-------------------------

Parameter files are ASDF format files that tell the pipeline which parameters should be used to run the pipeline. The JWST instrument teams create parameter files with the best set of pipeline parameters for the different observing modes - when pipeline-processed data is downloaded from MAST, or you run the pipeline yourself without any configuration, these are the 'default' parameter files that are used. Because parameter files can be time dependent either by nature (changes as the detector ages or changes), or due to improved understanding of the instrument, it is essential that they are version controlled. CRDS manages parameter files - when the pipeline is run, the CRDS software will determine the default parameter file associated with your dataset, just as it does with reference files.

The values in these files can be overridden in several ways - by providing your own parameter file, or overriding individual parameters when running the pipeline in Python or using the command line interface.

See `parameter files <https://jwst-pipeline.readthedocs.io/en/latest/jwst/introduction.html#parameter-files>`_ for more information.

What is a reference file?
-------------------------

Reference files are data files - seperate from the input data being processed - that the JWST pipeline requires to calibrate data and apply necessary corrections and calibrations to achieve science-ready data. An example of a reference file is a dark-current correction file, which is an array that represents the estimated dark-current for each pixel in an image. Each data set has a specific set of up-to-date reference files associated with it which supply the data for all the pipeline calibration steps.

Reference files are created and validated by the JWST instrument teams. Because many of these corrections are time dependent (e.g a monthly dark file), or are periodically updated and improved as understanding of the instrument improves, they must be version controlled to ensure users can access the exact set of files for a dataset as well as revert back to previous versions if needed. Managing these files and determining the exact set of reference files to apply to any given data file is not a trivial task: the CRDS (Calibration Reference Data System) manages these intricacies and is the interface for obtaining and managing pipeline reference files.

See `reference files <https://jwst-pipeline.readthedocs.io/en/latest/jwst/introduction.html#reference-files>`_ for more information.

