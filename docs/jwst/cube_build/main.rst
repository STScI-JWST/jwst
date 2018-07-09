Description
===========

This step takes  MIRI or NIRSPEC IFU calibrated 2D slope images and produces
3-D spectral cubes. The IFU slice distorted and disjointed 2D spectra are corrected
for distortion and put back together into a rectangular cube with three orthogonal axes: two
spatial and one spectral. The distortion information is  incorporated into the calibrated
images using the  calwebb_spec2 pipeline step ASSIGN_WCS. The detector pixel fluxes have already
been converted to surface brightness in the calwebb_spec2 pipeline step, PHOTOM.

The cube_build package can take as input either:

  * a single 2D input image

  * a model passed in containing 2D slope image

  * an association table (in json format) containing the list of exposures to combine

  * a model container holding several 2D slope data models.


There are a number of arguments the user can provide either in a configuration file or
on the command line that control the sampling size of the cube as well as the type of data that is combined to
create a cube. See the **Arguments** section in the documentation for more details.



Assumption
----------
It is assumed the assign_wcs step has been run on the data, attaching the distortion and pointing
information to the image. It is also assumed the photom step has converted the detector pixels from
units of flux to surface brightness. This step will only work with  MIRI or NIRSPEC IFU data.


Background Information
----------------------
The JWST integral field spectrographs obtain simultaneous spectral and spatial data on a relatively compact
region of the sky. The MIRI Medium Resolution Spectrometer (MRS) consists of four integral field units
providing four simultaneous and overlapping fields of view ranging from 3.3" X 3.7" to ~ 7.2" X 7.7" covering a
wavelength range from 5 to 28 microns. The optics system for the four IFUs is split into two paths. One path
is dedicated to the two short wavelength IFUs and the other one handles the two longer wavelength IFUs.
There is one 1024 X 1024 SCA for each path. Light entering the MRS is spectrally separated into four
channels by dichroic mirrors. Each of these channels has its own IFU that divides the image into several
slices. Each slice is then dispersed using a grating spectrograph and imaged on one half of a SCA. While
four channels are observed simultaneously, each exposure only records the spectral coverage of
approximately one third of the full wavelength range of each channel. The full 5 to 28 micron spectrum is then
obtained by making three exposures using three different gratings and three different dichroic sets.
We refer to a sub-channel as one of the three possible configurations (A/B/C) of the channel where each
sub-channel covers ~1/3 of the full spectrum for the channel. Each of the four channels have a different sampling
of the field, so the FOV, slice width, number of slices and plate scales are different for each channel.

The NIRSPEC IFU has a 3 X 3 arcsecond field of view that is sliced into thirty 0.1 arcsecond bands. Each slice is
dispersed by a prism or one of six diffraction gratings. When using diffraction gratings as dispersive elements three
separate gratings are employed in combination with specific filters in order to avoid the overlapping of spectra
caused by different grating orders. The three grating span four partially overlapping bands (1.0 - 1.8 microns;
1.7 - 3.0 microns; 2.9 - 5 microns) covering the total spectral range in four separate exposures.   Six gratings
provide high-resolution (R = 1400-3600) and medium resolution (R = 500-1300) spectroscopy over the wavelength
range 0.7 microns to 5 microns, while a prism yields lower-resolution (R = 30-300) spectroscopy over the range
0.6 microns to 5 microns.

The NIRSPEC detector focal plan consists of two HgCdTe sensor chip assemblies (SCAs). Each chip is a 2D array of 2048 X 2048
pixels. The light-sensitive portions of the two SCAS are separated by a physical gap of 3.144 mm which
corresponds to 17.8 arc seconds on the sky.  For low or medium resolution IFU data the 30 slices are imaged on
a single NIRSPEC SCA. In high resolution mode the 30 slices are imaged on the two NIRSPEC SCAs. The physical gap between the
SCAs causes a loss of spectral information over a range in wavelength that depends on the location of the target
and dispersive element used. The lost information can be recovered by dithering the targets.

Terminology
-----------

**We use the following terminology to define the spectral range divisions of MIRI:**

- *Channel*: The spectral range covered by each MIRI IFU. The channels are labeled as 1, 2, 3 or 4.
- *Sub-Channel*: The 3 sub-ranges that a channel is divided into. These are designated as *Short(A)*, *Medium(B)*, or *Long(C)*.
- *Band*:

  For **MIRI** band is one of the 12 contiguous wavelength intervals (four channels times three sub-channels each) into which
  the spectral range of the MRS is divided.  Each band (e.g., 1A, aka 1-SHORT) has a unique
  channel/sub-channel combination, i.e., the shortest wavelength range on MIRI is covered by Band 1-SHORT and the
  longest is covered by Band 4-LONG.



**NIRSPEC Optical Element and Filter possibilities for IFU mode:**

=======  ======  ====================
Grating  Filter  Wavelength (microns)
=======  ======  ====================
Prism    Clear   0.6 -5.3
G140M    F070LP  0.7 - 1.2
G140M    F100LP  1 - 1.8
G235M    F170LP  1.7 - 3.1
G395M    F290LP  2.9 - 5.2
G140H    F070LP  0.7 - 1.2
G140H    F100LP  1 - 1.8
G235H    F170LP  1.7 - 3.1
G395H    F290LP  2.9 - 5.2
=======  ======  ====================

For NIRSPEC we have defined a *band*  as a  single grating-filter combination, i.e. G140M-F070LP.

**Coordinate Systems:**

An integral field spectrograph measures the intensity of a region of the sky as a function of
wavelength. There are a number of different coordinate systems used in the cube building process. Here is an
overview of these coordinate systems:

- *Detector System*: is defined by the hardware and presents raw detector pixel values. Each detector or SCA
  will have its own pixel-based coordinate system. In the case of MIRI we have two detector systems because
  the the MIRI IFUs disperse data onto two SCAs.

- *Telescope (V2,V3)* : the V2,V3 coordinates locate points on  a spherical coordinate system. The frame is tied
  to JWST and applies to the whole field of view, encompassing all the instruments. The coordinates (V2,V3)
  are Euler angles in a spherical frame rather than Cartesian coordinates.

- *XAN,YAN*: similar to V2,V3 but flipped and shifted so the origin lies between the NIRCAM detectors instead of
   at the telescope boresight.
   Note what OSIM and OTE call 'V2,V3' are actually XAN,YAN.

- *Absolute* is the standard astronomical equatorial system of Ra-Dec.

- *Cube* is a three dimensional system with two spatial axes and one spectral axis.

- *MRS-FOV* this a MIRI specific system which is the angular coordinate system attached to the FOV of each MRS band.
  There are twelve MRS-FOV systems
  for MIRI, since there are twelve bands (1A, 1B, 1C,... 4C). Each system has two orthogonal axes, one parallel
  (**alpha**) and the other perpendicular (**beta**) to the projection of the long axes of the slices in the FOV.

Options which control what type of IFU cube to build.
-----------------------------------------------------
The input to cube build can be a single exposure or a set of exposures. There are a number of user options that control the
type of IFU Cube to create. For standard pipeline processing in calwebb_spec3, default settings are used and the output is a set of single
band IFU cubes. In the case of MIRI the standard IFU cubes will be single channel, single sub-channel cubes (e.g., 1A) and in
the case of NIRSPEC the standard output will be be single grating, single filters cubes. Since a single MIRI exposure
always covers two channels, there will at least be two IFU cubes as
the standard output.  The calwebb_spec2 pipeline produces intermediate cubes which are single IFU cubes for a single exposure.
In these intermediate cubes, the MIRI IFU spectral cube  contains two channels.

Below is a list of the user options that can be used to select the type of data to be used to create the IFU Cube:

- ``--channel #``

This is a MIRI only option and the only valid values for # are 1,2,3,4, or ALL.
If the ``channel`` argument is given, then only data corresponding to that channel  will be used in
constructing the cube.  If the user wants more than one  channel in the output spectral cube, then all the values are
contained in a comma separated string string. For example, to create a cube with channel 1 and 2 the argument list is
``--channel='1, 2'``. If this value is not specified the output will be a set of IFU Cubes for each channel/sub-channel combination
contained in the input data.

- ``--band [string]``

This is a MIRI option and the  only valid values  are SHORT,MEDIUM,LONG, or ALL.
If the ``band`` argument is given, then only data corresponding
to that sub-channel will be used in  constructing the cube. Only one option is possible, so IFU cubes are created either
per sub-channel or using all the sub-channels the input data cover.  If this value is not specified a set of IFU cubes are created
for each band. Note we used ``band`` instead of
``subchannel``, because the keyword ``band`` in the science fits is used to denote which MIRI subchannel the data covers.


* ``--grating [string]``

This is a NIRSPEC option and only valid values are PRISM, G140M, G140H, G235M, G235H, G395M, G395H, or ALL.
If the option ALL is used then all the gratings in the association are used.
Since association tables will only contain exposures of the same resolution, the use of ALL, will at most combine
data from grating G140M, G235M & G395M or G140H, G235H & G395H together. The user can supply a comma separated string
containing the gratings to use.

- ``--filter [string]``

This is a NIRSPEC  option and the only valid options are Clear, F100LP, F070LP, F170LP, F290LP, or ALL.
To cover the full wavelength range of NIRSPEC the option ALL can be used (provided the exposures in the association table
contain all the filters). The user can supply a comma separated string containing the filters to use.

- ``--output_type [string]``

This parameter has four valid options Band, Channel, Grating and Multi. The parameters can be combined
with the options above  [--band, --channel,--grating, and --filter] to fully control the type of IFU
cubes to make.

	 - ``--output_type=Band`` is the default mode and creates IFU cubes containing only one band
	   (channel/sub-channel or  grating/filter combination).

	 - ``--output_type = channel`` combines all the MIRI channels in the data or set by the
	   --channel option into a single IFU cube.

	 - ``--output_type = grating `` combines all the grating in the NIRSPEC data or set by the
	   --grating option into a single IFU cube.

	 - ``--output_type = multi`` combines data  into a single uber IFUCube. If in addition,
	   --channel, --band, --grating, or -filter are also set then only the data set by those
	   parameters will be combined into an uber cube.


- ``--weighting ['string]``

This is for MIRI data and the only valid values are STANDARD and MIRPSF. This option defines
how the distances between the point cloud members and spaxel centers are determined. The default value is STANDARD and the distances
are determined in the cube output coordinate system. If this parameter is set to MIRIPSF then the distances are determined in
the alpha-beta coordinate system of the point cloud member and are normalized by the PSF and LSF.

Output Format
-------------
The FITS files of spectral cubes consist of 4 IMAGE extensions and 1 ASDF table. The FITS primary data array is empty and the
primary header  holds the basic parameters of the observations by holding values for the first exposure used
to build the spectral cube. The 4 IMAGE extensions contain the following:

=======  =====  ========================  =========
EXTNAME  NAXIS  Dimensions                Data type
=======  =====  ========================  =========
SCI      3      2 spatial and 1 spectral  float
ERR      3      2 spatial and 1 spectral  float
DQ       3      2 spatial and 1 spectral  integer
WMAP     3      2 spatial and 1 spectral  integer
=======  =====  ========================  =========

The SCI image hold the surface brightness of the cube spaxel in units of mJy/arcsecond^2. The ERR image contains the
error on the surface brightnesses, the DQ image contains the data quality flag for each spaxel and the WMAP image contains
the number of point cloud elements contained in the region of interest of the spaxel.


Output Product Name
```````````````````
If the input data is passed as an Image Model then the IFU cube will be passed back as an IFU cube model. The IFU Cube will be
written to disk at the end of the calspec3  pipeline. In addition, if the user is running the cube_build pipeline
using the 'strun' pipeline methods the IFUCube will also be written to disk. The output name is based on a rootname plus a
string defining the type of IFU cube created plus the string 's3d.fits'.
If the input data is a single exposure then the rootname
is formed from the input filename; while if the input is an association table the rootname is defined in the association
table.
The string defining the type of IFU is created according to the following rules:

- for MIRI the output string name  is determined from the  channels and sub-channels used.
  The  IFU string for MIRI is 'ch'+ channel numbers used plus a string for the subchannel. For example if the IFU cube
  contains channel 1 and 2 data for the short subchannel, the output name would be, rootname_ch1-2_SHORT_s3d.fits.
  If all the sub-channels were used then the output name would be rootname_ch-1-2_ALL_s3d.fits.

- for NIRSPEC the output string is determined from the gratings and filters used. The gratings are grouped together in a dash (-)
  separated string and likewise for the filters. For example if the IFU cube contains data from
  grating G140M and G235M and from filter F070LP and F100LP,  the output name would be,
  rootname_G140M-G225_F070LP-F100LP_s3d.fits


Algorithm
---------
The default IFU Cubes contain data from a single band (channel/sub-channel or grating/filter). There are several
options which control the type of cubes to create (see description given above).
Based on the arguments defining the type of cubes to create, the program selects the data from
each exposure that should be included in the spectral cube. The output cube is defined using the WCS information of all
the included  input data.
This output cube WCS defines a field-of-view that encompasses the undistorted footprints on
the sky of all the input images. The output sampling scale in all three dimensions for the cube
is defined by a 'cubepars' reference file as a function of wavelength, and can also be changed by the user.
The cubepars reference file contains a predefined scale to use
for each dimension for each band. If the output IFU cube contains more than one band, then  for MIRI the
output scale corresponds to the channel with the smallest scale. In the case of NIRSPEC only gratings of the
same resolution are combined together in an IFU cube. The output spatial coordinate system is right ascension-declination.


All the pixels on each exposure that are included are mapped to the cube coordinate system. This input-to-output
pixel mapping is determined via a mapping function derived from the WCS of each input image and the WCS of output cube. The
mapping process corrects for the optical distortions and uses the spacecraft telemetry information to map each pixel location
to its projected location in the cube coordinate system. The mapping is actually a series of chained transformations
(detector -> alpha-beta-lambda), (alpha-beta-lambda -> v2-v3-lambda), (v2-v3-lambda - > right ascension-declination-lambda),
and (right ascension-declination-lambda -> Cube coordinate1-Cube Coordinate2-lambda).  The reverse of each transformation
is also possible.

The mapping process results in an irregular spaced "cloud of points" that sample the specific intensity
distribution at a series of locations on the sky. A schematic of this process is shown
in Figure 1.

.. figure:: pointcloud.png
   :scale: 50%
   :align: center

Figure 1: Schematic of two dithered exposures mapped to the IFU output coordinate system (black regular grid).
The plus symbols represent the point cloud mapping of detector pixels to effective sampling locations
relative to the output coordinate system at a given wavelength. The black points are from exposure one and the red points
are from exposure two.

Each point in the cloud represents a measurement of the specific intensity (with corresponding uncertainty)
of the astronomical scene at a particular location.  The final data cube is constructed by combining each of the
irregularly-distributed samples of the scene into a regularly-sampled grid in three dimensions for which each
**spaxel** (i.e., a spatial pixel in the cube) has a spectrum composed of many spectral elements.

The best algorithm with which to combine the irregularly-distributed samples of the point cloud to a rectilinear
data cube is the subject of ongoing study, and depends on both the optical characteristics of the IFU and
the science goals of a particular observing program.  At present, the default method uses a flux-conserving
variant of Shepards method in which the value of a given element of the cube is a distance-weighted average
of all point-cloud members within a given region of influence.  In order to explain this method we will introduce the follow definitions:

* xdistance = distance between point in the cloud and spaxel center  in units of arc seconds along the x axis
* ydistance = distance between point in the cloud and spaxel center in units of arc seconds along the y axis
* zdistance = distance between point cloud and spaxel center in the lambda dimension in units of microns along the wavelength axis

These distances are then normalized by the IFU cube sample size for the appropriate axis:

* xnormalized = xdistance/(cube sample size in x dimension [cdelt1])
* ynormalized = ydistance/(cube sample size in y dimension [cdelt2])
* znormalized = zdistance/(cube sample size in z dimension [cdelt3])

The final spaxel value at a given wavelength is determined as the weighted sum of the point cloud members with a spatial and
spectral region of influence centered on the spaxel.
The default size of the region of influence is defined in the cubepar reference file, but can be changed by the
user with the options: ``rois`` and ``roiw``.

If *n* point cloud members are located within the ROI of a spaxel, the  spaxel flux K =
:math:`\frac{ \sum_{i=1}^n Flux_i w_i}{\sum_{i=1}^n w_i}`

where

:math:`w_i =\frac{1.0} {\sqrt{({xnormalized}_i^2 + {ynormalized}_i^2 + {znormalized}_i^2)^{p} }}`


The default value for *p* is  2, although the optimal choice for this value (along with the size of the region of influence
and the cube sampling scale) is still under study.  Similarly, other algorithms such as a 3d generalization of the drizzle algorithm
are also being studied and may provide better performance for some science applications.

*Additional constraints for MIRI data if --weighting=MIRIPSF*

For MIRI the weighting function can be adapted to use the  width  of the PSF and LSF in weighting the point cloud members within the ROI
centered on the spaxel.  The width of the MIRI PSF varies with wavelength, broader for longer wavelengths.
The resolving power of  the MRS  varies with wavelength and band.  Adjacent point-cloud elements may in fact originate from
different exposures rotated from one another and even from different spectral bands. In order to properly weight the MIRI data  the
distances  between the point cloud element and spaxel the distances are determined in the alpha-beta coordinate system and
then normalized by the width of the PSF and the LSF.  To weight in the alpha-beta coordinates system each cube spaxel center must be
mapped to the alpha-beta system corresponding to the channel-band of the point cloud member. The xdistance and ydistances are redefined
to mean:

* xdistance = distance between point in the cloud and spaxel center along the alpha dimension in units of arc seconds
* ydistance = distance between point in the cloud and spaxel center along the beta dimension in units of arc seconds
* zdistance = distance between point cloud and spaxel center in the lambda dimension in units of microns along the wavelength axis

The spatial distances are then normalized by PSF width and the spectral distance is normalized by the LSF:

* xnormalized = xdistance/(width of the PSF in the alpha dimension in units of arc seconds)
* ynormalized = ydistance/(width of the PSF in the beta dimension  in units of arc seconds)
* znormalized = zdistance/( width of LSF in lambda dimension in units of microns)

