#
#  Module for  applying straylight correction.
# The routine correct_MRS applues a straylight correction to MRS science
# slope images. The straylight mask contains 0's for science regions and
# 1's for gaps between slices.
#
# there are two algorithms in the module
# the default one that is applied is correct_MRS_ModShepard
# correct_MRS was retained for testing purposes and can be removed
# after it is confirmed the new algorithm is better are removing the
# straylight

import numpy as np
import logging
import math
from .. import datamodels
from ..datamodels import dqflags
from astropy.convolution import convolve, Box2DKernel
#from matplotlib import pyplot as plt


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def correct_MRS(input_model, straylight_model):
    """
    Short Summary
    -------------
    Corrects the MIRI MRS data for straylight

    Parameter
    ----------
    input_model: data model object
        science data to be corrected

    straylight_model: holds the straylight mask for the correction

    Returns
    -------
    output: data model object
        straylight-subtracted science data

    """

    # Save some data parameterss for easy use later
    nrows, ncols = input_model.data.shape
    # mask is either 1 or 0 
    mask = straylight_model.data
 #   plt.imshow(mask)
 #   plt.show()

    x = float('nan')
    # The straylight mask has values of 1 and 0. The straylight task uses data
    # in-between the slices (also called slice gaps) of the MRS data to correct 
    # the science data in the slices. In the mask the pixels found in the gaps
    # have a value of 1 and the science pixels have a value of 0.

    # Create output as a copy of the input science data model
    # sci_mask is the input science image * mask
    # science regions  = 0 (reference pixel are  also = 0)

# used for testing and putting in a hot pixel 
#    drow = 412
#    dx1 = 470
#    dx2 = 480

    output = input_model.copy() # this is used in algorithm to
    # find the straylight correction.

    #_________________________________________________________________
    # if there are nans remove them because they mess up the correction
    index_inf = np.isinf(output.data).nonzero()
    output.data[index_inf] = 0.0
    mask[index_inf] = 0  # flag associated mask so we do not  use any
                         # slice gaps that are nans, now data=0 . 
    #_________________________________________________________________
    # flag bad pixels 
    mask_dq = input_model.dq.copy()# * mask # find DQ flags of the gap values 

    all_flags = (dqflags.pixel['DO_NOT_USE'] + 
                 dqflags.pixel['DEAD'] + dqflags.pixel['HOT'])
    
    # where are pixels set to any one of the all_flags cases 
    testflags = np.bitwise_and(mask_dq,all_flags)
    # where are testflags ne 0 and mask == 1
    bad_flags = np.where(testflags !=0)
    mask[bad_flags]  = 0 
    #_________________________________________________________________
    sci_mask = output.data * mask    #sci_maskcontains 0's in science regions of detector.
    straylight_image = output.data * 0.0

    #We Want Sci mask smoothed for GAP region with 3 X 3 box car filter
    # Handle edge cases for boxcar smoothing, by determining the
    # boxcar smoothing of the mask.

    sci_ave = convolve(sci_mask, Box2DKernel(3))
    mask_ave = convolve(mask, Box2DKernel(3))

    # catch /0 cases
    index = np.where(mask_ave == 0) # zero catches cases that would be #/0
                                   # near edges values are 0.3333 0.6667 1
    sci_ave[index] = 0
    mask_ave[index] = 1
    sci_smooth = sci_ave / mask_ave

    x = np.arange(ncols)

    # Loop over each row (0 to 1023)
    for j in range(nrows):
        row_mask = mask[j, :]
        if(np.sum(row_mask) > 0):
            # find the locations of slice gaps
            #determine the data in the slice gaps
            yuse = sci_smooth[j, np.where(row_mask == 1)]

            #find the x locations of the slice gaps
            xuse = x[np.where(row_mask == 1)]
            #if(j ==1):
            #    print 'row ',j+1,yuse.shape,yuse
            #    print 'xuse',xuse,xuse.shape,type(xuse)

            #find data in same gap area
            nn = len(xuse)
            #dx difference in adjacent slice gaps pixels --> used
            # to find x limits of each gap
            dx = xuse[1:nn] - xuse[0:nn - 1]

            # Find the number of slice gaps in  row
            idx = np.asarray(np.where(dx > 1))
            ngroups = len(idx[0]) + 1

            # xlimits are the x values that mark limits of a slice gaps
            xlimits = np.zeros(ngroups + 1)
            i = 1
            xlimits[0] = xuse[0]
            for index in idx[0]:
                xlimits[i] = xuse[index]
                i = i + 1

            xlimits[ngroups] = xuse[nn - 1]#+1
            #if(j ==1):
            #    print '# xlimits',ngroups
            #    print 'xlimits',xlimits


            xg = np.zeros(ngroups)
            yg = np.zeros(ngroups)
            # loop over all slice gaps in a row
            # find the mean y straylight value for each slice gaps
            # also find the mean x value for this slice gap
            for i in range(ngroups):
                lower_limit = xlimits[i]
                upper_limit = xlimits[i + 1]
                igap = np.asarray(np.where(np.logical_and(xuse > lower_limit, xuse <= upper_limit)))
                ymean = np.mean(yuse[0, igap[0]])
                xmean = np.mean(xuse[igap[0]])
                yg[i] = ymean
                xg[i] = xmean
        # else  entire row is zero
        else:
            xg = np.array([0, 1032])
            yg = np.array([0, 0])
       #using mean y value in slice gaps and x location of ymean
       #interpolate what straylight contribution based on yg and xg
       # for all points in row

        for k in x:
            if(x[k] >= xg[0] and x[k] <= xg[-1]):
                ynew = np.interp(x[k], xg, yg);
            else:
                if x[k] < xg[0]:
                    ynew = yg[0] + (x[k] - xg[0]) * (yg[1] - yg[0]) / (xg[1] - xg[0])
                elif x[k] > xg[-1]:
                    ynew = yg[-1] + (x[k] - xg[-1]) * (yg[-1] - yg[-2]) / (xg[-1] - xg[-2])
            straylight_image[j, k] = ynew

        # end loop over rows

    straylight_image[straylight_image < 0] = 0

    #print 'straylight image',straylight_image[debug_row,0:40]
    # pull out the science region (1024 pixel/row) to do boxcar smoothing on

    simage = convolve(straylight_image, Box2DKernel(25))

    # remove the straylight correction for the reference pixels
    simage[:, 1028:1032] = 0.0
    simage[:, 0:4] = 0.0


    output.data = output.data - simage
    return output


def correct_MRS_ModShepard(input_model, region_model,roi,power):
    """
    Short Summary
    -------------
    Corrects the MIRI MRS data for straylight using a Modified version of the
    Shepard algorithm. Straylight is determined using an inverse distance weighting
    function. The inverse distance weighting is determined by module
    Shepard2DKernel(roi,power):

    Parameter
    ----------
    input_model: data model object
        science data to be corrected

    regions_model: holds the pixel region mask for the correction
                   slice = (band*100+slice#)
                   gap = 0
    roi: region of inflence (size of radius)
    power: exponent of Shepard kernel
    sci_ngroups:int
        number of groups in input data


    Returns
    -------
    output: data model object
        straylight-subtracted science data

    """

    # Save some data parameterss for easy use later
    nrows, ncols = input_model.data.shape
    # mask is either non 0 for slices and 0 for gaps between slices
    # this algorithm using the pixels from gaps between the slices for 
    # correction
    sliceMap = region_model.regions.copy()

    #plt.imshow(sliceMap)
    #plt.show()

    # The regions mask has values of 0 and nonzero, values of 0 present pixel
    # in a slice gap and non zero values are science pixels.  The straylight task uses data
    # in-between the slices (also called slice gaps) of the MRS data to correct 
    # the science data in the slices. 

    # Create output as a copy of the input science data model
    output = input_model.copy() # this is used in algorithm to

    # kernel matrix 
    w = Shepard2DKernel(roi,power)

    #mask is same size as sliceMap - set = 0 everywhere
    mask = np.zeros_like(sliceMap)
    #mask = 1 for slice gaps
    mask[sliceMap==0] = 1

    # test putting in a hot pixels and see what it does to algorithm
#    drow = 412
#    dx1 = 470
#    dx2 = 480
#    dxhot = dx1 + 6
#    output.data[drow,dxhot] = 2000.0
#    output.data[drow,dxhot] = 2000.0

    # find if any of the gap pixels are bad pixels - if so mark them

    mask_dq = input_model.dq.copy()# * mask # find DQ flags of the gap values 

    all_flags = (dqflags.pixel['DEAD'] + dqflags.pixel['HOT'])
    
    # where are pixels set to any one of the all_flags cases 
    testflags = np.bitwise_and(mask_dq,all_flags)
    # where are testflags ne 0 and mask == 1
    bad_flags = np.where((testflags !=0) & (mask==1))
    mask[bad_flags]  = 0 

    # apply mask to the data
    image_gap = output.data*mask

    #avoid cosmic ray contamination
    # only using the science data for this cosmic ray test
#    print('max',np.max(output.data[sliceMap>0]))
    cosmic_ray_test = 0.02 * np.max(output.data[sliceMap>0])
#    print('Cosmic ray test %12.8f',cosmic_ray_test)
    image_gap[image_gap>cosmic_ray_test] = 0
    image_gap[image_gap<0] = 0 #set pixels less than zero to 0
    image_gap= convolve(image_gap,Box2DKernel(3)) # smooth gap pixels 
    image_gap*=mask #reset science pixels to 0
    #we not not want the reference pixels to be used in the convolution
    image_gap[:, 1028:1032] = 0.0
    image_gap[:, 0:4] = 0.0

    #convolve gap pixel image with weight kernel
    astropy_conv = convolve(image_gap,w)

    #normalize straylight flux by weights
    norm_conv = convolve(mask,w)
    astropy_conv /= norm_conv


    # remove the straylight correction for the reference pixels
    astropy_conv[:, 1028:1032] = 0.0
    astropy_conv[:, 0:4] = 0.0
    output.data = output.data - astropy_conv

    return output    

#______________________________________________________________________    
def Shepard2DKernel(roi,power):
# determine the 2D Modified Shepard Kernel 

    """
    Calculates the kernel matrix of Shepard's modified algorithm
    roi: region of influence
    power: exponent
    """

    distance_tolerance = 0.001 # for very small distances set min distance
                               # so denominator does not -> 0 and make w invalid. 
    xk,yk = np.meshgrid(np.arange(-roi/2,roi/2+1), np.arange(-roi/2, roi/2+1))
    d = np.sqrt(xk**2+yk**2)
    dtol = np.where(d < distance_tolerance)
    d[dtol] = distance_tolerance

#    print('number < tolerance',dtol[0])
    w = (np.maximum(0,roi-d)/(roi*d))**power
    w[d==0]=0

    return w
                                   
