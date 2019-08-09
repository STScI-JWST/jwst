"""
Unit tests for straylight correction
"""

from jwst.datamodels import IFUImageModel
from jwst.straylight import StraylightStep
from jwst.straylight.straylight import correct_mrs_modshepard, shepard_2d_kernel
from jwst.pipeline.collect_pipeline_cfgs import collect_pipeline_cfgs
import numpy as np



def test_correct_mrs_modshepard():
    """ Test Correct Straylight routine gives expected results for small region """

    image = IFUImageModel((16,16))
    image.data = np.ones((16,16)) + 30.0

    slice_map = np.ones((16,16))
    # create 2 slice gaps

    image.data[8,8] = 45.6 # set to pass second cr check easily
    slice_map[:,4:7] = 0
    slice_map[:,10:13] = 0
    image.data[:,4:7] = 0.5
    image.data[:,10:13] = 0.5
    roi = 8
    power = 1

    result = correct_mrs_modshepard(image, slice_map, roi, power)
    compare = np.zeros((16,16))
    compare[0,:] = [3.0696348e+01, 3.0638458e+01, 3.0652958e+01, 3.0663357e+01, 2.7708188e-01,
                    1.6659760e-01, 2.7704430e-01, 3.0657593e+01, 3.0652958e+01, 3.0657593e+01,
                    2.7704430e-01, 1.6659760e-01, 2.7708188e-01, 3.0663357e+01, 3.0652958e+01,
                    3.0638458e+01]
    compare[1,:] = [3.0688972e+01, 3.0629198e+01, 3.0642084e+01, 3.0645052e+01, 1.6645306e-01,
                    1.1563802e-03, 1.6646616e-01, 3.0641420e+01, 3.0642084e+01, 3.0641420e+01,
                    1.6646616e-01, 1.1563802e-03, 1.6645306e-01, 3.0645052e+01, 3.0642084e+01,
                    3.0629198e+01]
    compare[2,:] = [3.0682684e+01, 3.0620981e+01, 3.0632751e+01, 3.0632690e+01, 1.6629149e-01,
                    1.1218189e-03, 1.6630261e-01, 3.0629959e+01, 3.0632751e+01, 3.0629959e+01,
                    1.6630261e-01, 1.1218189e-03, 1.6629149e-01, 3.0632690e+01, 3.0632751e+01,
                    3.0620981e+01]
    compare[3,:] = [3.0677645e+01, 3.0614532e+01, 3.0626221e+01, 3.0626003e+01, 1.6621163e-01,
                    1.1279561e-03, 1.6622025e-01, 3.0623276e+01, 3.0626221e+01, 3.0623276e+01,
                    1.6622025e-01, 1.1279561e-03, 1.6621163e-01, 3.0626003e+01, 3.0626221e+01,
                    3.0614532e+01]
    compare[4,:] = [3.0673761e+01, 3.0609789e+01, 3.0621811e+01, 3.0622076e+01, 1.6616508e-01,
                    1.1363822e-03, 1.6617118e-01, 3.0619116e+01, 3.0621811e+01, 3.0619116e+01,
                    1.6617118e-01, 1.1363822e-03, 1.6616508e-01, 3.0622076e+01, 3.0621811e+01,
                    3.0609789e+01]
    compare[5,:] = [3.0666666e+01, 3.0601625e+01, 3.0614611e+01, 3.0616297e+01, 1.6612101e-01,
                    1.0898784e-03, 1.6612145e-01, 3.0612762e+01, 3.0614611e+01, 3.0612762e+01,
                    1.6612145e-01, 1.0898784e-03, 1.6612101e-01, 3.0616297e+01, 3.0614611e+01,
                    3.0601625e+01]

    compare[6:8,:]= compare[5,:]
    compare[8,:] = [3.0666666e+01, 3.0601625e+01, 3.0614611e+01, 3.0616297e+01, 1.6612101e-01,
                    1.0898784e-03, 1.6612145e-01, 3.0612762e+01, 4.5214607e+01, 3.0612762e+01,
                    1.6612145e-01, 1.0898784e-03, 1.6612101e-01, 3.0616297e+01, 3.0614611e+01,
                    3.0601625e+01]

    compare[9:11,:]= compare[5,:]

    compare[11,:] = [3.0673761e+01, 3.0609789e+01, 3.0621811e+01, 3.0622076e+01, 1.6616508e-01,
                     1.1363822e-03, 1.6617118e-01, 3.0619116e+01, 3.0621811e+01, 3.0619116e+01,
                     1.6617118e-01, 1.1363822e-03, 1.6616508e-01, 3.0622076e+01, 3.0621811e+01,
                     3.0609789e+01]
    compare[12,:] = [3.0677645e+01, 3.0614532e+01, 3.0626221e+01, 3.0626003e+01, 1.6621163e-01,
                     1.1279561e-03, 1.6622025e-01, 3.0623276e+01, 3.0626221e+01, 3.0623276e+01,
                     1.6622025e-01, 1.1279561e-03, 1.6621163e-01, 3.0626003e+01, 3.0626221e+01,
                     3.0614532e+01]
    compare[13,:] =  [3.0682684e+01, 3.0620981e+01, 3.0632751e+01, 3.0632690e+01, 1.6629149e-01,
                      1.1218189e-03, 1.6630261e-01, 3.0629959e+01, 3.0632751e+01, 3.0629959e+01,
                      1.6630261e-01, 1.1218189e-03, 1.6629149e-01, 3.0632690e+01, 3.0632751e+01,
                      3.0620981e+01]
    compare[14,:] = [3.0688972e+01, 3.0629198e+01, 3.0642084e+01, 3.0645052e+01, 1.6645306e-01,
                     1.1563802e-03, 1.6646616e-01, 3.0641420e+01, 3.0642084e+01, 3.0641420e+01,
                     1.6646616e-01, 1.1563802e-03, 1.6645306e-01, 3.0645052e+01, 3.0642084e+01,
                     3.0629198e+01]
    compare[15,:] = [3.0696348e+01, 3.0638458e+01, 3.0652958e+01, 3.0663357e+01, 2.7708188e-01,
                     1.6659760e-01, 2.7704430e-01, 3.0657593e+01, 3.0652958e+01, 3.0657593e+01,
                     2.7704430e-01, 1.6659760e-01, 2.7708188e-01, 3.0663357e+01, 3.0652958e+01,
                     3.0638458e+01]

    assert(np.allclose(compare, result.data))


def test_shepard_kernel():
    """ Test forming kernel gives expected results"""
    power = 1
    roi = 6
    wkernel = shepard_2d_kernel(roi,power)

    kcompare = np.array( [[0.0690355937, 0.110683431, 0.149561099, 0.166666667,
                 0.149561099, 1.10683431e-01, 6.90355937e-02],
                [1.10683431e-01, 1.86886724e-01, 2.80546929e-01, 3.33333333e-01,
                 2.80546929e-01, 1.86886724e-01, 1.10683431e-01],
                [1.49561099e-01, 2.80546929e-01, 5.40440115e-01, 8.33333333e-01,
                 5.40440115e-01, 2.80546929e-01, 1.49561099e-01],
                [1.66666667e-01, 3.33333333e-01, 8.33333333e-01, 9.99833333e+02,
                 8.33333333e-01, 3.33333333e-01, 1.66666667e-01],
                [1.49561099e-01, 2.80546929e-01, 5.40440115e-01, 8.33333333e-01,
                 5.40440115e-01, 2.80546929e-01, 1.49561099e-01],
                [1.10683431e-01, 1.86886724e-01, 2.80546929e-01, 3.33333333e-01,
                 2.80546929e-01, 1.86886724e-01, 1.10683431e-01],
                [6.90355937e-02, 1.10683431e-01, 1.49561099e-01, 1.66666667e-01,
                 1.49561099e-01, 1.10683431e-01, 6.90355937e-02]])
    assert(np.allclose(wkernel, kcompare))

def test_shepard_kernel2():
    """ Test forming kernel gives expected results with power 2, roi 4"""
    power = 2
    roi = 4
    wkernel = shepard_2d_kernel(roi,power)

    kcompare = np.array([[1.07233047e-02, 3.88932023e-02, 6.25000000e-02,
                          3.88932023e-02, 1.07233047e-02],
                         [3.88932023e-02, 2.08946609e-01, 5.62500000e-01,
                          2.08946609e-01, 3.88932023e-02],
                         [6.25000000e-02, 5.62500000e-01, 9.99500062e+05,
                          5.62500000e-01, 6.25000000e-02],
                         [3.88932023e-02, 2.08946609e-01, 5.62500000e-01,
                          2.08946609e-01, 3.88932023e-02],
                         [1.07233047e-02, 3.88932023e-02, 6.25000000e-02,
                          3.88932023e-02, 1.07233047e-02]])
    assert(np.allclose(wkernel, kcompare))


def test_correct_detector():
    image = IFUImageModel((20,20))
    image.data = np.random.random((20,20))
    image.meta.instrument.name  = 'MIRI'
    image.meta.instrument.detector = 'MIRIFULONG'
    image.data = np.random.random((50,50))
    collect_pipeline_cfgs('./config')
    result = StraylightStep.call(image,
                                 config_file='config/straylight.cfg')
    assert result.meta.cal_step.straylight == 'SKIPPED'
    assert type(image) is type(result)


def test_not_nirspec():
    image = IFUImageModel((20,20))
    image.data = np.random.random((20,20))
    image.meta.instrument.name  = 'NIRSPEC'
    image.meta.instrument.detector = 'NRS1'
    collect_pipeline_cfgs('./config')
    result = StraylightStep.call(image,
                                 config_file='config/straylight.cfg')
    assert result.meta.cal_step.straylight == 'SKIPPED'
    assert type(image) is type(result)
