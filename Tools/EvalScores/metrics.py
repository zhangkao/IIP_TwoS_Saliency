# -*- coding: utf-8 -*-
'''
Commonly used metrics for evaluating saliency map performance.

Most metrics are ported from Matlab implementation provided by http://saliency.mit.edu/
Bylinskii, Z., Judd, T., Durand, F., Oliva, A., & Torralba, A. (n.d.). MIT Saliency Benchmark.

Python implementation: Chencan Qian, Sep 2014
'''

from functools import partial
import numpy as np
from numpy import random
from skimage import exposure, img_as_float
from skimage.transform import resize
try:
    from cv2 import cv
except ImportError:
    print('please install Python binding of OpenCV to compute EMD')

EPS = 2.2204e-16

def normalize(x, method='standard', axis=None):

    x = np.array(x, copy=True)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def match_hist(image, cdf, bin_centers, nbins=256):
    '''Modify pixels of input image so that its histogram matches target image histogram, specified by:
    cdf, bin_centers = cumulative_distribution(target_image)

    Parameters
    ----------
    image : array
        Image to be transformed.
    cdf : 1D array
        Values of cumulative distribution function of the target histogram.
    bin_centers ; 1D array
        Centers of bins of the target histogram.
    nbins : int, optional
        Number of bins for image histogram.

    Returns
    -------
    out : float array
        Image array after histogram matching.

    References
    ----------
    [1] Matlab implementation histoMatch(MTX, N, X) by Simoncelli, 7/96.
    '''
    image = img_as_float(image)
    old_cdf, old_bin = exposure.cumulative_distribution(image, nbins) # Unlike [1], we didn't add small positive number to the histogram
    new_bin = np.interp(old_cdf, cdf, bin_centers)
    out = np.interp(image.ravel(), old_bin, new_bin)
    return out.reshape(image.shape)



def AUC_Judd(saliency_map, fixation_map, jitter=True):

    s_map = np.array(saliency_map, copy=True)
    f_map = np.array(fixation_map, copy=True) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(f_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape, order=3, mode='nearest')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        s_map += random.rand(*s_map.shape) * 1e-7
    # Normalize saliency map to have values between [0,1]
    s_map = normalize(s_map, method='range')

    S = s_map.ravel()
    F = f_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds)+2)
    fp = np.zeros(len(thresholds)+2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh) # Total number of saliency map values above threshold
        tp[k+1] = (k + 1) / float(n_fix) # Ratio saliency map values at fixation locations above threshold
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix) # Ratio other saliency map values above threshold
    return np.trapz(tp, fp) # y, x


def AUC_Borji(saliency_map, fixation_map, n_rep=100, step_size=0.1, rand_sampler=None):

    s_map = np.array(saliency_map, copy=True)
    f_map = np.array(fixation_map, copy=True) > 0.5
    # If there are no fixation to predict, return NaN
    if not np.any(f_map):
        print('no fixation to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape, order=3, mode='nearest')
    # Normalize saliency map to have values between [0,1]
    s_map = normalize(s_map, method='range')

    S = s_map.ravel()
    F = f_map.ravel()
    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # For each fixation, sample n_rep values from anywhere on the saliency map
    if rand_sampler is None:
        r = random.randint(0, n_pixels, [n_fix, n_rep])
        S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
    else:
        S_rand = rand_sampler(S, F, n_rep, n_fix)
    # Calculate AUC per random split (set of random locations)
    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc) # Average across random splits


# def AUC_shuffled(saliency_map, fixation_map, other_map, n_rep=100, step_size=0.1):
#
#     o_map = np.array(other_map, copy=True) > 0.5
#     if other_map.shape != fixation_map.shape:
#         raise ValueError('other_map.shape != fixation_map.shape')
#     # For each fixation, sample n_rep values (from fixated locations on other_map) on the saliency map
#     def sample_other(other, S, F, n_rep, n_fix):
#         fixated = np.nonzero(other)[0]
#         indexer = map(lambda x: random.permutation(x)[:n_fix], np.tile(range(len(fixated)), [n_rep, 1]))
#         r = fixated[np.transpose(indexer)]
#         S_rand = S[r] # Saliency map values at random locations (including fixated locations!? underestimated)
#         return S_rand
#     return AUC_Borji(saliency_map, fixation_map, n_rep, step_size, partial(sample_other, o_map.ravel()))
def AUC_shuffled(saliency_map, fixation_map, other_map, n_rep=100, step_size=0.1):

    s_map = np.array(saliency_map, copy=True)
    f_map = np.array(fixation_map, copy=True) > 0.5
    o_map = np.array(other_map, copy=True) > 0.5
    if other_map.shape != fixation_map.shape:
        raise ValueError('other_map.shape != fixation_map.shape')
    if not np.any(f_map):
        print('no fixation to predict')
        return np.nan
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape, order=3, mode='nearest')
    s_map = normalize(s_map, method='range')

    S = s_map.ravel()
    F = f_map.ravel()
    Oth = o_map.ravel()

    S_fix = S[F] # Saliency map values at fixation locations
    n_fix = len(S_fix)

    ind = np.nonzero(Oth)[0]
    n_ind = len(ind)
    n_fix_oth = min(n_fix,n_ind)

    r = random.randint(0, n_ind, [n_ind, n_rep])[:n_fix_oth,:]
    S_rand = S[ind[r]]

    auc = np.zeros(n_rep) * np.nan
    for rep in range(n_rep):
        thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
        tp = np.zeros(len(thresholds)+2)
        fp = np.zeros(len(thresholds)+2)
        tp[0] = 0; tp[-1] = 1
        fp[0] = 0; fp[-1] = 1
        for k, thresh in enumerate(thresholds):
            tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
            fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix_oth)
        auc[rep] = np.trapz(tp, fp)
    return np.mean(auc)

def NSS(saliency_map, fixation_map):

    s_map = np.array(saliency_map, copy=True)
    f_map = np.array(fixation_map, copy=True) > 0.5
    if s_map.shape != f_map.shape:
        s_map = resize(s_map, f_map.shape)
    # Normalize saliency map to have zero mean and unit std
    s_map = normalize(s_map, method='standard')
    # Mean saliency value at fixation locations
    return np.mean(s_map[f_map])


def KLD(saliency_map1, saliency_map2):

    map1 = np.array(saliency_map1, copy=True)
    map2 = np.array(saliency_map2, copy=True)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    return np.sum(map2 * np.log(EPS + map2 / (map1+EPS)))


def CC(saliency_map1, saliency_map2):

    map1 = np.array(saliency_map1, copy=True)
    map2 = np.array(saliency_map2, copy=True)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have zero mean and unit std
    map1 = normalize(map1, method='standard')
    map2 = normalize(map2, method='standard')
    # Compute correlation coefficient
    return np.corrcoef(map1.ravel(), map2.ravel())[0,1]


def SIM(saliency_map1, saliency_map2):

    map1 = np.array(saliency_map1, copy=True)
    map2 = np.array(saliency_map2, copy=True)
    if map1.shape != map2.shape:
        map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
    # Normalize the two maps to have values between [0,1] and sum up to 1
    map1 = normalize(map1, method='range')
    map2 = normalize(map2, method='range')
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute histogram intersection
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


def EMD(saliency_map1, saliency_map2, sub_sample=1/32.0):

    map2 = np.array(saliency_map2, copy=True)
    # Reduce image size for efficiency of calculation
    map2 = resize(map2, np.round(np.array(map2.shape)*sub_sample), order=3, mode='nearest')
    map1 = resize(saliency_map1, map2.shape, order=3, mode='nearest')
    # Histogram match the images so they have the same mass
    map1 = match_hist(map1, *exposure.cumulative_distribution(map2))
    # Normalize the two maps to sum up to 1,
    # so that the score is independent of the starting amount of mass / spread of fixations of the fixation map
    map1 = normalize(map1, method='sum')
    map2 = normalize(map2, method='sum')
    # Compute EMD with OpenCV
    # - http://docs.opencv.org/modules/imgproc/doc/histograms.html#emd
    # - http://stackoverflow.com/questions/5101004/python-code-for-earth-movers-distance
    # - http://stackoverflow.com/questions/12535715/set-type-for-fromarray-in-opencv-for-python
    r, c = map2.shape
    x, y = np.meshgrid(range(c), range(r))
    signature1 = cv.CreateMat(r*c, 3, cv.CV_32FC1)
    signature2 = cv.CreateMat(r*c, 3, cv.CV_32FC1)
    cv.Convert(cv.fromarray(np.c_[map1.ravel(), x.ravel(), y.ravel()]), signature1)
    cv.Convert(cv.fromarray(np.c_[map2.ravel(), x.ravel(), y.ravel()]), signature2)
    return cv.CalcEMD2(signature2, signature1, cv.CV_DIST_L2)


def InfoGain(saliencyMap, fixationMap, baselineMap):

    map1 = np.array(saliencyMap, copy=True)
    mapb = np.array(baselineMap, copy=True)

    map1 = resize(map1, fixationMap.shape, order=3, mode='nearest')
    mapb = resize(mapb, fixationMap.shape, order=3, mode='nearest')

    map1 = normalize(map1, method='range')
    mapb = normalize(mapb, method='range')
    map1 = normalize(map1, method='sum')
    mapb = normalize(mapb, method='sum')

    locs = np.array(fixationMap, copy=True) > 0.5

    return np.mean(np.log2(EPS + map1[locs]) - np.log2(EPS + mapb[locs]))
