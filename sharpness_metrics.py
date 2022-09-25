import numpy as np
import pywt
from skimage.filters import threshold_otsu
import sigpy.plot as plt
import matplotlib.pyplot as plt2
from scipy import ndimage
from scipy import stats
import logging


def estimate_noise(img):
    thresh = threshold_otsu(img[img > 0].ravel())
    # plt2.hist(img[img > 0].ravel(), 256)
    # plt2.show()
    bg_mask = img >= thresh
    fill_mask = img == 0
    # plt.ImagePlot(bg_mask)
    bg_mask = ndimage.binary_dilation(bg_mask, iterations=10)
    bg_mask = ndimage.binary_fill_holes(bg_mask)
    bg_mask = bg_mask + fill_mask
    # plt.ImagePlot(bg_mask)
    # plt.ImagePlot(~bg_mask)
    # plt.ImagePlot(img * bg_mask)
    # plt.ImagePlot(img * ~bg_mask)
    pool = img[~bg_mask]
    sample_size = 100
    counter = 0
    noise = 0
    repeats = 10000
    for ii in range(repeats):
        sample = np.random.choice(pool, (sample_size,))
        normality_test = stats.shapiro(sample)

        if normality_test.pvalue > 0.05:
            counter += 1
            noise += np.std(sample)
            # print("Pval: {}".format(normality_test.pvalue))
            # print("Noise: {}".format(noise / counter))
    return noise / counter


def estimate_noise_donoho(dcoeffs):
    """Calculate the robust median estimator of the noise standard deviation.
    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    Returns
    -------
    sigma : float
        The estimated noise standard deviation (see section 4.2 of [1]_).
    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
        by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
        DOI:10.1093/biomet/81.3.425
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = dcoeffs[np.nonzero(dcoeffs)]

    # 75th quantile of the underlying, symmetric noise distribution
    denom = stats.norm.ppf(0.75)
    sigma = np.median(np.abs(detail_coeffs)) / denom
    return sigma


def rwc(img, apply_thresh=True, donoho_noise=True):
    # Get approximation (cA) and detail(cD) coefficients (low and high resolution, respectively)
    decomp_level = 2
    coeffs = pywt.wavedecn(img, "db6", level=decomp_level)
    cA = coeffs[0].ravel()  # approximation coefficients(low res)
    detail_coeffs = []
    if donoho_noise:
        for level in range(1, decomp_level):
            for key, value in coeffs[level].items():
                # print(key)
                detail_coeffs.append(value.ravel())
        detail_coeffs = np.hstack(detail_coeffs)  # detail coefficients(high res)
        img_std = estimate_noise_donoho(detail_coeffs)
    else:
        for key, value in coeffs[-2].items():
            # print(key)
            detail_coeffs.append(value.ravel())
        detail_coeffs = np.hstack(detail_coeffs)  # detail coefficients(high res)
        img_std = estimate_noise(img)

    noise_thresh = img_std * np.sqrt(2 * np.log(np.size(img)) / np.size(img))
    # noise_thresh = img_std * np.sqrt(2 * np.log(np.size(img)))
    # plt2.hist(np.abs(detail_coeffs), 10000)
    # plt2.show()
    if apply_thresh:
        detail_coeffs = detail_coeffs[np.abs(detail_coeffs) > noise_thresh]

    H = np.sum(np.abs(detail_coeffs) ** 2)
    L = np.sum(np.abs(cA ** 2))

    return H / L, img_std, noise_thresh
