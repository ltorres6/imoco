import numpy as np
import sigpy.plot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import threshold_li, threshold_local, threshold_otsu
from skimage.morphology import ball
from skimage.segmentation import inverse_gaussian_gradient, morphological_geodesic_active_contour

from normalize import normalize

img = np.load("/home/ltorres/data/recon/ipf/103-002/mri/20151222/pre_contrast/MotionResolved_test/MotionResolvedLowRes.npy")
# img = gaussian_filter(normalize(np.abs(img[0]), 0, 1), [1] * 3)
img = normalize(np.abs(img[0]), 0, 1)
gradient = inverse_gaussian_gradient(img)
plt.ImagePlot(gradient)

bg_mask = morphological_geodesic_active_contour(
    gradient, iterations=60, init_level_set=np.ones_like(img), smoothing=1, balloon=-1
)
plt.ImagePlot(bg_mask)

strel = ball(2, dtype=np.uint8)
bg_mask = ndimage.binary_closing(bg_mask, structure=strel, iterations=10)
plt.ImagePlot(bg_mask)

bg_mask = ndimage.binary_fill_holes(bg_mask, structure=strel)
plt.ImagePlot(bg_mask)