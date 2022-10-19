import sigpy as sp
import numpy as np
from tqdm import tqdm, trange
import sigpy.plot as plt
from normalize import normalize


def rss(ksp, coord, dcf, device):
    xp = sp.Device(device).xp
    img = 0
    n_coils = ksp.shape[0]
    for c in trange(n_coils):
        ksp_c = sp.to_device(ksp[c], device=device)
        img_c = sp.nufft_adjoint(ksp_c * dcf, coord)
        img = img + sp.to_device(img_c * xp.conj(img_c), -1)
    img = np.sqrt(img)
    return img


def pipe_menon_dcf(
    coord,
    img_shape=None,
    device=sp.cpu_device,
    max_iter=30,
    kernel="kaiser_bessel",
    beta=8,
    width=4,
    show_pbar=True,
):
    r"""Compute Pipe Menon density compensation factor.

    Perform the following iteration:

    .. math::

        w = \frac{w}{|G^H G w|}

    with :math:`G` as the gridding operator.

    Args:
        coord (array): k-space coordinates.
        img_shape (None or list): Image shape.
        device (Device): computing device.
        max_iter (int): number of iterations.
        n (int): Kaiser-Bessel sampling numbers for gridding operator.
        beta (float): Kaiser-Bessel kernel parameter.
        width (float): Kaiser-Bessel kernel width.
        show_pbar (bool): show progress bar.

    Returns:
        array: density compensation factor.

    References:
        Pipe, James G., and Padmanabhan Menon.
        Sampling Density Compensation in MRI:
        Rationale and an Iterative Numerical Solution.
        Magnetic Resonance in Medicine 41, no. 1 (1999): 179–86.


    """
    device = sp.Device(device)
    xp = device.xp

    with device:
        w = xp.ones(coord.shape[:-1], dtype=coord.dtype)
        if img_shape is None:
            img_shape = sp.estimate_shape(coord)
        # G = sp.linop.Gridding(img_shape, coord, param=beta, width=width, kernel=kernel)
        G = sp.linop.NUFFTAdjoint(img_shape, coord, oversamp=1.5, width=width)
        with tqdm(total=max_iter, desc="PipeMenonDCF", disable=not show_pbar) as pbar:
            for it in range(max_iter):
                GHGw = G.H * G * w  # w_i
                w = w / xp.abs(GHGw)  # w_i/w_i conv C
                # w[w < 0] = 0
                resid = xp.abs(GHGw - 1).max().item()

                pbar.set_postfix(resid="{0:.2E}".format(resid))
                pbar.update()
        # scale
        # psf_raw = xp.real(F * w)
        # scale = xp.linalg.norm(psf_raw**2)
        # fov_scale = img_shape[0] * img_shape[1] * img_shape[2]
    # return w / (scale * fov_scale)
    return w


coord_path = ""
device = 0
# xp = sp.Device(device).xp
coord = np.load("/home/ltorres/data/new_oe_data/coord.npy")
dcf_original = np.load("/home/ltorres/data/new_oe_data/dcf.npy")
ksp = np.load("/home/ltorres/data/new_oe_data/ksp.npy")
plt.ImagePlot(ksp[:, :2000])
plt.ImagePlot(ksp[:, -2000:])

# print(coord.shape)
# print(coord.max())
# print(coord.min())
# coord /= coord.max()
# coord * 2 * np.pi
# plt.ImagePlot(coord[:500])
coord = sp.to_device(coord, device)
dcf_original = sp.to_device(dcf_original, device=device)
dcf = pipe_menon_dcf(
    coord, device=device, kernel="kaiser_bessel", beta=12, width=4, max_iter=20
)
# dcf = least_squares_density_compensation(
#     coord, device=0, kernel="kaiser_bessel", beta=8, width=4, max_iter=5
# )
# dcf = pipe_menon_dcf(coord, device=0, kernel="spline", beta=1, width=2, max_iter=20)
scaling_factor = 1e28
img_computed = normalize(rss(ksp, coord, dcf * scaling_factor, device), 0, 255)
img_original = normalize(rss(ksp, coord, dcf_original, device), 0, 255)
img_stacked = np.stack([img_original, img_computed], axis=0)
plt.ImagePlot(dcf[:500, :] * scaling_factor)
plt.ImagePlot(dcf_original[:500, :])
# plt.ImagePlot(img_original)
# plt.ImagePlot(img_computed)
plt.ImagePlot(img_stacked)
