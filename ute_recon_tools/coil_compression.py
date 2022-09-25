import numpy as np
import logging


def pca_cc(kdata=None, axis=0, target_channels=None):

    logger = logging.getLogger("PCA_CoilCompression")

    if isinstance(kdata, list):
        logger.debug("Passed k-space is a list, using encode 0 for compression")
        kdata_cc = kdata[0]
    else:
        kdata_cc = kdata

    logger.info(f"Compressing to {target_channels} channels, along axis {axis}")
    logger.debug(f"Initial  size = {kdata_cc.shape} ")

    # Put channel to first axis
    kdata_cc = np.moveaxis(kdata_cc, axis, -1)
    old_channels = kdata_cc.shape[-1]
    logger.info(f"Old channels =  {old_channels} ")

    # Subsample to reduce memory for SVD
    mask_shape = np.array(kdata_cc.shape)
    mask = np.random.choice([True, False], size=mask_shape[:-1], p=[0.05, 1 - 0.05])

    # Create a subsampled array
    kcc = np.zeros((old_channels, np.sum(mask)), dtype=kdata_cc.dtype)
    logger.debug(f"Kcc Shape = {kcc.shape} ")
    for c in range(old_channels):
        ktemp = kdata_cc[..., c]
        kcc[c, :] = ktemp[mask]

    kdata_cc = np.moveaxis(kdata_cc, -1, axis)

    #  SVD decomposition
    logger.debug(f"Working on SVD of {kcc.shape}")
    u, s, vh = np.linalg.svd(kcc, full_matrices=False)

    logger.debug(f"S = {s}")

    if isinstance(kdata, list):
        logger.debug("Passed k-space is a list, using encode 0 for compression")

        for e in range(len(kdata)):
            kdata[e] = np.moveaxis(kdata[e], axis, -1)
            kdata[e] = np.expand_dims(kdata[e], -1)
            logger.debug(f"Shape = {kdata[e].shape}")
            kdata[e] = np.matmul(u, kdata[e])
            kdata[e] = np.squeeze(kdata[e], axis=-1)
            kdata[e] = kdata[e][..., :target_channels]
            kdata[e] = np.moveaxis(kdata[e], -1, axis)

        for ksp in kdata:
            logger.debug(f"Final Shape {ksp.shape}")
    else:
        # Now iterate over and multiply by u
        kdata = np.moveaxis(kdata, axis, -1)
        kdata = np.expand_dims(kdata, -1)
        kdata = np.matmul(u, kdata)
        logger.debug(f"Shape = {kdata.shape}")

        # Crop to target channels
        kdata = np.squeeze(kdata, axis=-1)
        kdata = kdata[..., :target_channels]

        # Put back
        kdata = np.moveaxis(kdata, -1, axis)
        logger.debug(f"Final shape = {kdata.shape}")

    return kdata
