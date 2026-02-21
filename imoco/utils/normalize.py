import numpy as np


def normalize(input, minv, maxv):
    """Normalize array values to a specified range.

    Args:
        input (ndarray): Input array.
        minv (float): Minimum value of output range.
        maxv (float): Maximum value of output range.

    Returns:
        ndarray: Normalized array with values in [minv, maxv].
    """
    original = input.copy()
    out = (maxv - minv) * (original - np.min(original[:])) / (np.max(original[:]) - np.min(original[:])) + minv
    return out
