import numpy as np


def normalize(input, minv, maxv):
    original = input.copy()
    out = (maxv - minv) * (original - np.min(original[:])) / (np.max(original[:]) - np.min(original[:])) + minv
    return out
