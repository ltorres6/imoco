import numpy as np
from scipy.signal import (
    butter,
    convolve,
    firls,
    lfilter,
    savgol_filter,
)


def estimate_resp(dc, tr, n=9999, fl=0.1, fh=1.5, fw=0.01, usePhase=False):
    """Estimate respiratory signal from the DC (center of k-space) component.

    Performs band-pass filtering, per-channel robust normalization, and
    selects the channel with maximum variance as the respiratory surrogate.

    Reference: Section II-B of the JMRI paper.

    Args:
        dc (ndarray): Multi-channel DC array of shape [num_coils, num_tr].
        tr (float): Repetition time in seconds.
        n (int): Length of the FIR band-pass filter.
        fl (float): Lower cut-off frequency in Hz.
        fh (float): Upper cut-off frequency in Hz.
        fw (float): Transition width in Hz.
        usePhase (bool): If True, use phase of DC signal instead of magnitude.

    Returns:
        tuple: (resp, dc_selected) where resp is the normalized respiratory
            signal of length num_tr, and dc_selected is the raw DC signal
            of the selected coil channel.
    """
    if usePhase:
        dc = np.unwrap(np.angle(dc))
    else:
        dc = np.abs(dc)

    fs = 1 / tr
    bands = [0, fl - fw, fl, fh, fh + fw, fs / 2]
    desired = [0, 0, 1, 1, 0, 0]

    filt = firls(n, bands, desired, fs=fs)
    sigma_max = 0
    for c in range(len(dc)):
        dc_pad = np.pad(dc[c], [n // 2, n // 2], mode="reflect")
        resp_c = convolve(dc_pad, filt, mode="valid")
        sigma_c = 1.4826 * np.median(np.abs(resp_c - np.median(resp_c)))

        if sigma_c > sigma_max:
            resp = (resp_c - np.median(resp_c)) / sigma_c
            sigma_max = sigma_c
            c_selected = c

    return resp, dc[c_selected]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def estimate_respSavitzkyGolay(
    dc, tr, window=0.8, order=2, detrend_window=10.0, usePhase=False, useDetrend=False
):
    """Estimate respiratory signal using a Savitzky-Golay filter.

    Args:
        dc (ndarray): Multi-channel DC array of shape [num_coils, num_tr].
        tr (float): Repetition time in seconds.
        window (float): Smoothing window in seconds.
        order (int): Polynomial order.
        detrend_window (float): Detrending window in seconds.
        usePhase (bool): If True, use phase of DC signal.
        useDetrend (bool): If True, apply moving-mean detrending.

    Returns:
        tuple: (resp, dc_selected).
    """
    if usePhase is True:
        dc = np.angle(dc)
    else:
        dc = np.abs(dc)
    window_length = int(window / tr)
    if window_length % 2 == 0:
        window_length += 1
    sigma_max = 0
    for c in range(len(dc)):
        resp_c = savgol_filter(dc[c], window_length, order)
        sigma_c = 1.4826 * np.median(np.abs(resp_c - np.median(resp_c)))

        if sigma_c > sigma_max:
            resp = (resp_c - np.median(resp_c)) / sigma_c
            sigma_max = sigma_c
            c_selected = c
    resp = resp / resp.max()

    if useDetrend is True:
        detrend_window_length = int(detrend_window / tr)
        if detrend_window_length % 2 == 0:
            detrend_window_length += 1
        resp_moving_mean = np.convolve(
            resp, np.ones((detrend_window_length,)) / detrend_window_length, mode="same"
        )
        resp -= resp_moving_mean

    return resp, dc[c_selected]
