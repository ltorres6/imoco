import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import (
    butter,
    convolve,
    detrend,
    firls,
    lfilter,
    medfilt,
    savgol_filter,
)


def estimate_resp(dc, tr, n=9999, fl=0.1, fh=1.5, fw=0.01, usePhase=False):
    """Estimate respiratory signal from DC.

    The function performs:
    1) Filter DC with a band-pass filter with symmetric extension.
    2) Normalize each channel by a robust estimation of mean and variance.
    3) Return the channel with the maximum variance.

    Args:
        dc (array): multi-channel DC array of shape [num_coils, num_tr].
        tr (float): TR in seconds.
        n (int): length of band-pass filter.
        fl (float): lower cut-off of band-pass filter in Hz.
        fh (float): higher cut-off of band-pass filter in Hz.
        fw (float): transition width of band-pass filter.

    Returns:
        array: respiratory signal of length num_tr.
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
    """Estimate respiratory signal from DC.

    The function performs:
    1) Filter DC with savitzky Golay Filter

    Args:
        dc (array): multi-channel DC array of shape [num_coils, num_tr].
        tr (float): TR in seconds.
        window (float): smoothing window in seconds
        order (int): polynomial order.
    Returns:
        array: respiratory signal of length num_tr.
    """
    if usePhase is True:
        dc = np.angle(dc)
    else:
        dc = np.abs(dc)
    # fs = 1 / tr
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
        # resp_moving_median = medfilt(resp, window_length)
        resp_moving_mean = np.convolve(
            resp, np.ones((detrend_window_length,)) / detrend_window_length, mode="same"
        )

        # plt.plot(resp)
        # plt.plot(resp_moving_median)
        # plt.legend()
        # plt.show()
        resp -= resp_moving_mean
        # plt.plot(resp)
        # plt.show()
        # resp = detrend(resp)

    return resp, dc[c_selected]


def estimate_resp_bandpass(dc, tr, fl=0.5, fh=1.5, usePhase=False):
    """Estimate respiratory signal from DC.

    The function performs:
    1) Filter DC with butterworth bandpass filter
    Args:
        dc (array): multi-channel DC array of shape [num_coils, num_tr].
        tr (float): TR in seconds.
        fl (float): low threshold frequency
        fh (int):  high threshold frequency
    Returns:
        array: respiratory signal of length num_tr.
    """
    if usePhase is True:
        dc = np.angle(dc)
    else:
        dc = np.abs(dc)
    fs = 1 / tr
    sigma_max = 0
    for c in range(len(dc)):
        resp_c = butter_bandpass_filter(dc[c], fl, fh, fs, order=1)
        sigma_c = 1.4826 * np.median(np.abs(resp_c - np.median(resp_c)))
        plt.plot(resp_c)
        plt.show()
        if sigma_c > sigma_max:
            resp = (resp_c - np.median(resp_c)) / sigma_c
            sigma_max = sigma_c
    resp = resp / resp.max()
    return resp


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Estimate respiratory signal.")

    parser.add_argument("ksp_file", type=str, help="k-space file.")
    parser.add_argument("tr", type=float, help="TR in seconds.")
    parser.add_argument("resp_file", type=str, help="Output respiratory signal file.")

    args = parser.parse_args()

    ksp = np.load(args.ksp_file)
    dc = ksp[:, :, 0]
    resp = estimate_resp(dc, args.tr)
    np.save(args.resp_file, resp)
