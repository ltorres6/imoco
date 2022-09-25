import argparse
import numpy as np
from scipy.signal import savgol_filter, detrend, medfilt, butter, lfilter
import matplotlib.pyplot as plt


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


def estimate_respSavitzkyGolay(dc, tr, window=0.8, order=2, detrend_window=10.0, usePhase=False, useDetrend=False):
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
    resp = resp / resp.max()

    if useDetrend is True:
        detrend_window_length = int(detrend_window / tr)
        if detrend_window_length % 2 == 0:
            detrend_window_length += 1
        # resp_moving_median = medfilt(resp, window_length)
        resp_moving_mean = np.convolve(resp, np.ones((detrend_window_length,)) / detrend_window_length, mode="same")

        # plt.plot(resp)
        # plt.plot(resp_moving_median)
        # plt.legend()
        # plt.show()
        resp -= resp_moving_mean
        # plt.plot(resp)
        # plt.show()
        # resp = detrend(resp)

    return resp


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
    resp = estimate_respSavitzkyGolay(dc, args.tr)
    np.save(args.resp_file, resp)
