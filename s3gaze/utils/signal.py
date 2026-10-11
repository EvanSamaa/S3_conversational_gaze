# Vendored from EvansToolBox (https://github.com/EvanSamaa/EvansToolBox), MIT License,
# Copyright (c) 2023 EvanSamaa. Only the functions this package uses are kept.
import math

import numpy as np
from scipy.signal.windows import kaiser


def pad_arrays(arr1, arr2):
    """
    Pads the shorter of two NumPy arrays with zeros to make them the same length
    in the first dimension.

    Parameters:
    arr1 (numpy.ndarray): First input array
    arr2 (numpy.ndarray): Second input array

    Returns:
    numpy.ndarray: Padded version of arr1
    numpy.ndarray: Padded version of arr2
    """
    # Get the lengths of the arrays
    len1 = arr1.shape[0]
    len2 = arr2.shape[0]

    # Pad the shorter array with zeros to make them the same length
    if len1 < len2:
        pad_width = ((0, len2 - len1),) + ((0, 0),) * (arr1.ndim - 1)
        arr1 = np.pad(arr1, pad_width, mode='constant', constant_values=0)
    elif len2 < len1:
        pad_width = ((0, len1 - len2),) + ((0, 0),) * (arr2.ndim - 1)
        arr2 = np.pad(arr2, pad_width, mode='constant', constant_values=0)

    return arr1, arr2

def dx_dt(x: np.array, dt: float = 1, method=1):
    """
    This functio compute first derivative for the input function x using either central or forward differences

    :param x: input array to compute derivative, should be of shape [num of timestamp, num of attributes]
    :param dt: time stamp size
    :param method: method of computing derivative. 1 is forward difference, 2 is central differences
    :return: dx/dt, would be the same size as x. The first and last element are zero.
    """
    out_dx_dt = np.zeros(x.shape)
    if len(x.shape) == 2:
        for j in range(0, x.shape[1]):
            if method == 1:
                for i in range(0, x.shape[0]):
                    out_dx_dt[i, j] = (x[i, j] - x[i-1, j])/dt
                out_dx_dt[0, j] = out_dx_dt[1, j]
            if method == 2:
                for i in range(1, x.shape[0]-1):
                    out_dx_dt[i, j] = (x[i + 1, j] - x[i - 1, j]) / dt
                out_dx_dt[-1, j] = out_dx_dt[-2, j]
                out_dx_dt[0, j] = out_dx_dt[1, j]
            if method == 3:
                for i in range(0, x.shape[0]-1):
                    out_dx_dt[i, j] = (x[i+1, j] - x[i, j])/dt
                out_dx_dt[-1, j] = out_dx_dt[-2, j]

    elif len(x.shape) == 1:
        if method == 1:
            for i in range(1, x.shape[0]):
                out_dx_dt[i] = (x[i] - x[i-1]) / dt
            out_dx_dt[0] = 0
        if method == 2:
            for i in range(1, x.shape[0] - 1):
                out_dx_dt[i] = (x[i + 1] - x[i - 1]) / dt
            out_dx_dt[-1] = 0
            out_dx_dt[0] = 0
        if method == 3:
            for i in range(0, x.shape[0] - 1):
                out_dx_dt[i] = (x[i + 1] - x[i]) / dt
            out_dx_dt[-1] = out_dx_dt[-2]
    return out_dx_dt

def intensity_from_signal(x, win_size=441):
    """
    get the intensity of a signal, as implemented in praat
    :param x: input signal, should be a numpy array
    :param win_size: should be an integer, window size for the output signal, default for 441
    :return: the intensity of the signal, in the logarithmic scale (dB)
    """
    total_time_steps = np.ceil(x.shape[0] / win_size)
    kaiser_window = kaiser(win_size, 5)
    intensity = []
    for i in range(0, int(total_time_steps)):
        end_up = min((i+1)*win_size, x.shape[0])
        current_kaiser_window = kaiser_window[0:(end_up - i*win_size)]
        current = current_kaiser_window * x[i*win_size:end_up] * x[i*win_size:end_up]
        current_frame_intensity = current.sum()/kaiser_window.sum()/4.0e-10
        # current_frame_intensity = current.sum()/kaiser_window.sum()
        if current_frame_intensity <= 1.0e-30:
            current_frame_intensity = 0
        else:
        # if in the absolute value scale
            # current_frame_intensity = np.sqrt(current_frame_intensity)
        # in the case where logarithmic scale should be used
            current_frame_intensity = 10*np.log10(current_frame_intensity)
        intensity.append(current_frame_intensity)
    return np.array(intensity)

def laplacian_smoothing(arr, iteration=1):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)  # Convert to NumPy array if not already one
    
    shape = arr.shape
    for i in range(iteration):
        out = np.zeros_like(arr)
        out[0] = arr[0]
        out[-1] = arr[-1]
        # Vectorized update for the internal elements
        out[1:-1] = (arr[:-2] + arr[1:-1] + arr[2:]) / 3.0
        arr = out
    return out

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)
    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)
    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev
    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev
        # The filtered derivative of the signal.
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

def run_euro(t, data, min_cutoff=0.0001, beta=1.5):
    # input should be shape [N, D] with N being the number of time steps, D is the dimension of the data
    """
    """
    out = np.zeros(data.shape)
    for j in range(0, data.shape[1]):
        out[0, j] = data[0, j]
        one_euro_filter = OneEuroFilter(t[0], data[0, j],min_cutoff=min_cutoff,beta=beta)
        for i in range(1, t.shape[0]):
            out[i, j] = one_euro_filter(t[i], data[i, j])
    return out
