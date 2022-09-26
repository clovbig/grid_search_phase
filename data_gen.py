"""
Author: Claudia Bigoni, claudia.bigoni epfl.ch
Date: 05.01.2022
Description: Set of functions to generate synthetic EEG-like signals
"""
import numpy as np
import colorednoise as cn


def generate_sinusoid(f0, dur, fs):
    """Generate synthetic signal as a sine wave of main frequency f0 of length t.
    :param f0 = main frequency (Hz)
    :param dur = time duration of signal (s)
    :param fs = sampling frequency

    :return output_signal = sinusoid(s)
    """

    t = np.arange(0, dur, 1 / fs)
    output_signal = f0 * np.sin(2 * np.pi * f0 * t)

    return output_signal


def add_random_noise(input_signal, fs):
    """Add random noise as 1/f to an input signal sampled at fs
    :param input_signal
    :param fs

    :return noisy_signal = input_signal + random noise"""

    L = input_signal.shape[0]
    noise = cn.powerlaw_psd_gaussian(1, L)
    noisy_signal = input_signal + noise

    return noisy_signal