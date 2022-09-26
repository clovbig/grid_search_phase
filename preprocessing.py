"""
Author: Claudia Bigoni, claudia.bigoni@epfl.ch
Date: 05.01.2022, last update: 23.09.2022
Description: functions to help in the preprocessing of EEG. Frequency and spatial filters.
"""
import numpy as np
from scipy import signal

neighbours_ch = {
    'FC3': ['F5', 'C1', 'F1', 'C5'],
    'FC1': ['Fz', 'Cz', 'F3', 'C3'],
    'FC2': ['F4', 'C4', 'Fz', 'Cz'],
    'FC4': ['F2', 'C2', 'F6', 'C6'],
    'C5': ['FT7', 'FC3', 'TP7', 'CP3'],
    'C3': ['FC1', 'CP1', 'FC5', 'CP5'],
    'C1': ['Fz', 'CPz', 'FC3', 'CP3'],
    'Cz': ['FC1', 'CP1', 'FC2', 'CP2'],
    'C2': ['FC4', 'CP4', 'Fz', 'CPz'],
    'C4': ['FC2', 'CP2', 'FC6', 'CP6'],
    'C6': ['FC4', 'CP4', 'FT8', 'TP8'],
    'CP3': ['C1', 'P1', 'C5', 'P5'],
    'CP1': ['Cz', 'Pz', 'C3', 'P3'],
    'CPz': ['C1', 'P1', 'C2', 'P2'],
    'CP2': ['C4', 'P4', 'Cz', 'Pz'],
    'CP4': ['C2', 'P2', 'P6', 'C6'],
    'Oz': ['O1', 'O2', 'Iz', 'POz'],
    'Fz': ['FC1', 'FC2', 'AF3', 'AF4']}


def spatial_filter(input_signal, fname, ch_list_index, chs):
    """Compute the spatial filter for an input signal
    :param input_signal: MxN array with M=number of channels and N=number of samples
    :param fname: spatial filter name, accepted: car, small_laplacian (hjort)
    :param ch_list_index: list with correspndoing indices between channel names and position in input_signal
    :param chs: list of channels that need to be spatially filtered

    :return f_signal: same size as input signal, but filtered"""

    num_ch = input_signal.shape[0]
    f_signal = np.copy(input_signal)

    if fname == 'car':
        spatialfilter = np.ones([num_ch, num_ch]) / num_ch
        f_signal = input_signal - np.dot(spatialfilter, input_signal)

    elif fname == 'small_laplacian':
        for idx_ch, ch in enumerate(chs):
            average_neighbours = np.mean(input_signal[[ch_list_index.index(i) for i in neighbours_ch[ch]], :], axis=0)
            f_signal[ch_list_index.index(ch), :] = input_signal[ch_list_index.index(ch), :] - average_neighbours

    return f_signal


# 3 FIR filters
def fir_kaiser(input_signal, f_min, f_max, fs, order, ripple_db):
    """Bandpass an input signal between f_min and f_max with FIR window Kaiser method
        :param input_signal: 1xn array
        :param f_min: high-pass cutoff frequency (Hz)
        :param f_max: low-pass cutoff frequency (Hz)
        :param fs: sampling frequency (Hz)
        :param order: filter order !! here order stands for transition width !!
        :param ripple_db:

        :return filtered_signal"""
    if len(input_signal.shape) > 1:
        padlen = np.min([input_signal.shape[1] - 5, 1000])
    else:
        padlen = np.min([len(input_signal) - 5, 1000])
    transition_width = order / (0.5 * fs)  # transition width normalized to Nydquist frequency
    ntaps, beta = signal.kaiserord(ripple_db, transition_width)
    ntaps |= 1
    b = signal.firwin(ntaps, np.asarray([f_min, f_max]) / (fs / 2), window=('kaiser', beta), pass_zero='bandpass')
    filtered_signal = signal.filtfilt(b, 1.0, input_signal, padlen=padlen)

    return filtered_signal


def fir_window(input_signal, f_min, f_max, fs, order, fp):
    """Bandpass an input signal between f_min and f_max with FIR window Hamming method
            :param input_signal: 1xn array
            :param f_min: high-pass cutoff frequency (Hz)
            :param f_max: low-pass cutoff frequency (Hz)
            :param fs: sampling frequency (Hz)
            :param order: filter order
            :param fp: main frequency in the band of interest

            :return filtered_signal"""

    if len(input_signal.shape) > 1:
        padlen = np.min([input_signal.shape[1] - 5, 1000])
    else:
        padlen = np.min([len(input_signal) - 5, 1000])
    order = int(np.round(order * (fs / fp)))
    order |= 1
    b = signal.firwin(order, np.asarray([f_min, f_max]) / (fs / 2), window='hamming', pass_zero='bandpass')
    filtered_signal = signal.filtfilt(b, 1.0, input_signal, padlen=padlen)

    # filtered_signal = signal.lfilter(b, 1.0, input_signal)
    # filtered_signal2 = butterworth_filter(input_signal, f_min, f_max, fs, 4)

    return filtered_signal


def fir_ls(input_signal, f_min, f_max, fs, order, fp):
    """Bandpass an input signal between f_min and f_max with FIR least squres method
            :param input_signal: 1xn array
            :param f_min: high-pass cutoff frequency (Hz)
            :param f_max: low-pass cutoff frequency (Hz)
            :param fs: sampling frequency (Hz)
            :param order: filter order !! here order stands for transition width !!
            :param fp: main frequency in the band of interest
​
            :return filtered_signal"""

    order = int(np.round(order * (fs / fp)))
    order |= 1
    b = signal.firls(order, np.asarray([0, f_min - 2, f_min, f_max, f_max + 2, fs / 2]) / (fs / 2),
                     np.asarray([0, 0, 1, 1, 0, 0]))
    filtered_signal = signal.filtfilt(b, 1.0, input_signal, padlen=0)

    return filtered_signal


def iir_bandpass_filter(input_signal, f_min, f_max, fs, order=2, rs=5, rp=5, name='butter'):
    """Bandpass an input signal between f_min and f_max
    :param input_signal: 1xn array
    :param f_min: high-pass cutoff frequency (Hz)
    :param f_max: low-pass cutoff frequency (Hz)
    :param fs: sampling frequency (Hz)
    :param order: filter order
    :param rs:
    :param rp:
    :param name: name of filter type

    :return filtered_signal"""

    if name == 'butter':
        [b, a] = signal.butter(order, np.asarray([f_min, f_max]) / (fs / 2), btype='bandpass')
    elif name == 'cheby1':
        [b, a] = signal.cheby1(order, rp, np.asarray([f_min, f_max]), btype='bandpass', fs=fs)
    elif name == 'cheby2':
        [b, a] = signal.cheby2(order, rs, np.asarray([f_min, f_max]) / (fs / 2), btype='bandpass')
    elif name == 'ellip':
        [b, a] = signal.ellip(order, rp, rs, np.asarray([f_min, f_max]) / (fs / 2), btype='bandpass')
    elif name == 'bessel':
        [b, a] = signal.bessel(order, np.asarray([f_min, f_max]) / (fs / 2), btype='bandpass')
    elif name == 'peak':
        Q = order
        f_center = (f_max - f_min)/2
        [b, a] = signal.iirpeak(f_center, Q, fs)
    else:
        [b, a] = [1, 1]
    filtered_signal = signal.filtfilt(b, a, input_signal, padlen=0)

    return filtered_signal


def sos_iir_bandpass_filter(input_signal, f_min, f_max, fs, order, rs, rp, name):
    """Bandpass an input signal between f_min and f_max
    :param input_signal: 1xn array
    :param f_min: high-pass cutoff frequency (Hz)
    :param f_max: low-pass cutoff frequency (Hz)
    :param fs: sampling frequency (Hz)
    :param order: filter order
    :param rs: ripple stop band
    :param rp: ripple pass band
    :param name: name of filter type

    :return filtered_signal"""
    if name == 'butter':
        sos = signal.butter(order, np.asarray([f_min, f_max]) / (fs / 2), btype='bandpass', output='sos')
    elif name == 'cheby1':
        sos = signal.cheby1(order, rp, np.asarray([f_min, f_max]), btype='bandpass', fs=fs, output='sos')
    elif name == 'cheby2':
        sos = signal.cheby2(order, rs, np.asarray([f_min, f_max]), btype='bandpass', fs=fs, output='sos')
    elif name == 'ellip':
        sos = signal.ellip(order, rp, rs, np.asarray([f_min, f_max]), btype='bandpass', fs=fs, output='sos')
    elif name == 'bessel':
        sos = signal.bessel(order, np.asarray([f_min, f_max]) / (fs / 2), btype='bandpass', output='sos')
    else:
        print('No available filter selected')

    filtered_signal = signal.sosfiltfilt(sos, input_signal, padlen=0, axis=-1)

    return filtered_signal


def cmpl_morlet_wlt(f0, nc, t_array):
    """Create a complex Morlet wavelet with frequency f0 and number of cycles n in the time window given.
    :param f0: central frequency (Hz), float
    :param nc: number of cycles (float), positive
    :param t_array: 1D array of time window of wavelet."""
    # NB: wavelet must have the same sampling rate of the data
    # NB2: wavelet must be centered!!
    sin_wave = np.exp(1j * 2 * np.pi * f0 * t_array)  # complex sine wave
    s = nc / (2 * np.pi * f0)     # std deviation of gaussian
    gauss_win = np.exp((-t_array ** 2) / (2 * s ** 2))

    cmw = np.multiply(sin_wave, gauss_win)

    return cmw


def to_freq_domain(input_signal, n):
    """Apply FFT to signal.
    :param input_signal: 1D array
    :param n: integer, number of points for spectrum

    :return frequencies: 1D array containing frequencies (Hz)
    :return output_signal: 1D arrray, spectrum of input signal (same length of frequencies)."""
    y = np.fft.fft(input_signal, n)  # , n=fs)
    frequencies = np.fft.fftfreq(len(y))  # fs, d=1 / fs)
    output_signal = y

    return frequencies[:round(len(y) // 2)], output_signal


def morlet_filter(f0, n, input_signal, fs):
    """Convolution of continuous complex Morlet wavelet
    :param f0: central frequency (float)
    :param n: number of cycles (float)
    :param input_signal: 1D array - signal to be filtered
    :param fs: sampling frequency

    :return filtered_signal: 1D complex array
    """
    # 1. Create a complex morlet wavelet with central frequency fc and n number of cycles
    t_array = np.arange(-len(input_signal) / (2 * fs), len(input_signal) / (2 * fs), 1 / fs)
    cmw = cmpl_morlet_wlt(f0, n, t_array)
    # 2. Do multiplication of wavelet and signal in the frequency domain
    n_conv = len(cmw) + len(input_signal) - 1
    _, cmw_f = to_freq_domain(cmw, n_conv)
    cmw_f /= np.max(cmw_f)
    _, input_signal_f = to_freq_domain(input_signal, n_conv)
    conv_result = np.multiply(cmw_f, input_signal_f)
    # 3. Retrieve the filtered signal with iFFT
    half_wav = int(np.floor(len(cmw) / 2))
    filtered_signal = np.fft.ifft(conv_result)
    filtered_signal = filtered_signal[half_wav - 1:-half_wav]

    return filtered_signal


def morlet_filter_2(fs, f0, input_signal):
    """This is taken from github of Mike X Cohen - related to his paper in Neuroimage"""
    nfrex = 40
    frex = np.linspace(2, 25, nfrex)
    fwhm = np.linspace(.8, .7, nfrex)

    # setup wavelet and convolution parameters:
    wavet = np.arange(-2, 2, 1/fs)
    halfw = int(len(wavet)/2)
    n_conv = len(input_signal) + len(wavet)

    # spectrum od data
    dataX = np.fft.fft(input_signal, n_conv)
    # Find the frequency closest to the one looked for
    fi = np.argmin(abs(frex - f0))
    f = frex[fi]

    # create wavelet
    waveX = np.fft.fft(np.exp(2*1j*np.pi*f*wavet) * np.exp(-4*np.log(2)*wavet**2/fwhm[fi]**2), n_conv)
    waveX = waveX / np.abs(max(waveX))  # normalize

    ast = np.fft.ifft(waveX*dataX)  # convolve
    ast = ast[halfw-1:-halfw]

    filtered_signal = np.real(ast)

    return filtered_signal


def morlet_power(fc, n, input_signal, fs):
    """Bandpass filter using convolution with complex Morlet wavelet
    Inputs:
    - fc =  central frequency of Morlet wavelet, this is the main frequency you are interested in for your signal
    - n = no. cycles of the Morlet wavelet, this will change the time-frequency resolution"""
    # 1. Create a complex morlet wavelet with central frequency fc and n number of cycles
    t_array = np.arange(-len(input_signal) / (2 * fs), len(input_signal) / (2 * fs), 1/fs)
    cmw = cmpl_morlet_wlt(fc, n, t_array)
    # 2. Do multiplication of wavelet and signal in the frequency domain
    n_conv = len(cmw) + len(input_signal) - 1
    _, cmw_f = to_freq_domain(cmw, n_conv)
    cmw_f /= np.max(cmw_f)
    _, input_signal_f = to_freq_domain(input_signal, n_conv)
    conv_result = np.multiply(cmw_f, input_signal_f)
    # 3. Retrieve the filtered signal with iFFT
    half_wav = int(np.floor(len(cmw) / 2))
    filtered_signal = np.fft.ifft(conv_result)
    filtered_signal = filtered_signal[half_wav - 1:-half_wav]
    pxx = np.abs(filtered_signal)**2
    freq = []
    return freq, pxx