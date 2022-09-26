"""
Author: Claudia Bigoni
Date: 05.01.2022, last update: 23.09.2022
Description: Class to forecast the phase given a trial. It contains approaches of Tomasevic&Siebner 2018;
Zrenner et al., 2018 and 2020; Mansouri et al., 2017 and 2018.
"""
import numpy as np
from scipy import signal
from scipy import fft
from scipy.integrate import simps
from scipy import fftpack
import time
import matplotlib.pyplot as plt
from mne import time_frequency as mne_tf

import preprocessing as pp

import matlab.engine  # NB: matlab is needed to run the autoregressive (AR) model and to compute PSD with Burg AR method
eng = matlab.engine.start_matlab()


def instantaneous_phases_sin(input_signal):
    """Compute the phase for each sample in input signal using Hilbert Function. Phases are wrapped between -pi and pi.
    Phases are shifted by 90 deg because Hilbert gives phases for a cosine, while we look at a sine wave.
    :param input_signal: 1D np.array of len L
    :return phases_wrap: 1D np.array of len L with values between -pi and pi"""
    analytic_signal = signal.hilbert(input_signal)  # get the analytic signal, applying Hilbert transform
    instantaneous_phase = (np.angle(analytic_signal) + np.pi / 2)   # phase shift by 90deg the phases
    # Get phases between -pi and pi
    phases_wrap = np.remainder(instantaneous_phase, 2 * np.pi)
    mask = np.abs(phases_wrap) > np.pi
    phases_wrap[mask] -= 2 * np.pi * np.sign(phases_wrap[mask])

    return phases_wrap


def fft_psd(input_signal, fs, nfft, window, detrend='None'):
    """Compute the power spectral density (PSD) using the Fast Fourier Transform (FFT)
    :@param input_signal: time-domain signal in shape of 1D np.array with length L
    :@param fs: sampling frequency of input signal (Hz)
    :@param nfft: number of cycles to be used, this is related to resolution to be obtained. Must be less than L FIXME
    :@param window: np.array with specifc shape (e.g., Hanning) to be multiplied with the signal. Must be of length L
    :@param detrend: string to define any type of detrend to apply to input_signal before applying FFT

    :@return freq: list of frequencies kept in the spectrum used (Hz)
    :@return pxx: PSD of each frequency in freq (V^2/Hz)
    """
    if len(window) != len(input_signal):
        print('Error: window and input signal must have the same length')
        return None

    if detrend == 'mean':
        input_signal -= np.mean(input_signal)
    elif detrend == 'median':
        input_signal -= np.median(input_signal)

    # Multiply the signal by the window in the time domain
    input_signal_win = input_signal * window

    freq_signal = np.fft.fft(input_signal_win, nfft)
    freq_signal /= np.sum(window ** 2)
    pxx_ = abs(freq_signal) ** 2
    pxx = 2 * pxx_
    pxx[0] = pxx_[0]
    pxx[-1] = pxx_[-1]
    freq = fft.rfftfreq(nfft, 1 / fs)

    return freq, pxx[:int(len(pxx) / 2 + 1)]


class Power:
    def __init__(self, input_signal, f_min, f_max, fs, method='welch', f_res=1, summing='sum', nperseg=1, noverlap=0,
                 order=0.3):
        self.v = input_signal
        self.fs = fs
        self.f_res = f_res
        self.f_min = f_min
        self.f_max = f_max
        self.main_f = (f_max + f_min) / 2
        self.method = method
        self.order = int(len(self.v) * order)
        self.nseg = int(len(self.v) * nperseg)
        self.noverlap = noverlap
        self.N = int(self.fs / self.f_res)
        self.summing = summing

        self.pxx, self.frequency = [], []

    def pad_epoch(self):
        """Pad the epoch to achieve the wanted frequency resolution"""
        new_N = len(self.v)
        N_diff = self.N - new_N
        trial_new = np.zeros(self.N)
        if N_diff > 0:
            try:
                trial_new = np.pad(self.v, (0, N_diff))
            except ValueError:
                N_diff = self.N - new_N + 1
                trial_new = np.pad(self.v, (0, N_diff))

            self.v = trial_new

    def psd(self):
        self.pad_epoch()
        self.spectrum()
        if self.method != 'amplitude':
            try:
                if self.frequency[np.argmin(abs(self.frequency - self.f_min))] < 7:
                    self.f_max = self.frequency[np.argmin(abs(self.frequency - self.f_max))]
                idx_freq = np.logical_and(self.frequency <= self.f_max, self.frequency >= self.f_min)
                idx_freq_50 = np.argmin(abs(self.frequency - 50))
                if self.summing == 'sum':
                    pxx_total = np.sum(self.pxx[:idx_freq_50])
                else:
                    pxx_total = simps(self.pxx[:idx_freq_50])
                if np.where(idx_freq)[0].shape[0] == 1:
                    pxx_band = self.pxx[idx_freq][0]
                else:
                    if self.summing == 'sum':
                        pxx_band = np.sum(self.pxx[idx_freq])
                    else:
                        pxx_band = simps(self.pxx[idx_freq])
            except Exception as ex:
                print(f"Can't compute PSD: {ex}")
                pxx_total, pxx_band = np.nan, np.nan
        else:
            sos = signal.butter(2, [0, 49], 'bandpass', fs=self.fs, output='sos')
            sig_filtered = signal.sosfiltfilt(sos, self.v)
            pxx_total = np.sum(np.power(sig_filtered, 2))
            pxx_band = self.pxx

        return pxx_band, pxx_total

    def spectrum(self):
        # self.pad_epoch()
        if self.method == 'fft':
            h_win = np.hanning(len(self.v))
            freq, pxx = fft_psd(self.v, self.fs, self.N, h_win, 'mean')
        elif self.method == 'burg':
            order = int(0.2 * len(self.v))
            [pxx, freq] = eng.pburg(matlab.double(self.v.tolist()), matlab.double([order]), matlab.double([self.N]),
                                    matlab.double([self.fs]), nargout=2)
            pxx = np.asarray(pxx)[:, 0]
            freq = np.asarray(freq).flatten()

        elif self.method == 'welch':
            nseg = min(self.nseg, self.N)
            [freq, pxx] = signal.welch(self.v, fs=self.fs, nperseg=len(self.v), noverlap=self.noverlap,
                                       nfft=len(self.v))

        elif self.method == 'multitaper':
            try:
                [pxx, freq] = mne_tf.psd_array_multitaper(self.v, sfreq=self.fs, fmin=0, fmax=self.fs / 2,
                                                          bandwidth=self.f_res, normalization='full')
            except ValueError:
                [pxx, freq] = mne_tf.psd_array_multitaper(self.v, sfreq=self.fs, fmin=0, fmax=self.fs / 2,
                                                          normalization='full')
        elif self.method == 'wavelet':
            freq, pxx = pp.morlet_power(self.main_f, self.order, self.v, self.fs)
        elif self.method == 'amplitude':
            # filter in bandwidth
            sos = signal.butter(2, [self.f_min, self.f_max], 'bandpass', fs=self.fs, output='sos')
            sig_filtered = signal.sosfiltfilt(sos, self.v)
            freq = np.arange(int(self.fs / 2))
            pxx = np.sum(np.power(sig_filtered, 2))
        else:
            pxx = np.nan
            freq = np.nan
            print('Method chosen {method} is not available')

        self.pxx_db = 10 * np.log10(pxx)
        self.pxx = pxx
        self.frequency = freq

    def peak_frequency(self):
        bins = np.where(np.logical_and(self.frequency >= self.f_min, self.frequency <= self.f_max))[0]
        main_f = np.argmax(self.pxx[bins]) + self.f_min
        # TO DO: Update with FOOOF methods
        return main_f


class Epoch:
    def __init__(self, input_signal, noisy_signal, fs, f_min=0, f_max=100, f_name='butter', f_order=2, rs=20, rp=20):
        self.v = input_signal  # latest data from (online) EEG
        self.v_n = noisy_signal
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        self.f_name = f_name
        self.f_order = f_order
        self.rs = rs
        self.rp = rp
        self.t_init = time.time()  # time of initialization of epoch, this should be when data has been received
        self.dur = 0
        self.too_late = True
        self.len_signal = len(self.v) / self.fs

        self.s_filter = 'none'
        self.power = Power(self.v_n, self.f_min, self.f_max, self.fs, method='fft')
        self.not_forecasted = False
        self.ar_order = 30

    def forecast(self, method, edge_cut=0):
        self.t_init = time.time()

        if edge_cut == 0:
            forecast_point = Forecast(self.v, self.fs, len(self.v_n), self.v_n)
        else:
            forecast_point = Forecast(self.v[:-edge_cut+1], self.fs, len(self.v_n), self.v_n)

        triggered_point = np.random.randint(2, 0.3 * self.fs) + len(self.v_n)    # init the point as random
        try:
            if 'bigoni' in method:
                triggered_point, late_flag = forecast_point.bigoni(self.power.main_f)
            elif 'tomasevic' in method:
                triggered_point, late_flag = forecast_point.tomasevic()
            elif 'zrenner' in method:
                triggered_point, late_flag = forecast_point.zrenner(self.ar_order)
            elif 'mansouri' in method:
                triggered_point, late_flag = forecast_point.mansouri(self.f_min, self.f_max)
            else:
                print('No available algorithm chosen. Please set method to "bigoni", "tomasevic", "zrenner" or '
                      '"mansouri".')
                forecast_point.too_late = 1
                forecast_point.dur = np.nan
        except Exception as ex:
            print(f"Exception in finding peak with method {method}: {ex}")
            forecast_point.too_late = 1
            forecast_point.dur = np.nan
            self.not_forecasted = True
        self.dur = forecast_point.dur

        return triggered_point

    def robust_phase(self, filter_params):
        """Apply different filters (all those described in self.filter_params) to the input signal self.v and return the
         phases in the middle point of the signal. This is the sample of interest of the signal"""
        phases = []
        for f_type in self.filter_params:
            for f0 in self.filter_params[f_type]['f0']:
                for bw in self.filter_params[f_type]['bandwidth']:
                    f_min, f_max = f0 - bw, f0 + bw
                    for f_name in self.filter_params[f_type]['name']:
                        for f_order in self.filter_params[f_type]['order']:
                            for rs in self.filter_params[f_type]['rs']:
                                for rp in self.filter_params[f_type]['rp']:
                                    filtered_signal = pp.sos_iir_bandpass_filter(self.v, f_min, f_max, self.fs, f_order,
                                                                                 rs, rp, f_name)
                                    phases.append(instantaneous_phases_sin(filtered_signal)[self.half_len])

        return phases, self.circular_dev(phases)

    @staticmethod
    def circular_dev(phases):
        sd = np.sqrt(-2 * np.log(np.sqrt(sum(np.sin(phases)) ** 2 + sum(np.cos(phases)) ** 2) / len(phases)))

        return sd

    def frequency_filter(self):
        if self.f_name == 'fir':
            filtered_signal = pp.fir_ls(self.v, self.f_min, self.f_max, self.fs, self.f_order, self.power.main_f)
        elif self.f_name == 'wavelet':
            filtered_signal = pp.morlet_filter_2(self.fs, self.power.main_f, self.v)
        else:
            filtered_signal = pp.iir_bandpass_filter(self.v, self.f_min, self.f_max, order=int(self.f_order),
                                                     name=self.f_name, fs=self.fs, rs=self.rs, rp=self.rp)

        return filtered_signal

    # def spatial_filter(self):
    #     self.v = pp.spatial_filter(self.v, self.s_filter_name, ch_names, [ch])

    def phase_center_point(self):
        # Compute the phase in the middle of the signal
        # self.spatial_filter()
        self.half_len = int(len(self.v) / 2)
        filtered_signal = self.frequency_filter()
        phase = instantaneous_phases_sin(filtered_signal)

        return phase[self.half_len]



class Forecast:
    def __init__(self, input_signal, fs, len_init, noisy_signal):
        """

        :param input_signal: 1D np.array with the signal to be used for forecasting (no pre-processing will be done here)
        :param fs: sampling frequency
        :param len_init: length of the actual input signal (should be the same of length of noisy_signal)
        :param noisy_signal: 1D np.array - raw signal
        """
        self.v = input_signal
        self.v_n = noisy_signal
        self.len_signal = len(self.v)
        self.fs = fs
        self.len_min = len_init

        self.t_limit = int(0.3 * self.fs) + self.len_min  # 300ms + length of the signal (not considering possible edge cuts)
        self.t_init = time.time()
        self.dur = 10

    def last_peak(self):
        """Get the last positive peak in the signal"""
        idx, _ = signal.find_peaks(self.v)

        return idx[-1]

    def last_trough(self):
        """Get the last negative peak (i.e., trough) in thr signal"""
        idx, _ = signal.find_peaks(-self.v)     # use flipped signal and find positive peaks

        return idx[-1]

    def predict_future_peaks(self, t_peak, period):
        """Given the latest peak of the signal and the period of the frequency of interest, predict the next peaks by
        adding the period samples to the initial peak until we reach the limit of the future window in which we allow to
        predict (t_limit)
        :param t_peak: sample of the last peak in the input signal self.v
        :param period: integer defining nuber of samples that should separate subsequent peaks

        :return future_peaks: list of samples (max length=10) of predicted peaks in the future (up to self.t_limit)
        """
        future_peaks = np.asarray(
            [t_peak + k * period for k in range(10) if self.len_min < t_peak + k * period < self.t_limit])

        return future_peaks

    def first_available_stimulation_point(self, t_point_list):
        """Given an array of possible stimulation times,and taking into account the computation delay, say which point
        we should use. If there are no available points, give the last one and flag the issue.
         :param t_point_list:
         """
        available_points = t_point_list[t_point_list > int(self.dur * self.fs)]
        if len(available_points):
            stimulation_point = available_points[0]
            late_flag = False
        else:
            stimulation_point = t_point_list[-1]
            late_flag = True
            print('Too late!')

        return stimulation_point, late_flag

    def bigoni(self, main_f):
        """Find the last peak of the input signal and use the period as the main frequency in the spectrum of the input
        signal. The inverse of this will give the period
         :param main_f: frequency in Hz of main frequency in the band of interest

         :return triggered_point: sample number of when phase of interest is first found in the future
         :return late_flag: boolean saying if the triggered_point is outside the limit of the future window
        """
        period = int(1 / main_f * self.fs)
        peak = self.last_peak()
        future_peaks = self.predict_future_peaks(peak, period)
        self.dur = time.time() - self.t_init
        # Check if we can stimulate at one of those points time=0, is the end of the past window (i.e. beginning of
        # forecast period)
        triggered_point, late_flag = self.first_available_stimulation_point(future_peaks)

        return triggered_point, late_flag

    def tomasevic(self):
        """
        Find the last peak and trough of the input signal. The period is then 2*distance between peak and trough.

         :return triggered_point: sample number of when phase of interest is first found in the future
         :return late_flag: boolean saying if the triggered_point is outside the limit of the future window
        """
        peak = self.last_peak()
        trough = self.last_trough()
        period = abs(peak - trough)*2
        future_peaks = self.predict_future_peaks(peak, period)
        self.dur = time.time() - self.t_init
        # Check if we can stimulate at one of those points time=0, is the end of the past window (
        # i.e. beginning of forecast period)
        triggered_point, late_flag = self.first_available_stimulation_point(future_peaks)

        return triggered_point, late_flag

    def mansouri(self, f_min, f_max):
        """Pad the signal with 5000 samples using the last sample available of self.v. Find the main frequency and
        relative phase of the input signal using FFT. Then use these parameters to create a sinusoide with frequency and
        phase offset the values found with FFT. The sinusoid is of length
        L = len_input_signal + len_future_win(i.e., 300ms). Find the peaks in the future window. This approach is taken
        from Manosuri et al., 2017 Frontiers Neuroscience
         :param f_min:
         :param f_max:

         :return triggered_point: sample number of when phase of interest is first found in the future
         :return late_flag: boolean saying if the triggered_point is outside the limit of the future window
        """
        # 1. Pad the signal
        padded_signal = np.pad(self.v_n, 5000, mode='constant', constant_values=0)
        # 2. Apply the FFT to find main frequency in the band of interest and relative phase
        X = fftpack.fft(padded_signal)
        X_mod, X_ang = np.square(X), np.angle(X)
        freqs = fftpack.fftfreq(len(padded_signal)) * self.fs
        L = len(freqs)
        freqs = freqs[:L // 2 - 1]
        X_mod = X_mod[:L // 2 - 1]
        X_ang = X_ang[:L // 2 - 1]
        idx_freq_min, idx_freq_max = np.argmin(abs(freqs-f_min)), np.argmin(abs(freqs-f_max))
        main_f_idx = np.argmax(X_mod[idx_freq_min:idx_freq_max]) + idx_freq_min
        main_f, phase = freqs[main_f_idx], X_ang[main_f_idx]

        # 3. Sinusoid approximation
        t = np.arange(0, self.t_limit / self.fs, 1 / self.fs)   # array of time points on which sinusoid is defined
        # Create the sinousoid with freqeuncy the main found and phase offset phase found. Add additional 90deg to move
        # from cosine to sine.
        forecasted_signal = np.sin(main_f * 2 * np.pi * t - phase + np.pi/2)
        # 4. Find peaks in the future window
        future_peaks, _ = signal.find_peaks(forecasted_signal[self.len_min:])
        future_peaks += self.len_min
        self.dur = time.time() - self.t_init
        triggered_point, late_flag = self.first_available_stimulation_point(future_peaks[future_peaks>self.len_min])

        return triggered_point, late_flag

    def zrenner(self, ar_order):
        """Predict the signal in the future window with a Yule-Walker autoregressive model (function from Matlab). Find
        the peaks in the future. This approach is described in Zrenner et al., 2018 Brain Stimulation.

        :param ar_order: integer describing the order of the autoregressive model (i.e., number of latest points used
        to predict each new sample).

        :return triggered_point: sample number of when phase of interest is first found in the future
        :return late_flag: boolean saying if the triggered_point is outside the limit of the future window
         """
        signal_matlab = matlab.double(list(self.v))     # transform signal to Matlab readable
        ar_coefs_matlab = eng.aryule(signal_matlab, ar_order)   # get coefficients of AR model
        ar_coefs = np.asarray(ar_coefs_matlab[0])  # Transform coefficients them to python readable
        prediction_len = self.t_limit - self.len_min    # Define length of window for prediction
        data_predicted = np.zeros((1, prediction_len))  # Initialize array to save predictions
        data_past = self.v[-ar_order:].T
        for idx in range(prediction_len):
            # Coeff with different sing. Do not consider the first coeff because is like a normalization factor.
            # data_past is reveresed so that first coeff is multiplied by last value
            data_predicted[0, idx] = np.dot(-ar_coefs[1:], data_past[::-1])
            data_past = np.append(data_past[1:], data_predicted[0, idx])
        data_predicted = data_predicted[0]

        future_peaks, _ = signal.find_peaks(data_predicted)
        self.dur = time.time() - self.t_init
        future_peaks = np.sort(future_peaks) + self.len_signal
        triggered_point, late_flag = self.first_available_stimulation_point(future_peaks[future_peaks>self.len_min])

        return triggered_point, late_flag
