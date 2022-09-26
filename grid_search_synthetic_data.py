"""
Author: Claudia Bigoni, claudia.bigoni@epfl.ch
Date: 05.01.2022; last update: 23.09.2022
Description: Main script to run a grid-search for best phase forecasting parameters (i.e., EEG preprocessing and
forecasting methods) on synthetic EEG-like data. The input data is sine-wave in a freqeuncy band of interest (e.g.,
10Hz) with added pink (1/f) noise.
"""

import numpy as np
import pandas as pd
import itertools
import random as rnd
from scipy import signal
import os
import time
import concurrent.futures

import data_gen as data
import epoch as epoch
import preprocessing as pp


def try_get_phase(item):
    """
    Function that gets called by the parallel computing. It tries to call the actual function to get the forecasting and
    catches any error if they occur.
    :param item: 1D list of length 11. It contains the values needed for generating the signal and to forecast the
    phase.
    :return df:dataframe containing the value of the parameters used and the phase error obtained for all the trials.
    """
    try:
        df = get_phase(item)
    except Exception as ex:
        # If a problem occurs, still return a dataframe so that it can be checked with combinations did not work
        print(f'An exception has occurred while trying to forecast the phase. The exception caught is: {ex}')
        df = df_default.copy()
        row = 0
        algorithm, amp, bw, edge, fs, f_name, order, pad, rp, rs, win = item  # NB: they are in alphabetical order
        for trial_count in range(no_trials):
            df.loc[row] = pd.Series({
                'trial': trial_count, 'algorithm': algorithm, 'fs': fs, 'f_name': f_name, 'f_order': order,
                'f_bandwidth': bw, 'f_rs': rs, 'f_rp': rp, 'window_len': win, 'pad': pad, 'edge_cut': edge,
                'amplitude': amp, 'phase_diff': np.nan, 'dur': np.nan, 'not_forecasted': np.nan})
            row += 1

    return df


def get_phase(combo):
    """
    Forecast the phase using input combination of parameters.
    Steps are the following:
    1) Create the data (ground truth + noisy data to use for forecast)
    2) Re-sample signals if needed
    3) Detrend signals
    4) Get trials from signal
    5) For each trial:
    5) Pad the signal (if needed)
    6) Filter the signal
    7) Edge-cut the signal (if needed)
    8) Forecast time point where next wanted phase will be (e.g., peak)
    9) Compute error using the ground truth signal
    10) Save to dataframe

    :param combo: 1D list of length 11. It contains the values needed for generating the signal and to forecast the
    phase
    :return df: dataframe containing the value of the parameters used and the phase error obtained for all the trials
    """
    # Initialize dataframe where data will be saved
    df = df_default.copy()
    row = 0

    # Get the needed variables
    algorithm, amp, bw, edge_cut, fs, f_name, order, pad, rp, rs, win = combo   # NB: they are in alphabetical order
    f_min, f_max = f0 - bw / 2, f0 + bw / 2

    # 1. Make the signal (underlying one) + noisy one
    clean_data_orig = amp * data.generate_sinusoid(f0, dur, fs_original)
    if amp == 100:  # this is equivalent to condition where no noise is present - it's a control that the combination
        # generally works
        raw_data_orig = np.copy(clean_data_orig)
    else:
        raw_data_orig = data.add_random_noise(clean_data_orig, fs_original)

    # 2. Re-sample
    if fs != fs_original:
        raw_data = signal.resample(raw_data_orig, int(fs * len(raw_data_orig) / fs_original))
        clean_data = signal.resample(clean_data_orig, int(fs * len(clean_data_orig) / fs_original))
    else:
        raw_data = np.copy(raw_data_orig)
        clean_data = np.copy(clean_data_orig)

    # 3. De-trend both signals
    raw_data = signal.detrend(raw_data)
    clean_data = signal.detrend(clean_data)

    # Get the real phases in each time point
    clean_phases = epoch.instantaneous_phases_sin(clean_data)

    # 4. Create trials out of the data:
    win_smpl = int(fs * win)
    events = np.random.randint(fs + 1, dur * fs - fs - 1, no_trials)
    events_start, events_stop = events - int(win_smpl / 2), events + int(win_smpl / 2)
    trials = np.stack([raw_data[events_start[ii]:events_stop[ii]] for ii in range(len(events))])

    # 5. Pad the signal
    if pad == 0:
        pad_len = 0
        trials_pad = trials
    else:
        pad_len = int(0.5 * trials.shape[1])
        trials_pad = np.pad(trials, ((0, 0), (pad_len, pad_len)), mode='edge')

    # 6. Filter the signal
    if f_name == 'fir':
        trials_filtered = pp.fir_ls(trials_pad, f_min, f_max, fs, order, f0)
    elif f_name == 'wavelet':
        trials_filtered = pp.morlet_filter_2(fs, f0, trials_pad)
    else:
        trials_filtered = pp.sos_iir_bandpass_filter(trials_pad, f_min=f_min, f_max=f_max, fs=fs, order=order, rp=rp,
                                                     rs=rs, name=f_name)
    # Remove the padding
    trials_filtered = trials_filtered[:, :win_smpl]
    for trial_count, trial in enumerate(trials_filtered):
        # 7-8. Forecast time point when phase will be met for each trial
        trial_ep = epoch.Epoch(trial, trials[trial_count], fs, f_min, f_max, f_name, f_order=order)
        triggered_point = trial_ep.forecast(algorithm, int(edge_cut * win_smpl)) + events_start[trial_count]

        phase_diff = (clean_phases[triggered_point] - np.pi / 2)    # Compute phase error

        # 9. Save in dataframe
        df.loc[row] = pd.Series({
            'trial': trial_count, 'algorithm': algorithm, 'fs': fs, 'f_name': f_name, 'f_order': order, 'f_bandwidth':
                bw, 'f_rs': rs, 'f_rp': rp, 'window_len': win, 'pad': pad, 'edge_cut': edge_cut, 'amplitude': amp,
            'phase_diff': phase_diff, 'dur': trial_ep.dur, 'not_forecasted': trial_ep.not_forecasted})
        row += 1
    return df


# Default dataframe (with columns names)
df_default = pd.DataFrame(columns=['trial', 'algorithm', 'fs', 'f_name', 'f_order', 'f_bandwidth', 'f_rs', 'f_rp',
                                   'window_len', 'pad', 'edge_cut', 'amplitude', 'phase_diff', 'dur', 'not_forecasted'])
f0 = 10
no_trials = 30
fs_original = 5000  # fs with which we create the data (Hz)
dur = 50  # length in seconds of generated data
if __name__ == '__main__':
    # Define initial variables
    # Directories & file names
    results_dir = 'D:/phase_forecast/all'
    df_name = 'grid_search_synthetic_data_test'
    df_psd_name = 'grid_search_synthetic_data_psd_test'

    # Possible values for the parameters to be checked with the grid-search
    windows_array = [0.3]#, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]  # seconds
    fs_array = [250]#, 500, 1000, 5000]  # sampling frequency (Hz)
    edges_array = [0]#, 0.1, 0.3]  # in percentage of window dur -> edge to be removed from filtered window for edge
    # effects.
    padding_array = [0]#, 1]  # either pad the window before filtering or not. Next step could be to make a decision on
    # the length of padding. Here we use the same length as the input window.
    bandwidth_array = [1]#, 2, 3, 4]  # Hz - because we are looking at the alpha band for this example, only rather small
    # bandwidths are chosen
    order_array_iir = [2]#, 3, 4]
    order_array_fir = [0.2]#, 0.33]  # percentage of window to be used
    order_array_wavelet = [4]#, 4.5, 5, 5.5]  # number of cycles
    rs_array = [20]#, 40]
    rp_array = [0.1]#, 1]
    algorithms_array = ['bigoni']#, 'tomasevic', 'mansouri', 'zrenner']

    # Dataset creation
    fs_original = 5000  # fs with which we create the data (Hz)
    dur = 50  # length in seconds of generated data
    no_trials = 30
    amplitudes_array = [100, 5, 1, 0.5, 0.1, 0.05, 0.01]  # amplitude of the underlying signal, as opposite to noise
    f0 = 10  # underlying important frequency (Hz)

    # PSD methods computation:
    psd_method_array = ['welch', 'fft']
    pxx_total = {'welch': 0, 'fft': 0}
    pxx_alpha = {'welch': 0, 'fft': 0}
    nperseg_ = 1
    noverlap_ = 0
    order_ = 0.3
    f_res = 1

    # Create a dictionary with the possible values for the different parameters
    filters_dict = {
        'iir1': {
            'amplitude': amplitudes_array,  # NB: amplitude is not an actual hyperparameter. It is just easier for
            # computation to have it here
            'fs': fs_array,
            'window_length': windows_array,
            'algorithm': algorithms_array,
            'edges': edges_array,
            'pad': padding_array,
            'bandwidth': bandwidth_array,
            'name': ['butter', 'bessel'],
            'order': order_array_iir,
            'rs': [1], 'rp': [1]},  # for these filters, rs and rp are not important inputs
        'iir_ellip': {
            'amplitude': amplitudes_array,
            'fs': fs_array,
            'window_length': windows_array,
            'algorithm': algorithms_array,
            'edges': edges_array,
            'pad': padding_array,
            'bandwidth': bandwidth_array,
            'name': ['ellip'],
            'order': order_array_iir,
            'rs': rs_array, 'rp': rp_array},
        'iir_cheby1': {
            'amplitude': amplitudes_array,
            'fs': fs_array,
            'window_length': windows_array,
            'algorithm': algorithms_array,
            'edges': edges_array,
            'pad': padding_array,
            'bandwidth': bandwidth_array,
            'name': ['cheby1'],
            'order': order_array_iir,
            'rs': [1], 'rp': rp_array},
        'iir_cheby2': {
            'amplitude': amplitudes_array,
            'fs': fs_array,
            'window_length': windows_array,
            'algorithm': algorithms_array,
            'edges': edges_array,
            'pad': padding_array,
            'bandwidth': bandwidth_array,
            'name': ['cheby1'],
            'order': order_array_iir,
            'rs': rs_array, 'rp': [1]},
        'wavelet': {
            'amplitude': amplitudes_array,
            'fs': fs_array,
            'window_length': windows_array,
            'algorithm': algorithms_array,
            'edges': edges_array,
            'pad': padding_array,
            'bandwidth': [1],
            'name': ['wavelet'],
            'order': order_array_wavelet,
            'rs': [1], 'rp': [1]},
        'fir': {
            'amplitude': amplitudes_array,
            'fs': fs_array,
            'window_length': windows_array,
            'algorithm': algorithms_array,
            'edges': edges_array,
            'pad': padding_array,
            'bandwidth': bandwidth_array,
            'name': ['fir'],
            'order': order_array_fir,
            'rs': [1], 'rp': [1]}
    }

    # Create all possible combinations of parameters
    allKeys = sorted(filters_dict['iir1'])
    iir1_combo = itertools.product(*(filters_dict['iir1'][Name] for Name in allKeys))
    allKeys = sorted(filters_dict['iir_ellip'])
    iir_ellip_combo = itertools.product(*(filters_dict['iir_ellip'][Name] for Name in allKeys))
    allKeys = sorted(filters_dict['iir_cheby1'])
    iir_cheby1_combo = itertools.product(*(filters_dict['iir_cheby1'][Name] for Name in allKeys))
    allKeys = sorted(filters_dict['iir_cheby2'])
    iir_cheby2_combo = itertools.product(*(filters_dict['iir_cheby2'][Name] for Name in allKeys))
    allKeys = sorted(filters_dict['wavelet'])
    wav_combo = itertools.product(*(filters_dict['wavelet'][Name] for Name in allKeys))
    allKeys = sorted(filters_dict['fir'])
    fir_combo = itertools.product(*(filters_dict['fir'][Name] for Name in allKeys))

    # One per algorithm
    # Create dataframe to save all parameters
    df_psd = pd.DataFrame(columns=['trial', 'fs', 'window_dur', 'amplitude', 'psd_welch_alpha',
                                   'psd_welch_tot', 'psd_fft_alpha', 'psd_fft_tot'])

    df = pd.DataFrame(columns=['trial', 'algorithm', 'fs', 'f_name', 'f_order', 'bandwidth', 'window_dur', 'rs',
                               'rp', 'phase_diff', 'phase_diff_deg_abs', 'amplitude', 'dur',
                               'not_forecasted', 'pad', 'edge_cut', 'noise'])

    df_row = 0
    df_psd_row = 0
    df_psd_count = 0

    rnd.seed(25145)  # for consistency when using random

    ####################################################################################################################
    # Start to loop over all possible combinations of parameters - using parallel computing
    print('Starting the grid-search')
    # IIR1 filters
    combo_name = 'iir1'
    start_time = time.time()
    executor = concurrent.futures.ProcessPoolExecutor()
    futures = [executor.submit(try_get_phase, combo) for combo in iir1_combo]
    concurrent.futures.wait(futures)
    stop_time = time.time()
    print(f'Time to compute all the combinations for {combo_name}: {stop_time - start_time} s')
    start_time = time.time()
    results = [ii.result() for ii in futures]
    print(f'Time to get results: {time.time() - start_time}')
    start_time = time.time()
    df = pd.concat([results[i] for i in range(len(results))])
    print(f'Time to concatenate all the dataframes obtained: {time.time() - start_time}')
    start_time = time.time()
    df.to_csv(os.path.join(results_dir, f'{df_name}_{combo_name}.csv'))
    print(f'Time to save to csv: {time.time() - start_time}')

    # Ellip filter
    combo_name = 'ellip'
    start_time = time.time()
    executor = concurrent.futures.ProcessPoolExecutor()
    futures = [executor.submit(try_get_phase, combo) for combo in iir_ellip_combo]
    concurrent.futures.wait(futures)
    stop_time = time.time()
    print(f'Time to compute all the combinations for {combo_name}: {stop_time - start_time} s')
    start_time = time.time()
    results = [ii.result() for ii in futures]
    print(f'Time to get results: {time.time() - start_time}')
    start_time = time.time()
    df = pd.concat([results[i] for i in range(len(results))])
    print(f'Time to concatenate all the dataframes obtained: {time.time() - start_time}')
    start_time = time.time()
    df.to_csv(os.path.join(results_dir, f'{df_name}_{combo_name}.csv'))
    print(f'Time to save to csv: {time.time() - start_time}')

    # Same for the other combinations...
