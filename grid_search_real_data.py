"""
Author: Claudia Bigoni, claudia.bigoni@epfl.ch
Date: 05.01.2022, last update: 23.09.2022
Description: script to run a grid-search for best phase forecasting parameters on real resting state EEG data
"""
import numpy as np
import pandas as pd
import itertools
import random as rnd
from scipy import signal
import os
import time
import concurrent.futures
import mne

import epoch as epoch
import preprocessing as pp

# The default dataframe (columns' names)
df_default = pd.DataFrame(columns=['sub_id', 'ch', 'eye_cond', 'trial', 'algorithm', 'fs', 'f_name', 'f_order',
                                   'bandwidth', 'window_dur', 'rs', 'rp', 'phase_diff', 'phase_diff_2s', 'dur',
                                   'not_forecasted', 'pad', 'edge_cut', 'pxx_alpha_welch',  'pxx_total_welch'])

df_psd_default = pd.DataFrame(columns=['sub_id', 'ch', 'eye_cond', 'trial', 'fs', 'bandwidth', 'window_dur',
                                       'power_band', 'power_total', 'method', 'dur'])

# PSD methods computation:
psd_method_array = ['welch', 'fft']
pxx_total = {'welch': 0, 'fft': 0}
pxx_alpha = {'welch': 0, 'fft': 0}
nperseg_ = 1
noverlap_ = 0
order_ = 0.3
f_res = 1

def try_get_phase(item, sub_id, eye_cond, spatial_filter, eeg_data):
    """
    Use a "try except" on the get_phase function. If exception is found, return a dataframe with the same columns, but
    with nans in some fields.
    :param item: list with the following variables (in order!)
        - algorithm: string of name of forecasting algorithm to be used (bigoni, tomasevic, mansouri or zrenner)
        - bandwidth (bw): bandwidth around the main frequency of interest (Hz)
        - channel (ch): string of name of channel of interest (e.g., C3)
        - edge: fraction of input signal to be removed (this is related to filter edge effects)
        - eye condition (eye_cond): string describing if eyes were closed (EC) or open (EO) during resting state (RS)
        - sampling frequency (fs): sampling frequency to be used (this may not be the original of eeg_data) (Hz)
        - filter name (f_name): string
        - filter order (order): integer
        - padding (pad): boolean 1 or 0 if we need to pad the signal (if so, this is done with hard-coded manner. see
        get_phase function)
        - ripple bandpass (rp): float
        - ripple stopband (rs): float
        - window length (win): length of trials to be created from eeg_data (ms)
    :param sub_id: string describing the subject ID number
    :param eeg_data: mne format of raw eeg_data
    :return df: dataframe with inserted values for phase estimation error
    """
    df = df_default.copy()
    try:
        df = get_phase(item, sub_id, eye_cond, spatial_filter, eeg_data)
    except Exception as ex:
        print(f'There was an error in get_phase. Exception: {ex}')
        # Get the values inside item list
        algorithm, bw, ch, edge, fs, f_name, order, pad, rp, rs, win = item
        print(f'{algorithm}, {sub_id}, {bw}, {edge}, {fs}, {f_name}, {order}, {pad}, {rp}, {rs}, {win}')
        # Imagine there were no_trials for this combination of variables and save nans to the variables that were
        # supposed to be computed by get_phase
        for row in range(no_trials):
            df.loc[row] = pd.Series({'sub_id': sub_id, 'ch': ch, 'eye_cond': eye_cond, 'trial': row,
                                     'algorithm': algorithm, 'fs': fs, 'f_name': f_name, 'f_order': order, 'bandwidth':
                                         bw, 'rs': rs, 'rp': rp, 'window_dur': win,  'phase_diff': np.nan,
                                     'phase_diff_2s': np.nan, 'dur': np.nan, 'not_forecasted': True, 'edge_cut': edge,
                                     'pad': pad, 'pxx_alpha_welch': np.nan,  'pxx_total_welch': np.nan})
            row += 1

    return df


def get_phase(combo, sub_id, eye_cond, spatial_filter, eeg_data):
    """
    Given an EEG dataset (eeg_data), create no_trials epochs according to parameters in combo and predict for all the
    trials the sample in the future with the wanted phase (e.g., peak). Filters are done according to parameters in
    combo. Point prediction is done with functions in class Epoch.
    :param combo: list with the following variables (in order!)
        - algorithm: string of name of forecasting algorithm to be used (bigoni, tomasevic, mansouri or zrenner)
        - bandwidth (bw): bandwidth around the main frequency of interest (Hz)
        - channel (ch): string of name of channel of interest (e.g., C3)
        - edge: fraction of input signal to be removed (this is related to filter edge effects)
        - eye condition (eye_cond): string describing if eyes were closed (EC) or open (EO) during resting state (RS)
        - sampling frequency (fs): sampling frequency to be used (this may not be the original of eeg_data) (Hz)
        - filter name (f_name): string
        - filter order (order): integer
        - padding (pad): boolean 1 or 0 if we need to pad the signal (if so, this is done with hard-coded manner. see
        get_phase function)
        - ripple bandpass (rp): float
        - ripple stopband (rs): float
        - window length (win): length of trials to be created from eeg_data (ms)
    :param sub_id: string describing the subject ID number
    :param eeg_data: mne format of raw eeg_data
    :param spatial_filter: string describing type of spatial filter to be performed (None, car, laplacian)
    :return df: dataframe with inserted values for phase estimation error
    """
    # Initialize dataframe
    df = df_default.copy()
    row = 0

    # Get parameters from list combo
    algorithm, bw, ch, edge_cut, fs, f_name, order, pad, rp, rs, win = combo

    # Get info and data from eeg_data
    fs_original = int(eeg_data.info['sfreq'])
    raw_data_orig = eeg_data.get_data()
    ch_names = eeg_data.ch_names
    if spatial_filter:
        raw_data_ = pp.spatial_filter(raw_data_orig[:, :int(raw_data_orig.shape[1] / 2)], spatial_filter,
                                      ch_names, [ch])
        raw_data_ch = raw_data_[ch_names.index(ch)]
    else:
        raw_data_ch = raw_data_orig[ch_names.index(ch), :int(raw_data_orig.shape[1] / 2)]
    dur = len(raw_data_ch) / fs_original

    # Load for each subject their alpha peak frequency (previously computed) and get f_min and f_max
    df_mainf = pd.read_csv(alpha_peak_csv_dir)
    df_mainf['time_point'] = df_mainf['time_point'].fillna('')
    f0 = df_mainf[((df_mainf['group'] == 'Healthy') & (df_mainf['sub_id'] == sub_id) &
                   (df_mainf['time_point'] == '') & (df_mainf['channel'] == ch) &
                   (df_mainf['eye_cond'] == 'RS_EO'))]['max_alpha_welch'].values[0]
    f_min, f_max = f0 - bw / 2, f0 + bw / 2

    # Down-sample signal
    if fs != fs_original:
        raw_data = signal.resample(raw_data_ch, int(fs * len(raw_data_ch) / fs_original))
    else:
        raw_data = np.copy(raw_data_ch)
    # De-trend signal
    raw_data = signal.detrend(raw_data)

    del raw_data_orig
    del raw_data_ch
    del eeg_data

    # Create random trials out of the data:
    win_smpl = int(fs * win)
    events = np.random.randint(fs + 1, dur * fs - fs - 1, no_trials)
    events_start, events_stop = events - int(win_smpl / 2), events + int(win_smpl / 2)
    trials = np.stack([raw_data[events_start[ii]:events_stop[ii]] for ii in range(len(events))])

    # Pad the signal
    if pad == 0:
        pad_len = 0
        trials_pad = trials
    else:
        pad_len = int(0.5 * trials.shape[1])
        trials_pad = np.pad(trials, ((0, 0), (pad_len, pad_len)), mode='edge')

    # Filter all the trials and the full signal together
    if f_name == 'fir':
        trials_filtered = pp.fir_ls(trials_pad, f_min, f_max, fs, order, f0)
        all_filtered = pp.fir_ls(raw_data, f_min, f_max, fs, order, f0)
    elif f_name == 'wavelet':
        trials_filtered = trials_filtered = pp.morlet_filter_2(fs, f0, trials_pad)
        all_filtered = trials_filtered = pp.morlet_filter_2(fs, f0, raw_data)
    else:
        trials_filtered = pp.sos_iir_bandpass_filter(trials_pad, f_min=f_min, f_max=f_max, fs=fs, order=order, rp=rp,
                                                     rs=rs, name=f_name)
        all_filtered = pp.sos_iir_bandpass_filter(raw_data, f_min=f_min, f_max=f_max, fs=fs, order=order, rp=rp,
                                                  rs=rs, name=f_name)
    # get phases from the full filtered signal
    clean_phases = epoch.instantaneous_phases_sin(all_filtered)

    # Loop over trials to perdict the next peak
    trials_filtered = trials_filtered[:, :win_smpl]
    for trial_count, trial in enumerate(trials_filtered):
        trigger_early = 0
        trial_ep = epoch.Epoch(trial, trials[trial_count], fs, f_min, f_max, f_name=f_name, f_order=order)
        triggered_point = trial_ep.forecast(algorithm, int(edge_cut * win_smpl)) + events_start[trial_count]
        # if triggered_point < events_stop[trial_count]:
        #     triggered_point += int(fs/f0)
        #     trigger_early = 1

        # Compute phase error as the abs difference between 90deg (i.e., peak and expected phase that was forecasted)
        # and the actual phase
        phase_diff = (clean_phases[triggered_point] - np.pi / 2)

        # Compute phase error in epoch centered around the triggerd point and 2s long
        trial_2s_data = raw_data[triggered_point - fs:triggered_point + fs]
        trial_2s = epoch.Epoch(trial_2s_data, trial_2s_data, fs, f_min, f_max, f_name, order)
        phase_2s = trial_2s.phase_center_point()
        phase_diff_2s = (phase_2s - np.pi / 2)

        trial_pow = epoch.Power(trials[trial_count], f_min, f_max, fs, 'welch', f_res, nperseg_, noverlap_, order=order_)
        pxx_alpha, pxx_total = trial_pow.psd()
        # Save to dataframe
        df.loc[row] = pd.Series({'sub_id': sub_id, 'ch': ch, 'eye_cond': eye_cond, 'trial': trial_count,
                                 'algorithm': algorithm, 'fs': fs, 'f_name': f_name, 'f_order': order, 'bandwidth': bw,
                                 'rs': rs, 'rp': rp, 'window_dur': win, 'phase_diff': phase_diff, 'phase_diff_2s':
                                     phase_diff_2s, 'dur': trial_ep.dur, 'not_forecasted': trial_ep.not_forecasted,
                                 'edge_cut': edge_cut, 'pad': pad, 'pxx_alpha_welch': pxx_alpha,  'pxx_total_welch':
                                     pxx_total})
        row += 1

    return df

no_trials = 30
alpha_peak_csv_dir = 'C:/Users/bigoni/switchdrive/PhD/temp_data/subject_personal_psd.csv'  # csv containing alpha peaks

if __name__ == '__main__':
    # Define initial variables
    # Directories & file names
    data_dir = 'D:/Claudia/Resting_state'
    data_dir_sub = 'C:/Users/bigoni/Data/TMS-EEG'  # This directory is only used to get the subjects_IDs
    alpha_peak_csv_dir = 'C:/Users/bigoni/switchdrive/PhD/temp_data/subject_personal_psd.csv'  # csv containing alpha peaks
    # for each subject
    results_dir = 'D:/phase_forecast/real_gs/'
    df_name = 'grid_search_real_data_test'

    # Possible values for the parameters to be checked with the grid-search
    windows_array = [0.3] #, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]  # seconds
    fs_array = [250] #, 500, 1000, 5000]  # sampling frequency (Hz)
    edges_array = [0] #, 0.1, 0.3]  # in percentage of window dur -> edge to be removed from filtered window for edge
    # effects.
    padding_array = [0]#, 1]  # either pad the window before filtering or not. Next step could be to make a decision on
    # the length of padding. Here we use the same length as the input window.
    bandwidth_array = [1]#, 2, 3, 4]  # Hz - because we are looking at the alpha band for this example, only rather small
    # bandwidths are chosen
    order_array_iir = [2]#, 3, 4, 5, 6, 7, 8, 9, 10]
    order_array_fir = [0.2]#, 0.33]  # percentage of window to be used
    order_array_wavelet = [4]#, 4.5, 5, 5.5]  # number of cycles
    rs_array = [20]#, 40]
    rp_array = [0.1]#, 1]
    algorithms_array = ['bigoni']#, 'tomasevic', 'mansouri', 'zrenner']
    ch_list = ['C3']
    spatial_filter = 'None'

    # PSD methods computation:
    psd_method_array = ['welch', 'fft']
    pxx_total = {'welch': 0, 'fft': 0}
    pxx_alpha = {'welch': 0, 'fft': 0}
    nperseg_ = 1
    noverlap_ = 0
    order_ = 0.3
    f_res = 1

    rnd.seed(1)
    no_trials = 30

    # Create a dictionary with the possible values for the different parameters
    filters_dict = {
        'iir1': {
            'ch': ch_list,
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
            'ch': ch_list,
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
            'ch': ch_list,
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
            'ch': ch_list,
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
            'ch': ch_list,
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
            'ch': ch_list,
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
    psd_dict = {
        'ch': ch_list,
        'fs': fs_array,
        'window_length': windows_array,
        'method': psd_method_array,
        'bandwidth': bandwidth_array,
    }

    ####################################################################################################################
    # Start to loop over all possible combinations of parameters for a list of subjects
    df_count = 0
    eye_cond = 'RS_EO'
    for sub_id in ['WP12_006', 'WP12_015', 'WP12_020', 'WP12_001']:
        print(f'sub_id {sub_id}')
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

        allKeys = sorted(psd_dict)
        psd_dict_combo = itertools.product(*(psd_dict[Name] for Name in allKeys))

        # Load file once (for time/memory issues)
        sub_no = sub_id.split('_')[-1]
        filename = f'{eye_cond}_{sub_no}.set'
        eeg_data = mne.io.read_raw_eeglab(os.path.join(data_dir, 'Healthy', eye_cond, filename), preload=True)

        # PSD:
        # start_time = time.time()
        # executor = concurrent.futures.ProcessPoolExecutor(max_workers=3)
        # futures = [executor.submit(try_get_psd, combo, sub_id, eeg_data) for combo in psd_dict_combo]
        # concurrent.futures.wait(futures)
        # stop_time = time.time()
        # print(f'time to compute psd: {stop_time - start_time} s')
        # start_time = time.time()
        # results = [ii.result() for ii in futures]
        # print(f'Get results time: {time.time() - start_time}')
        # start_time = time.time()
        # df = pd.concat([results[i] for i in range(len(results))])
        # print(f'Time to concatenate: {time.time() - start_time}')
        # start_time = time.time()
        # df.to_csv(os.path.join(results_dir, 'all_filters', f'psd_{df_name}_{df_count}.csv'))
        # print(f'Time to save all together to csv: {time.time() - start_time}')
        # df_count += 1
        # del results
        # del df


        # IIR filters

        # IIR1 filters
        combo_name = 'iir1'
        start_time = time.time()
        executor = concurrent.futures.ProcessPoolExecutor()
        futures = [executor.submit(try_get_phase, combo, sub_id, eye_cond, spatial_filter, eeg_data) for combo in iir1_combo]
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

        # Elliptic filter
        combo_name = 'ellip'
        start_time = time.time()
        executor = concurrent.futures.ProcessPoolExecutor()
        futures = [executor.submit(try_get_phase, combo, sub_id, eye_cond, spatial_filter, eeg_data) for combo in iir_ellip_combo]
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
