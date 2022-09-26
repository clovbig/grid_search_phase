"""
Author: Claudia Bigoni, claudia.bigoni@epfl.ch
Date: 05.01.2022, last update: 23.09.2022
Description: Test specific combination of parameters for EEG processing & forecasting on synthetic and real data
"""
import numpy as np
import pandas as pd
import random as rnd
from scipy import signal
import os
import time
import mne
import copy

import data_gen as data
import epoch as epoch
import preprocessing as pp


def try_get_phase_psd(item, combo_no, add_sinusoids=0):
    """
    Function that gets called by the parallel computing. It tries to call the actual function to get the forecasting and
    catches any error if they occur.
    :param item: 1D list of length 11. It contains the values needed for generating the signal and to forecast the
    phase.
    :param combo_no: number of the combination tested (to recall among those that were pre-selected)
    :param add_sinusoids: boolean of adding to the synthetic signal sinusoids in frequencies nearby to that of interest
    :return df:dataframe containing the value of the parameters used and the phase error obtained for all the trials.
    """
    try:
        df = get_phase_psd(item, combo_no, add_sinusoids)
    except Exception as ex:
        # If a problem occurs, still return a dataframe so that it can be checked with combinations did not work
        print(f'An exception has occurred while trying to forecast the phase. The exception caught is: {ex}')
        df = df_dflt_syn.copy()
        row = 0

        algorithm, amp, bw, edge, f0, f_name, f_order, fs, pad, psd_method, rp, rs, win = item

        for trial_count in range(no_trials):
            df.loc[row] = pd.Series({
                'trial': trial_count, 'algorithm': algorithm, 'fs': fs, 'f_name': f_name, 'f_order': order, 'bandwidth':
                    bw, 'rs': rs, 'rp': rp, 'window_dur': win, 'amplitude': amp, 'phase_diff': np.nan, 'dur': np.nan,
                'not_forecasted': True, 'edge_cut': edge, 'pad': pad, 'psd_alpha': np.nan, 'psd_total': np.nan,
                'combo': combo_no, 'add_sinusoids': add_sinusoids})
            row += 1

    return df


def get_phase_psd(combo, combo_no, add_sinusoids):
    """
    Forecast the phase using input combination of parameters and also compute the PSD.
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
    :param combo_no: number of the combination tested (to recall among those that were pre-selected)
    :param add_sinusoids: boolean of adding to the synthetic signal sinusoids in frequencies nearby to that of interest
    :return df: dataframe containing the value of the parameters used and the phase error obtained for all the trials
    """
    # Initialize dataframe where data will be saved
    df = df_dflt_syn.copy()
    row = 0

    # Get the needed variables
    algorithm, amp, bw, edge, f0, f_name, f_order, fs, pad, psd_method, rp, rs, win = combo   # NB: they are in alphabetical order
    f_min, f_max = f0 - bw / 2, f0 + bw / 2

    # 1. Make the signal (underlying one) + noisy one
    clean_data_orig = amp * data.generate_sinusoid(f0, dur, fs_original)
    if amp == 100:  # this is equivalent to condition where no noise is present - it's a control that the combination
        # generally works
        raw_data_orig = np.copy(clean_data_orig)
    else:
        raw_data_orig = data.add_random_noise(clean_data_orig, fs_original)
    if add_sinusoids:
        t = np.arange(0, dur, 1 / fs_original)
        raw_data_orig += amp * (np.sin(9 * 2 * np.pi * t + np.pi / 4) + np.sin(8 * 2 * np.pi * t + np.pi / 8) + np.sin(
            11 * 2 * np.pi * t) + np.sin(12 * 2 * np.pi * t - np.pi / 3))

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
        trials_filtered = pp.fir_ls(trials_pad, f_min, f_max, fs, f_order, f0)
    elif f_name == 'wavelet':
        trials_filtered = pp.morlet_filter_2(fs, f0, trials_pad)
    else:
        trials_filtered = pp.sos_iir_bandpass_filter(trials_pad, f_min=f_min, f_max=f_max, fs=fs, order=f_order, rp=rp,
                                                     rs=rs, name=f_name)
    # Remove the padding
    trials_filtered = trials_filtered[:, :win_smpl]
    for trial_count, trial in enumerate(trials_filtered):
        # 7-8. Forecast time point when phase will be met for each trial
        trial_ep = epoch.Epoch(trial, trials[trial_count], fs, f_min, f_max, f_name, f_order=f_order)
        triggered_point = trial_ep.forecast(algorithm, int(edge * win_smpl)) + events_start[trial_count]

        phase_diff = (clean_phases[triggered_point] - np.pi / 2)    # Compute phase error

        # Compute power
        trial_pow = epoch.Power(trial, f_min, f_max, fs, psd_method, f_res, nperseg_, noverlap_, order=order_)
        pxx_alpha, pxx_total = trial_pow.psd()

        # 9. Save in dataframe
        df.loc[row] = pd.Series({
            'trial': trial_count, 'algorithm': algorithm, 'fs': fs, 'f_name': f_name, 'f_order': f_order, 'f_bandwidth':
                bw, 'f_rs': rs, 'f_rp': rp, 'window_len': win, 'pad': pad, 'edge_cut': edge, 'amplitude': amp,
            'phase_diff': phase_diff, 'dur': trial_ep.dur, 'not_forecasted': trial_ep.not_forecasted, 'psd_alpha':
                pxx_alpha, 'psd_total': pxx_total, 'combo': combo_no, 'add_sinusoids': add_sinusoids})
        row += 1

    return df


def try_get_phase_psd_real(group, sub_id, tp, eye_cond, algorithm, combo_no, spatial_filter, eeg_data):
    """
    Function that gets called by the parallel computing. It tries to call the actual function to get the forecasting and
    catches any error if they occur.
    :param group: string describing co-hort (stroke or healthy)
    :param sub_id: string defining ID
    :param tp: string defining time point (empty if healthy), otherwise T1, T2, T3, or T4
    :param eye_cond: string defining eyes condition (close or open during the resting-state)
    :param algorithm: string (bigoni, mansouri, tomasevic or zrenner)
    :param combo_no: number of the combination tested (to recall among those that were pre-selected)
    :param spatial_filter: string saying type of filter to be used (so far only car or small_laplacian/hjort)
    :param eeg_data: mne raw class with resting state
    :return df:dataframe containing the value of the parameters used and the phase error obtained for all the trials.
    """
    try:
        df = get_phase_psd_real(group, sub_id, tp, eye_cond, algorithm, combo_no, spatial_filter, eeg_data)
    except Exception as ex:
        # If a problem occurs, still return a dataframe so that it can be checked with combinations did not work
        print(f'There was an error when trying to forecast the phase in real data: Exception caught: {ex}')

        df = df_dflt_real.copy()
        row = 0
        bw, edge, f0, f_name, f_order, fs, pad, psd_method, rp, rs, win = [params_dict[algorithm][i] for i in
                                                                           sorted(params_dict[algorithm])]
        for trial_count in range(no_trials):
            # NB: channels are not known because we test all of them together at the same time in the function
            df.loc[row] = pd.Series({
                'group': group, 'sub_id': sub_id, 'eye_cond': eye_cond, 'ch': '', 'time_point': tp,
                'trial': trial_count, 'algorithm': algorithm, 'fs': fs, 'f_name': f_name, 'f_order': f_order,
                'bandwidth': bw, 'rs': rs, 'rp': rp, 'window_dur': win, 'phase_diff': np.nan, 'phase_diff_2s': np.nan,
                'dur': np.nan, 'not_forecasted': True, 'edge_cut': edge, 'pad': pad, 'psd_alpha': np.nan,
                'psd_total': np.nan, 'combo': combo_no, 'spatial_filter': spatial_filter})
            row += 1

    return df


def get_phase_psd_real(group, sub_id, tp, eye_cond, algorithm, combo_no, spatial_filter, eeg_data):
    """
    Forecast the phase using input combination of parameters and also compute the PSD.
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
    :param combo_no: number of the combination tested (to recall among those that were pre-selected)
    :param add_sinusoids: boolean of adding to the synthetic signal sinusoids in frequencies nearby to that of interest
    :return df: dataframe containing the value of the parameters used and the phase error obtained for all the trials
    """
    # Initialize dataframe where data will be saved
    df = df_dflt_real.copy()
    row = 0

    try:
        bw, edge, f0, f_name, f_order, fs, pad, psd_method, rp, rs, win = [params_dict[algorithm][i][combo_no] for i in
                                                                           sorted(params_dict[algorithm])]  # FIXME
        fs_original = int(eeg_data.info['sfreq'])

        # If in 1st step of validation choose to take only the first half and then just the second half. Else we expect
        # that with a different seed, not the same trials will be picked anyway.
        raw_data_orig = eeg_data.get_data()
        for ch in channels:
            if spatial_filter:
                raw_data_ = pp.spatial_filter(
                    raw_data_orig[:, :int(raw_data_orig.shape[1] / 2)],
                    spatial_filter, eeg_data.ch_names, [ch])
                raw_data_ch = raw_data_[eeg_data.ch_names.index(ch), :int(raw_data_orig.shape[1] / 2)]
            else:
                raw_data_ch = raw_data_orig[eeg_data.ch_names.index(ch), :int(raw_data_orig.shape[1] / 2)]

            # Take main frequency found from the full resting-state (pre-saved in a dataframe)
            df_mainf = pd.read_csv('C:/Users/bigoni/switchdrive/PhD/temp_data/subject_personal_psd.csv')
            df_mainf['time_point'] = df_mainf['time_point'].fillna('')
            f0 = df_mainf[((df_mainf['group'] == group) & (df_mainf['sub_id'] == sub_id) &
                           (df_mainf['time_point'] == tp) & (df_mainf['channel'] == ch) &
                           (df_mainf['eye_cond'] == 'RS_EO'))]['max_alpha_welch'].values[0]
            f_min, f_max = f0 - bw / 2, f0 + bw / 2
            if f_min < 7:
                f_delta = abs(7-f_min)
                f_min, f_max = 7, f_max + f_delta
            if f_max > 13:
                f_delta = abs(13-f_max)
                f_min, f_max = f_min - f_delta, 13

            # Re-sample
            if fs != fs_original:
                raw_data = signal.resample(raw_data_ch, int(fs * len(raw_data_ch) / fs_original))
            else:
                raw_data = np.copy(raw_data_ch)

            # Detrend signal
            raw_data = signal.detrend(raw_data)

            # Create trials out of the data:
            dur = len(raw_data_ch) / fs_original
            win_smpl = int(fs * win)
            events = np.random.randint(fs + 1, dur * fs - fs - 1, no_trials)
            events_start, events_stop = events - int(win_smpl / 2), events + int(win_smpl / 2)
            trials = np.stack([raw_data[events_start[ii]:events_stop[ii]] for ii in range(len(events))])

            # Padding
            if pad == 0:
                edge_cut = edge
                trials_pad = trials
            else:
                edge_cut = edge
                pad_len = int(0.5 * trials.shape[1])
                trials_pad = np.pad(trials, ((0, 0), (pad_len, pad_len)), mode='edge')
            try:
                if f_name == 'fir':
                    trials_filtered = pp.fir_ls(trials_pad, f_min, f_max, fs, f_order, f0)
                    all_filtered = pp.fir_ls(raw_data, f_min, f_max, fs, f_order, f0)
                elif f_name == 'wavelet':
                    trials_filtered = []
                    all_filtered = []
                else:
                    trials_filtered = pp.sos_iir_bandpass_filter(trials_pad, f_min=f_min, f_max=f_max, fs=fs,
                                                                 order=f_order, rp=rp, rs=rs, name=f_name)
                    all_filtered = pp.sos_iir_bandpass_filter(raw_data, f_min=f_min, f_max=f_max, fs=fs, order=f_order,
                                                              rp=rp, rs=rs, name=f_name)

                clean_phases = epoch.instantaneous_phases_sin(all_filtered)  # On full resting-state

                for trial_count, trial in enumerate(trials_filtered):
                    try:
                        trial_ep = epoch.Epoch(trial, trials[trial_count], fs, f_min, f_max, 'none', f_order=f_order)
                        triggered_point = trial_ep.forecast(algorithm, pad_len, int(edge_cut * win_smpl)) + \
                                          events_start[trial_count]
                        phase_diff = (clean_phases[triggered_point] - np.pi / 2)

                        # Create epoch around 2 s
                        trial_2s_data = raw_data[triggered_point - fs:triggered_point + fs]
                        trial_2s = epoch.EpochOffline(trial_2s_data, trial_2s_data, fs, f_min, f_max, f_name, f_order,
                                                      {})
                        phase_2s = trial_2s.phase_center_point()
                        phase_diff_2s = (phase_2s - np.pi / 2)
                        trial_pow = epoch.Power(trial, f_min, f_max, fs, psd_method, f_res, nperseg_, noverlap_,
                                                order=order_)
                        pxx_alpha, pxx_total = trial_pow.psd()

                        # Save in dataframe
                        df.loc[row] = pd.Series(
                            {'group': group, 'sub_id': sub_id, 'eye_cond': eye_cond, 'ch': ch, 'time_point': tp,
                             'trial': trial_count, 'algorithm': algorithm, 'fs': fs, 'f_name': f_name,
                             'f_order': f_order, 'bandwidth': bw, 'rs': rs, 'rp': rp, 'window_dur': win,
                             'phase_diff': phase_diff, 'phase_diff_2s': phase_diff_2s, 'dur': trial_ep.dur,
                             'not_forecasted': trial_ep.not_forecasted, 'edge_cut': edge, 'pad': pad,
                             'psd_alpha': pxx_alpha, 'psd_total': pxx_total, 'combo': combo_no,
                             'spatial_filter': spatial_filter})
                        row += 1

                    except Exception as ex:
                        print(f'An exception was caught with {algorithm} and trial {trial_count}: {ex}')
                        df.loc[row] = pd.Series(
                            {'group': group, 'sub_id': sub_id, 'eye_cond': eye_cond, 'ch': ch, 'time_point': tp,
                             'trial': trial_count, 'algorithm': algorithm, 'fs': fs, 'f_name': f_name,
                             'f_order': f_order, 'bandwidth': bw, 'rs': rs, 'rp': rp, 'window_dur': win,
                             'phase_diff': np.nan, 'phase_diff_2s': np.nan, 'dur': np.nan, 'not_forecasted':
                                 trial_ep.not_forecasted, 'edge_cut': edge, 'pad': pad, 'psd_alpha': np.nan,
                             'psd_total': np.nan, 'combo': combo_no, 'spatial_filter': spatial_filter})
                        row += 1

            except Exception as ex:
                print(f'An exception was caught with {algorithm} and filter {f_name, f_order}: {ex}')

    except Exception as ex:
        print(f'Outer excpetion: {ex}')

    return df


# PSD methods:
nperseg_ = 1
noverlap_ = 0
order_ = 0.3
f_res = 1

df_dflt_syn = pd.DataFrame(columns=['trial', 'algorithm', 'fs', 'f_name', 'f_order', 'bandwidth', 'window_dur',
                                    'rs', 'rp', 'phase_diff', 'amplitude', 'dur', 'not_forecasted', 'pad', 'edge_cut',
                                    'psd_alpha', 'psd_total', 'combo', 'add_sinusoids'])

df_dflt_real = pd.DataFrame(columns=['group', 'eye_cond', 'sub_id', 'time_point', 'ch', 'time_point', 'trial',
                                     'algorithm', 'fs', 'f_name', 'f_order', 'bandwidth', 'window_dur', 'rs', 'rp',
                                     'phase_diff', 'phase_diff_2s', 'dur', 'not_forecasted', 'pad', 'edge_cut',
                                     'psd_alpha', 'psd_total', 'combo', 'spatial_filter'])
if __name__ == '__main__':
    # Define initial variables
    # Directories & file names
    results_dir = 'D:/phase_forecast'
    df_params = pd.read_csv(f"D:/phase_forecast/best_combo/best_combo_mean_all_real_subjects.csv", sep=',')

    df_name = 'optimized_algorithms_test'

    no_trials = 50

    # Variables to create synthetic data
    amplitudes_array = [0.5, 1, 5, 0.05, 0.01, 0.001, 0.1, 100]  # amplitude of the underlying signal, as opposite to noise
    fs_original = 5000  # fs with which we create the data (Hz)
    dur = 50  # length in seconds of generated data
    f0 = 10

    # Variables for real data
    data_dir = 'D:/Claudia/Resting_state'
    data_dir_sub = 'D:/Claudia/TMS-EEG'
    eyes_cond = ['RS_EO', 'RS_EC']
    channels = ['Oz', 'C3', 'C4', 'Fz', 'Cz']
    spatial_filter = 'car'

    # Combos to use
    no_combo, combo_init = 10, 0    # Either use the first 10 combinations if you are still in the first process of
    # validation
    # If in second step of validation, you can directly say which combination for each algorithm to be used from the
    # initial best 10
    combos_alg = {'tomasevic': 4, 'zrenner': 6, 'mansouri': 0, 'bigoni': 7}

    rnd.seed(7500)

    # Default dataframes
    algorithms = ['tomasevic', 'mansouri', 'bigoni', 'zrenner']

    # Make dictionary of parameters to use, separately for each forecasting approach
    params_dict = {
        'bigoni': {
            'f_name': [], 'f_order': [], 'bandwidth': [], 'f0': [], 'win': [], 'fs': [],
            'edge': [], 'pad': [], 'rs': [], 'rp': [], 'psd_method': []},
        'mansouri': {
            'f_name': [], 'f_order': [], 'bandwidth': [], 'f0': [], 'win': [], 'fs': [],
            'edge': [], 'pad': [], 'rs': [], 'rp': [], 'psd_method': []},
        'tomasevic': {
            'f_name': [], 'f_order': [], 'bandwidth': [], 'f0': [], 'win': [], 'fs': [],
            'edge': [], 'pad': [], 'rs': [], 'rp': [], 'psd_method': []},
        'zrenner': {
            'f_name': [], 'f_order': [], 'bandwidth': [], 'f0': [], 'win': [], 'fs': [],
            'edge': [], 'pad': [], 'rs': [], 'rp': [], 'psd_method': []}
    }

    # Add values to params dict
    for algorithm in algorithms:
        df_data = df_params[df_params['algorithm'] == algorithm]
        for idx_line, line in enumerate(df_data.iterrows()):
            if idx_line >= no_combo + combo_init:
                break
            if idx_line < combo_init:
                continue
            try:
                params_dict[algorithm]['fs'].append(int(line[1]['fs']))
                params_dict[algorithm]['win'].append(line[1]['window_dur'])
                params_dict[algorithm]['f_name'].append(line[1]['f_name'])
                params_dict[algorithm]['f_order'].append(int(line[1]['f_order']))
                params_dict[algorithm]['bandwidth'].append(int(line[1]['bandwidth']))
                params_dict[algorithm]['edge'].append(float(line[1]['edge_cut']))
                params_dict[algorithm]['pad'].append(int(line[1]['pad']))
                params_dict[algorithm]['rs'].append(float(line[1]['rs']))
                params_dict[algorithm]['rp'].append(float(line[1]['rp']))
                params_dict[algorithm]['psd_method'].append('welch')
                params_dict[algorithm]['f0'].append(f0)
            except Exception as ex:
                continue

    ####################################################################################################################
    # # Synthetic data
    # start_time = time.time()
    # for add_sinusoids in [0, 1]:    # New option from grid_search - also add other sinusoidal components
    #     for algorithm in algorithms:
    #         for combo in range(combo_init, no_combo + combo_init):
    #             for amp in amplitudes_array:
    #                 params_list = [algorithm, amp] + [params_dict[algorithm][ii][combo - combo_init] for ii in sorted(
    #                     params_dict[algorithm])]
    #                 df_ = try_get_phase_psd(params_list, combo, add_sinusoids)
    #                 df_dflt_syn = pd.concat([df_dflt_syn, df_], ignore_index=True)
    # stop_time = time.time()
    # print(f'time to compute phase forecast and psd + concatenate data on synthetic data: {stop_time - start_time} s')
    #
    # start_time = time.time()
    # if not os.path.exists(os.path.join(results_dir, 'optimized_algorithms', 'synthetic')):
    #     os.makedirs(os.path.join(results_dir, 'optimized_algorithms', 'synthetic'))
    # df_dflt_syn.to_csv(os.path.join(results_dir, 'optimized_algorithms', 'synthetic', f'synthetic_{df_name}.csv'))
    # print(f'Time to save all together to csv: {time.time() - start_time}')

    ####################################################################################################################
    # Real data
    print('Real data')
    # Start to loop over all possible combinations of parameters
    params_dict = {
        'bigoni': {
            'f_name': [], 'f_order': [], 'bandwidth': [], 'f0': [], 'win': [], 'fs': [],
            'edge': [], 'pad': [], 'rs': [], 'rp': [], 'psd_method': []},
        'mansouri': {
            'f_name': [], 'f_order': [], 'bandwidth': [], 'f0': [], 'win': [], 'fs': [],
            'edge': [], 'pad': [], 'rs': [], 'rp': [], 'psd_method': []},
        'tomasevic': {
            'f_name': [], 'f_order': [], 'bandwidth': [], 'f0': [], 'win': [], 'fs': [],
            'edge': [], 'pad': [], 'rs': [], 'rp': [], 'psd_method': []},
        'zrenner': {
            'f_name': [], 'f_order': [], 'bandwidth': [], 'f0': [], 'win': [], 'fs': [],
            'edge': [], 'pad': [], 'rs': [], 'rp': [], 'psd_method': []}
    }

    start_time = time.time()
    for group in ['Healthy', 'Stroke']:
        no_subject = -1
        for sub_id in os.listdir(os.path.join(data_dir, group)):
            sub_no = sub_id.split('_')[-1]
            df_tot = df_dflt_real.copy()
            params_dict = copy.deepcopy(params_dict)
            # Add values to params dict
            for algorithm in algorithms:
                print(algorithm)
                df_data = df_params[df_params['algorithm'] == algorithm]
                for idx_line, line in enumerate(df_data.iterrows()):
                    if idx_line >= no_combo + combo_init:
                        break
                    if idx_line < combo_init:
                        continue
                    try:
                        params_dict[algorithm]['fs'].append(int(line[1]['fs']))
                        params_dict[algorithm]['win'].append(line[1]['window_dur'])
                        params_dict[algorithm]['f_name'].append(line[1]['f_name'])
                        params_dict[algorithm]['f_order'].append(int(line[1]['f_order']))
                        params_dict[algorithm]['bandwidth'].append(int(line[1]['bandwidth']))
                        params_dict[algorithm]['edge'].append(float(line[1]['edge_cut']))
                        params_dict[algorithm]['pad'].append(int(line[1]['pad']))
                        params_dict[algorithm]['rs'].append(float(line[1]['rs']))
                        params_dict[algorithm]['rp'].append(float(line[1]['rp']))
                        params_dict[algorithm]['psd_method'].append('welch')
                        params_dict[algorithm]['f0'].append(f0)
                    except Exception as ex:
                        continue
            for eye_cond in eyes_cond:
                if group == 'Stroke':
                    t_points = [i for i in os.listdir(os.path.join(data_dir_sub, group, sub_id, 'EEG')) if
                                os.path.isdir(os.path.join(data_dir_sub, group, sub_id, 'EEG', i))]
                else:
                    t_points = ['']
                for tp in t_points:
                    try:
                        if group == 'Healthy':
                            filename = f'{eye_cond}_{sub_no}.set'
                            eeg_data = mne.io.read_raw_eeglab(os.path.join(data_dir, group, eye_cond, filename),
                                                              preload=True)
                        elif group == 'Stroke':
                            filename = f'{eye_cond}1_{sub_no}_{tp}.set'
                            eeg_data = mne.io.read_raw_eeglab(os.path.join(data_dir, group, sub_no, tp, filename),
                                                              preload=True)
                        for algorithm in algorithms:
                            # combo = combos_alg[algorithm]
                            for combo in range(0, 10):  # If the best combination was already chosen, use the previous
                                # line, otherwise loop through all of them
                                df = try_get_phase_psd_real(group, sub_id, tp, eye_cond, algorithm, combo, 'None',
                                                            eeg_data)

                                df_tot = pd.concat([df_tot, df])
                    except FileNotFoundError:
                        print(f'Not found: {os.path.join(data_dir, group, sub_no, tp, filename)}')
            start_time = time.time()
            if not os.path.exists(os.path.join(results_dir, 'optimized_algorithms', 'real', 'trained_subjects')):
                os.mkdir(os.path.join(results_dir, 'optimized_algorithms', 'real', 'trained_subjects'))
            df_tot.to_csv(os.path.join(results_dir, 'optimized_algorithms', 'real',
                                       'trained_subjects', f'{df_name}_{group}_{sub_id}.csv'))
            print(f'Time to save all together to csv: {time.time() - start_time}')
