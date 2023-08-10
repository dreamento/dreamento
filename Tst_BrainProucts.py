# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:03:03 2023

@author: mahjaf
"""

import numpy as np
import yasa
import mne
import matplotlib.pyplot as plt

path_to_BrainAmp_file = 'P:\\3013097.06\\BrainAmpEEG_DCM\\'
subj = 'pilot_005.vhdr'

raw = mne.io.read_raw_brainvision(path_to_BrainAmp_file + subj, preload = True)
fs = raw.info['sfreq'] 
ch_names = raw.info['ch_names'] 

all_markers_with_timestamps = []
for counter, marker in enumerate(markers.values()):
    
    all_markers.append(marker)
    
for counter, marker in enumerate(markers.keys()):
    all_timestamp_markers.append(marker)

for i in np.arange(len(all_markers)):
    all_markers_with_timestamps.append(all_markers[i]+ '__'+ all_timestamp_markers[i])
select_marker_for_sync()

# =============================================================================
# # Get the data arrays for the specified channels
# data_a, times = raw[raw.ch_names.index('C3')]
# data_b, _ = raw[raw.ch_names.index('TP10')]
# 
# # Calculate the difference between channel_a and channel_b
# channel_diff = data_a - data_b
# 
# ch_info = mne.create_info(ch_names=['a_minus_b'], sfreq=raw.info['sfreq'], ch_types=['eeg'])
# 
# raw_diff = mne.io.RawArray(data=channel_diff, info=ch_info)
# 
# raw.add_channels([raw_diff], force_update_info=True)
# 
# raw_get = raw.get_data()
# 
# =============================================================================
# Concatenate the new channel with the original raw data
desired_EEG_channels = ['C3', 'TP10']
desired_EOG_channels  = ['HEOG', 'TP10']
desired_EMG_channels  = ['EMG1', 'TP10']
desired_channels = desired_EEG_channels + desired_EOG_channels + desired_EMG_channels
ch_indices = [raw.ch_names.index(ch) for ch in desired_channels]

# Compute the derivation data
derivation_data_eeg = raw.get_data(picks=ch_indices[0]) - raw.get_data(picks=ch_indices[1])
derivation_data_eog = raw.get_data(picks=ch_indices[2]) - raw.get_data(picks=ch_indices[3])
derivation_data_emg = raw.get_data(picks=ch_indices[4]) - raw.get_data(picks=ch_indices[5])

# Create an info object for the derivation channel
derivation_info_EEG = mne.create_info(['EEG ' + desired_channels[0] + '-' + desired_channels[1]], raw.info['sfreq'], ch_types='eeg')
derivation_info_EOG = mne.create_info(['EOG ' + desired_channels[2] + '-' + desired_channels[3]], raw.info['sfreq'], ch_types='eog')
derivation_info_EMG = mne.create_info(['EMG ' + desired_channels[4] + '-' + desired_channels[5]], raw.info['sfreq'], ch_types='emg')

# Create an Evoked object for the derivation data
derivation_evoked_EEG = mne.io.RawArray(data=derivation_data_eeg, info=derivation_info_EEG)
derivation_evoked_EOG = mne.io.RawArray(data=derivation_data_eog, info=derivation_info_EOG)
derivation_evoked_EMG = mne.io.RawArray(data=derivation_data_emg, info=derivation_info_EMG)

# Adding new derivations
raw.add_channels([derivation_evoked_EEG], force_update_info=True)
raw.add_channels([derivation_evoked_EOG], force_update_info=True)
raw.add_channels([derivation_evoked_EMG], force_update_info=True)


sls = yasa.SleepStaging(raw, eeg_name=derivation_info_EEG.ch_names[0],\
                        eog_name = derivation_evoked_EOG.ch_names[0],\
                        emg_name = derivation_evoked_EMG.ch_names[0])
    
hypno_pred = sls.predict()  # Predict the sleep stages
hypno_pred = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc
yasa.plot_hypnogram(hypno_pred);  # Plot

np.savetxt('output.txt', hypno_pred, fmt='%d')
