# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:31:25 2023

@author: mahjaf
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet, tfr_multitaper
%matplotlib qt

data_path = "P:\\3013102.01\\Data\\"
participant_session = ['NL_DNDRS_0004__ses-2']
LRLR_events = {'NL_DNDRS_0004__ses-2': [7197, 7226, 7276, 7457, 7482, 7492]}

for idx, c_subj in enumerate(participant_session):
    
    participant_number = c_subj.split('__')[0]
    session_number = c_subj.split('__')[1]
    
    # Reading EEG data
    path_EEG_L = data_path + participant_number +'\\' + session_number + '\\eeg\\EEG L.edf'
    path_EEG_R = data_path + participant_number +'\\' + session_number + '\\eeg\\EEG R.edf'

    # Load the EDF file
    raw_EEG_L = mne.io.read_raw_edf(path_EEG_L, preload=True)
    raw_EEG_R = mne.io.read_raw_edf(path_EEG_R, preload=True)
    
    fs = int(raw_EEG_L.info['sfreq'])
    
    # Set up the events for creating epochs
    num_rows = len(LRLR_events[c_subj])
    events = np.zeros((num_rows, 3), dtype = 'int')
    
    # Fill the first column with the values from the Python list
    events[:, 0] = [element * fs for element in LRLR_events[c_subj]] 
    events[:, 1] = 0
    events[:, 2] = 1

    # Create epochs for seconds 5-10 (event ID 1) and seconds 15-20 (event ID 2)
    epochs_EEG_L = mne.Epochs(raw_EEG_L, events, event_id=1, tmin=0, tmax=20, baseline=None)
    epochs_EEG_R = mne.Epochs(raw_EEG_R, events, event_id=1, tmin=0, tmax=20, baseline=None)
    
    epochs_EEG_L_get_data = np.transpose(np.squeeze(epochs_EEG_L.get_data()))
    epochs_EEG_R_get_data = np.transpose(np.squeeze(epochs_EEG_R.get_data()))
    
    for item in np.arange(len(LRLR_events[c_subj])):
        print(item)
        plt.figure()
        plt.plot(epochs_EEG_L_get_data[:, item])
        plt.plot(epochs_EEG_R_get_data[:, item])


epochs.plot()

epochs.plot_psd(fmin=.1, fmax=15, tmin=0, tmax=5, average=True)

freqs = np.arange(5., 30., 1.)

# You can trade time resolution or frequency resolution or both
# in order to get a reduction in variance

n_cycles = freqs / 2.
power = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
power.plot([0], baseline=(0., 0.1), mode='mean', vmin=, vmax=3.,
           title='Sim: Using Morlet wavelet')

## 
n_cycles = freqs *5
time_bandwidth = 2.0  # Least possible frequency-smoothing (1 taper)
power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                       time_bandwidth=time_bandwidth, return_itc=False)
power.plot()
# Plot results. Baseline correct based on first 100 ms.
power.plot([0], baseline=(5., 5.1), mode='mean', vmin=-1., vmax=3.,
           title='Sim: Least smoothing, most variance')