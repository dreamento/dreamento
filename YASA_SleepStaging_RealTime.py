# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:48:52 2023

@author: mahjaf
"""

import yasa
import mne
import numpy as np
%matplotlib qt
path_to_EEG_L = "P:\\3013097.06\\Data\\s37\\n9\\zmax\\EEG L.edf"
path_to_EEG_R = "P:\\3013097.06\\Data\\s37\\n9\\zmax\\EEG R.edf"



EEG_L_data = mne.io.read_raw_edf(path_to_EEG_L, preload = True)
EEG_R_data = mne.io.read_raw_edf(path_to_EEG_R, preload = True)

EEG_L_data_get = np.ravel(EEG_L_data.get_data())
EEG_R_data_get = np.ravel(EEG_R_data.get_data())

EEG_data = EEG_L_data_get - EEG_R_data_get

sls = yasa.SleepStaging(EEG_L_data, eeg_name = 'EEG L')

hypno_pred = sls.predict()
hypno_pred = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc

up_sampled_hypno = yasa.hypno_upsample_to_data(hypno= hypno_pred, sf_hypno = 1/30, data=EEG_L_data_get, sf_data=256, verbose=True)

yasa.plot_spectrogram(data = EEG_L_data_get, sf = 256, hypno = up_sampled_hypno)

## Simulating data stream after every epoch

epoch_by_epoch_staging = []

# =============================================================================
# for curr_epoch in np.arange(np.floor(len(EEG_L_data_get)/30/256)):
#     
#     curr_epoch = int(curr_epoch)
#     print('Analyzing epoch {curr_epoch}')
# =============================================================================
    
sls = yasa.SleepStaging(EEG_L_data_get, eeg_name = 'EEG L', ignore_raw = True)

hypno_pred = sls.predict()
hypno_pred = yasa.hypno_str_to_int(hypno_pred)  # Convert "W" to 0, "N1" to 1, etc

up_sampled_hypno = yasa.hypno_upsample_to_data(hypno= hypno_pred, sf_hypno = 1/30, data=EEG_L_data_get, sf_data=256, verbose=True)

yasa.plot_spectrogram(data = EEG_L_data_get, sf = 256, hypno = up_sampled_hypno)

    
    
    
    