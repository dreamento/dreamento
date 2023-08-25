import mne
import numpy as np
import matplotlib.pyplot as plt
import yasa
from mne.time_frequency import tfr_morlet, tfr_multitaper
%matplotlib qt

data_path = "P:\\3013102.01\\Data\\"
participant_session = ['NL_DNDRS_0004__ses-2']

for idx, c_subj in enumerate(participant_session):
    
    participant_number = c_subj.split('__')[0]
    session_number = c_subj.split('__')[1]
    
    # Reading EEG data
    path_EEG_L = data_path + participant_number +'\\' + session_number + '\\eeg\\EEG L.edf'
    path_EEG_R = data_path + participant_number +'\\' + session_number + '\\eeg\\EEG R.edf'

    # Load the EDF file
    raw_EEG_L = mne.io.read_raw_edf(path_EEG_L, preload=True)
    raw_EEG_R = mne.io.read_raw_edf(path_EEG_R, preload=True)
    
    raw_EEG_L.filter(l_freq=.3, h_freq=30)
    raw_EEG_R.filter(l_freq=.3, h_freq=30)

    raw_EEG_L_get_data = np.ravel(raw_EEG_L.get_data(units="uV"))
    raw_EEG_R_get_data = np.ravel(raw_EEG_R.get_data(units="uV"))
    
    loc = raw_EEG_L_get_data[7190*256:]
    roc = raw_EEG_R_get_data[7190*256:]
    
    plt.plot(loc)
    plt.plot(roc)

    REM_events = yasa.rem_detect(loc = loc, roc = roc, sf=256, hypno=None, include=4, amplitude=(30, 325), duration=(0.1, 1.2), freq_rem=(0.2, 8), remove_outliers=True, verbose=False)
    REM_events.summary()
    REM_events.plot_average(time_before=1, time_after=1);

        # Let's get a boolean mask of the REMs in data
    mask = REM_events.get_mask()
    
    loc_highlight = loc * mask[0, :]
    roc_highlight = roc * mask[1, :]

    loc_highlight[loc_highlight == 0] = np.nan
    roc_highlight[roc_highlight == 0] = np.nan
    
    plt.figure(figsize=(16, 4.5))
    plt.plot(loc, 'slategrey', label='LOC')
    plt.plot(roc, 'grey', label='ROC')
    plt.plot( loc_highlight, 'indianred')
    plt.plot( roc_highlight, 'indianred')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (uV)')
    plt.title('REM sleep EOG data')
    plt.legend()
