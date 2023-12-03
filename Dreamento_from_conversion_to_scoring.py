# -*- coding: utf-8 -*-
"""
Dreamento_from_conversion_to_scoring: This file is meant to convert several 'raw' .hyp ZMax recordings
into the corresponding .edf files, and automatically score the data.

=== Inputs ===
1. path_to_HDRecorder: path to HDRecorder.exe (specify the folder name only, default, C:/Program Files (x86)/Hypnodyne/ZMax)
2. source_folder: path to the folder, where all raw ".hyp" files are located. 
                  The folder may contain other folders including additional .hyp files.
3. Dreamento_main_folder_path: path to the main folder of Dreamento. Thid folder 
                               should contain DreamentoScorer folder inside.
=== Outputs ===
Converted edf files as well as the scoring results in destination folders!

Reference: Jafarzadeh Esfahani, M., Daraie, A. H., Zerr, P., Weber, F. D., & Dresler, M. (2023).
           Dreamento: An open-source dream engineering toolbox for sleep EEG wearables. 
           SoftwareX, 24, 101595. https://doi.org/10.1016/j.softx.2023.101595
"""
import os
import shutil
import numpy as np
import subprocess
import joblib
import pickle
import mne
import yasa
from scipy.signal import butter, filtfilt
from lspopt import spectrogram_lspopt
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
import time
import pandas as pd
import json
%matplotlib qt
# ============================= USER INPUT ==================================
# define path to HDRecorder.exe
path_to_HDRecorder = 'C:\\Program Files (x86)\\Hypnodyne\\ZMax\\'

# define folder including raw .hyp files to be converted 
source_folder = 'path\\to\\folders\\including\\hyp\\files\\'

Dreamento_main_folder_path = 'path\\to\\Dreamento\\folder\\'
    
DreamentoScorerModel ="PooledData_Full_20percent_valid.sav" 
# ============================= ///// \\\\\ ==================================

#%% Required functions 
os.chdir(path_to_HDRecorder)

# Finding .hyp files within the tree directory of the provided folder
def find_files_with_extension(folder_path, extension):
    # Initialize an empty list to store matching files
    matching_files = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has the desired extension
            if file.endswith(extension):
                # Append the full path of the matching file to the list
                matching_files.append(os.path.join(root, file))

    return matching_files

def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
    """
    Butterworth bandpass filter
    
    :param self: access the attributes and methods of the class
    :param data: data to be filtered
    :param lowcut: the lowcut of the filter
    :param highcut: the highcut of the filter
    :param fs: sampling frequency
    :param order: filter order
    
    :returns: y
    
    """
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band')
    #print(b,a)
    y = filtfilt(b, a, data)
    return y

def post_scoring_N1(print_replaced_epochs = True, replace_values_in_prediciton = True):
    
    y_pred_post_processed = y_pred
    first_N2_index = [i for i,j in enumerate(y_pred) if (y_pred[i]==2)][0]
    scores_before_first_N2 = y_pred[:first_N2_index]
    REM_before_first_N2 = [i for i,j in enumerate(scores_before_first_N2) if (scores_before_first_N2[i]==4)]
    
    for idx_actual_N1 in REM_before_first_N2:
        
        y_pred_post_processed[idx_actual_N1] = 1
        
        if print_replaced_epochs == True:
            
            print(f'the REM detected in epoch {idx_actual_N1} has been replaced by N1 ...')
            
        if replace_values_in_prediciton ==True:
                           
            y_pred[idx_actual_N1] = 1      
    return y_pred

def retrieve_sleep_statistics(hypno, sf_hyp = 1 / 30, sleep_stages = [0, 1, 2, 3, 5],\
                              show_sleep_stats = True):
    """Compute standard sleep statistics from an hypnogram.
    .. versionadded:: 0.1.9
    Parameters
    source: https://github.com/raphaelvallat/yasa/blob/master/yasa/sleepstats.py
    ----------
    hypno : array_like
        Hypnogram, assumed to be already cropped to time in bed (TIB,
        also referred to as Total Recording Time,
        i.e. "lights out" to "lights on").
        .. note::
            The default hypnogram format in YASA is a 1D integer
            vector where:
            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    sf_hyp : float
        The sampling frequency of the hypnogram. Should be 1/30 if there is one
        value per 30-seconds, 1/20 if there is one value per 20-seconds,
        1 if there is one value per second, and so on.
    Returns
    -------
    stats : dict
        Sleep statistics (expressed in minutes)
    Notes
    -----
    All values except SE, SME and percentages of each stage are expressed in
    minutes. YASA follows the AASM guidelines to calculate these parameters:
    * Time in Bed (TIB): total duration of the hypnogram.
    * Sleep Period Time (SPT): duration from first to last period of sleep.
    * Wake After Sleep Onset (WASO): duration of wake periods within SPT.
    * Total Sleep Time (TST): SPT - WASO.
    * Sleep Efficiency (SE): TST / TIB * 100 (%).
    * Sleep Maintenance Efficiency (SME): TST / SPT * 100 (%).
    * W, N1, N2, N3 and REM: sleep stages duration. NREM = N1 + N2 + N3.
    * % (W, ... REM): sleep stages duration expressed in percentages of TST.
    * Latencies: latencies of sleep stages from the beginning of the record.
    * Sleep Onset Latency (SOL): Latency to first epoch of any sleep.
    References
    ----------
    * Iber, C. (2007). The AASM manual for the scoring of sleep and
      associated events: rules, terminology and technical specifications.
      American Academy of Sleep Medicine.
    * Silber, M. H., Ancoli-Israel, S., Bonnet, M. H., Chokroverty, S.,
      Grigg-Damberger, M. M., Hirshkowitz, M., Kapen, S., Keenan, S. A.,
      Kryger, M. H., Penzel, T., Pressman, M. R., & Iber, C. (2007).
      `The visual scoring of sleep in adults
      <https://www.ncbi.nlm.nih.gov/pubmed/17557422>`_. Journal of Clinical
      Sleep Medicine: JCSM: Official Publication of the American Academy of
      Sleep Medicine, 3(2), 121–131.

    """
    stats = {}
    hypno = np.asarray(hypno)
    assert hypno.ndim == 1, 'hypno must have only one dimension.'
    assert hypno.size > 1, 'hypno must have at least two elements.'

    # TIB, first and last sleep
    stats['TIB'] = len(hypno)
    first_sleep = np.where(hypno > sleep_stages[0])[0][0]
    last_sleep = np.where(hypno > sleep_stages[0])[0][-1]

    # Crop to SPT
    hypno_s = hypno[first_sleep:(last_sleep + 1)]
    stats['SPT'] = hypno_s.size
    stats['WASO'] = hypno_s[hypno_s == sleep_stages[0]].size
    stats['TST'] = stats['SPT'] - stats['WASO']

    # Duration of each sleep stages
    stats['N1'] = hypno[hypno == sleep_stages[1]].size
    stats['N2'] = hypno[hypno == sleep_stages[2]].size
    stats['N3'] = hypno[hypno == sleep_stages[3]].size
    stats['REM'] = hypno[hypno == sleep_stages[4]].size
    stats['NREM'] = stats['N1'] + stats['N2'] + stats['N3']

    # Sleep stage latencies
    stats['SOL'] = first_sleep
    stats['Lat_N1'] = np.where(hypno == sleep_stages[1])[0].min() if sleep_stages[1] in hypno else np.nan
    stats['Lat_N2'] = np.where(hypno == sleep_stages[2])[0].min() if sleep_stages[2] in hypno else np.nan
    stats['Lat_N3'] = np.where(hypno == sleep_stages[3])[0].min() if sleep_stages[3] in hypno else np.nan
    stats['Lat_REM'] = np.where(hypno == sleep_stages[4])[0].min() if sleep_stages[4] in hypno else np.nan

    # Convert to minutes
    for key, value in stats.items():
        stats[key] = value / (60 * sf_hyp)

    # Percentage
    stats['%N1']   = "{:.2f}".format(100 * stats['N1'] / stats['TST'])
    stats['%N2']   = "{:.2f}".format(100 * stats['N2'] / stats['TST'])
    stats['%N3']   = "{:.2f}".format(100 * stats['N3'] / stats['TST'])
    stats['%REM']  = "{:.2f}".format(100 * stats['REM'] / stats['TST'])
    stats['%NREM'] = "{:.2f}".format(100 * stats['NREM'] / stats['TST'])
    stats['SE']    = "{:.2f}".format(100 * stats['TST'] / stats['TIB'])
    stats['SME']   = "{:.2f}".format(100 * stats['TST'] / stats['SPT'])
    
    stats = stats
      
    return stats
# %% DreamentoConverter
# Replace 'your_folder_path' with the actual path of your folder
folder_path = source_folder
extension = '.hyp'

# Call the function to find files with the specified extension
filenames           = find_files_with_extension(folder_path, extension)
destination_folders = []

# Definre destination folders
for file_path in filenames:
    
    hyp_files = os.path.basename(file_path)
    hyp_without_extension = hyp_files.split('.hyp')[0]
    hyp_dest_folder = file_path.split(hyp_files)[0]
    destination_folders.append(hyp_dest_folder + 'converted\\' + hyp_without_extension + '\\')

for conv in np.arange(len(filenames)):
    
    # Copy the .hyp file to HDRecoder folder for conversion
    src_path = filenames[conv]
    current_file = filenames[conv].split('\\')[-1]
    dst_path = path_to_HDRecorder + current_file
    shutil.copy(src_path, dst_path)
                    
    # Create a batch file to run conversion syntax
    myBat = open(r'DreamentoConverter.bat','w+')
    myBat.write('HDRecorder.exe -conv '+ current_file)
    myBat.close()
    
    # run the created .bat --> conversion
    print(f'Converting the file {conv+1}/{len(filenames)}...please be patient...')
    subprocess.call(path_to_HDRecorder + 'DreamentoConverter.bat')
    print(f'{conv+1}/{len(filenames)} files have been successfully converted')
    print(f'file {src_path} converted to path {destination_folders[conv]}')
    # Copy generated folder to the desired path
    shutil.copytree(path_to_HDRecorder + 'SDConvert\\', destination_folders[conv])
    
    # Remove the .hyp and .bat files from HDRecorder folder
    os.remove(path_to_HDRecorder + 'DreamentoConverter.bat')
    os.remove(dst_path)

print('All files have been successfully converted!')

#%% DreamentoScorer
print('Initiating requirements for autoscoring ...')

path_to_DreamentoScorer = Dreamento_main_folder_path + '\\DreamentoScorer\\'
model_path = path_to_DreamentoScorer + DreamentoScorerModel
standard_scaler_path = path_to_DreamentoScorer + "StandardScaler_PooledDataset_Full_20percent_valid.sav"
feat_selection_path = path_to_DreamentoScorer + "Selected_Features_BoturaAfterTD=3_Bidirectional_Donders2022_19-04-2023.pickle"
apply_post_scoring_N1_correction = True

try:
    # Change the current working Directory    
    os.chdir(path_to_DreamentoScorer)
    print("Loading DreamentoScorer class ... ")
    
except OSError:
    print("Can't change the Current Working Directory ... Dreamento main folder cannot be found")  

print(f'current path is {path_to_DreamentoScorer}')
from entropy.entropy import spectral_entropy
from DreamentoScorer import DreamentoScorer

fs=256
DS = DreamentoScorer(filename='', channel='', fs = fs, T = 30)
f_min = fmin = .3 #Hz
f_max = fmax =  30 #Hz

folders_to_be_autoscored = destination_folders
print(f'folders to be autoscored are detected as follows: {folders_to_be_autoscored}')

counter_scoring = 0
all_stats = dict()
for folder_autoscoring in folders_to_be_autoscored:
    print(folder_autoscoring)
    counter_scoring = counter_scoring + 1
    print(f'autoscoring folder: {folder_autoscoring} [{counter_scoring} / {len(folders_to_be_autoscored)}]')
    
    #sanity check:
    if folder_autoscoring[-1] != '/' or folder_autoscoring[-1] != '\\':
        folder_autoscoring = folder_autoscoring + '\\'
        
    data_L = mne.io.read_raw_edf(folder_autoscoring + 'EEG L.edf')
    raw_data_L = data_L.get_data()
    sigHDRecorder = np.ravel(raw_data_L)
    
    data_r = mne.io.read_raw_edf(folder_autoscoring + 'EEG R.edf')
    raw_data_r = data_r.get_data()
    sigHDRecorder_r = np.ravel(raw_data_r)
   
    # There has to be filtering in her anyways 
    print('Filtering ...')
    # if already filtered, take it, otherwise filter first
    EEG_L_filtered     = butter_bandpass_filter(data = sigHDRecorder,\
                                                     lowcut=.3, highcut=30, fs = 256, order = 2)
    EEG_R_filtered     = butter_bandpass_filter(data = sigHDRecorder_r,\
                                                         lowcut=.3, highcut=30, fs = 256, order = 2)
    T = 30 #secs
    len_epoch   = fs * T
    start_epoch = 0
    n_channels  = 1
    print(f'Data shape : {np.shape(EEG_R_filtered)}')
    print('Truncating the last epoch tails ...')
    EEG_L_filtered = EEG_L_filtered[0:EEG_L_filtered.shape[0] - EEG_L_filtered.shape[0]%len_epoch]
    EEG_R_filtered = EEG_R_filtered[0:EEG_R_filtered.shape[0] - EEG_R_filtered.shape[0]%len_epoch]
    
    print('Segmenting data into 30 s epochs ...')
    EEG_L_filtered_epoched = np.reshape(EEG_L_filtered,
                              (n_channels, len_epoch,
                               int(EEG_L_filtered.shape[0]/len_epoch)), order='F' )
    
    EEG_R_filtered_epoched = np.reshape(EEG_R_filtered,
                              (n_channels, len_epoch,
                               int(EEG_R_filtered.shape[0]/len_epoch)), order='F' )
    
    # Ensure equalituy of length for arrays:
    assert np.shape(EEG_L_filtered_epoched)[1] == np.shape(EEG_R_filtered_epoched)[1], 'Different epoch numbers!'
    
    print(f'SHAPE OF EEG_L_filtered_epoched {np.shape(EEG_L_filtered_epoched)}')
    print(f'shape after epoching: {np.shape(EEG_L_filtered_epoched)}')
    print('Extracting features for DreamentoScorer ...')
    for k in np.arange(np.shape(EEG_L_filtered_epoched)[0]):
        t_st = time.time()
        print('Extracting features from channel 1 ...')
        feat_L = DS.FeatureExtraction_per_subject(Input_data = EEG_L_filtered_epoched[k,:,:], fs = fs)
        print('Extracting features from channel 2 ...')
        feat_R = DS.FeatureExtraction_per_subject(Input_data = EEG_R_filtered_epoched[k,:,:], fs = fs)
        # Concatenate features
        print(f'concatenating features of size {np.shape(feat_L)} and {np.shape(feat_R)}')
        Feat_all_channels = np.column_stack((feat_L,feat_R))
        t_end = time.time()
        print(f'Features extracted in {t_end - t_st} s')
    # Scoring
    X_test  = Feat_all_channels
    X_test  = DS.replace_NaN_with_mean(X_test)

    # Replace any probable inf
    X_test  = DS.replace_inf_with_mean(X_test)
    
    # Z-score features
    print('Importing DreamentoScorer model ...')
    sc_fname = standard_scaler_path#'StandardScaler_TrainedonQS_1st_iter_To_transform_ZmaxDonders'
    sc = joblib.load(sc_fname)
    X_test = sc.transform(X_test)

    # Add time dependence to the data classification
    td = 3 # epochs of memory
    print('Adding time dependency ...')
    X_test_td  = DS.add_time_dependence_bidirectional(X_test,  n_time_dependence=td,\
                                                     padding_type = 'sequential')

    # Load selected features
    print('loading results of feature selection from the trained data')
    path_selected_feats = feat_selection_path
    with open(path_selected_feats, "rb") as f: 
        selected_feats_ind = pickle.load(f)
        
    X_test  = X_test_td[:, selected_feats_ind]

    # Load DreamentoScorer
    print('Loading scoring model ...')
    model_dir = model_path
    print(f'DreamentoScorer model retrieved from: {model_dir}')
    DreamentoScorer = joblib.load(model_dir)

    y_pred = DreamentoScorer.predict(X_test)
    y_pred_proba = DreamentoScorer.predict_proba(X_test)
    y_pred_proba = pd.DataFrame(y_pred_proba, columns = ['Wake', 'N1', 'N2', 'SWS', 'REM'])
    
    if apply_post_scoring_N1_correction == True: 
        # Post-scoring: Replacing the REM deetcted before the first N2 with N1
        y_pred = post_scoring_N1(print_replaced_epochs = True, replace_values_in_prediciton = True)

    stats = retrieve_sleep_statistics(hypno = y_pred, sf_hyp = 1 / 30,\
                                                 sleep_stages = [0, 1, 2, 3, 4])
        
    data1 = EEG_L_filtered
    data2 = EEG_R_filtered
    
    # Safety checks
    sf = 256
    win_sec = 30
    trimperc=2.5
    cmap="RdBu_r"
    vmin=None
    vmax=None
    assert isinstance(data1, np.ndarray), "data1 must be a 1D NumPy array."
    assert isinstance(data2, np.ndarray), "data1 must be a 1D NumPy array."
    assert isinstance(sf, (int, float)), "sf must be int or float."
    assert data1.ndim == 1, "data1 must be a 1D (single-channel) NumPy array."
    assert data2.ndim == 1, "data1 must be a 1D (single-channel) NumPy array."
    assert isinstance(win_sec, (int, float)), "win_sec must be int or float."
    assert isinstance(fmin, (int, float)), "fmin must be int or float."
    assert isinstance(fmax, (int, float)), "fmax must be int or float."
    assert fmin < fmax, "fmin must be strictly inferior to fmax."
    assert fmax < sf / 2, "fmax must be less than Nyquist (sf / 2)."
    
    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sf)
    assert data1.size > 2 * nperseg, "Data length must be at least 2 * win_sec."
    assert data2.size > 2 * nperseg, "Data length must be at least 2 * win_sec."
    
    f, t, Sxx = spectrogram_lspopt(data1, sf, nperseg=nperseg, noverlap=0)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz
    
    f2, t2, Sxx2 = spectrogram_lspopt(data2, sf, nperseg=nperseg, noverlap=0)
    Sxx2 = 10 * np.log10(Sxx2)  # Convert uV^2 / Hz --> dB / Hz
    
    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]
    t /= 3600  # Convert t to hours
    
    good_freqs2 = np.logical_and(f2>= fmin, f2 <= fmax)
    Sxx2 = Sxx2[good_freqs2, :]
    f2 = f2[good_freqs2]
    t2 /= 3600  # Convert t to hours
    
    # Normalization
    if vmin is None:
        vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        
    
    fig,AX = plt.subplots(nrows=4, figsize=(12, 6), gridspec_kw={'height_ratios': [3,3,1,1]})
    print('initiating the plot')
    # Increase font size while preserving original
    #plt.legend(loc = 'right', prop={'size': 6})
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 9})
    ax0 = plt.subplot(4,1,1)
    ax1 = plt.subplot(4,1,2)
    ax2 = plt.subplot(4,1,3)
    ax3 = plt.subplot(4,1,4)
    
    print('Subplots assigned')
    ax0.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True,
                       shading="auto")
    ax1.pcolormesh(t2, f2, Sxx2, norm=norm, cmap=cmap, antialiased=True,
                       shading="auto")
    print('plots completed')
    stages = y_pred
    #stages = np.row_stack((stages, stages[-1]))
    x      = np.arange(len(stages))
    stage_autoscoring = stages
     # Change the order of classes: REM and wake on top
    x = []
    y = []
    for i in np.arange(len(stages)):
        
        s = stages[i]
        if s== 0 :  p = -0
        if s== 4 :  p = -1
        if s== 1 :  p = -2
        if s== 2 :  p = -3
        if s== 3 :  p = -4
        if i!=0:
            
            y.append(p)
            x.append(i-1)   
    y.append(p)
    x.append(i)
    ax2.step(x, y, where='post', color = 'black', linewidth = 2)
    rem = [i for i,j in enumerate(y_pred) if (y_pred[i]==4)]
    for i in np.arange(len(rem)) -1:
        ax2.plot([rem[i]-1, rem[i]], [-1,-1] , linewidth = 2, color = 'red')

    #ax_autoscoring.scatter(rem, -np.ones(len(rem)), color = 'red')
# =============================================================================
#     ax2.set_yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'], fontsize = 8)
# =============================================================================
    ax2.set_yticks([0, -1, -2, -3, -4])
    ax2.set_yticklabels(['Wake', 'REM', 'N1', 'N2', 'SWS'])
    
    # Set the font size for y-tick labels
    ax2.tick_params(axis='y', labelsize=8)

    ax2.set_xlim([np.min(x), np.max(x)])
    ax3.set_xlim([np.min(x), np.max(x)])
    ax0.set_title(folder_autoscoring + 'EEG L')
    ax1.set_title(folder_autoscoring + 'EEG R')
    
    y_pred_proba.plot(ax = ax3, kind="area", alpha=0.8, stacked=True, lw=0, color = ['black', 'olive', 'deepskyblue', 'purple', 'red'])
    #ax3.legend().remove()
    #plt.tight_layout()
    # Save results?
    
                    
    save_path_autoscoring = folder_autoscoring + 'DreamentoScorer.txt'
    
    if os.path.exists(save_path_autoscoring):
        os.remove(save_path_autoscoring)
        
    saving_dir = save_path_autoscoring
    
    a_file = open(saving_dir, "w")
    a_file.write('=================== Dreamento: an open-source dream engineering toolbox! ===================\nhttps://github.com/dreamento/dreamento')
    a_file.write('\nThis file has been autoscored by DreamentoScorer v.01.00! \nSleep stages: Wake:0, N1:1, N2:2, SWS:3, REM:4.\n')
    a_file.write('============================================================================================\n')
    
    for row in stage_autoscoring[:,np.newaxis]:
        np.savetxt(a_file, row, fmt='%1.0f')
    a_file.close()
    
    # Save sleep metrics
    save_path_stats = folder_autoscoring + 'DreamentoScorer_sleep_stats.json'
    
    if os.path.exists(save_path_stats):
        os.remove(save_path_stats)
        
    with open(save_path_stats, 'w') as convert_file:
        convert_file.write(json.dumps(stats))
        
    all_stats[folder_autoscoring] = stats
    
    #save_figure
    save_path_plots = folder_autoscoring + 'Dreamento_TFR_autoscoring.png'
    
    if os.path.exists(save_path_plots):
        os.remove(save_path_plots)
        
    plt.savefig(save_path_plots,dpi = 300)  
    
# Store xlsx sleep stats
df = pd.DataFrame(data=all_stats)
df = (df.T)
path_to_save_all_stats = source_folder + '\\Dreamento_all_sleep_stats.xlsx'
print(f'stroing all stats in: {path_to_save_all_stats}')

# Remove if there is already a file with the same name ...
if os.path.exists(path_to_save_all_stats):
    os.remove(path_to_save_all_stats)
df.to_excel(path_to_save_all_stats)