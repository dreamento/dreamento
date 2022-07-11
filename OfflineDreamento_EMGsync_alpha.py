# -*- coding: utf-8 -*-
"""

Copyright (c) 2021-2022 Mahdad Jafarzadeh Esfahani, Amir Hossein Daraie

OfflineDreamento: The post-processing Dreamento!

"""
import tkinter as tk
from tkinter import LabelFrame, Label, Button, filedialog, messagebox,OptionMenu, StringVar, DoubleVar, PhotoImage, Entry
from tkinter import *
import mne
import numpy as np
from   numpy import loadtxt
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize, ListedColormap
from matplotlib import style
from scipy import signal
import mne
import json
import pickle
from scipy.signal import butter, filtfilt
import itertools
import matplotlib

matplotlib.use('TkAgg')

# =============================================================================
# %matplotlib qt
# =============================================================================
style.use('default')


class OfflineDreamento():
    
    def __init__(self, master):
        
        """
        Initiate the graphics, buttons, and the variables
        
        :param self: access the attributes and methods of the class
        :param master: The master window of the Tkinter

        """
        
        self.master = master
        
        master.title("OfflineDreamento: The post-processing Dreamento!")
        
        #%% Import section
        #### !!~~~~~~~~~~~~~~~~~ DEFINE INPUT DATAFRAME ~~~~~~~~~~~~~~~~~!!####
        
        self.frame_import = LabelFrame(self.master, text = "Analysis section", padx = 40, pady = 20,
                                  font = 'Calibri 18 bold')
        self.frame_import.grid(row = 0 , column = 0, padx = 5, pady = 5, columnspan = 8)        

        
        #### ==================== Help pop-up button ======================####
        
        self.popup_button = Button(self.master, text = "Help", command = self.help_pop_up_func,
                              font = 'Calibri 13 ', fg = 'white', bg = 'black')
        self.popup_button.grid(row = 1, column = 7)
        
        ###### ================== CopyRight ============================ ######
        self.label_CopyRight = Label(self.master, text = "© CopyRight (2021-22): Mahdad Jafarzadeh Esfahani, Amir Hossein Daraie",
                                  font = 'Calibri 10 italic')
        self.label_CopyRight.grid(row = 1 , column = 0, padx = 15, pady = 10)
        
        #### ==================== Import Hypnodyne data  ========================####
        # Label: Import EDF
        self.label_Hypnodyne = Label(self.frame_import, text = "Import Hypnodyne EDF file:",
                                  font = 'Calibri 13 ')
        self.label_Hypnodyne.grid(row = 0 , column = 0, padx = 15, pady = 10)
        
        # Button: Import EDF (Browse)
        self.button_Hypnodyne_browse = Button(self.frame_import, text = "Browse Hypnodyne",
                                           padx = 40, pady = 10,font = 'Calibri 12 ',
                                           command = self.load_hypnodyne_file_dialog, fg = 'blue',
                                           relief = RIDGE)
        self.button_Hypnodyne_browse.grid(row = 1, column = 0, padx = 15, pady = 10)
        
        #### ================== Import Dreamento file ====================####
        # Show a message about hypnograms
        self.label_Dreamento = Label(self.frame_import, text = "Import Dreamento output file (.txt):",
                                  font = 'Calibri 13 ')
        self.label_Dreamento.grid(row = 0 , column = 1, padx = 15, pady = 10)
        
        # Define browse button to import hypnos
        self.button_Dreamento_browse = Button(self.frame_import, text = "Browse Dreamento", 
                                           padx = 40, pady = 10, font = 'Calibri 12 ',
                                           command = self.load_Dreamento_file_dialog,fg = 'blue',
                                           relief = RIDGE)
        self.button_Dreamento_browse.grid(row = 1, column = 1, padx = 15, pady = 10)
        
        #### ================== Import markers Json file ====================####
        # Show a message about hypnograms
        self.label_marker_json = Label(self.frame_import, text = "Import marker file (.json):",
                                  font = 'Calibri 13 ')
        self.label_marker_json.grid(row = 0 , column = 2, padx = 15, pady = 10)
        
        # Define browse button to import hypnos
        self.button_marker_json_browse = Button(self.frame_import, text = "Browse markers", 
                                           padx = 40, pady = 10, font = 'Calibri 12 ',
                                           command = self.load_marker_file_dialog,fg = 'blue',
                                           relief = RIDGE)
        self.button_marker_json_browse.grid(row = 1, column = 2, padx = 15, pady = 10)
        
       
        #### ================== Import Brainvision EMG Json file ====================####
        # Show a message about hypnograms
        self.label_EMG = Label(self.frame_import, text = "Import EMG file (.vhdr):",
                                  font = 'Calibri 13 ')
        self.label_EMG.grid(row = 0 , column = 3, padx = 15, pady = 10)
        
        # Define browse button to import hypnos
        self.button_EMG_browse = Button(self.frame_import, text = "Browse EMG (.vhdr)", 
                                           padx = 40, pady = 10, font = 'Calibri 12 ',
                                           command = self.load_EMG_file_dialog,fg = 'blue',
                                           relief = RIDGE)
        self.button_EMG_browse.grid(row = 1, column = 3, padx = 15, pady = 10)
# =============================================================================
#         #### ============ Push Analyze button to assign markers =========####
#         #Label to read data and extract features
#         self.label_apply = Label(self.frame_import, text = "Assign markers",
#                                       font = 'Calibri 13 ')
#         self.label_apply.grid(row = 2 , column = 1)
# =============================================================================
        # Apply button
        self.button_apply = Button(self.frame_import, text = "Analyze!", padx = 40, pady=8,
                              font = 'Calibri 13 bold', relief = RIDGE, fg = 'blue',
                              command = self.Apply_button, state = tk.DISABLED)
        self.button_apply.grid(row = 3 , column =2, padx = 15, pady = 10)


    #%% =========================== Options for analysis =================== #%%
        #Label to read data and extract features
        self.label_analysis = Label(self.frame_import, text = "Analysis options:",
                                      font = 'Calibri 13 ')
        self.label_analysis.grid(row = 0 , column = 4)
        
    #%% init a label to give warning
    #%% Checkbox for filtering
        self.is_filtering = IntVar(value = 1)
        self.checkbox_is_filtering = Checkbutton(self.frame_import, text = "Band-pass filtering (.3-30 Hz)",
                                  font = 'Calibri 11 ', variable = self.is_filtering)
        
        self.checkbox_is_filtering.grid(row = 1, column = 4)
        
    #%% Checkbox for plotting syncing process
        self.plot_sync_output = IntVar()
        self.checkbox_plot_sync_output = Checkbutton(self.frame_import, text = "Plot alignment process",
                                  font = 'Calibri 11 ', variable = self.plot_sync_output)
        
        self.checkbox_plot_sync_output.grid(row = 2, column = 4)
        
    #%% Checkbox for plotting spectrograms with markers
        self.plot_additional_EMG = IntVar(value = 1)
        self.checkbox_plot_additional_EMG = Checkbutton(self.frame_import, text = "Plot EMG",
                                  font = 'Calibri 11 ', variable = self.plot_additional_EMG,\
                                  command=self.EMG_button_activator)
        
        self.checkbox_plot_additional_EMG.grid(row = 3, column = 4)
        
    #%% Checkbox for plotting periodogram 
        self.plot_psd = IntVar(value = 0)
        self.checkbox_plot_psd = Checkbutton(self.frame_import, text = "Plot peridogram",
                                  font = 'Calibri 11 ', variable = self.plot_psd)
        
        self.checkbox_plot_psd.grid(row = 4, column = 4)
        
    #%% EMG Y SCALE
        #Label to read data and extract features
        self.label_EMG_scale = Label(self.frame_import, text = "EMG amplitude (uV):",
                                      font = 'Calibri 13 ')
        self.label_EMG_scale.grid(row = 2 , column = 0)
        self.EMG_scale_options = ['Set desired EMG amplitude ...','100', '50', '20', '10']
        self.EMG_scale_options_val = StringVar()
        self.EMG_scale_options_val.set(self.EMG_scale_options[0])
        self.EMG_scale_option_menu = OptionMenu(self.frame_import, self.EMG_scale_options_val, *self.EMG_scale_options)
        self.EMG_scale_option_menu.config(fg = 'blue')
        self.EMG_scale_option_menu.grid(row = 3, column = 0)
    
    #%% EMG sync option
        #Label to read data and extract features
        
        self.button_sync_EMG = Button(self.frame_import, text = "Analyze! (+EMG)", padx = 40, pady=8,
                              font = 'Calibri 13 bold', relief = RIDGE, fg = 'blue',
                              command = self.EMG_sync_method_activator)
        self.button_sync_EMG.grid(row = 3 , column =1, padx = 15, pady = 10)
        
# =============================================================================
#         self.label_sync_EMG_option = Label(self.frame_import, text = "EMG sync method?",
#                                       font = 'Calibri 13 ')
#         self.label_sync_EMG_option.grid(row = 5 , column = 0)
#         self.sync_EMG_option = ['no sync', 'manual']
#         self.sync_EMG_option_val = StringVar()
#         self.sync_EMG_option_val.set(self.sync_EMG_option[0])
#         self.sync_EMG_option_menu = OptionMenu(self.frame_import, self.sync_EMG_option_val, *self.sync_EMG_option, command = self.EMG_sync_method_activator)
#         self.sync_EMG_option_menu.grid(row = 6, column = 0)
# =============================================================================
    #%% WarningNotEnoughDataMessage
        self.WarningNotEnoughDataMessage = "Dear user! \nAll required data are not uploaded! \n Please upload them all and try again."
    
    #%% Activation/inactivation of EMG button depending on the checkbox
    def EMG_button_activator(self):
        
        # EMG load button
        if self.button_EMG_browse['state'] == tk.DISABLED:
            self.button_EMG_browse['state'] = tk.NORMAL

        else:
            self.button_EMG_browse['state'] = tk.DISABLED
            
        # EMG option menu 
        if self.EMG_scale_option_menu['state'] == tk.DISABLED:
            self.EMG_scale_option_menu['state'] = tk.NORMAL

        else:
            self.EMG_scale_option_menu['state'] = tk.DISABLED
            
        # Sync button
        if self.button_sync_EMG['state'] == tk.DISABLED:
            self.button_sync_EMG['state'] = tk.NORMAL

        else:
            self.button_sync_EMG['state'] = tk.DISABLED
            
        # Sync button
        if self.button_apply['state'] == tk.NORMAL:
            self.button_apply['state'] = tk.DISABLED

        else:
            self.button_apply['state'] = tk.NORMAL
    #%% Moving average filter    
    def MA(self,x, N):
    
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
        
    #%% Activation/inactivation of EMG button depending on the checkbox
    def EMG_sync_method_activator(self):

        global sample_sync_EMG_Hypnodyne
        # Sanitary checks
        if 'hypnodyne_files_list' not in globals():
            messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The EEG L.edf file is not selected!")

        elif 'Dreamento_files_list' not in globals():
            messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The .txt file recorded by Dreamento is not selected!")
            
        elif 'marker_files_list' not in globals():
            messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The .json file of markers is not selected!")
            
        elif 'EMG_files_list' not in globals() and int(self.plot_additional_EMG.get()) == 1:
            messagebox.showerror("Dreamento", "Sorry, but no EMG files is loaded, though the plot EMG check box is activated! Change either of these and try again.")
            
        elif str(self.EMG_scale_options_val.get()) == 'Set desired EMG amplitude ...' and int(self.plot_additional_EMG.get()) == 1:
            messagebox.showerror("Dreamento", "Sorry, but a parameter is missing ...\nThe EMG amplitude is not set!")
            
        else: 
            Dreamento_files_list, hypnodyne_files_list, marker_files_list
            
            self.ZmaxDondersRecording = Dreamento_files_list[0]    
            self.HDRecorderRecording  = hypnodyne_files_list[0]
            self.path_to_json_markers = marker_files_list[0]
            self.path_to_EMG          = EMG_files_list[0]
            
        # Opening JSON file
        f = open(self.path_to_json_markers,)
        
        print('reading annotation file ...')
        # returns JSON object as a dictionary
        markers = json.load(f)
        markers_details = list(markers.values())
        marker_keys = list(markers.keys())

        clench_event = []
        counter_sync = []
        time_sync_event = []

        for counter, marker in enumerate(markers.values()):
            if marker.split()[0] == 'clench':
                print(marker.split())
                counter_sync.append(counter)
        
        for counter, marker in enumerate(markers.keys()):
            if counter in counter_sync:
                time_sync_event.append(int(marker.split()[-1]))

        print('Loading EEG file ... Please wait')                            
        path_Txt = self.ZmaxDondersRecording

        sigScript = np.loadtxt(path_Txt, delimiter=',')

        sigScript_org = sigScript
        sigScript_org_R = sigScript[:, 0]
        sigScript_org = sigScript_org[:, 1]

        # Read EMG
        print('Loading EEG file ... Please wait ...')                            
        EMG_data = mne.io.read_raw_brainvision(self.path_to_EMG , preload = True)
        
        print('EEG and EMG imported successfully')
        # Read annotations
        
        # Reading sampling frequencies
        EMG_sf = int(EMG_data.info['sfreq'])
        EEG_sf = 256   
        print(f'samlping frequency of EMG and EEG are: {EMG_sf} , {EEG_sf} Hz, respectively ... ')
       
        if EEG_sf < EMG_sf:
            print(f'resampling EMG to {EEG_sf} Hz ...')
            EMG_data.resample(EEG_sf)
        
        EMG_data_get = EMG_data.get_data()
        EMG_data_get = EMG_data_get[0,:] * 1e6
        
        #Filtering 
        sigScript_org   = self.butter_bandpass_filter(data = sigScript_org, lowcut=5, highcut=100, fs = 256, order = 2)
        EMG_data_get   = self.butter_bandpass_filter(data = EMG_data_get, lowcut=5, highcut=100, fs = 256, order = 2)
        t_sync = np.arange(time_sync_event[0] - 256*10, time_sync_event[0] + 256*20)
        
        # Truncate sync period
        EEG_to_sync_period = sigScript_org[t_sync]
        EMG_to_sync_period = EMG_data_get[t_sync]
        
        # Rectified signal
        EEG = EEG_to_sync_period
        EMG = EMG_to_sync_period
        EMG_Abs= abs(EMG)
        EEG_Abs = abs(EEG)
        
        MA_EEG = self.MA(EEG_Abs, 512)
        MA_EMG = self.MA(EMG_Abs, 512)
        
        fig, axs = plt.subplots(3, figsize = (10,10))
        axs[0].set_title('EEG during sync event')
        axs[0].plot(EEG_Abs, color = 'powderblue')
        axs[1].set_title('EMG during sync event')
        axs[1].plot(EMG_Abs, color = 'plum')
        axs[2].plot(EEG_Abs, color = 'powderblue')
        axs[2].set_title('EMG vs EEG plotted on top of each other')
        axs[2].plot(EMG_Abs, color = 'plum')
        
        MsgBox = tk.messagebox.askquestion ('EEG vs EMG synchronization','Look at the data durong sync period. Does the data require further synchronization?',icon = 'warning')
        plt.show()
        if MsgBox == 'yes':
            self.flag_sync_EEG_EMG = True
            print('Proceeding to synchronization process ...')
            while True:
                MsgBox = tk.messagebox.askquestion ('Synchronization?','Proceed to automatic synchronization? For manual sync, press No.',icon = 'warning')
                if MsgBox == 'yes':
    
                    fig, axs = plt.subplots(4, figsize = (15,10))
                    axs[0].plot(EEG_Abs, color = 'powderblue')
                    axs[0].plot(MA_EEG, color = 'navy', linewidth=3)
                    axs[0].set_title('EEG')
                    axs[1].plot(EMG_Abs, color = 'plum')
                    axs[1].plot(MA_EMG, color = 'purple', linewidth=3)
                    axs[1].set_title('EMG')
                    
    # =============================================================================
    #                 x = EEG_Abs
    #                 y = EMG_Abs
    # =============================================================================
                    x = MA_EEG
                    y = MA_EMG                
    
                    N = max(len(x), len(y))
                    n = min(len(x), len(y))
    
                    if N == len(y):
                        lags = np.arange(-N + 1, n)
    
                    else:
                        lags = np.arange(-n + 1, N)
    
                    c = signal.correlate(x / np.std(x), y / np.std(y), 'full')
    
    
                    axs[2].plot(lags, c / n, color='k', label="Cross-correlation")
    
                    axs[2].set_title('Cross-correlation')
    
    
                    corr2 = np.correlate(x, y, "full")
                    lag2 = np.argmax(corr2)
    
                    samples_before_begin = lag2 + 1 - len(y)
                    samples_after_end = len(x) - samples_before_begin - len(y)
                    index_start_script = samples_before_begin
                    index_end_script = len(x) - samples_after_end
                    print(f"samples_before_begin {samples_before_begin}")
                    print(f"samples_after_end {samples_after_end}")
                    print(f"index_start_script {index_start_script}")
                    print(f"index_end_script {index_end_script}")
                    
                    
                    if samples_before_begin < 0:
                        axs[3].plot(EEG, color ='powderblue')
                        axs[3].plot(MA_EEG, color = 'navy', linewidth=3)
                        axs[3].plot(EMG[-samples_before_begin:], color = 'plum')
                        axs[3].plot(MA_EMG[-samples_before_begin:], color = 'purple', linewidth=3)
                        axs[3].set_title('EEG and EMG after sync')
                        self.samples_before_begin_EMG_Dreamento = -samples_before_begin
                        self.flag_sign_samples_before_begin_EMG_Dreamento = 'eeg_event_earlier'
                    else:
                        axs[3].plot(EEG, color ='powderblue')
                        axs[3].plot(MA_EEG, color = 'navy', linewidth=3)
                        tmp = np.zeros(samples_before_begin)
                        synced_EMG = np.append(tmp, EMG)
                        synced_EMG_MA = np.append(tmp, MA_EMG)
                        axs[3].plot(synced_EMG, color = 'plum')
                        axs[3].plot(synced_EMG_MA, color = 'purple', linewidth=3)
                        axs[3].set_title('EEG and EMG after sync')
                        self.samples_before_begin_EMG_Dreamento = tmp
                        self.flag_sign_samples_before_begin_EMG_Dreamento = 'emg_earlier'
                    MsgBox = tk.messagebox.askquestion ('Satisfying results?','Are the results satisfying? If not click on No to try again with the other method.',icon = 'warning')
                    if MsgBox == 'yes':
                        messagebox.showinfo("Information",f"Perfect! Now we proceed with the main analysis")
                        plt.show()
                        break
                        
                else: 
                    MsgBox = tk.messagebox.askquestion ('Synchronization?','Do you want to manually synchronize data?',icon = 'warning')
                    if MsgBox == 'yes':
                        # Truncate sync period
                        EEG_to_sync_period = sigScript_org[(time_sync_event[0] - 256*5):(time_sync_event[0] + 256*20)]
                        EMG_to_sync_period = EMG_data_get[(time_sync_event[0] - 256*5):(time_sync_event[0] + 256*20)]
                        
                        # Rectified signal
                        self.EEG = EEG_to_sync_period
                        self.EMG = EMG_to_sync_period
                        EMG_Abs= abs(self.EMG)
                        EEG_Abs = abs(self.EEG)
                            
                        self.points = []
                        self.n = 2
        
                        self.fig, self.axs = plt.subplots(3 ,figsize=(15, 10))
                        line = self.axs[0].plot(EEG_Abs, picker=2, color = 'powderblue')
                        self.axs[0].set_title('Manual drift estimation ... \nPlease click on two points to create the estimate line ...')
                        
                        self.MA_EEG = self.MA(EEG_Abs, 512)
                        self.MA_EMG = self.MA(EMG_Abs, 512)
        
                        self.axs[0].set_xlim([0,len(self.EMG)])
                        self.axs[1].set_xlim([0,len(self.EMG)])
                        self.axs[2].set_xlim([0,len(self.EMG)])
        
                        self.axs[0].set_ylabel('Lag (samples)')
        
                        line = self.axs[1].plot(EMG_Abs, picker=2, color = 'plum')
        
                        plt.show()
                        self.fig.canvas.mpl_connect('pick_event', self.onpick)
                        MsgBox = tk.messagebox.askquestion ('Satisfying results?','Are the results satisfying? If not click on No to try again with the other method.',icon = 'warning')
                        if MsgBox == 'yes':
                            messagebox.showinfo("Information",f"Perfect! Now we proceed with the main analysis ... Please wait ...")
                            plt.show()
                            break
                        
            
        # Loading all available data            
        self.noise_path = hypnodyne_files_list[0].split('EEG')[0] + 'NOISE.edf'
        self.noise_obj = mne.io.read_raw_edf(self.noise_path)
        self.noise_data = self.noise_obj.get_data()[0]
        
        # Acc
        self.acc_x_path = hypnodyne_files_list[0].split('EEG')[0] + 'dX.edf'
        self.acc_y_path = hypnodyne_files_list[0].split('EEG')[0] + 'dY.edf'
        self.acc_z_path = hypnodyne_files_list[0].split('EEG')[0] + 'dz.edf'
        
        self.acc_x_obj = mne.io.read_raw_edf(self.acc_x_path)
        self.acc_y_obj = mne.io.read_raw_edf(self.acc_y_path)
        self.acc_z_obj = mne.io.read_raw_edf(self.acc_z_path)
        
        self.acc_x = self.acc_x_obj.get_data()[0]
        self.acc_y = self.acc_y_obj.get_data()[0]
        self.acc_z = self.acc_z_obj.get_data()[0]
        
        print('Acceleration and noise data imported successfully ...')
        
        self.samples_before_begin, self.sigHDRecorder_org_synced, self.sigScript_org, self.sigScript_org_R = self.calculate_lag(
                               plot=(int(self.plot_sync_output.get()) ==1) , path_EDF=self.HDRecorderRecording,\
                               path_Txt=self.ZmaxDondersRecording,\
                               T = 30,\
                               t_start_sync = 100,\
                               t_end_sync   = 130)
        print('The lag between Dreamento and Hypndoyne EEG computed ...')
        # Filter?
        if int(self.is_filtering.get()) == 1: 
            print('Bandpass filtering (.3-30 Hz) started')
            self.sigScript_org   = self.butter_bandpass_filter(data = self.sigScript_org, lowcut=.3, highcut=30, fs = 256, order = 2)
            self.sigScript_org_R = self.butter_bandpass_filter(data = self.sigScript_org_R, lowcut=.3, highcut=30, fs = 256, order = 2)
            print('Band-pass filter applied to data ...')
        else:
            print('No filtering applied ...')

        # Plot psd?
        if int(self.plot_psd.get()) == 1:
            print('plotting peridogram ...')
            self.plot_welch_periodogram(data = self.sigScript_org, sf = 256, win_size = 5)
            print('PSD plotted ...')
            
        # Plot EMG as well?
        print('Loading the main window of Dreamento ... Please wait')
        if int(self.plot_additional_EMG.get()) == 1:
            fig, markers_details = self.AssignMarkersToRecordedData_EEG_TFR(data = self.sigScript_org, data_R = self.sigScript_org_R, sf = 256,\
                                             path_to_json_markers=self.path_to_json_markers,EMG_path=self.path_to_EMG,\
                                             markers_to_show = ['light', 'manual', 'sound'],\
                                             win_sec=30, fmin=0.5, fmax=25,\
                                             trimperc=5, cmap='RdBu_r', add_colorbar = False)
                
        else:
                fig, markers_details = self.AssignMarkersToRecordedData_EEG_TFR_noEMG(data = self.sigScript_org, data_R = self.sigScript_org_R, sf = 256,\
                                             path_to_json_markers=self.path_to_json_markers,\
                                             markers_to_show = ['light', 'manual', 'sound'],\
                                             win_sec=30, fmin=0.5, fmax=25,\
                                             trimperc=5, cmap='RdBu_r', add_colorbar = False)
        # Activate save section
        self.create_save_options()

    #%% Manual sync function        
    def onpick(self,event):
        point1_x = []
        point2_x = []
        point1_y = []
        point2_y = []
        
        if len(self.points) < self.n:
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            point = tuple(zip(xdata[ind], ydata[ind]))
            self.points.append(point)
            print('onpick point:', point)
            print(f'You already chose {len(self.points)} points')
            
            if len(self.points) == self.n :
                print('done')
                
                for i in self.points[0]:
                    
                    point1_x.append(i[0])
                    point1_y.append(i[1])
                    
                for i in self.points[1]:
                    point2_x.append(i[0])
                    point2_y.append(i[1])
                    
                mean_x1 = np.mean(point1_x)
                mean_x2 = np.mean(point2_x)
                mean_y1 = np.mean(point1_y)
                mean_y2 = np.mean(point2_y)
                
                if mean_x1 > mean_x2:
                    self.axs[2].plot(self.EEG, color = 'powderblue')
                    
                    tmp_sync = np.zeros(int(mean_x1-mean_x2))
                    self.synced_EMG = np.append(tmp_sync, self.EMG)
                    self.synced_MA_EMG = np.append(tmp_sync, self.MA_EMG)
                    self.axs[2].plot(self.synced_EMG, color = 'plum')
                    self.axs[2].plot(self.MA_EEG, color = 'navy', linewidth = 2)
                    self.axs[2].plot(self.synced_MA_EMG, color = 'purple', linewidth = 2)
                    n_sample_sync = len(tmp_sync)
                    
                    self.samples_before_begin_EMG_Dreamento = tmp_sync
                    self.flag_sign_samples_before_begin_EMG_Dreamento = 'emg_event_earlier'
                    
                if mean_x1 < mean_x2:
                    self.axs[2].plot(self.EEG, color = 'powderblue')
                    tmp_sync = int(mean_x2-mean_x1)
                    self.axs[2].plot(self.EMG[tmp_sync:], color = 'plum')
                    self.axs[2].plot(self.MA_EEG, color = 'navy', linewidth = 2)
                    self.axs[2].plot(self.MA_EEG[tmp_sync:], color = 'purple', linewidth = 2)
                    self.samples_before_begin_EMG_Dreamento = tmp_sync
                    self.flag_sign_samples_before_begin_EMG_Dreamento = 'eeg_event_earlier'
                #self.axs[1].plot([mean_x1, mean_x2], [mean_y1, mean_y2], linewidth = 3, color = 'plum')
                self.fig.canvas.draw()
            #return drift_estimate
        return self.points
        #%% Save section    
    def create_save_options(self):
        """
        Store all the raw and processed variables to be used in Matlab
    
        :param self: access the attributes and methods of the class
        
        :returns: user_defined_name.mat

        """
        
        
        
        # Label: Save outcome
        self.label_save_path = Label(self.frame_import, text = "Saving path:",
                                  font = 'Calibri 13 ')
        self.label_save_path.grid(row = 4 , column = 0, padx = 15, pady = 10)
        
        # Define browse button to import hypnos
        self.button_save_browse = Button(self.frame_import, text = "Browse ...", 
                                           padx = 40, pady = 10, font = 'Calibri 12 ',
                                           command = self.save_path_finder,fg = 'blue',
                                           relief = RIDGE)
        self.button_save_browse.grid(row = 5, column = 0, padx = 15, pady = 10)
        
        
        
        # Label: Save name
        self.label_save_filename = Label(self.frame_import, text = "Saving filename:",
                                  font = 'Calibri 13 ')
        self.label_save_filename.grid(row = 4 , column = 1)#, padx = 15, pady = 10)
        
        # Create entry for user
        self.entry_save_name = Entry(self.frame_import,text = " enter filename.mat ")#, borderwidth = 2, width = 10)
        self.entry_save_name.grid(row = 5, column = 1)#, padx = 15, pady = 10)
        
        self.button_save_mat = Button(self.frame_import, text = "Save", padx = 40, pady=8,
                              font = 'Calibri 13 bold', relief = RIDGE, fg = 'blue',
                              command = lambda: self.save_results_button(data_EEG_L = self.sigScript_org,\
                                        data_EEG_R=self.sigScript_org_R, markers = self.markers_details,\
                                        marker_keys= self.marker_keys, samples_to_sync = self.samples_before_begin,\
                                        raw_EMG1 = self.EMG_raw, raw_EMG2 = self.EMG_filtered_data2, raw_EMG3 = self.EMG_filtered_data1_minus_2,\
                                        microphone_data = self.noise_data,\
                                        acc_x = self.acc_x, acc_y = self.acc_y, acc_z = self.acc_z,\
                                        Hypnodyne_EEG_L = self.sigHDRecorder_org_synced))
        self.button_save_mat.grid(row = 5 , column =2, padx = 15, pady = 10)
        

    #%% ################### DEFINE FUNCTIONS OF BUTTON(S) #######################
    #%% Function: Import EDF (Browse)
    def load_hypnodyne_file_dialog(self):
        """
        The function to load the EEG.L file from the HDrecorder.

        :param self: access the attributes and methods of the class
        
        :returns: global hypnodyne_files_list

        """
    
        global hypnodyne_files_list
        
        self.filenames        = filedialog.askopenfilenames(title = 'select data files', 
                                                       filetype = (("edf", "*.edf"), ("All Files", "*.*")))
        
        # Make a list of imported file names (full path)
        hypnodyne_files_list       = self.frame_import.tk.splitlist(self.filenames)
        self.n_data_files     = len(hypnodyne_files_list)
        
        # check if the user chose somthing
        if not hypnodyne_files_list:
            
            self.label_data       = Label(self.frame_import, text = "No file has been selected!",
                                          fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 0)
    
        else:
            self.label_data       = Label(self.frame_import, text = "The EDF files has been loaded!",
                                          fg = 'green', font = 'Helvetica 9 bold').grid(row = 2, column = 0)
            
    #%% Function: Import Hypnogram (Browse)
    def load_Dreamento_file_dialog(self):
        
        """
        Loading the .txt file recorded by Dreamento.
        
        :param self: access the attributes and methods of the class
        
        :returns: global Dreamento_files_list

        """
    
        global Dreamento_files_list
        
        self.filenames    = filedialog.askopenfilenames(title = 'select label files', 
                                                       filetype = (("txt", "*.txt"),("csv", "*.csv"), ("All Files", "*.*")))
        Dreamento_files_list  = self.frame_import.tk.splitlist(self.filenames)
        self.n_label_files     = len(Dreamento_files_list)
        
        # check if the user chose somthing
        if not Dreamento_files_list:
            
            self.label_labels  = Label(self.frame_import, text = "No hypnogram has been selected!",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 1)
            
        else:
            
            self.label_labels  = Label(self.frame_import, text ="The Dreamento data file has been loaded!",
                                  fg = 'green', font = 'Helvetica 9 bold').grid(row = 2, column = 1)
            
            
            
    #%% Function: Import Hypnogram (Browse)
    def load_marker_file_dialog(self):
        
        """
        Load the anntations file (.json) created by Dreamento.
        
        :param self: access the attributes and methods of the class
         
        :returns: global marker_files_list
        
        """
    
        global marker_files_list
        
        self.filenames    = filedialog.askopenfilenames(title = 'select marker files', 
                                                       filetype = (("json", "*.json"),("csv", "*.csv"), ("All Files", "*.*")))
        marker_files_list  = self.frame_import.tk.splitlist(self.filenames)
        self.n_label_files     = len(marker_files_list)
        
        # check if the user chose somthing
        if not marker_files_list:
            
            self.label_labels  = Label(self.frame_import, text = "No marker (.json) file has been selected!",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 2)
            
        else:
            
            self.label_labels  = Label(self.frame_import, text = "The marker (.json) file has been loaded!",
                                  fg = 'green', font = 'Helvetica 9 bold').grid(row = 2, column = 2)
            
    #%% Function: Import EMG (Browse)        
    def load_EMG_file_dialog(self):
        
        """
        Load the optional recorded data (e.g.,EMG.vhdr)
        
        :param self: access the attributes and methods of the class
        
        :returns: global EMG_files_list
        
        """
    
        global EMG_files_list
        
        self.filenames    = filedialog.askopenfilenames(title = 'select EMG file (.vhdr)', 
                                                       filetype = (("vhdr", "*.vhdr"),("vhdr", "*.vhdr"), ("All Files", "*.*")))
        EMG_files_list  = self.frame_import.tk.splitlist(self.filenames)
        self.n_label_files     = len(EMG_files_list)
        
        # check if the user chose somthing
        if not EMG_files_list:
            
            self.label_labels  = Label(self.frame_import, text = "No EMG (.vhdr) file has been selected!",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 2, column = 3)
            
        else:
            
            self.label_labels  = Label(self.frame_import, text = "The EMG (.vhdr) file has been loaded!",
                                  fg = 'green', font = 'Helvetica 9 bold').grid(row = 2, column = 3)
     
    #%% Function: Import Hypnogram (Browse)
    def save_path_finder(self):
        
        """
        Button to define the directory to save the .mat output
        
        :param self: access the attributes and methods of the class
        
        :returns: global where_to_save_path
        
        """
    
        global where_to_save_path
        where_to_save_path    = filedialog.askdirectory()

        
    #%% Function: Import Hypnogram (Browse)
    def Apply_button(self):
        
        """
        Start the processing, once all the parameters are set.
        
        :param self: access the attributes and methods of the class
        
        :returns display: The main post-processing Dreamento window, saving options

        """
        
        # Sanitary checks
        if 'hypnodyne_files_list' not in globals():
            messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The EEG L.edf file is not selected!")

        elif 'Dreamento_files_list' not in globals():
            messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The .txt file recorded by Dreamento is not selected!")
            
        elif 'marker_files_list' not in globals():
            messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The .json file of markers is not selected!")
            
        elif 'EMG_files_list' not in globals() and int(self.plot_additional_EMG.get()) == 1:
            messagebox.showerror("Dreamento", "Sorry, but no EMG files is loaded, though the plot EMG check box is activated! Change either of these and try again.")
            
        elif str(self.EMG_scale_options_val.get()) == 'Set desired EMG amplitude ...' and int(self.plot_additional_EMG.get()) == 1:
            messagebox.showerror("Dreamento", "Sorry, but a parameter is missing ...\nThe EMG amplitude is not set!")
            
        else: 
            Dreamento_files_list, hypnodyne_files_list, marker_files_list
            
                
            self.ZmaxDondersRecording = Dreamento_files_list[0]
            self.HDRecorderRecording  = hypnodyne_files_list[0]
            self.path_to_json_markers = marker_files_list[0]
            
            if int(self.plot_additional_EMG.get()) == 1:
                self.path_to_EMG          = EMG_files_list[0]
                        
            self.noise_path = hypnodyne_files_list[0].split('EEG')[0] + 'NOISE.edf'
            self.noise_obj = mne.io.read_raw_edf(self.noise_path)
            self.noise_data = self.noise_obj.get_data()[0]
            
            # Acc
            self.acc_x_path = hypnodyne_files_list[0].split('EEG')[0] + 'dX.edf'
            self.acc_y_path = hypnodyne_files_list[0].split('EEG')[0] + 'dY.edf'
            self.acc_z_path = hypnodyne_files_list[0].split('EEG')[0] + 'dz.edf'
            
            self.acc_x_obj = mne.io.read_raw_edf(self.acc_x_path)
            self.acc_y_obj = mne.io.read_raw_edf(self.acc_y_path)
            self.acc_z_obj = mne.io.read_raw_edf(self.acc_z_path)
            
            self.acc_x = self.acc_x_obj.get_data()[0]
            self.acc_y = self.acc_y_obj.get_data()[0]
            self.acc_z = self.acc_z_obj.get_data()[0]
            
            print('Required files imported successfully ...')

            self.samples_before_begin, self.sigHDRecorder_org_synced, self.sigScript_org, self.sigScript_org_R = self.calculate_lag(
                               plot=(int(self.plot_sync_output.get()) ==1) , path_EDF=self.HDRecorderRecording,\
                               path_Txt=self.ZmaxDondersRecording,\
                               T = 30,\
                               t_start_sync = 100,\
                               t_end_sync   = 130)
            # Filter?
            if int(self.is_filtering.get()) == 1: 
                print('Bandpass filtering (.3-30 Hz) started')
                self.sigScript_org   = self.butter_bandpass_filter(data = self.sigScript_org, lowcut=.3, highcut=30, fs = 256, order = 2)
                self.sigScript_org_R = self.butter_bandpass_filter(data = self.sigScript_org_R, lowcut=.3, highcut=30, fs = 256, order = 2)
                print(f'EMG scale is: {self.EMG_scale_options_val.get()} uV')

            # Plot psd?
            if int(self.plot_psd.get()) == 1:
                print('plotting peridogram ...')
                self.plot_welch_periodogram(data = self.sigScript_org, sf = 256, win_size = 5)
                
            # Plot spectrogram as well?
            if int(self.plot_additional_EMG.get()) == 1:
                
                fig, markers_details = self.AssignMarkersToRecordedData_EEG_TFR(data = self.sigScript_org, data_R = self.sigScript_org_R, sf = 256,\
                                             path_to_json_markers=self.path_to_json_markers,EMG_path=self.path_to_EMG,\
                                             markers_to_show = ['light', 'manual', 'sound'],\
                                             win_sec=30, fmin=0.5, fmax=25,\
                                             trimperc=5, cmap='RdBu_r', add_colorbar = False)
                    
            
            else:
                fig, markers_details = self.AssignMarkersToRecordedData_EEG_TFR_noEMG(data = self.sigScript_org, data_R = self.sigScript_org_R, sf = 256,\
                                             path_to_json_markers=self.path_to_json_markers,\
                                             markers_to_show = ['light', 'manual', 'sound'],\
                                             win_sec=30, fmin=0.5, fmax=25,\
                                             trimperc=5, cmap='RdBu_r', add_colorbar = False)
            # Activate save section
            self.create_save_options()
            
       
    #%% band-pass filtering
    
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order = 2):
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
    
    #%% Save button
    def save_results_button(self, data_EEG_L, data_EEG_R, markers, marker_keys, Hypnodyne_EEG_L, microphone_data, acc_x, acc_y, acc_z, samples_to_sync, raw_EMG1, raw_EMG2, raw_EMG3):
        
        """
        The command to save the results of the ra and processed data to a matfile.
        
        :param self: access the attributes and methods of the class
        :param data_EEG_L:  Dreamento EEG L
        :param data_EEG_R: Dreamento EEG R
        :param Hypnodyne_EEG_L: HDRecorder EEG L
        :param markers: List of annotations
        :param marker_keys: The keys to define the dictionaries of annotations
        :param samples_to_sync: The number of samples to be compensated for sync.                    
        :param microphone_data: mic recording from the headband                
        :param acc_x: The x-axis of accleration
        :param acc_y: The y-axis of accleration
        :param acc_z: The z-axis of accleration
        :param raw_EMG1: The first recorded EMG channel
        :param raw_EMG2: The second recorded EMG channel
        :param raw_EMG3: The third recorded EMG channel

        :returns:  matfile.mat

        """
        
        from scipy.io import savemat
        
        self.dict_all_results = dict()
        self.dict_all_results['EEG_L_Dreamento'] = data_EEG_L #self.sigScript_org
        self.dict_all_results['EEG_R_Dreamento'] = data_EEG_R # self.sigScript_org_R
        self.dict_all_results['markers']        = markers # markers_details
        self.dict_all_results['marker_keys']    = marker_keys # marker_keys
        self.dict_all_results['Hypnodyne_EEG_L']= Hypnodyne_EEG_L # marker_key

        self.dict_all_results['samples_to_sync']= samples_to_sync # marker_keys
        self.dict_all_results['microphone_data']= microphone_data # marker_keys
        self.dict_all_results['acc_x']= acc_x # marker_keys
        self.dict_all_results['acc_y']= acc_y # marker_keys
        self.dict_all_results['acc_z']= acc_z # marker_keys
        self.dict_all_results['raw_EMG1']= self.EMG_filtered_data1 # marker_keys
        self.dict_all_results['raw_EMG2']= self.EMG_filtered_data2 # marker_keys
        self.dict_all_results['raw_EMG1_minus_EMG2']= self.EMG_filtered_data1_minus_2


        
        if self.entry_save_name.get()[-4:] == '.mat':
           saving_dir = where_to_save_path + '/' + self.entry_save_name.get()
        else:
           saving_dir = where_to_save_path + '/' + self.entry_save_name.get() + '.mat'
        #savemat('saving_dir.MAT', self.dict_all_results)
        savemat(saving_dir, self.dict_all_results)
        messagebox.showinfo(title = "Output sucessfully saved", message = f'file saved in {saving_dir}!')
        
    #%% Coimpute correlation
    def plot_xcorr(self, x, y, ax=None):
        """
        Plot  the cross-correlation between signals.
        
        :param self: access the attributes and methods of the class
        :param signal1: sigHDRecorderTrimmed
        :param signal2: sig Dreamento
        :param axis:
        
        :returns: plot
        
        """
        # "Plot cross-correlation (full) between two signals."
        N = max(len(x), len(y))
        n = min(len(x), len(y))
    
        if N == len(y):
            lags = np.arange(-N + 1, n)
    
        else:
            lags = np.arange(-n + 1, N)
    
        c = signal.correlate(x / np.std(x), y / np.std(y), 'full')
    
        if ax is None:
            plt.plot(lags, c / n)
            plt.show()
    
        else:
            ax.plot(lags, c / n, color='k', label="Cross-correlation")
    
    #%% Compute lag
    def calculate_lag(self, plot=False, path_EDF=None, path_Txt=None,
                      T = 30,
                      t_start_sync = 5,
                      t_end_sync   = 7):
        
        """
        Calculate the lag between the HDRecorder and Dreamento and compensate it.
        
        :param self: access the attributes and methods of the class
        :param plot: activate/deactivate plotting
        :param path_EDF: path to HDRecorder EEG L.edf file
        :param path_Txt: path to Dreamento .txt file
        :param T: The duration to use for synchronization (seconds)
        :param t_start_sync: the start time of a nevent for sync. 
        :param t_end_sync: the end point of a nevent for sync. 
        
        :returns: samples_before_begin, sigHDRecorder_org_synced, sigScript_org, sigScript_org_R fwsd
        
        """
        global sigHDRecorder, sigScript
        sigScript = np.loadtxt(path_Txt, delimiter=',')
        
        sigScript_org = sigScript
        sigScript_org_R = sigScript[:, 0]
        sigScript_org = sigScript_org[:, 1]
        sigScript = sigScript[int(t_start_sync * 256):int(t_end_sync * 256), 1]
    
        data = mne.io.read_raw_edf(path_EDF)
        raw_data = data.get_data()
        sigHDRecorder = np.ravel(raw_data)
        sigHDRecorder = sigHDRecorder
        sigHDRecorder_org = sigHDRecorder
    
        # for example: epoch 15 to 17
        # T = 30, t_start_sync = 15, t_end_sync = 17
        sigHDRecorder = sigHDRecorder[int(t_start_sync  * 256):int(t_end_sync * 256)]
        print('Calculating the lag between signals ...')
        corr = signal.correlate(sigHDRecorder, sigScript)  # Compute correlation
        lag = np.argmax(np.abs(corr))  # find lag
    
        corr2 = np.correlate(sigHDRecorder, sigScript, "full")
        lag2 = np.argmax(corr2)
    
        xCorrZeros = np.zeros(lag2 + 1 - len(sigScript))
        xShiftedForward = np.append(xCorrZeros, sigScript)
    
        samples_before_begin = lag2 + 1 - len(sigScript)
        samples_after_end = len(sigHDRecorder) - samples_before_begin - len(sigScript)
        index_start_script = samples_before_begin
        index_end_script = len(sigHDRecorder) - samples_after_end
        print(f"samples_before_begin {samples_before_begin}")
        print(f"samples_after_end {samples_after_end}")
        print(f"index_start_script {index_start_script}")
        print(f"index_end_script {index_end_script}")
        sigHDRecorderTrimmed = sigHDRecorder[index_start_script:index_end_script + 1]
    
        print(f"HDRecorder {samples_before_begin} samples longer")
    
        if plot:
            fig, ax = plt.subplots(4, 1, figsize=(16, 12))
            ax[0].plot(sigScript, color='r', label="sigScript")
            ax[1].plot(sigHDRecorder, color='g', label="sigHDRecorder")
            ax[2].plot(sigScript, color='r', label="sigScript")
            ax[2].plot(sigHDRecorderTrimmed * 1e6, color='g', label=f"HDRecorder {samples_before_begin} longer")
            self.plot_xcorr(sigHDRecorderTrimmed, sigScript, ax=ax[3])
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            ax[3].legend()
            plt.tight_layout()
            # plt.savefig('similarity.png')
            # plt.savefig('similarity.pdf')
            # plt.savefig('similarity.svg')
            # plt.savefig('similarityT.png', transparent=True)
            # plt.savefig('similarityT.pdf', transparent=True)
            # plt.savefig('similarityT.svg', transparent=True)
            plt.show()
    
            # from scipy.signal import butter, filtfilt
            # sigScriptFiltered = sigScript_org / 1e6
            # lowcut = 0.3
            # highcut = 30
            # nyquist_freq = 256 / 2.
            # low = lowcut / nyquist_freq
            # high = highcut / nyquist_freq
            # # Req channel
            # b, a = butter(2, [low, high], btype='band')
            # sigScriptFiltered_L = filtfilt(b, a, data)
            # sigScriptFiltered_R = filtfilt(b, a, data)
    
            # Now plot the complete signals
            plt.figure()
            plt.title("Synced Signals")
            plt.plot(sigScript_org, label='EEG L - Dreamento')
            plt.plot(sigHDRecorder_org[samples_before_begin:]*1e6, label='EEG L - HDRecorder')
            plt.legend()
            plt.show()
            
        # Synced data
        sigHDRecorder_org_synced = sigHDRecorder_org[samples_before_begin:]
        self.sigHDRecorder_org_synced = sigHDRecorder_org_synced
        
        print('Lag cmputation finished')
        return samples_before_begin, sigHDRecorder_org_synced, sigScript_org, sigScript_org_R
    #%% Plot PSD
    def plot_welch_periodogram(self, data,  sf=256, win_size = 4, log_power = False):
        
        """
        Plot the welch periodogram
        
        :param self: access the attributes and methods of the class
        :param data: EEG data
        :param sf: sampoling frequency 
        :param win_size: the size of sliding windows for Welch method.
        :param log_power: compute logarithmic power
        
        :returns: plot
        
        """
        
        from scipy import signal
        import seaborn as sns
        import yasa
        
        # Define window length (4 seconds)
        win = win_size * sf
        freqs, psd = signal.welch(data, sf, nperseg=win)
        
        if log_power:
            psd = 20 * np.log10(psd)
        
        # Compute vvalues: 
        ret = yasa.bandpower(data, sf=sf)
        # Plot the power spectrum
        sns.set(font_scale=1.2, style='white')
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, psd, color='k', lw=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power spectral density (V^2 / Hz)')
        plt.ylim([0, psd.max() * 1.1])
        plt.title("Welch's periodogram")
        plt.xlim([0, 30])
        sns.despine()
        
        
        # Delta
        plt.axvline(.5, linestyle='--', color='black')
        plt.axvline(4, linestyle='--', color='black')
    
        # Theta
        plt.axvline(8, linestyle='--', color='black')
    
        # Alpha
        plt.axvline(12, linestyle='--', color='black')
        
        loc_pow_vals = np.mean([np.min(psd), np.max(psd)])
        loc_labels = loc_pow_vals
        plt.text(1.3, loc_labels, 'Delta', size=8)
        plt.text(4.8, loc_labels, 'Theta', size=8)
        plt.text(8.8, loc_labels, 'Alpha', size=8)
        plt.text(12.8, loc_labels, 'Beta', size=8)
    
        # legend - values
        plt.legend([f'Delta (0.5-4 Hz)): {round(ret["Delta"][0] * 100, 2)}% \n\
    Theta (4-8 Hz): {round(ret["Theta"][0] * 100, 2)}% \n\
    Alpha (8-12 Hz): {round(ret["Alpha"][0] * 100, 2)}% \n\
    Sigma (12-16 Hz): {round(ret["Sigma"][0] * 100, 2)}% \n\
    Beta (16-30 Hz): {round(ret["Beta"][0] * 100, 2)}% \n\
    Gamma (30-40 Hz): {round(ret["Gamma"][0] * 100, 2)}%'], prop={'size': 10}, frameon=False)
        
    #%% AssignMarkersToRecordedData EEG + TFR
    def AssignMarkersToRecordedData_EEG_TFR(self, data, data_R, sf, path_to_json_markers,EMG_path, markers_to_show = ['light', 'manual', 'sound'],\
                                win_sec=30, fmin=0.3, fmax=40,
                                trimperc=5, cmap='RdBu_r', add_colorbar = False):
        """
        The main function: create the main display window of Offline Dreamento
        
        :param self: access the attributes and methods of the class
        :param data: EEG L data
        :param data_R: EEG R data
        :param sf: sampling frequency
        :param path_to_json_markers: path to the annotation file (generated by Dreamento)
        :param EMG_path: path to the EMG file
        :param markers_to_show: the desired type of markers to show ['light', 'manual', 'sound']
        :param win_sec: the window size to compute spectrogram
        :param fmin: relavant min frequency for spectrogram
        :param fmax: relavant max frequency for spectrogram
        :param trimperc: bidirectional trim factor for normalizing the spectrogram colormap
        :param cmap: colormap of the spectrogram
        :param add_colorbar: adding colorbar to spectrogram

        :returns: fig, markers_details
        """
        
# =============================================================================
#         Source code for plotting spectrogram: YASA (https://raphaelvallat.com/yasa/build/html/_modules/yasa/plotting.html#plot_spectrogram)
#         [modified]                                           	
# =============================================================================
        # Increase font size while preserving original
        old_fontsize = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': 12})
        
        # Safety checks
        assert isinstance(data, np.ndarray), 'Data must be a 1D NumPy array.'
        assert isinstance(sf, (int, float)), 'sf must be int or float.'
        assert data.ndim == 1, 'Data must be a 1D (single-channel) NumPy array.'
        assert isinstance(win_sec, (int, float)), 'win_sec must be int or float.'
        assert isinstance(fmin, (int, float)), 'fmin must be int or float.'
        assert isinstance(fmax, (int, float)), 'fmax must be int or float.'
        assert fmin < fmax, 'fmin must be strictly inferior to fmax.'
        assert fmax < sf / 2, 'fmax must be less than Nyquist (sf / 2).'
    
        # Calculate multi-taper spectrogram
        nperseg = int(10 * sf) #int(win_sec * sf / 8)
        assert data.size > 2 * nperseg, 'Data length must be at least 2 * win_sec.'
        f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
        Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz
    
        # Select only relevant frequencies (up to 30 Hz)
        good_freqs = np.logical_and(f >= fmin, f <= fmax)
        Sxx = Sxx[good_freqs, :]
        f = f[good_freqs]
        #t *= 256  # Convert t to hours
    
        # Normalization
        vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        #!!!!!!!!!! SHORT SPECTROGRAM!! Calculate multi-taper spectrogram
        nperseg2 = int(.2 * sf) #int(win_sec * sf / 8)
        assert data.size > 2 * nperseg2, 'Data length must be at least 2 * win_sec.'
        f2, t2, Sxx2 = spectrogram_lspopt(data, sf, nperseg=nperseg2, noverlap=0)
        Sxx2 = 10 * np.log10(Sxx2)  # Convert uV^2 / Hz --> dB / Hz
        print(f'f: {np.shape(f)}, f2: {np.shape(f2)}, t: {np.shape(t)}, t2: {np.shape(t2)}, Sxx: {np.shape(Sxx)}, Sxx2: {np.shape(Sxx2)}')
        # Select only relevant frequencies (up to 30 Hz)
        good_freqs2 = np.logical_and(f2 >= fmin, f2 <= fmax)
        Sxx2 = Sxx2[good_freqs2, :]
        f2 = f2[good_freqs2]
        #t *= 256  # Convert t to hours
    
        # Normalization
        vmin2, vmax2 = np.percentile(Sxx2, [0 + trimperc, 100 - trimperc])
        norm2 = Normalize(vmin=vmin2, vmax=vmax2)
    # =============================================================================
    #     gs1 = gridspec.GridSpec(2, 1)
    #     gs1.update(wspace=0.005, hspace=0.0001)
    # =============================================================================
        fig,AX = plt.subplots(nrows=12, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 10,1,1, 2, 2,4, 4, 4, 4, 10]})
        
        ax1 = plt.subplot(12,1,1)
        ax2 = plt.subplot(12,1,2, sharex = ax1)
        ax3 = plt.subplot(12,1,3, sharex = ax1)
        
        ax_epoch_marker = plt.subplot(12,1,4, )
        ax_epoch_light = plt.subplot(12,1,5, sharex = ax_epoch_marker)
        
        ax_acc = plt.subplot(12,1,6, sharex = ax_epoch_marker)
        ax_noise = plt.subplot(12,1,7, sharex = ax_epoch_marker)
        

        
        ax_EMG = plt.subplot(12,1,8, sharex = ax_epoch_marker)
        ax_EMG2 = plt.subplot(12,1,9, sharex = ax_epoch_marker)
        ax_EMG3 = plt.subplot(12,1,10, sharex = ax_epoch_marker)
        ax_TFR_short = plt.subplot(12,1,11, sharex = ax_epoch_marker)
        ax4 = plt.subplot(12,1,12, sharex = ax_epoch_marker)
        ax4.grid(True)

        ax1.get_xaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax_epoch_marker.get_xaxis().set_visible(False)
        ax_epoch_light.get_xaxis().set_visible(False)
        ax_EMG.get_xaxis().set_visible(False)
        ax_acc.get_xaxis().set_visible(False)
        ax_acc.set_yticks([])
        ax_noise.set_yticks([])
        ax_noise.get_xaxis().set_visible(False)
        
        ax_acc.set_ylim([-1.4, 1.4])
        
        plt.subplots_adjust(hspace = 0)
        ax1.set_title('Dreamento: post-processing ')
        im = ax3.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True,
                           shading="auto")
        ax3.set_xlim([0, len(data)/256])
        ax3.set_ylim((fmin, 25))
        ax3.set_ylabel('Frequency [Hz]')
        
        im2 = ax_TFR_short.pcolormesh(t2, f2, Sxx2, norm=norm2, cmap=cmap, antialiased=True,
                           shading="auto")
        
        # Add colorbar
        if add_colorbar == True:
            cbar = fig.colorbar(im, ax=ax3, shrink=0.95, fraction=0.1, aspect=25, pad=0.01)
            cbar.ax3.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=5)
            
        # PLOT EEG
        ax4.plot(np.arange(len(data))/256, data, color = (160/255, 70/255, 160/255), linewidth = 1)
        ax4.plot(np.arange(len(data_R))/256, data_R, color = (0/255, 128/255, 190/255), linewidth = 1)
        #axes[1].set_ylim([-200, 200])
        #ax4.set_xlim([0, len(data)])
        ax4.set_ylabel('EEG (uV)')
        ax4.set_ylim([-150, 150])
        ax_TFR_short.set_ylabel('TFR (current window', rotation = 0, labelpad=30, fontsize=8)

                   
        # Opening JSON file
        f = open(path_to_json_markers,)
         
        # returns JSON object as a dictionary
        markers = json.load(f)
        markers_details = list(markers.values())
        
        self.markers_details = markers_details
        self.marker_keys = list(markers.keys())

        self.counter_markers = 0
        self.palette = itertools.cycle(sns.color_palette())

        for counter, marker in enumerate(markers.keys()):
                
            if marker.split()[0] == 'MARKER':
                if 'manual' in markers_to_show:
                    self.counter_markers = self.counter_markers + 1
                    self.color_markers = next(self.palette)
                    marker_loc = int(marker.split()[-1])
                    ax1.plot([marker_loc/256, marker_loc/256], [fmin, fmax] ,  label =  str(self.counter_markers)+'. '+markers_details[counter], linewidth = 3, color = self.color_markers)
                    ax1.set_ylim([fmin, fmax])
                    ax_epoch_marker.plot([marker_loc/256, marker_loc/256], [fmin, fmax] ,  label =  markers_details[counter], linewidth = 3, color = self.color_markers)
                    
                    ax_epoch_marker.text(marker_loc/256+.1, int((fmin+fmax)/2), str(self.counter_markers ), verticalalignment='center', color = self.color_markers)

            if marker.split()[0] == 'SOUND':
                if 'sound' in markers_to_show:
                    marker_loc = int(marker.split()[-1])
                    ax2.plot([marker_loc/256, marker_loc/256], [fmin, fmax] ,  label =  'Audio: '+markers_details[counter].split('/')[-1], linewidth = 3, color = 'blue')
                    ax2.set_ylim([fmin, fmax])
                    ax_epoch_light.plot([marker_loc/256, marker_loc/256], [fmin, fmax] ,  label = 'Audio: '+ markers_details[counter].split('/')[-1], linewidth = 3, color = 'blue')
                    ax_epoch_light.text(marker_loc/256+.1, int((fmin+fmax)/2),'Audio', verticalalignment='center', color = 'blue')

            elif marker.split()[0] == 'LIGHT':
                if 'light' in markers_to_show:
                    marker_loc = int(marker.split()[-1])
                    if 'Vib: False' in markers_details[counter]:
                        ax2.plot([marker_loc/256, marker_loc/256],  [fmin, fmax] , label =  markers_details[counter].split(',')[0],color = 'red', linewidth = 3)
                        ax2.set_ylim([fmin, fmax])
                        ax_epoch_light.plot([marker_loc/256, marker_loc/256],  [fmin, fmax] , label =  markers_details[counter].split(',')[0],color = 'red', linewidth = 3)
                        ax_epoch_light.text(marker_loc/256+.1, int((fmin+fmax)/2),'Light', verticalalignment='center', color = 'red')

                    elif 'Vib: True' in markers_details[counter]:
                        ax2.plot([marker_loc/256, marker_loc/256],  [fmin, fmax] , label =  markers_details[counter].split(',')[0],color = 'green', linewidth = 3)
                        ax2.set_ylim([fmin, fmax])
                        ax_epoch_light.plot([marker_loc/256, marker_loc/256],  [fmin, fmax] , label =  markers_details[counter].split(',')[0],color = 'green', linewidth = 3)
                        ax_epoch_light.text(marker_loc/256+.1, int((fmin+fmax)/2),'Vibration', verticalalignment='center', color = 'green')
                    
        ax1.set_yticks([])
        ax2.set_yticks([])
        ax_epoch_marker.set_yticks([])
        ax_epoch_light.set_yticks([])
        #ax_epoch_marker.set_xticks([])
        #ax_epoch_light.set_xticks([])
    
        ax2.set_ylabel('Stimulation(overall)', rotation = 0, labelpad=30, fontsize=8)#.set_rotation(0)
        ax1.set_ylabel('Markers(overall)', rotation = 0, labelpad=30, fontsize=8)#.set_rotation(0)
        ax_epoch_marker.set_ylabel('Markers(epoch)', rotation = 0, labelpad=30, fontsize=8)
        ax_epoch_light.set_ylabel('Stimulation(epoch)', rotation = 0, labelpad=30,fontsize=8)
        ax_acc.set_ylabel('Acceleration', rotation = 0, labelpad=30, fontsize=8)
        ax_noise.set_ylabel('Noise', rotation = 0, labelpad=30, fontsize=8)
        ax_EMG.set_ylabel('EMG1 (uV)',rotation = 0, labelpad=30, fontsize=8)
        ax_EMG2.set_ylabel('EMG2 (uV)',rotation = 0, labelpad=30, fontsize=8)
        ax_EMG3.set_ylabel(' EMG1 - EMG2 (uV) ',rotation = 0, labelpad=30, fontsize=8)
        
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
# =============================================================================
#         ax_epoch_light.spines[["left", 'right']].set_visible(False)
#         ax_epoch_marker.spines[["left", 'right']].set_visible(False)
#         ax_acc.spines[["top", "bottom"]].set_visible(False)
#         ax_noise.spines[["top", "bottom"]].set_visible(False)
# =============================================================================
        
        time_axis = np.round(np.arange(0, len(data)) / 256 , 2)
    
        ax4.set_xlabel('time (s)')
        #ax4.set_xticks(np.arange(len(data)), time_axis)
        ax4.set_xlim([0, 30])#len(data)])
        leg = ax1.legend(fontsize=9, bbox_to_anchor=(1, 0), loc = 'upper left')
        leg.set_draggable(state=True)
        
        # plot acc
        ax_acc.plot(np.arange(len(self.acc_x[self.samples_before_begin:]))/256, self.acc_x[self.samples_before_begin:], linewidth = 2 , color = 'blue')
        ax_acc.plot(np.arange(len(self.acc_y[self.samples_before_begin:]))/256, self.acc_y[self.samples_before_begin:], linewidth = 2, color = 'red')
        ax_acc.plot(np.arange(len(self.acc_z[self.samples_before_begin:]))/256, self.acc_z[self.samples_before_begin:], linewidth = 2, color = 'green')
        
        # plot noise
        ax_noise.plot(np.arange(len(self.noise_data[self.samples_before_begin:]))/256, self.noise_data[self.samples_before_begin:], linewidth = 2, color = 'black')
        
        #ax4.get_xaxis().set_visible(False)
        
        fig.canvas.mpl_connect('key_press_event', self.pan_nav)
        fig.canvas.mpl_connect('button_press_event', self.onclick)
        #ax1.set_xlim([0, 7680])#len(data)])
        #ax2.set_xlim([0, 7680])#len(data)])
        #ax3.set_xlim([0, len(data)])
        #ax4.set_xlim([0, 7680])#len(data)])
        
    
                
        # read EMG
        self.EMG_raw = mne.io.read_raw_brainvision(self.path_to_EMG)
        self.EMG_raw = self.EMG_raw.resample(int(256))
        self.EMG_filtered = self.EMG_raw.filter(l_freq=10, h_freq=100)
        self.EMG_filtered_data1 = self.EMG_filtered.get_data()[0] 
        self.EMG_filtered_data2 = self.EMG_filtered.get_data()[1] 
        self.EMG_filtered_data1_minus_2 = self.EMG_filtered_data1 - self.EMG_filtered_data2
        
        self.desired_EMG_scale_val = int(self.EMG_scale_options_val.get())
        self.desired_EMG_scale= [-1e-6* self.desired_EMG_scale_val, 1e-6* self.desired_EMG_scale_val]
        
        # Check whether the user already synced EMG vs. EEG or not
        print(f'shape EMG signals = {np.shape(self.EMG_filtered_data1)}')
        print(f'sync samples: {str(self.samples_before_begin_EMG_Dreamento)} with shape {str(np.shape(self.samples_before_begin_EMG_Dreamento))}')
        print(f'sync criterion: {self.flag_sign_samples_before_begin_EMG_Dreamento}')
        if self.flag_sync_EEG_EMG == True:
            
            if self.flag_sign_samples_before_begin_EMG_Dreamento == 'eeg_event_earlier':
                print('Detected that the event occured earlier in EEG than the EMG signal')
                self.EMG_filtered_data1 = self.EMG_filtered_data1[self.samples_before_begin_EMG_Dreamento:]
                self.EMG_filtered_data2 = self.EMG_filtered_data2[self.samples_before_begin_EMG_Dreamento:]
                self.EMG_filtered_data1_minus_2 = self.EMG_filtered_data1_minus_2[self.samples_before_begin_EMG_Dreamento:]
                
            elif self.flag_sign_samples_before_begin_EMG_Dreamento == 'emg_event_earlier':
                print('Detected that the event occured earlier in EMG than the EEG signal')
                print(f'EMG shape before alignment is : {np.shape(self.EMG_filtered_data1)}')
                self.EMG_filtered_data1 = np.append(self.samples_before_begin_EMG_Dreamento, self.EMG_filtered_data1)
                self.EMG_filtered_data2 = np.append(self.samples_before_begin_EMG_Dreamento, self.EMG_filtered_data2)
                self.EMG_filtered_data1_minus_2 = np.append(self.samples_before_begin_EMG_Dreamento, self.EMG_filtered_data1_minus_2)
                print(f'EMG shape after alignment is : {np.shape(self.EMG_filtered_data1)}')
        
        ax_EMG.plot(np.arange(len(self.EMG_filtered_data1))/256, self.EMG_filtered_data1, color = (84/255,164/255,75/255))
        ax_EMG2.plot(np.arange(len(self.EMG_filtered_data2))/256, self.EMG_filtered_data2, color = (24/255,100/255,160/255))
        ax_EMG3.plot(np.arange(len(self.EMG_filtered_data1_minus_2))/256, self.EMG_filtered_data1_minus_2, color = (160/255,10/255,22/255))
        ax_EMG.set_ylim(self.desired_EMG_scale)
        ax_EMG2.set_ylim(self.desired_EMG_scale)
        ax_EMG3.set_ylim(self.desired_EMG_scale)
        ax_EMG.grid(True)
        ax_EMG2.grid(True)
        ax_EMG3.grid(True)
        ax_EMG.set_yticks([])
        ax_EMG3.set_yticks([])
        ax_EMG2.set_yticks((self.desired_EMG_scale[0], 0, self.desired_EMG_scale[1]))
        plt.show()
# =============================================================================
#         ax_EMG.set_xticks([])
#         ax_EMG2.set_xticks([])
#         ax_EMG3.set_xticks([])
# =============================================================================
        return fig, markers_details
    
    #%% AssignMarkersToRecordedData EEG + TFR
    def AssignMarkersToRecordedData_EEG_TFR_noEMG(self, data, data_R, sf, path_to_json_markers, markers_to_show = ['light', 'manual', 'sound'],\
                                win_sec=30, fmin=0.3, fmax=40,
                                trimperc=5, cmap='RdBu_r', add_colorbar = False):
        """
        The main function: create the main display window of Offline Dreamento
        
        :param self: access the attributes and methods of the class
        :param data: EEG L data
        :param data_R: EEG R data
        :param sf: sampling frequency
        :param path_to_json_markers: path to the annotation file (generated by Dreamento)
        :param markers_to_show: the desired type of markers to show ['light', 'manual', 'sound']
        :param win_sec: the window size to compute spectrogram
        :param fmin: relavant min frequency for spectrogram
        :param fmax: relavant max frequency for spectrogram
        :param trimperc: bidirectional trim factor for normalizing the spectrogram colormap
        :param cmap: colormap of the spectrogram
        :param add_colorbar: adding colorbar to spectrogram

        :returns: fig, markers_details
        """
        
# =============================================================================
#         Source code for plotting spectrogram: YASA (https://raphaelvallat.com/yasa/build/html/_modules/yasa/plotting.html#plot_spectrogram)
#         [modified]                                           	
# =============================================================================
        # Increase font size while preserving original
        old_fontsize = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': 12})
        
        # Safety checks
        assert isinstance(data, np.ndarray), 'Data must be a 1D NumPy array.'
        assert isinstance(sf, (int, float)), 'sf must be int or float.'
        assert data.ndim == 1, 'Data must be a 1D (single-channel) NumPy array.'
        assert isinstance(win_sec, (int, float)), 'win_sec must be int or float.'
        assert isinstance(fmin, (int, float)), 'fmin must be int or float.'
        assert isinstance(fmax, (int, float)), 'fmax must be int or float.'
        assert fmin < fmax, 'fmin must be strictly inferior to fmax.'
        assert fmax < sf / 2, 'fmax must be less than Nyquist (sf / 2).'
    
        # Calculate multi-taper spectrogram
        nperseg = int(10 * sf) #int(win_sec * sf / 8)
        assert data.size > 2 * nperseg, 'Data length must be at least 2 * win_sec.'
        f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
        Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz
    
        # Select only relevant frequencies (up to 30 Hz)
        good_freqs = np.logical_and(f >= fmin, f <= fmax)
        Sxx = Sxx[good_freqs, :]
        f = f[good_freqs]
        #t *= 256  # Convert t to hours
    
        # Normalization
        vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        #!!!!!!!!!! SHORT SPECTROGRAM!! Calculate multi-taper spectrogram
        nperseg2 = int(.2 * sf) #int(win_sec * sf / 8)
        assert data.size > 2 * nperseg2, 'Data length must be at least 2 * win_sec.'
        f2, t2, Sxx2 = spectrogram_lspopt(data, sf, nperseg=nperseg2, noverlap=0)
        Sxx2 = 10 * np.log10(Sxx2)  # Convert uV^2 / Hz --> dB / Hz
        print(f'f: {np.shape(f)}, f2: {np.shape(f2)}, t: {np.shape(t)}, t2: {np.shape(t2)}, Sxx: {np.shape(Sxx)}, Sxx2: {np.shape(Sxx2)}')
        # Select only relevant frequencies (up to 30 Hz)
        good_freqs2 = np.logical_and(f2 >= fmin, f2 <= fmax)
        Sxx2 = Sxx2[good_freqs2, :]
        f2 = f2[good_freqs2]
        #t *= 256  # Convert t to hours
    
        # Normalization
        vmin2, vmax2 = np.percentile(Sxx2, [0 + trimperc, 100 - trimperc])
        norm2 = Normalize(vmin=vmin2, vmax=vmax2)
    # =============================================================================
    #     gs1 = gridspec.GridSpec(2, 1)
    #     gs1.update(wspace=0.005, hspace=0.0001)
    # =============================================================================
        fig,AX = plt.subplots(nrows=9, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 10,1,1, 2, 2, 4, 10]})
        
        ax1 = plt.subplot(9,1,1)
        ax2 = plt.subplot(9,1,2, sharex = ax1)
        ax3 = plt.subplot(9,1,3, sharex = ax1)
        
        ax_epoch_marker = plt.subplot(9,1,4, )
        ax_epoch_light = plt.subplot(9,1,5, sharex = ax_epoch_marker)
        
        ax_acc = plt.subplot(9,1,6, sharex = ax_epoch_marker)
        ax_noise = plt.subplot(9,1,7, sharex = ax_epoch_marker)
        

        ax_TFR_short = plt.subplot(9,1,8, sharex = ax_epoch_marker)
        ax4 = plt.subplot(9,1,9, sharex = ax_epoch_marker)
        ax4.grid(True)

        ax1.get_xaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax_epoch_marker.get_xaxis().set_visible(False)
        ax_epoch_light.get_xaxis().set_visible(False)
        ax_acc.get_xaxis().set_visible(False)
        ax_acc.set_yticks([])
        ax_noise.set_yticks([])
        ax_noise.get_xaxis().set_visible(False)
        
        ax_acc.set_ylim([-1.4, 1.4])
        
        plt.subplots_adjust(hspace = 0)
        ax1.set_title('Dreamento: post-processing ')
        im = ax3.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True,
                           shading="auto")
        ax3.set_xlim([0, len(data)/256])
        ax3.set_ylim((fmin, 25))
        ax3.set_ylabel('Frequency [Hz]')
        
        im2 = ax_TFR_short.pcolormesh(t2, f2, Sxx2, norm=norm2, cmap=cmap, antialiased=True,
                           shading="auto")
        
        # Add colorbar
        if add_colorbar == True:
            cbar = fig.colorbar(im, ax=ax3, shrink=0.95, fraction=0.1, aspect=25, pad=0.01)
            cbar.ax3.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=5)
            
        # PLOT EEG
        ax4.plot(np.arange(len(data))/256, data, color = (160/255, 70/255, 160/255), linewidth = 1)
        ax4.plot(np.arange(len(data_R))/256, data_R, color = (0/255, 128/255, 190/255), linewidth = 1)
        #axes[1].set_ylim([-200, 200])
        #ax4.set_xlim([0, len(data)])
        ax4.set_ylabel('EEG (uV)')
        ax4.set_ylim([-150, 150])
        ax_TFR_short.set_ylabel('TFR (current window', rotation = 0, labelpad=30, fontsize=8)

                   
        # Opening JSON file
        f = open(path_to_json_markers,)
         
        # returns JSON object as a dictionary
        markers = json.load(f)
        markers_details = list(markers.values())
        
        self.markers_details = markers_details
        self.marker_keys = list(markers.keys())

        self.counter_markers = 0
        self.palette = itertools.cycle(sns.color_palette())

        for counter, marker in enumerate(markers.keys()):
                
            if marker.split()[0] == 'MARKER':
                if 'manual' in markers_to_show:
                    self.counter_markers = self.counter_markers + 1
                    self.color_markers = next(self.palette)
                    marker_loc = int(marker.split()[-1])
                    ax1.plot([marker_loc/256, marker_loc/256], [fmin, fmax] ,  label =  str(self.counter_markers)+'. '+markers_details[counter], linewidth = 3, color = self.color_markers)
                    ax1.set_ylim([fmin, fmax])
                    ax_epoch_marker.plot([marker_loc/256, marker_loc/256], [fmin, fmax] ,  label =  markers_details[counter], linewidth = 3, color = self.color_markers)
                    
                    ax_epoch_marker.text(marker_loc/256+.1, int((fmin+fmax)/2), str(self.counter_markers ), verticalalignment='center', color = self.color_markers)

            if marker.split()[0] == 'SOUND':
                if 'sound' in markers_to_show:
                    marker_loc = int(marker.split()[-1])
                    ax2.plot([marker_loc/256, marker_loc/256], [fmin, fmax] ,  label =  'Audio: '+markers_details[counter].split('/')[-1], linewidth = 3, color = 'blue')
                    ax2.set_ylim([fmin, fmax])
                    ax_epoch_light.plot([marker_loc/256, marker_loc/256], [fmin, fmax] ,  label = 'Audio: '+ markers_details[counter].split('/')[-1], linewidth = 3, color = 'blue')
                    ax_epoch_light.text(marker_loc/256+.1, int((fmin+fmax)/2),'Audio', verticalalignment='center', color = 'blue')

            elif marker.split()[0] == 'LIGHT':
                if 'light' in markers_to_show:
                    marker_loc = int(marker.split()[-1])
                    if 'Vib: False' in markers_details[counter]:
                        ax2.plot([marker_loc/256, marker_loc/256],  [fmin, fmax] , label =  markers_details[counter].split(',')[0],color = 'red', linewidth = 3)
                        ax2.set_ylim([fmin, fmax])
                        ax_epoch_light.plot([marker_loc/256, marker_loc/256],  [fmin, fmax] , label =  markers_details[counter].split(',')[0],color = 'red', linewidth = 3)
                        ax_epoch_light.text(marker_loc/256+.1, int((fmin+fmax)/2),'Light', verticalalignment='center', color = 'red')

                    elif 'Vib: True' in markers_details[counter]:
                        ax2.plot([marker_loc/256, marker_loc/256],  [fmin, fmax] , label =  markers_details[counter].split(',')[0],color = 'green', linewidth = 3)
                        ax2.set_ylim([fmin, fmax])
                        ax_epoch_light.plot([marker_loc/256, marker_loc/256],  [fmin, fmax] , label =  markers_details[counter].split(',')[0],color = 'green', linewidth = 3)
                        ax_epoch_light.text(marker_loc/256+.1, int((fmin+fmax)/2),'Vibration', verticalalignment='center', color = 'green')
                    
        ax1.set_yticks([])
        ax2.set_yticks([])
        ax_epoch_marker.set_yticks([])
        ax_epoch_light.set_yticks([])
        #ax_epoch_marker.set_xticks([])
        #ax_epoch_light.set_xticks([])
    
        ax2.set_ylabel('Stimulation(overall)', rotation = 0, labelpad=30, fontsize=8)#.set_rotation(0)
        ax1.set_ylabel('Markers(overall)', rotation = 0, labelpad=30, fontsize=8)#.set_rotation(0)
        ax_epoch_marker.set_ylabel('Markers(epoch)', rotation = 0, labelpad=30, fontsize=8)
        ax_epoch_light.set_ylabel('Stimulation(epoch)', rotation = 0, labelpad=30,fontsize=8)
        ax_acc.set_ylabel('Acceleration', rotation = 0, labelpad=30, fontsize=8)
        ax_noise.set_ylabel('Noise', rotation = 0, labelpad=30, fontsize=8)
        
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["left"].set_visible(False)
        ax_epoch_light.spines[["left", 'right']].set_visible(False)
        ax_epoch_marker.spines[["left", 'right']].set_visible(False)
        ax_acc.spines[["top", "bottom"]].set_visible(False)
        ax_noise.spines[["top", "bottom"]].set_visible(False)
        
        time_axis = np.round(np.arange(0, len(data)) / 256 , 2)
    
        ax4.set_xlabel('time (s)')
        #ax4.set_xticks(np.arange(len(data)), time_axis)
        ax4.set_xlim([0, 30])#len(data)])
        leg = ax1.legend(fontsize=9, bbox_to_anchor=(1, 0), loc = 'upper left')
        leg.set_draggable(state=True)
        
        # plot acc
        ax_acc.plot(np.arange(len(self.acc_x[self.samples_before_begin:]))/256, self.acc_x[self.samples_before_begin:], linewidth = 2 , color = 'blue')
        ax_acc.plot(np.arange(len(self.acc_y[self.samples_before_begin:]))/256, self.acc_y[self.samples_before_begin:], linewidth = 2, color = 'red')
        ax_acc.plot(np.arange(len(self.acc_z[self.samples_before_begin:]))/256, self.acc_z[self.samples_before_begin:], linewidth = 2, color = 'green')
        
        # plot noise
        ax_noise.plot(np.arange(len(self.noise_data[self.samples_before_begin:]))/256, self.noise_data[self.samples_before_begin:], linewidth = 2, color = 'black')
        
        #ax4.get_xaxis().set_visible(False)
        
        fig.canvas.mpl_connect('key_press_event', self.pan_nav_noEMG)
        fig.canvas.mpl_connect('button_press_event', self.onclick_noEMG)
        #ax1.set_xlim([0, 7680])#len(data)])
        #ax2.set_xlim([0, 7680])#len(data)])
        #ax3.set_xlim([0, len(data)])
        #ax4.set_xlim([0, 7680])#len(data)])
        
        return fig, markers_details
    
    #%% Function: Help pop-up
    def help_pop_up_func(self):
        """
        Help button of the software. Introduction to the applications and hot keeys
        
        :param self: access the attributes and methods of the class
        """
        
        line1 = "Welcome to Dreamento!\n"
        line2 = "Here you can sync the recordings from hypnodyne and Dreamento!\n"
        line3 = "** Notes:\n- First load the *EEG L* from Hypnodyne recording (.edf format).\n"
        line4 = "- Then load Dreamento output (.txt format).\n"
        line5 = "- Afterwards load the marker file (.json).\n"
        line5_5 = "- And eventually the .vhdr file from EMG recording (optional).\n"
        line_5_75 = "If there is no EMG, uncheck the PLOT EMG checkbox.\n"
        
        line6 = "- Hotkeys to navigate in the final plot.\n"

        line7 = "- Keyboard down arrow: zoom out.\n"
        line8 = "- Keyboard up arrow: zoom in.\n"
        line9 = "- Keyboard right arrow: next epoch (30s).\n"
        line10 = "- Keyboard left arrow: previous epoch (30s).\n\n"
        line11 = "Markers and light stimulations of the whole data are shown on first two rows.\n"
        line12 = "However, current-epoch markers are shown in the middle.\n"
        line13 = "IMPORTANT: To keep the location navigator accurate, please keep the cursur on\n" 
        line14 = "the spectrogrma plot while pressing left/right buttons to navigate.\n\n"
        
        line15 = "Matlab compatability: After the analysis you can export an .mat file\n\n"
        line16 = "*** CopyRight (2021-2022): Mahdad Jafarzadeh Esfahani, Amir Hossein Daraie ***"
      
        lines = line1 + line2 + line3 + line4 + line5 + line5_5+ line6 + line7+ line8+ line9+ line10+ line11 + line12 + line13 + line14+ line15 + line16
        messagebox.showinfo(title = "Help", message = lines)
    
    #%% Navigating via keyboard in the figure
    def pan_nav(self, event):
        """
        The keyboard controller of the software
        
        :param self: accessing  the attributes and methods of the class
        :param up arrow keyboard button: increase the EEG amplitude scale
        :param down arrow keyboard button: lower the EEG amplitude scale
        :param left arrow keyboard button: navigate to the previous epoch
        :param right arrow keyboard button: navigate to the next epoch

        """
        ax_tmp = plt.gca()
        if event.key == 'left':
            lims = ax_tmp.get_xlim()
            adjust = (lims[1] - lims[0]) 
            ax_tmp.set_xlim((lims[0] - adjust, lims[1] - adjust))
            curr_ax = event.inaxes
            if str(curr_ax) == 'AxesSubplot(0.125,0.67;0.775x0.175)':
                print('spectrogram axis detected')
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([int(np.mean((lims[0] - adjust, lims[1] - adjust))), int(np.mean((lims[0] - adjust, lims[1] - adjust)))], [-150, 150], color = 'black')
                #curr_ax.set_ylim((0.1,25))
            
            plt.draw()
        elif event.key == 'right':
            lims = ax_tmp.get_xlim()
            adjust = (lims[1] - lims[0]) 
            ax_tmp.set_xlim((lims[0] + adjust, lims[1] + adjust))
            print(event.xdata)
            print(lims)
            #ax3.axvline(x=event.xdata, color="k")
            plt.draw()
            print(f'The xdata is : {event.xdata}')
            print(f'The ydata is : {event.ydata}')
            
            print(f'The x is : {event.x}')
            print(f'The y is : {event.y}')
            
            print(f'favailable axes: {event.inaxes}')
            
            curr_ax = event.inaxes
            if str(curr_ax) == 'AxesSubplot(0.125,0.67;0.775x0.175)':
                print('spectrogram axis detected')
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([int(np.mean((lims[0] + adjust, lims[1] + adjust))), int(np.mean((lims[0] + adjust, lims[1] + adjust)))], [-150, 150], color = 'black')
                #curr_ax.set_ylim((0.1,25))
        elif event.key == 'up':
            lims = ax_tmp.get_ylim()
            adjust_up = lims[1] - lims[1]/5
            adjust_down = lims[0] +lims[1]/5
            ax_tmp.set_ylim((adjust_down, adjust_up))
            plt.draw()
            
        elif event.key == 'down':
            lims = ax_tmp.get_ylim()
            adjust_up = lims[1] + lims[1]/5
            adjust_down = lims[0] - lims[1]/5
            ax_tmp.set_ylim((adjust_down, adjust_up))
            plt.draw()
    
    #%% Define event while clicking
    
    def onclick(self, event):
        """
        Clicking on the TFR to go to the desired epoch.
        
        :param self: access the attributes and methods of the class
        :param event: mouse click
        """
        ax_tmp = plt.gca()
        if event.button == 1: 

            print('mouse cliked --> move plot')
            ax_tmp.set_xlim((np.floor(event.xdata)- int(7680/256/2), np.floor(event.xdata)+ int(7680/256/2)))
            plt.draw()
            print(f'clicked sample{ {event.xdata}}')
            print(f'adjust xlm {(np.floor(event.xdata)- int(7680/2), np.floor(event.xdata)+ int(7680/2))}')
            print(f'{event.inaxes}')
            curr_ax = event.inaxes
            if str(curr_ax) == 'AxesSubplot(0.125,0.67;0.775x0.175)':
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([event.xdata, event.xdata], [0.3, 40], color = 'black')
                curr_ax.set_ylim((0.1,25))
                
                
    #%% Navigating via keyboard in the figure
    def pan_nav_noEMG(self, event):
        """
        The keyboard controller of the software
        
        :param self: accessing  the attributes and methods of the class
        :param up arrow keyboard button: increase the EEG amplitude scale
        :param down arrow keyboard button: lower the EEG amplitude scale
        :param left arrow keyboard button: navigate to the previous epoch
        :param right arrow keyboard button: navigate to the next epoch

        """
        ax_tmp = plt.gca()
        if event.key == 'left':
            lims = ax_tmp.get_xlim()
            adjust = (lims[1] - lims[0]) 
            ax_tmp.set_xlim((lims[0] - adjust, lims[1] - adjust))
            curr_ax = event.inaxes
            if str(curr_ax) == 'AxesSubplot(0.125,0.59125;0.775x0.240625)':
                print('spectrogram axis detected')
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([int(np.mean((lims[0] - adjust, lims[1] - adjust))), int(np.mean((lims[0] - adjust, lims[1] - adjust)))], [-150, 150], color = 'black')
                #curr_ax.set_ylim((0.1,25))
            
            plt.draw()
        elif event.key == 'right':
            lims = ax_tmp.get_xlim()
            adjust = (lims[1] - lims[0]) 
            ax_tmp.set_xlim((lims[0] + adjust, lims[1] + adjust))
            print(event.xdata)
            print(lims)
            #ax3.axvline(x=event.xdata, color="k")
            plt.draw()
            print(f'The xdata is : {event.xdata}')
            print(f'The ydata is : {event.ydata}')
            
            print(f'The x is : {event.x}')
            print(f'The y is : {event.y}')
            
            print(f'favailable axes: {event.inaxes}')
            
            curr_ax = event.inaxes
            if str(curr_ax) == 'AxesSubplot(0.125,0.59125;0.775x0.240625)':
                print('spectrogram axis detected')
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([int(np.mean((lims[0] + adjust, lims[1] + adjust))), int(np.mean((lims[0] + adjust, lims[1] + adjust)))], [-150, 150], color = 'black')
                #curr_ax.set_ylim((0.1,25))
        elif event.key == 'up':
            lims = ax_tmp.get_ylim()
            adjust_up = lims[1] - lims[1]/5
            adjust_down = lims[0] +lims[1]/5
            ax_tmp.set_ylim((adjust_down, adjust_up))
            plt.draw()
            
        elif event.key == 'down':
            lims = ax_tmp.get_ylim()
            adjust_up = lims[1] + lims[1]/5
            adjust_down = lims[0] - lims[1]/5
            ax_tmp.set_ylim((adjust_down, adjust_up))
            plt.draw()
    
    #%% Define event while clicking
    
    def onclick_noEMG(self, event):
        """
        Clicking on the TFR to go to the desired epoch.
        
        :param self: access the attributes and methods of the class
        :param event: mouse click
        """
        ax_tmp = plt.gca()
        if event.button == 1: 

            print('mouse cliked --> move plot')
            ax_tmp.set_xlim((np.floor(event.xdata)- int(7680/256/2), np.floor(event.xdata)+ int(7680/256/2)))
            plt.draw()
            print(f'clicked sample{ {event.xdata}}')
            print(f'adjust xlm {(np.floor(event.xdata)- int(7680/2), np.floor(event.xdata)+ int(7680/2))}')
            print(f'{event.inaxes}')
            curr_ax = event.inaxes
            if str(curr_ax) == 'AxesSubplot(0.125,0.59125;0.775x0.240625)':
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([event.xdata, event.xdata], [0.3, 40], color = 'black')
                curr_ax.set_ylim((0.1,25))
if __name__ == "__main__":

    #%% Test section
    root = Tk()
    my_gui = OfflineDreamento(root)
    #root.iconphoto(False, PhotoImage(file=".\\Donders_Logo.png"))
    root.mainloop()
