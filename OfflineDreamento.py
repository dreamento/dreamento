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
import os
import threading
import yasa

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
        self.label_CopyRight.grid(row = 1 , column = 0, padx = 10, pady = 10)
        
        #### ==================== Import Hypnodyne data  ========================####
        # Label: Import EDF
        self.label_Hypnodyne = Label(self.frame_import, text = "Import Hypnodyne EEG L.edf file:",
                                  font = 'Calibri 13 ')
        self.label_Hypnodyne.grid(row = 0 , column = 0, padx = 10, pady = 10)
        
        # Button: Import EDF (Browse)
        self.button_Hypnodyne_browse = Button(self.frame_import, text = "Browse Hypnodyne",
                                           padx = 40, pady = 10,font = 'Calibri 12 ',
                                           command = self.load_hypnodyne_file_dialog, fg = 'blue',
                                           relief = RIDGE)
        self.button_Hypnodyne_browse.grid(row = 1, column = 0, padx = 10, pady = 10)
        
        #### ================== Import Dreamento file ====================####
        # Show a message about hypnograms
        self.label_Dreamento = Label(self.frame_import, text = "Import Dreamento output file (.txt):",
                                  font = 'Calibri 13 ')
        self.label_Dreamento.grid(row = 0 , column = 1, padx = 10, pady = 10)
        
        # Define browse button to import hypnos
        self.button_Dreamento_browse = Button(self.frame_import, text = "Browse Dreamento", 
                                           padx = 40, pady = 10, font = 'Calibri 12 ',
                                           command = self.load_Dreamento_file_dialog,fg = 'blue',
                                           relief = RIDGE)
        self.button_Dreamento_browse.grid(row = 1, column = 1, padx = 10, pady = 10)
        
        #### ================== Import markers Json file ====================####
        # Show a message about hypnograms
        self.label_marker_json = Label(self.frame_import, text = "Import marker file (.json):",
                                  font = 'Calibri 13 ')
        self.label_marker_json.grid(row = 0 , column = 2, padx = 10, pady = 10)
        
        # Define browse button to import hypnos
        self.button_marker_json_browse = Button(self.frame_import, text = "Browse markers", 
                                           padx = 40, pady = 10, font = 'Calibri 12 ',
                                           command = self.load_marker_file_dialog,fg = 'blue',
                                           relief = RIDGE)
        self.button_marker_json_browse.grid(row = 1, column = 2, padx = 10, pady = 10)
        
       
        #### ================== Import Brainvision EMG Json file ====================####
        # Show a message about hypnograms
        self.label_EMG = Label(self.frame_import, text = "Import EMG file (.vhdr):",
                                  font = 'Calibri 13 ')
        self.label_EMG.grid(row = 0 , column = 3, padx = 10, pady = 10)
        
        # Define browse button to import hypnos
        self.button_EMG_browse = Button(self.frame_import, text = "Browse EMG (.vhdr)", 
                                           padx = 40, pady = 10, font = 'Calibri 12 ',
                                           command = self.load_EMG_file_dialog,fg = 'blue',
                                           relief = RIDGE)
        self.button_EMG_browse.grid(row = 1, column = 3, padx = 10, pady = 10)
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
        self.button_apply.grid(row = 3 , column =2, padx = 10, pady = 10)


    #%% =========================== Options for analysis =================== #%%
        #Label to read data and extract features
        self.label_analysis = Label(self.frame_import, text = "Analysis options:",
                                      font = 'Calibri 13 ')
        self.label_analysis.grid(row = 0 , column = 4, sticky="w")
        
    #%% init a label to give warning
    #%% Bulk autoscoring
        self.bulk_autoscoring_val = IntVar(value = 0)
        self.checkbox_bulk_autoscoring = Checkbutton(self.frame_import, text = "Bulk autoscoring",
                                  font = 'Calibri 11 ', variable = self.bulk_autoscoring_val, 
                                  command = self.bulk_autoscoring_popup)
        
        self.checkbox_bulk_autoscoring.grid(row = 1, column = 4, sticky="w")
    #%% Checkbox for autoscoring
        self.is_autoscoring = IntVar(value = 0)
        self.checkbox_is_autoscoring = Checkbutton(self.frame_import, text = "Single-file autoscoring",
                                  font = 'Calibri 11 ', variable = self.is_autoscoring)#, command = self.scoring_caution)
        
        self.checkbox_is_autoscoring.grid(row = 2, column = 4, sticky="w") 
        
    #%% Checkbox for filtering
        self.is_filtering = IntVar(value = 1)
        self.checkbox_is_filtering = Checkbutton(self.frame_import, text = "Band-pass filtering (.3-30 Hz)",
                                  font = 'Calibri 11 ', variable = self.is_filtering)
        
        self.checkbox_is_filtering.grid(row = 3, column = 4, sticky="w")
        

    #%% Checkbox for plotting syncing process
        self.plot_sync_output = IntVar()
        self.checkbox_plot_sync_output = Checkbutton(self.frame_import, text = "Plot EEG alignment process",
                                  font = 'Calibri 11 ', variable = self.plot_sync_output)
        
        self.checkbox_plot_sync_output.grid(row = 4, column = 4, sticky="w")
    
    #%% Checkbox for plotting EMG quality TFR
        self.plot_EMG_quality_evaluation = IntVar()
        self.checkbox_plot_EMG_quality_evaluation = Checkbutton(self.frame_import, text = "EMG quality evaluation",
                                  font = 'Calibri 11 ', variable = self.plot_EMG_quality_evaluation)
        
        self.checkbox_plot_EMG_quality_evaluation.grid(row = 5, column = 4, sticky="w")
        
    #%% Checkbox for plotting periodogram 
        self.plot_psd = IntVar(value = 0)
        self.checkbox_plot_psd = Checkbutton(self.frame_import, text = "Plot peridogram",
                                  font = 'Calibri 11 ', variable = self.plot_psd)
        
        self.checkbox_plot_psd.grid(row = 6, column = 4, sticky="w", pady = 10)
    #%% Label to select the desired analysis
        #Label to read data and extract features
        self.label_analysis_data = Label(self.frame_import, text = "Select the data to analyze:",
                                      font = 'Calibri 13 ')
        self.label_analysis_data.grid(row = 7 , column = 4, sticky="w")
        
    #%% Checkbox for plotting spectrograms with markers
        self.analysis_signal_options = IntVar(value = 1)
        self.checkbox_plot_additional_EMG = Radiobutton(self.frame_import, text = "Dreamento + HDRecorder + EMG",
                                  font = 'Calibri 11 ', variable = self.analysis_signal_options,\
                                  value = 1, command=self.analysis_signal_options_button_activator)
        
        self.checkbox_plot_additional_EMG.grid(row = 8, column = 4, sticky="w", pady = 10)
        
        
    #%% Checkbox for analyzing ZMax Hypndoyne only
        self.ZMax_Hypno_Dreamento = IntVar(value = 0)
        self.checkbox_ZMax_Hypno_Dreamento = Radiobutton(self.frame_import, text = "Dreamento + HDRecorder",
                                  font = 'Calibri 11 ', variable = self.analysis_signal_options, value = 2,\
                                  command=self.analysis_signal_options_button_activator)
        
        self.checkbox_ZMax_Hypno_Dreamento.grid(row = 9, column = 4, sticky="w", pady = 10)
    #%% Checkbox for analyzing ZMax Hypndoyne only
        self.ZMax_Hypno_only = IntVar(value = 0)
        self.checkbox_ZMax_Hypno_only = Radiobutton(self.frame_import, text = "HDRecorder",
                                  font = 'Calibri 11 ', variable = self.analysis_signal_options, value = 3,\
                                  command=self.analysis_signal_options_button_activator)
        
        self.checkbox_ZMax_Hypno_only.grid(row = 10, column = 4, sticky="w", pady = 10)
        
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
    def analysis_signal_options_button_activator(self):
        
        int_val = int(self.analysis_signal_options.get())
        
        if int_val == 1:
        # EMG load button
            self.button_EMG_browse['state'] = tk.NORMAL
            
        # EMG option menu 
            self.EMG_scale_option_menu['state'] = tk.NORMAL

        # Sync button
            self.button_sync_EMG['state'] = tk.NORMAL

        # Sync button
            self.button_apply['state'] = tk.DISABLED
            
        # EMG Evaluation
            self.checkbox_plot_EMG_quality_evaluation['state'] = tk.NORMAL
        elif int_val == 2:
            
            # EMG load button
            self.button_EMG_browse['state'] = tk.DISABLED
                
            # EMG option menu 
            self.EMG_scale_option_menu['state'] = tk.DISABLED

            # Sync button
            self.button_sync_EMG['state'] = tk.DISABLED

            # Sync button
            self.button_apply['state'] = tk.NORMAL
            
            # EMG Evaluation
            self.checkbox_plot_EMG_quality_evaluation['state'] = tk.DISABLED
            
        elif int_val == 3:
            
            # EMG load button
            self.button_EMG_browse['state'] = tk.DISABLED
                
            # EMG option menu 
            self.EMG_scale_option_menu['state'] = tk.DISABLED

            # Sync button
            self.button_sync_EMG['state'] = tk.DISABLED

            # Sync button
            self.button_apply['state'] = tk.NORMAL
            
            # Browse Dreamento button 
            self.button_Dreamento_browse['state'] = tk.DISABLED
            
            # Browse Markers browse button 
            self.button_marker_json_browse['state'] = tk.DISABLED
            
            # EMG Evaluation
            self.checkbox_plot_EMG_quality_evaluation['state'] = tk.DISABLED
        
    #%% Moving average filter    
    def MA(self,x, N):
    
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    #%% Def progress bar
    def progress_bar(self):
        from tkinter import ttk
        from time import sleep
        #start progress bar
        popup = tk.Toplevel()
        tk.Label(popup, text="Analysing in progress").grid(row=0,column=0)
        teams = range(30)

        progress = 0
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(popup, variable=progress_var, maximum=100)
        progress_bar.grid(row=1, column=0)#.pack(fill=tk.X, expand=1, side=tk.BOTTOM)
        popup.pack_slaves()
    
        progress_step = float(100.0/len(teams))
        for team in teams:
            popup.update()
            sleep(5) # lauch task
            progress += progress_step
            progress_var.set(progress)
    
        return 0
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
            
        elif 'EMG_files_list' not in globals() and int(self.analysis_signal_options.get()) == 1:
            messagebox.showerror("Dreamento", "Sorry, but no EMG files is loaded, though the plot EMG check box is activated! Change either of these and try again.")
            
        elif str(self.EMG_scale_options_val.get()) == 'Set desired EMG amplitude ...' and int(self.analysis_signal_options.get()) == 1:
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
        sync_event_is_selected = 0
        self.all_markers = []
        self.all_timestamp_markers = []
        self.all_markers_with_timestamps = []
        for counter, marker in enumerate(markers.values()):
            self.all_markers.append(marker)
            
        for counter, marker in enumerate(markers.keys()):
            self.all_timestamp_markers.append(marker)
        
        for i in np.arange(len(self.all_markers)):
            self.all_markers_with_timestamps.append(self.all_markers[i]+ '__'+ self.all_timestamp_markers[i])
        self.select_marker_for_sync()

        self.selected_marker = self.markers_sync_event.get()

        messagebox.showinfo('syncing in process', f'The syncing event has been selected. Please wait for sync window to pop up.')
        print(f'lets split the marker into ... {self.selected_marker.split()}')
        time_sync_event.append(int(self.selected_marker.split()[-1]))
# =============================================================================
#         for counter, marker in enumerate(markers.values()):
#             try:
#                 if marker.split()[0] == 'clench' or marker.split()[0] == 'Clench':
#                     print(marker.split())
#                     counter_sync.append(counter)
#                     
#             except:
#                 print(f'an exception occured for marker = {marker}')
#                 continue
#         print(counter_sync)
#         
#         for counter, marker in enumerate(markers.keys()):
#             if counter in counter_sync:
#                 time_sync_event.append(int(marker.split()[-1]))
# 
# =============================================================================
        print('Loading EEG file ... Please wait')                            
        path_Txt = self.ZmaxDondersRecording

        sigScript = np.loadtxt(path_Txt, delimiter=',')

        sigScript_org = sigScript
        sigScript_org_R = sigScript[:, 0]
        sigScript_org = sigScript_org[:, 1]

        # Read EMG
        print('Loading EEG file ... Please wait')                            
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
        EMG_data_get1 = EMG_data_get[0,:] * 1e6
        EMG_data_get2 = EMG_data_get[1,:] * 1e6
        EMG_data_get3 = EMG_data_get1 - EMG_data_get2
        
        #Filtering 
        sigScript_org   = self.butter_bandpass_filter(data = sigScript_org, lowcut=5, highcut=100, fs = 256, order = 2)
        EMG_data_get1   = self.butter_bandpass_filter(data = EMG_data_get1, lowcut=5, highcut=100, fs = 256, order = 2)
        EMG_data_get2   = self.butter_bandpass_filter(data = EMG_data_get2, lowcut=5, highcut=100, fs = 256, order = 2)
        EMG_data_get3   = self.butter_bandpass_filter(data = EMG_data_get3, lowcut=5, highcut=100, fs = 256, order = 2)
        t_sync          = np.arange(time_sync_event[0] - 256*5, time_sync_event[0] + 256*20)
        
        # Truncate sync period
        EEG_to_sync_period  = sigScript_org[t_sync]
        EMG_to_sync_period1 = EMG_data_get1[t_sync]
        EMG_to_sync_period2 = EMG_data_get2[t_sync]
        EMG_to_sync_period3 = EMG_data_get3[t_sync]
        
        # Rectified signal
        EEG = EEG_to_sync_period
        EEG_Abs = abs(EEG)
        
        EMG1 = EMG_to_sync_period1
        EMG2 = EMG_to_sync_period2
        EMG3 = EMG_to_sync_period3
        EMG_Abs1= abs(EMG1)
        EMG_Abs2= abs(EMG2)
        EMG_Abs3= abs(EMG3)
        
        MA_EEG  = self.MA(EEG_Abs, 512)
        
        MA_EMG1 = self.MA(EMG_Abs1, 512)
        MA_EMG2 = self.MA(EMG_Abs2, 512)
        MA_EMG3 = self.MA(EMG_Abs3, 512)
        
        fig, axs = plt.subplots(5, figsize = (10,10))
        axs[0].set_title('EEG during sync event')
        axs[0].plot(EEG_Abs, color = 'powderblue')
        
        axs[1].set_title('EMG 1 during sync event')
        axs[1].plot(EMG_Abs1, color = 'plum')
        
        axs[2].set_title('EMG 2 during sync event')
        axs[2].plot(EMG_Abs2, color = 'orchid')
        
        axs[3].set_title('EMG 1 - EMG 2 during sync event')
        axs[3].plot(EMG_Abs3, color = 'thistle')
        
        axs[4].plot(EEG_Abs, color = 'powderblue')
        axs[4].set_title('EMG vs EEG plotted on top of each other')
        axs[4].plot(EMG_Abs1, color = 'plum')
        axs[4].plot(EMG_Abs2, color = 'orchid')
        axs[4].plot(EMG_Abs3, color = 'thistle')
        
        plt.tight_layout()
        MsgBox = tk.messagebox.askquestion ('EEG vs EMG synchronization','Look at the data during sync period. Does the data require further synchronization (recommended to sync further)?',icon = 'warning')
        plt.show()
        if MsgBox == 'no':
            messagebox.showinfo("Information",f"OK! Now we proceed with the main analysis ... Please click on OK and wait ...")
            self.flag_sync_EEG_EMG = False
        if MsgBox == 'yes':
            self.flag_sync_EEG_EMG = True
            print('Proceeding to synchronization process ...')
            while True:
                MsgBox = tk.messagebox.askquestion ('Synchronization?','Proceed to automatic synchronization? For manual sync, press No.',icon = 'warning')
                if MsgBox == 'yes':
    
                    fig, axs = plt.subplots(6, figsize = (15,10))
                    axs[0].plot(EEG_Abs, color = 'powderblue')
                    axs[0].plot(MA_EEG, color = 'olive', linewidth=3)
                    axs[0].set_title('EEG', fontsize = 10)
                    
                    axs[1].plot(EMG_Abs1, color = 'plum')
                    axs[1].plot(MA_EMG1, color = 'purple', linewidth=3)
                    axs[1].set_title('EMG1' , fontsize = 10)
                    
                    axs[2].plot(EMG_Abs2, color = 'orchid')
                    axs[2].plot(MA_EMG2, color = 'slateblue', linewidth=3)
                    axs[2].set_title('EMG2', fontsize = 10)
                    
                    axs[3].plot(EMG_Abs3, color = 'thistle')
                    axs[3].plot(MA_EMG3, color = 'blueviolet', linewidth=3)
                    axs[3].set_title('EMG1 - EMG2', fontsize = 10)
                    
    # =============================================================================
    #                 x = EEG_Abs
    #                 y = EMG_Abs
    # =============================================================================
                    x = MA_EEG
                    y = MA_EMG3                
    
                    N = max(len(x), len(y))
                    n = min(len(x), len(y))
    
                    if N == len(y):
                        lags = np.arange(-N + 1, n)
    
                    else:
                        lags = np.arange(-n + 1, N)
    
                    c = signal.correlate(x / np.std(x), y / np.std(y), 'full')
    
    
                    axs[4].plot(lags, c / n, color='k', label="Cross-correlation")
    
                    axs[4].set_title('Cross-correlation')
    
    
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
                        axs[5].plot(EEG, color ='powderblue')
                        axs[5].plot(MA_EEG, color = 'olive', linewidth=3)
                        
                        axs[5].plot(EMG1[-samples_before_begin:], color = 'plum')
                        axs[5].plot(MA_EMG1[-samples_before_begin:], color = 'purple', linewidth=3)
                        
                        axs[5].plot(EMG2[-samples_before_begin:], color = 'orchid')
                        axs[5].plot(MA_EMG2[-samples_before_begin:], color = 'slateblue', linewidth=3)
                        
                        axs[5].plot(EMG3[-samples_before_begin:], color = 'thistle')
                        axs[5].plot(MA_EMG3[-samples_before_begin:], color = 'blueviolet', linewidth=3)
                        
                        axs[5].set_title('EEG and EMG after sync', fontsize = 10)
                        
                        self.samples_before_begin_EMG_Dreamento = -samples_before_begin
                        self.flag_sign_samples_before_begin_EMG_Dreamento = 'eeg_event_earlier'
                    else:
                        axs[5].plot(EEG, color ='powderblue')
                        axs[5].plot(MA_EEG, color = 'olive', linewidth=3)
                        
                        tmp = np.zeros(samples_before_begin)
                        
                        synced_EMG1 = np.append(tmp, EMG1)
                        synced_EMG_MA1 = np.append(tmp, MA_EMG1)
                        
                        synced_EMG2 = np.append(tmp, EMG2)
                        synced_EMG_MA2 = np.append(tmp, MA_EMG2)
                        
                        synced_EMG3 = np.append(tmp, EMG3)
                        synced_EMG_MA3 = np.append(tmp, MA_EMG3)
                        
                        axs[5].plot(synced_EMG1, color = 'plum')
                        axs[5].plot(synced_EMG_MA1, color = 'purple', linewidth=3)
                        
                        axs[5].plot(synced_EMG2, color = 'orchid')
                        axs[5].plot(synced_EMG_MA2, color = 'slateblue', linewidth=3)
                        
                        axs[5].plot(synced_EMG3, color = 'thistle')
                        axs[5].plot(synced_EMG_MA3, color = 'blueviolet', linewidth=3)
                        
                        axs[5].set_title('EEG and EMG after sync', fontsize = 10)
                        self.samples_before_begin_EMG_Dreamento = tmp
                        self.flag_sign_samples_before_begin_EMG_Dreamento = 'emg_earlier'
                        
                    plt.tight_layout()
                    MsgBox = tk.messagebox.askquestion ('Satisfying results?','Are the results satisfying? If not click on No to try again with the other method.',icon = 'warning')
                    if MsgBox == 'yes':
                        messagebox.showinfo("Information",f"Perfect! Now we proceed with the main analysis ... Click on OK and wait ...")
                        plt.show()
                        break
                        
                else: 
                    MsgBox = tk.messagebox.askquestion ('Synchronization?','Do you want to manually synchronize data?',icon = 'warning')
                    if MsgBox == 'yes':
                        # Truncate sync period
                        EEG_to_sync_period = sigScript_org[(time_sync_event[0] - 256*5):(time_sync_event[0] + 256*20)]
                        EMG_to_sync_period1 = EMG_data_get1[(time_sync_event[0] - 256*5):(time_sync_event[0] + 256*20)]
                        EMG_to_sync_period2 = EMG_data_get2[(time_sync_event[0] - 256*5):(time_sync_event[0] + 256*20)]
                        EMG_to_sync_period3 = EMG_data_get3[(time_sync_event[0] - 256*5):(time_sync_event[0] + 256*20)]
                        
                        # Rectified signal
                        self.EEG = EEG_to_sync_period
                        #EEG_Abs = abs(self.EEG)
                        EEG_Abs= self.EEG
                        
                        self.EMG1 = EMG_to_sync_period1
                        #EMG_Abs1= abs(self.EMG1)
                        EMG_Abs1= self.EMG1
                        
                        self.EMG2 = EMG_to_sync_period2
                        #EMG_Abs2= abs(self.EMG2)
                        EMG_Abs2= self.EMG2
                        
                        self.EMG3 = EMG_to_sync_period3
                        #EMG_Abs3 = abs(self.EMG3)
                        EMG_Abs3 = self.EMG3
                        
                        self.points = []
                        self.n = 2
        
                        self.fig, self.axs = plt.subplots(5 ,figsize=(15, 10))
                        line = self.axs[0].plot(EEG_Abs, picker=2, color = 'powderblue')
                        self.axs[0].set_title('Manual drift estimation ... \nPlease click on two points to create the estimate line ...')
                        
                        self.MA_EEG = self.MA(EEG_Abs, 512)
                        self.MA_EMG1 = self.MA(EMG_Abs1, 512)
                        self.MA_EMG2 = self.MA(EMG_Abs2, 512)
                        self.MA_EMG3 = self.MA(EMG_Abs3, 512)
        
                        self.axs[0].set_xlim([0,len(self.EMG1)])
                        self.axs[1].set_xlim([0,len(self.EMG1)])
                        self.axs[2].set_xlim([0,len(self.EMG1)])
                        self.axs[3].set_xlim([0,len(self.EMG1)])
                        self.axs[4].set_xlim([0,len(self.EMG1)])
        
                        self.axs[0].set_ylabel('Lag (samples)')
        
                        line = self.axs[1].plot(EMG_Abs1, picker=2, color = 'plum')
                        line = self.axs[2].plot(EMG_Abs2, picker=2, color = 'orchid')
                        line = self.axs[3].plot(EMG_Abs3, picker=2, color = 'thistle')
                        plt.tight_layout()
                        plt.show()
                        self.fig.canvas.mpl_connect('pick_event', self.onpick)
                        MsgBox = tk.messagebox.askquestion ('Satisfying results?','Are the results satisfying? If not click on No to try again with the other method.',icon = 'warning')
                        if MsgBox == 'yes':
                            messagebox.showinfo("Information",f"Perfect! Now we proceed with the main analysis ... Please click on OK and wait ...")
                            plt.show()
                            break
                        
         
        self.ppg_path = hypnodyne_files_list[0].split('EEG')[0] + 'OXY_IR_AC.edf'
        self.ppg_obj = mne.io.read_raw_edf(self.ppg_path)
        self.ppg_data = self.ppg_obj.get_data()[0]
        self.ppg_data = self.butter_bandpass_filter(data = self.ppg_data, lowcut=.3, highcut=100, fs = 256, order = 2)
            
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
        
        data = mne.io.read_raw_edf(self.HDRecorderRecording)
        raw_data = data.get_data()
        self.sigHDRecorder = np.ravel(raw_data)
        
        data_r = mne.io.read_raw_edf(self.HDRecorderRecording.split('EEG L.edf')[0] + 'EEG R.edf')
        raw_data_r = data_r.get_data()
        self.sigHDRecorder_r = np.ravel(raw_data_r)
                
        print('Acceleration and noise data imported successfully ...')
        
        self.samples_before_begin, self.sigHDRecorder_org_synced, self.sigScript_org, self.sigScript_org_R = self.calculate_lag(
                               plot=(int(self.plot_sync_output.get()) ==1) , path_EDF=self.HDRecorderRecording,\
                               path_Txt=self.ZmaxDondersRecording,\
                               T = 30,\
                               t_start_sync = 100,\
                               t_end_sync   = 200)
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
        if int(self.analysis_signal_options.get()) == 1:
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
        if int(self.is_autoscoring.get()) == 1: 
            self.create_save_options_autoscoring()
            
        # Create export option
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
                    self.axs[4].plot(self.EEG, color = 'powderblue')
                    self.axs[4].plot(self.MA_EEG, color = 'olive', linewidth = 2)
                    
                    tmp_sync = np.zeros(int(mean_x1-mean_x2))
                    self.synced_EMG1    = np.append(tmp_sync, self.EMG1)
                    self.synced_MA_EMG1 = np.append(tmp_sync, self.MA_EMG1)
                    
                    self.synced_EMG2    = np.append(tmp_sync, self.EMG2)
                    self.synced_MA_EMG2 = np.append(tmp_sync, self.MA_EMG2)
                    
                    self.synced_EMG3    = np.append(tmp_sync, self.EMG3)
                    self.synced_MA_EMG3 = np.append(tmp_sync, self.MA_EMG3)
                    
                    self.axs[4].plot(self.synced_EMG1, color = 'plum')
                    self.axs[4].plot(self.synced_MA_EMG1, color = 'purple', linewidth = 2)
                    self.axs[4].plot(self.synced_EMG2, color = 'orchid')
                    self.axs[4].plot(self.synced_MA_EMG2, color = 'blueviolet', linewidth = 2)
                    self.axs[4].plot(self.synced_EMG3, color = 'thistle')
                    self.axs[4].plot(self.synced_MA_EMG3, color = 'slateblue', linewidth = 2)
                    
                    n_sample_sync = len(tmp_sync)
                    
                    self.samples_before_begin_EMG_Dreamento = tmp_sync
                    self.flag_sign_samples_before_begin_EMG_Dreamento = 'emg_event_earlier'
                    
                if mean_x1 < mean_x2:
                    tmp_sync = int(mean_x2-mean_x1)
                    
                    self.axs[4].plot(self.EEG, color = 'powderblue')
                    self.axs[4].plot(self.MA_EEG, color = 'olive', linewidth = 2)
                    
                    self.axs[4].plot(self.EMG1[tmp_sync:], color = 'plum')
                    self.axs[4].plot(self.MA_EMG1[tmp_sync:], color = 'purple', linewidth = 2)
                    
                    self.axs[4].plot(self.EMG2[tmp_sync:], color = 'plum')
                    self.axs[4].plot(self.MA_EMG2[tmp_sync:], color = 'purple', linewidth = 2)
                    
                    self.axs[4].plot(self.EMG3[tmp_sync:], color = 'plum')
                    self.axs[4].plot(self.MA_EMG3[tmp_sync:], color = 'purple', linewidth = 2)

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
        self.label_save_path.grid(row = 6 , column = 0, padx = 15, pady = 10)
        
        # Define browse button to import hypnos
        self.button_save_browse = Button(self.frame_import, text = "Browse ...", 
                                           padx = 40, pady = 10, font = 'Calibri 12 ',
                                           command = self.save_path_finder,fg = 'blue',
                                           relief = RIDGE)
        self.button_save_browse.grid(row = 7, column = 0, padx = 15, pady = 10)
        
        
        
        # Label: Save name
        self.label_save_filename = Label(self.frame_import, text = "Saving filename:",
                                  font = 'Calibri 13 ')
        self.label_save_filename.grid(row = 6 , column = 1)#, padx = 15, pady = 10)
        
        # Create entry for user
        self.entry_save_name = Entry(self.frame_import,text = " enter filename.mat ")#, borderwidth = 2, width = 10)
        self.entry_save_name.grid(row = 7, column = 1)#, padx = 15, pady = 10)
        
        self.button_save_mat = Button(self.frame_import, text = "Save", padx = 40, pady=8,
                              font = 'Calibri 13 bold', relief = RIDGE, fg = 'blue',
                              command = self.save_results_button)
        self.button_save_mat.grid(row = 7 , column =2, padx = 15, pady = 10)
        
    #%% create save options for autoscoring
    def create_save_options_autoscoring(self):
        """
        Store the scoring results and the sleep metrics as a txt file
    
        :param self: access the attributes and methods of the class
        
        :returns: user_defined_name.mat

        """
        
        
        
        # Label: Save outcome
        self.label_save_path_autoscoring = Label(self.frame_import, text = "Path to save autoscoring results:",
                                  font = 'Calibri 13 ')
        self.label_save_path_autoscoring.grid(row = 4 , column = 0, padx = 15, pady = 10)
        
        # Define browse button to import hypnos
        self.button_save_browse_autoscoring = Button(self.frame_import, text = "Browse ...", 
                                           padx = 40, pady = 10, font = 'Calibri 12 ',
                                           command = self.save_path_finder,fg = 'blue',
                                           relief = RIDGE)
        self.button_save_browse_autoscoring.grid(row = 5, column = 0, padx = 15, pady = 10)
        
        
        
        # Label: Save name
        self.label_save_filename_autoscoring = Label(self.frame_import, text = "Saving filename:",
                                  font = 'Calibri 13 ')
        self.label_save_filename_autoscoring.grid(row = 4 , column = 1)#, padx = 15, pady = 10)
        
        # Create entry for user
        self.entry_save_name_autoscoring = Entry(self.frame_import,text = "Subject#_night#.txt ")#, borderwidth = 2, width = 10)
        self.entry_save_name_autoscoring.grid(row = 5, column = 1)#, padx = 15, pady = 10)
        
        self.button_save_mat_autoscoring = Button(self.frame_import, text = "Save", padx = 40, pady=8,
                              font = 'Calibri 13 bold', relief = RIDGE, fg = 'blue',
                              command = self.save_autoscoring_button)
        self.button_save_mat_autoscoring.grid(row = 5 , column =2, padx = 15, pady = 10)
    #%% ################### DEFINE FUNCTIONS OF BUTTON(S) #######################
    #%% Save autoscorrig button
    def save_autoscoring_button(self):
        
        if self.entry_save_name_autoscoring.get()[-4:] == '.txt':
           saving_dir = where_to_save_path + '/' + self.entry_save_name_autoscoring.get()
        else:
           saving_dir = where_to_save_path + '/' + self.entry_save_name_autoscoring.get() + '.txt'
           
           a_file = open(saving_dir, "w")
           a_file.write('=================== Dreamento: an open-source dream engineering toolbox! ===================\nhttps://github.com/dreamento')
           a_file.write('\nThis file has been autoscored by DreamentoScorer! \nSleep stages: Wake:0, N1:1, N2:2, SWS:3, REM:4.\n')
           a_file.write('N.B. this is an alpha version of DreamentoScorer, always double-check with manual scoring!\n')
           a_file.write('============================================================================================\n')
           
           for row in self.stage_autoscoring[:,np.newaxis]:
               np.savetxt(a_file, row, fmt='%1.0f')
           
           a_file.close()
           
           # Save sleep metrics
           with open(saving_dir[:-4]+'_metrics.txt', 'w') as convert_file:
               convert_file.write(json.dumps(self.stats))
        messagebox.showinfo(title = "Done!", message = f'Autoscoring results successfully saved in {saving_dir}!')
        
    #%% Function: Import EDF (Browse
    def load_hypnodyne_file_dialog(self):
        """
        The function to load the EEG.L file from the HDrecorder.

        :param self: access the attributes and methods of the class
        
        :returns: global hypnodyne_files_list

        """
    
        global hypnodyne_files_list
        
        self.filenames        = filedialog.askopenfilenames(title = 'select EEG L.edf file', 
                                                       filetype = (("EEG L edf", "*.edf"), ("All Files", "*.*")))
        
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
    #%% load txt files including paths to the folders for bulk autoscoring
    def browse_txt_for_bulk_autoscoring(self):
                
        """
        Load the autoscoring bulk (.txt) file comprising paths to all the folders to be scored
        
        :param self: access the attributes and methods of the class
        
        
        """        
        self.filename_paths_for_bulk_autoscoring    = filedialog.askopenfilenames(title = 'select a .txt file including paths to the folders to be autoscored', 
                                                       filetype = (("txt", "*.txt"),("All Files", "*.*")))

        self.bulk_autoscoring_files_list  = self.popupWin_bulk_autoscoring.tk.splitlist(self.filename_paths_for_bulk_autoscoring)
        
        # check if the user chose somthing
        if not self.bulk_autoscoring_files_list:
            
            self.label_no_bulk_scoring  = Label(self.popupWin_bulk_autoscoring , text = "No file has been selected!",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 3, column = 1)
        else:
            
            self.label_no_bulk_scoring  = Label(self.popupWin_bulk_autoscoring, text = "The (.txt) file has been loaded!",
                                  fg = 'green', font = 'Helvetica 9 bold').grid(row = 3, column = 1)
    
    #%% load txt files including paths to the folders for bulk autoscoring
    def browse_destination_folder_for_bulk_autoscoring(self):
                
        """
        Load the destination folder of autoscoring bulk (.txt) file
        :param self: access the attributes and methods of the class

        
        """

        self.destination_bulk_autoscoring_files_list  = filedialog.askdirectory()
        
        # check if the user chose somthing
        if not self.destination_bulk_autoscoring_files_list:
            
            self.label_no_bulk_scoring_destination  = Label(self.popupWin_bulk_autoscoring, text = "No destination folder has been selected!",
                                  fg = 'red', font = 'Helvetica 9 bold').grid(row = 3, column = 2)
        else:
            
            self.label_no_bulk_scoring_destination  = Label(self.popupWin_bulk_autoscoring, text = "The destination folder has been selectded!",
                                  fg = 'green', font = 'Helvetica 9 bold').grid(row = 3, column = 2)
    
     
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
        if int(self.analysis_signal_options.get()) == 2:
            
            self.sync_with_dreamento = True
            
            # Sanitary checks
            if 'hypnodyne_files_list' not in globals():
                messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The EEG L.edf file is not selected!")
    
            elif 'Dreamento_files_list' not in globals():
                messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The .txt file recorded by Dreamento is not selected!")
                
            elif 'marker_files_list' not in globals():
                messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The .json file of markers is not selected!")
                
# =============================================================================
#             elif 'EMG_files_list' not in globals() and int(self.analysis_signal_options.get()) == 1:
#                 messagebox.showerror("Dreamento", "Sorry, but no EMG files is loaded, though the plot EMG check box is activated! Change either of these and try again.")
#                 
#             elif str(self.EMG_scale_options_val.get()) == 'Set desired EMG amplitude ...' and int(self.analysis_signal_options.get()) == 1:
#                 messagebox.showerror("Dreamento", "Sorry, but a parameter is missing ...\nThe EMG amplitude is not set!")
#                 
# =============================================================================

                
            Dreamento_files_list, hypnodyne_files_list, marker_files_list
        
            
            self.ZmaxDondersRecording = Dreamento_files_list[0]
            self.HDRecorderRecording  = hypnodyne_files_list[0]
            self.path_to_json_markers = marker_files_list[0]
            
            data = mne.io.read_raw_edf(self.HDRecorderRecording)
            raw_data = data.get_data()
            self.sigHDRecorder = np.ravel(raw_data)
            
            data_r = mne.io.read_raw_edf(self.HDRecorderRecording.split('EEG L.edf')[0] + 'EEG R.edf')
            raw_data_r = data_r.get_data()
            self.sigHDRecorder_r = np.ravel(raw_data_r)
            
            self.noise_path = hypnodyne_files_list[0].split('EEG')[0] + 'NOISE.edf'
            self.noise_obj = mne.io.read_raw_edf(self.noise_path)
            self.noise_data = self.noise_obj.get_data()[0]
            
            self.ppg_path = hypnodyne_files_list[0].split('EEG')[0] + 'OXY_IR_AC.edf'
            self.ppg_obj = mne.io.read_raw_edf(self.ppg_path)
            self.ppg_data = self.ppg_obj.get_data()[0]
            self.ppg_data = self.butter_bandpass_filter(data = self.ppg_data, lowcut=.3, highcut=100, fs = 256, order = 2)
            
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
                               t_end_sync   = 200)
            # Filter?
            if int(self.is_filtering.get()) == 1: 
                print('Bandpass filtering (.3-30 Hz) started')
                self.sigScript_org   = self.butter_bandpass_filter(data = self.sigScript_org, lowcut=.3, highcut=30, fs = 256, order = 2)
                self.sigScript_org_R = self.butter_bandpass_filter(data = self.sigScript_org_R, lowcut=.3, highcut=30, fs = 256, order = 2)

            # Plot psd?
            if int(self.plot_psd.get()) == 1:
                print('plotting peridogram ...')
                self.plot_welch_periodogram(data = self.sigScript_org, sf = 256, win_size = 5)
                
            # Plot spectrogram as well?
            
            fig, markers_details = self.AssignMarkersToRecordedData_EEG_TFR_noEMG(data = self.sigScript_org, data_R = self.sigScript_org_R, sf = 256,\
                                         path_to_json_markers=self.path_to_json_markers,\
                                         markers_to_show = ['light', 'manual', 'sound'],\
                                         win_sec=30, fmin=0.5, fmax=25,\
                                         trimperc=5, cmap='RdBu_r', add_colorbar = False)
            # Activate save section
            if int(self.is_autoscoring.get()) == 1:      
                self.create_save_options_autoscoring()
                
        #%% Analyzing only HDRecorder
        elif int(self.analysis_signal_options.get()) == 3:         
        
            self.sync_with_dreamento = False
            # Sanitary checks
            if 'hypnodyne_files_list' not in globals():
                messagebox.showerror("Dreamento", "Sorry, but a file is missing ...\n The EEG L.edf file is not selected!")
                
            else:
                hypnodyne_files_list
                
                print('Analyzing ZMax Hypndoyne data ...')

                self.HDRecorderRecording  = hypnodyne_files_list[0]

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
                
                self.ppg_path = hypnodyne_files_list[0].split('EEG')[0] + 'OXY_IR_AC.edf'
                self.ppg_obj = mne.io.read_raw_edf(self.ppg_path)
                self.ppg_data = self.ppg_obj.get_data()[0]
                self.ppg_data = self.butter_bandpass_filter(data = self.ppg_data, lowcut=.3, highcut=100, fs = 256, order = 2)
                print('Required files imported successfully ...')
                
                data = mne.io.read_raw_edf(self.HDRecorderRecording)
                raw_data = data.get_data()
                self.sigHDRecorder = np.ravel(raw_data)
                
                data_r = mne.io.read_raw_edf(self.HDRecorderRecording.split('EEG L.edf')[0] + 'EEG R.edf')
                raw_data_r = data_r.get_data()
                self.sigHDRecorder_r = np.ravel(raw_data_r)
                
                # Filter?
                if int(self.is_filtering.get()) == 1: 
                    print('Bandpass filtering (.3-30 Hz) started')
                    self.sigHDRecorder   = self.butter_bandpass_filter(data = self.sigHDRecorder, lowcut=.3, highcut=30, fs = 256, order = 2)
                    self.sigHDRecorder_r = self.butter_bandpass_filter(data = self.sigHDRecorder_r, lowcut=.3, highcut=30, fs = 256, order = 2)
                    print('Filtered successfully')
    
                # Plot psd?
                if int(self.plot_psd.get()) == 1:
                    print('plotting peridogram ...')
                    self.plot_welch_periodogram(data = self.sigHDRecorder, sf = 256, win_size = 5)
                    print('PSD plotted successfully')
                    
                self.AnalyzeZMaxHypnodyne(data= self.sigHDRecorder, data_R = self.sigHDRecorder_r, sf = 256,\
                                        win_sec=30, fmin=0.3, fmax=25,\
                                        trimperc=5, cmap='RdBu_r', add_colorbar = False)

                        
        # Activate save section
        if int(self.is_autoscoring.get()) == 1:     
            self.create_save_options_autoscoring()
            
        # Create export option
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
    def save_results_button(self):
        
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
        if int(self.analysis_signal_options.get()) == 1:    
            
            
            self.dict_all_results['EEG_L_Dreamento'] = self.sigScript_org #self.sigScript_org
            self.dict_all_results['EEG_R_Dreamento'] = self.sigScript_org_R # self.sigScript_org_R
            self.dict_all_results['markers']        = self.markers_details # markers_details
            self.dict_all_results['marker_keys']    = self.marker_keys # marker_keys
            self.dict_all_results['Hypnodyne_EEG_L']= self.sigHDRecorder_org_synced # marker_key
    
            self.dict_all_results['samples_to_sync']= self.samples_before_begin # marker_keys
            self.dict_all_results['microphone_data']= self.noise_data # marker_keys
            self.dict_all_results['acc_x']= self.acc_x # marker_keys
            self.dict_all_results['acc_y']= self.acc_y # marker_keys
            self.dict_all_results['acc_z']= self.acc_z # marker_keys
            self.dict_all_results['raw_EMG1']= self.EMG_filtered_data1 # marker_keys
            self.dict_all_results['raw_EMG2']= self.EMG_filtered_data2 # marker_keys
            self.dict_all_results['raw_EMG1_minus_EMG2']= self.EMG_filtered_data1_minus_2
            #self.dict_all_results['sample_to_remove_from_EMG_to_sync_with_Dreamento']=self.samples_before_begin_EMG_Dreamento
        
        elif int(self.analysis_signal_options.get()) == 2:    
            self.dict_all_results = dict()
            self.dict_all_results['EEG_L_Dreamento'] = self.sigScript_org #self.sigScript_org
            self.dict_all_results['EEG_R_Dreamento'] = self.sigScript_org_R # self.sigScript_org_R
            self.dict_all_results['markers']        = self.markers_details # markers_details
            self.dict_all_results['marker_keys']    = self.marker_keys # marker_keys
            self.dict_all_results['Hypnodyne_EEG_L']= self.sigHDRecorder_org_synced # marker_key
    
            self.dict_all_results['samples_to_sync']= self.samples_before_begin # marker_keys
            self.dict_all_results['microphone_data']= self.noise_data # marker_keys
            self.dict_all_results['acc_x']= self.acc_x # marker_keys
            self.dict_all_results['acc_y']= self.acc_y # marker_keys
            self.dict_all_results['acc_z']= self.acc_z # marker_keys

        elif int(self.analysis_signal_options.get()) == 3:         
            self.dict_all_results['Hypnodyne_EEG_L']= self.sigHDRecorder # marker_key
            self.dict_all_results['Hypnodyne_EEG_R']= self.sigHDRecorder_r # marker_key

            self.dict_all_results['microphone_data']= self.noise_data # marker_keys
            self.dict_all_results['acc_x']= self.acc_x # marker_keys
            self.dict_all_results['acc_y']= self.acc_y # marker_keys
            self.dict_all_results['acc_z']= self.acc_z # marker_keys

        
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
        if int(self.is_autoscoring.get()) == 0:
            fig,AX = plt.subplots(nrows=13, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 10,1,1, 2, 2, 2, 4, 4, 4, 4, 10]})
            
            ax1 = plt.subplot(13,1,1)
            ax2 = plt.subplot(13,1,2, sharex = ax1)
            ax3 = plt.subplot(13,1,3, sharex = ax1)
            
            ax_epoch_marker = plt.subplot(13,1,4, )
            ax_epoch_light = plt.subplot(13,1,5, sharex = ax_epoch_marker)
            
            ax_acc = plt.subplot(13,1,6, sharex = ax_epoch_marker)
            ax_ppg = plt.subplot(13,1,7, sharex = ax_epoch_marker)
            ax_noise = plt.subplot(13,1,8, sharex = ax_epoch_marker)
            
    
            
            ax_EMG = plt.subplot(13,1,9, sharex = ax_epoch_marker)
            ax_EMG2 = plt.subplot(13,1,10, sharex = ax_epoch_marker)
            ax_EMG3 = plt.subplot(13,1,11, sharex = ax_epoch_marker)
            ax_TFR_short = plt.subplot(13,1,12, sharex = ax_epoch_marker)
            ax4 = plt.subplot(13,1,13, sharex = ax_epoch_marker)
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
            ax_TFR_short.set_ylabel('TFR', rotation = 90)#, labelpad=30, fontsize=8)
    
                       
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
            ax_acc.set_ylabel('Acc', rotation = 90)#, labelpad=30, fontsize=8)
            ax_noise.set_ylabel('Sound', rotation = 0, labelpad=30, fontsize=8)
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
            
            # Plot ppg
            ax_ppg.plot(np.arange(len(self.ppg_data))/256, self.ppg_data, linewidth = 2 , color = 'olive')
            ax_ppg.set_ylim([-100, 100])
            ax_ppg.set_ylabel('PPG')
            ax_ppg.set_yticks([])
            

            # plot noise
            ax_noise.plot(np.arange(len(self.noise_data[self.samples_before_begin:]))/256, self.noise_data[self.samples_before_begin:], linewidth = 2, color = 'navy')
            
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
    
            if self.flag_sync_EEG_EMG == True:
                
                print(f'sync samples: {str(self.samples_before_begin_EMG_Dreamento)} with shape {str(np.shape(self.samples_before_begin_EMG_Dreamento))}')
                print(f'sync criterion: {self.flag_sign_samples_before_begin_EMG_Dreamento}')
                
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
            #plt.subplots_adjust(left=0.1, right=0.1, top=0.9, bottom=0.1)
            plt.tight_layout()
            plt.show()
            
            global sig_emg_1, sig_emg_2, sig_emg_3
             
            sig_emg_1 = self.EMG_filtered_data1
            sig_emg_2 = self.EMG_filtered_data2
            sig_emg_3 = self.EMG_filtered_data1_minus_2
            
            if int(self.plot_EMG_quality_evaluation.get()) == 1:
                
                self.assess_EMG_data_quality()
            
        else:
            
           fig,AX = plt.subplots(nrows=15, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 10, 4, 4,1,1,2, 2, 2,4, 4, 4, 4, 10]})
           
           ax1 = plt.subplot(15,1,1)
           ax2 = plt.subplot(15,1,2, sharex = ax1)
           ax3 = plt.subplot(15,1,3, sharex = ax1)
           
           ax_autoscoring = plt.subplot(15,1,4, )
           ax_proba = plt.subplot(15,1,5, )
           ax_epoch_marker = plt.subplot(15,1,6, )
           ax_epoch_light = plt.subplot(15,1,7, sharex = ax_epoch_marker)
           
           ax_acc = plt.subplot(15,1,8, sharex = ax_epoch_marker)
           ax_ppg = plt.subplot(15,1,9, sharex = ax_epoch_marker)
           ax_noise = plt.subplot(15,1,10, sharex = ax_epoch_marker)
                      
           ax_EMG = plt.subplot(15,1,11, sharex = ax_epoch_marker)
           ax_EMG2 = plt.subplot(15,1,12, sharex = ax_epoch_marker)
           ax_EMG3 = plt.subplot(15,1,13, sharex = ax_epoch_marker)
           ax_TFR_short = plt.subplot(15,1,14, sharex = ax_epoch_marker)
           ax4 = plt.subplot(15,1,15, sharex = ax_epoch_marker)
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
           ax_TFR_short.set_ylabel('TFR', rotation = 90)#, labelpad=30, fontsize=8)

                      
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
           ax_acc.set_ylabel('Acc', rotation = 90)#, labelpad=30, fontsize=8)
           ax_noise.set_ylabel('Sound', rotation = 0, labelpad=30, fontsize=8)
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
           
           # Plot ppg
           ax_ppg.plot(np.arange(len(self.ppg_data))/256, self.ppg_data, linewidth = 2 , color = 'olive')
           ax_ppg.set_ylim([-100, 100])
           ax_ppg.set_ylabel('PPG')
           ax_ppg.set_yticks([])
           
           
           # plot noise
           ax_noise.plot(np.arange(len(self.noise_data[self.samples_before_begin:]))/256, self.noise_data[self.samples_before_begin:], linewidth = 2, color = 'navy')
           
           #ax4.get_xaxis().set_visible(False)
           
           fig.canvas.mpl_connect('key_press_event', self.pan_nav_EMG_autoscoring)
           fig.canvas.mpl_connect('button_press_event', self.onclick_EMG_autoscoring)
           #ax1.set_xlim([0, 7680])#len(data)])
           #ax2.set_xlim([0, 7680])#len(data)])
           #ax3.set_xlim([0, len(data)])
           #ax4.set_xlim([0, 7680])#len(data)])
           self.sync_with_dreamento = True
           
           self.autoscoring()

           stages = self.y_pred
           #stages = np.row_stack((stages, stages[-1]))
           x      = np.arange(len(stages))
           self.stage_autoscoring = stages
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
           ax_autoscoring.step(x, y, where='post', color = 'black')
           rem = [i for i,j in enumerate(self.y_pred) if (self.y_pred[i]==4)]
           for i in np.arange(len(rem)) -1:
               ax_autoscoring.plot([rem[i]-1, rem[i]], [-1,-1] , linewidth = 2, color = 'red')

           #ax_autoscoring.scatter(rem, -np.ones(len(rem)), color = 'red')
           ax_autoscoring.set_yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'],\
                                     fontsize = 8)
           ax_autoscoring.set_xlim([np.min(x), np.max(x)])
           
           ax_proba.set_xlim([np.min(x), np.max(x)])
           self.y_pred_proba.plot(ax = ax_proba, kind="area", alpha=0.8, stacked=True, lw=0, color = ['black', 'olive', 'deepskyblue', 'purple', 'red'])
           ax_proba.legend(loc = 'right', prop={'size': 6})
           
           ax_proba.set_yticks([])
           ax_proba.set_xticks([])

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

           if self.flag_sync_EEG_EMG == True:
               
               print(f'sync samples: {str(self.samples_before_begin_EMG_Dreamento)} with shape {str(np.shape(self.samples_before_begin_EMG_Dreamento))}')
               print(f'sync criterion: {self.flag_sign_samples_before_begin_EMG_Dreamento}')
               
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
           #plt.subplots_adjust(left=0.1, right=0.1, top=0.9, bottom=0.1)
           plt.tight_layout()
           plt.show()
           
           if int(self.plot_EMG_quality_evaluation.get()) == 1:
               
               self.assess_EMG_data_quality()           

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
        if int(self.is_autoscoring.get()) == 0:
            fig,AX = plt.subplots(nrows=10, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 10,1,1, 2, 2, 2, 4, 10]})
            
            ax1 = plt.subplot(10,1,1)
            ax2 = plt.subplot(10,1,2, sharex = ax1)
            ax3 = plt.subplot(10,1,3, sharex = ax1)
            
            ax_epoch_marker = plt.subplot(10,1,4, )
            ax_epoch_light = plt.subplot(10,1,5, sharex = ax_epoch_marker)
            
            ax_acc = plt.subplot(10,1,6, sharex = ax_epoch_marker)
            ax_ppg = plt.subplot(10,1,7, sharex = ax_epoch_marker)
            ax_noise = plt.subplot(10,1,8, sharex = ax_epoch_marker)

            ax_TFR_short = plt.subplot(10,1,9, sharex = ax_epoch_marker)
            ax4 = plt.subplot(10,1,10, sharex = ax_epoch_marker)
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
            ax_TFR_short.set_ylabel('TFR', rotation = 90)#, labelpad=30, fontsize=8)
    
                       
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
            ax_acc.set_ylabel('Acc', rotation = 90)#, labelpad=30, fontsize=8)
            ax_noise.set_ylabel('Sound', rotation = 0, labelpad=30, fontsize=8)
            
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
            
            # Plot ppg
            ax_ppg.plot(np.arange(len(self.ppg_data))/256, self.ppg_data, linewidth = 2 , color = 'olive')
            ax_ppg.set_ylim([-100, 100])
            ax_ppg.set_ylabel('PPG')
            ax_ppg.set_yticks([])
            
            
            # plot noise
            ax_noise.plot(np.arange(len(self.noise_data[self.samples_before_begin:]))/256, self.noise_data[self.samples_before_begin:], linewidth = 2, color = 'navy')
            
            #ax4.get_xaxis().set_visible(False)
            
            fig.canvas.mpl_connect('key_press_event', self.pan_nav_noEMG)
            fig.canvas.mpl_connect('button_press_event', self.onclick_noEMG)
        else:
            
            fig,AX = plt.subplots(nrows=12, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 1, 10,4,4,1,1,2, 2, 2, 4, 10]})
            
            ax1 = plt.subplot(12,1,1)
            ax2 = plt.subplot(12,1,2, sharex = ax1)
            ax3 = plt.subplot(12,1,3, sharex = ax1)
            
            ax_autoscoring = plt.subplot(12,1,4, )
            ax_proba = plt.subplot(12,1,5, )
            ax_epoch_marker = plt.subplot(12,1,6, )
            ax_epoch_light = plt.subplot(12,1,7, sharex = ax_epoch_marker)
            
            ax_acc = plt.subplot(12,1,8, sharex = ax_epoch_marker)
            ax_ppg = plt.subplot(12,1,9, sharex = ax_epoch_marker)
            ax_noise = plt.subplot(12,1,10, sharex = ax_epoch_marker)
            ax_TFR_short = plt.subplot(12,1,11, sharex = ax_epoch_marker)
            ax4 = plt.subplot(12,1,12, sharex = ax_epoch_marker)
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
            ax_TFR_short.set_ylabel('TFR', rotation = 90)#, labelpad=30, fontsize=8)
    
                       
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
            ax_acc.set_ylabel('Acc', rotation = 90)#, labelpad=30, fontsize=8)
            ax_noise.set_ylabel('Sound', rotation = 0, labelpad=30, fontsize=8)
            
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
            
            # Plot ppg
            ax_ppg.plot(np.arange(len(self.ppg_data))/256, self.ppg_data, linewidth = 2 , color = 'olive')
            ax_ppg.set_ylim([-100, 100])
            ax_ppg.set_ylabel('PPG')
            ax_ppg.set_yticks([])
            
            
            # plot noise
            ax_noise.plot(np.arange(len(self.noise_data[self.samples_before_begin:]))/256, self.noise_data[self.samples_before_begin:], linewidth = 2, color = 'navy')
            
            #ax4.get_xaxis().set_visible(False)
            self.sync_with_dreamento = True
            self.autoscoring()
            stages = self.y_pred
            #stages = np.row_stack((stages, stages[-1]))
            x      = np.arange(len(stages))
            self.stage_autoscoring = stages
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
            ax_autoscoring.step(x, y, where='post', color = 'black')
            rem = [i for i,j in enumerate(self.y_pred) if (self.y_pred[i]==4)]
            
            for i in np.arange(len(rem)) -1:
                ax_autoscoring.plot([rem[i]-1, rem[i]], [-1,-1] , linewidth = 2, color = 'red')

                    
            ax_autoscoring.set_yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'])
            ax_autoscoring.set_xlim([np.min(x), np.max(x)])
            
            ax_proba.set_xlim([np.min(x), np.max(x)])
            self.y_pred_proba.plot(ax = ax_proba, kind="area", alpha=0.8, stacked=True, lw=0, color = ['black', 'navy', 'deepskyblue', 'purple', 'red'])
            ax_proba.legend(loc = 'right', prop={'size': 6})
            ax_proba.set_yticks([])
            ax_proba.set_xticks([])
            
            fig.canvas.mpl_connect('key_press_event', self.pan_nav_noEMG)
            fig.canvas.mpl_connect('button_press_event', self.onclick_noEMG)


        
        return fig, markers_details
    
    #%% Autoscoring caution
# =============================================================================
#     def scoring_caution(self):
#         if int(self.is_autoscoring.get()) == 1:
#             messagebox.showinfo("Caution",f"The current version of DreamentoScorer is alpha - thus its generalizability is limited! Always double-check with manual scoring! \n N.B. For a proper performance of the DreamentoScorer both EEG channels should have acceptable quality. The scoring for the epochs of data loss is not reliable.")
# =============================================================================

    #%% Autoscoring    
    def autoscoring(self, DreamentoScorer_path = '.\\DreamentoScorer\\',\
                    model_path = "DreamentoScorer_model_beta_January2023.sav",
                    standard_scaler_path = "StandardScaler_td=3_bidirectional_Trainedon_500_estimator_3013097-06_1st_iter_121222.sav",
                    feat_selection_path = "Selected_Features_BoturaAfterTD=3_Bidirectional_500_estimator_3013097-06_121222.pickle",
                    fs = 256):
        
        # old CS model
        # 'Dreamento_autoscoring_Lightgbm_td=3_bidirectional_version_alpha_trained_on_69_data.sav',\
        # 'StandardScaler_Trainedon_3013097-06_1st_iter_091222.sav',\
        # 'Selected_Features_BoturaAfterTD=3_Bidirectional_3013097-06_061222.pickle',\
        # ==================================
        import joblib
        path_to_DreamentoScorer = DreamentoScorer_path
        
        # Init dir tio read libraries
        try:
            # Change the current working Directory    
            os.chdir(path_to_DreamentoScorer)
            print("Loading DreamentoScorer class ... ")
            
        except OSError:
            print("Can't change the Current Working Directory")     
        print(f'current path is {path_to_DreamentoScorer}')
        from entropy.entropy import spectral_entropy
        from DreamentoScorer import DreamentoScorer
        self.SSN = DreamentoScorer(filename='', channel='', fs = fs, T = 30)
        f_min = .3 #Hz
        f_max = 30 #Hz
        tic   = time.time()
        
        if self.sync_with_dreamento == True:
            if int(self.is_filtering.get()) == 1: 
                # If it was initially set to filter data, there is no need to re-filter here     
                print('already filtered ... proceed with scoring ...')
                EEG_L_filtered =  self.sigHDRecorder[self.samples_before_begin:]
                EEG_R_filtered = self.sigHDRecorder_r[self.samples_before_begin:]
                
            else:
                # If it was not initially set to filter data, it has to be filtered here
                print('Filtering ...')
                # if already filtered, take it, otherwise filter first
                EEG_L_filtered     = self.butter_bandpass_filter(data = self.sigHDRecorder[self.samples_before_begin:],\
                                                                 lowcut=.3, highcut=30, fs = 256, order = 2)
                EEG_R_filtered     = self.butter_bandpass_filter(data = self.sigHDRecorder_r[self.samples_before_begin:],\
                                                                 lowcut=.3, highcut=30, fs = 256, order = 2)
        else:
            if int(self.is_filtering.get()) == 1: 
                # If it was initially set to filter data, there is no need to re-filter here     
                print('already filtered ... proceed with scoring ...')
                EEG_L_filtered =  self.sigHDRecorder
                EEG_R_filtered = self.sigHDRecorder_r
                
            else:
                # If it was not initially set to filter data, it has to be filtered here
                print('Filtering ...')
                # if already filtered, take it, otherwise filter first
                EEG_L_filtered     = self.butter_bandpass_filter(data = self.sigHDRecorder,\
                                                                 lowcut=.3, highcut=30, fs = 256, order = 2)
                EEG_R_filtered     = self.butter_bandpass_filter(data = self.sigHDRecorder_r,\
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

        # Extract features
        tic = time.time()
        print(f'shape after epoching: {np.shape(EEG_L_filtered_epoched)}')
        print('Extracting features for DreamentoScorer ...')
        for k in np.arange(np.shape(EEG_L_filtered_epoched)[0]):
            print('Extracting features from channel 1 ...')
            feat_L = self.SSN.FeatureExtraction_per_subject(Input_data = EEG_L_filtered_epoched[k,:,:], fs = fs)
            print('Extracting features from channel 2 ...')
            feat_R = self.SSN.FeatureExtraction_per_subject(Input_data = EEG_R_filtered_epoched[k,:,:], fs = fs)
            # Concatenate features
            print(f'concatenating features of size {np.shape(feat_L)} and {np.shape(feat_R)}')
            Feat_all_channels = np.column_stack((feat_L,feat_R))
            
        # Scoring
        X_test  = Feat_all_channels
        X_test  = self.SSN.replace_NaN_with_mean(X_test)

        # Replace any probable inf
        X_test  = self.SSN.replace_inf_with_mean(X_test)

        # Z-score features
        print('Importing DreamentoScorer model ...')
        sc_fname = standard_scaler_path#'StandardScaler_TrainedonQS_1st_iter_To_transform_ZmaxDonders'
        sc = joblib.load(sc_fname)
        X_test = sc.transform(X_test)

        # Add time dependence to the data classification
        td = 3 # 5epochs of memory
        print('Adding time dependency ...')
        X_test_td  = self.SSN.add_time_dependence_bidirectional(X_test,  n_time_dependence=td,\
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
        DreamentoScorer = joblib.load(model_dir)

        self.y_pred = DreamentoScorer.predict(X_test)
        
        y_pred_proba = DreamentoScorer.predict_proba(X_test)
        self.y_pred_proba = pd.DataFrame(y_pred_proba, columns = ['Wake', 'N1', 'N2', 'SWS', 'REM'])
        self.sleep_stats = self.retrieve_sleep_statistics(hypno = self.y_pred, sf_hyp = 1 / 30,\
                                                     sleep_stages = [0, 1, 2, 3, 4])
        
        
        

    #%% Bulk autoscoring function
    def bulk_autoscoring(self, DreamentoScorer_path = ".\\DreamentoScorer\\",\
                    model_path = "DreamentoScorer_model_beta_January2023.sav",\
                    standard_scaler_path = "StandardScaler_td=3_bidirectional_Trainedon_500_estimator_3013097-06_1st_iter_121222.sav",\
                    feat_selection_path = "Selected_Features_BoturaAfterTD=3_Bidirectional_500_estimator_3013097-06_121222.pickle",\
                    fs = 256):
        
        import joblib
        path_to_DreamentoScorer = DreamentoScorer_path
        messagebox.showinfo(title = "Bulk Autoscoring", message = 'Autoscoring started ... \nDepending on the number of scorings, this may take a while ... \nIf you selected to store the results, they will be stored in the data folder ...\nPlease click on OK and be patient ...')
        # Init dir tio read libraries
        try:
            # Change the current working Directory    
            os.chdir(path_to_DreamentoScorer)
            print("Loading DreamentoScorer class ... ")
            
        except OSError:
            print("Can't change the Current Working Directory")     
        print(f'current path is {path_to_DreamentoScorer}')
        from entropy.entropy import spectral_entropy
        from DreamentoScorer import DreamentoScorer
        self.SSN = DreamentoScorer(filename='', channel='', fs = fs, T = 30)
        f_min = fmin = .3 #Hz
        f_max = fmax =  30 #Hz
        tic   = time.time()
        print(f'i reached here {self.bulk_autoscoring_files_list[0]}')
        self.folders_to_be_autoscored = pd.read_csv(self.bulk_autoscoring_files_list[0], header=None).to_numpy()
        print(f'folders to be autoscored are detected as follows: {self.folders_to_be_autoscored}')
        
        counter_scoring = 0
        for folder_autoscoring in self.folders_to_be_autoscored:
            folder_autoscoring = folder_autoscoring[0]
            counter_scoring = counter_scoring + 1
            print(f'autoscoring folder: {folder_autoscoring} [{counter_scoring} / {len(self.folders_to_be_autoscored)}]')
            
            #sanity check:
            if folder_autoscoring[-1] != '/' or folder_autoscoring[-1] != '\\':
                folder_autoscoring = folder_autoscoring + '\\'
                
            data_L = mne.io.read_raw_edf(folder_autoscoring + 'EEG L.edf')
            raw_data_L = data_L.get_data()
            self.sigHDRecorder = np.ravel(raw_data_L)
            
            data_r = mne.io.read_raw_edf(folder_autoscoring + 'EEG R.edf')
            raw_data_r = data_r.get_data()
            self.sigHDRecorder_r = np.ravel(raw_data_r)
           
            # There has to be filtering in her anyways 
            print('Filtering ...')
            # if already filtered, take it, otherwise filter first
            EEG_L_filtered     = self.butter_bandpass_filter(data = self.sigHDRecorder,\
                                                             lowcut=.3, highcut=30, fs = 256, order = 2)
            EEG_R_filtered     = self.butter_bandpass_filter(data = self.sigHDRecorder_r,\
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
    
            # Extract features
            tic = time.time()
            print(f'shape after epoching: {np.shape(EEG_L_filtered_epoched)}')
            print('Extracting features for DreamentoScorer ...')
            for k in np.arange(np.shape(EEG_L_filtered_epoched)[0]):
                t_st = time.time()
                print('Extracting features from channel 1 ...')
                feat_L = self.SSN.FeatureExtraction_per_subject(Input_data = EEG_L_filtered_epoched[k,:,:], fs = fs)
                print('Extracting features from channel 2 ...')
                feat_R = self.SSN.FeatureExtraction_per_subject(Input_data = EEG_R_filtered_epoched[k,:,:], fs = fs)
                # Concatenate features
                print(f'concatenating features of size {np.shape(feat_L)} and {np.shape(feat_R)}')
                Feat_all_channels = np.column_stack((feat_L,feat_R))
                t_end = time.time()
                print(f'Features extracted in {t_end - t_st} s')
            # Scoring
            X_test  = Feat_all_channels
            X_test  = self.SSN.replace_NaN_with_mean(X_test)
    
            # Replace any probable inf
            X_test  = self.SSN.replace_inf_with_mean(X_test)
    
            # Z-score features
            print('Importing DreamentoScorer model ...')
            sc_fname = standard_scaler_path#'StandardScaler_TrainedonQS_1st_iter_To_transform_ZmaxDonders'
            sc = joblib.load(sc_fname)
            X_test = sc.transform(X_test)
    
            # Add time dependence to the data classification
            td = 3 # epochs of memory
            print('Adding time dependency ...')
            X_test_td  = self.SSN.add_time_dependence_bidirectional(X_test,  n_time_dependence=td,\
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
    
            self.y_pred = DreamentoScorer.predict(X_test)
            y_pred_proba = DreamentoScorer.predict_proba(X_test)
            self.y_pred_proba = pd.DataFrame(y_pred_proba, columns = ['Wake', 'N1', 'N2', 'SWS', 'REM'])
            plt.legend(loc = 'right', prop={'size': 6})

            self.sleep_stats = self.retrieve_sleep_statistics(hypno = self.y_pred, sf_hyp = 1 / 30,\
                                                         sleep_stages = [0, 1, 2, 3, 4])
                
            # Increase font size while preserving original
            old_fontsize = plt.rcParams["font.size"]
            plt.rcParams.update({"font.size": 9})
            
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
            stages = self.y_pred
            #stages = np.row_stack((stages, stages[-1]))
            x      = np.arange(len(stages))
            self.stage_autoscoring = stages
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
            ax2.step(x, y, where='post', color = 'black', linewidth = 1)
            rem = [i for i,j in enumerate(self.y_pred) if (self.y_pred[i]==4)]
            for i in np.arange(len(rem)) -1:
                ax2.plot([rem[i]-1, rem[i]], [-1,-1] , linewidth = 2, color = 'red')

            
            
            #ax_autoscoring.scatter(rem, -np.ones(len(rem)), color = 'red')
            ax2.set_yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'],\
                                      fontsize = 8)
            ax2.set_xlim([np.min(x), np.max(x)])
            ax3.set_xlim([np.min(x), np.max(x)])
            ax0.set_title(folder_autoscoring + 'EEG L')
            ax1.set_title(folder_autoscoring + 'EEG R')
            
            self.y_pred_proba.plot(ax = ax3, kind="area", alpha=0.8, stacked=True, lw=0, color = ['black', 'olive', 'deepskyblue', 'purple', 'red'])
            
            #plt.tight_layout()
            # Save results?
            if int(self.checkbox_save_bulk_autoscoring_txt_results_val.get()) == 1:
                
                save_path_autoscoring = folder_autoscoring + 'DreamentoScorer_autoscoring_vAlpha.txt'
                
                if os.path.exists(save_path_autoscoring):
                    os.remove(save_path_autoscoring)
                    
                saving_dir = save_path_autoscoring
                
                a_file = open(saving_dir, "w")
                a_file.write('=================== Dreamento: an open-source dream engineering toolbox! ===================\nhttps://github.com/dreamento')
                a_file.write('\nThis file has been autoscored by DreamentoScorer! \nSleep stages: Wake:0, N1:1, N2:2, SWS:3, REM:4.\n')
                a_file.write('N.B. this is an alpha version of DreamentoScorer, always double-check with manual scoring!\n')
                a_file.write('============================================================================================\n')
                
                for row in self.stage_autoscoring[:,np.newaxis]:
                    np.savetxt(a_file, row, fmt='%1.0f')
                a_file.close()
                
                # Save sleep metrics
                save_path_stats = folder_autoscoring + 'DreamentoScorer_autoscoring_vAlpha_stats.json'
                
                if os.path.exists(save_path_stats):
                    os.remove(save_path_stats)
                    
                with open(save_path_stats, 'w') as convert_file:
                    convert_file.write(json.dumps(self.stats))
                    
            #save_figure
            if int(self.checkbox_save_bulk_autoscoring_plot_val.get()) == 1:
                save_path_plots = folder_autoscoring + 'Dreamento_TFR_autoscoring.png'
                
                if os.path.exists(save_path_plots):
                    os.remove(save_path_plots)
                    
                plt.savefig(save_path_plots,dpi = 300)  
                
            if int(self.checkbox_close_plots_val.get()) == 1:
                plt.close()
                
           
    #%% AssignMarkersToRecordedData EEG + TFR
    def AnalyzeZMaxHypnodyne(self, data, data_R, sf,\
                                win_sec=30, fmin=0.3, fmax=40,
                                trimperc=5, cmap='RdBu_r', add_colorbar = False):
        """
        The main function: create the main display window of Offline Dreamento
        
        :param self: access the attributes and methods of the class
        :param data: EEG L data
        :param data_R: EEG R data
        :param sf: sampling frequency
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
        #plt.rcParams["figure.autolayout"] = True
        
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
    
        if int(self.is_autoscoring.get()) == 0: 
            fig,AX = plt.subplots(nrows=6, figsize=(16, 9), gridspec_kw={'height_ratios': [2,1,1,1,1,2]})
            
    
            ax3 = plt.subplot(6,1,1, )
            ax_acc = plt.subplot(6,1,2,)
            ax_ppg = plt.subplot(6,1,3, sharex = ax_acc)
            ax_noise = plt.subplot(6,1,4,  sharex = ax_acc)
            ax_TFR_short = plt.subplot(6,1,5, sharex = ax_acc)
            ax4 = plt.subplot(6,1,6, sharex = ax_acc)
            ax4.grid(True)
    
            ax3.get_xaxis().set_visible(False)
    
            ax_acc.get_xaxis().set_visible(True)
            ax_acc.set_yticks([])
            ax_noise.set_yticks([])
            ax_noise.get_xaxis().set_visible(False)
            
            ax_acc.set_ylim([-1.4, 1.4])
            
            
            ax3.set_title('Dreamento: post-processing ')
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
            ax4.plot(np.arange(len(data))/256, data  * 1e6, color = (160/255, 70/255, 160/255), linewidth = 1)
            ax4.plot(np.arange(len(data_R))/256, data_R * 1e6, color = (0/255, 128/255, 190/255), linewidth = 1)
            #axes[1].set_ylim([-200, 200])
            #ax4.set_xlim([0, len(data)])
            ax4.set_ylabel('EEG (uV)')
            ax4.set_ylim([-150, 150])
            ax_TFR_short.set_ylabel('TFR', rotation = 90)#, labelpad=30, fontsize=8)
    
                       
            # Opening JSON file]
            ax_acc.set_ylabel('Acc', rotation = 90)#, labelpad=30, fontsize=8)
            ax_noise.set_ylabel('Sound', rotation = 90)#, labelpad=30, fontsize=8)
    
            ax_acc.spines[["top", "bottom"]].set_visible(False)
            ax_noise.spines[["top", "bottom"]].set_visible(False)
            
            time_axis = np.round(np.arange(0, len(data)) / 256 , 2)
        
            ax4.set_xlabel('time (s)')
            #ax4.set_xticks(np.arange(len(data)), time_axis)
            ax4.set_xlim([0, 30])#len(data)])
            # plot acc
            ax_acc.plot(np.arange(len(self.acc_x))/256, self.acc_x, linewidth = 2 , color = 'blue')
            ax_acc.plot(np.arange(len(self.acc_y))/256, self.acc_y, linewidth = 2, color = 'red')
            ax_acc.plot(np.arange(len(self.acc_z))/256, self.acc_z, linewidth = 2, color = 'green')
            
            # Plot ppg
            ax_ppg.plot(np.arange(len(self.ppg_data))/256, self.ppg_data, linewidth = 2 , color = 'olive')
            ax_ppg.set_ylim([-100, 100])
            ax_ppg.set_ylabel('PPG')
            ax_ppg.set_yticks([])
            
            # plot noise
            ax_noise.plot(np.arange(len(self.noise_data))/256, self.noise_data, linewidth = 2, color = 'navy')
            
            #
            
            fig.canvas.mpl_connect('key_press_event', self.pan_nav_ZMaxHypnodyneOnly)
            fig.canvas.mpl_connect('button_press_event', self.onclick_ZMaxHypnodyneOnly)
            #ax1.set_xlim([0, 7680])#len(data)])
            #ax2.set_xlim([0, 7680])#len(data)])
            #ax3.set_xlim([0, len(data)])
            #ax4.set_xlim([0, 7680])#len(data)])
            plt.subplots_adjust(hspace = 0)
        elif int(self.is_autoscoring.get()) == 1:
            print('Plot results with autoscoring')
            fig,AX = plt.subplots(nrows=8, figsize=(16, 9), gridspec_kw={'height_ratios': [2,1,1,1,1,1,1,2]})
    
            ax3 = plt.subplot(8,1,1, )
            ax_autoscoring = plt.subplot(8,1,2, )
            ax_proba = plt.subplot(8,1,3, )
            ax_acc = plt.subplot(8,1,4,)
            ax_ppg = plt.subplot(8,1,5, sharex = ax_acc)
            ax_noise = plt.subplot(8,1,6,  sharex = ax_acc)    
            ax_TFR_short = plt.subplot(8,1,7, sharex = ax_acc)
            ax4 = plt.subplot(8,1,8, sharex = ax_acc)
            ax4.grid(True)
    
            ax3.get_xaxis().set_visible(False)
    
            ax_acc.get_xaxis().set_visible(True)
            ax_acc.set_yticks([])
            ax_noise.set_yticks([])
            ax_noise.get_xaxis().set_visible(False)
            
            ax_acc.set_ylim([-1.4, 1.4])
            
            ax3.set_title('Dreamento: post-processing ')
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
            ax4.plot(np.arange(len(data))/256, data  * 1e6, color = (160/255, 70/255, 160/255), linewidth = 1)
            ax4.plot(np.arange(len(data_R))/256, data_R * 1e6, color = (0/255, 128/255, 190/255), linewidth = 1)
            #axes[1].set_ylim([-200, 200])
            #ax4.set_xlim([0, len(data)])
            ax4.set_ylabel('EEG (uV)')
            ax4.set_ylim([-150, 150])
            ax_TFR_short.set_ylabel('TFR', rotation = 90)#, labelpad=30, fontsize=8)
    
                       
            # Opening JSON file]
            ax_acc.set_ylabel('Acc', rotation = 90)#, labelpad=30, fontsize=8)
            ax_noise.set_ylabel('Sound', rotation = 0, labelpad=30, fontsize=8)
    
            ax_acc.spines[["top", "bottom"]].set_visible(False)
            ax_noise.spines[["top", "bottom"]].set_visible(False)
            
            time_axis = np.round(np.arange(0, len(data)) / 256 , 2)
        
            ax4.set_xlabel('time (s)')
            #ax4.set_xticks(np.arange(len(data)), time_axis)
            ax4.set_xlim([0, 30])#len(data)])
            
            self.samples_before_begin = 0
            self.autoscoring()
            
            stages = self.y_pred
            #stages = np.row_stack((stages, stages[-1]))
            x      = np.arange(len(stages))
            self.epoch_autoscoring = x
            self.stage_autoscoring = stages
            
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
            

            
            ax_autoscoring.step(x, y, where='post', color = 'black')
            #ax_autoscoring.scatter(rem, -np.ones(len(rem)), color = 'red')
            ax_autoscoring.set_yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'])
            ax_autoscoring.set_xlim([np.min(x), np.max(x)])
            ax_proba.set_xlim([np.min(x), np.max(x)])
            self.y_pred_proba.plot(ax = ax_proba, kind="area", alpha=0.8, stacked=True, lw=0, color = ['black', 'olive', 'deepskyblue', 'purple', 'red'])
            ax_proba.legend(loc = 'right', prop={'size': 6})
            ax_proba.set_yticks([])
            ax_proba.set_xticks([])
            
            rem = [i for i,j in enumerate(self.y_pred) if (self.y_pred[i]==4)]
            for i in np.arange(len(rem)) -1:
                ax_autoscoring.plot([rem[i]-1, rem[i]], [-1,-1] , linewidth = 2, color = 'red')

            
            # plot acc
            ax_acc.plot(np.arange(len(self.acc_x))/256, self.acc_x, linewidth = 2 , color = 'blue')
            ax_acc.plot(np.arange(len(self.acc_y))/256, self.acc_y, linewidth = 2, color = 'red')
            ax_acc.plot(np.arange(len(self.acc_z))/256, self.acc_z, linewidth = 2, color = 'green')
            
            # Plot ppg
            ax_ppg.plot(np.arange(len(self.ppg_data))/256, self.ppg_data, linewidth = 2 , color = 'olive')
            ax_ppg.set_ylim([-100, 100])
            ax_ppg.set_ylabel('PPG')
            ax_ppg.set_yticks([])
            
            
            # plot noise
            ax_noise.plot(np.arange(len(self.noise_data))/256, self.noise_data, linewidth = 2, color = 'navy')
            
            ax4.get_xaxis().set_visible(True)
            plt.subplots_adjust(hspace = 0)
            fig.canvas.mpl_connect('key_press_event', self.pan_nav_ZMaxHypnodyneOnly)
            fig.canvas.mpl_connect('button_press_event', self.onclick_ZMaxHypnodyneOnly)
            messagebox.showinfo(title = "AASM sleep metrics", message = self.stats)
            
        return fig
    #%% Retrieve sleep statistics (from yasa)
    def retrieve_sleep_statistics(self, hypno, sf_hyp = 1 / 30, sleep_stages = [0, 1, 2, 3, 5],\
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
        
        self.stats = stats
        
    #%% Assess EMG quality
    def assess_EMG_data_quality(self, 
                                win_sec = 30,
                                fmin = 10,
                                fmax = 100,
                                sf = 256,
                                noverlap = 0,
                                trimperc=5,
                                cmap='RdBu_r',
                                log_power = False):
        
        from lspopt import spectrogram_lspopt
        from matplotlib.colors import Normalize, ListedColormap

        """        
        source: https://github.com/raphaelvallat/yasa/blob/master/notebooks/10_spectrogram.ipynb
       
        Adjusted by Mahdad for comparative purposes
        """
        
        sig1 = self.EMG_filtered_data1
        sig2 = self.EMG_filtered_data2
        sig3 = self.EMG_filtered_data1_minus_2
        print('EMG signals for quality assessment were loaded')
        # parameters
        win_sec = win_sec
        fmin = fmin
        fmax = fmax
        sf = sf
        noverlap = noverlap
        trimperc=trimperc
        cmap=cmap

        # Increase font size while preserving original
        old_fontsize = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': 12})

        # Safety checks
        assert isinstance(sig1, np.ndarray), 'Data1 must be a 1D NumPy array.'
        assert isinstance(sig2, np.ndarray), 'Data2 must be a 1D NumPy array.'
        assert isinstance(sig3, np.ndarray), 'Data1 must be a 1D NumPy array.'

        assert isinstance(sf, (int, float)), 'sf must be int or float.'

        assert sig1.ndim == 1, 'Data1 must be a 1D (single-channel) NumPy array.'
        assert sig2.ndim == 1, 'Data2 must be a 1D (single-channel) NumPy array.'
        assert sig3.ndim == 1, 'Data1 must be a 1D (single-channel) NumPy array.'

        assert isinstance(win_sec, (int, float)), 'win_sec must be int or float.'
        assert isinstance(fmin, (int, float)), 'fmin must be int or float.'
        assert isinstance(fmax, (int, float)), 'fmax must be int or float.'
        assert fmin < fmax, 'fmin must be strictly inferior to fmax.'
        assert fmax < sf / 2, 'fmax must be less than Nyquist (sf / 2).'

        print('Sanity checks completed!')
        # Calculate multi-taper spectrogram
        nperseg = int(win_sec * sf)

        assert sig1.size > 2 * nperseg, 'Data1 length must be at least 2 * win_sec.'
        assert sig2.size > 2 * nperseg, 'Data2 length must be at least 2 * win_sec.'
        assert sig3.size > 2 * nperseg, 'Data1 length must be at least 2 * win_sec.'


        f1, t1, Sxx1 = spectrogram_lspopt(sig1, sf, nperseg=nperseg, noverlap=noverlap)
        f2, t2, Sxx2 = spectrogram_lspopt(sig2, sf, nperseg=nperseg, noverlap=noverlap)
        f3, t3, Sxx3 = spectrogram_lspopt(sig3, sf, nperseg=nperseg, noverlap=noverlap)

        Sxx1 = 10 * np.log10(Sxx1)  # Convert uV^2 / Hz --> dB / Hz
        Sxx2 = 10 * np.log10(Sxx2)  # Convert uV^2 / Hz --> dB / Hz
        Sxx3 = 10 * np.log10(Sxx3)  # Convert uV^2 / Hz --> dB / Hz

        # Select only relevant frequencies (up to 30 Hz)
        good_freqs1 = np.logical_and(f1 >= fmin, f1 <= fmax)
        good_freqs2 = np.logical_and(f2 >= fmin, f2 <= fmax)
        good_freqs3 = np.logical_and(f3 >= fmin, f3 <= fmax)

        Sxx1 = Sxx1[good_freqs1, :]
        Sxx2 = Sxx2[good_freqs2, :]
        Sxx3 = Sxx3[good_freqs3, :]


        f1 = f1[good_freqs1]
        f2 = f2[good_freqs2]
        f3 = f3[good_freqs3]


        t1 /= 3600  # Convert t to hours
        t2 /= 3600  # Convert t to hours
        t3 /= 3600  # Convert t to hours


        # Normalization
        vmin1, vmax1 = np.percentile(Sxx1, [0 + trimperc, 100 - trimperc])
        vmin2, vmax2 = np.percentile(Sxx2, [0 + trimperc, 100 - trimperc])
        vmin3, vmax3 = np.percentile(Sxx3, [0 + trimperc, 100 - trimperc])

        norm1 = Normalize(vmin=vmin1, vmax=vmax1)
        norm2 = Normalize(vmin=vmin2, vmax=vmax2)
        norm3 = Normalize(vmin=vmin3, vmax=vmax3)
            
        # Plot signal 1
        #fig, ax = plt.subplots(3, 2)
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 5), gridspec_kw={'width_ratios':[2,1], 'height_ratios':[1,1,1]})


        im = ax[0,0].pcolormesh(t1, f1, Sxx1, norm=norm1, cmap=cmap, antialiased=True,
                           shading="auto")
        ax[0,0].set_xlim(0, t1.max())
        ax[0,0].set_ylabel('Frequency [Hz]')
        ax[0,0].set_xlabel('Time [hrs]')
        ax[0,0].set_title('EMG 1')
        # Add colorbar
        # =============================================================================
        #         cbar = fig.colorbar(im, ax=ax[0], shrink=0.95, fraction=0.1, aspect=25)
        #         cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)
        # =============================================================================

        # Plot signal 2
        im = ax[1,0].pcolormesh(t2, f2, Sxx2, norm=norm1, cmap=cmap, antialiased=True,
                           shading="auto") # Normalized with respect to the same freq range as sig1
        ax[1,0].set_xlim(0, t2.max())
        ax[1,0].set_ylabel('Frequency [Hz]')
        ax[1,0].set_title('EMG 2')
        # =============================================================================
        #         cbar = fig.colorbar(im, ax=ax[1], shrink=0.95, fraction=0.1, aspect=25)
        #         cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)
        # =============================================================================
        # Plot signal 3
        im = ax[2,0].pcolormesh(t3, f3, Sxx3, norm=norm1, cmap=cmap, antialiased=True,
                           shading="auto") # Normalized with respect to the same freq range as sig1
        ax[2,0].set_xlim(0, t3.max())
        ax[2,0].set_ylabel('Frequency [Hz]')
        ax[2,0].set_title('EMG 1 - EMG 2')

        # Periodogram
        win_size = 5
        win = win_size * sf
        freqs1, psd1 = signal.welch(x=sig1, fs=sf, nperseg=win)
        freqs2, psd2 = signal.welch(x=sig2, fs=sf, nperseg=win)
        freqs3, psd3 = signal.welch(x=sig3, fs=sf, nperseg=win)

        log_power = log_power
        if log_power:
            psd1 = 20 * np.log10(psd1)
            psd2 = 20 * np.log10(psd2)
            psd3 = 20 * np.log10(psd3)

        # Compute vvalues: 
        ret1 = yasa.bandpower(data=sig1, sf=sf)
        ret2 = yasa.bandpower(data=sig2, sf=sf)
        ret3 = yasa.bandpower(data=sig3, sf=sf)

        ax[0,1].plot(freqs1, psd1, color='k', lw=2)
        ax[0,1].set_title('EMG 1')
        ax[1,1].plot(freqs2, psd2, color='k', lw=2)
        ax[1,1].set_title('EMG 2')
        ax[2,1].plot(freqs3, psd3, color='k', lw=2)
        ax[2,1].set_title('EMG 1 - EMG 2')
        
        plt.subplots_adjust(hspace = 0.2)
    #%% Function: Help pop-up
    def help_pop_up_func(self):
        """
        Help button of the software. Introduction to the applications and hot keeys
        
        :param self: access the attributes and methods of the class
        """
        
        line_msg = "Welcome to Dreamento!\n" +\
        "You can Use offline Dreamento in different cases (1) to merely analyze ZMax Hypnodyne recording" +\
        " (2) to integrate ZMAX Hypnodyne and Dremento recordings, " +\
        " and (3) to integrate ZMax Hypnodyne, Dremento, and EMG recordings \n \n" +\
        "From the analysis options, check and uncheck the most relavant properties\n" +\
        "Then load the relavant files ...\n \n" +\
        "- Hotkeys to navigate in the final plot.\n" +\
        "- Keyboard down arrow: zoom out.\n" +\
        "- Keyboard up arrow: zoom in.\n" +\
        "- Keyboard right arrow: next epoch (30s).\n" +\
        "- Keyboard left arrow: previous epoch (30s).\n\n" +\
        "- To navigate through data, click on the desired part of the time-freq representation." +\
        "- In case you want a specific part of the EEG (in the current epoch) comes to the middle, click on it." +\
        "\n\n In case of full ZMax Hypnodyne, Dreamento, and EMG analysis:\n" +\
        "Markers and light stimulations of the whole data are shown on first two rows.\n" +\
        "However, current-epoch markers are shown in the middle.\n\n" +\
        "IMPORTANT: To keep the location navigator accurate, please keep the cursur on "  +\
        "the spectrogrma plot while pressing left/right buttons to navigate.\n\n" +\
        "Matlab compatability: After the analysis you can export an .mat file\n\n\n" +\
        "Contact: Mahdad.Jafarzadehesfahani@donders.ru.nl \n" +\
        "CopyRight (2021-2022): Mahdad Jafarzadeh Esfahani, Amir Hossein Daraie**" 
            
        
        messagebox.showinfo(title = "Help", message = line_msg)
    
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
            if str(curr_ax) == 'AxesSubplot(0.125,0.67913;0.775x0.167391)':
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
            if str(curr_ax) == 'AxesSubplot(0.125,0.67913;0.775x0.167391)':
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
            if str(curr_ax) == 'AxesSubplot(0.125,0.67913;0.775x0.167391)':
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([event.xdata, event.xdata], [0.3, 40], color = 'black')
                curr_ax.set_ylim((0.1,25))
                
    #%% Navigating via keyboard in the figure
    def pan_nav_EMG_autoscoring (self, event):
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
            if str(curr_ax) == 'AxesSubplot(0.125,0.708889;0.775x0.142593)':
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
            if str(curr_ax) == 'AxesSubplot(0.125,0.708889;0.775x0.142593)':
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
    
    def onclick_EMG_autoscoring(self, event):
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
            if str(curr_ax) == 'AxesSubplot(0.125,0.708889;0.775x0.142593)':
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
            if (str(curr_ax) == 'AxesSubplot(0.125,0.66;0.775x0.183333)' or str(curr_ax) =='AxesSubplot(0.125,0.608235;0.775x0.226471)'):
                print('spectrogram axis detected')
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([int(np.mean((lims[0] - adjust, lims[1] - adjust))), int(np.mean((lims[0] - adjust, lims[1] - adjust)))], [-150, 150], color = 'black')

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
            if (str(curr_ax) == 'AxesSubplot(0.125,0.66;0.775x0.183333)' or str(curr_ax) =='AxesSubplot(0.125,0.608235;0.775x0.226471)'):
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
    
    #%% Navigating via keyboard in the figure
    def pan_nav_ZMaxHypnodyneOnly(self, event):
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
            print('going back in data')
            lims = ax_tmp.get_xlim()
            adjust = (lims[1] - lims[0]) 
            ax_tmp.set_xlim((lims[0] - adjust, lims[1] - adjust))
            curr_ax = event.inaxes
            if (str(curr_ax) == 'AxesSubplot(0.125,0.726;0.775x0.154)' or str(curr_ax) == 'AxesSubplot(0.125,0.6875;0.775x0.1925)'):
                print('spectrogram axis detected')
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([int(np.mean((lims[0] - adjust, lims[1] - adjust))), int(np.mean((lims[0] - adjust, lims[1] - adjust)))], [-150, 150], color = 'black')

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
            if (str(curr_ax) == 'AxesSubplot(0.125,0.726;0.775x0.154)' or str(curr_ax) == 'AxesSubplot(0.125,0.6875;0.775x0.1925)'):
                print('spectrogram axis detected')
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([int(np.mean((lims[0] + adjust, lims[1] + adjust))), int(np.mean((lims[0] + adjust, lims[1] + adjust)))], [-150, 150], color = 'black')
                    
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
            if (str(curr_ax) == 'AxesSubplot(0.125,0.66;0.775x0.183333)' or str(curr_ax) =='AxesSubplot(0.125,0.608235;0.775x0.226471)'):
                if len(curr_ax.lines) > 0 :
                    curr_ax.lines[-1].remove()
                curr_ax.plot([event.xdata, event.xdata], [0.3, 40], color = 'black')
                curr_ax.set_ylim((0.1,25))
    #%% pop-up markers selection
    def select_marker_for_sync(self):
        self.popupWin = Toplevel(root)
        self.alert = Label(self.popupWin, text='Please select an event to sync EMG and EEG (recommendation: teeth clench')
        
        self.markers_sync_event = StringVar()        
        self.markers_sync_event_option_menu = OptionMenu(self.popupWin, self.markers_sync_event, *self.all_markers_with_timestamps)
        self.markers_sync_event_option_menu.pack()
        
        self.button_popupwin = Button(self.popupWin, text = "OK", command=self.select_marker)
        self.alert.pack()
        self.button_popupwin.pack()
        root.wait_window(self.popupWin)
    
    #%% pop-up markers selection
    def bulk_autoscoring_popup(self):
        self.popupWin_bulk_autoscoring = Toplevel(root)
        
        # button: load a txt file including paths of folders to be autoscored
        self.load_txt_bulk_autoscoring_paths_label = Label(self.popupWin_bulk_autoscoring, text='select a txt file including paths to folders')
        self.load_txt_bulk_autoscoring_paths_label.grid(row = 1 , column =1)
        self.button_load_bulk_autoscoring = Button(self.popupWin_bulk_autoscoring, text = "Browse ...",
                              font = 'Calibri 13 bold', relief = RIDGE,
                              command = self.browse_txt_for_bulk_autoscoring)
        self.button_load_bulk_autoscoring.grid(row = 2 , column =1)
        
        # button: Path to export autoscoring
# =============================================================================
#         self.select_folder_export_bulk_autoscoring_bulk_autoscoring_label = Label(self.popupWin_bulk_autoscoring, text='select destination folder')
#         self.select_folder_export_bulk_autoscoring_bulk_autoscoring_label.grid(row = 1 , column =2)
#         self.button_select_folder_export_bulk_autoscoring = Button(self.popupWin_bulk_autoscoring, text = "Browse ...",
#                               font = 'Calibri 13 bold', relief = RIDGE,
#                               command = self.browse_destination_folder_for_bulk_autoscoring)
#         self.button_select_folder_export_bulk_autoscoring.grid(row = 2 , column =2)
# =============================================================================
        
        # Button: start bulk autoscoring
        self.button_start_bulk_autoscoring_label = Label(self.popupWin_bulk_autoscoring, text='start autoscoring!')
        self.button_start_bulk_autoscoring_label.grid(row = 1 , column =3)
        self.button_start_bulk_autoscoring = Button(self.popupWin_bulk_autoscoring, text = "Start!",
                              font = 'Calibri 13 bold', relief = RIDGE,
                              command = self.bulk_autoscoring)
        self.button_start_bulk_autoscoring.grid(row = 2 , column =3)
        
        
        self.checkbox_save_bulk_autoscoring_txt_results_val = IntVar(value = 0)
        self.checkbox_bulk_autoscoring_txt_results = Checkbutton(self.popupWin_bulk_autoscoring, text = "Save autoscoring results?",
                                  font = 'Calibri 11 ', variable = self.checkbox_save_bulk_autoscoring_txt_results_val)
        self.checkbox_bulk_autoscoring_txt_results.grid(row = 1, column = 3)
        
        
        self.checkbox_save_bulk_autoscoring_plot_val = IntVar(value = 1)
        self.checkbox_bulk_autoscoring_plot = Checkbutton(self.popupWin_bulk_autoscoring, text = "Save TFR + autoscoring plots?",
                                  font = 'Calibri 11 ', variable = self.checkbox_save_bulk_autoscoring_plot_val)
        self.checkbox_bulk_autoscoring_plot.grid(row = 1, column = 4)
        
        self.checkbox_close_plots_val = IntVar(value = 1)
        self.checkbox_close_plot = Checkbutton(self.popupWin_bulk_autoscoring, text = "Close plots after save?",
                                  font = 'Calibri 11 ', variable = self.checkbox_close_plots_val)
        self.checkbox_close_plot.grid(row = 1, column = 5)
        
        self.auotscoring_model_Optionmenu_Label = Label(self.popupWin_bulk_autoscoring, text='Autoscoring model')
        self.auotscoring_model_Optionmenu_Label.grid(row = 2, column = 0)
        
        self.autoscoring_model_options = ['Select model','Lightgbm_td=3_Bidirectional', 'Usleep for ZMax']
        self.autoscoring_model_options_val = StringVar()

        self.auotscoring_model_Optionmenu = Optionmenu(self.popupWin_bulk_autoscoring, self.autoscoring_model_options_val, *self.autoscoring_model_options)

# =============================================================================
#         self.alert = Label(self.popupWin, text='Please select an event to sync EMG and EEG (recommendation: teeth clench')
#         
#         self.markers_sync_event = StringVar()        
#         self.markers_sync_event_option_menu = OptionMenu(self.popupWin, self.markers_sync_event, *self.all_markers_with_timestamps)
#         self.markers_sync_event_option_menu.pack()
#         
#         self.button_popupwin = Button(self.popupWin, text = "OK", command=self.select_marker)
#         self.alert.pack()
#         self.button_popupwin.pack()
# =============================================================================
        root.wait_window(self.popupWin_bulk_autoscoring)
    #%% select marker for sync command
    def select_marker(self):
        print(f'syncing based on the following event: {self.markers_sync_event.get()}')
        self.popupWin.destroy()
    #%% Click ZMax Hypnodyne only            
    def onclick_ZMaxHypnodyneOnly(self, event):
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
            if (str(curr_ax) == 'AxesSubplot(0.125,0.726;0.775x0.154)' or str(curr_ax) == 'AxesSubplot(0.125,0.6875;0.775x0.1925)'):
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