# -*- coding: utf-8 -*-
"""
Copyright (C) 2020-22, Mahdad Jafarzadeh Esfahani

THIS IS THE CLASS FOR "AUTOMATIC SLEEP SCORING". 

The class is capable of extracting relevant features, applying various machine-
learning algorithms and finally applying Randomized grid search to tune hyper-
parameters of different classifiers.

To see the example codes and instructions how to use each method of class, 
please visit: https://github.com/MahdadJafarzadeh/DreamentoScorer/


"""
#%% Importing libs
import numpy as np
import pandas as pd 
import pywt
from scipy.signal import butter, lfilter, periodogram, spectrogram, welch, filtfilt, iirnotch
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import argrelextrema
from sklearn.model_selection import cross_val_score,KFold, cross_validate
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from entropy.entropy import spectral_entropy
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew
from scipy.fftpack import fft
# =============================================================================
# import h5py
# =============================================================================
import time
#import pyeeg
from scipy.integrate import simps
import scipy
import scipy.fftpack

#import tensorflow as tf

class DreamentoScorer():
    
    def __init__(self, filename, channel, fs, T):
        
        self.filename = filename
        self.channel  = channel
        self.fs       = fs
        self.T        = T
        
    #%% Notch-filter    
    def NotchFilter(self, data, Fs, f0, Q):
        w0 = f0/(Fs/2)
        b, a = iirnotch(w0, Q)
        y = filtfilt(b, a, data)
        return y
    
    #%% Loading existing featureset
    def LoadFeatureSet(self, path, fname, feats, labels):
        # Reading N3 epochs
        with h5py.File(path + fname + '.h5', 'r') as rf:
            X  = rf['.'][feats].value
            y  = rf['.'][labels].value
        return X, y
    
    #%% Low=pass butterworth
    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y    
    
    #%% Band-pass Filtering section
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order = 2):
        nyq = 0.5 * fs
        low = lowcut /nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='band')
        #print(b,a)
        y = filtfilt(b, a, data)
        return y
    
    #%% high-pass Filtering section
    def butter_highpass_filter(self, data, highcut, fs, order):
        nyq = 0.5 * fs
        high = highcut/nyq
        b, a = butter(order, high, btype='highpass')
        y = filtfilt(b, a, data)
        return y

    #%% Combining epochs
    def CombineEpochs(self, directory = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/train_test/',
                      ch = 'fp2-M1', N3_fname  = 'tr90_N3_fp1-M2_fp2-M1',
                      REM_fname = 'tr90_fp1-M2_fp2-M1',
                      saving = False, fname_save = 'tst'):
        # Initialization 
        tic       = time.time() 
        # Defining the directory of saved files
        directory = directory 
        # Define channel of interest (currently fp1-M2 and fp2-M1 are only active)
        ch = ch
        # N3 epochs Filename 
        N3_fname  = N3_fname
        # REM epochs Filename
        REM_fname = REM_fname
        
        # Reading N3 epochs
        with h5py.File(directory + N3_fname + '.h5', 'r') as rf:
            xtest_N3  = rf['.']['x_test_' + ch].value
            xtrain_N3 = rf['.']['x_train_' + ch].value
            ytest_N3  = rf['.']['y_test_' + ch].value
            ytrain_N3 = rf['.']['y_train_' + ch].value
        print(f'N3 epochs were loaded successfully in {time.time()-tic} secs')    
        
        # Reading REM epochs
        tic       = time.time() 
        with h5py.File(directory + REM_fname + '.h5', 'r') as rf:
            xtest_REM  = rf['.']['x_test_' + ch].value
            xtrain_REM = rf['.']['x_train_' + ch].value
            ytest_REM  = rf['.']['y_test_' + ch].value
            ytrain_REM = rf['.']['y_train_' + ch].value
        print(f'REM epochs were loaded successfully in {time.time()-tic} secs') 
           
        # Combining epochs
        xtest   = np.row_stack((xtest_N3, xtest_REM))
        xtrain  = np.row_stack((xtrain_N3, xtrain_REM))
        ytest   = np.row_stack((ytest_N3, ytest_REM))
        ytrain  = np.row_stack((ytrain_N3, ytrain_REM))
        print('Epochs were successfully concatenated')
        
        # Save concatenated results:
        # SAVE train/test splits
        if saving == True:
            tic = time.time()
            fname_save = fname_save
            with h5py.File((directory+fname_save + '.h5'), 'w') as wf:
                dset = wf.create_dataset('y_test_' +ch, ytest.shape, data=ytest)
                dset = wf.create_dataset('y_train_'+ch, ytrain.shape, data=ytrain)
                dset = wf.create_dataset('x_test_' +ch, xtest.shape, data=xtest)
                dset = wf.create_dataset('x_train_'+ch, xtrain.shape, data=xtrain)
            print('Time to save H5: {}'.format(time.time()-tic))
            return xtrain, ytrain, xtest, ytest
        else:
            print('Outputs were generated but not saved')
            return xtrain, ytrain, xtest, ytest

    #%% Feature extarction
    def FeatureExtraction(self):
        
        ''' ~~~~~~################## INSTRUCTION #################~~~~~~~~
        ----
        THIS IS A FUNCTION TO EXTRACT FEATURES AND THEN USE THEM FOR ANY KIND OF
        SUPERVISED MACHINE LEARNING ALGORITHM.
    
        INPUTS: 
        1) filename : full directory of train-test split (e.g. .h5 file saved via Prepare_for_CNN.py)
        2) channel  : channel of interest, e.g. 'fp2-M1'
        
        OUTPUTS:
        1) X        : Concatenation of all featureset after random permutation.
        2) y        : Relevant labels of "X".
        '''
        # Loading data section
        # Load data
        tic = time.time() 
        fname = self.filename
        
        # choose channel to extract features from
        ch = self.channel
        fs = self.fs #Hz
        T  = self.T #sec
        # Split train and test 
        with h5py.File(fname, 'r') as rf:
            xtest  = rf['.']['x_test_' + ch].value
            xtrain = rf['.']['x_train_' + ch].value
            ytest  = rf['.']['y_test_' + ch].value
            ytrain = rf['.']['y_train_' + ch].value
        print('train and test data loaded in : {} secs'.format(time.time()-tic))
        
        # Flatten data for filter and normalization
        X_train = np.reshape(xtrain, (np.shape(xtrain)[0] * np.shape(xtrain)[1] ,1))
        X_test  = np.reshape(xtest, (np.shape(xtest)[0] * np.shape(xtest)[1] ,1))
        
        #%% Filtering section
        ## Defining preprocessing function ##
        def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
            nyq = 0.5 * fs
            low = lowcut /nyq
            high = highcut/nyq
            b, a = butter(order, [low, high], btype='band')
            #print(b,a)
            y = lfilter(b, a, data)
            return y
        
        # Apply filter
        X_train = butter_bandpass_filter(data=X_train, lowcut=.1, highcut=30, fs=fs, order=2)
        X_test  = butter_bandpass_filter(data=X_test , lowcut=.1, highcut=30, fs=fs, order=2)
        
        #%% Normalization section - DEACTIVATED
        #sc = StandardScaler()
        #X_train = sc.fit_transform(X_train)
        #X_test  = sc.transform(X_test)
        
        #%% Reshaping data per epoch
        X_train = np.reshape(X_train, (int(len(X_train) / (fs*T)), fs*T))
        X_test  = np.reshape(X_test,  (int(len(X_test) / (fs*T)), fs*T))
        
        # Concatenate to extract feats
        X       = np.concatenate((X_train, X_test))
        
        
        
        #%% Feature Extraction section
        
        # Defining EEG bands:
        eeg_bands = {'Delta'     : (0.5, 4),
                     'Theta_low' : (4  , 6),
                     'Theta_high': (6  , 8),
                     'Alpha'     : (8  , 11),
                     'Beta'      : (16 , 24),
                     'Sigma'     : (12 , 15),
                     'Sigma_slow': (10 , 12)}
        
        # Initializing variables of interest
        eeg_band_fft      = dict()
        freq_ix           = dict()
        Features = np.empty((0, 42))
        # Settings of peridogram    
        Window = 'hann'
        # zero-padding added with respect to (Nfft=2^(nextpow2(len(window))))
        Nfft = 2 ** 15 
        # Defining freq. resoultion
        fm, _ = periodogram(x = X[0,:], fs = fs, nfft = Nfft , window = Window)  
        tic = time.time()
        # Finding the index of different freq bands with respect to "fm" #
        for band in eeg_bands:
            freq_ix[band] = np.where((fm >= eeg_bands[band][0]) &   
                               (fm <= eeg_bands[band][1]))[0]    
        
        
        # Defining for loop to extract features per epoch
        for i in np.arange(len(X)):
            
            data = X[i,:]
            
            # Initialization for wavelet 
            cA_values  = []
            cD_values  = []
            cA_mean    = []
            cA_std     = []
            cA_Energy  = []
            cD_mean    = []
            cD_std     = []
            cD_Energy  = []
            Entropy_D  = []
            Entropy_A  = []
            first_diff = np.zeros(len(data)-1)
            
            ''' POWER --> Periodogram with padding '''
            # Compute the "total" power inside the investigational window
            _ , pxx = periodogram(x = data, fs = fs, nfft = Nfft , window = Window) 
            
            pow_total      = np.sum(pxx[np.arange(freq_ix['Delta'][0], freq_ix['Beta'][-1]+1)])
            Pow_Delta      = np.sum(pxx[freq_ix['Delta']]) 
            Pow_Theta_low  = np.sum(pxx[freq_ix['Theta_low']]) 
            Pow_Theta_high = np.sum(pxx[freq_ix['Theta_high']]) 
            Pow_Alpha      = np.sum(pxx[freq_ix['Alpha']]) 
            Pow_Beta       = np.sum(pxx[freq_ix['Beta']])  
            Pow_Sigma      = np.sum(pxx[freq_ix['Sigma']]) 
            Pow_Sigma_slow = np.sum(pxx[freq_ix['Sigma_slow']])  
            
            
            '''Power ratio in differnt freq ranges ''' 
            # Total pow is defined form 0.5 - 20 Hz
            Pow_Delta_ratio      = np.sum(pxx[freq_ix['Delta']]) / pow_total
            Pow_Theta_low_ratio  = np.sum(pxx[freq_ix['Theta_low']]) / pow_total
            Pow_Theta_high_ratio = np.sum(pxx[freq_ix['Theta_high']]) / pow_total
            Pow_Alpha_ratio      = np.sum(pxx[freq_ix['Alpha']]) / pow_total
            Pow_Beta_ratio       = np.sum(pxx[freq_ix['Beta']])  / pow_total
            Pow_Sigma_ratio      = np.sum(pxx[freq_ix['Sigma']]) / pow_total
            Pow_Sigma_slow_ratio = np.sum(pxx[freq_ix['Sigma_slow']]) / pow_total
            
            '''Apply Welch to see the dominant Max power in each freq band''' 
            ff, Psd             = welch(x = data, fs = fs, window = 'hann', nperseg= 512, nfft = Nfft)
            Pow_max_Total       = np.max(Psd[np.arange(freq_ix['Delta'][0], freq_ix['Beta'][-1]+1)])
            Pow_max_Delta       = np.max(Psd[freq_ix['Delta']])
            Pow_max_Theta_low   = np.max(Psd[freq_ix['Theta_low']])
            Pow_max_Theta_high  = np.max(Psd[freq_ix['Theta_high']])
            Pow_max_Alpha       = np.max(Psd[freq_ix['Alpha']])
            Pow_max_Beta        = np.max(Psd[freq_ix['Beta']])
            Pow_max_Sigma       = np.max(Psd[freq_ix['Sigma']])
            Pow_max_Sigma_slow  = np.max(Psd[freq_ix['Sigma_slow']])
            
            ''' Spectral Entropy '''
            Entropy_Welch = spectral_entropy(x = data, sf=fs, method='welch', nperseg = 512)
            Entropy_fft   = spectral_entropy(x = data, sf=fs, method='fft')
               
            ''' Wavelet Decomposition ''' 
            cA,cD=pywt.dwt(data,'coif1')
            cA_values.append(cA)
            cD_values.append(cD)
            cA_mean.append(np.mean(cA_values))
            cA_std.append(np.std(cA_values))
            cA_Energy.append(np.sum(np.square(cA_values)))
            cD_mean.append(np.mean(cD_values))
            cD_std.append(np.std(cD_values))
            cD_Energy.append(np.sum(np.square(cD_values)))
            Entropy_D.append(np.sum(np.square(cD_values) * np.log(np.square(cD_values))))
            Entropy_A.append(np.sum(np.square(cA_values) * np.log(np.square(cA_values))))
            
            ''' Hjorth Parameters '''
            hjorth_activity     = np.var(data)
            diff_input          = np.diff(data)
            diff_diffinput      = np.diff(diff_input)
            hjorth_mobility     = np.sqrt(np.var(diff_input)/hjorth_activity)
            hjorth_diffmobility = np.sqrt(np.var(diff_diffinput)/np.var(diff_input))
            hjorth_complexity   = hjorth_diffmobility / hjorth_mobility
             
            ''' Statisctical features'''
            Kurt     = kurtosis(data, fisher = False)
            Skewness = skew(data)
            Mean     = np.mean(data)
            Median   = np.median(data)
            Std      = np.std(data)
            ''' Coefficient of variation '''
            coeff_var = Std / Mean
            
            ''' First and second difference mean and max '''
            sum1  = 0.0
            sum2  = 0.0
            Max1  = 0.0
            Max2  = 0.0
            for j in range(len(data)-1):
                    sum1     += abs(data[j+1]-data[j])
                    first_diff[j] = abs(data[j+1]-data[j])
                    
                    if first_diff[j] > Max1: 
                        Max1 = first_diff[j] # fi
                        
            for j in range(len(data)-2):
                    sum2 += abs(first_diff[j+1]-first_diff[j])
                    if abs(first_diff[j+1]-first_diff[j]) > Max2 :
                    	Max2 = first_diff[j+1]-first_diff[j] 
                        
            diff_mean1 = sum1 / (len(data)-1)
            diff_mean2 = sum2 / (len(data)-2) 
            diff_max1  = Max1
            diff_max2  = Max2
            
            ''' Variance and Mean of Vertex to Vertex Slope '''
            t_max   = argrelextrema(data, np.greater)[0]
            amp_max = data[t_max]
            t_min   = argrelextrema(data, np.less)[0]
            amp_min = data[t_min]
            tt      = np.concatenate((t_max,t_min),axis=0)
            if len(tt)>0:
                tt.sort() #sort on the basis of time
                h=0
                amp = np.zeros(len(tt))
                res = np.zeros(len(tt)-1)
                
                for l in range(len(tt)):
                        amp[l] = data[tt[l]]
                        
                out = np.zeros(len(amp)-1)     
                 
                for j in range(len(amp)-1):
                    out[j] = amp[j+1]-amp[j]
                amp_diff = out
                
                out = np.zeros(len(tt)-1)  
                
                for j in range(len(tt)-1):
                    out[j] = tt[j+1]-tt[j]
                tt_diff = out
                
                for q in range(len(amp_diff)):
                        res[q] = amp_diff[q]/tt_diff[q] #calculating slope        
                
                slope_mean = np.mean(res) 
                slope_var  = np.var(res)   
            else:
                slope_var, slope_mean = 0, 0
                
            ''' Spectral mean '''
            Spectral_mean = 1 / (freq_ix['Beta'][-1] - freq_ix['Delta'][0]) * (Pow_Delta + 
                    Pow_Theta_low + Pow_Theta_high + Pow_Alpha + Pow_Beta + 
                    Pow_Sigma) 
            
            ''' Correlation Dimension Feature '''
            #cdf = nolds.corr_dim(data,1)
            
            ''' Detrended Fluctuation Analysis ''' 

            ''' Wrapping up featureset '''
            feat = [pow_total, Pow_Delta, Pow_Theta_low, Pow_Theta_high, Pow_Alpha,
                    Pow_Beta, Pow_Sigma, Pow_Sigma_slow, cA_mean[0], cA_std[0],
                    cA_Energy[0], cD_Energy[0],  cD_mean[0], cD_std[0],
                    Entropy_D[0], Entropy_A[0], Entropy_Welch, Entropy_fft,
                    Kurt, Skewness, Mean, Median, Spectral_mean, hjorth_activity,
                    hjorth_mobility, hjorth_complexity, Std, coeff_var,
                    diff_mean1, diff_mean2, diff_max1, diff_max2, slope_mean, 
                    slope_var, Pow_max_Total, Pow_max_Delta, Pow_max_Theta_low,
                    Pow_max_Theta_high, Pow_max_Alpha, Pow_max_Beta, Pow_max_Sigma,
                    Pow_max_Sigma_slow]
            
            Features = np.row_stack((Features,feat))
            
        #%% Replace the NaN values of features with the mean of each feature column
        print('Features were successfully extracted in: {} secs'.format(time.time()-tic))
        
        aa, bb = np.where(np.isnan(Features))
        for j in np.arange(int(len(aa))):
            Features[aa[j],bb[j]] = np.nanmean(Features[:,bb[j]])
        print('the NaN values were successfully replaced with the mean of related feature.')    
        #%% Normalizing features
        Feat_train = Features[:int(len(X_train)),:]
        Feat_test = Features[int(len(X_train)):,:]
        sc = StandardScaler()
        Feat_train = sc.fit_transform(Feat_train)
        Feat_test = sc.transform(Feat_test)
        
        #%% Shuffle train and test data with rand perumtation
        rp_train = np.random.RandomState(seed=42).permutation(len(Feat_train))
        rp_test  = np.random.RandomState(seed=42).permutation(len(Feat_test))
        
        Feat_train_rp = Feat_train[rp_train,:]
        Feat_test_rp  = Feat_test[rp_test,:]
        y_train_rp    = ytrain[rp_train,:]
        y_test_rp     = ytest[rp_test,:]
        
        X_train = Feat_train_rp
        X_test  = Feat_test_rp 
        y_train = y_train_rp
        y_test  = y_test_rp
        
        return X_train, X_test, y_train, y_test
    
    
    #%% Feature extraction PER_Subject
    def FeatureExtraction_per_subject(self, Input_data, fs):
        
        ''' n_time_dependence_epochs: number of previous and subsequent epochs
        to consider with the current epoch to account for time-dependence'''
        # Loading data section
        # Load data
        tic = time.time() 

        fs = self.fs #Hz
        T  = self.T #sec
        
        x = Input_data
        X = np.transpose(x)
# =============================================================================
#         X = x.flatten('F')
#         
#         #%% Filtering section
#         ## Defining preprocessing function ##
#         def butter_bandpass_filter(data, lowcut, highcut, fs, order = 2):
#             nyq = 0.5 * fs
#             low = lowcut /nyq
#             high = highcut/nyq
#             b, a = butter(order, [low, high], btype='band')
#             #print(b,a)
#             y = lfilter(b, a, data)
#             return y
#         
#         # Apply filter
#         X = butter_bandpass_filter(data=X, lowcut=.1, highcut=30, fs=fs, order=2)
#         
#         #%% Reshaping data per epoch
#         X = np.reshape(X, (int(len(X) / (fs*T)), fs*T))
# =============================================================================
        
        #%% Feature Extraction section
        
        # Defining EEG bands:
        eeg_bands = {'Delta' : (0.5, 4),
                 'Theta_low' : (4  , 6),
                 'Theta_high': (6  , 8),
                 'Alpha'     : (8  , 11),
                 'Beta'      : (16 , 24),
                 'Sigma'     : (12 , 15),
                 'Sigma_slow': (10 , 12)}
        
        # Initializing variables of interest
        eeg_band_fft      = dict()
        freq_ix           = dict()
        freq_ix_welch     = dict()
        Features = np.empty((0, 76))
        # Settings of peridogram    
        Window = 'hann'
        # zero-padding added with respect to (Nfft=2^(nextpow2(len(window))))
        Nfft = 2 ** 15 
        # Defining freq. resoultion
        fm, _ = periodogram(x = X[0,:], fs = fs, nfft = Nfft , window = Window)  
        tic = time.time()
        
        # Finding the index of different freq bands with respect to "fm" PERIODOGRAM #
        for band in eeg_bands:
            freq_ix[band] = np.where((fm >= eeg_bands[band][0]) &   
                               (fm <= eeg_bands[band][1]))[0]    
            
        window_len = 4 # secs
        ff, _      = welch(x = X[0,:], fs = fs, window = 'hann', nperseg = fs*window_len)
        
        # Finding the index of different freq bands with respect to "fm" WELCH#
        for band in eeg_bands:
            freq_ix_welch[band] = np.where((ff >= eeg_bands[band][0]) &   
                               (ff <= eeg_bands[band][1]))[0]    
            
        # Defining for loop to extract features per epoch
        for i in np.arange(len(X)):
            
            data = X[i,:]
                        
            ### Initialization for wavelet 
            
            # 4th appr coef
            cA_values4  = []
            
            # 1st to 4th det coef
            cD_values4  = []
            cD_values3  = []
            cD_values2  = []
            cD_values1  = []
            
            # mean and std of appr coef
            cA_mean4    = []
            cA_std4     = []
            
            # mean and std of det coefs
            cD_mean4    = []
            cD_std4     = []
            cD_mean3    = []
            cD_std3     = []
            cD_mean2    = []
            cD_std2     = []
            cD_mean1    = []
            cD_std1     = []
            
            # Energy of appr coefs
            cA_Energy4  = []
            
            # Energy of det coefs
            cD_Energy4  = []
            cD_Energy3  = []
            cD_Energy2  = []
            cD_Energy1  = []
            
            # Entropy of appr coef
            Entropy_A4  = []
            
            # Entropy of det coefs
            Entropy_D4  = []
            Entropy_D3  = []
            Entropy_D2  = []
            Entropy_D1  = []

            first_diff = np.zeros(len(data)-1)
            
            ''' Power of signal --> Peridogram with padding'''
            # Compute the "total" power inside the investigational window
            _ , pxx = periodogram(x = data, fs = fs, nfft = Nfft , window = Window) 
            freq_resolu_per= fm[1] - fm[0]
            
            pow_total      = simps(pxx, dx = freq_resolu_per)
            Pow_Delta      = simps(pxx[freq_ix['Delta']], dx = freq_resolu_per) 
            Pow_Theta_low  = simps(pxx[freq_ix['Theta_low']], dx = freq_resolu_per) 
            Pow_Theta_high = simps(pxx[freq_ix['Theta_high']], dx = freq_resolu_per) 
            Pow_Alpha      = simps(pxx[freq_ix['Alpha']], dx = freq_resolu_per) 
            Pow_Beta       = simps(pxx[freq_ix['Beta']], dx = freq_resolu_per)  
            Pow_Sigma      = simps(pxx[freq_ix['Sigma']], dx = freq_resolu_per) 
            Pow_Sigma_slow = simps(pxx[freq_ix['Sigma_slow']], dx = freq_resolu_per)  
            
            
            '''Power ratio in differnt freq ranges (Periodogram)''' 
            # Total pow is defined form 0.5 - 20 Hz
            Pow_Delta_ratio      = Pow_Delta / pow_total
            Pow_Theta_low_ratio  = Pow_Theta_low / pow_total
            Pow_Theta_high_ratio = Pow_Theta_high / pow_total
            Pow_Alpha_ratio      = Pow_Alpha / pow_total
            Pow_Beta_ratio       = Pow_Beta / pow_total
            Pow_Sigma_ratio      = Pow_Sigma / pow_total
            Pow_Sigma_slow_ratio = Pow_Sigma_slow / pow_total
            
            '''Apply WELCH to see the ABSOLUTE power in each freq band'''
            window_len = 4 # secs
            ff, Psd             = welch(x = data, fs = fs, window = 'hann', nperseg = fs*window_len)
            freq_resolu_welch   = ff[1] - ff[0]
            
            Pow_welch_Total       = simps(Psd, dx = freq_resolu_welch)
            Pow_welch_Delta       = simps(Psd[freq_ix_welch['Delta']], dx = freq_resolu_welch)
            Pow_welch_Theta_low   = simps(Psd[freq_ix_welch['Theta_low']], dx = freq_resolu_welch)
            Pow_welch_Theta_high  = simps(Psd[freq_ix_welch['Theta_high']], dx = freq_resolu_welch)
            Pow_welch_Alpha       = simps(Psd[freq_ix_welch['Alpha']], dx = freq_resolu_welch)
            Pow_welch_Beta        = simps(Psd[freq_ix_welch['Beta']], dx = freq_resolu_welch)
            Pow_welch_Sigma       = simps(Psd[freq_ix_welch['Sigma']], dx = freq_resolu_welch)
            Pow_welch_Sigma_slow  = simps(Psd[freq_ix_welch['Sigma_slow']], dx = freq_resolu_welch)
            
            '''Apply WELCH to see the RELATIVE power in each freq band'''

            Pow_welch_Delta_rel       = Pow_welch_Delta / Pow_welch_Total
            Pow_welch_Theta_low_rel   = Pow_welch_Theta_low / Pow_welch_Total
            Pow_welch_Theta_high_rel  = Pow_welch_Theta_high / Pow_welch_Total
            Pow_welch_Alpha_rel       = Pow_welch_Alpha / Pow_welch_Total
            Pow_welch_Beta_rel        = Pow_welch_Beta / Pow_welch_Total
            Pow_welch_Sigma_rel       = Pow_welch_Sigma / Pow_welch_Total
            Pow_welch_Sigma_slow_rel  = Pow_welch_Sigma_slow / Pow_welch_Total
            
            ''' Spectral Entropy '''
            Entropy_Welch = spectral_entropy(x = data, sf=fs, method='welch', nperseg = fs* window_len)
            Entropy_fft   = spectral_entropy(x = data, sf=fs, method='fft')
               
            ''' Wavelet Decomposition ''' 
            # Extract 4 det compositions
            coeffs = pywt.wavedec(data, 'db10', level=4)
            cA4, cD4, cD3, cD2, cD1 = coeffs
            
            # Appending appr values
            cA_values4.append(cA4)
            
            # Appending det coefs values
            cD_values4.append(cD4)
            cD_values3.append(cD3)
            cD_values2.append(cD2)
            cD_values1.append(cD1)
            
            # Calculate mean and std of appr coefs
            cA_mean4.append(np.mean(cA_values4))
            cA_std4.append(np.std(cA_values4))
            
            # Calculate mean of det coefs
            cD_mean4.append(np.mean(cD_values4))
            cD_mean3.append(np.mean(cD_values3))
            cD_mean2.append(np.mean(cD_values2))
            cD_mean1.append(np.mean(cD_values1))
            
            # Calculate std of det coefs
            cD_std4.append(np.std(cD_values4))
            cD_std3.append(np.std(cD_values3))
            cD_std2.append(np.std(cD_values2))
            cD_std1.append(np.std(cD_values1))
            
            # Calculate energy of appr coefs
            cA_Energy4.append(np.sum(np.square(cA_values4)))
            
            # Calculate energy of det coefs
            cD_Energy4.append(np.sum(np.square(cD_values4)))
            cD_Energy3.append(np.sum(np.square(cD_values3)))
            cD_Energy2.append(np.sum(np.square(cD_values2)))
            cD_Energy1.append(np.sum(np.square(cD_values1)))
            
            # Entropy of appr coefs
            Entropy_A4.append(np.sum(np.square(cA_values4) * np.log(np.square(cA_values4))))
            
            # Entropy of det coefs
            Entropy_D4.append(np.sum(np.square(cD_values4) * np.log(np.square(cD_values4))))
            Entropy_D3.append(np.sum(np.square(cD_values3) * np.log(np.square(cD_values3))))
            Entropy_D2.append(np.sum(np.square(cD_values2) * np.log(np.square(cD_values2))))
            Entropy_D1.append(np.sum(np.square(cD_values1) * np.log(np.square(cD_values1))))

            
            ''' Hjorth Parameters '''
            hjorth_activity     = np.var(data)
            diff_input          = np.diff(data)
            diff_diffinput      = np.diff(diff_input)
            hjorth_mobility     = np.sqrt(np.var(diff_input)/hjorth_activity)
            hjorth_diffmobility = np.sqrt(np.var(diff_diffinput)/np.var(diff_input))
            hjorth_complexity   = hjorth_diffmobility / hjorth_mobility
             
            ''' Statisctical features'''
            Kurt     = kurtosis(data, fisher = False)
            Skewness = skew(data)
            Mean     = np.mean(data)
            Median   = np.median(data)
            Std      = np.std(data)
            ''' Coefficient of variation '''
            coeff_var = Std / Mean
            
            ''' First and second difference mean and max '''
            sum1  = 0.0
            sum2  = 0.0
            Max1  = 0.0
            Max2  = 0.0
            for j in range(len(data)-1):
                    sum1     += abs(data[j+1]-data[j])
                    first_diff[j] = abs(data[j+1]-data[j])
                    
                    if first_diff[j] > Max1: 
                        Max1 = first_diff[j] # fi
                        
            for j in range(len(data)-2):
                    sum2 += abs(first_diff[j+1]-first_diff[j])
                    if abs(first_diff[j+1]-first_diff[j]) > Max2 :
                    	Max2 = first_diff[j+1]-first_diff[j] 
                        
            diff_mean1 = sum1 / (len(data)-1)
            diff_mean2 = sum2 / (len(data)-2) 
            diff_max1  = Max1
            diff_max2  = Max2
            
            ''' Variance and Mean of Vertex to Vertex Slope '''
            t_max   = argrelextrema(data, np.greater)[0]
            amp_max = data[t_max]
            t_min   = argrelextrema(data, np.less)[0]
            amp_min = data[t_min]
            tt      = np.concatenate((t_max,t_min),axis=0)
            if len(tt)>0:
                tt.sort() #sort on the basis of time
                h=0
                amp = np.zeros(len(tt))
                res = np.zeros(len(tt)-1)
                
                for l in range(len(tt)):
                        amp[l] = data[tt[l]]
                        
                out = np.zeros(len(amp)-1)     
                 
                for j in range(len(amp)-1):
                    out[j] = amp[j+1]-amp[j]
                amp_diff = out
                
                out = np.zeros(len(tt)-1)  
                
                for j in range(len(tt)-1):
                    out[j] = tt[j+1]-tt[j]
                tt_diff = out
                
                for q in range(len(amp_diff)):
                        res[q] = amp_diff[q]/tt_diff[q] #calculating slope        
                
                slope_mean = np.mean(res) 
                slope_var  = np.var(res)   
            else:
                slope_var, slope_mean = 0, 0
                
            ''' Spectral mean '''
            Spectral_mean = 1 / (freq_ix['Beta'][-1] - freq_ix['Delta'][0]) * (Pow_Delta + 
                    Pow_Theta_low + Pow_Theta_high + Pow_Alpha + Pow_Beta + 
                    Pow_Sigma) 
            """ 
            ''' Correlation Dimension Feature '''
            try:
                cdf = nolds.corr_dim(data,1)
            except np.linalg.LinAlgError:
                cdf = np.NaN
              
            """ 
            
# =============================================================================
#             ''' Hurst component '''
#             try:
#                 Hurst = pyeeg.hurst(data)
#             except np.linalg.LinAlgError:
#                 Hurst = np.NaN
#                 
#             ''' Detrended Fluctuation Analysis ''' 
#             try:
#                 DFA = pyeeg.dfa(data)
#             except np.linalg.LinAlgError:
#                 DFA = np.NaN
#             
#             '''Compute Petrosian Fractal Dimension '''
#             try: 
#                 PFD = pyeeg.pfd(data, D=None)
#             except np.linalg.LinAlgError:
#                 PFD = np.NaN
#             
# =============================================================================
            '''Waveform length(WL)'''
            WL = sum(abs(np.diff(data)))
            
            '''Zerocrossing(ZC) '''
            zero_crossings = np.where(np.diff(np.signbit(data)))[0]
            num_ZC = (len(zero_crossings))
            
            ''' Mean absolute value ''' 
            MAV = sum(np.abs(data)) / len(data)
            
            '''Simple Square Integral (SSI)'''
            SSI = sum(np.abs(data)**2)
            
            ''' Root mean square '''
            rms = np.sqrt(1 / len(data) * sum(np.abs(data)**2))

            ''' Spectral edge frequency features --> SEF50 and SEF95'''
            # Imtiaz et al. proposed the freq band of investigation: 8 - 16 Hz
            data_SEF =  self.butter_bandpass_filter(data=data, lowcut=8, highcut=16, fs=fs, order=2)
            
            # compute fft^2
            FFT_ = abs(np.fft.fft(data, n = None))
            FFT_ = FFT_[0:int(len(FFT_)/2)+1] 
            FFT_ = abs(FFT_ ** 2)
            
            # Compute frequency samples
            freq_fft, _ = periodogram(x = data_SEF, fs = fs, nfft = None , window = Window)  
            
            # Defining accumulative and total power
            acc_pow = 0
            tot_pow = 0
            
            # Calculating summation of all powers
            for i,j in enumerate(freq_fft):
                tot_pow = tot_pow + FFT_[i]
            
            # defining SEF50
            for i,j in enumerate(freq_fft):
                acc_pow = acc_pow + FFT_[i]    
                if acc_pow >= .5 * tot_pow:
                    SEF50 = j
                    break
                
            # Defining SEF 95   
            acc_pow = 0
            for i,j in enumerate(freq_fft):
                acc_pow = acc_pow + FFT_[i]    
                if acc_pow >= .95 * tot_pow:
                    SEF95 = j
                    break
            del acc_pow, tot_pow
            
            ''' Spectral edge frequency features --> SEFd'''
            
            # definig subepochs : create window size of 2 secs
            time_win     = 2
            samp_per_win = fs * time_win
            
            # each column is a subepoch of 2s
            data_SEF_per_win = np.reshape(data_SEF,
                                          (samp_per_win, int(len(data_SEF) / samp_per_win)), order='F' )
            
            SEFds = []
            for j in np.arange(0, np.shape(data_SEF_per_win)[1]):
                
                # Calculating fft
                data_tmp = data_SEF_per_win[:,j]
                C        = abs(np.fft.fft(data_tmp, n = 512))
                C        = C[0:int(len(C)/2)+1] 
                C        = abs(C ** 2)
                freqs, _ = periodogram(x = data_tmp, fs = fs, nfft = 512 , window = Window)  
                
                # Calculating SEFds
                acc_pow = 0
                tot_pow = 0
                # Calculating summation of all powers
                for i,j in enumerate(freqs):
                    tot_pow = tot_pow + C[i]
                
                # defining SEF50
                for i,j in enumerate(freqs):
                    acc_pow = acc_pow + C[i]    
                    if acc_pow >= .5 * tot_pow:
                        SEF50_tmp = j
                        break
                # Defining SEF 95   
                acc_pow = 0
                for i,j in enumerate(freqs):
                    acc_pow = acc_pow + C[i]    
                    if acc_pow >= .95 * tot_pow:
                        SEF95_tmp = j
                        break
                SEFd_tmp = SEF95_tmp - SEF50_tmp
                # Comuting SEFds array
                SEFds.append(SEFd_tmp)
                del acc_pow, tot_pow, SEF95_tmp, SEF50_tmp
                
            SEFd = 1 / len(SEFds)  * sum(SEFds)
            
            
            
            
            ''' Wrapping up featureset '''
            feat = [pow_total, Pow_Delta, Pow_Theta_low, Pow_Theta_high, Pow_Alpha,
                    Pow_Beta, Pow_Sigma, Pow_Sigma_slow, Pow_Delta_ratio, Pow_Theta_low_ratio, 
                    Pow_Theta_high_ratio, Pow_Alpha_ratio,
                    Pow_Beta_ratio, Pow_Sigma_ratio, Pow_Sigma_slow_ratio, cA_mean4[0], cA_std4[0],
                    cD_mean4[0],cD_mean3[0], cD_mean2[0], cD_mean1[0], cD_std4[0],
                    cD_std3[0], cD_std2[0], cD_std1[0], cA_Energy4[0], cD_Energy4[0],
                    cD_Energy3[0], cD_Energy2[0], cD_Energy1[0], Entropy_A4[0],
                    Entropy_D4[0], Entropy_D3[0], Entropy_D2[0], Entropy_D1[0],
                    Entropy_Welch, Entropy_fft, Kurt, Skewness, Mean, Median, 
                    Spectral_mean, hjorth_activity, hjorth_mobility, 
                    hjorth_complexity, Std, coeff_var, diff_mean1, diff_mean2, 
                    diff_max1, diff_max2, slope_mean, slope_var, Pow_welch_Total, 
                    Pow_welch_Delta, Pow_welch_Theta_low, Pow_welch_Theta_high, 
                    Pow_welch_Alpha, Pow_welch_Beta, Pow_welch_Sigma, Pow_welch_Sigma_slow,
                    Pow_welch_Delta_rel, Pow_welch_Theta_low_rel, Pow_welch_Theta_high_rel, 
                    Pow_welch_Alpha_rel, Pow_welch_Beta_rel, Pow_welch_Sigma_rel, 
                    Pow_welch_Sigma_slow_rel, SEF50, SEF95, SEFd, 
                    WL, num_ZC, MAV,  SSI, rms] #  DFA, PFD, Hurst
            
            Features = np.row_stack((Features,feat))
        
        # Replace the NaN values of features with the mean of each feature column
        aa, bb = np.where(np.isnan(Features))
        for j in np.arange(int(len(aa))):
            Features[aa[j],bb[j]] = np.nanmean(Features[:,bb[j]])
        print('the NaN values were successfully replaced with the mean of related feature.')   
        
        return Features
    
    #%% Acceleration feature extraction
    def Acc_feature_extraction(self, AccNorm, Acc, fs, axes_acc_status = 'deactive'):
        "Acc is numpy.array including 3 components, namely X-, Y-, Z- components of acceleration. One can read these using self.Read_Acceleration_data. \
        axes_acc_status: shows weather feature extraction is of interest in a aces-based manner."
        
        # init feature list
        Features = np.empty((0, 22))
        
        # Take X-y-z components of Acc and split them
# =============================================================================
#         x_acc = Acc[0].flatten()
#         y_acc = Acc[1].flatten()
#         z_acc = Acc[2].flatten()
# =============================================================================
        
        # Flattening Acc-Norm
        AccNorm = AccNorm.flatten()
# =============================================================================
#         x_acc   = x_acc[0:x_acc.shape[0] - x_acc.shape[0]%self.len_epoch]
#         y_acc   = y_acc[0:y_acc.shape[0] - y_acc.shape[0]%self.len_epoch]
#         z_acc   = z_acc[0:z_acc.shape[0] - z_acc.shape[0]%self.len_epoch]
# =============================================================================

        
        #### ======================  pre-processing ========================= #
        # Band-pass filtering to remove baseline and unnecessary noise
        AccNorm_filt = self.butter_bandpass_filter(AccNorm, lowcut = .25, highcut = 80, fs= self.fs)
        
        # low-pass filtering axes-based acceleration signal
# =============================================================================
#         x_acc_filt = self.butter_lowpass_filter(x_acc, cutoff = 80, fs = self.fs, order = 2)
#         y_acc_filt = self.butter_lowpass_filter(y_acc, cutoff = 80, fs = self.fs, order = 2)
#         z_acc_filt = self.butter_lowpass_filter(z_acc, cutoff = 80, fs = self.fs, order = 2)
# =============================================================================
        
        #### ================= Reshaping data per epoch ================== ####
        AccNorm_filt = np.reshape(AccNorm_filt, (int(len(AccNorm_filt) / (self.fs*self.T)), self.fs*self.T))
# =============================================================================
#         x_acc_filt   = np.reshape(x_acc_filt, (int(len(x_acc_filt) / (self.fs*self.T)), self.fs*self.T))
#         y_acc_filt   = np.reshape(y_acc_filt, (int(len(y_acc_filt) / (self.fs*self.T)), self.fs*self.T))
#         z_acc_filt   = np.reshape(z_acc_filt, (int(len(z_acc_filt) / (self.fs*self.T)), self.fs*self.T))
# =============================================================================
        
        ##### ============== Feature extraction: AccNorm ================ #####
        # init 
        window = 'hann'
        Nfft = 2 ** 15
        
        # Defining for loop to extract features per epoch
        for i in np.arange(len(AccNorm_filt)):
            
            # pick the current epoch
            data = AccNorm_filt[i,:]
            
            # Statistical feats : NormAcc
            mean_AccNorm = np.mean(data)
            std_AccNorm  = np.std(data)       
            var_AccNorm  = np.var(data)
            skew_AccNorm = skew(data)
            kurt_AccNorm = kurtosis(data)
            
            # Power
            fm , pxx = periodogram(x = data, fs = fs, nfft = Nfft , window = window) 
            freq_resolu_per= fm[1] - fm[0]  
            power_AccNorm      = simps(pxx, dx = freq_resolu_per)
            
            # Initialization for wavelet 
            cA_values  = []
            cD_values  = []
            cA_mean    = []
            cA_std     = []
            cA_Energy  = []
            cD_mean    = []
            cD_std     = []
            cD_Energy  = []
            Entropy_D  = []
            Entropy_A  = []
            
            # Wavelet Decomposition
            cA,cD=pywt.dwt(data,'coif1')
            cA_values.append(cA)
            cD_values.append(cD)
            cA_mean.append(np.mean(cA_values))
            cA_std.append(np.std(cA_values))
            cA_Energy.append(np.sum(np.square(cA_values)))
            cD_mean.append(np.mean(cD_values))
            cD_std.append(np.std(cD_values))
            cD_Energy.append(np.sum(np.square(cD_values)))
            Entropy_D.append(np.sum(np.square(cD_values) * np.log(np.square(cD_values))))
            Entropy_A.append(np.sum(np.square(cA_values) * np.log(np.square(cA_values))))
            
            # Hjorth Parameters
            hjorth_activity     = np.var(data)
            diff_input          = np.diff(data)
            diff_diffinput      = np.diff(diff_input)
            hjorth_mobility     = np.sqrt(np.var(diff_input)/hjorth_activity)
            hjorth_diffmobility = np.sqrt(np.var(diff_diffinput)/np.var(diff_input))
            hjorth_complexity   = hjorth_diffmobility / hjorth_mobility
            
            # Waveform length(WL)
            WL = sum(abs(np.diff(data)))
            
            # Zerocrossing(ZC) 
            zero_crossings = np.where(np.diff(np.signbit(data)))[0]
            num_ZC = (len(zero_crossings))
            
            # Mean absolute value
            MAV = sum(np.abs(data)) / len(data)
            
            # Simple Square Integral (SSI)
            SSI = sum(np.abs(data)**2)
            
            # Root mean square
            rms = np.sqrt(1 / len(data) * sum(np.abs(data)**2))
            
            #### ==================== Wrapping up ======================== ####
            
            feat= [mean_AccNorm, std_AccNorm, var_AccNorm, skew_AccNorm, kurt_AccNorm,
                   power_AccNorm, cA_mean[0], cA_std[0], cA_Energy[0], cD_mean[0], cD_std[0], cD_Energy[0],
                   Entropy_D[0], Entropy_A[0], hjorth_mobility, hjorth_diffmobility, hjorth_complexity,
                   WL, num_ZC, MAV, SSI, rms]
            
            Features = np.row_stack((Features,feat))
            
        return Features
        #%% Normalizing features
    
    def SaveFeatureSet(self, X, y, path, filename):
        path     = path  
        filename = filename
        with h5py.File((path+filename + '.h5'), 'w') as wf:
            dset = wf.create_dataset('featureset', X.shape, data = X)
            dset = wf.create_dataset('labels', y.shape, data = y)
        print('Features have been successfully saved!')

        ######################## DEFINING FEATURE SELECTION METHODS ######################
    #%% Feature selection section - 1. Boruta method
    def FeatSelect_Boruta(self, X,y, max_iter = 50, max_depth = 7):
        #import lib
        tic = time.time()
        from boruta import BorutaPy
        #instantiate an estimator for Boruta. 
        rf = RandomForestClassifier(n_jobs=-1, class_weight=None, max_depth=max_depth)
        # Initiate Boruta object
        feat_selector = BorutaPy(rf, max_iter = max_iter, n_estimators='auto', verbose=2, random_state=0)
        # fir the object
        feat_selector.fit(X=X, y=y)
        # Find index of selected feats
        selected_feats_ind = feat_selector.support_
        # Check selected features
        print(selected_feats_ind)
        # Select the chosen features from our dataframe.
        Feat_selected = X[:,selected_feats_ind]
        print(f'Selected Feature Matrix Shape {Feat_selected.shape}')
        toc = time.time()
        print(f'Feature selection using Boruta took {toc-tic}')
        ranks = feat_selector.ranking_
        
        return ranks, Feat_selected, selected_feats_ind
    
    #%% Feature selection using LASSO as regression penalty
    def FeatSelect_LASSO(self, X, y, C = 1):
        from sklearn.linear_model import LogisticRegression
        tic = time.time()
        from sklearn.linear_model import Lasso
        from sklearn.feature_selection import SelectFromModel
        #create object
        sel_ = SelectFromModel(LogisticRegression(C=C, penalty='l1'))
        sel_.fit(X, y)
        # find the selected feature indices
        selected_ = sel_.get_support()
        # Select releavnt features
        Feat_selected = X[:, selected_]
        toc = time.time()
        print(f'Total time for LASSO feature selection was: {toc-tic}')
        print(f'total of {len(Feat_selected)} was selected out of {np.shape(X)[1]} features')
        return Feat_selected
    
    #%% Feature Selection with Univariate Statistical Tests
    def FeatSelect_ANOVA(self, X, y, k=20):
        tic = time.time()
        from sklearn.feature_selection import SelectKBest, f_classif
        test = SelectKBest(score_func=f_classif, k=k)
        fit = test.fit(X, y)
        # summarize scores
        print(f' scores: {fit.scores_}')
        Feat_selected = fit.transform(X)
        toc = time.time()
        print(f'Total time for ANOVA feature selection was: {toc-tic}')

        return Feat_selected
    #%% # Recursive Feature Elimination
    def FeatSelect_Recrusive(self, X,y, k = 20):
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
            # feature extraction
        model = LogisticRegression(solver='lbfgs')
        rfe = RFE(model, n_features_to_select = k)
        fit = rfe.fit(X, y)
        ranks = fit.ranking_
        selected_ = fit.support_
        Feat_selected = X[:, selected_]
        print("Num Features: %d" % fit.n_features_)
        print("Selected Features: %s" % fit.support_)
        print("Feature Ranking: %s" % fit.ranking_)
        
        return ranks, Feat_selected

    #%% PCA
    def FeatSelect_PCA(self, X, y, n_components = 5):
        tic = time.time()
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        PCA_out = pca.fit(X)
        # summarize components
        print("Explained Variance: %s" % PCA_out.explained_variance_ratio_)
        print(PCA_out.components_)
        toc = time.time()
        print(f'Total time for PCA feature selection was: {toc-tic}')

        return PCA_out
    
    ######################## DEFINING SUPERVISED CLASSIFIERs ######################
    
    #%% Naive Bayes
    def Naive_bayes_clissifer(self, X_train, y_train,X_test, y_test):
        from sklearn.naive_bayes import GaussianNB
        classifier_gnb = GaussianNB()
        
        if np.shape(y_train)[1] > 1:
            y_train = self.binary_to_single_column_label(y_train)
            y_train = np.ravel(y_train)
            
        classifier_gnb.fit(X_train, y_train)
        y_pred = classifier_gnb.predict(X_test)

        return y_pred
    #%% Random Forest
    def RandomForest_Modelling(self, X_train, y_train,X_test, y_test, n_estimators = 500):
        
        classifier_RF = RandomForestClassifier(n_jobs=-1, n_estimators = n_estimators)
        classifier_RF.fit(X_train, y_train)
        y_pred = classifier_RF.predict(X_test)

        return y_pred
    
    #%% Kernel SVM
    def KernelSVM_Modelling(self,X_train, y_train,X_test, y_test, kernel='rbf'):
        tic = time.time()
        from sklearn.svm import SVC
        classifier_SVM = SVC(kernel = kernel)
        #results_SVM = cross_validate(estimator = classifier_SVM, X = X, 
        #                         y = y, scoring = scoring, cv = KFold(n_splits = cv))
        #Acc_cv10_SVM = accuracies_SVM.mean()
        #std_cv10_SVM = accuracies_SVM.std()
        #print(f'Cross validation finished: Mean Accuracy {Acc_cv10_SVM} +- {std_cv10_SVM}')
# =============================================================================
#         if np.shape(y_train)[1] > 1:
#             y_train = self.binary_to_single_column_label(y_train)
#             y_train = np.ravel(y_train)
# =============================================================================
        classifier_SVM.fit(X_train, y_train)
        y_pred = classifier_SVM.predict(X_test)
        print('Cross validation for SVM took: {} secs'.format(time.time()-tic))
        return y_pred
    
    
    #%% Logistic regression
    def LogisticRegression_Modelling(self, X, y, scoring, cv = 10, max_iter = 500):
        tic = time.time()
        from sklearn.linear_model import LogisticRegression
        classifier_LR = LogisticRegression(max_iter = max_iter)
        results_LR = cross_validate(estimator = classifier_LR, X = X, 
                                 y = y, scoring = scoring, cv = KFold(n_splits = cv))
        #Acc_cv10_LR = accuracies_LR.mean()
        #std_cv10_LR = accuracies_LR.std()
        #print(f'Cross validation finished: Mean Accuracy {Acc_cv10_LR} +- {std_cv10_LR}')
        print('Cross validation for LR took: {} secs'.format(time.time()-tic))
        return results_LR
    #%% XGBoost
    def XGB_Modelling(self, X_train, y_train,X_test, y_test, n_estimators = 1000, 
                      max_depth=3, learning_rate=.1):
        tic = time.time()
        from xgboost import XGBClassifier
        classifier_xgb = XGBClassifier(n_jobs=-1, n_estimators = n_estimators, max_depth = max_depth,
                                       learning_rate = learning_rate)
        
# =============================================================================
#         if np.shape(y_train)[1] > 1:
#             y_train = self.binary_to_single_column_label(y_train)
#             y_train = np.ravel(y_train)
# =============================================================================
            
        classifier_xgb.fit(X_train, y_train)
        y_pred = classifier_xgb.predict(X_test)

        y_pred = np.expand_dims(y_pred, axis=1)
        '''
        if plot_confusion == True:
            from sklearn.metrics import plot_confusion_matrix
            labels = ['Wake', 'N1', 'N2','N3','REM']
            if np.shape(y_pred)[1] != np.shape(y_test)[1]:
                y_true = self.binary_to_single_column_label(y_test)
            plot_confusion_matrix(classifier_xgb, X_test, y_true, normalize='all')  # doctest: +SKIP
            plt.show()  # doctest: +SKIP
            '''
        return y_pred
    
    #%% Kernel LGBM
    def LightGBM_Modelling(self, X_train, y_train, X_test, y_test,\
                           boosting_type='gbdt',\
                           n_estimators=400,\
                           max_depth=5, num_leaves=90,\
                           colsample_bytree=0.5,importance_type='gain'):
        tic = time.time()
        from lightgbm import LGBMClassifier
        params = dict(
            boosting_type = boosting_type,
            n_estimators = n_estimators,
            max_depth = max_depth,
            num_leaves = num_leaves,
            colsample_bytree = colsample_bytree,
            importance_type = importance_type,
        )
        classifier_LGBM = LGBMClassifier(**params)
        classifier_LGBM.fit(X_train, y_train)
        y_pred = classifier_LGBM.predict(X_test)
        print('Cross validation for LGBM took: {} secs'.format(time.time() - tic))
        return y_pred, classifier_LGBM
        
    #%% ANN
    def ANN_classifier(self, X_train, y_train,X_test, y_test, path, units_h1, units_h2, units_output, activation_out,
                  init = 'uniform', activation = 'relu', optimizer = 'adam',
                  loss = 'mean_squared_logarithmic_error', metrics = ['accuracy'],
                  h3_status = 'deactive', units_h3 = 50, epochs = 10, batch_size = 32,\
                  patience = 5, best_model_name = 'best_model_ANN.h5'):
        # Importing the Keras libraries and packages
        import tensorflow as tf
        from tensorflow import keras
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, BatchNormalization
        from keras.callbacks import EarlyStopping, ModelCheckpoint
        
        # Initialising the ANN
        classifier = Sequential()
        
        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units = units_h1, init = init, activation = activation, input_dim = np.shape(X_train)[1]))
        
        # Add dropout and batch normalization
        classifier.add(BatchNormalization())
        classifier.add(Dropout(0.3))
        
        # Adding the second hidden layer
        classifier.add(Dense(units = units_h2 , init = init, activation = activation))
        
        # Add dropout and batch normalization
        classifier.add(BatchNormalization())
        classifier.add(Dropout(0.3))
        
        # Adding the third hidden layer
        if h3_status == 'active':
            classifier.add(Dense(units = units_h3 , init = init, activation = activation))
            
        # Adding the output layer
        classifier.add(Dense(units = units_output, init = init, activation = activation_out))
        
        # Compiling the ANN
        classifier.compile(optimizer = optimizer, loss = loss , metrics = metrics)
        
        #callbacks
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        mc = ModelCheckpoint(path+ best_model_name, \
                          monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        
        # Fit and train
        history = classifier.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=batch_size, \
                            verbose=1, callbacks=[es, mc])
        
        # Predict
        y_pred = classifier.predict_classes(X_test)
        
        return y_pred, history
    
    #%% Extra randomized trees
    def Extra_randomized_trees(self, X_train, y_train, X_test,y_test, n_estimators= 250, max_depth = None, min_samples_split =2,
                               max_features="sqrt"):
        
        from sklearn.ensemble import ExtraTreesClassifier
        
        clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,
          min_samples_split=min_samples_split, random_state=0)
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
        return y_pred
    #%% Ada boost
    def ADAboost_Modelling(self, X_train, y_train,X_test, y_test, n_estimators = 250):
        
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=n_estimators)
        
        if np.shape(y_train)[1] > 1:
            y_train = self.binary_to_single_column_label(y_train)
            y_train = np.ravel(y_train)
            
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # Add dimension
        y_pred = np.expand_dims(y_pred, axis=1)
        
        return y_pred
    #%% Gradient boosting ensemble method
    def gradient_boosting_classifier(self,  X_train, y_train,X_test, y_test, 
                                     n_estimators = 250, learning_rate= 1.0, max_depth=1):
    
        from sklearn.ensemble import GradientBoostingClassifier
        
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
         max_depth=max_depth, random_state=0)
        
        if np.shape(y_train)[1] > 1:
            y_train = self.binary_to_single_column_label(y_train)
            y_train = np.ravel(y_train)
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        y_pred = np.expand_dims(y_pred, axis=1)
        
        return y_pred
    
    #%% Ensemble voting classifiers
    def Ensemble_voting_classifier(self, X_train, y_train,X_test, y_test, n_estimators_RF=250,
                                   n_estimators_GBC=250,n_estimators_XGB=250,learning_rate_XGB = .1,
                                   max_depth_XGB = 3):
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
        from xgboost import XGBClassifier
        from sklearn.svm import SVC
        
        
        # Define separate classifiers
         
        classifier_SVM = SVC(kernel = 'rbf')
        classifier_RF  = RandomForestClassifier(n_estimators = n_estimators_RF)
        classifier_GBC = GradientBoostingClassifier(n_estimators=n_estimators_GBC)
        classifier_XGB = XGBClassifier(n_estimators = n_estimators_XGB, max_depth = max_depth_XGB,
                                       learning_rate = learning_rate_XGB)
        # Apply the voting classifier
        eclf = VotingClassifier(estimators=[('rf', classifier_RF), ('svm', classifier_SVM), 
                                            ('xgb',classifier_XGB), ('gbc', classifier_GBC)], voting='hard')
        
        # make y_train compatible
        if np.shape(y_train)[1] > 1:
            y_train = self.binary_to_single_column_label(y_train)
            y_train = np.ravel(y_train)
            
        # Fitting
        eclf.fit(X_train, y_train)
        
        # Prediction
        y_pred = eclf.predict(X_test)
        
        return y_pred
        
    #%% Stacked classifier
    def Stacking_classifier(self, X_train, y_train,X_test, y_test, n_estimators_RF=250,
                                   n_estimators_GBC=250,n_estimators_XGB=250,learning_rate_XGB = .1,
                                   max_depth_XGB = 3):
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
        from xgboost import XGBClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import StackingClassifier
        
        # Define separate classifiers

        classifier_SVM = SVC(kernel = 'rbf')
        classifier_RF  = RandomForestClassifier(n_estimators = n_estimators_RF)
        classifier_GBC = GradientBoostingClassifier(n_estimators=n_estimators_GBC)
        classifier_XGB = XGBClassifier(n_estimators = n_estimators_XGB, max_depth = max_depth_XGB,
                                       learning_rate = learning_rate_XGB)
        
        # Apply the voting classifier
        estimators=[('rf', classifier_RF),('xgb',classifier_XGB), ('gbc', classifier_GBC)]
        
        stacking_clf = StackingClassifier(estimators=estimators, final_estimator=classifier_SVM)
        
        # make y_train compatible
        if np.shape(y_train)[1] > 1:
            y_train = self.binary_to_single_column_label(y_train)
            y_train = np.ravel(y_train)
            
        # Fitting
        stacking_clf.fit(X_train, y_train)
        
        # Prediction
        y_pred = stacking_clf.predict(X_test)
        
        return y_pred
        
    #%% Evaluation using multi-label confusion matrix
    def multi_label_confusion_matrix(self,y_true, y_pred, print_results = 'on'):
        from sklearn.metrics import multilabel_confusion_matrix, cohen_kappa_score, accuracy_score
        
        try: 
            if np.shape(y_true)[1] != np.shape(y_pred)[1]:
                y_true = self.binary_to_single_column_label(y_true)
        except IndexError:
            y_true = self.binary_to_single_column_label(y_true)
        
        out_dic = dict()
        mcm     = multilabel_confusion_matrix(y_true, y_pred)
        tn      = mcm[:, 0, 0]
        tp      = mcm[:, 1, 1]
        fn      = mcm[:, 1, 0]
        fp      = mcm[:, 0, 1]
        Recall  = tp / (tp + fn)
        prec    = tp / (tp + fp)
        f1_sc   = 2 * Recall * prec / (Recall + prec)
        Acc     = (tp + tn) / (tp + fp + fn+ tn)
        kappa   = cohen_kappa_score(y_true, y_pred)
        Acc_all = accuracy_score(y_true, y_pred)
        mf1     = np.mean(f1_sc)
                
        if print_results == 'on':
            print(f'Overall Accuracy: {Acc_all}')
            print(f'mf1: {mf1}')
            print(f'Accuracy for Wake,N1,N2,N3,REM were respectively: {Acc}')
            print(f'Precision for Wake,N1,N2,N3,REM were respectively: {prec}')
            print(f'Recall for Wake,N1,N2,N3,REM were respectively: {Recall}')
            print(f'f1-score for Wake,N1,N2,N3,REM were respectively: {f1_sc}')
            print(f'Cohen kappa was calculated as: {kappa}')
            
        out_dic['Acc']     = Acc
        out_dic['Acc_all'] = Acc_all
        out_dic['Recall']  = Recall
        out_dic['prec']    = prec
        out_dic['f1_sc']   = f1_sc
        out_dic['mf1']     = mf1
        out_dic['kappa']   = kappa
        out_dic['mcm']     = mcm
            
        return out_dic# Acc, Acc_all, Recall, prec, f1_sc, mf1, kappa, mcm

    #%% Evaluation using two-class confusion matrix
    def two_class_confusion_matrix(self, y_true, y_pred, print_results = True):
        from sklearn.metrics import confusion_matrix
        import sklearn.metrics as skmetrics
        
        # calculate amount of true labels per class (NREM:0, REM = 1)
        N_NREM  = len([w for w,j in enumerate(y_true[:,0]) if y_true[w,0]==0])
        N_REM   = len([w for w,j in enumerate(y_true[:,0]) if y_true[w,0]==1])
        N_total = N_NREM + N_REM
        
        cm        = confusion_matrix(y_true, y_pred)
        tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0] # tp, tn is defined this way in sklearn!
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn) 
        #f1        = 2 * (precision * recall) / (precision + recall)
        f1        = skmetrics.f1_score(y_true, y_pred,average="macro")
        f1_micro  = skmetrics.f1_score(y_true, y_pred,average="micro")
        acc       = (tp+tn) / (tp+fp+tn+fn)
        #kappa = cohen_kappa_score(y_true, y_pred)
        
        if print_results == True:
            print("////// ====== Performance metrics \\\\\\ ======")
            print(f'total true labels: REM = {N_REM}, NREM = {N_NREM}, Total = {N_total}')
            print('Accuracy for REM was : {:.1f}'.format(acc*100))
            print('Precision for REM was : {:.1f}'.format(precision*100))
            print('Recall for REM was : {:.1f}'.format(recall*100))
            print('macro f1-score for REM detection was : {:.1f}'.format(f1*100))
            print('micro f1-score for REM detection was : {:.1f}'.format(f1_micro*100))
            #print(f'Cohen kappa was calculated as: {kappa}')
            
        return acc, recall, precision, f1 #, kappa
    
    #%% Randomized and grid search 
    ######################## DEFINING RANDOMIZED SEARCH ###########################
    #       ~~~~~~!!!!! THIS IS FOR RANDOM FOREST AT THE MOMENT ~~~~~~!!!!!!
    def RandomSearchRF(self, X, y, scoring, estimator = RandomForestClassifier(),
                        n_estimators = [int(x) for x in np.arange(10, 1000, 50)],
                        max_features = ['log2', 'sqrt'],
                        max_depth = [int(x) for x in np.arange(10, 100, 30)],
                        min_samples_split = [2, 5, 10],
                        min_samples_leaf = [1, 2, 4],
                        bootstrap = [True, False],
                        n_iter = 100, cv = 10):
        from sklearn.model_selection import RandomizedSearchCV
        tic = time.time()
        # DEFINING PARAMATERS
        # Number of trees in random forest
        n_estimators = n_estimators
        # Number of features to consider at every split
        max_features = max_features
        # Maximum number of levels in tree
        max_depth = max_depth
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = min_samples_split
        # Minimum number of samples required at each leaf node
        min_samples_leaf = min_samples_leaf
        # Method of selecting samples for training each tree
        bootstrap = bootstrap
        
        # Create the random grid
        param_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap,
                       'criterion' :['gini', 'entropy']}
        
        rf_random = RandomizedSearchCV(estimator = estimator,
                                   param_distributions = param_grid,
                                   n_iter = n_iter, cv = cv, scoring = scoring,
                                   verbose=2, n_jobs = -1)
        
        grid_result = rf_random.fit(X, y)
    
        BestParams_RandomSearch = rf_random.best_params_
        Bestsocre_RandomSearch   = rf_random.best_score_
    
        # summarize results
        
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        print('Randomized search was done in: {} secs'.format(time.time()-tic))
        print("Best: %f using %s" % (Bestsocre_RandomSearch, BestParams_RandomSearch))
        
        return BestParams_RandomSearch, Bestsocre_RandomSearch ,means, stds, params
        #%% Plot feature importance
        
    def Feat_importance_plot(self, Input ,labels, n_estimators = 250):
            classifier = RandomForestClassifier(n_estimators = n_estimators)
            classifier.fit(Input, labels)
            FeatureImportance = pd.Series(classifier.feature_importances_).sort_values(ascending=False)
            sb.barplot(y=FeatureImportance, x=FeatureImportance.index)
            plt.show()
            
    #%% mix pickle and h5 features
    def mix_pickle_h5(self, picklefile, saving_fname,
                          h5file = ("P:/3013080.02/ml_project/scripts/1D_TimeSeries/train_test/tr90_N3_fp1-M2_fp2-M1.h5"),
                          saving = False, ch = 'fp2-M1'):
        import pickle 

        # Define pickle file name
        
        picklefile = picklefile
        pickle_in = open(picklefile + ".pickle","rb")
        Featureset = pickle.load(pickle_in)
        # Open relative h5 file to map labels
        fname = h5file # N3
        ch = ch
        with h5py.File(fname, 'r') as rf:
            xtest  = rf['.']['x_test_' + ch].value
            xtrain = rf['.']['x_train_' + ch].value
            ytest  = rf['.']['y_test_' + ch].value
            ytrain = rf['.']['y_train_' + ch].value
        
        y = np.concatenate((ytrain[:,1], ytest[:,1]))
        
        rp = np.random.permutation(len(y))
        
        X = Featureset[rp,:]
        y = y[rp]
        
        # saving
        if saving == True:
            directory = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/' 
            fname = saving_fname
            with h5py.File((directory+fname + '.h5'), 'w') as wf:
                # Accuracies
                dset = wf.create_dataset('X', X.shape, data =X)
                dset = wf.create_dataset('y' , y.shape, data  = y)
                
        return X, y
        
    #%% Detect and remove bad signals
    def remove_bad_signals(self, hypno_labels, input_feats):
        bad        = [i for i,j in enumerate(hypno_labels[:,0]) if ((j==-1) or (j==8) or (j==6))]
        out_feats  = np.delete(input_feats, bad, axis=2)
        out_labels = np.delete(hypno_labels, bad, axis=0)
        
        return out_feats, out_labels
    
    #%% Detect and remove bad signals --> If the featureset has is 2-d
    def remove_bad_epochs(self, input_feats, hypno_labels):
        # Unkwown etc.
        bad        = [i for i,j in enumerate(hypno_labels[:,0]) if ((j==-1) or (j==8) or (j==6))]
        out_feats  = np.delete(input_feats, bad, axis=0)
        out_labels = np.delete(hypno_labels, bad, axis=0)
        
        #arousals
        bad        = [i for i,j in enumerate(out_labels[:,1]) if (j==1)]
        out_feats  = np.delete(out_feats, bad, axis=0)
        out_labels = np.delete(out_labels, bad, axis=0)
        
        return out_feats, out_labels
    #%% remove channels without scoring
    def remove_channels_without_scoring(self, hypno_labels, input_feats, Feats_Acc, Acc_feats = False):
        
        bad        = [i for i,j in enumerate(hypno_labels[:,0]) if (j==-1)]
        out_feats  = np.delete(input_feats, bad, axis=2)
        out_labels = np.delete(hypno_labels, bad, axis=0)
        if Acc_feats == True:
            out_acc_feats = np.delete(Feats_Acc, bad, axis=0)
            return out_feats, out_labels, out_acc_feats
        else:
            return out_feats, out_labels
    
    #%% remove channels without scoring
    def remove_channels_without_scoring2(self, hypno_labels, input_feats, Feats_Acc, Acc_feats = False):
        
        bad        = [i for i,j in enumerate(hypno_labels[:,0]) if (j==6)]
        out_feats  = np.delete(input_feats, bad, axis=2)
        out_labels = np.delete(hypno_labels, bad, axis=0)
        if Acc_feats == True:
            out_acc_feats = np.delete(Feats_Acc, bad, axis=0)
            return out_feats, out_labels, out_acc_feats
        else:
            return out_feats, out_labels
    #%% Remove disconnections:
    def remove_disconnection(self, hypno_labels, input_feats, Feats_Acc, Acc_feats = False):
        
        bad        = [i for i,j in enumerate(hypno_labels[:,0]) if (j==8)]
        out_feats  = np.delete(input_feats, bad, axis=2)
        out_labels = np.delete(hypno_labels, bad, axis=0)
        
        if Acc_feats == True:
            out_acc_feats = np.delete(Feats_Acc, bad, axis=0)
            return out_feats, out_labels, out_acc_feats
        
        else:
            return out_feats, out_labels
    #%% Detect and remove arousal and wake: useful for classifying only sleep stages
    def remove_artefact(self, hypno_labels, input_feats):
        bad        = [i for i,j in enumerate(hypno_labels[:,0]) if ((hypno_labels[i,1]==1) or (hypno_labels[i,1]==2))]
        out_feats  = np.delete(input_feats, bad, axis=2)
        out_labels = np.delete(hypno_labels, bad, axis=0)
        
        return out_feats, out_labels
    
        #%% Detect and remove bad signals --> If the featureset has is 2-d
    def remove_bad_epochs_from_zmax(self, ch1, ch2, hypno):
        # Unkwown etc.
        bad        = [i for i,j in enumerate(hypno[:,0]) if ((j==-1) or (j==8) or (j==6))]
        out_ch1    = np.delete(ch1, bad, axis=2)
        out_ch2    = np.delete(ch2, bad, axis=2)
        out_hypno  = np.delete(hypno, bad, axis=0)
        
        #arousals
        bad        = [i for i,j in enumerate(out_hypno[:,1]) if (j==1) or (j==2)]
        out_ch1    = np.delete(out_ch1, bad, axis=2)
        out_ch2    = np.delete(out_ch2, bad, axis=2)
        out_hypno  = np.delete(out_hypno, bad, axis=0)
        
        return out_ch1, out_ch2, out_hypno
    #%% Create 2 class hypnogram (NREM vs REM)
    def create_REM_NREM_hyp(self, hyp):
        
        # Copy initial hyp to output
        Out = np.zeros((len(hyp), 1))
        
        # Find index of different classes
        Wake_idx = [i for i,j in enumerate(hyp[:,0]) if (j == 0) ]
        N1_idx   = [i for i,j in enumerate(hyp[:,0]) if (j == 1) ]
        N2_idx   = [i for i,j in enumerate(hyp[:,0]) if (j == 2) ]
        N3_idx   = [i for i,j in enumerate(hyp[:,0]) if (j == 3) ]
        REM_idx  = [i for i,j in enumerate(hyp[:,0]) if (j == 5) ]
        
        # Replacing values of each class in corresponding column
        Out[Wake_idx, 0] = 0
        Out[N1_idx,   0] = 0
        Out[N2_idx,   0] = 0
        Out[N3_idx,   0] = 0
        Out[REM_idx,  0] = 1
        
        return Out
        
    #%% find unscored values in Zmax
    def find_unscored(self, hyp, subject_no, fname_save = "unscored_subjective"):
    
        # Initialize the .txt file for saving
        import datetime
        fmt='%d/%m/%Y ----- %H:%M:%S'
        
        # calculate unscored values
        unscored  = [i for i,j in enumerate(hyp[:,1]) if (hyp[i,1]=='U')]
        len_unscored = len(unscored)
        
        
        with  open(fname_save + ".txt", "a") as f:
            #f.write("This results file was created on : %s \r\n\n" % datetime.datetime.now().strftime(fmt))
            f.write("==================================================================\n")
            f.write("Subject %s : \n" % str(subject_no))
            f.write("==================================================================\r\n")
            f.write("Total of %s rows were not scored out of %s (%s percent)\n\r\n" % (str(len_unscored), str(np.shape(hyp)[0]), str(len_unscored/np.shape(hyp)[0]*100)))

            #f.write("%s  %s (%s percent)" % (str(len_unscored), str(np.shape(hyp)[0]),str(len_unscored/np.shape(hyp)[0]*100:.2f)))


    #%% Replace the stage of arousal with wake
    def replace_arousal_with_wake(self, hypno_labels, input_feats):
        arousal    = [i for i,j in enumerate(hypno_labels[:,0]) if (hypno_labels[i,1]==1)]
        out_labels = hypno_labels
        out_labels[arousal,0] = 0
        return out_labels
    
    #%% Create one column of binary values for each class
    def binary_labels_creator(self, labels):
        ''' column 0: wake - column 1: N1 - column 2: N2 - column 3: SWS - column 4: REM
        '''
        from sklearn.preprocessing import OneHotEncoder
        onehotencoder = OneHotEncoder(categorical_features = [0])
        out = onehotencoder.fit_transform(labels).toarray()
        out = out[:,0:5]
        
        return out
    #%% Plot confusion matrix
    def plot_confusion_matrix(self, y_test,y_pred, target_names = ['Wake','N1','N2','SWS','REM'],
                          title='Confusion matrix of DreamentoScorer algorithm',
                          cmap = None,
                          normalize=True):
    
        import matplotlib.pyplot as plt
        import numpy as np
        import itertools
        import matplotlib as m

        cdict = {
          'red'  :  ( (0.0, 0.25, .25), (0.02, .59, .59), (1., 1., .8)),
          'green':  ( (0.0, 0.0, 0.0), (0.3, .45, .45), (1., .97, .97)),
          'blue' :  ( (0.0, 1.0, 1.0), (0.02, .75, .75), (1., 0.45, 0.45))
        }

        Colors = m.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
        
        if np.shape(y_test)[1] > 1:
            y_test = self.binary_to_single_column_label(y_test)
            y_test = np.ravel(y_test)
    
        cm = confusion_matrix(y_test, y_pred)
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy
    
        if cmap is None:
            cmap = plt.get_cmap('Blues')
    
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest')
        plt.title(title)
        plt.colorbar()
    
        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    
        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
    
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()
    #%% Create one column of binary values for each class    
    def binary_labels_creator_categories(self, labels):
        ''' column 0: wake - column 1: N1 - column 2: N2 - column 3: SWS - column 4: REM
        '''
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
                
        ct = ColumnTransformer([("col", OneHotEncoder(), [0])], remainder = 'passthrough')
        labels = ct.fit_transform(labels)
        
        labels = labels[:,0:5]
        
        return labels
    #%% One-Hot Encoding WITH AROUSALS
    def One_hot_encoding(self, hyp):
        ''' column 0: wake - column 1: N1 - column 2: N2 - column 3: SWS - column 4: REM
            
        '''
       
        Out = np.zeros((len(hyp), 5))
        
        # Find index of classes and EXCLUDE the ones contaminated with ARTEFACT
            
# =============================================================================
#         Wake_idx = [i for i,j in enumerate(hyp[:,0]) if ((j == 0) and (hyp[i,1]==0))]
#         N1_idx   = [i for i,j in enumerate(hyp[:,0]) if ((j == 1) and (hyp[i,1]==0))]
#         N2_idx   = [i for i,j in enumerate(hyp[:,0]) if ((j == 2) and (hyp[i,1]==0))]
#         N3_idx   = [i for i,j in enumerate(hyp[:,0]) if ((j == 3) and (hyp[i,1]==0))]
#         REM_idx  = [i for i,j in enumerate(hyp[:,0]) if ((j == 5) and (hyp[i,1]==0))]
# =============================================================================
            
        # Find index of classes and also INCLUDE the ones contaminated with ARTEFACT   

        Wake_idx = [i for i,j in enumerate(hyp[:,0]) if (j == 0) ]
        N1_idx   = [i for i,j in enumerate(hyp[:,0]) if (j == 1) ]
        N2_idx   = [i for i,j in enumerate(hyp[:,0]) if (j == 2) ]
        N3_idx   = [i for i,j in enumerate(hyp[:,0]) if (j == 3) ]
        REM_idx  = [i for i,j in enumerate(hyp[:,0]) if (j == 5) ]
        
        
        # Replacing values of each class in corresponding column
        Out[Wake_idx, 0] = 1
        Out[N1_idx,   1] = 1
        Out[N2_idx,   2] = 1
        Out[N3_idx,   3] = 1
        Out[REM_idx,  4] = 1
        
        return Out
    

    #%% Make sure all the label rows have a value:
    def Unlabaled_rows_detector(self, labels):
        
        Unlabeled = 0
        for i,j in enumerate(labels[:,0]):
            if sum (labels[i,:]) == 1:
                pass
            else:
                Unlabeled = Unlabeled + 1
        if Unlabeled!= 0:
            raise ValueError(f"Some of the rows do not have a label!!!")
                
        print(f'Total of {Unlabeled} unlabeled rows were found.')
    #%% Save the feature-label pair as a pickle file
    def save_dictionary(self, path, fname, labels_dic, features_dic):
        import pickle        
        with open(path+fname+'.pickle',"wb") as f:
            pickle.dump([features_dic, labels_dic], f)
            
    #%% Load pickle files to access features and labels     
    def load_dictionary(self, path, fname):
        import pickle
        with open(path + fname + '.pickle', "rb") as f: 
            feats, y = pickle.load(f)
            
        return feats, y
    #%% Z-score the featureset
    def Standardadize_features(self, X_train, X_test):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        return X_train, X_test, sc
    
    #%% Replace the NaN values of features with the mean of each feature column
    def replace_NaN_with_mean(self, Features):
        aa, bb = np.where(np.isnan(Features))
        for j in np.arange(int(len(aa))):
            Features[aa[j],bb[j]] = np.nanmean(Features[:,bb[j]])
        
        return Features
    
    #%% REMOVE the NaN values of features
    def remove_NaN_from_features(self, Features, hyp, axis):
        aa     = np.where(np.isnan(Features[:, axis]))
        aa     = aa[0]
        
        Features_out = np.delete(Features, aa, axis=axis)
        hyp_out = np.delete(hyp     , aa, axis=axis)
        
        print('NaN values were removed from the featureset and labels')
        
        return Features_out, hyp_out
    #%% Replace the inf values of features with the mean of each feature column
    def replace_inf_with_mean(self, Features):
        feat_tmp = Features
        aa, bb = np.where(Features== np.inf)  
        feat_tmp = np.delete(feat_tmp,aa,0)
        
        for j in np.arange(int(len(aa))):
            Features[aa[j],bb[j]] = np.nanmean(feat_tmp[:,bb[j]])
        
        return Features
    
    #%% create hyppno single column array
    def binary_to_single_column_label(self, y_pred):
        # Find the index of each sleep stage (class)
        wake = [w for w,j in enumerate(y_pred[:,0]) if y_pred[w,0]==1]
        n1   = [w for w,j in enumerate(y_pred[:,1]) if y_pred[w,1]==1]
        n2   = [w for w,j in enumerate(y_pred[:,2]) if y_pred[w,2]==1]
        n3   = [w for w,j in enumerate(y_pred[:,3]) if y_pred[w,3]==1]
        rem  = [w for w,j in enumerate(y_pred[:,4]) if y_pred[w,4]==1]
        # Initialize hyp array
        hyp_pred = np.zeros((len(y_pred),1))
        # Replace the values of each sleep stage in hyp array
        hyp_pred[wake]  = 0
        hyp_pred[n1]    = 1
        hyp_pred[n2]    = 2
        hyp_pred[n3]    = 3
        hyp_pred[rem]   = 4
        
        return hyp_pred
    
    #%% Plot hypno
    def plot_hyp(self, hyp, mark_REM = 'active', ax = None):
        
        import matplotlib.pyplot as plt
        
        stages = hyp
        #stages = np.row_stack((stages, stages[-1]))
        x      = np.arange(len(stages))
        
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
        

#        plt.figure(figsize = [20,14])
        plt.step(x, y, where='post')
        plt.yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'])
        plt.ylabel('Sleep Stage')
        plt.xlabel('# Epoch')
        plt.title('Hypnogram')
        plt.rcParams.update({'font.size': 15})
        plt.xlim([np.min(x), np.max(x)])
        # Mark REM epochs
        if mark_REM == 'active':
            rem = [i for i,j in enumerate(hyp) if (hyp[i]==4)]
            for i in np.arange(len(rem)) -1:
                if rem[i+1] - rem[i] == 1:
                    plt.plot([rem[i], rem[i+1]], [-1,-1] , linewidth = 5, color = 'red')
                elif rem[i] - rem[i-1] == 1:
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
        
                elif ((rem[i+1] - rem[i] != 1) and (rem[i] - rem[i-1] != 1)):
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
                    
    #%% Plot 2-class hypnogram (NREM vs. REM)
    def plot_NREM_REM_hyp(self, y_true, y_pred, acc, f1, precision, recall, sub_name, mark_REM = True,\
                          save_fig = False,\
                          directory = "P:/3013080.02/Loreta_data/REM_detection"):
        import matplotlib.pyplot as plt
            
        fig, axs = plt.subplots(2,1, figsize=(26, 14))
        
        ###### PLOT TRUE HYP ######
        plt.axes(axs[0])
        plt.step(np.arange(len(y_true)), y_true, where='post')
        plt.yticks([0,1], ['NREM','REM'])
        plt.ylabel('Sleep Stage')
        plt.title('Hypnogram - True')
        plt.rcParams.update({'font.size': 15})
        plt.text(0, 1, ("acc={:.1f}, mf1={:.1f}, precision= {:.1f}, recall= {:.1f}".format(
        acc*100.0,
        f1*100.0,
        precision*100.0,
        recall*100.0
    )), transform=plt.gca().transAxes)
                # Mark REM epochs
        if mark_REM ==  True:
            rem = [i for i,j in enumerate(y_true) if (y_true[i]==1)]
            for i in np.arange(len(rem)) -1:
                if rem[i+1] - rem[i] == 1:
                    plt.plot([rem[i], rem[i+1]], [1, 1] , linewidth = 5, color = 'red')
                elif rem[i] - rem[i-1] == 1:
                    plt.plot([rem[i], rem[i]+1], [1,1] , linewidth = 5, color = 'red')
        
                elif ((rem[i+1] - rem[i] != 1) and (rem[i] - rem[i-1] != 1)):
                    plt.plot([rem[i], rem[i]+1], [1,1] , linewidth = 5, color = 'red')
        
        ###### PLOT PRED HYP ######
        plt.axes(axs[1])
        plt.step(np.arange(len(y_pred)), y_pred, where='post')
        plt.yticks([0,1], ['NREM','REM'])
        plt.ylabel('Sleep Stage')
        plt.xlabel('# Epoch')
        plt.title('Hypnogram - AI-predicted')
        plt.rcParams.update({'font.size': 15})

        
        # Mark REM epochs
        if mark_REM ==  True:
            rem = [i for i,j in enumerate(y_pred) if (y_pred[i]==1)]
            for i in np.arange(len(rem)) -1:
                if rem[i+1] - rem[i] == 1:
                    plt.plot([rem[i], rem[i+1]], [1, 1] , linewidth = 5, color = 'red')
                elif rem[i] - rem[i-1] == 1:
                    plt.plot([rem[i], rem[i]+1], [1,1] , linewidth = 5, color = 'red')
        
                elif ((rem[i+1] - rem[i] != 1) and (rem[i] - rem[i-1] != 1)):
                    plt.plot([rem[i], rem[i]+1], [1,1] , linewidth = 5, color = 'red')
                    
        #Save figure
        if save_fig == True:
            self.save_figure(directory=directory, saving_name="hyp_"+sub_name,
                             dpi=1000, saving_format = '.png',
                full_screen = False)  
                    
    #%% Mean results of leave-one-out cross-validation
    def Mean_leaveOneOut(self, metrics_per_fold):
        mean_acc      = np.empty((0,5))
        mean_prec     = np.empty((0,5))
        mean_recall   = np.empty((0,5))
        mean_f1_score = np.empty((0,5))
        mean_kappa    = []
        
        for i in np.arange(len(metrics_per_fold)):
            itr = str(i+1)
            iteration_tmp = metrics_per_fold['iteration'+itr]
            
            tmp_acc       = iteration_tmp[0]
            tmp_recall    = iteration_tmp[1]
            tmp_prec      = iteration_tmp[2]
            tmp_f1_score  = iteration_tmp[3]
            tmp_kappa     = iteration_tmp[4]
            # concatenate them all per metric
            mean_acc      = np.row_stack((mean_acc, tmp_acc))
            mean_prec     = np.row_stack((mean_prec, tmp_prec))
            mean_recall   = np.row_stack((mean_recall, tmp_recall))
            mean_f1_score = np.row_stack((mean_f1_score, tmp_f1_score))
            mean_kappa.append(tmp_kappa)
            # remove temp arrays
            del tmp_acc, tmp_recall, tmp_prec, tmp_f1_score
            
        # Show results - Mean
        Acc_Mean           = np.nanmean(mean_acc, axis = 0)
        Recall_Mean        = np.nanmean(mean_recall, axis = 0)
        Prec_Mean          = np.nanmean(mean_prec, axis = 0)
        F1_score_Mean      = np.nanmean(mean_f1_score, axis = 0)
        kappa_mean         = np.nanmean(mean_kappa)
        # Show results
        Acc_std           = np.nanstd(mean_acc, axis = 0)
        Recall_std        = np.nanstd(mean_recall, axis = 0)
        Prec_std          = np.nanstd(mean_prec, axis = 0)
        F1_score_std      = np.nanstd(mean_f1_score, axis = 0)
        kappa_std         = np.nanstd(mean_kappa)
        # Show results
        print(f'Mean Acc, Recall, Precision, and F1-score of leave-one-out cross-validation for Wake, N1, N2, SWS, and REM, respectively:\n{Acc_Mean}\n{Recall_Mean}\n{Prec_Mean}\n{F1_score_Mean}')
        print(f'Mean kappa: {kappa_mean}+-{kappa_std}')
        #print(f'std Acc, Recall, Precision, and F1-score of leave-one-out cross-validation for Wake, N1, N2, SWS, and REM, respectively:\n{Acc_std}\n{Recall_std}\n{Prec_std}\n{F1_score_std}')

        
    #%% def comparative hypnograms (True vs predicted)

    def plot_comparative_hyp(self, hyp_true, hyp_pred, mark_REM = 'active',
                             Title = 'True Hypnogram', labels = [0,1,2,3,5]):
        import matplotlib.pyplot as plt
# =============================================================================
#         hyp_true = self.binary_to_single_column_label(y_true)
#         if y_pred[1]:
#             hyp_pred = self.binary_to_single_column_label(y_pred)
#         else: 
#             hyp_pred = y_pred
# =============================================================================
    
        stages = hyp_true
        #stages = np.row_stack((stages, stages[-1]))
        x      = np.arange(len(stages))
        
        # Change the order of classes: REM and wake on top
        x = []
        y = []
        for i in np.arange(len(stages)):
            s = stages[i]
            if s== labels[0] :  p = -0
            if s== labels[-1] :  p = -1
            if s== labels[1] :  p = -2
            if s== labels[2] :  p = -3
            if s== labels[3] :  p = -4
            if i!=0:
                y.append(p)
                x.append(i-1)   
        y.append(p)
        x.append(i)
        
        #plt.figure(figsize = [20,14])
        fig, axs = plt.subplots(2,1, figsize=(26, 14))
        plt.axes(axs[0])
        plt.step(x, y, where='post')
        plt.yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'])
        plt.ylabel('Sleep Stage')
        #plt.xlabel('# Epoch')
        plt.title(Title)
        plt.rcParams.update({'font.size': 15})
        
        # Mark REM epochs
        if mark_REM == 'active':
            rem = [i for i,j in enumerate(hyp_true) if (hyp_true[i]==labels[-1])]
            for i in np.arange(len(rem)) -1:
                if rem[i+1] - rem[i] == 1:
                    plt.plot([rem[i], rem[i+1]], [-1,-1] , linewidth = 5, color = 'red')
                elif rem[i] - rem[i-1] == 1:
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
        
                elif ((rem[i+1] - rem[i] != 1) and (rem[i] - rem[i-1] != 1)):
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
                    
        del x,y, stages            
	    
        stages = hyp_pred
        #stages = np.row_stack((stages, stages[-1]))
        x      = np.arange(len(stages))
        
        # Change the order of classes: REM and wake on top
        x = []
        y = []
        for i in np.arange(len(stages)):
            s = stages[i]
            if s== labels[0] :  p = -0
            if s== labels[-1] :  p = -1
            if s== labels[1] :  p = -2
            if s== labels[2] :  p = -3
            if s== labels[3] :  p = -4
            if i!=0:
                y.append(p)
                x.append(i-1)   
        y.append(p)
        x.append(i)
        
    
        #plt.figure(figsize = [20,14])
        
        plt.axes(axs[1])
        plt.step(x, y, where='post')
        plt.yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'])
        plt.ylabel('Sleep Stage')
        plt.xlabel('# Epoch')
        plt.title('Predicted Hypnogram')
        plt.rcParams.update({'font.size': 15})
        
        # Mark REM epochs
        if mark_REM == 'active':
            rem = [i for i,j in enumerate(hyp_pred) if (hyp_pred[i]==labels[-1])]
            for i in np.arange(len(rem)) -1:
                if rem[i+1] - rem[i] == 1:
                    plt.plot([rem[i], rem[i+1]], [-1,-1] , linewidth = 5, color = 'red')
                elif rem[i] - rem[i-1] == 1:
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
        
                elif ((rem[i+1] - rem[i] != 1) and (rem[i] - rem[i-1] != 1)):
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
    
#%% def comparative hypnograms (True vs predicted)

    def plot_comparative_hyp_with_performance_metrics(self, hyp_true, hyp_pred, sub_name, PerformanceMetrics, mark_REM = 'active',
                             Title = 'True Hypnogram', save_fig = True, write_metrics = True,\
                             directory = "P:/3013080.02/Mahdad/Github/DreamentoScorer/",\
                             ):
        import matplotlib.pyplot as plt
# =============================================================================
#         hyp_true = self.binary_to_single_column_label(y_true)
#         if y_pred[1]:
#             hyp_pred = self.binary_to_single_column_label(y_pred)
#         else: 
#             hyp_pred = y_pred
# =============================================================================
    
        stages = hyp_true
        #stages = np.row_stack((stages, stages[-1]))
        x      = np.arange(len(stages))
        
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
        
        #plt.figure(figsize = [20,14])
        fig, axs = plt.subplots(2,1, figsize=(26, 14))
        plt.axes(axs[0])
        plt.step(x, y, where='post')
        plt.yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'])
        plt.ylabel('Sleep Stage')
        #plt.xlabel('# Epoch')
        plt.title(Title)
        plt.rcParams.update({'font.size': 15})
        
        # Mark REM epochs
        if mark_REM == 'active':
            rem = [i for i,j in enumerate(hyp_true) if (hyp_true[i]==4)]
            for i in np.arange(len(rem)) -1:
                if rem[i+1] - rem[i] == 1:
                    plt.plot([rem[i], rem[i+1]], [-1,-1] , linewidth = 5, color = 'red')
                elif rem[i] - rem[i-1] == 1:
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
        
                elif ((rem[i+1] - rem[i] != 1) and (rem[i] - rem[i-1] != 1)):
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
                    
        del x,y, stages            
	    
        stages = hyp_pred
        #stages = np.row_stack((stages, stages[-1]))
        x      = np.arange(len(stages))
        
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
        
    
        #plt.figure(figsize = [20,14])
        
        plt.axes(axs[1])
        plt.step(x, y, where='post')
        plt.yticks([0,-1,-2,-3,-4], ['Wake','REM', 'N1', 'N2', 'SWS'])
        plt.ylabel('Sleep Stage')
        plt.xlabel('# Epoch')
        plt.title('Predicted Hypnogram')
        plt.rcParams.update({'font.size': 15})
        
        if write_metrics == True:
            plt.text(2, 6, (f"n={len(np.arange(100))}, acc={np.round(PerformanceMetrics['Acc_all']*100.0,2)},\
            f1={np.round(PerformanceMetrics['f1_sc']*100.0,2)}, kappa= {np.round(PerformanceMetrics['kappa']*100.0,2)}"))

        # Mark REM epochs
        if mark_REM == 'active':
            rem = [i for i,j in enumerate(hyp_pred) if (hyp_pred[i]==4)]
            for i in np.arange(len(rem)) -1:
                if rem[i+1] - rem[i] == 1:
                    plt.plot([rem[i], rem[i+1]], [-1,-1] , linewidth = 5, color = 'red')
                elif rem[i] - rem[i-1] == 1:
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
        
                elif ((rem[i+1] - rem[i] != 1) and (rem[i] - rem[i-1] != 1)):
                    plt.plot([rem[i], rem[i]+1], [-1,-1] , linewidth = 5, color = 'red')
                    
        #Save figure
        if save_fig == True:
            self.save_figure(directory=directory, saving_name="hyp_"+sub_name,
                             dpi=500, saving_format = '.png',
                full_screen = False)
                                   
    #%% save figure
    def save_figure(self, directory, saving_name, dpi, saving_format = '.png',
                    full_screen = False):
        if full_screen == True:
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()
        plt.savefig(directory+saving_name+saving_format,dpi = dpi)    
        
    #%% Add time-dependency to the featureset
    def add_time_dependence_bidirectional(self, featureset, n_time_dependence=3,
                                        padding_type = 'sequential'):
        
        ''' n_time_dependece (forward and backward): 
            number of epochs preceding and proceeding the current investigational epoch.
            '''
        # Calculate number of features (columns)
        nf = np.shape(featureset)[1]
        #time dependence
        td = n_time_dependence
        # Initializa new feature array
        X_new = np.empty((np.shape(featureset)[0], nf * (2*td+1)))
        
        # Fill in the values for the AREA BETWEEN "TD" UNTIL "len(data)-td")
        for i in np.arange(td, np.shape(featureset)[0] - td):
            # Current epoch goes into the middle column
            X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]
            # proceeding and preceding epochs come here:
            for j in np.arange(1, td+1):
                # Fill in previous epochs
                X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i-j,:]
                # Fill in next epochs
                X_new[i,nf * (td+j) : nf* (td+j+1)] = featureset [i+j,:]
        del i,j
        if padding_type == 'same':
            # Fill in the values for the AREA BEFORE "TD"
            for i in np.arange(0,td):
                # Current epoch goes into the middle column
                X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]  
                # proceeding and preceding epochs come here:
                for j in np.arange(1, td+1):
                    # Fill in previous epochs with the same values as the current
                    X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i,:]
                    # Fill in next epochs
                    X_new[i,nf * (td+j) : nf* (td+j+1)] = featureset [i+j,:]
            del i,j       
            # Fill in the values for the AREA AFTER "TD"
            for i in np.arange(np.shape(featureset)[0] - td, np.shape(featureset)[0]):
                # Current epoch goes into the middle column
                X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]
                for j in np.arange(1, td+1):
                    # Fill in previous epochs
                    X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i-j,:]
                    # Fill in next epochs
                    X_new[i,nf * (td+j) : nf* (td+j+1)] = featureset [i,:]
            del i,j        
        
        if padding_type == 'sequential':
            # Fill in the values for the AREA BEFORE "TD"
            for i in np.arange(0,td):
                # Current epoch goes into the middle column
                X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]  
                # proceeding and preceding epochs come here:
                for j in np.arange(1, td+1):
                    
                    # Make usre if there is an epoch before the current:
                    if i - j >= 0:
                        # Fill in previous epochs with the values of previous section
                        X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i-j,:]
                    else:
                        # Fill in previous epochs with the values of 0th epoch
                        X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [0,:]
                    # Fill in next epochs
                    X_new[i,nf * (td+j) : nf* (td+j+1)] = featureset [i+j,:]
            del i,j        
            
            # Fill in the values for the AREA AFTER "TD"
            for i in np.arange(np.shape(featureset)[0] - td, np.shape(featureset)[0]):
                
                # Current epoch goes into the middle column
                X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]

                for j in np.arange(1, td+1):
                    # Make usre if there is an epoch after the current:
                    if (i + j) <= np.shape(featureset)[0] - 1:
                        # Fill in next epochs
                        X_new[i,nf * (td+j) : nf* (td+j+1)] = featureset [i+j,:]
                    else: 
                        X_new[i,nf * (td+j) : nf* (td+j+1)] = featureset [-1,:]
                    # Fill in previos epochs
                    X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i-j,:]
                    
        return X_new
    
    
    #%% Add time-dependency to the featureset
    def add_time_dependence_backward(self, featureset, n_time_dependence=3,
                                        padding_type = 'sequential'):
        
        ''' n_time_dependece (only backward): 
            number of epochs preceding the current investigational epoch.
            '''
        # Calculate number of features (columns)
        nf = np.shape(featureset)[1]
        #time dependence
        td = n_time_dependence
        # Initializa new feature array
        X_new = np.empty((np.shape(featureset)[0], nf * (td+1)))
        
        # Fill in the values for the AREA BETWEEN "TD" UNTIL "len(data)-td")
        for i in np.arange(td, np.shape(featureset)[0] - td):
            # Current epoch goes into the middle column
            X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]
            # proceeding and preceding epochs come here:
            for j in np.arange(1, td+1):
                # Fill in previous epochs
                X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i-j,:]
        del i,j
        
        if padding_type == 'sequential':
            # Fill in the values for the AREA BEFORE "TD"
            for i in np.arange(0,td):
                # Current epoch goes into the middle column
                X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]  
                # proceeding and preceding epochs come here:
                for j in np.arange(1, td+1):
                    
                    # Make usre if there is an epoch before the current:
                    if i - j >= 0:
                        # Fill in previous epochs with the values of previous section
                        X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i-j,:]
                    else:
                        # Fill in previous epochs with the values of 0th epoch
                        X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [0,:]

            del i,j        
            # Fill in the values for the last "TD" epochs
            for i in np.arange(np.shape(featureset)[0] - td, np.shape(featureset)[0]):
                
                # Current epoch goes into the middle column
                X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]
                
                for j in np.arange(1, td+1):
                    # Fill in previos epochs
                    X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i-j,:]
                    
        elif padding_type == 'same':
            # Fill in the values for the AREA BEFORE "TD"
            for i in np.arange(0,td):
                # Current epoch goes into the middle column
                X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]  
                # proceeding and preceding epochs come here:
                for j in np.arange(1, td+1):
                    # Fill in previous epochs with the same values as the current
                    X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i,:]
            del i,j       
            # Fill in the values for the AREA AFTER "TD"
            for i in np.arange(np.shape(featureset)[0] - td, np.shape(featureset)[0]):
                # Current epoch goes into the middle column
                X_new[i, td * nf : (td+1) * nf]  = featureset[i,:]
                for j in np.arange(1, td+1):
                    # Fill in previous epochs
                    X_new[i,nf * (td-j) : nf* (td-j+1)] = featureset [i-j,:]       
                    
        return X_new
    

    #%% Create subjective outcomes
    def create_subjecive_results(self, y_true, y_pred, test_subjects_list,
                                 subjects_data_dic, fname_save = "results"):
        
        from sklearn.metrics import multilabel_confusion_matrix, cohen_kappa_score

        #Initializing counter
        counter = 0
        
        # Reshaping y_true to compare with y_pred
        if np.shape(y_true)[1] > 1:
            y_true = self.binary_to_single_column_label(y_true)
            y_true = np.ravel(y_true)
            
        # Initialize the .txt file for saving
        import datetime
        fmt='%d/%m/%Y ----- %H:%M:%S'
        
        with  open(fname_save + ".txt", "a") as f:
            f.write("\nDreamentoScorer: Automatic sleep scoring package\r\n")
            f.write("Link to package: https://github.com/MahdadJafarzadeh/DreamentoScorer \r\n")
            f.write("This results file was created on : %s \r\n\n" % datetime.datetime.now().strftime(fmt))
        
        # Retrieving the size of each test subject's data
        for c, sub_name in enumerate(test_subjects_list):
            
            # Calculating shape of the current subject
            current_size = np.shape(subjects_data_dic[test_subjects_list[c]])[0]
            
            # Load corresponding prediction and true values
            y_pred_tmp   = y_pred[counter:counter + current_size]
            y_true_tmp   = y_true[counter:counter + current_size]
            
            # Defining metrics
            mcm = multilabel_confusion_matrix(y_true_tmp, y_pred_tmp)
            tn     = mcm[:, 0, 0]
            tp     = mcm[:, 1, 1]
            fn     = mcm[:, 1, 0]
            fp     = mcm[:, 0, 1]
            Recall = tp / (tp + fn)
            prec   = tp / (tp + fp)
            f1_sc  = 2 * Recall * prec / (Recall + prec)
            Acc = (tp + tn) / (tp + fp + fn+ tn)
            kappa = cohen_kappa_score(y_true_tmp, y_pred_tmp)
            
            # Write down these results in a txt file          
            with  open(fname_save + ".txt", "a") as f:
                f.write("==================================================================\n")
                f.write("Results for %s : \n" % str(sub_name))
                f.write("==================================================================\r\n")
                f.write("Accuracy for Wake, N1, N2, N3, and REM were respectively: %s \r\n" % str(Acc))
                f.write("Recall for Wake, N1, N2, N3, and REM were respectively: %s \r\n" % str(Recall))
                f.write("Precision for Wake, N1, N2, N3, and REM were respectively: %s \r\n" % str(prec))
                f.write("F1-score for Wake, N1, N2, N3, and REM were respectively: %s \r\n" % str(f1_sc))
                f.write("Cohen kappa was calculated as: %s \r\n" % str(kappa))
                #f.write("..................................................................\r\n") 
            
            # Update the counter
            counter = counter + current_size
            
            # Delete variables of the loop
            del current_size, y_pred_tmp, y_true_tmp, Acc, Recall, prec, kappa, f1_sc, mcm, c, sub_name, tn, tp, fp, fn
        
        # Compute overall metrics 
        mcm = multilabel_confusion_matrix(y_true, y_pred)
        tn     = mcm[:, 0, 0]
        tp     = mcm[:, 1, 1]
        fn     = mcm[:, 1, 0]
        fp     = mcm[:, 0, 1]
        Recall = tp / (tp + fn)
        prec   = tp / (tp + fp)
        f1_sc  = 2 * Recall * prec / (Recall + prec)
        Acc = (tp + tn) / (tp + fp + fn+ tn)
        kappa = cohen_kappa_score(y_true, y_pred)
            
        # Add overall results
        with  open(fname_save + ".txt", "a") as f:
            f.write("==================================================================\n")
            f.write("Overall results:\n")
            f.write("==================================================================\r\n")
            f.write("Accuracy for Wake, N1, N2, N3, and REM were respectively: %s \r\n" % str(Acc))
            f.write("Recall for Wake, N1, N2, N3, and REM were respectively: %s \r\n" % str(Recall))
            f.write("Precision for Wake, N1, N2, N3, and REM were respectively: %s \r\n" % str(prec))
            f.write("F1-score for Wake, N1, N2, N3, and REM were respectively: %s \r\n" % str(f1_sc))
            f.write("Cohen kappa was calculated as: %s \r\n" % str(kappa))
        f.close()
        
    #%% Plot subjective hypno
    def plot_subjective_hypno(self,y_true, y_pred, test_subjects_list,
                                 subjects_data_dic, save_fig = False, 
                                 directory="C:/PhD/Github/DreamentoScorer"):
        
        #Initializing counter
        counter = 0
        
        # Reshaping y_true to compare with y_pred
        if np.shape(y_true)[1] > 1:
            y_true = self.binary_to_single_column_label(y_true)
            y_true = np.ravel(y_true)

        # Retrieving the size of each test subject's data
        for c, sub_name in enumerate(test_subjects_list):
            
            # Calculating shape of the current subject
            current_size = np.shape(subjects_data_dic[test_subjects_list[c]])[0]
            
            # Load corresponding prediction and true values
            y_pred_tmp   = y_pred[counter:counter + current_size]
            y_true_tmp   = y_true[counter:counter + current_size]
            
            self.plot_comparative_hyp(hyp_true=y_true_tmp, hyp_pred=y_pred_tmp, mark_REM = 'active',
                                      Title = 'True hypnogram - ' + sub_name)
            
            #Save figure
            if save_fig == True:
                self.save_figure(directory=directory, saving_name="hyp_"+sub_name,
                                 dpi=1000, saving_format = '.png',
                    full_screen = False)
            # Update the counter
            counter = counter + current_size
            
            del current_size, y_pred_tmp, y_true_tmp
            
    #%% Plot spectrogram

    def spectrogram_creation(self, sig, fs, explanation='', saving_directory=None):
        from lspopt import spectrogram_lspopt
        import numpy as np
        import matplotlib.pyplot as plt
        for i in range(len(sig)):
            f, t, Sxx = spectrogram_lspopt(x=sig[i], fs=fs, c_parameter=20.0, nperseg=int(30*fs), \
                                           scaling='density')
            Sxx = 10 * np.log10(Sxx) #power to db
            
    
        #==== 1st Way =======
        plt.figure()
        ax = plt.axes()
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]', size=20)
        plt.xlabel('Time [sec]', size=20)
        plt.title('C3 Multi-taper Spectrogram', size=25)
        ax.tick_params(labelsize=15) #chnage size of tick parameters on x and y axes
        
        plt.colorbar()
        #==== 1st Way =======
        
        #=== Maximize ====
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(32, 18)
        plt.show()
        #=== Maximize ====
        
        #===== Save Figure ======
        if(saving_directory != None):
        
           plt.savefig(saving_directory + '/' + explanation + '_' + str(i), pad_inches=0, \
                       bbox_inches='tight', dpi=400)
           print('Figure has saved successfully!')
        #===== Save Figure ======
           plt.close()
    #%% Plot subjective confusion matrix
    def plot_confusion_mat_subjective(self, y_true, y_pred, test_subjects_list,
                                 subjects_data_dic):
        
        #Initializing counter
        counter = 0
        
        '''# Reshaping y_true to compare with y_pred
        if np.shape(y_true)[1] > 1:
            y_true = self.binary_to_single_column_label(y_true)
            y_true = np.ravel(y_true)'''

        # Retrieving the size of each test subject's data
        for c, sub_name in enumerate(test_subjects_list):
            
            # Calculating shape of the current subject
            current_size = np.shape(subjects_data_dic[test_subjects_list[c]])[0]
            
            # Load corresponding prediction and true values
            y_pred_tmp   = y_pred[counter:counter + current_size]
            y_true_tmp   = y_true[counter:counter + current_size]
            
            #Plot confusion matrix
            self.plot_confusion_matrix(y_true_tmp,y_pred_tmp,
                                       title='Confusion matrix of '+sub_name)
            
            # Update the counter
            counter = counter + current_size
            
    #%% Find number of samples per class
    def find_number_of_samples_per_class(self, labels, including_artefact = False):
        
        # Convert output structure to Onehotencioded version:
        
        wake = [w for w,j in enumerate(labels[:,0]) if labels[w,0]==1]
        n1   = [w for w,j in enumerate(labels[:,1]) if labels[w,1]==1]
        n2   = [w for w,j in enumerate(labels[:,2]) if labels[w,2]==1]
        n3   = [w for w,j in enumerate(labels[:,3]) if labels[w,3]==1]
        rem  = [w for w,j in enumerate(labels[:,4]) if labels[w,4]==1]
            
        # Calculate the number per class
        n_w   = len(wake)
        n_n1  = len(n1)
        n_n2  = len(n2)
        n_n3  = len(n3)
        n_rem = len(rem)
        if including_artefact == True:            
            artefact  = [w for w,j in enumerate(labels[:,5]) if labels[w,5]==1]
            n_MA  = len(artefact)
            print(f'Nmber of epochs per class:\nWake:{n_w}, N1:{n_n1}, N2:{n_n2}, SWS:{n_n3}, REM:{n_rem}, Movement Artefact:{n_MA} ')
        else:
            print(f'Nmber of epochs per class:\nWake:{n_w}, N1:{n_n1}, N2:{n_n2}, SWS:{n_n3}, REM:{n_rem}')


        
    #%% ensure train data and labels have the same length
    def Ensure_data_label_length(self, X, y):
        len_x, len_y = np.shape(X)[2], np.shape(y)[0]
        if len_x == len_y:
            print("Length of data and hypnogram are identical! Perfect!")
        else:
            raise ValueError("Lengths of data epochs and hypnogram labels are different!!!")
    #%% ensure train data and labels have the same length
    def Ensure_feature_label_length(self, X, y):
        len_x, len_y = np.shape(X)[0], np.shape(y)[0]
        if len_x == len_y:
            print("Length of data and hypnogram are identical! Perfect!")
        else:
            raise ValueError("Lengths of data epochs and hypnogram labels are different!!!")   
            
    #%% Plot accceleration data
    def Read_Acceleration_data(self, folder_acc , axis_files = ["dX", "dY", "dZ"],
                           file_format = ".edf", plot_Acc = False):
        import mne

        # Loading Acceleration data
        Acc_X   = mne.io.read_raw_edf(folder_acc + axis_files[0] + file_format)
        Acc_Y   = mne.io.read_raw_edf(folder_acc + axis_files[1] + file_format)
        Acc_Z   = mne.io.read_raw_edf(folder_acc + axis_files[2] + file_format)
        
        # Get data
        Acc_X   = Acc_X.get_data()
        Acc_Y   = Acc_Y.get_data()
        Acc_Z   = Acc_Z.get_data()
                
        # Define length of epochs
        self.len_epoch   = self.fs * self.T
        
        # Cut remaining tail from the last epoch; use modulo to find full epochs 
        Acc_X = Acc_X[:, 0:Acc_X.shape[1] - Acc_X.shape[1]%self.len_epoch]
        Acc_Y = Acc_Y[:, 0:Acc_Y.shape[1] - Acc_Y.shape[1]%self.len_epoch]
        Acc_Z = Acc_Z[:, 0:Acc_Z.shape[1] - Acc_Z.shape[1]%self.len_epoch]
        
        # Acc data in an array format
        Acc     = [Acc_X, Acc_Y, Acc_Z]
        
        # Compute length of data
        N       = np.shape(Acc_X)[1]
        
        # Init norm of acc
        AccNorm = np.empty((1,N))
        
        # Compute norm of Acc
        for i in np.arange(0,N):
            
            AccNorm[0,i] = np.sqrt(Acc_X[0,i]**2 + Acc_Y[0,i]**2 + Acc_Z[0,i]**2 )
        
        # Calculate mean of Acc
        MeanAcc        = np.mean(AccNorm)
        
        # Remove mean 
        AccNorm_filt   = AccNorm - MeanAcc
        
        if plot_Acc == True:
            
            # Create figure
            fig, axs = plt.subplots(2,1, figsize=(20, 10))
            
            # First subplot shows axis-based acceleration
            samples = np.arange(0, N)
            plt.axes(axs[0])
            plt.plot(samples, np.ravel(Acc_X), color = 'blue', label = 'Acc_X')
            plt.plot(samples, np.ravel(Acc_Y), color = 'green', label = 'Acc_Y')
            plt.plot(samples, np.ravel(Acc_Z), color = 'red', label = 'Acc_Z')
            plt.ylabel('Amplitude (g)')
            plt.xlabel('Sapmles')
            plt.title('Acceleration of different axes')
            plt.xlim(0, N) 
            plt.legend()
            # Second plot is just norm of Acc
            plt.axes(axs[1])
            plt.plot(samples, np.ravel(AccNorm_filt), color = 'black')
            plt.ylabel('Amplitude (g)')
            plt.xlabel('Sapmles')
            plt.title('Norm of  accelartion')
            plt.xlim(0, N)
        
        return AccNorm, Acc
   
    #%% Read csv files
    def read_csv(self, file, folder = "C:/PhD/Zmax/", 
                 column_labels=["col1", "epoch", "stage_no", "stage", 
                "start","end","date&time_start", "date&time_end"],
                 slicing_col = 2, delimiter = None):
        
        import pandas as pd        
        
        data = pd.read_csv(folder + file + ".csv", names = column_labels, delimiter= delimiter)
        
        df   = data.set_index("epoch", drop = False)
        
        sliced_data = df.iloc[:, slicing_col]
        
        return data, sliced_data
    #%% Compute metrics
    def compute_performance_metrics(self, y_ts, y_pred, labels=[0,1,2,3,5]):
        """Computer performance metrics from confusion matrix.
    
        It computers performance metrics from confusion matrix.
        It returns:
        - Total number of samples
        - Number of samples in each class
        - Accuracy
        - Macro-F1 score
        - Per-class precision
        - Per-class recall
        - Per-class f1-score
        
        """
        import sklearn.metrics as skmetrics

        acc = skmetrics.accuracy_score(y_ts, y_pred)
        f1_score = skmetrics.f1_score(y_true=y_ts, y_pred=y_pred, average="macro")
        cm = skmetrics.confusion_matrix(y_true=y_ts, y_pred=y_pred, labels = labels)
        tp = np.diagonal(cm).astype(np.float)
        tpfp = np.sum(cm, axis=0).astype(np.float) # sum of each col
        tpfn = np.sum(cm, axis=1).astype(np.float) # sum of each row
        acc = np.sum(tp) / np.sum(cm)
        precision = tp / tpfp
        recall = tp / tpfn
        f1 = (2 * precision * recall) / (precision + recall)
        mf1 = np.mean(f1)
    
        total = np.sum(cm)
        n_each_class = tpfn

        return total, n_each_class, acc, mf1, precision, recall, f1

    #%% summarize outcome
    def summarize_performance(self, metrics):
        
        print("Total: {}".format(metrics[0]))
        print("Number of samples from each class: {}".format(metrics[1]))
        print("Accuracy: {:.1f}".format(metrics[2]*100.0))
        print("Macro F1-Score: {:.1f}".format(metrics[3]*100.0))
        print("Per-class Precision: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[4]]))
        print("Per-class Recall: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[5]]))
        print("Per-class F1-Score: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[6]]))

# =============================================================================
#     #%% LSTM classifier
#     def LSTM_classifier(self, X_train, X_test, y_train, neurons_l1 = 80, dropout = .3,
#                          neurons_l2 = 80, epochs = 100, batch_size = 512, verbose = 1,
#                          loss='mean_squared_error', optimizer='adam',
#                          metrics = ['accuracy'],
#                          print_model_summary = False, bidirectional_ = True):
#         
#         """This function requires the hand-crafted features to be fed in: 
#             
#         Parameters:
#             
#         neurons_l1: neurouns/units in 1st layer.
#         
#         neurons_l2: neurouns/units in 2nd layer.
#         
#         bidirectional_: activates bidirectional LSTM (default: True)
#         """
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Importing libraries ~~~~~~~~~~~~~~~~~~~~~~ #
#         from keras.models import Sequential
#         from keras.layers import Dense
#         from keras.layers import LSTM, Bidirectional, TimeDistributed
#         
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Reshaping input data~~~~~~~~~~~~~~~~~~~~~~ #
#         
#         # check if the existing data has 3 dimensions(reshape from [samples, timesteps] into [samples, timesteps, features])
#             
#         trainX = X_train.reshape(-1,1, np.shape(X_train)[1])
#         testX  = X_test.reshape(-1,1, np.shape(X_test)[1])
#         n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
#         trainY = y_train.reshape(-1,1, np.shape(y_train)[1])
#         
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Creating model ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #   
#         
#         # init
#         model = Sequential()
#         
#         #! Check bidirectional flag to activate/deactivate bidirectional training
#         if bidirectional_ == True:
#             
#             # 1st layer + dropout
#             model.add(Bidirectional(LSTM(neurons_l1, input_shape=(n_timesteps, n_features), recurrent_dropout=dropout, return_sequences=True)))
#             
#             # 2nd layer + dropout
#             model.add(Bidirectional(LSTM(neurons_l2, recurrent_dropout=dropout, return_sequences=True)))
#         else:
#             # 1st layer + dropout
#             model.add(LSTM(neurons_l1, input_shape=(n_timesteps, n_features), recurrent_dropout=dropout, return_sequences=True))
#             
#             # 2nd layer + dropout
#             model.add(LSTM(neurons_l2, recurrent_dropout=dropout, return_sequences=True))
#          
#         # Adding dense
#         model.add(TimeDistributed(Dense(5, activation='softmax')))
#         
#         # compile
#         model.compile(loss=loss, optimizer=optimizer, metrics = metrics)
#         
#         # Fit to train data
#         model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
#         
#         # Predict
#         y_pred = model.predict_classes(testX)
#         
#         # print model summary
#         if print_model_summary == True:
#             print(model.summary())
#         
#         return y_pred
#     
#     #%% Deep CNN classifier
#     
#     def CNN_Classifier(self, X_train, y_train, X_test, fs, verbose = 1, epochs = 100,
#                        batch_size = 512):
#         
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Importing libraries ~~~~~~~~~~~~~~~~~~~~~~ #
#         import tensorflow as tf
#         from tensorflow import keras
#         from keras.models import Sequential
#         from keras.layers import Dense
#         from keras.layers import Flatten
#         from keras.layers import Dropout, BatchNormalization
#         from keras.layers.convolutional import Conv1D
#         from keras.layers.convolutional import MaxPooling1D
#         from keras.optimizers import adam
# 
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Reshaping input data~~~~~~~~~~~~~~~~~~~~~~ #
#         
#         # reshape from [samples, timesteps] into [samples, timesteps, features]
#         
#         trainX = np.transpose(X_train)
#         testX  = np.transpose(X_test)
#         n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
#         
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Creating model ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#         model = Sequential()
#         
#         # Conv1
#         model.add(Conv1D(filters = 128, kernel_size= 50, activation='relu', strides = 5, input_shape=(n_timesteps,n_features)))
#         
#         # Add dropout and batch normalization 1
#         model.add(BatchNormalization())
#         model.add(Dropout(0.2))
#         
#         # Conv2
#         model.add(Conv1D(filters = 256, kernel_size= 5, activation='relu', strides = 1))
#         
#         # Add dropout and batch normalization 2
#         model.add(BatchNormalization())
#         model.add(Dropout(0.2))
#         
#         # Max-pooling1
#         model.add(MaxPooling1D(pool_size=2, strides = 1))
#         
#         # Conv3
#         model.add(Conv1D(filters = 300, kernel_size= 5, activation='relu', strides = 2))
#                
#         # Add dropout and batch normalization 3
#         model.add(BatchNormalization())
#         model.add(Dropout(0.2))
# 
#         # Max-pooling 2
#         model.add(MaxPooling1D(pool_size=2, strides = 1))
#         
#         # Flatten
#         model.add(Flatten())
#         
#         # Dense 1
#         model.add(Dense(1500, activation='relu'))
#         
#         # Add dropout and batch normalization 4
#         model.add(BatchNormalization())
#         model.add(Dropout(0.5))
#         
#         # Dense 2
#         model.add(Dense(1500, activation='relu'))
#         
#         # Add dropout and batch normalization 5
#         model.add(BatchNormalization())
#         model.add(Dropout(0.5))
#         
#         # Output layer
#         model.add(Dense(5, activation='softmax'))
#         
#         # Compile
#         model.compile(loss='mean_squared_logarithmic_error', optimizer= 'adam', metrics=['accuracy'])   
#         
#         # Fit
#         model.fit(trainX, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
#         
#         # Predict
#         y_pred = model.predict_classes(testX)
#         
#         return y_pred
# 
#     #%% (CNN+LSTM) stack classifier
#     def CNN_LSTM_stack_calssifier(self, X_train, X_test, y_train, y_test, fs,path, n_filters = [8, 16, 32], 
#                         kernel_size = [50, 8, 8], LSTM_units = 64, n_LSTM_layers = 4,
#                         recurrent_dropout = .3,loss='mean_squared_error', 
#                         optimizer='adam',metrics = ['accuracy'],
#                         epochs = 10, batch_size = 128, verbose = 1,
#                         show_summarize =True, plot_model_graph =True, show_shapes = False,
#                         patience = 6):
#         
#         """ This model has a CNN on top for feature extraction and layers of LSTM 
#         at the bottom to account for time dependency.
#         
#         Please note: The data flow for training is troughout the model and there
#         is no pretraining for CNN. So the CNN weight will be also upodated here
#         after each iteration.
#         """
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Importing libraries ~~~~~~~~~~~~~~~~~~~~~~ #
#         #import tensorflow as tf
#         #from tensorflow import keras
#         import keras
#         from keras.utils import plot_model
#         from keras.models import Model
#         from keras.layers import Input
#         from keras.layers import Dense
#         from keras.layers.recurrent import LSTM
#         from keras.layers.merge import concatenate
#         from keras.layers.convolutional import Conv1D
#         from keras.layers.pooling import MaxPooling1D
#         from keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed,Flatten, Bidirectional
#         from keras.callbacks import EarlyStopping, ModelCheckpoint
#         import pickle
# # =============================================================================
# #         import tensorflow
# #         #from tensorflow.python.keras.utils import plot_model
# #         from tensorflow.python.keras.models import Model
# #         from tensorflow.python.keras.layers import Input
# #         from tensorflow.python.keras.layers import Dense
# #         from tensorflow.python.keras.layers.recurrent import LSTM
# #         from tensorflow.python.keras.layers.merge import concatenate
# #         from tensorflow.python.keras.layers.convolutional import Conv1D
# #         from tensorflow.python.keras.layers.pooling import MaxPooling1D
# #         from tensorflow.python.keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed,Flatten, Bidirectional
# #         from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# #         import pickle
# # =============================================================================
#         
#         # ~~~~~~~~~~~~~~~~~~~~~~~~Defining input data~~~~~~~~~~~~~~~~~~~~~~~~ #
#     
#         # input is 30-s epoch of sleep (samples, timesteps, features])
#         input_sig1 = Input(shape=(30*fs,1))
#                 
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Creating CNN model ~~~~~~~~~~~~~~~~~~~~~~~ #
#         
#         ### === Layer 1 === ###
#         
#         # Conv 1
#         conv1 = Conv1D(n_filters[0], kernel_size = kernel_size[0], strides = 1, 
#                         padding='same', kernel_initializer='glorot_uniform')(input_sig1)
#         # Batch normalization 1
#         batch1 = BatchNormalization()(conv1)
#         # Activation 1
#         act1 = Activation('relu')(batch1)
#         # Maxpooling 1
#         maxpooling1 = MaxPooling1D(pool_size=8, strides=1)(act1)
#         
#         ### === Layer 2 === ###
#         
#         # Conv 2
#         conv2 = Conv1D(n_filters[1], kernel_size= kernel_size[1], strides = 1, 
#               padding='same', kernel_initializer='glorot_uniform')(maxpooling1)
#         # Batch normalization 2
#         batch2 = BatchNormalization()(conv2)
#         # Activation 2
#         act2 = Activation('relu')(batch2)
#         # Max-pooling 2
#         maxpooling2 = MaxPooling1D(pool_size=8, strides=1)(act2)
#         
#         ### === Layer 3 === ###
#         
#         # Conv 3
#         conv3 = Conv1D(n_filters[2], kernel_size= kernel_size[2], strides = 1, 
#               padding='same', kernel_initializer='glorot_uniform')(maxpooling2)
#         # Batch normalization 3
#         batch3 = BatchNormalization()(conv3)
#         # Activation 3
#         act3 = Activation('relu')(batch3)
#         # Max-pooling 3
#         maxpooling3 = MaxPooling1D(pool_size=8, strides=1)(act3)
#             
#         # ~~~~~~~~~~~~~~~~~~~~~~~ Creating LSTM model ~~~~~~~~~~~~~~~~~~~~~~~ #
#     
#         # Adding LSTM 1
#         lstm1 = Bidirectional(LSTM(units = LSTM_units, recurrent_dropout= recurrent_dropout, return_sequences=True))(maxpooling3) 
#         # Adding LSTM 2         
#         lstm2 = Bidirectional(LSTM(units = LSTM_units,recurrent_dropout= recurrent_dropout, return_sequences=True))(lstm1)
#         # Adding LSTM 3
#         lstm3 = Bidirectional(LSTM(units = LSTM_units,recurrent_dropout= recurrent_dropout, return_sequences=True))(lstm2)
#         # Adding LSTM 4
#         lstm4 = Bidirectional(LSTM(units = LSTM_units,recurrent_dropout= recurrent_dropout, return_sequences=True))(lstm3)
#         
#         # flattening
#         flattened = Flatten()(lstm4)
#         
#         # Final Dense layer of CNN + LSTM stack
#         final = Dense(5, activation = 'softmax')(flattened)
#         
#         # Activates tf variable and cpu/gpu
#         #with tf.device('/gpu:0'):
#         model = Model(inputs=[input_sig1], outputs=[final])
#         
#         # compile
#         #optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
#         #                                  beta_2=0.999, epsilon=1e-08)
#         optimizer = keras.optimizers.Adam()
#         model.compile(loss=loss, optimizer= optimizer, metrics = metrics)   
#         
#         # plot graph
# # =============================================================================
# #         if plot_model_graph ==True:  
# #             plot_model(model, show_shapes = show_shapes)
# # =============================================================================
#             
#         # summarize layers
#         if show_summarize ==True:
#             print(model.summary())
#             
#         #callbacks
#         es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
#         mc = ModelCheckpoint(path+'best_model.h5', \
#                           monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
#             
#             
#         # Fit and train
#         history = model.fit(np.transpose(X_train), y_train, epochs=epochs, validation_data=(np.transpose(X_test), y_test), batch_size=batch_size, \
#                             verbose=verbose, callbacks=[es, mc])
#                
#         #pickle best model
#         best_model = pickle.load(open(path+'best_model.h5', 'wb'))
#         Acc, Recall, prec, f1_sc, kappa, mcm= self.multi_label_confusion_matrix(y_test, best_model.predict_classes(np.transpose(X_test)))
#         
#         # Predict
#         y_pred = model.predict_classes(np.transpose(X_test))
#         
#         return y_pred, history
#     
#     #%% CRNN classifier (both the premodel and model training comes here)
#     def CRNN_premodel_classifier(self, X_train, y_train, fs, n_filters = [8, 16, 32], 
#                         kernel_size = [50, 8, 8], loss='mean_squared_error', 
#                         optimizer='adam',metrics = ['accuracy'],
#                         epochs = 10, batch_size = 128, verbose = 1,
#                         show_summarize =True, plot_model_graph =True, show_shapes = False):
#         
#         """ This is a CNN model which is the premodel of CRNN network. The user have to
#         pretrain this network first and then the subsequent LSTM network 
#         (CRNN_model_classifier) outputs the final classification.
#         
#         Please note: So the CNN weights will be adjusted here and then will be fixed
#         dutring the main training of '(CRNN_model_classifier)'
#         """
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Importing libraries ~~~~~~~~~~~~~~~~~~~~~~ #
#         import keras
#         from keras.utils import plot_model
#         from keras.models import Model
#         from keras.layers import Input
#         from keras.layers import Dense
#         from keras.layers.recurrent import LSTM
#         from keras.layers.merge import concatenate
#         from keras.layers.convolutional import Conv1D
#         from keras.layers.pooling import MaxPooling1D
#         from keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed,Flatten, Bidirectional
# 
#         # ~~~~~~~~~~~~~~~~~~~~~~~~Defining input data~~~~~~~~~~~~~~~~~~~~~~~~ #
#     
#         # input is 30-s epoch of sleep (samples, timesteps, features])
#         input_sig1 = Input(shape=(30*fs,1))
#                 
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Creating CNN model ~~~~~~~~~~~~~~~~~~~~~~~ #
#         
#         ### === Layer 1 === ###
#         
#         # Conv 1
#         conv1 = Conv1D(n_filters[0], kernel_size = kernel_size[0], strides = 1, 
#                         padding='same', kernel_initializer='glorot_uniform')(input_sig1)
#         # Batch normalization 1
#         batch1 = BatchNormalization()(conv1)
#         # Activation 1
#         act1 = Activation('relu')(batch1)
#         # Maxpooling 1
#         maxpooling1 = MaxPooling1D(pool_size=8, strides=1)(act1)
#         
#         ### === Layer 2 === ###
#         
#         # Conv 2
#         conv2 = Conv1D(n_filters[1], kernel_size= kernel_size[1], strides = 1, 
#               padding='same', kernel_initializer='glorot_uniform')(maxpooling1)
#         # Batch normalization 2
#         batch2 = BatchNormalization()(conv2)
#         # Activation 2
#         act2 = Activation('relu')(batch2)
#         # Max-pooling 2
#         maxpooling2 = MaxPooling1D(pool_size=8, strides=1)(act2)
#         
#         ### === Layer 3 === ###
#         
#         # Conv 3
#         conv3 = Conv1D(n_filters[2], kernel_size= kernel_size[2], strides = 1, 
#               padding='same', kernel_initializer='glorot_uniform')(maxpooling2)
#         # Batch normalization 3
#         batch3 = BatchNormalization()(conv3)
#         # Activation 3
#         act3 = Activation('relu')(batch3)
#         # Max-pooling 3
#         maxpooling3 = MaxPooling1D(pool_size=8, strides=1)(act3) 
#         
#         # flattening
#         flattened = Flatten()(maxpooling3)
#         
#         # Final Dense layer of CNN
#         final_premodel = Dense(5, activation = 'softmax')(flattened)
#         
#         # Activates tf variable and cpu/gpu
#         with tf.device('/cpu:0'):
#             premodel = Model(inputs=[input_sig1], outputs=[final_premodel])
#         
#         # compile
#         optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
#                                           beta_2=0.999, epsilon=1e-08)
#         premodel.compile(loss=loss, optimizer= optimizer, metrics = metrics)   
#         
#         # plot graph
#         if plot_model_graph ==True:  
#             plot_model(premodel, show_shapes = True)
#             
#         # summarize layers
#         if show_summarize ==True:
#             print(premodel.summary())
#             
#         # Fit and train
#         premodel.fit(np.transpose(X_train), y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
#              
#         return premodel
#     
#     #### ====================== Main CRNN model ======================== ###### 
#     # CRNN classifier
#     def CRNN_main_classifier(self, premodel, X_train, y_train, fs, before_flatten_layer,
#                             loss='mean_squared_error', 
#                             LSTM_units = 64, recurrent_dropout = .3,
#                             optimizer='adam',metrics = ['accuracy'],
#                             epochs = 10, batch_size = 128, verbose = 1,
#                             show_summarize =True, plot_model_graph =True, show_shapes = False):
#         
#         """ This is the LSTM model which is the final section of CRNN network.
#         
#         Please note: So the CNN weights will be adjusted here and then will be fixed
#         dutring the main training of '(CRNN_model_classifier)'
#         """
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Importing libraries ~~~~~~~~~~~~~~~~~~~~~~ #
#         import keras
#         from keras.utils import plot_model
#         from keras.models import Model
#         from keras.layers import Input, Reshape
#         from keras.layers import Dense
#         from keras.layers.recurrent import LSTM
#         from keras.layers.merge import concatenate
#         from keras.layers.convolutional import Conv1D
#         from keras.layers.pooling import MaxPooling1D
#         from keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed,Flatten, Bidirectional
# 
#         # ~~~~~~~~ Take tha last layer before Flattening of base model~~~~~~~ #
#         # i is the index of the layer from which you want to take the model output
#         before_flatten = premodel.layers[before_flatten_layer].output 
#         
#         # Reshape and preparation for LSTM
#         conv2lstm_reshape = Reshape((-1, 2))(before_flatten)
#         
#         # Deactivate training of pre-trained model (CNN)
#         premodel.trainable = False
# 
#         # ~~~~~~~~~~~~~~~~~~~~~~~ Creating LSTM model ~~~~~~~~~~~~~~~~~~~~~~~ #
#     
#         # Adding LSTM 1
#         lstm1 = Bidirectional(LSTM(units = LSTM_units, recurrent_dropout= recurrent_dropout, return_sequences=True))(conv2lstm_reshape) 
#         # Adding LSTM 2         
#         lstm2 = Bidirectional(LSTM(units = LSTM_units,recurrent_dropout= recurrent_dropout, return_sequences=True))(lstm1)
#         # Adding LSTM 3
#         lstm3 = Bidirectional(LSTM(units = LSTM_units,recurrent_dropout= recurrent_dropout, return_sequences=True))(lstm2)
#         # Adding LSTM 4
#         lstm4 = Bidirectional(LSTM(units = LSTM_units,recurrent_dropout= recurrent_dropout, return_sequences=True))(lstm3)
#         
#         # flattening
#         flattened = Flatten()(lstm4)
#         
#         # Final Dense layer of CNN + LSTM stack
#         final = Dense(5, activation = 'softmax')(flattened)
#         
#         # Activates tf variable and cpu/gpu
#         with tf.device('/cpu:0'):
#             model = Model(inputs=[premodel.input], outputs=[final])
#         
#         # compile
#         optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
#                                           beta_2=0.999, epsilon=1e-08)
#         model.compile(loss=loss, optimizer= optimizer, metrics = metrics)   
#         
#         # plot graph
#         if plot_model_graph ==True:  
#             plot_model(model, show_shapes = show_shapes)
#             
#         # summarize layers
#         if show_summarize ==True:
#             print(model.summary())
#             
#         # Fit and train
#         #model.fit(np.transpose(X_train), y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
#              
#         return model
#     #### ==================== RUN full CRNN model ====================== ###### 
#     def run_CRNN_model(self,  X_train, y_train, before_flatten_layer = 12, fs=200, n_filters = [8, 16, 32], 
#                         kernel_size = [50, 8, 8], loss='mean_squared_error', 
#                         LSTM_units = 64, recurrent_dropout = .3,
#                         optimizer='adam',metrics = ['accuracy'],
#                         epochs = 10, batch_size = 128, verbose = 1,
#                         show_summarize =True, plot_model_graph =True, show_shapes = False):
#         
#         # Premodel definition (CNN)
#         self.premodel = self.CRNN_premodel_classifier(self, X_train, y_train, fs=fs, n_filters = n_filters, 
#                         kernel_size = kernel_size, loss=loss, 
#                         optimizer=optimizer,metrics = metrics,
#                         epochs = epochs, batch_size = batch_size, verbose = verbose,
#                         show_summarize =show_summarize, plot_model_graph =plot_model_graph,
#                         show_shapes = show_shapes)
# 
#         # main model CRNN (LSTM)
#         model = self.CRNN_main_classifier(self, self.premodel, X_train, y_train, fs, 
#                                           before_flatten_layer=before_flatten_layer,
#                                           loss=loss, LSTM_units = LSTM_units, recurrent_dropout = recurrent_dropout,
#                                           optimizer=optimizer,metrics =metrics,
#                                           epochs = epochs, batch_size = batch_size, verbose = verbose,
#                                           show_summarize =show_summarize, 
#                                           plot_model_graph =plot_model_graph, 
#                                           show_shapes = show_shapes)
#         
#         # return final trained model
#         return model
#     
#     #%% Plot deep neural networks model
#     def plot_models(self, model, show_shapes = True):
#      
#         from keras.utils import plot_model
#         plot_model(model, show_shapes = show_shapes)
#         
#     #%% Show summary of  deep neural networks
#     def summary_models(self, model):
#           print(model.summary())
#     #%% DeepSleepNet
#     
#     def DeepSleepNet_pretraining_classifier(self, X_train, X_test, fs,
#                                             loss='mean_squared_error',
#                                             metrics = ['accuracy'],
#                                             epochs = 10, batch_size = 128, verbose = 1,
#                                             show_summarize =True, plot_model_graph =True,
#                                             show_shapes = False):
#         
#         "This is based on the pre-model created called: DeepSleepNet" 
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Importing libraries ~~~~~~~~~~~~~~~~~~~~~~ #
#         import keras
#         from keras.utils import plot_model
#         from keras.models import Model
#         from keras.layers import Input
#         from keras.layers import Dense
#         from keras.layers.recurrent import LSTM
#         from keras.layers.merge import concatenate
#         from keras.layers.convolutional import Conv1D
#         from keras.layers.pooling import MaxPooling1D
#         from keras.layers import Dropout, BatchNormalization, Activation, TimeDistributed,Flatten, Bidirectional
# 
# 
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Reshaping input data~~~~~~~~~~~~~~~~~~~~~~ #
#         
#         # reshape from [samples, timesteps] into [samples, timesteps, features]
#         
#         trainX = np.transpose(X_train)
#         testX  = np.transpose(X_test)
#         n_timesteps, n_features = trainX.shape[1], trainX.shape[2]
#         
#         # ~~~~~~~ Create model 1: Large Filter(lf) --> Frequency feats ~~~~~~~# 
#         
#         # ~~~~~~~~~~~~~~~~~~~~~~~~Defining input data~~~~~~~~~~~~~~~~~~~~~~~~ #
#     
#         # input is 30-s epoch of sleep (samples, timesteps, features])
#         input_sig = Input(shape=(30*fs,1))
#         
#         #conv1 + Batch normalize (before activation) + RELU activation
#         model_lf_conv1  = Conv1D(filters = 64, kernel_size= int(fs/2), strides = int(fs/16), use_bias=False)(input_sig)
#         model_lf_bacth1 = BatchNormalization()(model_lf_conv1)
#         model_lf_act1   = Activation("relu")(model_lf_bacth1)
# 
#         
#         # Max-pooling1
#         model_lf_maxpooling1 = MaxPooling1D(pool_size=8, strides = 8)(model_lf_act1)
#         
#         # Dropout1
#         model_lf_dropout1 = Dropout(.5)(model_lf_maxpooling1)
#         
#         # conv2 + Batch normalize (before activation) + RELU activation
#         model_lf_conv2  = Conv1D(filters = 128, kernel_size= 8, use_bias=False)(model_lf_dropout1)
#         model_lf_bacth2 = BatchNormalization()(model_lf_conv2)
#         model_lf_act2   = Activation("relu")(model_lf_bacth2)
#         
#         # conv3 + Batch normalize (before activation) + RELU activation
#         model_lf_conv3  = Conv1D(filters = 128, kernel_size= 8, use_bias=False)(model_lf_act2)
#         model_lf_batch3 = BatchNormalization()(model_lf_conv3)
#         model_lf_act3   = Activation("relu")(model_lf_batch3)
#         
#         # conv4 + Batch normalize (before activation) + RELU activation
#         model_lf_conv4  = Conv1D(filters = 128, kernel_size= 8, use_bias=False)(model_lf_act3)
#         model_lf_batch4 = BatchNormalization()(model_lf_conv4)
#         model_lf_act4   = Activation("relu")(model_lf_batch4)
#         
#         # Max-pooling 2
#         model_lf_maxpooling2 = MaxPooling1D(pool_size = 4, strides = 4)(model_lf_act4)
#         
#         # Flattening 
#         model_lf_flattened = Flatten()(model_lf_maxpooling2)
#         
#         # ~~~~~~~ Create model 2: Small Filter(sf) --> Temporal feats ~~~~~~~~#
#         # init --> takes the same input as model_lf
#         
#         # Conv1
#         model_sf_conv1  = Conv1D(filters = 64, kernel_size= int(fs*4), use_bias=False, strides = int(fs/2))(input_sig)
#         model_sf_bacth1 = BatchNormalization()(model_sf_conv1)
#         model_sf_act1   = Activation("relu")(model_sf_bacth1)
#         
#         # Max-pooling 1
#         model_sf_maxpooling1 = MaxPooling1D(pool_size = 4, strides = 4)(model_sf_act1)
#         
#         # Dropout1
#         model_sf_dropout1 = Dropout(.5)(model_sf_maxpooling1)
#         
#         # Conv2
#         model_sf_conv2  = Conv1D(filters = 128, kernel_size= 6, use_bias=False)(model_sf_dropout1)
#         model_sf_batch2 = BatchNormalization()(model_sf_conv2)
#         model_sf_act2   = Activation("relu")(model_sf_batch2)
#         
#         # Conv3
#         model_sf_conv3  = Conv1D(filters = 128, kernel_size= 6, use_bias=False)(model_sf_act2)
#         model_sf_batch3 = BatchNormalization()(model_sf_conv3)
#         model_sf_act3   = Activation("relu")(model_sf_batch3)
#         
#         # Conv4
#         model_sf_conv4  = Conv1D(filters = 128, kernel_size= 6, use_bias=False)(model_sf_act3)
#         model_sf_batch4 = BatchNormalization()(model_sf_conv4)
#         model_sf_act4   = Activation("relu")(model_sf_batch4)
#         
#         # Max-pooling 2
#         model_sf_maxpooling2 = MaxPooling1D(pool_size = 2, strides = 2)(model_sf_act4)
#         
#         # Flattening 
#         model_sf_flattened = Flatten()(model_sf_maxpooling2)
#         
#         # ====================== Concatenation of models ==================== #
#         
#         # Merge CNN_sf and CNN_lf
#         merged_CNNs = concatenate([model_lf_flattened, model_sf_flattened])
#         
#         # Dopout
#         dropout_after_concatenation = Dropout(.5)(merged_CNNs)
#         
#         # Final layer for pre-training (5 neurons for wake to REM)
#         dense_after_concatenation = Dense(5, activation='softmax')(dropout_after_concatenation)
#     
#         # Create final model:Input is the same array of training, but twice 
#         
#         with tf.device('/cpu:0'):
#             premodel = Model(inputs=[input_sig], outputs=[dense_after_concatenation])
#             
#         # compile
#         optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
#                                           beta_2=0.999, epsilon=1e-08)
#         premodel.compile(loss=loss, optimizer= optimizer, metrics = metrics)   
#         
#         # plot graph
#         if plot_model_graph ==True:  
#             plot_model(premodel, show_shapes = True)
#             
#         # summarize layers
#         if show_summarize ==True:
#             print(premodel.summary())
#             
#         # Fit and train
#         #premodel.fit(np.transpose(X_train), y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
#         
#         return premodel
#     
#     #%% DeepSleepNet - LSTM
#     def DeepSleepNet_LSTM(self, input_, vec_len, timesteps):
#         
#         "This is based on the LSTM-model created in DeepSleepNet" 
#         
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Importing libraries ~~~~~~~~~~~~~~~~~~~~~~ #
#         import tensorflow as tf
#         from keras.models import Sequential, Model
#         from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation, concatenate, Bidirectional, LSTM
#         from keras.layers.convolutional import Conv1D
#         from keras.layers.convolutional import MaxPooling1D
#         from keras.optimizers import adam
# 
#         # Define input shape from CNNs (True?!!)
#         input_shape = (timesteps, vec_len)
#         
#         # ~~~~~~~~~~~~~~~~~~~~~ Creating fully_connected ~~~~~~~~~~~~~~~~~~~~ #
#         
#         # Initialize
#         model_fc = Sequential()
#         
#         # Add fully connected layer (feedback from CNN to LSTM output)
#         model_fc.add(Dense(1024,input_shape = input_shape))
#         
#         # BatchNormalization
#         model_fc.add(BatchNormalization())
#         
#         # Activation
#         model_fc.add(Activation("relu"))
# 
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Creating bidir-LSTM ~~~~~~~~~~~~~~~~~~~~~~ #
#         
#         # Initialize
#         model_lstm = Sequential()
#         
#         # 1st LSTM
#         model_lstm.add(Bidirectional(LSTM(512, input_shape = input_shape)))
# 
#         # Dopout
#         model_lstm.add(Dropout(.5))
#         
#         # 2nd LSTM
#         model_lstm.add(Bidirectional(LSTM(512)))
# 
#         # Dopout
#         model_lstm.add(Dropout(.5))        
#         
#         # ~~~~~~~~~~~~~~~~~~~~~~~~ Merge LSTM and fc ~~~~~~~~~~~~~~~~~~~~~~~~ #
#         merged = concatenate([model_lstm, model_fc], axis = -1)
#         
#         # Flatten
#         merged.flatten()
#         
#         # Dopout
#         merged.add(Dropout(.5))
#         
#         # Final layer for pre-training (5 neurons for wake to REM)
#         merged.add(Dense(5, activation='softmax'))
#         
#         # Compile 
#         merged.compile(loss='categorical_crossentropy', 
#                             optimizer= 'adam', metrics=[tf.keras.metrics.Recall()])   
#         
#         # Create final model:Input is the same array of training, but twice 
#         
#         with tf.device('/cpu:0'):
#             model = Model(inputs=[input_, input_], outputs=[merged])
#             
#         return model
# 
# 
# 
# 
# =============================================================================
