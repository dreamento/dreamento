# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 20:19:56 2022

@author: mahjaf
"""

# Import libs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import mne
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize, ListedColormap
import os
%matplotlib qt

################################### File path #################################
# Define path
global path_data
path_data = "P:\\3013097.06\\Data\\s35\\n15\\somno\\"
subject_number = "s35_n15_(2).edf"
###############################################################################

# Define functions
def on_click(event):
    
    global ix, iy, counter_event
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y = {iy}')
    counter_event = counter_event + 1
    global coords
    coords.append((ix, iy))

    if len(coords) % 2 == 0:
        print('You have already selected all the points!')

def Unscorable(event):
    tmp_idx_min = []
    tmp_idx_max = []
    current_coords = [coords[counter_event-3][0], coords[counter_event-2][0]]
    ax[4].fill_between(current_coords, [6, 6], color = 'gray')
    fig.canvas.draw()
    print("marked as unscorable!")
    
    for i in t1:
        tmp_idx_min.append(abs(i-current_coords[0]))
        tmp_idx_max.append(abs(i-current_coords[-1]))
        
    idx_start = np.argmin(tmp_idx_min)
    idx_end   = np.argmin(tmp_idx_max)

    scoring_area = np.arange(idx_start, idx_end + 1)
    Scoring_list[scoring_area, 0] = np.ones(len(scoring_area)) * -1
    
def Wake(event):
    tmp_idx_min = []
    tmp_idx_max = []
    current_coords = [coords[counter_event-3][0], coords[counter_event-2][0]]
    ax[4].fill_between(current_coords, [5, 5], color = 'black')
    fig.canvas.draw()
    print("marked as Wake!")
    
    for i in t1:
        tmp_idx_min.append(abs(i-current_coords[0]))
        tmp_idx_max.append(abs(i-current_coords[-1]))
        
    idx_start = np.argmin(tmp_idx_min)
    idx_end   = np.argmin(tmp_idx_max)

    scoring_area = np.arange(idx_start, idx_end + 1)
    Scoring_list[scoring_area, 0] = np.ones(len(scoring_area)) * 0

def REM(event):
    tmp_idx_min = []
    tmp_idx_max = []
    current_coords = [coords[counter_event-3][0], coords[counter_event-2][0]]
    ax[4].fill_between(current_coords, [4, 4], color = 'red')
    fig.canvas.draw()
    print("marked as REM!")
    
    for i in t1:
        tmp_idx_min.append(abs(i-current_coords[0]))
        tmp_idx_max.append(abs(i-current_coords[-1]))
        
    idx_start = np.argmin(tmp_idx_min)
    idx_end   = np.argmin(tmp_idx_max)

    scoring_area = np.arange(idx_start, idx_end + 1)
    Scoring_list[scoring_area, 0] = np.ones(len(scoring_area)) * 5

def N1(event):
    tmp_idx_min = []
    tmp_idx_max = []
    current_coords = [coords[counter_event-3][0], coords[counter_event-2][0]]
    ax[4].fill_between(current_coords, [3, 3], color = 'thistle')
    fig.canvas.draw()
    print("marked as N1!")
    
    for i in t1:
        tmp_idx_min.append(abs(i-current_coords[0]))
        tmp_idx_max.append(abs(i-current_coords[-1]))
        
    idx_start = np.argmin(tmp_idx_min)
    idx_end   = np.argmin(tmp_idx_max)

    scoring_area = np.arange(idx_start, idx_end + 1)
    Scoring_list[scoring_area, 0] = np.ones(len(scoring_area)) * 1

def N2(event):
    tmp_idx_min = []
    tmp_idx_max = []
    current_coords = [coords[counter_event-3][0], coords[counter_event-2][0]]
    ax[4].fill_between(current_coords, [2, 2], color = 'deepskyblue')
    fig.canvas.draw()
    print("marked as N2!")
    
    for i in t1:
        tmp_idx_min.append(abs(i-current_coords[0]))
        tmp_idx_max.append(abs(i-current_coords[-1]))
        
    idx_start = np.argmin(tmp_idx_min)
    idx_end   = np.argmin(tmp_idx_max)

    scoring_area = np.arange(idx_start, idx_end + 1)
    Scoring_list[scoring_area, 0] = np.ones(len(scoring_area)) * 2

def N3(event):
    tmp_idx_min = []
    tmp_idx_max = []
    current_coords = [coords[counter_event-3][0], coords[counter_event-2][0]]
    ax[4].fill_between(current_coords, [1, 1], color = 'slateblue')
    fig.canvas.draw()
    print("marked as SWS (N3)!")
    
    for i in t1:
        tmp_idx_min.append(abs(i-current_coords[0]))
        tmp_idx_max.append(abs(i-current_coords[-1]))
        
    idx_start = np.argmin(tmp_idx_min)
    idx_end   = np.argmin(tmp_idx_max)

    scoring_area = np.arange(idx_start, idx_end + 1)
    Scoring_list[scoring_area, 0] = np.ones(len(scoring_area)) * 3

def Movement_Artefact(event):
    tmp_idx_min = []
    tmp_idx_max = []
    current_coords = [coords[counter_event-3][0], coords[counter_event-2][0]]
    ax[4].fill_between(current_coords, [.5, .5], color = 'olive')
    fig.canvas.draw()
    print("marked as Movement artefact!")
    
    for i in t1:
        tmp_idx_min.append(abs(i-current_coords[0]))
        tmp_idx_max.append(abs(i-current_coords[-1]))
        
    idx_start = np.argmin(tmp_idx_min)
    idx_end   = np.argmin(tmp_idx_max)

    scoring_area = np.arange(idx_start, idx_end + 1)
    Scoring_list[scoring_area, 1] = np.ones(len(scoring_area)) * 1

def Remove_scoring(event):
    tmp_idx_min = []
    tmp_idx_max = []
    current_coords = [coords[counter_event-3][0], coords[counter_event-2][0]]
    ax[4].fill_between(current_coords, [6, 6], color = 'white')
    fig.canvas.draw()
    print("The selected area' scoring removed!")
    
    for i in t1:
        tmp_idx_min.append(abs(i-current_coords[0]))
        tmp_idx_max.append(abs(i-current_coords[-1]))
        
    idx_start = np.argmin(tmp_idx_min)
    idx_end   = np.argmin(tmp_idx_max)

    scoring_area = np.arange(idx_start, idx_end + 1)
    Scoring_list[scoring_area, 0] = np.ones(len(scoring_area)) * -1

def Export_scoring(event):
    saving_dir = path_data + '/scoring/' + subject_number.split('.edf')[0] + '_QAASM_MJE.txt'
    if os.path.exists(saving_dir):
        os.remove(saving_dir)
    np.savetxt(saving_dir, Scoring_list, fmt='%1.0f', delimiter='\t')
    plt.savefig(path_data + '/scoring/' + subject_number.split('.edf')[0] + '_QAASM_MJE.png',dpi = 300)  
    print(f'Scoring results have been saved in: {saving_dir}')

# Read data
data = mne.io.read_raw_edf(path_data + subject_number)

# Find index of required channels     
raw_data = data.get_data()
availableChannels = data.ch_names

RequiredChannels = ['F4:A1']
Idx_F4_A1 = []
for indx, c in enumerate(RequiredChannels):
    if c in availableChannels:
        Idx_F4_A1.append(availableChannels.index(c))

RequiredChannels = ['C4:A1']
Idx_C4_A1 = []
for indx, c in enumerate(RequiredChannels):
    if c in availableChannels:
        Idx_C4_A1.append(availableChannels.index(c))

RequiredChannels = ['O2:A1']
Idx_O2_A1 = []
for indx, c in enumerate(RequiredChannels):
    if c in availableChannels:
        Idx_O2_A1.append(availableChannels.index(c))

RequiredChannels = ['EMG']
Idx_EMG = []
for indx, c in enumerate(RequiredChannels):
    if c in availableChannels:
        Idx_EMG.append(availableChannels.index(c))
        
# Fetch the data from the required channel
data_F4_A1 = raw_data[Idx_F4_A1, :]
data_C4_A1 = raw_data[Idx_C4_A1, :]
data_O2_A1 = raw_data[Idx_O2_A1, :]
data_EMG   = raw_data[Idx_EMG  , :]

# Init
win_sec = 30
fmin_EEG = .1
fmax_EEG = 30
fmin_EMG = 10
fmax_EMG = 100
sf = 256
noverlap = 0
trimperc= 5
cmap='RdBu_r'
log_power = False

# Increase font size while preserving original
old_fontsize = plt.rcParams['font.size']
plt.rcParams.update({'font.size': 12})

sig1  = data_F4_A1[0]
sig2  = data_C4_A1[0]
sig3  = data_O2_A1[0]
sig4  = data_EMG[0]

# Safety checks
assert isinstance(sig1, np.ndarray), 'Data1 must be a 1D NumPy array.'
assert isinstance(sig2, np.ndarray), 'Data2 must be a 1D NumPy array.'
assert isinstance(sig3, np.ndarray), 'Data1 must be a 1D NumPy array.'
assert isinstance(sig4, np.ndarray), 'Data2 must be a 1D NumPy array.'

assert isinstance(sf, (int, float)), 'sf must be int or float.'

assert sig1.ndim == 1, 'Data1 must be a 1D (single-channel) NumPy array.'
assert sig2.ndim == 1, 'Data2 must be a 1D (single-channel) NumPy array.'
assert sig3.ndim == 1, 'Data1 must be a 1D (single-channel) NumPy array.'
assert sig4.ndim == 1, 'Data2 must be a 1D (single-channel) NumPy array.'

assert isinstance(win_sec, (int, float)), 'win_sec must be int or float.'
# =============================================================================
# assert isinstance(fmin, (int, float)), 'fmin must be int or float.'
# assert isinstance(fmax, (int, float)), 'fmax must be int or float.'
# assert fmin < fmax, 'fmin must be strictly inferior to fmax.'
# assert fmax < sf / 2, 'fmax must be less than Nyquist (sf / 2).'
# 
# =============================================================================
# Calculate multi-taper spectrogram
nperseg = int(win_sec * sf)

assert sig1.size > 2 * nperseg, 'Data1 length must be at least 2 * win_sec.'
assert sig2.size > 2 * nperseg, 'Data2 length must be at least 2 * win_sec.'
assert sig3.size > 2 * nperseg, 'Data1 length must be at least 2 * win_sec.'
assert sig4.size > 2 * nperseg, 'Data2 length must be at least 2 * win_sec.'

global t1
f1, t1, Sxx1 = spectrogram_lspopt(sig1, sf, nperseg=nperseg, noverlap=noverlap)
f2, t2, Sxx2 = spectrogram_lspopt(sig2, sf, nperseg=nperseg, noverlap=noverlap)
f3, t3, Sxx3 = spectrogram_lspopt(sig3, sf, nperseg=nperseg, noverlap=noverlap)
f4, t4, Sxx4 = spectrogram_lspopt(sig4, sf, nperseg=nperseg, noverlap=noverlap)

Sxx1 = 10 * np.log10(Sxx1)  # Convert uV^2 / Hz --> dB / Hz
Sxx2 = 10 * np.log10(Sxx2)  # Convert uV^2 / Hz --> dB / Hz
Sxx3 = 10 * np.log10(Sxx3)  # Convert uV^2 / Hz --> dB / Hz
Sxx4 = 10 * np.log10(Sxx4)  # Convert uV^2 / Hz --> dB / Hz

# Select only relevant frequencies (up to 30 Hz)
good_freqs1 = np.logical_and(f1 >= fmin_EEG, f1 <= fmax_EEG)
good_freqs2 = np.logical_and(f2 >= fmin_EEG, f2 <= fmax_EEG)
good_freqs3 = np.logical_and(f3 >= fmin_EEG, f3 <= fmax_EEG)
good_freqs4 = np.logical_and(f4 >= fmin_EMG, f4 <= fmax_EMG)

Sxx1 = Sxx1[good_freqs1, :]
Sxx2 = Sxx2[good_freqs2, :]
Sxx3 = Sxx3[good_freqs3, :]
Sxx4 = Sxx4[good_freqs4, :]


f1 = f1[good_freqs1]
f2 = f2[good_freqs2]
f3 = f3[good_freqs3]
f4 = f4[good_freqs4]


t1 /= 3600  # Convert t to hours
t2 /= 3600  # Convert t to hours
t3 /= 3600  # Convert t to hours
t4 /= 3600  # Convert t to hours


# Normalization
vmin1, vmax1 = np.percentile(Sxx1, [0 + trimperc, 100 - trimperc])
vmin2, vmax2 = np.percentile(Sxx2, [0 + trimperc, 100 - trimperc])
vmin3, vmax3 = np.percentile(Sxx3, [0 + trimperc, 100 - trimperc])
vmin4, vmax4 = np.percentile(Sxx4, [0 + trimperc, 100 - trimperc])

norm1 = Normalize(vmin=vmin1, vmax=vmax1)
norm2 = Normalize(vmin=vmin2, vmax=vmax2)
norm3 = Normalize(vmin=vmin3, vmax=vmax3)
norm4 = Normalize(vmin=vmin4, vmax=vmax4)

# Plotting
fig, ax = plt.subplots(5 ,figsize=(10, 4))
global Scoring_list
Scoring_list = -1 * np.ones((len(t1), 2))
Scoring_list[:, 1] = np.zeros(len(t1))
coords = []
counter_event = 0

ax[0].pcolormesh(t1, f1, Sxx1, norm=norm1, cmap=cmap, antialiased=True,
                               shading="auto")
ax[0].set_ylabel('EEG (F4:A1)')
ax[1].pcolormesh(t2, f2, Sxx2, norm=norm2, cmap=cmap, antialiased=True,
                               shading="auto")
ax[1].set_ylabel('EEG (C4:A1)')
ax[2].pcolormesh(t3, f3, Sxx3, norm=norm3, cmap=cmap, antialiased=True,
                               shading="auto")
ax[2].set_ylabel('EEG (O2:A1)')
ax[3].pcolormesh(t4, f4, Sxx4, norm=norm4, cmap=cmap, antialiased=True,
                               shading="auto")
ax[3].set_ylabel('EMG')

ax[4].set_xlim([t1.min(), t1.max()])
ax[4].set_yticks([6, 5, 4, 3, 2, 1, .5], ['Unscorable', 'Wake','REM', 'N1', 'N2', 'SWS', 'Movement Artefact'])
ax[4].set_ylabel('Scoring')

plt.connect('button_press_event', on_click)

axcut = plt.axes([0.15, 0.0, 0.15, 0.075])
axcut2 = plt.axes([0.35, 0.0, 0.05, 0.075])
axcut3 = plt.axes([0.45, 0.0, 0.05, 0.075])
axcut4 = plt.axes([0.55, 0.0, 0.05, 0.075])
axcut5 = plt.axes([0.65, 0.0, 0.05, 0.075])
axcut6 = plt.axes([0.75, 0.0, 0.05, 0.075])
axcut7 = plt.axes([0.85, 0.0, 0.05, 0.075])
axcut8 = plt.axes([0.95, 0.0, 0.05, 0.075])
axcut9 = plt.axes([0.0, 0.0, 0.05, 0.075])

bcut = Button(axcut, 'Unscorable', color='lightgray', hovercolor='dimgray')
bcut2 = Button(axcut2, 'N1', color= 'thistle', hovercolor='purple')
bcut3 = Button(axcut3, 'N2', color='deepskyblue', hovercolor='purple')
bcut4 = Button(axcut4, 'N3', color='slateblue', hovercolor='purple')
bcut5 = Button(axcut5, 'REM', color='red', hovercolor='hotpink')
bcut6 = Button(axcut6, 'Wake', color='grey', hovercolor='green')
bcut7 = Button(axcut7, 'MA', color='olive', hovercolor='blue')
bcut8 = Button(axcut8, 'Remove', color='white', hovercolor='gray')
bcut9 = Button(axcut9, 'Export scoring', color='salmon', hovercolor='plum')


bcut.on_clicked(Unscorable)
bcut2.on_clicked(N1)
bcut3.on_clicked(N2)
bcut4.on_clicked(N3)
bcut5.on_clicked(REM)
bcut6.on_clicked(Wake)
bcut7.on_clicked(Movement_Artefact)
bcut8.on_clicked(Remove_scoring)
bcut9.on_clicked(Export_scoring)


plt.show()