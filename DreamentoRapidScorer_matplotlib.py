# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 20:19:56 2022

@author: mahjaf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import mne
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize, ListedColormap

%matplotlib qt

path_data = "P:\\3013097.06\\Data\\s37\\n15\\zmax\\EEG L.edf"

data = mne.io.read_raw_edf(path_data)
raw_data = data.get_data()
dataY = np.ravel(raw_data)
dataX = np.arange(len(dataY))

# Init
win_sec = 30
fmin = .1
fmax = 30
sf = 256
noverlap = 0
trimperc=5
cmap='RdBu_r'
log_power = False

# Increase font size while preserving original
old_fontsize = plt.rcParams['font.size']
plt.rcParams.update({'font.size': 12})

# Safety checks
data = dataY
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
 
fig, ax = plt.subplots(2 ,figsize=(10, 4))

coords = []
counter_event = 0

def on_click(event):
    
    global ix, iy, counter_event
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y = {iy}')
    counter_event = counter_event + 1
    global coords
    coords.append((ix, iy))

    if len(coords) % 2 == 0:
        print('You have already selected all the points!')
# =============================================================================
#     
#     if event.dblclick:
#        ax.plot((event.xdata, event.xdata), (mean-standardDeviation, mean+standardDeviation), 'r-')
#        plt.show()
# =============================================================================

def Unscorable(event):
    ax[1].fill_between([coords[counter_event-3][0], coords[counter_event-2][0]], [6, 6], color = 'gray')
    fig.canvas.draw()
    print("marked as Unscorable!")
    
def Wake(event):
    ax[1].fill_between([coords[counter_event-3][0], coords[counter_event-2][0]], [5, 5], color = 'black')
    fig.canvas.draw()
    print("marked as Wake!")

def REM(event):
    ax[1].fill_between([coords[counter_event-3][0], coords[counter_event-2][0]], [4, 4], color = 'red', linewidth = 2)
    fig.canvas.draw()
    print("marked as REM!")
    
def N1(event):
    ax[1].fill_between([coords[counter_event-3][0], coords[counter_event-2][0]], [3, 3], color = 'blue')
    fig.canvas.draw()
    print("marked as N1!")
    
def N2(event):
    ax[1].fill_between([coords[counter_event-3][0], coords[counter_event-2][0]], [2, 2], color = 'deepskyblue')
    fig.canvas.draw()
    print("marked as N2!")
    
def N3(event):
    ax[1].fill_between([coords[counter_event-3][0], coords[counter_event-2][0]], [1, 1], color = 'slateblue')
    fig.canvas.draw()
    print("marked as SWS!")
     

mean = np.mean(dataY)
standardDeviation = np.std(dataY)

ax[0].pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True,
                               shading="auto")
ax[1].set_xlim([t.min(), t.max()])
ax[1].set_yticks([6, 5, 4, 3, 2, 1], ['Unscorable', 'Wake','REM', 'N1', 'N2', 'SWS'])
plt.connect('button_press_event', on_click)

axcut = plt.axes([0.15, 0.0, 0.15, 0.075])
axcut2 = plt.axes([0.35, 0.0, 0.05, 0.075])
axcut3 = plt.axes([0.45, 0.0, 0.05, 0.075])
axcut4 = plt.axes([0.55, 0.0, 0.05, 0.075])
axcut5 = plt.axes([0.65, 0.0, 0.05, 0.075])
axcut6 = plt.axes([0.75, 0.0, 0.05, 0.075])

bcut = Button(axcut, 'Unscorable', color='gray', hovercolor='dimgray')
bcut2 = Button(axcut2, 'N1', color= 'lavender', hovercolor='purple')
bcut3 = Button(axcut3, 'N2', color='thistle', hovercolor='purple')
bcut4 = Button(axcut4, 'N3', color='plum', hovercolor='purple')
bcut5 = Button(axcut5, 'REM', color='hotpink', hovercolor='red')
bcut6 = Button(axcut6, 'Wake', color='olive', hovercolor='green')


bcut.on_clicked(Unscorable)
bcut2.on_clicked(N1)
bcut3.on_clicked(N2)
bcut4.on_clicked(N3)
bcut5.on_clicked(REM)
bcut6.on_clicked(Wake)


plt.show()