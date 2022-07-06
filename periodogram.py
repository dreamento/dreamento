import matplotlib.pyplot as plt
from matplotlib.mlab import psd
import numpy as np
from scipy.integrate import simps
import scipy.signal as ssignal
from yasa.spectral import bandpower as yas_bandpower    # for periodogram label values calculation

# %% Plot Power spectral density
def calculatePowerSpectralDensity(sig, fs, noverlap, NFFT=2 ** 11,  scale_by_freq=True):

    # Compute power spectrums
    try:
        # psd_sig, f_psd_sig = psd(x=sig, Fs=fs, NFFT=NFFT, scale_by_freq=scale_by_freq, noverlap=noverlap)
        f_psd_sig, psd_sig = ssignal.periodogram(x=sig, fs=fs)#, NFFT=NFFT, scale_by_freq=scale_by_freq, noverlap=noverlap)

    except ValueError:
        f_psd_sig, psd_sig = ssignal.periodogram(x=sig, fs=fs)
        # psd_sig, f_psd_sig = psd(x=np.ravel(sig), Fs=fs, NFFT=NFFT, scale_by_freq=scale_by_freq, noverlap=noverlap)

        # Compute log of power (optional)

    # ======================== Compute band power =========================== #

    # Defining EEG bands:
    eeg_bands = {'Delta': (0.5, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 11),
                 'Beta': (12, 24),
                 'Sigma': (12, 15)}

    freq_ix = dict()

    fxx = f_psd_sig
    pxx = psd_sig

    for band in eeg_bands:
        freq_ix[band] = np.where((fxx >= eeg_bands[band][0]) &
                                 (fxx <= eeg_bands[band][1]))[0]

    freq_resolu_per = fxx[1] - fxx[0]

    pow_total = simps(pxx[np.arange(freq_ix['Delta'][0], freq_ix['Beta'][-1])], dx=freq_resolu_per)
    Pow_Delta = simps(pxx[freq_ix['Delta']], dx=freq_resolu_per)
    Pow_Theta = simps(pxx[freq_ix['Theta']], dx=freq_resolu_per)
    Pow_Alpha = simps(pxx[freq_ix['Alpha']], dx=freq_resolu_per)
    Pow_Beta = simps(pxx[freq_ix['Beta']], dx=freq_resolu_per)
    Pow_Sigma = simps(pxx[freq_ix['Sigma']], dx=freq_resolu_per)

    # Ratios
    Pow_Delta_ratio = Pow_Delta / pow_total
    Pow_Theta_ratio = Pow_Theta / pow_total
    Pow_Alpha_ratio = Pow_Alpha / pow_total
    Pow_Beta_ratio = Pow_Beta / pow_total
    Pow_Sigma_ratio = Pow_Sigma / pow_total

    return psd_sig, f_psd_sig, Pow_Delta_ratio, Pow_Theta_ratio, Pow_Alpha_ratio, Pow_Beta_ratio, Pow_Sigma_ratio

def plotPowerSpectralDensity(figure=None, axis=None, sig=None,
                             filtering_status=True,
                             lowcut=.3,
                             highcut=30,
                             ):
    log_power = False
    ylimit = 'auto'  # [-50, 10]
    label = 'psd'
    freq_range = [0, 30]
    f = 20
    fs = 256
    Ts = 1 / fs
    t = np.arange(0, 30, Ts)
    if sig is None:
        sig = 1e-6* np.sin(2 * np.pi * f * t) + 2e-6* np.sin(2 * np.pi * 3 * t) + 10e-6*np.squeeze(np.random.rand(len(t), 1))
    else:
        if filtering_status:
            lowcut = lowcut
            highcut = highcut
            nyquist_freq = fs / 2.
            low = lowcut / nyquist_freq
            high = highcut / nyquist_freq
            # Req channel
            print("filtering for periodogram")
            b, a = ssignal.butter(3, [low, high], btype='band')
            sig = ssignal.filtfilt(b, a, sig)
    psd_sig, f_psd_sig, Pow_Delta_ratio, Pow_Theta_ratio, Pow_Alpha_ratio, Pow_Beta_ratio, Pow_Sigma_ratio = \
        calculatePowerSpectralDensity(sig=sig, fs=256, noverlap= 0, NFFT=len(sig), scale_by_freq=True)

    if log_power:
        psd_sig = 20 * np.log10(psd_sig)

    if figure == None or axis == None:
        # Open a new fig
        figure, axis = plt.subplots(1, 1, figsize=(20, 10))

    # Global setting for axes values size
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)

    # Plot signals
    axis.plot(f_psd_sig, psd_sig, label=label, color='blue')

    # Delta
    axis.axvline(.5, linestyle='--', color='black')
    axis.axvline(4, linestyle='--', color='black')

    # Theta
    axis.axvline(8, linestyle='--', color='black')

    # Alpha
    axis.axvline(12, linestyle='--', color='black')

    # Title and labels
    axis.set(title='Power spectral density', \
             xlabel='Frequency (Hz)', \
             ylabel='Power spectral density (dB/ Hz)')

    # Legend
#     legend = axis.legend([f"Delta: {round(Pow_Delta_ratio * 100, 2)}% \n\
# Theta: {round(Pow_Theta_ratio * 100, 2)}% \n\
# Alpha: {round(Pow_Alpha_ratio * 100, 2)}% \n\
# Beta: {round(Pow_Beta_ratio * 100, 2)}%"], prop={'size': 10}, frameon=False)
#     legend.set_draggable(state=True)

    # new decision: calculate powers from YASA and show them on the legend
    ret = yas_bandpower(sig, sf=fs)
    legend = axis.legend([f'Delta: {round(ret["Delta"][0] * 100, 2)}% \n\
Theta: {round(ret["Theta"][0] * 100, 2)}% \n\
Alpha: {round(ret["Alpha"][0] * 100, 2)}% \n\
Sigma: {round(ret["Sigma"][0] * 100, 2)}% \n\
Beta: {round(ret["Beta"][0] * 100, 2)}% \n\
Gamma: {round(ret["Gamma"][0] * 100, 2)}%'], prop={'size': 10}, frameon=False)
    legend.set_draggable(state=True)

    # Deactivate grid
    plt.grid(False)

    # Adding labels
    loc_pow_vals = np.mean([np.min(psd_sig), np.max(psd_sig)])
    loc_labels = loc_pow_vals
    axis.text(1.3, loc_labels, 'Delta', size=8)
    axis.text(4.8, loc_labels, 'Theta', size=8)
    axis.text(8.8, loc_labels, 'Alpha', size=8)
    axis.text(12.8, loc_labels, 'Beta', size=8)

    # loc_percents = np.min(psd_sig) + 10
    # # Write power percentages
    # axis.text(1, loc_percents, f'{round(Pow_Delta_ratio * 100, 2)}%', size=8)
    # axis.text(5, loc_percents, f'{round(Pow_Theta_ratio * 100, 2)}%', size=8)
    # axis.text(9, loc_percents, f'{round(Pow_Alpha_ratio * 100, 2)}%', size=8)
    # axis.text(13, loc_percents, f'{round(Pow_Beta_ratio * 100, 2)}%', size=8)

    # Limiting x-axis to 0-30 Hz
    axis.set_xlim(freq_range)

#     plt.figtext(0.5, 0, f"Delta: {round(Pow_Delta_ratio * 100, 2)}% | \
# Theta: {round(Pow_Theta_ratio * 100, 2)}% | \
# Alpha: {round(Pow_Alpha_ratio * 100, 2)}% | \
# Beta: {round(Pow_Beta_ratio * 100, 2)}%", ha="center", fontsize=12, \
#                 bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    if ylimit == 'auto':
        axis.set_ylim([np.min(psd_sig), np.max(psd_sig)])

    else:
        axis.set_ylim(ylimit)

    if figure == None or axis == None:
        plt.show()
        # DONT FORGET PLT.SHOW() IF YOU HAVE YOUR OWN FIGURE AND AXIS AS ARGUMENT

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    plotPowerSpectralDensity(figure=fig, axis=ax)
    plt.tight_layout()
    plt.show()