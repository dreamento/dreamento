#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`lsp`
==================

.. module:: lsp
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-13

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import six
import numpy as np
from scipy.signal import spectrogram

from data import C, WEIGHTS, f_h


def lspopt(n, c_parameter=20.0):
    """Calculate the multitaper windows described in [1], given window length and C parameter.

    Parameters
    ----------
    n : int
        Length of multitaper windows
    c_parameter : float
        The parameter `c` in [1]. Default is 20.0

    Returns
    -------
    H : ndarray
        Multitaper windows, of size [n_windows x n].
    w : ndarray
        Array of taper window weights.

    References
    ----------
    ...[1] Hansson-Sandsten, M. (2011). Optimal multitaper Wigner spectrum
           estimation of a class of locally stationary processes using Hermite functions.
           EURASIP Journal on Advances in Signal Processing, 2011, 10.

    """
    k = int(round((c_parameter - 1) * 10))
    if c_parameter != C[k]:
        print("Using c={0} instead of desired {1}.".format(C[k], c_parameter))

    weights = WEIGHTS[:, k][np.nonzero(WEIGHTS[:, k])][:10]
    weights /= np.sum(weights)
    K = len(weights)

    t1 = np.arange(-(n / 2) + 1, (n / 2) + 0.1, step=1.0) / f_h(n, K)
    h = np.vstack((np.ones((n,)), 2 * t1))
    for i in six.moves.range(1, K - 1):
        h = np.vstack((h, (2 * t1 * h.T[:, i]) - 2 * i * h.T[:, i - 1]))

    H = h.T * np.outer(np.exp(-(t1 ** 2) / 2), np.ones((K,), "float"))
    H /= np.array([np.linalg.norm(x) for x in H.T])

    return H.T.copy(), weights


def spectrogram_lspopt(
    x,
    fs=1.0,
    c_parameter=20.0,
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
):
    """Convenience method for calling :py:meth:`scipy.signal.spectrogram` with
    :py:mod:`lspopt` multitaper windows.

    Parameters
    ----------
    x : array_like
        Time series of measurement values
    c_parameter : float
        The parameter `c` in [1].

    For all other parameters, see the :py:meth:`scipy.signal.spectrogram`
    docstring.

    Returns
    -------
    f : ndarray
        Array of sample frequencies.
    t : ndarray
        Array of segment times.
    Sxx : ndarray
        Spectrogram of x. By default, the last axis of Sxx corresponds to the
        segment times.

    Notes
    -----
    An appropriate amount of overlap will depend on the choice of window
    and on your requirements. In contrast to welch's method, where the entire
    data stream is averaged over, one may wish to use a smaller overlap (or
    perhaps none at all) when computing a spectrogram, to maintain some
    statistical independence between individual segments.

    References
    ----------
    ...[1] Hansson-Sandsten, M. (2011). Optimal multitaper Wigner spectrum
           estimation of a class of locally stationary processes using Hermite functions.
           EURASIP Journal on Advances in Signal Processing, 2011, 10.

    Examples
    --------
    >>> from scipy.signal import chirp
    >>> from lspopt import lsp
    >>> import matplotlib.pyplot as plt

    Generate a test signal.

    >>> fs = 10e3
    >>> N = 1e5
    >>> amp = 2 * np.sqrt(2)
    >>> noise_power = 0.001 * fs / 2
    >>> time = np.arange(N) / fs
    >>> freq = np.linspace(1e3, 2e3, N)
    >>> x = (amp * chirp(time, 1e3, 2.0, 6e3, method='quadratic') +
    >>>      np.random.normal(scale=np.sqrt(noise_power), size=time.shape))

    Compute and plot the spectrogram.

    >>> f, t, Sxx = lsp.spectrogram_lspopt(x, fs=fs, c_parameter=20.0)
    >>> plt.pcolormesh(t, f, Sxx)
    >>> plt.ylabel('Frequency [Hz]')
    >>> plt.xlabel('Time [sec]')
    >>> plt.show()

    """
    # Less overlap than welch, so samples are more statisically independent
    if noverlap is None:
        noverlap = nperseg // 8

    H, taper_weights = lspopt(n=nperseg, c_parameter=c_parameter)
    S_out = None
    # Call spectrogram method for each taper window.
    for taper_window, taper_weight in zip(H, taper_weights):
        f, t, Pxx = spectrogram(
            x,
            fs=fs,
            window=taper_window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=axis,
        )
        if S_out is None:
            S_out = taper_weight * Pxx
        else:
            S_out += taper_weight * Pxx
    return f, t, S_out
