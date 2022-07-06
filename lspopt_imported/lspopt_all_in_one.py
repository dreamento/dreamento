#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-13

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

# from pkg_resources import resource_filename

import numpy as np

__all__ = ["C", "WEIGHTS", "f_h"]

C = np.array([ 1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2. ,
        2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3. ,  3.1,
        3.2,  3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4. ,  4.1,  4.2,
        4.3,  4.4,  4.5,  4.6,  4.7,  4.8,  4.9,  5. ,  5.1,  5.2,  5.3,
        5.4,  5.5,  5.6,  5.7,  5.8,  5.9,  6. ,  6.1,  6.2,  6.3,  6.4,
        6.5,  6.6,  6.7,  6.8,  6.9,  7. ,  7.1,  7.2,  7.3,  7.4,  7.5,
        7.6,  7.7,  7.8,  7.9,  8. ,  8.1,  8.2,  8.3,  8.4,  8.5,  8.6,
        8.7,  8.8,  8.9,  9. ,  9.1,  9.2,  9.3,  9.4,  9.5,  9.6,  9.7,
        9.8,  9.9, 10. , 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8,
       10.9, 11. , 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9,
       12. , 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13. ,
       13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8, 13.9, 14. , 14.1,
       14.2, 14.3, 14.4, 14.5, 14.6, 14.7, 14.8, 14.9, 15. , 15.1, 15.2,
       15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 16. , 16.1, 16.2, 16.3,
       16.4, 16.5, 16.6, 16.7, 16.8, 16.9, 17. , 17.1, 17.2, 17.3, 17.4,
       17.5, 17.6, 17.7, 17.8, 17.9, 18. , 18.1, 18.2, 18.3, 18.4, 18.5,
       18.6, 18.7, 18.8, 18.9, 19. , 19.1, 19.2, 19.3, 19.4, 19.5, 19.6,
       19.7, 19.8, 19.9, 20. , 20.1, 20.2, 20.3, 20.4, 20.5, 20.6, 20.7,
       20.8, 20.9, 21. , 21.1, 21.2, 21.3, 21.4, 21.5, 21.6, 21.7, 21.8,
       21.9, 22. , 22.1, 22.2, 22.3, 22.4, 22.5, 22.6, 22.7, 22.8, 22.9,
       23. , 23.1, 23.2, 23.3, 23.4, 23.5, 23.6, 23.7, 23.8, 23.9, 24. ,
       24.1, 24.2, 24.3, 24.4, 24.5, 24.6, 24.7, 24.8, 24.9, 25. , 25.1,
       25.2, 25.3, 25.4, 25.5, 25.6, 25.7, 25.8, 25.9, 26. , 26.1, 26.2,
       26.3, 26.4, 26.5, 26.6, 26.7, 26.8, 26.9, 27. , 27.1, 27.2, 27.3,
       27.4, 27.5, 27.6, 27.7, 27.8, 27.9, 28. , 28.1, 28.2, 28.3, 28.4,
       28.5, 28.6, 28.7, 28.8, 28.9, 29. , 29.1, 29.2, 29.3, 29.4, 29.5,
       29.6, 29.7, 29.8, 29.9, 30. , 30.1, 30.2, 30.3, 30.4, 30.5, 30.6,
       30.7, 30.8, 30.9, 31. , 31.1, 31.2, 31.3, 31.4, 31.5, 31.6, 31.7,
       31.8, 31.9, 32. , 32.1, 32.2, 32.3, 32.4, 32.5, 32.6, 32.7, 32.8,
       32.9, 33. , 33.1, 33.2, 33.3, 33.4, 33.5, 33.6, 33.7, 33.8, 33.9,
       34. , 34.1, 34.2, 34.3, 34.4, 34.5, 34.6, 34.7, 34.8, 34.9, 35. ,
       35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 35.9, 36. , 36.1,
       36.2, 36.3, 36.4, 36.5, 36.6, 36.7, 36.8, 36.9, 37. , 37.1, 37.2,
       37.3, 37.4, 37.5, 37.6, 37.7, 37.8, 37.9, 38. , 38.1, 38.2, 38.3,
       38.4, 38.5, 38.6, 38.7, 38.8, 38.9, 39. , 39.1, 39.2, 39.3, 39.4,
       39.5, 39.6, 39.7, 39.8, 39.9, 40. , 40.1, 40.2, 40.3, 40.4, 40.5,
       40.6, 40.7, 40.8, 40.9, 41. , 41.1, 41.2, 41.3, 41.4, 41.5, 41.6,
       41.7, 41.8, 41.9, 42. , 42.1, 42.2, 42.3, 42.4, 42.5, 42.6, 42.7,
       42.8, 42.9, 43. , 43.1, 43.2, 43.3, 43.4, 43.5, 43.6, 43.7, 43.8,
       43.9, 44. , 44.1, 44.2, 44.3, 44.4, 44.5, 44.6, 44.7, 44.8, 44.9,
       45. , 45.1, 45.2, 45.3, 45.4, 45.5, 45.6, 45.7, 45.8, 45.9, 46. ,
       46.1, 46.2, 46.3, 46.4, 46.5, 46.6, 46.7, 46.8, 46.9, 47. , 47.1,
       47.2, 47.3, 47.4, 47.5, 47.6, 47.7, 47.8, 47.9, 48. , 48.1, 48.2,
       48.3, 48.4, 48.5, 48.6, 48.7, 48.8, 48.9, 49. , 49.1, 49.2, 49.3,
       49.4, 49.5, 49.6, 49.7, 49.8, 49.9, 50. ])



# An array of C parameter values for which weights have been pre-calculated.
# C = np.load(resource_filename("lspopt.data", "c.npy")).flatten()
# The pre-calculated Hermite polynomial coefficients
# for the C parameter values above.
# WEIGHTS = np.load(resource_filename("lspopt.data", "weights.npy"))
WEIGHTS = np.load("weights.npy")


def f_h(n, k):
    """Returns f_h value.

    :param n: Window length of multitaper windows.
    :type n: int
    :param k: Length of non-zero Hermite polynomial coefficient array.
    :type k: int
    :return: The f_h value.
    :rtype: float

    """
    return n / _K_TO_VALUE_.get(k)


# Given length of Hermite polynomial coefficient array, return
# a value to divide N with.
_K_TO_VALUE_ = {
    1: 5.4,
    2: 6.0,
    3: 7.3,
    4: 8.1,
    5: 8.7,
    6: 9.3,
    7: 9.8,
    8: 10.3,
    9: 10.9,
    10: 11.2,
}

import numpy as np
from scipy.linalg import sqrtm


def create_lsp_realisation(N, c, Fs, random_seed=None):
    """Create a realisation of a locally stationary process (LSP).

    For details, see section 2 in [1].

    Parameters
    ----------
    N : int
        Length of the process to generate.
    c: float
        Measure of stationarity of the process to generate.
    Fs: float
        Frequency.
    random_seed: int
        Random seed to apply before generating process.

    Returns
    -------
    x : ndarray
        The process realisation.
    H : ndarray
        The process-generating matrix H.
    Rx : ndarray
        The covariance matrix R_x(t, s).

    References
    ----------
    ...[1] Hansson-Sandsten, M. (2011). Optimal multitaper Wigner spectrum
           estimation of a class of locally stationary processes using Hermite functions.
           EURASIP Journal on Advances in Signal Processing, 2011, 10.

    """
    r_x = lambda a, b: _q((a + b) / 2) * _r(a - b, c)
    return _create_realisation(r_x, N, Fs, random_seed)


def create_lscp_realisation(N, c, Fs, m, d, random_seed=None):
    """Create a realisation of a locally stationary chirp process.

    For details, see section 2 in [1].

    Parameters
    ----------
    N : int
        Length of the process to generate
    c: float
        Measure of stationarity of the process to generate.
    m: float
        Chirp frequency.
    d: float
        Start of the chirp frequency.
    random_seed: int
        Random seed to apply before generating process.

    Returns
    -------
    x : ndarray
        The process realisation.
    H : ndarray
        The process-generating matrix H.
    Rx : ndarray
        The covariance matrix R_x(t, s).

    References
    ----------
    ...[1] Hansson-Sandsten, M. (2011). Optimal multitaper Wigner spectrum
           estimation of a class of locally stationary processes using Hermite functions.
           EURASIP Journal on Advances in Signal Processing, 2011, 10.

    """
    r_x = (
        lambda a, b: _q((a + b) / 2)
        * _r(a - b, c)
        * np.exp(1j * m * (a - b) * ((a + b) / 2 - d))
    )
    return _create_realisation(r_x, N, Fs, random_seed)


def create_mlsp_realisation(N, c_vector, Fs, random_seed=None):
    """Create a realisation of a Multicomponent locally stationary process.

    For details, see section 2 in [1].

    Parameters
    ----------
    N : int
        Length of the process to generate
    c_vector: list, tuple
        A list of stationarity measure values of the process to generate.
    Fs: float
        Frequency.
    random_seed: int
        Random seed to apply before generating process.

    Returns
    -------
    x : ndarray
        The process realisation.
    H : ndarray
        The process-generating matrix H.
    Rx : ndarray
        The covariance matrix R_x(t, s).

    References
    ----------
    ...[1] Hansson-Sandsten, M. (2011). Optimal multitaper Wigner spectrum
           estimation of a class of locally stationary processes using Hermite functions.
           EURASIP Journal on Advances in Signal Processing, 2011, 10.

    """
    r_x = lambda a, b: np.sum([_q((a + b) / 2) * _r(a - b, c) for c in c_vector])
    return _create_realisation(r_x, N, Fs, random_seed)


def _create_realisation(r_x, N, Fs, random_seed):
    # Sampling vector.
    t_vector = np.arange(-(N / 2), N / 2, 1) / Fs

    # Create real or complex matrix depending on covariance function output.
    if np.iscomplex(r_x(t_vector[0], t_vector[-1])):
        Rx = np.zeros((N, N), "complex")
    else:
        Rx = np.zeros((N, N), "float")

    # Calculate covariance matrix.
    for i, t in enumerate(t_vector):
        for j, s in enumerate(t_vector):
            if j < i:
                continue
            Rx[i, j] = r_x(t, s)
            Rx[j, i] = Rx[i, j]
    # Apply matrix square root to Rx to find H from (5) in [1].
    # Since Rx is symmetric, its square root will also be symmetric.
    H = sqrtm(Rx)

    # Use only the real part of the matrix if the imaginary side is small.
    # The covariance matrix is symmetric and positive semi-definite and
    # as such it will have a principal square root, i.e. a strictly real
    # square root matrix.
    if (np.linalg.norm(np.imag(H)) / np.linalg.norm(H)) < 1e-6:
        H = np.real(H)

    # Generate Gaussian zero mean stochastic process.
    np.random.seed(random_seed)
    x = np.random.normal(size=(N,))

    # Create process realisation and return all relevant data.
    return H.dot(x), H, Rx


def _q(tau):
    return np.exp(-(tau ** 2) / 2)


def _r(tau, c):
    return np.exp(-((c / 4) * (tau ** 2)) / 2)


import six
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
