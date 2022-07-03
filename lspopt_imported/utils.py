#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`utils`
==================

.. module:: utils
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-15, 22:47

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

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
