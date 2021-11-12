# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:17:54 2021

@author: mbate
"""

import numpy as np
import math
import scipy as sp
from scipy.fft import fft
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os



def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)

def compute_spectrogram(xb, fs):
    numBlocks = xb.shape[0]
    afWindow = 0.5 - (0.5 * np.cos(2 * np.pi / xb.shape[1] * np.arange(xb.shape[1])))
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])
    f_min = 0
    f_max = fs/2
    f = np.linspace(f_min, f_max, xb.shape[0]+2)
    fInHz = f[1:xb.shape[0]+1]
    
    for n in range(0, numBlocks):
        # apply window
        tmp = abs(fft(xb[n,:] * afWindow))*2/xb.shape[1]
    
        # compute magnitude spectrum
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))] 
        # normalize
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) 

    return X, fInHz

def track_pitch_fftmax(x, blockSize, hopSize, fs):
    # Block Audio Input
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    # Compute Spectrogram of Audio
    X, fInHz = compute_spectrogram(xb, fs)
    
    # Identify the maximum magnitudes within each block
    maxMag = X.max(0)
    
    # Initialize masking array
    maxNdx = np.zeros(maxMag.shape[0])
    
    # Iterate through each block to find the index at which maximum occurs, fill in mask
    for n in range(0, X.shape[1]):
        block = X[:, n]
        maxNdx[n] = np.argwhere(block == maxMag[n])

    # Convert mask from float to integer
    maxNdx = np.int_(maxNdx)
    # Apply mask to frequency vector to determine f0 (frequency of occurence of the maximum magnitude)
    f0 = fInHz[maxNdx]
    
    return f0, timeInSec


def get_f0_from_Hps(X, fs, order):
    nyquist = fs / 2
    f0 = np.zeros(X.shape[1])
    for i, block in enumerate(X.T):
        hps = block
        for j in range(2, order+1):
            down_sampled = block[::j]
            hps[:len(down_sampled)] *= down_sampled

        freq_ind = np.argmax(hps)
        freq = nyquist * freq_ind / (X.shape[0]) - 1
        f0[i] = freq

    return f0

def track_pitch_hps(x, blockSize, hopSize, fs):
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    X, fInHz = compute_spectrogram(xb, fs)
    f0 = get_f0_from_Hps(X, fs, order=4)

    return f0, timeInSec


def extract_rms(xb):
    # number of results
    numBlocks = xb.shape[0]
    # allocate memory
    vrms = np.zeros(numBlocks)
    for n in range(0, numBlocks):
        # calculate the rms
        vrms[n] = np.sqrt(np.dot(xb[n,:], xb[n,:]) / xb.shape[1])
    # convert to dB
    epsilon = 1e-5  # -100dB
    vrms[vrms < epsilon] = epsilon
    rmsDb = 20 * np.log10(1/vrms)
    
    return (rmsDb)    

def create_voicing_mask(rmsDb, thresholdDb):
    
    mask = np.argwhere(rmsDb <= thresholdDb)
    
    return mask

def apply_voicing_mask(f0, mask):
    
    f0Adj = f0
    f0Adj[mask] = 0
    
    return f0Adj

def eval_voiced_fp(estimation, annotation):
    
    denomenator_mask = np.argwhere(annotation == 0)
    denomenator = len(denomenator_mask)
    numerator = estimation[denomenator_mask]
    numerator = len(np.argwhere(numerator != 0))
    
    pfp = (numerator / denomenator) * 100
    
    return pfp

def eval_voiced_fn(estimation, annotation):
    
    denomenator_mask = np.argwhere(annotation != 0)
    denomenator = len(denomenator_mask)
    numerator = estimation[denomenator_mask]
    numerator = len(np.argwhere(numerator == 0))
    
    pfn = (numerator / denomenator) * 100
    
    return pfn


    