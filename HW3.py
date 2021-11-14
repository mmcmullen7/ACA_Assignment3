# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:17:54 2021

@author: mbate
"""

import numpy as np
import math
import scipy as sp
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")



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
        maxNdx[n] = np.argmax(block)#np.argwhere(block == maxMag[n])

    # Convert mask from float to integer
    #maxNdx = np.int_(maxNdx)
    # Apply mask to frequency vector to determine f0 (frequency of occurence of the maximum magnitude)
    #f0 = fInHz[maxNdx]

    nyquist = fs/2
    f0 = nyquist * maxNdx / (X.shape[0] - 1)
    
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
        freq = nyquist * freq_ind / ((X.shape[0]) - 1)
        f0[i] = freq

    return f0

def track_pitch_hps(x, blockSize, hopSize, fs):
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    X, fInHz = compute_spectrogram(xb, fs)
    f0 = get_f0_from_Hps(X, fs, order=4)

    return f0, timeInSec

def comp_acf(inputVector, blsNormalized=True):
    inputVector = np.concatenate([np.zeros(len(inputVector)), inputVector])
    freq = np.fft.fft(inputVector)
    r_freq = freq * np.conjugate(freq)
    r = np.fft.ifft(r_freq).real
    r = r[:len(r) // 2]

    if blsNormalized:
        r = r / np.max(np.abs(r))

    return r


def get_f0_from_acf(r, fs):
    peaks, _ = find_peaks(r)
    proms = peak_prominences(r, peaks)[0]
    max_prom = np.max(proms)
    ind = np.where(proms==max_prom)
    T = peaks[ind] / fs
    f0 = 1 / T
    return f0


def track_pitch_acf(x, blockSize, hopSize, fs):
    xb, timeInSec = block_audio(x, blockSize, hopSize, fs)
    f0 = np.zeros(len(xb))
    for i, block in enumerate(xb):
        r = comp_acf(block)
        f0[i] = get_f0_from_acf(r, fs)
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
    mask = rmsDb <= thresholdDb
    thresh_line = np.ones(rmsDb.shape[0]) * thresholdDb

    fig, axs = plt.subplots(2)
    axs[0].plot(rmsDb)
    axs[0].plot(thresh_line)
    axs[0].set_title("RMS")
    axs[1].plot(mask)
    axs[1].set_title(("Mask with threshold:" + str(thresholdDb)))
    plt.show()

    #mask = np.argwhere(rmsDb <= thresholdDb)
    
    return mask

def apply_voicing_mask(f0, mask):
    
    f0Adj = np.multiply(f0, mask)

    #print("mask shape:", mask.shape)
    #print("f0Adj shape:", f0Adj.shape)

    #f0Adj[mask] = 0
    
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


def eval_pitchtrack_v2(estimation, annotation):
    cents = []
    for est, true in zip(estimation, annotation):
        if true > 0 and est > 0:
            cents.append(1200 * np.log2(est / true))


    errCentRms = np.sqrt(np.mean(np.power(np.array(cents), 2)))
    pfp = eval_voiced_fp(estimation, annotation)
    pfn = eval_voiced_fn(estimation, annotation)

    return errCentRms, pfp, pfn  

def executeassign3():
    fs = 44100
    blockSize = 1024
    hopSize = 512

    t = np.arange(0, fs) / fs
    signal_1 = np.sin(2 * np.pi * 441 * t)
    signal_2 = np.sin(2 * np.pi * 882 * t)
    test_signal = np.concatenate([signal_1, signal_2])
    f0, timeInSec = track_pitch_fftmax(test_signal, blockSize, hopSize, fs)
    ref = np.zeros(len(f0))
    ref[:len(f0)//2 - 1] = 441
    ref[len(f0)//2 - 1 :] = 882
    error = ref - f0

    fig, axs = plt.subplots(2)
    axs[0].plot(timeInSec, f0)
    axs[0].plot(timeInSec, ref)
    axs[0].set_title("Pitch from FFT Max")
    axs[0].set(xlabel=("Time in seconds"), ylabel=("Frequency in Hz"))
    axs[1].plot(timeInSec, error)
    axs[1].set_title("Error in estimated pitch")
    axs[1].set(xlabel=("Time in seconds"), ylabel=("Error in Hz"))
    plt.show()

    f0, timeInSec = track_pitch_hps(test_signal, blockSize, hopSize, fs)
    error = ref - f0
    fig, axs = plt.subplots(2)
    axs[0].plot(timeInSec, f0)
    axs[0].plot(timeInSec, ref)
    axs[0].set_title("Pitch from HPS")
    axs[0].set(xlabel=("Time in seconds"), ylabel=("Frequency in Hz"))
    axs[1].plot(timeInSec, error)
    axs[1].set_title("Error in estimated pitch")
    axs[1].set(xlabel=("Time in seconds"), ylabel=("Error in Hz"))
    plt.show()

    blockSize = 2048
    hopSize = 512
    f0, timeInSec = track_pitch_fftmax(test_signal, blockSize, hopSize, fs)
    error = ref - f0

    fig, axs = plt.subplots(2)
    axs[0].plot(timeInSec, f0)
    axs[0].plot(timeInSec, ref)
    axs[0].set_title("Pitch from FFT Max with increased block size")
    axs[0].set(xlabel=("Time in seconds"), ylabel=("Frequency in Hz"))
    axs[1].plot(timeInSec, error)
    axs[1].set_title("Error in estimated pitch")
    axs[1].set(xlabel=("Time in seconds"), ylabel=("Error in Hz"))
    plt.show()


def run_evaluation(complete_path_to_data_folder):
    wav_paths = []
    txt_paths = []
    for f_name in os.listdir(complete_path_to_data_folder):
        if f_name.endswith(".wav"):
            wav_paths.append(os.path.join(complete_path_to_data_folder, f_name))
        elif f_name.endswith(".txt"):
            txt_paths.append(os.path.join(complete_path_to_data_folder, f_name))

    wav_paths.sort()
    txt_paths.sort()

    for wav, txt in zip(wav_paths, txt_paths):
        fs, y = wavfile.read(wav)
        txt_arr = np.loadtxt(txt)
        onsets = txt_arr[:, 0]
        f_true = txt_arr[:, 2]
        f_est, timeInSec = track_pitch_acf(y, 1024, 512, fs)
        rms = eval_pitchtrack(f_est, f_true)
        print("RMS:", rms)

def e3():
    complete_path_to_data_folder = "developmentSet/trainData/"
    wav_paths = []
    txt_paths = []
    for f_name in os.listdir(complete_path_to_data_folder):
        if f_name.endswith(".wav"):
            wav_paths.append(os.path.join(complete_path_to_data_folder, f_name))
        elif f_name.endswith(".txt"):
            txt_paths.append(os.path.join(complete_path_to_data_folder, f_name))

    wav_paths.sort()
    txt_paths.sort()

    for wav, txt in zip(wav_paths, txt_paths):
        fs, y = wavfile.read(wav)
        txt_arr = np.loadtxt(txt)
        onsets = txt_arr[:, 0]
        f_true = txt_arr[:, 2]
        f_est, timeInSec = track_pitch_fftmax(y, 1024, 512, fs)

        rms, pfp, pfn = eval_pitchtrack_v2(f_est, f_true)
        print("RMS:", rms)
        print("PFP:", pfp)
        print("PFN:", pfn)


def e4():
    complete_path_to_data_folder = "developmentSet/trainData/"
    wav_paths = []
    txt_paths = []
    for f_name in os.listdir(complete_path_to_data_folder):
        if f_name.endswith(".wav"):
            wav_paths.append(os.path.join(complete_path_to_data_folder, f_name))
        elif f_name.endswith(".txt"):
            txt_paths.append(os.path.join(complete_path_to_data_folder, f_name))

    wav_paths.sort()
    txt_paths.sort()

    for wav, txt in zip(wav_paths, txt_paths):
        fs, y = wavfile.read(wav)
        txt_arr = np.loadtxt(txt)
        onsets = txt_arr[:, 0]
        f_true = txt_arr[:, 2]
        f_est, timeInSec = track_pitch_hps(y, 1024, 512, fs)

        rms, pfp, pfn = eval_pitchtrack_v2(f_est, f_true)
        print("RMS:", rms)
        print("PFP:", pfp)
        print("PFN:", pfn)

    
def track_pitch(x, blockSize, hopSize, fs, method, voicingThres):

    if method == "acf":
        f0, timeInSec = track_pitch_acf(x, blockSize, hopSize, fs)
    elif method == "max":
        f0, timeInSec = track_pitch_fftmax(x, blockSize, hopSize, fs)
    elif method == "hps":
        f0, timeInSec = track_pitch_hps(x, blockSize, hopSize, fs)
    else:
        print("no method specified")

    xb, _ = block_audio(x, blockSize, hopSize, fs)
    rmsDb = extract_rms(xb)
    mask = create_voicing_mask(rmsDb, voicingThres)
    f0Adj = apply_voicing_mask(f0, mask)

    return f0Adj, timeInSec

def e5():

    complete_path_to_data_folder = "developmentSet/trainData/"
    wav_paths = []
    txt_paths = []
    for f_name in os.listdir(complete_path_to_data_folder):
        if f_name.endswith(".wav"):
            wav_paths.append(os.path.join(complete_path_to_data_folder, f_name))
        elif f_name.endswith(".txt"):
            txt_paths.append(os.path.join(complete_path_to_data_folder, f_name))

    wav_paths.sort()
    txt_paths.sort()

    trackers = ["acf", "max", "hps"]
    thresholds = [-40, -20]

    for tracker in trackers:
        for threshold in thresholds:
            for wav, txt in zip(wav_paths, txt_paths):
                print("Tracking pitch for", wav, "using", tracker, "and threshold of", threshold)
                fs, y = wavfile.read(wav)
                txt_arr = np.loadtxt(txt)
                onsets = txt_arr[:, 0]
                f_true = txt_arr[:, 2]
                f_est, timeInSec = track_pitch(y, 1024, 512, fs, tracker, threshold)
                plt.plot(f_true, label="true")
                plt.plot(f_est, label="est")
                plt.show()
                rms, pfp, pfn = eval_pitchtrack_v2(f_est, f_true)
                print("RMS:", rms)
                print("PFP:", pfp)
                print("PFN:", pfn)
