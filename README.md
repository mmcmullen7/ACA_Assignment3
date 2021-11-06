# ACA_Assignment3

## Revisited: Fundamental Frequency Detection/Pitch Tracking

We had looked at a time-domain based monophonic pitch tracker in Assignment 1. In this assignment, we investigate frequency-domain approaches to monophonic pitch tracking.  We also look at other performance evaluation metrics and implement a very basic RMS based voicing detection technique.

## General Instructions:

use the provided function declarations and/or function headers. Submitting modified function headers will result in point deductions.
ensure input and output dimensions of your functions are accurate
all vectors and matrices returned have to be np arrays, no lists etc.
no thirdparty modules except numpy, scipy, and matplotlib are allowed, plus either os or glob
when asked for discussion and plot, submit them with the corresponding question number in one pdf
all plot axes must be labeled and the plots easily understandable
submit all your functions defined in one file: ass3solution.py
DO NOT change file names or function names as that will break the test scripts and will result in point deductions
DO NOT plagiarize. Any similarity with publicly available code or between different submissions will result in 0 points for the complete assignment and, if significant, might be reported.
when asked to use function from previous assignments, always use the provided references, not your implementations.
 

 

## A. Maximum spectral peak based pitch tracker
[5 points] Implement a function [X, fInHz] = compute_spectrogram(xb, fs) that computes the magnitude spectrum for each block of audio in xb (calculated using the reference block_audio() from previous assignments) and returns the magnitude spectrogram X (dimensions blockSize/2+1 X numBlocks) and a frequency vector fInHz (dim blockSize/2+1,) containing the central frequency of each bin. Do not use any third party spectrogram function. Note: remove the redundant part of the spectrum. Also note that you will have to apply a von-Hann window of appropriate length to the blocks before computing the fft.
[10 points] Implement a function: [f0, timeInSec] = track_pitch_fftmax(x, blockSize, hopSize, fs) that estimates the fundamental frequency f0 of the audio signal based on a block-wise maximum spectral peak finding approach. Note: This function should use compute_spectrogram().
[5 points] If the blockSize = 1024 for blocking, what is the exact time resolution of your pitch tracker? Can this be improved without changing the block-size? If yes, how? If no, why? (Use a sampling rate of 44100Hz for all calculations).
 
## B. HPS (Harmonic Product Spectrum) based pitch tracker
[15 points] Implement a function [f0] = get_f0_from_Hps(X, fs, order) that computes the block-wise fundamental frequency f0 given the magnitude spectrogram X and the samping rate based on a HPS approach of order order.
[5 points] Implement a function [f0, timeInSec] = track_pitch_hps(x, blockSize, hopSize, fs) that estimates the fundamental frequency f0 of the audio signal based on  HPS based approach. Use blockSize = 1024 in compute_spectrogram(). Use order = 4 for get_f0_from_Hps()
 
## C. Voicing Detection
[1 points] Take the  function [rmsDb] = extract_rms(xb) from the second assignment which takes the blocked audio as input and computes the RMS (Root Mean Square) amplitude of each block.
[6 points] Implement a function [mask] = create_voicing_mask(rmsDb, thresholdDb) which takes a vector of decibel values for the different blocks of audio and creates a binary mask based on the threshold parameter. Note: A binary mask in this case is a simple column vector of the same size as 'rmsDb' containing 0's and 1's only. The value of the mask at an index is 0 if the rmsDb value at that index is less than 'thresholdDb' and the value is 1 if 'rmsDb' value at that index is greater than or equal to the threshold. 
[6 points] Implement a function [f0Adj] = apply_voicing_mask(f0, mask)  which applies the voicing mask to the previously computed f0 so that the f0 of blocks with low energy is set to 0.

## D. Different evaluation metrics
[5 points] Implement a function [pfp] = eval_voiced_fp(estimation, annotation) that computes the percentage of false positives for your fundamental frequency estimation
False Positive : The denominator would be the number of blocks for which annotation = 0. The numerator would be how many of these blocks were classified as voiced (with a fundamental frequency not equal to 0) is your estimation. 
[5 points] Implement a function [pfn] = eval_voiced_fn(estimation, annotation) that computes the percentage of false negatives for your fundamental frequency estimation
False Negative: In this case the denominator would be number of blocks which have non-zero fundamental frequency in the annotation. The numerator would be number of blocks out of these that were detected as zero is the estimation.
[5 points] Now modify the eval_pitchtrack() method that you wrote in Assignment 1 to [errCentRms, pfp, pfn] = eval_pitchtrack_v2(estimation, annotation), to return all the 3 performance metrics for your fundamental frequency estimation.  Note: the errorCentRms computation might need to slightly change now considering that your estimation might also contain zeros.

## E. Evaluation
[5 points] In a separate function executeassign3(), generate a test signal (sine wave, f = 441 Hz from 0-1 sec and f = 882 Hz from 1-2 sec), apply your track_pitch_fftmax(), (blockSize = 1024, hopSize = 512) and plot the f0 curve. Also, plot the absolute error per block and discuss the possible causes for the deviation. Repeat for track_pitch_hps() with the same signal and parameters. Why does the HPS method fail with this signal?
[5 points] Next use (blockSize = 2048, hopSize = 512) and repeat the above experiment (only for the max spectra method). Do you see any improvement in performance? 
[5 points] Evaluate your track_pitch_fftmax() using the development set  Download development set(see assignment 1) and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512). Report the average performance metrics across the development set.
[5 points] Evaluate your track_pitch_hps() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512). Report the average performance metrics across the development set. 
[6 points] Implement a MATLAB wrapper function [f0Adj, timeInSec] = track_pitch(x, blockSize, hopSize, fs, method, voicingThres) that takes audio signal ‘x’ and related paramters (fs, blockSize, hopSize), calls the appropriate pitch tracker based on the method parameter (‘acf’,‘max’, ‘hps’) to compute the fundamental frequency and then applies the voicing mask based on the threshold parameter. 
[6 points] Evaluate your track_pitch() using the development set and the eval_pitchtrack_v2() method (use blockSize = 1024, hopSize = 512) over all 3 pitch trackers (acf, max and hps) and report the results with two values of threshold (threshold = -40, -20)
 

## Bonus: Improving your Pitch Trackers
[10 points, capped at max] Implement a function: [f0, timeInSec] = track_pitch_mod(x, blockSize, hopSize, fs) that combines ideas from different pitch trackers you have tried thus far and thereby provides better f0 estimations. You may include voicing detection within this method with parameters of your choosing. Please explain your approach in the report. Your function will be tested using a testing set (not provided) with a block size of 1024 and a hopsize of 512, and points will be given based on its performance compared to the other groups. Best performing group gets 10 points and worst performing group gets 1 point. 
