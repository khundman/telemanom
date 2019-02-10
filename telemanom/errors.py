import numpy as np
import math
import pandas as pd 
from itertools import groupby
from operator import itemgetter
import more_itertools as mit
from elasticsearch import Elasticsearch
import time
import json
import sys
import os
import math
from scipy.stats import norm

from telemanom._globals import Config

config = Config("config.yaml")

def get_errors(y_test, y_hat, anom, smoothed=True):
    """Calculate the difference between predicted telemetry values and actual values, then smooth residuals using
    ewma to encourage identification of sustained errors/anomalies.

    Args:
        y_test (np array): array of test targets corresponding to true values to be predicted at end of each sequence
        y_hat (np array): predicted test values for each timestep in y_test  
        anom (dict): contains anomaly information for a given input stream
        smoothed (bool): If False, return unsmooothed errors (used for assessing quality of predictions)


    Returns:
        e (list): unsmoothed errors (residuals)
        e_s (list): smoothed errors (residuals)
    """

    e = [abs(y_h-y_t[0]) for y_h,y_t in zip(y_hat, y_test)]

    if not smoothed:
        return e

    smoothing_window = int(config.batch_size * config.window_size * config.smoothing_perc)
    if not len(y_hat) == len(y_test):
        raise ValueError("len(y_hat) != len(y_test), can't calculate error: %s (y_hat) , %s (y_test)" %(len(y_hat), len(y_test)))
    
    e_s = list(pd.DataFrame(e).ewm(span=smoothing_window).mean().values.flatten())

    # for values at beginning < sequence length, just use avg
    if not anom['chan_id'] == 'C-2': #anom occurs early in window (limited data available for channel)
        e_s[:config.l_s] = [np.mean(e_s[:config.l_s*2])]*config.l_s 

    np.save(os.path.join("data", anom['run_id'], "smoothed_errors", anom["chan_id"] + ".npy"), np.array(e_s))

    return e_s



def process_errors(y_test, y_hat, e_s, anom, logger):
    '''Using windows of historical errors (h = batch size * window size), calculate the anomaly 
    threshold (epsilon) and group any anomalous error values into continuos sequences. Calculate 
    score for each sequence using the max distance from epsilon.

    Args:
        y_test (np array): test targets corresponding to true telemetry values at each timestep t
        y_hat (np array): test target predictions at each timestep t
        e_s (list): smoothed errors (residuals) between y_test and y_hat
        anom (dict): contains anomaly information for a given input stream
        logger (obj): logging object

    Returns:
        E_seq (list of tuples): Start and end indices for each anomaloues sequence
        anom_scores (list): Score for each anomalous sequence
    '''
   
    i_anom = [] # anomaly indices
    window_size = config.window_size

    num_windows = int((y_test.shape[0] - (config.batch_size*window_size)) / config.batch_size)

    # decrease the historical error window size (h) if number of test values is limited
    while num_windows < 0:
        window_size -= 1
        if window_size <= 0:
            window_size = 1
        num_windows = int((y_test.shape[0] - (config.batch_size*window_size)) / config.batch_size)
        if window_size == 1 and num_windows < 0:
            raise ValueError("Batch_size (%s) larger than y_test (len=%s). Adjust in config.yaml." %(config.batch_size, y_test.shape[0]))
        
    # Identify anomalies for each new batch of values
    for i in range(1, num_windows+2):
        prior_idx = (i-1) * (config.batch_size)
        idx = (config.window_size*config.batch_size) + ((i-1) * config.batch_size)
        
        if i == num_windows+1:
            idx = y_test.shape[0]

        window_e_s = e_s[prior_idx:idx]
        window_y_test = y_test[prior_idx:idx]

        epsilon = find_epsilon(window_e_s, config.error_buffer)
        window_anom_indices = get_anomalies(window_e_s, window_y_test, epsilon, i-1, i_anom, len(y_test))

        # update indices to reflect true indices in full set of values (not just window)
        i_anom.extend([i_a + (i-1)*config.batch_size for i_a in window_anom_indices])

    # group anomalous indices into continuous sequences
    i_anom = sorted(list(set(i_anom)))
    groups = [list(group) for group in mit.consecutive_groups(i_anom)]
    E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

    # calc anomaly scores based on max distance from epsilon for each sequence
    anom_scores = []
    for e_seq in E_seq:
        score = max([abs(e_s[x] - epsilon) / (np.mean(e_s) + np.std(e_s)) for x in range(e_seq[0], e_seq[1])])
        anom_scores.append(score)

    return E_seq, anom_scores




def find_epsilon(e_s, error_buffer, sd_lim=12.0):
    '''Find the anomaly threshold that maximizes function representing tradeoff between a) number of anomalies
    and anomalous ranges and b) the reduction in mean and st dev if anomalous points are removed from errors
    (see https://arxiv.org/pdf/1802.04431.pdf)

    Args:
        e_s (array): residuals between y_test and y_hat values (smoothes using ewma)
        error_buffer (int): if an anomaly is detected at a point, this is the number of surrounding values 
            to add the anomalous range. this promotes grouping of nearby sequences and more intuitive results
        sd_lim (float): The max number of standard deviations above the mean to calculate as part of the 
            argmax function

    Returns:
        sd_threshold (float): the calculated anomaly threshold in number of standard deviations above the mean
    '''
    
    mean = np.mean(e_s)
    sd = np.std(e_s)
    
    max_s = 0
    sd_threshold = sd_lim # default if no winner or too many anomalous ranges
    
    for z in np.arange(2.5, sd_lim, 0.5):
        epsilon = mean + (sd*z)
        pruned_e_s, pruned_i, i_anom  = [], [], []

        for i,e in enumerate(e_s):
            if e < epsilon:
                pruned_e_s.append(e)
                pruned_i.append(i)
            if e > epsilon:
                for j in range(0, error_buffer):
                    if not i + j in i_anom and not i + j >= len(e_s):
                        i_anom.append(i + j)
                    if not i - j in i_anom and not i - j < 0:
                        i_anom.append(i - j)
    
        if len(i_anom) > 0:
            # preliminarily group anomalous indices into continuous sequences (# sequences needed for scoring)
            i_anom = sorted(list(set(i_anom)))
            groups = [list(group) for group in mit.consecutive_groups(i_anom)]
            E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]
                
            perc_removed = 1.0 - (float(len(pruned_e_s)) / float(len(e_s)))
            mean_perc_decrease = (mean - np.mean(pruned_e_s)) / mean
            sd_perc_decrease = (sd - np.std(pruned_e_s)) / sd
            s = (mean_perc_decrease + sd_perc_decrease) / (len(E_seq)**2 + len(i_anom))

            # sanity checks
            if s >= max_s and len(E_seq) <= 5 and len(i_anom) < (len(e_s)*0.5):
                sd_threshold = z
                max_s = s
    
    return sd_threshold #multiply by sd to get epsilon



def compare_to_epsilon(e_s, epsilon, len_y_test, inter_range, chan_std, 
    std, error_buffer, window, i_anom_full):
    '''Compare smoothed error values to epsilon (error threshold) and group consecutive errors together into 
    sequences. 

    Args:
        e_s (list): smoothed errors between y_test and y_hat values
        epsilon (float): Threshold for errors above which an error is considered anomalous
        len_y_test (int): number of timesteps t in test data
        inter_range (tuple of floats): range between 5th and 95 percentile values of error values
        chan_std (float): standard deviation on test values
        std (float): standard deviation of smoothed errors
        error_buffer (int): number of values surrounding anomalous errors to be included in anomalous sequence
        window (int): Count of number of error windows that have been processed
        i_anom_full (list): list of all previously identified anomalies in test set

    Returns:
        E_seq (list of tuples): contains start and end indices of anomalous ranges
        i_anom (list): indices of errors that are part of an anomlous sequnce
        non_anom_max (float): highest smoothed error value below epsilon
    '''

    i_anom = []
    E_seq = []
    non_anom_max = 0

    # Don't consider anything in window because scale of errors too small compared to scale of values   
    if not (std > (.05*chan_std) or max(e_s) > (.05 * inter_range)) or not max(e_s) > 0.05:    
        return E_seq, i_anom, non_anom_max

    # ignore initial error values until enough history for smoothing, prediction, comparisons
    num_to_ignore = config.l_s * 2
    # if y_test is small, ignore fewer
    if len_y_test < 2500: 
        num_to_ignore = config.l_s
    if len_y_test < 1800: 
        num_to_ignore = 0
    
    for x in range(0, len(e_s)):
        
        anom = True
        if not e_s[x] > epsilon or not e_s[x] > 0.05 * inter_range:
            anom = False

        if anom:
            for b in range(0, error_buffer):
                if not x + b in i_anom and not x + b >= len(e_s) and ((x + b) >= len(e_s) - config.batch_size or window == 0):
                    if not (window == 0 and x + b < num_to_ignore):
                        i_anom.append(x + b)
                # only considering new batch of values added to window, not full window
                if not x - b in i_anom and ( (x - b) >= len(e_s) - config.batch_size or window == 0):
                    if not (window == 0 and x - b < num_to_ignore):
                        i_anom.append(x - b)

    # capture max of values below the threshold that weren't previously identified as anomalies 
    # (used in filtering process)
    for x in range(0, len(e_s)):
        adjusted_x = x + window*config.batch_size
        if e_s[x] > non_anom_max and not adjusted_x in i_anom_full and not x in i_anom:
            non_anom_max = e_s[x]

    # group anomalous indices into continuous sequences
    i_anom = sorted(list(set(i_anom)))
    groups = [list(group) for group in mit.consecutive_groups(i_anom)]
    E_seq = [(g[0], g[-1]) for g in groups if not g[0] == g[-1]]

    return E_seq, i_anom, non_anom_max



def prune_anoms(E_seq, e_s, non_anom_max, i_anom):
    '''Remove anomalies that don't meet minimum separation from the next closest anomaly or error value

    Args:
        E_seq (list of lists): contains start and end indices of anomalous ranges
        e_s (list): smoothed errors between y_test and y_hat values
        non_anom_max (float): highest smoothed error value below epsilon
        i_anom (list): indices of errors that are part of an anomlous sequnce

    Returns:
        i_pruned (list): remaining indices of errors that are part of an anomlous sequnces 
            after pruning procedure  
    '''

    E_seq_max, e_s_max = [], []
    for e_seq in E_seq:
        if len(e_s[e_seq[0]:e_seq[1]]) > 0:
            E_seq_max.append(max(e_s[e_seq[0]:e_seq[1]]))
            e_s_max.append(max(e_s[e_seq[0]:e_seq[1]]))
    e_s_max.sort(reverse=True)
    
    if non_anom_max and non_anom_max > 0:
        e_s_max.append(non_anom_max) # for comparing the last actual anomaly to next highest below epsilon

    i_to_remove = []
    p = config.p
    
    for i in range(0,len(e_s_max)):
        if i+1 < len(e_s_max): 
            if (e_s_max[i] - e_s_max[i+1]) / e_s_max[i] < p:
                i_to_remove.append(E_seq_max.index(e_s_max[i]))
                # p += 0.03 # increase minimum separation by this amount for each step further from max error
            else:
                i_to_remove = []
    for idx in sorted(i_to_remove, reverse=True):    
        del E_seq[idx]



    i_pruned = []
    for i in i_anom:
        keep_anomaly_idx = False

        for e_seq in E_seq:
            if i >= e_seq[0] and i <= e_seq[1]:
                keep_anomaly_idx = True

        if keep_anomaly_idx == True:
            i_pruned.append(i)

    return i_pruned


def get_anomalies(e_s, y_test, z, window, i_anom_full, len_y_test):
    '''Find anomalous sequences of smoothed error values that are above error threshold (epsilon). Both
    smoothed errors and the inverse of the smoothed errors are evaluated - large dips in errors often
    also indicate anomlies.

    Args:
        e_s (list): smoothed errors between y_test and y_hat values
        y_test (np array): test targets corresponding to true telemetry values at each timestep for given window
        z (float): number of standard deviations above mean corresponding to epsilon
        window (int): number of error windows that have been evaluated
        i_anom_full (list): list of all previously identified anomalies in test set
        len_y_test (int): num total test values available in dataset

    Returns:
        i_anom (list): indices of errors that are part of an anomlous sequnces 
    '''

    perc_high, perc_low = np.percentile(y_test, [95,5])
    inter_range = perc_high - perc_low

    mean = np.mean(e_s)
    std = np.std(e_s)
    chan_std = np.std(y_test)

    e_s_inv = [mean + (mean-e) for e in e_s] #flip it around the mean
    z_inv = find_epsilon(e_s_inv, config.error_buffer)

    epsilon = mean + (float(z)*std)
    epsilon_inv = mean + (float(z_inv)*std)

    # find sequences of anomalies greater than epsilon
    E_seq, i_anom, non_anom_max = compare_to_epsilon(e_s, epsilon, len_y_test,
        inter_range, chan_std, std, config.error_buffer, window, i_anom_full)

    # find sequences of anomalies using inverted error values (lower than normal errors are also anomalous)
    E_seq_inv, i_anom_inv, inv_non_anom_max = compare_to_epsilon(e_s_inv, epsilon_inv, 
        len_y_test, inter_range, chan_std, std, config.error_buffer, window, i_anom_full)

    if len(E_seq) > 0:
        i_anom = prune_anoms(E_seq, e_s, non_anom_max, i_anom)

    if len(E_seq_inv) > 0:
        i_anom_inv = prune_anoms(E_seq_inv, e_s_inv, inv_non_anom_max, i_anom_inv)

    i_anom = list(set(i_anom + i_anom_inv))

    return i_anom



def evaluate_sequences(E_seq, anom):
    '''Compare identified anomalous sequences with labeled anomalous sequences

    Args:
        E_seq (list of lists): contains start and end indices of anomalous ranges
        anom (dict): contains anomaly information for a given input stream

    Returns:
        anom (dict): with updated anomaly information (whether identified, scores, etc.)
    '''

    anom["false_positives"] = 0
    anom["false_negatives"] = 0
    anom["true_positives"] = 0
    anom["fp_sequences"] = []
    anom["tp_sequences"] = []
    anom["num_anoms"] = len(anom["anomaly_sequences"])   

    E_seq_test = eval(anom["anomaly_sequences"])

    if len(E_seq) > 0:

        matched_E_seq_test = []

        for e_seq in E_seq:

            valid = False

            for i, a in enumerate(E_seq_test):

                if (e_seq[0] >= a[0] and e_seq[0] <= a[1]) or (e_seq[1] >= a[0] and e_seq[1] <= a[1]) or\
                    (e_seq[0] <= a[0] and e_seq[1] >= a[1]) or (a[0] <= e_seq[0] and a[1] >= e_seq[1]):

                    anom["tp_sequences"].append(e_seq)

                    valid = True

                    if i not in matched_E_seq_test:
                        anom["true_positives"] += 1
                        matched_E_seq_test.append(i)
                    
            if valid == False:
                anom["false_positives"] += 1
                anom["fp_sequences"].append([e_seq[0], e_seq[1]])

        anom["false_negatives"] += (len(E_seq_test) - len(matched_E_seq_test))

    else:
        anom["false_negatives"] += len(E_seq_test)

    return anom


