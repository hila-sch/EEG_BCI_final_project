from random import randint
import mne
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import csv
import statistics
from mnelab.io import read_raw

def filter_raw(raw, bandpass=(0.5, 45), notch=(50), notch_width=10):
    '''
    Filter raw data with a bandpass filter and a notch filter.
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw object containing the data.
    bandpass : tuple
        Tuple containing the lower and upper frequencies of the bandpass filter.
            Example: bandpass = (0.5, 45)
    notch : int
        Frequency of the notch filter.
            Example: notch = 50
    notch_width : int
        Width of the notch filter.
            Example: notch_width = 10
            
    Returns:
    --------
    raw : mne.io.Raw
        Raw object containing the filtered data.'''
    #raw.notch_filter(notch, notch_widths=notch_width)
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])
    return raw

def calculate_EI (epoch, bands):
    '''
    Calculate the EI for each epoch and return a list of EI values.

    Parameters:
    -----------
    epoch : mne.epochs.Epochs
            Epochs object containing the data.
    bands : dictionary
            Dictionary containing the bands' names and their limits.
            Example: bands = {'theta': (4,8), 'alpha': (8, 12), 'beta': (12, 30)}

    Returns:
    --------
    ei : list
            List of EI values for each epoch.
    avg_bands : dictionary
            Dictionary containing the average PSD for each band.
            Example: avg_bands = {'theta': [0.1, 0.2, 0.3, ...], 'alpha': [0.1, 0.2, 0.3, ...], 'beta': [0.1, 0.2, 0.3, ...]}
    '''
    '''

    To calculate the EI:
    The function first calculates the PSD for each epoch,
    then averages the PSD over all electrodes and then averages the PSD over the
    frequencies in the band of interest. 
    The EI is then calculated by dividing the average PSD of the beta band by 
    the sum of the average PSD of the theta and alpha bands.'''

    avg_bands = {}
    for band_name, band_limits in bands.items():
        low, high = band_limits
        psds = epoch.compute_psd(method='welch', fmin=low, fmax=high)
        avg_over_electrodes= psds.get_data().mean(1)
        avg_over_band = avg_over_electrodes.mean(1)
        avg_bands[band_name] = avg_over_band.tolist()
        # print(avg_bands[band_name])
    
    sum_lists = np.add(avg_bands['theta'], avg_bands['alpha'])
    ei = np.divide(avg_bands['beta'], sum_lists)
    # ei_mid = signal.medfilt(ei, kernel_size=3)
    return ei, avg_bands['alpha'], avg_bands['beta'], avg_bands['theta']



# def moving_average(x, n_epochs):
#     moving_averages = []
    
#     # Loop through the array to consider
#     # every window of size 3
#     window_size = n_epochs * 2
#     i = n_epochs
#     while i < len(x) - window_size + 1:
        
#         # Store elements from i to i+window_size
#         # in list to get the current window
#         window = x[i : i + window_size]
    
#         # Calculate the average of current window
#         window_average = math.ceil(np.mean(window))
        
#         # Store the average of current
#         # window in moving average list
#         moving_averages.append(window_average)
        
#         # Shift window to right by one position
#         i += n_epochs

#     return moving_averages

# def moving_average_no_double(x, n_epochs):
#     moving_averages = []
    
#     # Loop through the array to consider
#     # every window of size 3
#     window_size = n_epochs
#     i = n_epochs
#     while i < len(x) - window_size + 1:
        
#         # Store elements from i to i+window_size
#         # in list to get the current window
#         window = x[i : i + window_size]
    
#         # Calculate the average of current window
#         window_average = math.ceil(np.mean(window))
        
#         # Store the average of current
#         # window in moving average list
#         moving_averages.append(window_average)
        
#         # Shift window to right by one position
#         i += n_epochs

#     return moving_averages

def ewma(x, alpha):
    '''
    Returns the exponentially weighted moving average of x.

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}

    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    x = np.array(x)
    n = x.size

    # Create an initial weight matrix of (1-alpha), and a matrix of powers
    # to raise the weights by
    w0 = np.ones(shape=(n,n)) * (1-alpha)
    p = np.vstack([np.arange(i,i-n,-1) for i in range(n)])

    # Create the weight matrix
    w = np.tril(w0**p,0)

    # Calculate the ewma
    return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)



if __name__ == "__main__":
    window = 5
    overlap = 0

    # get participant number
    with open('./Results/participant.txt', 'r') as f:
        participant = int(f.read())

    # define bands and channels
    bands = {'theta': (4,8), 'alpha': (8, 12), 'beta': (12, 30)}
    channels = ['F7', 'F3', 'Fz', 'F4', 'F8', 'O1', 'O2', 'Oz']
    fname = f'C:/Users/cogexp/Desktop/Hila_thesis/calib{participant}.xdf'

    # read raw data
    raw_calib = read_raw(fname, stream_ids=[2], preload=True)

    # choose electrodes
    raw_calib.pick_channels(channels)

    # resample data
    raw_calib.resample(128)

    # filter data
    calib_filtered = filter_raw(raw_calib)

    # epoch data
    epochs_calib = mne.make_fixed_length_epochs(calib_filtered, duration=window, overlap = overlap)

    # calculate EI
    ei_calib, alpha_calib, beta_calib, theta_calib = calculate_EI(epochs_calib, bands)

    # median filtering
    ei_mid_calib = scipy.ndimage.median_filter(ei_calib, size = 3)

    # EWMA
    ei_ewma_calib = ewma(ei_calib, 0.2)

    # get annotations' time stamps
    annotations_calib = raw_calib.annotations

    # put onsets into a list
    onsets_calib = annotations_calib.onset

    # put type of annotations into a list
    types_calib = annotations_calib.description
    # remove the .0 from the end of the string
    types_calib = [x[:-2] for x in types_calib]


    # find min and max EI values
    min_ei = min(ei_ewma_calib)
    max_ei = max(ei_ewma_calib)

    # save min and max EI values as numpy array:
    np.save(f'./Results/min_max_ei_{participant}.npy', [min_ei, max_ei])

