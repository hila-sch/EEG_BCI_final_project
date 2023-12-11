import numpy as np
import scipy
import random
import queue
import multiprocessing as mp
import threading
import numpy as np

from mne.datasets import sample
from mne.io import read_raw_fif
import time
import math


from mne_realtime import LSLClient, MockLSLStream
import numpy as np
from pylsl import StreamInfo, StreamOutlet

import os
from pathlib import Path

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
    return ei

def filter_raw(raw, bandpass=(0.5, 45), notch=(50), notch_width=10):
    raw.notch_filter(notch, notch_widths=notch_width)
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])
    return raw


def event_manager(q_to_lsl, markers, event_queue, bs_event, ma_event, end_event, outlet):
    '''This function is responsible for managing the event queue.
    It checks if the queue is empty, and if not, it gets the first item in the queue.
    If the item is a marker, it is put in the event queue and sent to the LSL outlet.
    If the item is the last marker, the end event is set and the function breaks.

    Parameters:
    -----------
    q_to_lsl : queue.Queue
        Queue containing the markers to be sent to the LSL outlet.
    markers : dictionary
        Dictionary containing the markers' names and their values.
        Example: markers = {'baseline started': 'bs', 'baseline ended': 'be', 'MA started': 'ms', 'MA ended': 'me'}
    event_queue : queue.Queue
        Queue containing the markers to be sent to the LSL outlet.
    bs_event : threading.Event
        Event that is set when the relaxation baseline start marker is received.
    ma_event : threading.Event
        Event that is set when the mental arithmetic task start marker is received.
    end_event : threading.Event
        Event that is set when the last marker is received.
    outlet : pylsl.StreamOutlet
        LSL outlet to which the markers are sent.
    

    Returns:
    --------
    None
'''
    while True:
        if not q_to_lsl.empty():
            msg = q_to_lsl.get_nowait()
            print(msg)
            if msg in markers.values():
                event_queue.put_nowait(msg)
                outlet.push_sample([msg])

                if msg == markers['baseline started']:
                    print('baeline started')
                    bs_event.set()
                elif msg == markers['baseline ended']:
                    print('baeline ended')
                    bs_event.clear()
                elif msg == markers['MA started']:
                    print('MA started')
                    ma_event.set()
                elif msg == markers['MA ended']:
                    print('MA ended')
                    ma_event.clear()
                    end_event.set()
                    break





def lsl_calib(q_from_lsl, q_to_lsl, markers):
    '''
    This function is responsible for the calibration phase.
    It starts a child process that is responsible for managing the event queue.
    It then starts the LSL client and gets the data from the LSL stream.
    The data is filtered and the EI is calculated.
    The EI is then saved to a file.
    
    Parameters:
    -----------
    q_from_lsl : queue.Queue
        Queue containing the data from the LSL stream.
    q_to_lsl : queue.Queue
        Queue containing the markers to be sent to the LSL outlet.
    markers : dictionary
        Dictionary containing the markers' names and their values.
        Example: markers = {'baseline started': 'bs', 'baseline ended': 'be', 'MA started': 'ms', 'MA ended': 'me'}

    Returns:
    --------
    None
    '''
    bs_event = threading.Event()
    ma_event = threading.Event()
    end_event = threading.Event()
    event_array = np.empty((0,2))
    event_queue = queue.Queue()

    info = StreamInfo(name='Trigger_stream_calibration', type='Markers', channel_count=1, nominal_srate=0, source_id='psy_marker')
    outlet = StreamOutlet(info) 
    threading.Thread(target = event_manager, args=(q_to_lsl, markers, event_queue, bs_event, ma_event, end_event, outlet), daemon=True).start()
    
    

    # this is the host id that identifies your stream on LSL
    #host = 'mne_stream'
    host = 'mne_stream_EEG_4_250.0_float32_UT177560'
    # this is the max wait time in seconds until client connection
    wait_max = 5


    #for the loop:
    stop_time = 300
    t = 0 

    n_epochs = 3
    n_sec = 5
    bands = {'theta': (4,8), 'alpha': (8, 12), 'beta': (12, 30)}
    channels = ['FP1', 'FP2', 'F7', 'F3', 'F4', 'F8', 'O1', 'O2']
    ei_score = []
    ei_mid_to_file = []
    ei_ewma_to_file = []
    ei_ma = []
    ei_bs = []
    ewma_result = []

    bandpass = (0.5, 45)


    

    with LSLClient(host=host) as client:
        # start_time = time.time()
        client_info = client.get_measurement_info()
        sfreq = int(client_info['sfreq'])

        # print message from parent process
        print('Child process started')


        time.sleep(5)
        while end_event.is_set() == False:
            # while video_event.is_set():
            epoch = client.get_data_as_epoch(n_samples=sfreq*n_sec)
            # resample
            epoch.pick_channels(channels)
            epoch.resample(128)

            # filter data
            epoch.filter(l_freq=bandpass[0], h_freq=bandpass[1])
            # filtered_epoch = filter_raw(epoch)
            #epoch.filter(l_freq = 0.5, h_freq = 40)

            # calculate EI
            ei = calculate_EI(epoch, bands)
            temp_score = ei.item(-1)
            print(temp_score)
            ei_score.append(temp_score) # list that saves all the ei scores
            # print(ei_score)
        

            ei_arr = np.array(ei_score)
            print("ei array created")
            #ei_mid = signal.medfilt(ei_arr, kernel_size=3)

            # apply median filter
            if len(ei_arr) < 3:
                ei_mid = ei_arr
            else:
                ei_mid = scipy.ndimage.median_filter(ei_arr, size = 3)
            print("median filter completed")

            # apply exponential weighted moving average
            ei_ewma = ewma(ei_mid, alpha = 0.2)

            ewma_result.append(ei_ewma[-1])

            print("ewma completed")

            # for ii in range(n_epochs):
            #             print('Got epoch %d/%d' % (ii + 1, n_epochs))
            #             epoch = client.get_data_as_epoch(n_samples=sfreq*n_sec, picks=channels)
            #             # choose channels and resample
            #             # epoch.pick_channels(channels)
            #             epoch.resample(128)

            #             # filter data
            #             filtered_epoch = filter_raw(epoch)
            #             #epoch.filter(l_freq = 0.5, h_freq = 40)

            #             # calculate EI
            #             ei = calculate_EI(filtered_epoch, bands)
            #             temp_score.append(ei.item())

            # while t < stop_time:
            # temp_score = []
            # for ii in range(n_epochs):
            # # print('Got epoch %d/%d' % (ii + 1, n_epochs))
            # epoch = client.get_data_as_epoch(n_samples=sfreq*n_sec, picks = channels)
            # epoch.resample(128)
            # epoch.filter(l_freq=bandpass[0], h_freq=bandpass[1])
            # # epoch.filter(l_freq = 0.5, h_freq = 40)
            # ei = calculate_EI(epoch, bands).item()
            # print(f'ei is: {ei}')
            # # temp_score.append(ei.item())

            # # ei_arr = np.array(temp_score)
            # #ei_mid = signal.medfilt(ei_arr, kernel_size=3)
            
            # if bs_event.is_set():
            #     ei_bs.append(ei)
            #     print("added bs score")
            # elif ma_event.is_set():
            #     ei_ma.append(ei)
            #     print("added ma score")
            # avg_ei.append(ei_result)
            # print(f'ei{msg}')
            # ei_score.extend(temp_score)

    
            print('Streams closed')
            with open('./Results/participant.txt', 'r') as f:
                participant = int(f.read())
            np.save(f'./Results/min_max_ei_{participant}.npy', [min(ewma_result), max(ewma_result)])
            # fname = f'C:/Users/cogexp/Desktop/Hila_thesis/calib{participant}.xdf'
            # convert lists to numpy array


 
# main function is necessary here to enable script as own program
# in such way a child process can be started (primarily for Windows)
# if __name__ == '__main__':
    # q_from_lsl = queue.Queue()
    # q_to_lsl = queue.Queue()
    # lsl_main(q_from_lsl, q_to_lsl)