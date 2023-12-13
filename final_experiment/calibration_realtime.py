import numpy as np
import scipy
import random
import queue
import multiprocessing as mp
import threading
import numpy as np

# from mne.datasets import sample
# from mne.io import read_raw_fif
import time
import math
from mne_lsl.stream import StreamLSL as Stream
import mne


# from mne_realtime import LSLClient, MockLSLStream
import numpy as np
from pylsl import StreamInfo, StreamOutlet
import pylsl

import os
from pathlib import Path

class A:
    def __init__(self, i):
        self.b = i
    def __exit__(*args, **kwargs):
        print ("exit")
    def __enter__(*args, **kwargs):
        print ("enter")
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

# def calculate_EI (epoch, bands):
#     '''
#     Calculate the EI for each epoch and return a list of EI values.

#     Parameters:
#     -----------
#     epoch : mne.epochs.Epochs
#             Epochs object containing the data.
#     bands : dictionary
#             Dictionary containing the bands' names and their limits.
#             Example: bands = {'theta': (4,8), 'alpha': (8, 12), 'beta': (12, 30)}

#     Returns:
#     --------
#     ei : list
#             List of EI values for each epoch.

#     To calculate the EI:
#     The function first calculates the PSD for each epoch,
#     then averages the PSD over all electrodes and then averages the PSD over the
#     frequencies in the band of interest. 
#     The EI is then calculated by dividing the average PSD of the beta band by 
#     the sum of the average PSD of the theta and alpha bands.'''

#     avg_bands = {}
#     for band_name, band_limits in bands.items():
#         low, high = band_limits
#         psds = epoch.compute_psd(method='welch', fmin=low, fmax=high)
#         avg_over_electrodes= psds.get_data().mean(1)
#         avg_over_band = avg_over_electrodes.mean(1)
#         avg_bands[band_name] = avg_over_band.tolist()
#         # print(avg_bands[band_name])
    
#     sum_lists = np.add(avg_bands['theta'], avg_bands['alpha'])
#     ei = np.divide(avg_bands['beta'], sum_lists)
#     return ei


def calculate_EI (raw, bands):
    '''
    Calculate the EI for each raw and return a list of EI values.

    Parameters:
    -----------
    raw : mne.raws.raws
            raws object containing the data.
    bands : dictionary
            Dictionary containing the bands' names and their limits.
            Example: bands = {'theta': (4,8), 'alpha': (8, 12), 'beta': (12, 30)}

    Returns:
    --------
    ei : list
            List of EI values for each raw.

    To calculate the EI:
    The function first calculates the PSD for each raw,
    then averages the PSD over all electrodes and then averages the PSD over the
    frequencies in the band of interest. 
    The EI is then calculated by dividing the average PSD of the beta band by 
    the sum of the average PSD of the theta and alpha bands.'''
    
    avg_bands = {}
    for band_name, band_limits in bands.items():
        low, high = band_limits
        psds = raw.compute_psd(method='welch', fmin=low, fmax=high)
        # dimensions of psds
        # print(psds.get_data().shape)
        avg_over_electrodes= psds.get_data().mean(0)
        # print(avg_over_electrodes.shape)
        avg_over_band = avg_over_electrodes.mean(0)
    #     avg_over_band = avg_over_electrodes.mean(1)
        avg_bands[band_name] = avg_over_band.tolist()
        #print(avg_bands[band_name])
    
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
    stream_name = '0'
    while stream_name != 'EE225-000000-000758-02-DESKTOP-8G8988B':
    #  while True:
        streams = pylsl.resolve_streams()
        for ii, stream in enumerate(streams):
            stream_name = stream.name()
            print('%d: %s' % (ii, stream_name))
            if stream_name == 'EE225-000000-000758-02-DESKTOP-8G8988B':
                break

    print('found stream')
    # name = 
    wait_max = 5



    bandpass = (1, 45)
    # notch = 50
    # notch_width = 10


    # number of epochs to compute EI for
    n_epochs = 3

    # number of seconds per epoch
    n_sec = 5

    bands = {'theta': (4,8), 'alpha': (8, 12), 'beta': (12, 30)}
    #channels = ['F9', 'F10', 'F3', 'F4', 'FCz', 'O1', 'O2']
    channels = ['4', '5', '1', '3', '20', '21', '2']
    ei_score = []
    avg_ei = []
    temp_result = []
    mid_result = []
    ewma_result = []
    norm_result = []

    
    stream = Stream(bufsize=5, name= stream_name)  # 5 seconds of buffer
    stream.connect(acquisition_delay=0.2)
    print(stream.info)

    # with LSLClient(host=host) as client:
    # client_info = client.get_measurement_info()
    sfreq = stream.info["sfreq"]

    # print message from parent process
    print('Child process started')


    with A(1):
        # start_time = time.time()
        # client_info = client.get_measurement_info()
        # sfreq = int(client_info['sfreq'])

        # print message from parent process
        # print('Child process started')


        time.sleep(5)

        while end_event.is_set() == False:
            # while video_event.is_set():
            #epoch = client.get_data_as_epoch(n_samples=sfreq*n_sec)

            winsize = n_sec
            data, ts = stream.get_data(winsize)
            # resample
            raw = mne.io.RawArray(data=data, info=stream.info, verbose=False)
                # raw = client.get_data_as_raw(n_samples=sfreq*n_sec)

                # resample
                # print(raw.ch_names)
            raw.pick(channels)
                # raw.resample(200)
            # epoch.pick_channels(channels)
            # epoch.resample(200)
            raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

            # filtered_epoch = filter_raw(epoch)
            #epoch.filter(l_freq = 0.5, h_freq = 40)

            # calculate EI
            #ei = calculate_EI(epoch, bands)
            ei = calculate_EI(raw, bands)
            print(f'ei computed is {ei}')

            temp_score = ei.item(-1)
            print(temp_score)

            ei_score.append(temp_score) # list that saves all the ei scores
            print(ei_score)
        

            ei_arr = np.array(ei_score)
            print("ei array created")
            #ei_mid = signal.medfilt(ei_arr, kernel_size=3)

            ei_mid = scipy.ndimage.median_filter(ei_arr, size = 3)
            print("median filter completed")

            # apply exponential weighted moving average
            ei_ewma = ewma(ei_mid, alpha = 0.2)

            ewma_result.append(ei_ewma[-1])

            print("ewma completed")

            # sleep for 5 seconds
            time.sleep(5)

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

    
        with open('./Results/participant.txt', 'r') as f:
            participant = int(f.read())
        np.save(f'./Results/min_max_ei_{participant}.npy', [min(ewma_result), max(ewma_result)])
        # np.save(f'./Results/min_max_ei_{participant}.npy', [0.01, 0.07])
        # fname = f'C:/Users/cogexp/Desktop/Hila_thesis/calib{participant}.xdf'
        # convert lists to numpy array

        print('Streams closed')
 
# main function is necessary here to enable script as own program
# in such way a child process can be started (primarily for Windows)
# if __name__ == '__main__':
    # q_from_lsl = queue.Queue()
    # q_to_lsl = queue.Queue()
    # lsl_main(q_from_lsl, q_to_lsl)