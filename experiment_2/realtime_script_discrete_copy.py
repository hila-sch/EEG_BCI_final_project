import numpy as np
import signal
import matplotlib.pyplot as plt
import scipy
import random
import select
import msvcrt
import queue
import multiprocessing as mp
import threading
from datetime import datetime
import pandas as pd
from mne_lsl.stream import StreamLSL as Stream
import mne

# from mne.datasets import sample
# from mne.io import read_raw_fif
import time
import math
import sys
import csv

# from mne_realtime import LSLClient, MockLSLStream
from mne_lsl.lsl import local_clock
# import mne_realtime
from mne_lsl.stream import StreamLSL as Stream
import numpy as np
from pylsl import StreamInfo, StreamOutlet
import pylsl

# def filter_raw(raw, bandpass=(0.5, 45), notch=(50), notch_width=5):
#     raw.notch_filter(notch, notch_widths=notch_width)
#     raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])
#     return raw
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

def find_min_max(participant_number):
    '''This function finds the min and max values of the EI scores from the calibration phase.
    It is used to normalize the EI scores in the realtime script.

    Parameters:
    -----------
    participant_number : int
        The participant number of the participant whose min and max values are to be found.

    Returns:
    --------
    ei_min : float
        The minimum value of the EI scores.
    ei_max : float
        The maximum value of the EI scores.

    '''
    fname = './Results/min_max_ei_' + str(participant_number) + '.npy'
    min_max = np.load(fname)
    ei_min, ei_max = min_max[0], min_max[1]

    return ei_min, ei_max


#function to check if the queue is empty, and otherwise return the first item
def event_manager(q_to_lsl, markers, event_queue, video_event, end_event, outlet):
    '''This function is responsible for managing the event queue.
    It checks if the queue is empty, and if not, it gets the first item in the queue.
    If the item is a marker, it is put in the event queue and sent to the LSL outlet.
    If the item is the last marker, the end event is set and the function breaks.

    Parameters:
    -----------
    q_to_lsl : queue
        The queue that contains the markers that are sent from the parent process.
    markers : dictionary
        Dictionary containing the markers' names and their values.
        Example: markers = {'video start': 0,
                            'video end': 1,
                            'video paused': 2,
                            'video resumed': 3,
                            'attention': 4}
    event_queue : queue
        The queue that contains the markers that are sent to the LSL outlet.
    video_event : threading.Event
        The event that is set when the video starts and cleared when the video ends.
    end_event : threading.Event
        The event that is set when the last video ends.
    outlet : pylsl.StreamOutlet 
        The LSL outlet that sends the markers to the LSL stream.

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

                if msg == markers['video start']:
                    print('video started')
                    video_event.set()
                elif msg == markers['video end']:
                    print('video ended')
                    video_event.clear()

                if msg == markers['last video ended']:
                    print('last video ended')
                    end_event.set()
                    break

def lsl_main(q_from_lsl, q_to_lsl, markers):
    '''This function is responsible for the realtime computation of the EI.
    It gets the data from the LSL stream, filters it, calculates the EI and sends it to the parent process.
    It also saves the EI scores to a file.

    Parameters:
    -----------
    q_from_lsl : queue
        The queue that contains the markers that are sent from the parent process.
    q_to_lsl : queue
        The queue that contains the markers that are sent to the LSL outlet.
    markers : dictionary
        Dictionary containing the markers' names and their values.
        Example: markers = {'video start': 0,
                            'video end': 1,
                            'video paused': 2,
                            'video resumed': 3,
                            'attention': 4}
    
    Returns:
    --------
    None
    '''
    video_event = threading.Event()
    end_event = threading.Event()
    during_video = threading.Event()
    # event_array = np.empty((0,2))
    event_queue = queue.Queue()



    date_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    df_file = './Results/ei_realtime_' + date_now + '.csv'

    # load participant number from file:
    with open('./Results/participant.txt', 'r') as f:
        participant_number = int(f.read())

    # load min and max values from file
    ei_min, ei_max = find_min_max(participant_number)
    
    
    #threading.Thread(target = event_manager, args=(q_to_lsl, markers, event_queue, video_event, end_event, outlet), daemon=True).start()


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


    # Load a file to stream raw data
    # #data_path = sample.data_path()
    # data_path = 'sub1.fif'
    # #raw_fname = data_path  / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
    # raw = read_raw_fif(data_path).load_data()


    bandpass = (1, 45)
    # notch = 50
    # notch_width = 10


    # number of raws to compute EI for
    # n_raws = 3

    # number of seconds per raw
    n_sec = 5

    bands = {'theta': (4,8), 'alpha': (8, 12), 'beta': (12, 30)}
    #channels = ['F9', 'F10', 'F3', 'F4', 'FCz', 'O1', 'O2']
    #channels = ['0','1']
    channels = ['4', '5', '1', '3', '20', '21', '2']
    ei_score = []
    avg_ei = []
    temp_result = []
    mid_result = []
    ewma_result = []
    norm_result = []


    # create a stream outlet
    info = StreamInfo(name='Trigger_stream', type='Markers', channel_count=1, nominal_srate=0, source_id='psy_marker')
    outlet = StreamOutlet(info) 
    # Broadcast the stream.   
    #outlet.push_sample(['start'])

    


    # start the event manager thread
    threading.Thread(target = event_manager, args=(q_to_lsl, markers, event_queue, video_event, end_event, outlet), daemon=True).start()

    # with MockLSLStream(host, raw, 'eeg'):
    
    stream = Stream(bufsize=5, name= stream_name)  # 5 seconds of buffer
    stream.connect(acquisition_delay=0.2)
    print(stream.info)

    # with LSLClient(host=host) as client:
    # client_info = client.get_measurement_info()
    sfreq = stream.info["sfreq"]

    # print message from parent process
    print('Child process started')

    # wait for video start
    msg = event_queue.get()
    
    # start computing EI when video starts
    with A(1):
        if msg == markers['video start']:
            print("video started- will now compute engagement index")
            count = 0
            while end_event.is_set() == False:
                # while video_event.is_set():
                # for ii in range(n_raws):
                
                tic = time.perf_counter()
                count+=1

                # print('Got raw %d/%d' % (ii + 1, n_raws))
                #winsize = int(n_sec * sfreq)
                winsize = n_sec
                data, ts = stream.get_data(winsize)
                #check the size of data
                print('size', data.shape)
                raw = mne.io.RawArray(data=data, info=stream.info, verbose=False)
                # raw = client.get_data_as_raw(n_samples=sfreq*n_sec)

                # resample
                # print(raw.ch_names)
                raw.pick(channels)
                # raw.resample(200)

                # filter data
                raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])

                # calculate EI
                ei = calculate_EI(raw, bands)
                print(f'ei computed is {ei}')

                temp_score = ei.item(-1)
                print(temp_score)

                ei_score.append(temp_score) # list that saves all the ei scores
                #print(ei_score)

                # update min and max values if needed
                
                # create a list for the the last 30 seconds of EI scores
                # ei_window = []
                # add the last 3 EI scores (15 seconds):
                # if len(ei_score) > 0 :
                #     ei_window = ei_score[-3:] + temp_score
                # else:
                # ei_window = ei_score + temp_score
            

                ei_arr = np.array(ei_score)
                print("ei array created")
                #ei_mid = signal.medfilt(ei_arr, kernel_size=3)

                # apply median filter
                # if len(ei_arr) < 3:
                #     ei_mid = ei_arr
                # else:
                ei_mid = scipy.ndimage.median_filter(ei_arr, size = 3)
                print(ei_mid)
                print("median filter completed")

                # apply exponential weighted moving average
                ei_ewma = ewma(ei_mid, alpha = 0.2)
                print("ewma completed")

                    
                if min(ei_ewma) < ei_min:
                    ei_min = min(ei_ewma)
                    print("new min: ", ei_min)
                if max(ei_ewma) > ei_max:
                    ei_max = max(ei_ewma)
                    print("new max: ", ei_max)

                print('current min: ', ei_min, 'current max: ', ei_max)

                # normalize the scores
                ei_norm = ((ei_ewma - ei_min) / (ei_max - ei_min)) * 100

                # ei_norm = (ei_ewma - 0) / (1 - 0) * 100
                print("normalization completed")

                # ei_result = math.ceil(np.mean(ei_norm))
                ei_result = math.ceil(ei_norm[-1])

                temp_result.append(temp_score)
                mid_result.append(ei_mid[-1])
                ewma_result.append(ei_ewma[-1])
                norm_result.append(ei_result)

                print(f'ei score is : {ei_result}')
                toc = time.perf_counter()

         

                print(f"computation took {toc - tic:0.4f} seconds")
                # sleep for 5 seconds minus the time it took to compute the EI
                time.sleep(4 - (toc - tic))

                
                msg = ei_result

                # send the score to the parent process for feedback only during the video
                if video_event.is_set():
                    print('sending ei to parent process')
                    q_from_lsl.put(msg)


                    # every 5 minutes save to df
                if count == 12:       
                #df = pd.DataFrame({'temp_result': ei_score, 'mid_result': mid_result, 'ewma_result': ewma_result, 'norm_result': norm_result})
                    df = pd.DataFrame({'temp_result': ei_score, 'mid_result': ei_mid, 'ewma_result': ewma_result, 'norm_result': norm_result})
                    df.to_csv(df_file, index=False)
                    count = 0

                # sleep

                #print('sleeping for 5 seconds')

            df = pd.DataFrame({'temp_result': ei_score, 'mid_result': ei_mid, 'ewma_result': ewma_result, 'norm_result': norm_result})
            df.to_csv(df_file, index=False)
    
    print('Streams closed')



# main function is necessary here to enable script as own program
# in such way a child process can be started (primarily for Windows)

if __name__ == '__main__':

    q_from_lsl = queue.Queue()
    q_to_lsl = queue.Queue()

    markers = {'video start': 0,
                'video end': 1,
                'video paused': 2,
                'video resumed': 3,
                'attention': 4}
    
    lsl_main(q_from_lsl, q_to_lsl, markers)