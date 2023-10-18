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
import pandas as pd

from mne.datasets import sample
from mne.io import read_raw_fif
import time
import math
import sys
import csv

from mne_realtime import LSLClient, MockLSLStream
import mne_realtime
import numpy as np
from pylsl import StreamInfo, StreamOutlet

# def filter_raw(epoch, bandpass=(0.5, 45), notch=(50), notch_width=5):
#     epoch.notch_filter(notch, notch_widths=notch_width)
#     raw.filter(l_freq=bandpass[0], h_freq=bandpass[1])
#     return raw

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
    return ei

def find_min_max():
    # load csv file to numpy array
    bs = np.loadtxt("./Results/ei_bs.csv", delimiter=",")
    ma = np.loadtxt("./Results/ei_ma.csv", delimiter=",")

    # find min and max values
    ei_min = min(bs)
    print(ei_min)
    ei_max = max(ma)
    print(ei_max)

    return ei_min, ei_max


#function to check if the queue is empty, and otherwise return the first item
def event_manager(q_to_lsl, markers, event_queue, video_event, end_event, outlet):
    '''This function is responsible for managing the event queue and the event array.
    It is a separate thread that runs in parallel with the main thread.
    It added the EEG markers to the LSL stream, in addition to adding them to the event array.
    The event array is used for during process updates.'''
    while True:
        if not q_to_lsl.empty():
            msg = q_to_lsl.get_nowait()
            print(msg)
            if msg in markers.values():
                # newrow = np.array([msg['event_type'], msg['event_onset']])
                # event_array = np.vstack([event_array, newrow])
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
                    # ma_event.clear()
                    end_event.set()
                    # np.save('event_array.npy', event_array)
                    break

def lsl_main(q_from_lsl, q_to_lsl, markers):
    video_event = threading.Event()
    end_event = threading.Event()
    # event_array = np.empty((0,2))
    event_queue = queue.Queue()
    video_counter = queue.Queue()

    # load participant number from file:
    with open('./Results/participant.txt', 'r') as f:
        participant_number = int(f.read())
    
    # load min max from file
    fname = './Results/min_max_ei_' + str(participant_number) + '.npy'
    min_max = np.load(fname)
    ei_min, ei_max = min_max[0], min_max[1]
    # threading.Thread(target = event_manager, args=(q_to_lsl, markers, event_queue, video_event, end_event, outlet), daemon=True).start()
        # print(__doc__)

    # this is the host id that identifies your stream on LSL
    host = 'mne_stream'
    # this is the max wait time in seconds until client connection
    wait_max = 5


    # Load a file to stream raw data
    # #data_path = sample.data_path()
    data_path = 'C:/Users/hilas/Desktop/Pilot_results/P2pilot.fif'
    # #raw_fname = data_path  / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
    raw = read_raw_fif(data_path).load_data()


    bandpass = (0.5, 45)
    notch = 50
    notch_width = 10



    # For this example, let's use the mock LSL stream.
    n_epochs = 3
    n_sec = 5
    bands = {'theta': (4,8), 'alpha': (8, 12), 'beta': (12, 30)}
    channels = ['F7', 'F3', 'Fz', 'F4', 'F8', 'O1', 'O2', 'Oz']
    ei_score = []
    ei_mid_to_file = []
    ei_ewma_to_file = []
    avg_ei = []
    temp_result = []
    mid_result = []
    ewma_result = []
    norm_result = []

    # date_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    # df_file = './Results/ei_realtime_' + date_now + '.csv'


    info = StreamInfo(name='Trigger_stream', type='Markers', channel_count=1, nominal_srate=0, source_id='psy_marker')
    outlet = StreamOutlet(info) 
    # Broadcast the stream.   
    # outlet.push_sample(['start'])
    threading.Thread(target = event_manager, args=(q_to_lsl, markers, event_queue, video_event, end_event, outlet), daemon=True).start()

    with MockLSLStream(host, raw, 'eeg'):
        with LSLClient(host=host) as client:
            start_time = time.time()
            client_info = client.get_measurement_info()
            sfreq = int(client_info['sfreq'])

            # print message from parent process
            print('Child process started')


            # if msg == 'video':
            #     print("video started- will now compute engagement index")
            msg = event_queue.get()
            

            video_count = 0


            if msg == markers['video start']:
                print("video started- will now compute engagement index")
                while end_event.is_set() == False:
                    # while video_event.is_set():
                    for ii in range(n_epochs):
                        tic = time.perf_counter()
                        print('Got epoch %d/%d' % (ii + 1, n_epochs))
                        epoch = client.get_data_as_epoch(n_samples=sfreq*n_sec)
                        # resample
                        epoch.pick_channels(channels)
                        epoch.resample(128)

                        # filter data
                        # epoch.notch_filter(notch, notch_widths=notch_width)
                        epoch.filter(l_freq=bandpass[0], h_freq=bandpass[1])
                        # filtered_epoch = filter_raw(epoch)
                        #epoch.filter(l_freq = 0.5, h_freq = 40)

                        # calculate EI
                        ei = calculate_EI(epoch, bands)
                        temp_score = ei.item(-1)
                        print(temp_score)
                        ei_score.append(temp_score) # list that saves all the ei scores
                        # print(ei_score)

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
                        if len(ei_arr) < 3:
                            ei_mid = ei_arr
                        else:
                            ei_mid = scipy.ndimage.median_filter(ei_arr, size = 3)
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
                        if ei_norm[-1] > 100:
                            ei_norm[-1] = 100
                            max_count += 1
                        # ei_norm = (ei_ewma - 0) / (1 - 0) * 100
                        print("normalization completed")

                        # calculate the average of the EI scores
                        # ei_result = math.ceil(np.mean(ei_norm))
                        ei_result = math.ceil(ei_norm[-1])

                        temp_result.append(temp_score)
                        mid_result.append(math.ceil(ei_mid[-1]))
                        ewma_result.append(math.ceil(ei_ewma[-1]))
                        norm_result.append(ei_result)
                        print(f'ei score is : {ei_result}')
                        toc = time.perf_counter()
                        print(f"computation took {toc - tic:0.4f} seconds")


                    print('sending feedcack to parent process')
                    msg = ei_result

                    # send the score to the parent process for feedback
                    q_from_lsl.put(msg)

                    

                    # add the new scores to the list
                    # ei_score.extend(temp_score)

                # avg_ei = np.array(avg_ei)
                # np.save('./Results/avg_ei.npy', avg_ei)

                        
        # with open(r'./Results/avg_ei.txt', 'w') as fp:
        #     for item in avg_ei:
        #         # write each item on a new line
        #         fp.write("%s\n" % item)

    # make one pandas dataframe out of temp_result, mid_results, ewma_results, norm_results
    df = pd.DataFrame({'temp_result': ei_score, 'mid_result': mid_result, 'ewma_result': ewma_result, 'norm_result': norm_result})
    print(df)
    df.to_csv('./Results/ei_scores.csv', index=False)


    with open(r'./Results/ei_avg_1.txt', 'w') as fp:
        fp.write("\n".join(str(item) for item in norm_result))
    print('Streams closed')

    # convert avg_ei to numpy array and save to file
    print(avg_ei)
    





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