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
    '''This function is responsible for managing the event queue and the event array.
    It is a separate thread that runs in parallel with the main thread.
    It added the EEG markers to the LSL stream, in addition to adding them to the event array.
    The event array is used for during process updates.'''
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
    bs_event = threading.Event()
    ma_event = threading.Event()
    end_event = threading.Event()
    event_array = np.empty((0,2))
    event_queue = queue.Queue()

    info = StreamInfo(name='Trigger_stream_calibration', type='Markers', channel_count=1, nominal_srate=0, source_id='psy_marker')
    outlet = StreamOutlet(info) 
    threading.Thread(target = event_manager, args=(q_to_lsl, markers, event_queue, bs_event, ma_event, end_event, outlet), daemon=True).start()
    
    

    # this is the host id that identifies your stream on LSL
    host = 'mne_stream'
    # this is the max wait time in seconds until client connection
    wait_max = 5


    # Load a file to stream raw data
    #data_path = sample.data_path()
    # data_path = 'sub1.fif'
    # #raw_fname = data_path  / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
    # raw = read_raw_fif(data_path).load_data()

    

    #for the loop:
    stop_time = 300
    t = 0 
    #generate_data = True



    # For this example, let's use the mock LSL stream.
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

    #recording
    # record_dir = Path('~/bsl_data/examples').expanduser()
    # os.makedirs(record_dir, exist_ok=True)
    # print (record_dir)

    

    with LSLClient(host=host) as client:
        # start_time = time.time()
        client_info = client.get_measurement_info()
        sfreq = int(client_info['sfreq'])

        # print message from parent process
        print('Child process started')



        # if msg == 'video':
        #     print("video started- will now compute engagement index")
        # msg = q_to_lsl.get()
        # # if msg != None:
        # #     msg_str = str(msg)
        # #     print(f'msg is: {msg}')
        
        # if event_queue.get_nowait() == markers['video started']:
        #     print("video started- will now compute engagement index")
        # if video_event.is_set():
        time.sleep(5)
        while end_event.is_set() == False:
            # while video_event.is_set():
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


        

    # ei_bs_arr = np.array(ei_bs)
    # ei_ma_arr = np.array(ei_ma)

    # # calculate median filter
    # ei_mid_bs = scipy.ndimage.median_filter(ei_bs_arr, size = 3)
    # ei_mid_ma = scipy.ndimage.median_filter(ei_ma_arr, size = 3)

    # # calculate ewma
    # ei_ewma_bs = ewma(ei_mid_bs, alpha = 0.2)
    # ei_ewma_ma = ewma(ei_mid_ma, alpha = 0.2)


    # # save numpy array to csv file
    # np.savetxt(r'./Results/ei_bs.csv', ei_ewma_bs, delimiter=',')
    # np.savetxt(r'./Results/ei_ma.csv', ei_ewma_ma, delimiter=',')


            # ei_result = np.mean(ei_ewma)


            # recorder.stop()
            # print (recorder)



    # with open(r'C:/Users/hilas/OneDrive/RUG_CCS/year2/Thesis/Project/ei.txt', 'w') as fp:
    #     fp.write("\n".join(str(item) for item in ei_score))

    # with open(r'C:/Users/hilas/OneDrive/RUG_CCS/year2/Thesis/Project/ei_mid.txt', 'w') as fp:
    #     fp.write("\n".join(str(item) for item in ei_mid_to_file))

    # with open(r'C:/Users/hilas/OneDrive/RUG_CCS/year2/Thesis/Project/ei_ewma.txt', 'w') as fp:
    #     fp.write("\n".join(str(item) for item in ei_ewma_to_file))

    # with open(r'ei_bs.txt', 'w') as fp:
    #     fp.write("\n".join(str(item) for item in avg_ei_bs))
    
    # with open(r'ei_ma.txt', 'w') as fp:
    #     fp.write("\n".join(str(item) for item in avg_ei_ma))



# main function is necessary here to enable script as own program
# in such way a child process can be started (primarily for Windows)
# if __name__ == '__main__':
    # q_from_lsl = queue.Queue()
    # q_to_lsl = queue.Queue()
    # lsl_main(q_from_lsl, q_to_lsl)