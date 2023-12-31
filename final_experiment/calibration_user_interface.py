import PySimpleGUI as sg
import pandas as pd
import csv
import vlc
from sys import platform as PLATFORM
import random
from random import *
import re
from playsound import playsound
import time
from pathlib import Path
# from colour import Color
from playsound import playsound
import multiprocessing as mp
import threading
from calibration_realtime import *
from mne_lsl.stream import StreamLSL as Stream
import mne
import os



def info_window(txt, key_txt):
    '''
    This function takes a txt file and returns a layout for a window with the text in the txt file.
    
    Parameters
    ----------
    txt : str
        The path to the txt file.
        key_txt : str
        The key for the button in the window.
        
    Returns
    -------
    layout : list
        The layout for the window.
        
        '''
    my_file = open(txt, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    layout = []
    for line in data_into_list:
        layout.append([sg.Text(line, font='Arial 20')])
        layout.append([sg.Text('', size=(1,1))])

    layout.append([sg.Button('Ok', key = key_txt)])

    return layout

def trial_window():
    '''
    This function returns a layout for a window with an equation and an input field.'''
    layout = [[sg.Text(' ', size = (1,10))],
        [sg.Text(' ', key = 'eq', font='Arial 25')],
        [sg.InputText(size = (4,1), key='answer', font='Arial 20')],
        [sg.Text(' ', key='feedback')]]
    
    return layout

def relax_window(txt, key_txt):
    '''
    This function takes a txt file and returns a layout for a window with the text in the txt file.
    It is used for the relaxation period during calibration.

    Parameters
    ----------
    txt : str
        The path to the txt file.

    key_txt : str
        The key for the button in the window.

    Returns
    -------
    sg.Window
        The window with the text in the txt file.
    '''
    my_file = open(txt, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    layout = []
    for line in data_into_list:
        layout.append([sg.Text(line, font='Arial 20')])
        layout.append([sg.Text('', size=(1,1))])

    layout.append([sg.Text('120', font=('Helvetica', 48), key='-TIMER-')])
    layout.append([sg.Text('', size=(1,5))])
    layout.append([sg.Button('Start')])
    layout.append([sg.Button('Ok', key = key_txt, visible = False)])

    return sg.Window('Calibration', layout, element_justification='center', finalize=True, resizable=True)

def clear_input(window, values):
    '''
    This function clears the input field in the window.

    Parameters
    ----------
    window : sg.Window
        The window with the input field.
    values : dict
        The values of the window.

    Returns
    -------
    None.
    '''
    for key in values:
        window[key]('')
    return None

def calibration_app(q_from_lsl: mp.Queue(), q_to_lsl: mp.Queue(), markers):
    '''
    This function is the main function for the calibration user interface.
    It takes the queues for the communication with the lsl calibration script and the markers dictionary.
    It starts the lsl calibration script as a process and then starts the calibration user interface.
    It also takes the participant name, the number of trials and the number of blocks as input.
    
    
    Parameters
    ----------
    q_from_lsl : mp.Queue()
        The queue for the communication from the lsl calibration script.
    q_to_lsl : mp.Queue()
        The queue for the communication to the lsl calibration script.
    markers : dict
        The markers dictionary.
    
    Returns
    -------
    None.
    '''

    participant = input("Participant: ")

    # save participant number
    with open('./Results/participant.txt', 'w') as f:
        f.write(participant)
    n_trials = 10
    max_bloc = 4
    sound = False


    info_txt = "./UI/calibration_info.txt"
    welcome_txt = "./UI/calibration_welcome.txt"
    pause_txt = "./UI/pause.txt"
    end_txt = "./UI/calibration_end.txt"
    relax_txt = "./UI/calibration_relax.txt"

    sg.theme('Reddit')
    sg.set_options(font='Arial', keep_on_top=True)

    # Create the window
    layout = [[sg.Column(info_window(welcome_txt, 'ok_welcome'), key = 'welcome'),
               sg.Column(info_window(info_txt, 'ok_info'), key = 'info', visible = False),
              sg.Column(info_window(pause_txt, 'ok_pause'), key= 'pause', visible = False)]]

    welcome, trial, relax, end =  sg.Window('Main_window', layout, resizable=True, element_justification='center', finalize= True, size=(1280,650)), None, None, None
    welcome.maximize()

    # start lsl calibration script as a process
    p = mp.Process(target = lsl_calib, args=(q_from_lsl, q_to_lsl, markers))
    p.start()
    start_time = time.time()
    

    # count trials and blocs
    t_count = 0
    n_bloc = 0

    while True:
        window, event, values = sg.read_all_windows()

        # if n_bloc == max_bloc:
        #     break
        
        if event == 'ok_end':
            break 

        if window == welcome: 
            if event == sg.WIN_CLOSED or event == 'Exit':
                break

            if event == 'ok_pause':
                welcome.hide()
                welcome['pause'].update(visible=False)
                trial = sg.Window('trial', trial_window(), element_justification='center', finalize=True, resizable=True, return_keyboard_events=True)
                n1 = randint(11, 100)
                n2 = randint(11, 100)
                result = n1+n2
                trial['eq'].update(f'{n1} + {n2} = ')
                trial.maximize()

            if event == 'ok_info':
                msg = markers['MA started']
                q_to_lsl.put(msg, block=False)
                welcome['info'].update(visible=False)
                welcome.hide()
                trial = sg.Window('trial', trial_window(), element_justification='center', finalize=True, resizable=True, return_keyboard_events=True)
                n1 = randint(11, 100)
                n2 = randint(11, 100)
                result = n1+n2
                trial['eq'].update(f'{n1} + {n2} = ')
                trial.maximize()
            
            if event == 'ok_welcome':
                welcome['welcome'].update(visible=False)
                welcome.hide()
                relax = relax_window(relax_txt, 'ok_from_relax')
                relax.maximize()

                
        
        if window == trial:
            while True:
                event_t, values_t = trial.read()

                if t_count <= n_trials and values_t.get('answer') == str(result):
                    # clear_input()
                    # clear input:
                    for key in values:
                        window[key]('')
                    n1 = randint(11, 100)
                    n2 = randint(11, 100)
                    result = n1+n2
                    trial['eq'].update(f'{n1} + {n2} = ')
                    t_count+=1
                
                if t_count > n_trials and n_bloc < max_bloc:
                    trial.close()
                    n_bloc +=1 
                    t_count = 0

                    welcome.un_hide()
                    welcome['pause'].update(visible=True)
                    welcome.maximize()
                    break

                if t_count > n_trials and n_bloc == max_bloc:
                    msg = markers['MA ended']
                    q_to_lsl.put(msg, block=False)
                    trial.close()
                    end = sg.Window('trial', info_window(end_txt, 'ok_end'), element_justification='center', finalize=True, resizable=True, return_keyboard_events=True)
                    end.maximize()
                    break


                if event_t == 'ok_trial':
                    continue
                
                if event_t == sg.WIN_CLOSED or event == 'Exit':
                    break

        if window == relax:
                timer_running = False
                timer_paused = False
                time_remaining = -1

                # Event loop
                while True:
                    event, values = window.read(timeout=1000) # wait up to 1s for an event
                    if event in (sg.WINDOW_CLOSED, 'Exit'):
                        break
                    elif event == 'Start':
                        msg = markers['baseline started']
                        q_to_lsl.put(msg, block=False)
                        timer_running = True
                        time_remaining = 120
                        timer_paused = False
                        relax['Start'].update(visible = False)

                    if timer_running and not timer_paused:
                        time_remaining = time_remaining - 1
                        # print(time_remaining)
                        
                        # seconds = time_remaining % 60
                        # print(seconds)
                        time_string = f'{time_remaining:02d}'
                        relax['-TIMER-'].update(time_string)
                        
                    if time_remaining == 0 and sound == False:
                        timer_running = False
                        timer_paused = True
                        audio = 'C:/Users/NIRS/Documents/GitHub/EEG_BCI_final_project/ding.mp3'
                        # media = vlc.MediaPlayer(audio)
                        # media.play()
                        os.system("mpg123 " + audio)
                        # playsound()
                    
                        sound = True
                        relax['ok_from_relax'].update(visible = True)
                        msg = markers['baseline ended']
                        q_to_lsl.put(msg, block=False)
                        

                    if event == 'ok_from_relax':

                        relax.close()
                        welcome.un_hide()
                        welcome['info'].update(visible=True)
                        welcome.maximize()
                        break





if __name__ == "__main__":

    markers = {'baseline started': 0,
            'baseline ended': 99,
            'MA started': 100,
            'MA ended': 200}

    q_from_lsl = mp.Queue()
    q_to_lsl = mp.Queue()
    
    calibration_app(q_from_lsl, q_to_lsl, markers)
    




        
                



                
                



