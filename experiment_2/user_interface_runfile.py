import PySimpleGUI as sg
# import PySimpleGUIQt as sg
import pandas as pd
import csv
import vlc
from sys import platform as PLATFORM
import random
import re
from playsound import playsound
import time
from pathlib import Path
from colour import Color
import queue
from datetime import datetime

import multiprocessing as mp
from realtime_script_runfile import *
import threading

QUESTION_FONT = 'Arial 16 bold'


def question_window(questions_file, video):
    '''This function receives the path for a csv file for the questions and  a path for a mp4 video, 
    and creates a page with questions. 
    This layout has 2 questions per page. '''
    df = pd.read_csv(questions_file, encoding = 'cp1252')
    df_vid = df[df['video'] == video]

    #shuffle the order of questions:
    df_vid = df_vid.sample(frac=1).reset_index(drop=True)

    questions =  list(df_vid['question'])
    layout_list = []
    question_page = []
    layout = []
    q_per_page = 2

    #column numbers of answers: 
    column = [2,3,4,5]
    page = 0

    for i in range(0, len(questions)):
        layout.append([sg.Text(questions[i], font=QUESTION_FONT, k=questions[i])])
        random.shuffle(column)
        answers = df_vid.iloc[i, column]
        correct = df_vid.iloc[i, 6]
        for j in range(0,len(answers)):
            if answers[j] == correct:
                layout.append([sg.Radio(answers[j], f'answers_for_{i}', k = f'{i}.{j}'+'.fact')])
            else:
                layout.append([sg.Radio(answers[j], f'answers_for_{i}', k = f'{i}.{j}'+'.lie')])

        if (i%2)!=0:
            layout.append([sg.Text('', size=(1,5))])
            layout.append([sg.Button('continue', size=(8, 1), pad=(1, 1), visible=True, k=f'continue_{page}')])
            page+=1
    
    #split into multiple lists:
    # How many elements each list should have?
    n = 12

    # using list comprehension
    layout_list = [layout[i * n:(i + 1) * n] for i in range((len(layout) + n - 1) // n )]

    # make into a list of columns, each column is a page layout with two questions
    for i in range(0,len(layout_list)):
        question_page.append(sg.Column(layout_list[i], key = f'page_{i}', visible = False))

    return [question_page]
    # return sg.Window('questions', layout, element_justification='center', finalize=True, resizable=True)

def between_videos_window():
    '''this function returns a layout with a page for two simple questions following the video:
    how familiar were you with the topic, and what was your attention level'''
    layout = [[sg.Text('How familiar were you with the topic of the video prior to watching it?', font=QUESTION_FONT)],
                                [sg.Radio('Not at all familiar', 'familiar', key='-2_familiar')],
                                [sg.Radio('Slightly familiar', 'familiar', key='-1_familiar')],
                                [sg.Radio('Somewhat familiar', 'familiar', key='0_familiar')],
                                [sg.Radio('Moderately familiar ', 'familiar',  key = '1_familiar')],
                                [sg.Radio('Extremely familiar', 'familiar', key = '2_familiar')],
                                [sg.Text('', size= (1,1))],
                                [sg.Text('How would you estimate your attention level during the video lecture?', font=QUESTION_FONT)],
                                [sg.Text('very low'), sg.Slider(range=(1,5), default_value = 3, orientation='horizontal', tick_interval=1, key = 'attention'),
                                 sg.Text('very high')],
                                 [sg.Text('', size= (1,5))],
                                 [sg.Button('Continue', key = 'continue_between')]
                                ]
    return layout

def video_window ():
    '''This function returns a window with all the elements ready for presenting a video.'''
    layout = [[sg.Image('', size=(300, 170), key='-VID_OUT-')],
              [sg.Text(' ', key='feedback', size=(200,1), border_width =3)],
            #   [sg.ProgressBar(100, orientation='h', size=(100, 20), key='feedback', visible = False, bar_color = ('#E8E8E8', '#E8E8E8'), border_width = 1, style = 'xpnative')],
              [sg.Text('If you are ready to start watching the video, click ', font='Arial 20', key='ready'),
               sg.Button('start', size=(6, 1), pad=(1, 1), visible=True), 
               sg.Button('play', size=(6, 1), pad=(1, 1), visible=False), 
               sg.Button('pause', size=(6, 1), pad=(1, 1), visible=False),
               sg.Button('continue', size=(8, 1), pad=(1, 1), visible=False)],
              [sg.Text('', key='-MESSAGE_AREA-')]]

    return sg.Window('Lecture', layout, element_justification='center', finalize=True, resizable=True, return_keyboard_events=True, use_default_focus=False)

def welcome_window():
    '''This function creates the layout for the welcome window'''
    layout = [[sg.Text('Welcome!', font='Arial 20'), ],
              [sg.Text('We appreciate your participation.', font='Arial 20'), ],
              [sg.Text('We will first start with a few short questions.', font='Arial 20'), ],
              [sg.Text('', size=(1,5))],
              [sg.Button('Ok'), sg.Button('Exit')]]

    return layout

def txt_window(txt, key_txt):
    '''This function recieves the path to a txt file and a string indicating the key for the 'ok' button,
    to use later in relevant events.
    It returns the layout of a window with the text added and an Ok button.'''
    my_file = open(txt, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    layout = []
    for line in data_into_list:
        layout.append([sg.Text(line, font='Arial 20')])
        layout.append([sg.Text('', size=(1,1))])

    layout.append([sg.Button('Ok', key = key_txt, visible = False)])

    return layout

def relax_window(txt, key_txt):
    '''This function returns a window that shows a timer counting down when the 'Start' button is clicked. '''
    my_file = open(txt, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    my_file.close()
    layout = []
    for line in data_into_list:
        layout.append([sg.Text(line, font='Arial 20')])
        layout.append([sg.Text('', size=(1,1))])

    layout.append([sg.Text('30', font=('Helvetica', 48), key='-TIMER-')])
    layout.append([sg.Text('', size=(1,5))])
    layout.append([sg.Button('Start')])
    layout.append([sg.Button('Ok', key = key_txt, visible = False)])

    return sg.Window('Calibration', layout, element_justification='center', finalize=True, resizable=True)


def form_window():
    '''This function returns a layout for basic data entry form.'''
    layout = [
        [sg.Text('Please fill out the following fields:', font=QUESTION_FONT)],
        [sg.Text('', size=(1,1))],
        [sg.Text('Age', size=(3,1), font=QUESTION_FONT), sg.InputText(size = (3,1), key='Age')],
        [sg.Text('', size=(1,1))],
        [sg.Text('Gender', size=(7,1), font=QUESTION_FONT)],
        [sg.Radio('Female', 'Gender', size=(10,1), key='Female')], 
        [sg.Radio('Male', "Gender", size=(10,1), key='Male')],
        [sg.Radio('Non-binary', "Gender", size=(10,1), key='Nonbinary')], 
        [sg.Radio('Prefer not to share', "Gender", size=(16,1), key='prefer_not')],
        [sg.Text('', size=(1,1))],
        [sg.Text('Native Language', size=(15,1), font=QUESTION_FONT),
                                sg.Checkbox('Dutch', key='Dutch'),
                                sg.Checkbox('German', key='German'),
                                sg.Checkbox('English', key='English'),
                                sg.Checkbox('Other: (type here)', key = 'Other'),
                                sg.InputText(key='Other_language', size=(10,1))],
        [sg.Text('', size=(1,1))],
        [sg.Text('Are you a Student?', font=QUESTION_FONT), sg.Radio('Yes', 'Student', size=(10,1), key='Student'), sg.Radio('No', "Student", size=(10,1), key='not_student')],
        [sg.Text("If you are a student, please indicate your level: "), sg.Combo(['Undergraduate', 'Graduate', 'Postgraduate'], key='study_level')],
        [sg.Text("and your study program: "), sg.InputText(key='studies', size=(25,1))],
        [sg.Text('', size=(1,1))],
        [sg.Submit(), sg.Button('< Prev'), sg.Exit()]
    ]
    return layout

def attention_questions():
    '''This function returns a layout for basic data entry form.'''
    col1 = [
        [sg.Text('Please respond to the following questions:', font=QUESTION_FONT)],
        [sg.Text('', size=(1,1))],
        [sg.Text('Are you diagnosed with Attention deficit hyperactivity disorder (ADHD)?', font=QUESTION_FONT)],
        [sg.Radio('Yes', 'adhd', size=(10,1), key='adhd_yes'), sg.Radio('No', "adhd", size=(10,1), key='adhd_no')], 
        [sg.Text('', size=(1,1))],
        [sg.Text('I get distracted easily', font=QUESTION_FONT)],
                                [sg.Radio('Strongly Disagree', 'distract', key='-2_distract')],
                                [sg.Radio('Disagree', 'distract', key='-1_distract')],
                                [sg.Radio('Neither Agree nor Disagree', 'distract', key='0_distract')],
                                [sg.Radio('Agree ', 'distract',  key = '1_distract')],
                                [sg.Radio('Strongly Agree', 'distract', key = '2_distract')],
                                [sg.Text('', size= (1,1))],
        [sg.Text('I often catch myself daydreaming while working/studying', font=QUESTION_FONT)],
                                [sg.Radio('Strongly Disagree', 'daydream', key='-2_daydream')],
                                [sg.Radio('Disagree', 'daydream', key='-1_daydream')],
                                [sg.Radio('Neither Agree nor Disagree', 'daydream', key='0_daydream')],
                                [sg.Radio('Agree ', 'daydream',  key = '1_daydream')],
                                [sg.Radio('Strongly Agree', 'daydream', key = '2_daydream')],
                                [sg.Text('', size= (1,1))]
    ]

    col2 = [
        [sg.Text('I have difficulty concentrating on one task', font=QUESTION_FONT)],
                                [sg.Radio('Strongly Disagree', 'concentrate', key='-2_concentrate')],
                                [sg.Radio('Disagree', 'concentrate', key='-1_concentrate')],
                                [sg.Radio('Neither Agree nor Disagree', 'concentrate', key='0_concentrate')],
                                [sg.Radio('Agree ', 'concentrate',  key = '1_concentrate')],
                                [sg.Radio('Strongly Agree', 'concentrate', key = '2_concentrate')],
                                [sg.Text('', size= (1,1))],
        [sg.Text('Feedback on my attention level can help me focus better', font=QUESTION_FONT)],
                                [sg.Radio('Strongly Disagree', 'att_feedback', key='-2_feedback')],
                                [sg.Radio('Disagree', 'att_feedback', key='-1_feedback')],
                                [sg.Radio('Neither Agree nor Disagree', 'att_feedback', key='0_feedback')],
                                [sg.Radio('Agree ', 'att_feedback',  key = '1_feedback')],
                                [sg.Radio('Strongly Agree', 'att_feedback', key = '2_feedback')],
                                [sg.Text('', size= (1,1))],
        [sg.Text('', size=(1,1))]
    ]
    layout = [[sg.Column(col1), sg.Column(col2)],
        [sg.Submit(key = 'submit_attention'), sg.Exit()]]
    return layout

def the_thread(window: sg.Window, q_from_lsl: mp.Queue, ei_queue: queue.Queue, video_event: threading.Event):
    '''This function runs in a separate thread and checks if there is a message in the queue from the LSL stream.'''
    while True:
        try:
            if not q_from_lsl.empty() and video_event.is_set():
                msg = q_from_lsl.get_nowait()
                print(f'msg accepted {msg}')
                ei_queue.put(msg)
                #window.write_event_value('-THREAD-', msg)
                
        except Exception as e:
            print(e)
            pass
def improve_data(df, conditions):
    '''this function takeus the data filled up in the short form, and makes it more readble for the analysis.'''
    if df['Female'].values[0] == True:
        df['Gender'] = ['Female']
    elif df['Male'].values[0] == True:
        df['Gender'] = ['Male']
    elif df['Nonbinary'].values[0] == True:
        df['Gender'] = ['Nonbinary']
    elif df['prefer_not'].values[0] == True:
        df['Gender'] = ['Prefer not to share']
    
    languages = ['Dutch', 'German', 'English', 'Other']

    df['Native_language'] = ['']
    for language in languages:
        if df[language].values[0] == True:
            df['Native_language'] = [df['Native_language'].values[0] + ', ' + language]
        if language == 'Other':
            if df['Other_language'].values[0] != '':
                df['Native_language'] = [df['Native_language'].values[0] + ', ' + df['Other_language'].values[0]]
    
    df['video1'] = [conditions[0]]
    df['video2'] = [conditions[1]]
    df['video3'] = [conditions[2]]
    df.drop(['Female', 'Male', 'Nonbinary', 'prefer_not', 'Dutch', 'German', 'English', 'Other', 'Other_language', 'attention', '-2_familiar', '-1_familiar', '0_familiar', '1_familiar', '2_familiar'], axis=1, inplace=True)

def get_false_feedback(window, n_epochs):
    '''This function returns a list of false feedback for the given window size.'''
    #read the false feedback csv file to list
    file = open('./UI/false_feedback.csv', "r")
    false_feedback = list(csv.reader(file, delimiter=","))[0]
    file.close()
    
    len_window = round((60 * 13) / (window * n_epochs))
    print(len(false_feedback))
    print(len_window)
    max_index = len(false_feedback) - len_window
    
    # randomly select an index to start with
    start_index = random.randint(0, max_index)

    return false_feedback[start_index:start_index+len_window]

def app(q_from_lsl: mp.Queue, q_to_lsl: mp.Queue, markers: dict):
    questions_file = './UI/questions.csv'
    explainer_file = "./UI/experiment_info.txt"
    relax_file = './UI/relax.txt'
    end_file = './UI/end.txt'
    video_path = ['./videos/arthistory.mp4', './videos/funct.mp4', './videos/lang.mp4']
    thread = False #variable to check if thread is running
    date_now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    demo_file = './Results/' + date_now + '.csv'
    answer_file = './Results/' + date_now + '_answers.csv'
    
    random.shuffle(video_path)

    window_sec = 5 
    n_epochs = 3

    # prepare colors:
    red = Color("red")
    yellow = Color("yellow")
    green = Color("Green")
    n = 100
    colors = list(red.range_to(yellow,int(n/2)))
    colors.extend(list(yellow.range_to(green,int(n/2))))

    relax = False
    sound = False
    timer_running = False
    timer_paused = False
    time_remaining = -1

    # prepare condition
    # NF = no feedback
    # F = feedback
    # FF = false feedback
    conditions = ['NF', 'F', 'FF']
    # shuffle conditions
    random.shuffle(conditions)

    # save condition information to txt file
    with open('./Results/' + date_now + '_conditions.txt', 'w') as f:
       f.write("\n".join(str(item) for item in conditions))
    
    sg.theme('Reddit')
    # sg.set_options(font='Arial', keep_on_top=True)
    sg.set_options(font='Arial')
    layout = [[sg.Column(welcome_window(), key = 'welcome'),
               sg.Column(form_window(), key = 'form', visible = False),
               sg.Column(txt_window(explainer_file, 'ok_from_explain'), key= 'explainer', visible = False),
               sg.Column(txt_window(relax_file, 'ok_from_relax'), key= 'relax', visible = False),
               sg.Column(attention_questions(), key= 'attention_q', visible = False),
               sg.Column(txt_window(end_file, 'ok_from_end'), key = 'end', visible = False),
               sg.Column(between_videos_window(), key= 'between', visible = False)]]
    
    main, video, questions, relax =  sg.Window('Main_window', layout, resizable=True, element_justification='center', finalize= True, size=(1280,650)), None, None, None
    main.maximize()

    p = mp.Process(target = lsl_main, args=(q_from_lsl, q_to_lsl, markers))
    p.start()

    



    video_count = 0 # counts the video we are at (max 3)
    q_count = 0 # counts the questions- before/after (max 2)
    key = 0 # counts key presses (to send trigger of low attention)

    df = pd.DataFrame()
    df_total_answer = pd.DataFrame()
    df_answer = pd.DataFrame()

    video_event = threading.Event()


    while True:
        window, event, values = sg.read_all_windows()

        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        if window == main:
            if event == 'Ok':
                # if 'ok' from welcome --> move to basic data entry window 
                window['welcome'].update(visible=False)
                window['form'].update(visible=True)

            if event == '< Prev':
                # if back from form --> move to welcome window
                window['form'].update(visible=False)
                window['welcome'].update(visible=True)
                
            if event == 'Submit':
                # values of basic form submitted, save and then --> explanation window 
                df = pd.DataFrame.from_dict([values])
                print(df)
                improve_data(df, conditions)
                print(df)
                df.to_csv(demo_file, index=False)
                window['form'].update(visible=False)
                window['explainer'].update(visible=True)
                main['ok_from_explain'].update(visible=True)


            if event == 'ok_from_explain' or event == 'ok_from_relax' and video_count < 3: 
                # from explanation and from calibration --> question window
                main.hide()
                video_name = video_path[video_count].replace('.mp4', '').replace('./videos/', '')
                questions = sg.Window('questions', question_window(questions_file, video_name), element_justification='center', finalize=True, resizable=True)
                questions['page_0'].update(visible=True)
                questions.maximize()

            if event == 'continue_between' and video_count < 3:
                df['attention_'+str(video_count)] = values['attention'] # save attention score
                for i in range(-2, 3): # save familiarity scores
                    if values[str(i)+'_familiar'] == True:
                        df['familiar_'+str(video_count)] = i
                
                print(df)
                keys = ['attention', '-2_familiar', '-1_familiar', '0_familiar', '1_familiar', '2_familiar']

                for key in keys:
                    values[key] = None
                    if key == 'attention':
                        values[key] = 3
                    if key == '0_familiar':
                        values[key] = True
                

                main.hide()
                relax = relax_window(relax_file, 'ok_from_relax')
                sound = False
                relax.maximize()

            if event == 'submit_attention':
                main['attention_q'].update(visible=False)
                for i in range(-2, 3): # save distract score
                    if values[str(i)+'_distract'] == True:
                        df['distract'] = i

                for i in range(-2, 3): # save daydream score
                    if values[str(i)+'_daydream'] == True:
                        df['daydream'] = i
                
                for i in range(-2, 3): # save concentration score
                    if values[str(i)+'_concentrate'] == True:
                        df['concentrate'] = i
                
                for i in range(-2, 3):
                    if values[str(i)+'_feedback'] == True:
                        df['feedback'] = i
                main['end'].update(visible=True)
                df_answer.to_csv(answer_file, index=False)
                df.to_csv(demo_file, index=False)

            if event == 'continue_between' and video_count == 3 or event == sg.WIN_CLOSED:
                msg = markers['last video ended']
                q_to_lsl.put(msg)
                df['attention_'+str(video_count)] = values['attention'] # save attention score
                for i in range(-2, 3): # save familiarity scores
                    if values[str(i)+'_familiar'] == True:
                        df['familiar_'+str(video_count)] = i
                
                print(df)
                # move to summary window
                main['between'].update(visible=False)
                main['attention_q'].update(visible=True)


            # if event == 'start':
            #     # start timer 
            #     timer_running = True
            #     time_remaining = 10 # 5 minutes
            #     timer_paused = False

            # if timer_running and not timer_paused:
            #     time_remaining = time_remaining - 1
            #     time_string = f'{time_remaining:02d}'
            #     main['-TIMER-'].update(time_string)
        
            # if time_remaining == 0:
            #     timer_running = False
            #     timer_paused = True
                

        if window == questions:
            q_count+=1
            end_video_sent = False
            while True:
                event_q, values_q = questions.read()

                if event_q in ['continue_0', 'continue_1', 'continue_2']:
                    current_page = int(re.findall(r'\d+', event_q)[0])                        
                    questions[f'page_{current_page}'].update(visible=False)
                    questions[f'page_{current_page+1}'].update(visible=True)

                if event_q == 'continue_3':
                    count_correct = 0
                    for answer_key in values_q:
                        if 'fact' in answer_key:
                            if values_q[answer_key] == True:
                                count_correct +=1
                    
                    # correct_answer = {'video.'+str(video_count)+'.'+str(q_count): [count_correct]}
                    print('correct answers: ' + str(count_correct))
                    df_answer['video.'+str(video_count)+'.'+str(q_count)] = [count_correct]
                    # print(values_q)
                    # df_total_answer =  pd.concat([df_total_answer , df_answer])
                    # print(df_answer)
                    questions.hide()

                    if q_count == 1:
                        video = video_window()
                        video.maximize()
                        break
                    else:
                        q_count = 0
                        video_count+=1
                        questions.close()
                        # move from questions to --> relax window between videos
                        main['explainer'].update(visible=False)
                        main['between'].update(visible=True)
                        main.un_hide()
                        main.maximize()
                        break

                        # if video_count < 3: 
                        #     main['explainer'].update(visible=False)
                        #     main['between'].update(visible=True)
                        #     # relax = relax_window(relax_file, 'ok_from_relax')
                        #     # sound = False
                        #     # relax.maximize()
                        #     break
                        # else:
                        #     main.un_hide()
                        #     main.maximize()
                        #     main['end'].update(visible=True)

                if event_q == sg.WIN_CLOSED:
                    questions.close()
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
                    # start timer
                    msg = markers['between video calibartion start']
                    q_to_lsl.put(msg)
                    timer_running = True
                    time_remaining = 10 
                    timer_paused = False
                    relax['Start'].update(visible = False)

                if timer_running and not timer_paused:
                    time_remaining = time_remaining - 1

                    time_string = f'{time_remaining:02d}'
                    relax['-TIMER-'].update(time_string)
                    
                if time_remaining == 0 and sound == False:
                    timer_running = False
                    timer_paused = True
                    audio = 'ding.mp3'
                    playsound('./' + audio)
                    sound = True
                    relax['ok_from_relax'].update(visible = True)
                    

                if event == 'ok_from_relax' and video_count < 3  : 
                    msg = markers['between video calibration end']
                    q_to_lsl.put(msg)
                    relax.hide()
                    video_name = video_path[video_count].replace('.mp4', '').replace('./videos/', '')
                    questions = sg.Window('questions', question_window(questions_file, video_name), finalize=True,  element_justification='center', resizable=True)
                    questions['page_0'].update(visible=True)
                    questions.maximize()
                    break

            if video_count == 3:
                # move to summary window
                pass



        if window == video:
            # start a thread to read from the queue  
            ei_queue = queue.Queue()
            video_event.set()
            threading.Thread(target = the_thread, args=(window, q_from_lsl, ei_queue, video_event), daemon=True).start()  

            # set condition
            condition = conditions[video_count]
            print('condition: ' + condition)
            
            if condition == 'FF':
                ff_list = get_false_feedback(window_sec, n_epochs)
                i = 0

            video['-VID_OUT-'].expand(True, True) 

            inst = vlc.Instance()
            list_player = inst.media_list_player_new()
            media_list = inst.media_list_new([])
            list_player.set_media_list(media_list)
            
            player = list_player.get_media_player()

            player.set_hwnd(video['-VID_OUT-'].Widget.winfo_id())

            media_list.add_media(video_path[video_count])
            list_player.set_media_list(media_list)

            if event == 'start':
                msg = markers['video start']
                q_to_lsl.put(msg)
                list_player.play()
                video['ready'].update(visible = False)
                video['start'].update(visible = False)
                video['play'].update(visible = True)
                video['pause'].update(visible = True)
                video['feedback'].update(visible = True)
                

            while True:
                event1, values1 = video.read(timeout=1000)
                if event1 == 'play':
                    list_player.play()
                    msg = markers['video resumed']
                    q_to_lsl.put(msg)
                    
                if not ei_queue.empty():
                    ei = int(ei_queue.get())
                    print('message accepted from thread: ', ei, ' yay')
                    if ei < 100 and condition == 'F':
                        video['feedback'].update(background_color = colors[ei].hex)
                        # video['feedback'].UpdateBar(ei)
                        # video['feedback'].update(bar_color = (colors[ei].hex, '#E8E8E8'))
                    if condition == 'FF':
                        ei = int(ff_list[i])
                        print(ei)
                        video['feedback'].update(background_color = colors[ei].hex)
                        i+=1

                if event1 == 'pause':
                    list_player.pause()
                    msg = markers['video paused']
                    q_to_lsl.put(msg)
                
                # if list_player.is_playing():
                #     video['-MESSAGE_AREA-'].update("{:02d}:{:02d} / {:02d}:{:02d}".format(*divmod(player.get_time()//1000, 60),
                #                                                         *divmod(player.get_length()//1000, 60)))
                
                if list_player.get_state() == vlc.State.Ended or event1 == 'q':
                    list_player.stop()
                    if end_video_sent == False:
                        msg = markers['video end']
                        q_to_lsl.put(msg)
                        end_video_sent = True
                    video['play'].update(visible = False)
                    video['pause'].update(visible = False)
                    video['continue'].update(visible = True)
                    video_event.clear()



                # if event1 == ' ':
                #     msg = markers['attention']
                #     q_to_lsl.put(msg)
                    # video['feedback'].update(background_color = colors[c].hex)

                # if event1 == '-THREAD-':
                #     ei = int(values1[event1])
                #     print('message accepted from thread: ', ei, ' yay')
                #     if ei < 100 :
                #         video['feedback'].UpdateBar(ei)
                #         video['feedback'].update(bar_color = (colors[ei].hex, '#E8E8E8'))

            

                    

                if event1 == sg.WIN_CLOSED or event == 'Exit':
                    list_player.stop()
                    video.close()
                    break

                if event1 == 'continue':
                    end_video_sent = False
                    video.close()
                    questions = sg.Window('questions', question_window(questions_file, video_name), finalize=True,  element_justification='center', resizable=True)
                    questions['page_0'].update(visible=True)
                    questions.maximize()
                    break
                
                c = random.randint(1,100) 


if __name__ == "__main__":
    q_from_lsl = mp.Queue()
    q_to_lsl = mp.Queue()

    markers = {'video start': 0,
            'video end': 1,
            'video paused': 2,
            'video resumed': 3,
            'attention': 4,
            'between video calibartion start': 5,
            'between video calibration end': 6,
            'last video ended': 7}
    
    app(q_from_lsl, q_to_lsl, markers)
                


    

