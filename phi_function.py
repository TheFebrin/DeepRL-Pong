import numpy as np
import cv2


def preprocess(image):
    '''
    Preprocessing the raw Atari frames. Converting to grayscale,
    downsampling and cropping image
    
    param: image: 210x160x3 raw Atari frame
    
    return: preprocessed_image: 84x84 grayscale Atari frame
    '''
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    downsampled = cv2.resize(grayscale, (84, 110))
    cropped = downsampled[18:102, :]
    return cropped


def phi(state):
    '''
    Preprocessing to the last 4 frames of a history 
    and stacks them to produce the input to the Q-function
   
    param: state: The last 4 (or less) raw Atari frames
                  In order: oldest to latest
    
    return: phi: The last 4 preprocessed Atari frames
    '''

    if len(state) < 4:
        state = [state[0].copy() for _ in range(4 - len(state))] + state

    preprocessed_state = np.stack(list(map(preprocess, state)), axis=2)
    return preprocessed_state
