'''
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data/fer2013.csv
35887 rows
1st col is emotion:
    0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
2nd col is pixels of image:
    48x48 graysacle
3rd col is indicator of test or training data:
    ignored as we will randomly split training and test sets in code
'''

import numpy as np
import pandas as pd
from PIL import Image

emotions = ['Angry', 'Disgust', 'Fear',
            'Happy', 'Sad', 'Surprise', 'Neutral']


def process():
    data_frame = pd.read_csv('../input/fer2013.csv', usecols=[0, 1])
    X = np.array(data_frame.iloc[:, 1])
    T = np.array(data_frame.iloc[:, 0])
    return X, T


def show_image_and_print_label(index, X, T):
    '''
    for sanity checks:
    r = np.random.random_integers(0, len(T))
    show_image_and_print_label(r)
    '''
    print(emotions[T[index]])
    image_bytes = [int(i) for i in X[index].split(' ')]
    image = Image.frombytes('L', (48, 48), bytes(image_bytes))
    image.show()
