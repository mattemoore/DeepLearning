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
import datetime as dt

IMG_WIDTH = 48
EMOTIONS = ['Angry', 'Disgust', 'Fear',
            'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTIONS_IDX = 0
PIXELS_IDX = 1


# X is array of pixels comprising each image
# T is the labelled emotion for each image
def process():

    X, T = load_data()

    # sanity checks for loading and parsing data:
    # print(X.shape, T.shape)
    # (35887, 2304), (35887, 1)
    # r = np.random.random_integers(0, N - 1)
    # print(EMOTIONS[T[r, 0]])
    # show_image(X[r, :].tolist())

    return X, T


def load_data():
    print('Loading data files...')
    start_time = dt.datetime.now()

    data_frame = pd.read_csv('../input/fer2013.csv',
                             usecols=[PIXELS_IDX, EMOTIONS_IDX])

    print('Loading completed in {0} seconds'.format(
          (dt.datetime.now() - start_time).seconds))

    print('Parsing data files...')
    start_time = dt.datetime.now()

    T = np.array(data_frame.iloc[:, EMOTIONS_IDX], dtype=int)
    T = np.reshape(T, (len(T), 1))

    N = len(data_frame)
    X = np.zeros((N, IMG_WIDTH * IMG_WIDTH), dtype=int)

    for i, row in enumerate(data_frame.itertuples(index=False, name=None)):
        pixels = row[PIXELS_IDX].split(' ')
        X[i] = pixels

    print('Parsing completed in {} seconds'.format(
          (dt.datetime.now() - start_time).seconds))

    return X, T


def show_image(int_list):
    Image.frombytes('L', (IMG_WIDTH, IMG_WIDTH), bytes(int_list)).show()
