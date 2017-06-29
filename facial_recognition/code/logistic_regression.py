import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import process

# np.set_printoptions(threshold=np.inf)

# get data
X, T = process.process()
N = len(X)
NUM_EMOTIONS = len(process.EMOTIONS)

# multinomial logistic regression

# step1: one-hot encode dependant variable
zeros = np.zeros((N, 1), dtype=int)
for i in range(NUM_EMOTIONS):
    T = np.concatenate((T, zeros), axis=1)

for row in range(N):
    emotion = T[row, 0]
    T[row, emotion + 1] = 1

# remove emotion column now that it has been encoded
# remove redundant last column
T = np.delete(T, [0, NUM_EMOTIONS], axis=1)


def sanity_check_one_hot_encoding():
    r = np.random.random_integers(0, N - 1)
    print('Emotion value', get_emotion(r))
    print('Emotion name', process.EMOTIONS[get_emotion(r)])
    process.show_image(X[r, :].tolist())


def get_emotion(row_index):
    row = T[row_index]
    for i in range(T.shape[1]):
        if row[i] == 1:
            return i
    return NUM_EMOTIONS - 1


sanity_check_one_hot_encoding()
