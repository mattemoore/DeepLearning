import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import process

# np.set_printoptions(threshold=np.inf)

# get data
X, T = process.process()
N = len(X)
D_X = X.shape[1]
NUM_EMOTIONS = len(process.EMOTIONS)

# multinomial logistic regression

# STEP1: one-hot encode dependant variable
zeros = np.zeros((N, 1), dtype=int)
for i in range(NUM_EMOTIONS):
    T = np.concatenate((T, zeros), axis=1)
D_T = T.shape[1]

for row in range(N):
    emotion = T[row, 0]
    T[row, emotion + 1] = 1

# remove emotion column now that it has been encoded
# the last encoded (Neutral-6) becomes our reference variable
T = np.delete(T, [0, NUM_EMOTIONS], axis=1)


def sanity_check_one_hot_encoding():
    r = np.random.random_integers(0, N - 1)
    print('Emotion value', get_emotion(r))
    for i in range(len(T[r])):
        if T[r, i] == 1:
            print('Emotion index (should match value):', i)
    print('Emotion name', process.EMOTIONS[get_emotion(r)])
    process.show_image(X[r].tolist())


def get_emotion(row_index):
    row = T[row_index]
    for i in range(T.shape[1]):
        if row[i] == 1:
            return i
    return NUM_EMOTIONS - 1


sanity_check_one_hot_encoding()

# STEP2: Create NUM_EMOTIONS -1 logistic regression models


# TODO: all of these methods need to be modded to accept one specific column of T
# so we can have one model per column
''' T_old = [1
            0
            0
            0
            ...]
    T_new = [1, 0, 0, ...]
            [0, 1, 0, ...]
            [...]
'''


def classification_rate(T, P):
    return np.mean(T == P)


def calculate_cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E


# bias term
ones = np.ones((N, 1))
X = np.concatenate((X, ones), axis=1)

T_index = 0

# random weights
w = np.random.randn(D_X + 1)
b = 0

Y = expit(X.dot(w) + b)

learning_rate = 0.001
errors = []
for t in range(10000):
    cost = calculate_cross_entropy(T[:, T_index], Y)
    if (t % 100 == 0):
        print('Cost:', cost)
    errors.append(cost)

    w += learning_rate * ((X.T.dot(T[:, T_index] - Y)))
    b -= learning_rate * (Y - T[:, T_index]).sum()

    Y = expit(X.dot(w) + b)
    print('X', X)
    print('W', w)
    print('Y', Y)

plt.plot(errors)
plt.title('Cross-entroy per iteration')
plt.show()

print('Final w', w)
print('Classification rate', classification_rate(T[T_index], np.round(Y)))
