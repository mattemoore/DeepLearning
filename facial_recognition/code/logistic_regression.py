import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit 
import process

X, T = process.process()
print(X.shape, T.shape)
