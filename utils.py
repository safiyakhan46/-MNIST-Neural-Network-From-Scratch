import numpy as np
import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def normalize(X):
    return X / 255.0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y