import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, concatenate, Bidirectional


z=5912
for x in range(1,10000):
    if z % x == 0:
        print(x)