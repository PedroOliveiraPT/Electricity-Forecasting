import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional

def create_model(features, timesteps=1):
    model = Sequential()
    model.add(Dense(20))
    model.add(Bidirectional(LSTM(20)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))