#!/usr/bin/env python
# coding: utf-8

# Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# create network
def create_model(rate, features, timesteps=1):
    model = Sequential()
    model.add(LSTM(10, input_shape=(timesteps, features)))
    model.add(Dropout(rate))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    return model
    
