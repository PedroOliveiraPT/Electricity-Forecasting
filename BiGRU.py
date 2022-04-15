#!/usr/bin/env python
# coding: utf-8

# Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout, Bidirectional

# create network
def create_model(cells, rate, features, timesteps=1):
    model = Sequential()
    model.add(Bidirectional(GRU(cells, input_shape=(timesteps, features))))
    model.add(Dropout(rate))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    return model
    
