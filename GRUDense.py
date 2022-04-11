#!/usr/bin/env python
# coding: utf-8

# Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU

# create network
def create_model(dense_cell, features, timesteps=1):
    model = Sequential()
    model.add(GRU(10, input_shape=(timesteps, features)))
    model.add(Dense(dense_cell))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    return model
