#!/usr/bin/env python
# coding: utf-8
# Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Flatten

# create network
def create_model(n_stack, features, timesteps=1):
    model = Sequential()
    model.add(Dense(20))
    model.add(GRU(10, input_shape=(timesteps, features), return_sequences=True))
    for i in range(n_stack-1):
        model.add(Dense(20))
        model.add(GRU(10, return_sequences=True))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
     
    return model
