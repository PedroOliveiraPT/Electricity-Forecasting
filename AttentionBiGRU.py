import numpy as np
from keras.models import *
from keras.layers import *
from keras import backend as K

class attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)

def create_model(cells, rate, features, timesteps=1):
    model = Sequential()
    model.add(Bidirectional(GRU(cells, input_shape=(timesteps, features))))
    model.add(attention(return_sequences=True)) # receive 3D and output 3D
    model.add(Dropout(rate))
    model.add(Dense(1, activation='sigmoid'))

    