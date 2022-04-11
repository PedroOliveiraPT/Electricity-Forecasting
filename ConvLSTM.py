
# Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Conv1D

# create network
def create_model(n_stack, features, timesteps=1):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[timesteps, features]))
    for i in range(n_stack-1):
        model.add(Conv1D(filters=32, kernel_size=3,
                      strides=1, padding="causal",
                      activation="relu"))
    model.add(LSTM(10, return_sequences=True))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    return model
