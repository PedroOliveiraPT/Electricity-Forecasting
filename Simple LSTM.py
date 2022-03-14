#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px # to plot the time series plot
from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf 


# In[3]:


df = pd.read_csv("data/mongo_data.csv", index_col='ts')
df = df.drop('Unnamed: 0', 1)
#Remove cols with the same value

def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)

df = df.loc[:,np.invert(unique_cols(df))]
print(df.shape)
df.tail()


# In[4]:


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


# In[5]:


history_window = 60 #seconds
prediction_window = 1
values = df.values
# frame as supervised learning
reframed = series_to_supervised(values, history_window, prediction_window)
# drop columns we don't want to predict
print(reframed.head())


# In[18]:


timesteps = prediction_window
features = (history_window + prediction_window) * 39


# In[19]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import LayerNormalization

# design network
model = Sequential()
model.add(LayerNormalization(axis=0))
model.add(LSTM(100, input_shape=(timesteps, features)))
model.add(Dense(39))
model.compile(loss='mae', optimizer='adam')


# In[20]:


# split into train and test sets
values = reframed.values
n_train_seconds = 2*24*60*60 #2 days
n_test_seconds =  3*24*60*60 #3 days
train = values[:n_train_seconds, :]
test = values[n_train_seconds:n_test_seconds, :]
# split into input and outputs
train_X, train_y = train[:, :-39], train[:, -39:]
test_X, test_y = test[:, :-39], test[:, -39:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[21]:


history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[25]:


history_results = pd.DataFrame([history.history['loss'], history.history['val_loss']], columns=['Loss', 'Validation Loss'])


# In[27]:


history_results.to_csv('results/Simple_LSTM_60secs.csv')
model.save('models/Simple_LSTM_60secs')


# In[ ]:




