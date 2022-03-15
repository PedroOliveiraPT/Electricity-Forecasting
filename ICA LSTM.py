#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import tensorflow as tf 


# In[2]:


df = pd.read_csv("data/mongo_data.csv", index_col='ts')
df = df.drop('Unnamed: 0', 1)
#Remove cols with the same value

def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)

df = df.loc[:,np.invert(unique_cols(df))]
print(df.shape)
df.tail()


# In[3]:


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


# In[4]:


scaler = MinMaxScaler()
d = scaler.fit_transform(df)
scaled_df = pd.DataFrame(d, columns=df.columns, index=df.index)
scaled_df.head()


# In[6]:


from sklearn.decomposition import FastICA
pcn=6
ica = FastICA(n_components=pcn)

ds = ica.fit_transform(scaled_df)


# In[7]:


history_window = 60 #seconds
prediction_window = 1
# frame as supervised learning
reframed = series_to_supervised(ds, history_window, prediction_window)
# drop columns we don't want to predict
print(reframed.head())


# In[13]:


values = reframed.values
len_values = values.shape[0]
# split into train and test sets 
n_train_seconds = int(0.7*len_values) #70% dos valores
n_test_seconds =  int(0.9*len_values) #20% dos valores
train = values[:n_train_seconds, :]
test = values[n_train_seconds:n_test_seconds, :]
# split into input and outputs
train_X, train_y = train[:, :-pcn], train[:, -pcn:]
test_X, test_y = test[:, :-pcn], test[:, -pcn:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# design network
model = Sequential()
model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.25))
model.add(Dense(pcn))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)


# In[ ]:


history_results = pd.DataFrame(list(zip(history.history['loss'], history.history['val_loss'])), columns=['Loss', 'Validation Loss'])
history_results.to_csv('results/ICA_Drop_LSTM_60secs_MSE_loss.csv')
model.save('models/ICA_Drop_LSTM_60secs_MSE.h5')


# In[ ]:


from sklearn.metrics import mean_squared_error

#Test for the day after
n_rtest_seconds =  int(0.1*24*60*60) #10%
rtest = values[-n_rtest_seconds:, :]

rtest_X, rtest_y = rtest[:, :-pcn], rtest[:, -pcn:]
rtest_X = rtest_X.reshape((rtest_X.shape[0], 1, rtest_X.shape[1]))
# make a prediction
yhat = model.predict(rtest_X)


# In[ ]:


Xhat_ica = ica.inverse_transform(yhat)


# In[ ]:


prediction_results = pd.DataFrame(Xhat_ica)
prediction_results.to_csv('results/ICA_Drop_LSTM_60secs_MSE_prediction.csv')


# In[ ]:




