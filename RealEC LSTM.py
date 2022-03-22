#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[9]:


corr = {
    'RealE_SUM':
        ['RealEc_SUM', # Sum of Consumed Energy 
        'RealEc_L1', # Real Energy Consumed L1
        'RealEc_L2', # Real Energy Consumed L2
        'RealEc_L3', # Real Energy Consumed L3
        'RealE_SUM', # Sum of Real Energy 
        'RealE_L2', # Real Energy L2
        'RealE_L3', # Real Energy L3
        'RealE_L1', # Real Energy L1
        'AE_SUM', # Apparent Energy Sum
        'AE_L1', # Apparent Energy L1
        'AE_L2', # Apparent Energy L2
        'AE_L3', # Apparent Energy L3
        'ReacE_L1']
} #Reactive Energy L1


# In[5]:


df = pd.read_csv("data/mongo_data.csv", index_col='ts')
df = df.drop('Unnamed: 0', 1)
df.index = pd.to_datetime(df.index)


# In[10]:


def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)

df = df.loc[:,np.invert(unique_cols(df))]
df = df[corr['RealE_SUM']]
print(df.shape)
df.tail()


# In[11]:


# Average window
df_2 = df.groupby(np.arange(len(df))//60).mean()
print(df_2.shape)
df_2.head()


# In[13]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
d = scaler.fit_transform(df_2)
scaled_df = pd.DataFrame(d, columns=df_2.columns, index=df_2.index)
scaled_df.head()


# In[15]:


#scaled_df.plot()


# In[16]:


# convert series to supervised learning
def create_supervised_dataset(df, target, feats, n_in=1, n_out=1):
    cols, names = list(), list()
    n_vars = len(feats)
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df[feats].shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[target].shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(1)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(1)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg.values


# In[17]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# create network
def create_model(features, timesteps=1):
    model = Sequential()
    #model.add(LayerNormalization(center=True, scale=True))
    model.add(LSTM(30, input_shape=(timesteps, features)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    return model


# In[19]:


history_window =  30 # 8*15secs = 120secs
prediction_window = 1 #predict 15 secs
for k in corr:
    values = create_supervised_dataset(scaled_df, k, corr[k], n_in=history_window, n_out=prediction_window)
    len_values = values.shape[0]
    # split into train and test sets 
    n_train_seconds = int(0.7*len_values) #70% dos valores
    n_cv_seconds =  int(0.9*len_values) #20% dos valores
    train = values[:n_train_seconds, :]
    cv = values[n_train_seconds:n_cv_seconds, :]
    
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1:]
    cv_X, cv_y = cv[:, :-1], cv[:, -1:]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    cv_X = cv_X.reshape((cv_X.shape[0], 1, cv_X.shape[1]))
    print(train_X.shape, train_y.shape, cv_X.shape, cv_y.shape)    
    model = create_model(train_X.shape[2])
    history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(cv_X, cv_y), verbose=2, shuffle=False)
    history_results = pd.DataFrame(list(zip(history.history['loss'], history.history['val_loss'])), columns=['Loss', 'Validation Loss'])
    history_results.to_csv('results/Spec_Norm_LSTM_'+k+'_history.csv')
    model.save('models/Spec_Norm_LSTM_'+k+'_model.h5')

    #Test for the day after
    print("Starting Test", k)
    n_test_seconds =  int(0.1*len_values) #10% dos valores
    test = values[-n_test_seconds:, :]

    test_X, test_y = test[:, :-1], test[:, -1:]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # make a prediction
    yhat = model.predict(test_X)
    prediction_results = pd.DataFrame(yhat)
    prediction_results.to_csv('results/Spec_Norm_LSTM'+k+'predict.csv')


# In[ ]:





# In[ ]:




