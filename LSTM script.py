#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os
from sklearn import metrics # for the evaluation

# Model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import LayerNormalization

# create network
def create_model(lstm_cells, features, timesteps=1):
    model = Sequential()
    model.add(LSTM(lstm_cells, input_shape=(timesteps, features)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    return model

def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)

corr_group = {
    'P_SUM': #Var to Predict
        ['S_SUM', # Sum of apparent power S1, S2, S3
        'S_L3', # Apparent power S3 
        'S_L2', # Apparent Power S2
        'S_L1', # Apparent power S1
        'C_phi_L1', #Fund power CosPhi factor L1
        'C_phi_L2', #Fund power CosPhi factor L2
        'P_SUM', # Sum of powers P1, P2, P3
        'P_L1', # Real Power 1
        'P_L2', # Real Power 2
        'P_L3', # Real Power 3
        'Q_SUM', # SUm of fund reactive power
        'Q_L1', #Fundamental Reactive Power Q1
        'Q_L2', #Fundamental Reactive Power Q2
        'Q_L3', #Fundamental Reactive Power Q3
        'I_L1', # Current L1
        'I_L2', # Current L2
        'I_L3'], # Current L3
    'U_L1_N':
        ['U_L1_L2', # Voltage L1_l2
        'U_L3_L1', # Voltage L3_l1
        'U_L3_N', # Voltage L3_N
        'U_L2_L3', # Voltage L2_l3
        'U_L2_N', # Voltage L2_N
        'U_L1_N'], # VOltage L1_N 
    'I_SUM': 
        ['I_SUM'], # Current Sum 
    'F': 
        ['F'], # Measured Freq
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
        'ReacE_L1'], #Reactive Energy L1
    'C_phi_L3': 
        ['C_phi_L3'] #Fund power CosPhi factor L3
}

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

def write_results(model_desc, res):
    if not os.path.isfile('./results/rmse_results.csv'):
        with open('./results/rmse_results.csv', 'w') as writer:
            writer.write("model,"+",".join(list(corr_group.keys)))
    with open('./results/rmse_results.csv', 'a') as writer:
        writer.append(model_desc+","+",".join(res))
# In[2]:

if __name__ == '__main__':

    df = pd.read_csv("data/mongo_data.csv", index_col='ts')
    df = df.drop('Unnamed: 0', 1)
    df.index = pd.to_datetime(df.index)

    df = df.loc[:,np.invert(unique_cols(df))]

    # Average window
    df_2 = df.groupby(np.arange(len(df))//60).mean()
    print(df_2.shape)
    df_2.head()


    scaler = MinMaxScaler()
    d = scaler.fit_transform(df_2)
    scaled_df = pd.DataFrame(d, columns=df_2.columns, index=df_2.index)

    history_window = sys.argv[1]
    model_cells = sys.argv[2]
    prediction_window = 1

    rmse_res = []

    for k in corr_group:
        values = create_supervised_dataset(scaled_df, k, corr_group[k], n_in=history_window, n_out=prediction_window)
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
        model = create_model(model_cells, train_X.shape[2])
        history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(cv_X, cv_y), shuffle=False)

        #Test for the day after
        n_test_seconds =  int(0.1*len_values) #10% dos valores
        test = values[-n_test_seconds:, :]

        test_X, test_y = test[:, :-1], test[:, -1:]
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        # make a prediction
        yhat = model.predict(test_X)
        rmse_res.append(np.sqrt(mean_squared_error(test, fc)))

    write_results(f"LSTM{model_cells}_{history_window}secs", rmse_res)
