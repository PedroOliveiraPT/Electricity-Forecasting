import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn import metrics # for the evaluation
from settings import CORR_GROUP, SEED1, SEED2, SEED3
from keras.callbacks import EarlyStopping
import tensorflow as tf
import logging

import AttentionBiLSTM
import SimpleGRU
import ConvLSTM


models = {
    'P_SUM': AttentionBiLSTM.create_model(8, 0.5, 19*15),
    'U_L1_N': SimpleGRU.create_model(10, 6*15),
    'I_SUM': SimpleGRU.create_model(20, 6*15),
    'H_TDH_I_L3_N': AttentionBiLSTM.create_model(8, 0.1, 15),
    'F': ConvLSTM.create_model(3, 15),
    'C_phi_L3': SimpleGRU.create_model(70, 15)
}

def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)


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


df = pd.read_csv("data/mongo_filtered_av101_mins.csv", index_col='ts')
#df = df.drop('Unnamed: 0', 1)
df.index = pd.to_datetime(df.index)

df_2 = df.loc[:,np.invert(unique_cols(df))]

# Average window
# df_2 = df.groupby(np.arange(len(df))//60).mean()

scaler = MinMaxScaler()
d = scaler.fit_transform(df_2)
scaled_df = pd.DataFrame(d, columns=df_2.columns, index=df_2.index)

callback = EarlyStopping(monitor='val_loss', patience=10)
history_window = 15
prediction_window = 1

for m in models:
    values = create_supervised_dataset(scaled_df, m, CORR_GROUP[m], n_in=history_window, n_out=prediction_window)
    len_values = values.shape[0]
    # split into train and test sets 
    n_train_seconds = int(0.8*len_values) #70% dos valores
    n_cv_seconds =  int(1*len_values) #20% dos valores
    train = values[:n_train_seconds, :]
    cv = values[n_train_seconds:n_cv_seconds, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1:]
    cv_X, cv_y = cv[:, :-1], cv[:, -1:]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    cv_X = cv_X.reshape((cv_X.shape[0], 1, cv_X.shape[1]))
    model = models[m]
    history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(cv_X, cv_y), verbose=1, shuffle=False, callbacks=[callback])
    model.save('models/' + m + '_model.h5')