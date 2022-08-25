from pprint import pprint

from sklearn import metrics

import autosklearn.regression
from settings import INPUT_FILE, CORR_GROUP, OUTPUT_FILE
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import InputLayer, Bidirectional

import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='implementation.log', level=logging.DEBUG)
logging.info('Started training')

# create network
def create_model(features, timesteps=1):
    model = Sequential()
    model.add(InputLayer(input_shape=(timesteps, features)))
    model.add(Bidirectional(GRU(features, activation='tanh', return_sequences=True), trainable=True))
    model.add(Bidirectional(GRU(features, activation='tanh', return_sequences=True), trainable=True))
    model.add(Bidirectional(GRU(features, activation='tanh'), trainable=True))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    return model

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

if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE, index_col='ts')
    #df = df.drop('Unnamed: 0', 1)
    df.index = pd.to_datetime(df.index)

    df_2 = df.loc[:,np.invert(unique_cols(df))]


    scaler = MinMaxScaler()
    d = scaler.fit_transform(df_2)
    scaled_df = pd.DataFrame(d, columns=df_2.columns, index=df_2.index)
    results = []

    history_window = 15
    prediction_window = 1
    for k in CORR_GROUP:
        logging.info(k + " started training")
        values = create_supervised_dataset(scaled_df, k, CORR_GROUP[k], n_in=history_window, n_out=prediction_window)
        len_values = values.shape[0]
        feats = len(CORR_GROUP[k])
        # split into train and test sets 
        n_train_seconds = int(0.7*len_values) #70% dos valores
        n_cv_seconds =  int(0.9*len_values) #20% dos valores
        train = values[:n_train_seconds, :]
        cv = values[n_train_seconds:n_cv_seconds, :]

        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1:]
        cv_X, cv_y = cv[:, :-1], cv[:, -1:]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((-1, history_window, feats))
        cv_X = cv_X.reshape((-1, history_window, feats))
        model = create_model(feats, timesteps=history_window)
        history = model.fit(train_X, train_y, epochs=20, batch_size=96, validation_data=(cv_X, cv_y), verbose=1, shuffle=False)

        #Test for the day after
        test = values[n_cv_seconds:, :]

        test_X, test_y = test[:, :-1], test[:, -1:]
        test_X = test_X.reshape((test_X.shape[0], history_window, feats))
        # make a prediction
        yhat = model.predict(test_X)
        results.append(np.sqrt(metrics.mean_squared_error(test_y, yhat)))

    with open(OUTPUT_FILE, 'a') as writer:
        writer.write("BiGRU3,"+",".join([f'{num:.6f}' for num in results])+'\n')