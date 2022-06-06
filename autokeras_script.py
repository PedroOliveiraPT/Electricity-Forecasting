import pandas as pd
import tensorflow as tf

import autokeras as ak

from settings import INPUT_FILE, CORR_GROUP, OUTPUT_FILE
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='autokeras.log', level=logging.DEBUG)
logging.info('Started training')\

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

df = pd.read_csv(INPUT_FILE, index_col='ts')
#df = df.drop('Unnamed: 0', 1)
df.index = pd.to_datetime(df.index)

df_2 = df.loc[:,np.invert(unique_cols(df))]
# Average window
# df_2 = df.groupby(np.arange(len(df))//60).mean()

scaler = MinMaxScaler()
d = scaler.fit_transform(df_2)
scaled_df = pd.DataFrame(d, columns=df_2.columns, index=df_2.index)
results = []
train_split = int(0.7*scaled_df.shape[0])
val_split = int(0.9*scaled_df.shape[0])
predict_from = 1
predict_until = 1
lookback = 15

for k in CORR_GROUP:
    logging.INFO("started training " + k)
    data_train = scaled_df[:train_split]
    data_cv = scaled_df[train_split:val_split]
    
    data_x = data_train[CORR_GROUP[k]]
    data_y = data_train[k]

    data_x_val = data_cv[CORR_GROUP[k]]
    data_y_val = data_cv[k]

    data_test = create_supervised_dataset(scaled_df, k, CORR_GROUP[k], n_in=lookback, n_out=1)[val_split:]
    data_x_test, data_y_test = data_test[:, :-1], data_test[:, -1]
    data_x_test = data_x_test.reshape(-1, lookback, len(CORR_GROUP[k]))
    print(data_x_test.shape)
    print(data_y_test.shape)

    clf = ak.TimeseriesForecaster(
        lookback=lookback,
        predict_from=predict_from,
        predict_until=predict_until,
        max_trials=100,
        project_name=f'autokeras_ml/{k}_forecaster',
        objective="mean_squared_error",
    )

    # Train the TimeSeriesForecaster with train data
    clf.fit(
        x=data_x,
        y=data_y,
        validation_data=(data_x_val, data_y_val),
        batch_size=96,
        epochs=20,
        overwrite=True
    )

    model = clf.export_model()
    logging.INFO("exporting " + k)
    model.save(f'models/{k}_autokeras.h5')
    # Evaluate the best model with testing data.