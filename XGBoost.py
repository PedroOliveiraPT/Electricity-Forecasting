import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn import metrics # for the evaluation
from settings import CORR_GROUP
from xgboost import XGBRegressor
import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='training.log', level=logging.DEBUG)
logging.info('Started training')

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

def write_results(model_desc, res):
    with open('./results/rmse_results2.csv', 'a') as writer:
        writer.write(model_desc+","+",".join([f'{num:.3f}' for num in res])+'\n')


if __name__ == '__main__':

    df = pd.read_csv("data/mongo_data.csv", index_col='ts')
    df = df.drop('Unnamed: 0', 1)
    df.index = pd.to_datetime(df.index)

    df = df.loc[:,np.invert(unique_cols(df))]

    # Average window
    df_2 = df.groupby(np.arange(len(df))//60).mean()

    scaler = MinMaxScaler()
    d = scaler.fit_transform(df_2)
    scaled_df = pd.DataFrame(d, columns=df_2.columns, index=df_2.index)

    history_window = 15
    prediction_window = 1

    n_est = int(sys.argv[1])
    model = None

    rmse_res = []

    for k in CORR_GROUP:
        values = create_supervised_dataset(scaled_df, k, CORR_GROUP[k], n_in=history_window, n_out=prediction_window)
        len_values = values.shape[0]
        # split into train and test sets 
        n_train_seconds = int(0.7*len_values) #70% dos valores
        train = values[:n_train_seconds, :]
        train_X, train_y = train[:, :-1], train[:, -1:]
        
        model = XGBRegressor(objective='reg:squarederror', n_estimators=n_est)
        model.fit(train_X, train_y)
        
        n_test_seconds =  int(0.3*len_values) #10% dos valores
        test = values[-n_test_seconds:, :]

        test_X, test_y = test[:, :-1], test[:, -1:]
        yhat = model.predict(test_X)
        rmse_res.append(np.sqrt(metrics.mean_squared_error(test_y, yhat)))

    write_results(f"XGBoost_{n_est}", rmse_res)
