from pprint import pprint

import sklearn.metrics

import autosklearn.regression
from settings import INPUT_FILE, CORR_GROUP, OUTPUT_FILE
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='automl.log', level=logging.DEBUG)
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

if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE, index_col='ts')
    #df = df.drop('Unnamed: 0', 1)
    df.index = pd.to_datetime(df.index)

    df_2 = df.loc[:,np.invert(unique_cols(df))]

    # Average window
    # df_2 = df.groupby(np.arange(len(df))//60).mean()

    scaler = MinMaxScaler()
    scaler = MinMaxScaler()
    d = scaler.fit_transform(df_2)
    scaled_df = pd.DataFrame(d, columns=df_2.columns, index=df_2.index)
    results = []
    for k in CORR_GROUP:
        logging.info(k + " started training")
        values = create_supervised_dataset(scaled_df, k, CORR_GROUP[k], n_in=15, n_out=1)
        len_values = values.shape[0]
        # split into train and test sets 
        n_train_seconds = int(0.7*len_values) #70% dos valores
        n_cv_seconds =  int(1*len_values) #20% dos valores
        train = values[:n_train_seconds, :]
        cv = values[n_train_seconds:n_cv_seconds, :]

        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1:]
        test_X, test_y = cv[:, :-1], cv[:, -1:]
        automl = autosklearn.regression.AutoSklearnRegressor(
            time_left_for_this_task=3600,
            per_run_time_limit=120,
            tmp_folder='./tmp2/autosklearn_regression_'+k+'_final_tmp',
        )
        try:
            automl.fit(train_X, train_y, dataset_name=k)
            pickle.dump(automl, open('models/autosklearn_'+k, 'wb'))
            test_predictions = automl.predict(test_X)
            results.append(sklearn.metrics.mean_squared_error(test_y, test_predictions, squared=False))
        except Exception:
            logging.info('error')
            results.append(1)
        del automl
    with open(OUTPUT_FILE, 'a') as writer:
        writer.write("AutoML,"+",".join([f'{num:.6f}' for num in results])+'\n')