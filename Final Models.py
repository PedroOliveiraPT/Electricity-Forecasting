import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn import metrics # for the evaluation
from settings import CORR_GROUP, SEED1, SEED2, SEED3, INPUT_FILE, OUTPUT_FILE
from keras.callbacks import EarlyStopping
import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)s %(message)s', filename='results.log', level=logging.DEBUG)
logging.info('Started training final models')

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
    d = scaler.fit_transform(df_2)
    scaled_df = pd.DataFrame(d, columns=df_2.columns, index=df_2.index)

    callback = EarlyStopping(monitor='val_loss', patience=10)
    history_window = 15
    prediction_window = 1
    
    results = pd.read_csv(OUTPUT_FILE, index_col='model')
    model_per_group = results.idxmin().to_dict()
    
    for group, model_name in model_per_group.items():
        values = create_supervised_dataset(scaled_df, group, CORR_GROUP[group], n_in=history_window, n_out=prediction_window)
        len_values = values.shape[0]
        # split into train and test sets 
        n_train_seconds = int(0.8*len_values) #70% dos valores
        n_cv_seconds =  len_values #20% dos valores
        train = values[:n_train_seconds, :]
        cv = values[n_train_seconds:n_cv_seconds, :]

        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1:]
        cv_X, cv_y = cv[:, :-1], cv[:, -1:]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        cv_X = cv_X.reshape((cv_X.shape[0], 1, cv_X.shape[1]))
        
        model_params = model_name.split('_')
        model_type = model_params[0]
        if model_type == 'SimpleLSTM':
            import SimpleLSTM
            model_cells = int(model_params[1])
            model = SimpleLSTM.create_model(model_cells, train_X.shape[2])
        elif model_type == 'SimpleGRU':
            import SimpleGRU
            model_cells = int(model_params[1])
            model = SimpleGRU.create_model(model_cells, train_X.shape[2])
        elif model_type == 'DropoutLSTM':
            import DropoutLSTM
            rate = int(model_params[1])/100
            model = DropoutLSTM.create_model(rate, train_X.shape[2])
        elif model_type == 'DropoutGRU':
            import DropoutGRU
            rate = int(model_params[1])/100
            model = DropoutGRU.create_model(rate, train_X.shape[2])
        elif model_type == 'DenseLSTM':
            import DenseLSTM
            dense_cells = int(model_params[1])
            model = DenseLSTM.create_model(dense_cells, train_X.shape[2])
        elif model_type == 'DenseGRU':
            import DenseGRU
            dense_cells = int(model_params[1])
            model = DenseGRU.create_model(dense_cells, train_X.shape[2])
        elif model_type == 'LSTMDense':
            import LSTMDense
            dense_cells = int(model_params[1])
            model = LSTMDense.create_model(dense_cells, train_X.shape[2])
        elif model_type == 'GRUDense':
            import GRUDense
            dense_cells = int(model_params[1])
            model = GRUDense.create_model(dense_cells, train_X.shape[2])
        elif model_type == 'StackedLSTM':
            import StackedLSTM
            nstacks = int(model_params[1])
            model = StackedLSTM.create_model(nstacks, train_X.shape[2])
        elif model_type == 'StackedGRU':
            import StackedGRU
            nstacks = int(model_params[1])
            model = StackedGRU.create_model(nstacks, train_X.shape[2])
        elif model_type == 'ConvLSTM':
            import ConvLSTM
            nstacks = int(model_params[1])
            model = ConvLSTM.create_model(nstacks, train_X.shape[2])
        elif model_type == 'ConvGRU':
            import ConvGRU
            nstacks = int(model_params[1])
            model = ConvGRU.create_model(nstacks, train_X.shape[2])
        elif model_type == 'BiLSTM':
            import BiLSTM
            ncells = int(model_params[1])
            rate = int(model_params[2])
            model = BiLSTM.create_model(ncells, rate/100, train_X.shape[2])
        elif model_type == 'BiGRU':
            import BiGRU
            ncells = int(model_params[1])
            rate = int(model_params[2])
            model = BiGRU.create_model(ncells, rate/100, train_X.shape[2])
        elif model_type == 'AttentionBiLSTM':
            import AttentionBiLSTM
            ncells = int(model_params[1])
            rate = int(model_params[2])
            model = AttentionBiLSTM.create_model(ncells, rate/100, train_X.shape[2])
        elif model_type == 'AttentionBiGRU':
            import AttentionBiGRU
            ncells = int(model_params[1])
            rate = int(model_params[2])
            model = AttentionBiGRU.create_model(ncells, rate/100, train_X.shape[2])
        elif model_type == 'LinRegression':
            from sklearn.linear_model import LinearRegression
            train_X, train_y = values[:, :-1], values[:, -1:]
            train_y = np.reshape(train_y, newshape=train_y.shape[0])
            model = LinearRegression().fit(train_X, train_y)
            pickle.dump(model, open(f'models/{k}_model.sav', 'wb'))
            continue
        
        history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(cv_X, cv_y), verbose=1, shuffle=False, callbacks=[callback])
        model.save('models/' + m + '_model.h5')