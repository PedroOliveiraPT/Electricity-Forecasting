import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from sklearn import metrics # for the evaluation
from settings import CORR_GROUP, SEED1, SEED2, SEED3, INPUT_FILE, OUTPUT_FILE
from keras.callbacks import EarlyStopping
import tensorflow as tf
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
    with open(OUTPUT_FILE, 'a') as writer:
        writer.write(model_desc+","+",".join([f'{num:.6f}' for num in res])+'\n')


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

    model_type = sys.argv[1]
    model = None

    rmse_res = []

    for k in CORR_GROUP:
        values = create_supervised_dataset(scaled_df, k, CORR_GROUP[k], n_in=history_window, n_out=prediction_window)
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
        model_name = None
        change_result = False
        if model_type == 'SimpleLSTM':
            import SimpleLSTM
            model_cells = int(sys.argv[2])
            model = SimpleLSTM.create_model(model_cells, train_X.shape[2])
            model_name = f"SimpleLSTM_{model_cells}_15secs"
        elif model_type == 'SimpleGRU':
            import SimpleGRU
            model_cells = int(sys.argv[2])
            model_name = f"SimpleGRU_{model_cells}_15secs"
            model = SimpleGRU.create_model(model_cells, train_X.shape[2])
        elif model_type == 'DropoutLSTM':
            import DropoutLSTM
            rate = int(sys.argv[2])/100
            model_name = f"DropoutLSTM_{sys.argv[2]}_15secs"
            model = DropoutLSTM.create_model(rate, train_X.shape[2])
        elif model_type == 'DropoutGRU':
            import DropoutGRU
            rate = int(sys.argv[2])/100
            model_name = f"DropoutGRU_{sys.argv[2]}_15secs"
            model = DropoutGRU.create_model(rate, train_X.shape[2])
        elif model_type == 'DenseLSTM':
            import DenseLSTM
            dense_cells = int(sys.argv[2])
            model_name = f"DenseLSTM_{dense_cells}_15secs"
            model = DenseLSTM.create_model(dense_cells, train_X.shape[2])
        elif model_type == 'DenseGRU':
            import DenseGRU
            dense_cells = int(sys.argv[2])
            model_name = f"DenseGRU_{dense_cells}_15secs"
            model = DenseGRU.create_model(dense_cells, train_X.shape[2])
        elif model_type == 'LSTMDense':
            import LSTMDense
            dense_cells = int(sys.argv[2])
            model_name = f"LSTMDense_{dense_cells}_15secs"
            model = LSTMDense.create_model(dense_cells, train_X.shape[2])
        elif model_type == 'GRUDense':
            import GRUDense
            dense_cells = int(sys.argv[2])
            model_name = f"GRUDense_{dense_cells}_15secs"
            model = GRUDense.create_model(dense_cells, train_X.shape[2])
        elif model_type == 'StackedLSTM':
            import StackedLSTM
            nstacks = int(sys.argv[2])
            change_result = True
            model_name = f"StackedLSTM_{nstacks}_15secs"
            model = StackedLSTM.create_model(nstacks, train_X.shape[2])
        elif model_type == 'StackedGRU':
            import StackedGRU
            nstacks = int(sys.argv[2])
            change_result = True
            model_name = f"StackedGRU_{nstacks}_15secs"
            model = StackedGRU.create_model(nstacks, train_X.shape[2])
        elif model_type == 'ConvLSTM':
            import ConvLSTM
            nstacks = int(sys.argv[2])
            change_result = True
            model_name = f"ConvLSTM10_{nstacks}_15secs"
            model = ConvLSTM.create_model(nstacks, train_X.shape[2])
        elif model_type == 'ConvGRU':
            import ConvGRU
            nstacks = int(sys.argv[2])
            change_result = True
            model_name = f"ConvGRU_{nstacks}_15secs"
            model = ConvGRU.create_model(nstacks, train_X.shape[2])
        elif model_type == 'BiLSTM':
            import BiLSTM
            ncells = int(sys.argv[2])
            rate = int(sys.argv[3])
            model_name = f'BiLSTM_{ncells}_{rate}_15secs'
            model = BiLSTM.create_model(ncells, rate/100, train_X.shape[2])
        elif model_type == 'BiGRU':
            import BiGRU
            ncells = int(sys.argv[2])
            rate = int(sys.argv[3])
            model_name = f'BiGRU_{ncells}_{rate}_15secs'
            model = BiGRU.create_model(ncells, rate/100, train_X.shape[2])
        elif model_type == 'AttentionBiLSTM':
            import AttentionBiLSTM
            ncells = int(sys.argv[2])
            change_result = True
            rate = int(sys.argv[3])
            model_name = f'AttentionBiLSTM_{ncells}_{rate}_15secs'
            model = AttentionBiLSTM.create_model(ncells, rate/100, train_X.shape[2])
        elif model_type == 'AttentionBiGRU':
            import AttentionBiGRU
            ncells = int(sys.argv[2])
            change_result = True
            rate = int(sys.argv[3])
            model_name = f'AttentionBiGRU_{ncells}_{rate}_15secs'
            model = AttentionBiGRU.create_model(ncells, rate/100, train_X.shape[2])
        
        if model is None:
            logging.warning(f"No model found for {model_type}")
            break
        results = []
        for s in [SEED1, SEED2, SEED3]:
            logging.info(f"{k} for {model_name} started training with {s}")
            tf.keras.utils.set_random_seed(s)
            history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(cv_X, cv_y), verbose=1, shuffle=False, callbacks=[callback])

            #Test for the day after
            n_test_seconds =  int(0.1*len_values) #10% dos valores
            test = values[-n_test_seconds:, :]

            test_X, test_y = test[:, :-1], test[:, -1:]
            test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
            # make a prediction
            yhat = model.predict(test_X)
            if change_result:
                yhat = yhat.reshape(yhat.shape[0], yhat.shape[1])
            results.append(np.sqrt(metrics.mean_squared_error(test_y, yhat)))
            logging.info(f"{k} for {model_name}, {s} was run for {len(history.history['loss'])} epochs")
        rmse_res.append(sum(results)/3)

    write_results(model_name, rmse_res)
