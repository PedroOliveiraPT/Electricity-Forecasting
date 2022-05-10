import pandas as pd
import numpy as np
from settings import CORR_GROUP, INPUT_FILE, OUTPUT_FILE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics # for the evaluation

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

def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)

if __name__ == '__main__':
    df = pd.read_csv(INPUT_FILE, index_col='ts')
    df.index = pd.to_datetime(df.index)
    
    df_2 = df.loc[:,np.invert(unique_cols(df))]
    

    scaler = MinMaxScaler()
    d = scaler.fit_transform(df_2)
    scaled_df = pd.DataFrame(d, columns=df_2.columns, index=df_2.index)
    
    history_window =  15 # 8*15secs = 120secs
    prediction_window = 1 #predict 15 secs
    results = []
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
        train_y = np.reshape(train_y, newshape=train_y.shape[0])
        model = LinearRegression().fit(train_X, train_y)
        r_sq = model.score(train_X, train_y)

        #Test for the day after
        print("Starting Test", k)
        n_test_seconds =  int(0.1*len_values) #10% dos valores
        test = values[-n_test_seconds:, :]

        test_X, test_y = test[:, :-1], test[:, -1:]
        test_y = np.reshape(test_y, newshape=test_y.shape[0])
        # make a prediction
        yhat = model.predict(test_X)
        results.append(np.sqrt(metrics.mean_squared_error(test_y, yhat)))
    
    with open(OUTPUT_FILE, 'w') as writer:
        writer.write("model,"+','.join(CORR_GROUP)+"\n")
        writer.write("LinRegression,"+",".join([f'{num:.6f}' for num in results])+'\n')
