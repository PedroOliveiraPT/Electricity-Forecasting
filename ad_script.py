import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import autokeras as ak
from sklearn import metrics # for the evaluation
from settings import CORR_GROUP, AD_THRESHOLD, INPUT_FILE, SEED1
import tensorflow as tf
import logging
import random
import pickle


logging.basicConfig(format='%(asctime)s %(message)s', filename='anomalydetection.log', level=logging.DEBUG)
logging.info('Started script')

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

for k in CORR_GROUP:
    scaled_df['AD'] = " "
    scaled_df[k + ' AD Detected'] = " "
    scaled_df[k + ' Predict'] = " "

random.seed(SEED1)
anomaly_df = scaled_df.tail(int(0.1*len(scaled_df)))
control = 0
for index, row in anomaly_df.iterrows():
    if control != 0:
        if control > 5:
            anomaly_df.at[index, 'AD'] = True
            for k in CORR_GROUP:
                anomaly_df.at[index, k] /= 2
            control = 0
            continue

        anomaly_df.at[index, 'AD'] = True
        for k in CORR_GROUP:
            anomaly_df.at[index, k] = 0
        control += 1
        continue

    is_anomaly = random.random() < 0.01
    if is_anomaly:
        anomaly_df.at[index, 'AD'] = True
        for k in CORR_GROUP:
            anomaly_df.at[index, k] /= 2
        control += 1
        
    else:
        anomaly_df.at[index, 'AD'] = False

scaled_df.iloc[-int(0.1*len(scaled_df)):] = anomaly_df

models_linreg = ['ReacEc_L1', 'ReacEc_L3', 'RealE_SUM']

for var in CORR_GROUP:
    logging.info(var + " started script")
    model = None
    if var in models_linreg: model = pickle.load(open(f'models/{var}_model.sav', 'rb'))
    else: model = tf.keras.models.load_model(f'models/{var}_autokeras.h5')
    features = []
    counter = 0
    history_window = 15
    for index, row in scaled_df.iterrows():

        if counter >= history_window:
            if row['AD'] == True or row['AD'] == False:
                tensor = np.array(features).reshape(-1, history_window, len(CORR_GROUP[var]))
                res = model.predict(tensor, verbose=0)
                scaled_df.at[index, var + ' Predict'] = res
                ad_detected = abs(res - row[var]) > AD_THRESHOLD[var]
                scaled_df.at[index, var + ' AD Detected'] = ad_detected
            features = features[len(CORR_GROUP[var]):]
            
        counter += 1
        predictors = row[CORR_GROUP[var]]
        features += predictors.to_list()


anomaly_df = scaled_df.tail(int(0.1*len(scaled_df)))
anomaly_df.to_csv('results/ad_df.csv')
results = {k:[0,0,0,0] for k in CORR_GROUP}
for index, row in anomaly_df.iterrows():
    for k in CORR_GROUP:
        if row[k + ' AD']:
            if row[k + ' AD Detected']: results[k][0]+=1
            else: results[k][1]+=1
        else:
            if row[k + ' AD Detected']: results[k][2]+=1
            else: results[k][3]+=1
results_df = pd.DataFrame.from_dict(results, orient='index')
results.to_csv('results/ad_results.csv')
#scaled_df.tail(int(0.1*len(scaled_df))).to_csv('results/ad_results.csv')


