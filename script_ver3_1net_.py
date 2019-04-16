from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import History
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from keras.layers import BatchNormalization

import csv
import numpy as np
import pandas as pd
import os
import json
import re

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.neural_network import MLPRegressor

import sys
import argparse

def make_my_data(data_to_corr, params):
    df = pd.DataFrame()
    for name in params.keys():
        temp = data_to_corr[name].copy()
        temp = temp.replace('N', np.NaN).replace(',', '.', regex=True).astype('float')
        for k in params[name]:
            shift_int = int(k)
            if shift_int > 0:
                df[name + '+' + str(k)] = temp.shift(-shift_int)
            elif shift_int == 0:
                df[name] = temp
            elif shift_int < 0:
                df[name + str(k)] = temp.shift(-shift_int)


    df = df.dropna()
    df = df[(df != 'N').all(axis=1)]
    #df.to_csv('/data/Lera_November/Data/list.csv', sep=';', decimal=',')
    return df

def my_mae(y, y_pred): return (np.mean(np.abs(y - y_pred)))


def my_rmse(y, y_pred): return (np.sqrt(np.mean((y - y_pred) ** 2)))


def my_corr(y, y_pred): return (np.corrcoef(y, y_pred)[1, 0])


def my_R2(y, y_pred):
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    R2 = 1 - ss_res / ss_tot
    return (R2)


def PrintErrors(y_test, y_train, y_predicted_test, y_predicted_train):
    mae1 = float(my_mae(y_test, y_predicted_test))
    mae2 = float(my_mae(y_train, y_predicted_train))

    rmse1 = float(my_rmse(y_test, y_predicted_test))
    rmse2 = float(my_rmse(y_train, y_predicted_train))

    r1 = float(my_R2(y_test, y_predicted_test))
    r2 = float(my_R2(y_train, y_predicted_train))

    s = pd.DataFrame({'MAE': [mae1, mae2],
                      'RMSE': [rmse1, rmse2],
                      'R2': [r1, r2]},
                     index=pd.Series(['test', 'train']))

    print("MAE (test set): ", mae1, "MAE (train_set): ", mae2, sep = ' ', end='\n')
    print("RMSE (test set): ", rmse1,
          "RMSE (train_set): ", rmse2, sep = ' ', end='\n')
    print("R2 (test set): ", r1,
          "R2 (train_set): ", r2, sep = ' ', end='\n')

    return s

def PrintErrorsOnce(y, y_predicted):
    mae = float(my_mae(y, y_predicted))

    rmse = float(my_rmse(y, y_predicted))

    r = float(my_R2(y, y_predicted))

    s = pd.DataFrame({'MAE': [mae],
                      'RMSE': [rmse],
                      'R2': [r]})

    return s

def compile_perceptron(X, y, checkpoint_path, output_path, horizon = 1, neurons=64, epochs=2800, val_size=0.2, x_scaler=MinMaxScaler(), y_scaler=MinMaxScaler(),
                       activation='sigmoid'):

    y_test = y.iloc[(len(y) - 300):]
    y_train = y.iloc[:(len(y) - 300)]

    X_test = X.iloc[(len(X) - 300):]
    X_train = X.iloc[:(len(X) - 300)]

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    # print()
    y_train_scaled = y_scaler.fit_transform(np.array(y_train).reshape(-1, 1))

    X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = train_test_split(X_train_scaled, y_train_scaled,
                                                                                  test_size=val_size, random_state=1)

    train_file = pd.DataFrame()
    test_file = pd.DataFrame()
    train_file['y train'] = y_train.copy()
    # train_file.index = pd.to_datetime(y_train.index)
    train_file.index = train_file.index.shift(horizon, freq='D')
    test_file['y test'] = y_test.copy()
    test_file.index = pd.to_datetime(y_test.index)
    test_file.index = test_file.index.shift(horizon, freq='D')

    train_av = pd.DataFrame()
    test_av = pd.DataFrame()

    stats_test = pd.DataFrame()
    stats_train = pd.DataFrame()

    stats_epochs = pd.DataFrame(columns=['es_epochs'])
    for i in range(5):

        m = Sequential()
        m.add(Dense(neurons, input_dim=X_train_scaled.shape[1], activation=activation))
        m.add(BatchNormalization())
        m.add(Dense(1, activation='linear'))

        m.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.5), metrics=['mae', 'mse'])
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=400, verbose=0, mode='min')

        history = m.fit(X_train_scaled, y_train_scaled, epochs=epochs, batch_size=10, shuffle=True, validation_data=(X_val_scaled, y_val_scaled), verbose=0, callbacks=[checkpoint, es])
        m.load_weights(checkpoint_path)

        y_pred_test = m.predict(X_test_scaled, batch_size=1, verbose=0, steps=None)
        y_pred_train = m.predict(X_train_scaled, batch_size=1, verbose=0, steps=None)


        this_stats_train = PrintErrorsOnce(y_train_scaled, y_pred_train.flatten())

        y_pred_test = y_scaler.inverse_transform(y_pred_test)
        y_pred_train = y_scaler.inverse_transform(y_pred_train)

        this_stats_test = PrintErrorsOnce(y_test, y_pred_test.flatten())
        test_av['test'+str(i)]=y_pred_test.flatten()
        train_av['train'+str(i)]=y_pred_train.flatten()

        this_epochs = es.stopped_epoch

        #print(this_epochs)


        stats_test = pd.concat([stats_test, this_stats_test])
        stats_train = pd.concat([stats_train, this_stats_train])
        stats_epochs = stats_epochs.append({'es_epochs': this_epochs}, ignore_index=True)

        if i==0:
            hist = pd.DataFrame(history.history)


    test_av['mean'] = test_av.mean(axis=1)
    train_av['mean'] = train_av.mean(axis=1)


    test_file['y_predicted_test'] = test_av['mean'].values
    train_file['y_predicted_train'] = train_av['mean'].values

    test_file.to_csv(output_path + "/y_test" + ".csv", sep=';')
    train_file.to_csv(output_path + "/y_train" + ".csv", sep=';')
    stats_epochs.to_csv(output_path + "/early_stopping_epochs" + ".csv", sep=';')
    hist.to_csv(output_path + "/loss_history.csv", sep = ';')

    f = PrintErrors(np.array(y_test).flatten(), np.array(y_train).flatten(), np.array(test_av['mean']).flatten(), np.array(train_av['mean']).flatten())
    f.to_csv(output_path + "/statistics.csv", sep=';')


parser = argparse.ArgumentParser(description='Parameters for computing perceptron output.')
parser.add_argument('--data', type=str, default='', help="Path for the initial dataset.")
parser.add_argument('--checkpoint', type=str, default='', help="Path for the saved models.")
parser.add_argument('--params', type=str, default='', help="Json file with horizon and depths parameters.")
parser.add_argument('--output', type=str, default='', help='Directory for output files.')

parser.add_argument('--horizon', type=int, default='1', help='Prediction horizon. ')
parser.add_argument('--threshold', type=str, default='None', help='Correlation threshold. ')
parser.add_argument('--neurons', type=int, default=1, help='Number of neurons.')
parser.add_argument('--mode', type=str, default='None', help='Mode: SW/None.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs.')
parser.add_argument('--layers', type=int, default=1, help='Number of layers.')
parser.add_argument('--corr_value', type=str, default='lg(E_fp_24)', help='The value all the others will be correlated with.')

args = parser.parse_args()

data = pd.read_csv(args.data, sep=';', decimal=',')
data = data.replace('N', np.NaN).replace(',', '.', regex=True).astype('float')
date = pd.to_datetime(data[["year", "month", "day"]])
data = data.set_index(date)

with open(args.params) as data_file:
    params = json.load(data_file)

data_file.close()

for name in ['AE_mid_24',
                 'AE_max_24',
                 'ULFgr_mid_24',
                 'ULFgr_max_24',
                 'ULFgeo_mid_24',
                 'ULFgeo_max_24',
                 'ULFimf_mid_24',
                 'ULFimf_max_24',
                 'ULFden_mid_24',
                 'ULFden_max_24']:

    if name in params:
        del params[name]
    if name in data.columns:
        data.drop([name], axis=1, inplace=True)

if args.threshold=='None':
    threshold=None
else: threshold = float(args.threshold)

my_data = make_my_data(data, params)
#print(my_data.head())
l = []
for (key, value) in params.items():
    for shift in value:
        if shift>0:
            l+=[key+'+'+str(shift)]
        elif shift<0:
            l+=[key+str(shift)]
        else: l+=[key]

my_l = [value for value in l if value in list(my_data)] 
data_new = my_data[my_l]
for name in my_l:
    if name.startswith(args.corr_value+'+'):
        y_name=name

y = data_new[y_name]
#print(y.head())
X = data_new.loc[:, data_new.columns != y_name]
#print(X.head())
#print(y.columns, x.columns, sep='\n')

'''
checkpoint_path - directory to store best weights for each model
output path - directory to store output for each model
'''


mode_name = args.mode
checkpoint_path = ''.join([args.checkpoint, '_epochs', str(args.epochs), '_neurons', str(args.neurons), '_horizon', str(args.horizon), '_mode', mode_name, '_threshold', str(threshold)])
output_path = ''.join([args.output, '_epochs', str(args.epochs), '_neurons', str(args.neurons), '_horizon', str(args.horizon), '_mode', mode_name, '_threshold', str(threshold)])

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)

print('Learning has started...')
print('Mode: ', args.mode, 'Path:', output_path)
compile_perceptron(X, y, epochs=int(args.epochs), horizon=int(args.horizon),neurons=int(args.neurons), x_scaler=MinMaxScaler(feature_range=(-1,1)), y_scaler=MinMaxScaler(feature_range=(-1,1)),
                   activation='tanh', checkpoint_path=checkpoint_path + '/best_weights.hdf5', output_path=output_path)
print('Done.')
