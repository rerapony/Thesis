import csv
import pandas as pd
import numpy as np

from scipy.stats.stats import pearsonr

def make_my_data(data_to_corr, params):
    value_all = pd.DataFrame()
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
