import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy

def load_pickle(path):
    with open(path,'rb') as f:
        load_data = pickle.load(f)
    return load_data

def save_pickle(path,data):
    with open(path, 'wb') as f:
        pickle.dump(data,f)
        
def weighted_mean(df):
    wm = list()
    for date,rows in df.iterrows():
        rows = rows[[0,1,2,3]].copy()
        
        
        available_days = rows.dropna().index.astype(int).to_numpy()
        weights = available_days+1
        weights = weights[::-1]
        wm.append(sum(np.array([ weights[idx]*rows[day] for idx,day in enumerate(available_days)]))/sum(weights))
        
    df['weighted_mean'] = wm
    
    return df
    
def acc_mean(ps):
    DAYS = list(ps[0].keys())
    acc_mean = []
    for day in DAYS:
        _ = np.array([ps[p][day] for p in range(4) if ps[p][day] is not None])
        if len(_) != 0: acc_mean.append(_.mean())
        
    return acc_mean

def  confidence_interval(data, confidence = 0.95):
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)
    
    stderr = stats.sem(data)

    # length_of_one_interval
    interval = stderr * stats.t.ppf( (1 + confidence) / 2 , n-1) # ppf : inverse of cdf

    return mean, mean - interval, mean + interval ,interval

# Y0 => decoder의 첫번째 input(day2)
def generate_window(raw_dict,input_seq_len,output_seq_len=4):
    window_len = input_seq_len + output_seq_len - 1
    patients = list(raw_dict.keys())
    
    X,Y,Y0 = dict(),dict(),dict()
    
    for patient in patients:
        df = raw_dict[patient]
        if len(df) < window_len:
            continue
        #i = window start pos
        for i in range(0,len(df) - window_len + 1):
            window_df = df.iloc[i:i+window_len,]
            input_data = window_df.iloc[:input_seq_len,:].iloc[:,:-1].to_numpy(dtype=float)
            output_data = window_df.iloc[input_seq_len-1:,-1]
            
            decoder_input = np.array([0.0,0.0])
            
            data_id = f'{patient}_{i}'
            X[data_id] = input_data
            Y[data_id] = output_data
            Y0[data_id] = decoder_input 

    return X,Y,Y0

def normX(x,scaler):
    mini_batch,w,fea_dim = x.shape

    x = scaler.transform(x.reshape(mini_batch * w,fea_dim).numpy())
    x = torch.from_numpy(x)
    x = x.reshape(mini_batch,w,fea_dim)

    return x