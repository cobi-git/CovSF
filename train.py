import sys
sys.path.append('/home/qotmd01/CovSF_2/')
sys.path.append('/home/qotmd01/CovSF_2/module/')

import argparse
import os
from pathlib import Path

from trainer import Trainer

'''
Adjust arguments here
'''
DATASET_DIR = '/home/qotmd01/CovSF_2/Data/train/0625/3_4/'
SAVE_DIR = '/home/qotmd01/CovSF_2/trained/'

hidden_layers = 2
hidden_dim = 32
recurrent_type = 'gru'
recurrent_dropout = 0.2

model_args = {
    'encoder_args' : {
        'recurrent_type' : recurrent_type,
        'input_dim' : 16, # Feature numbers
        'hidden_dim' : hidden_dim,
        'hidden_layers' : hidden_layers,
        'recurrent_dropout' : recurrent_dropout
    },
    'decoder_args' : {
        'recurrent_type' : recurrent_type,
        'output_dim' : 2,
        'output_len' : 4, # output window size
        'hidden_dim' : hidden_dim,
        'hidden_layers' : hidden_layers,
        'recurrent_dropout' : recurrent_dropout,
        'fc_layers' : [16,8],
        'fc_dropout' : 0.4,
        'norm_method' : 'batch'
    }
}

train_args = {
    'batch_size' : 32,
    'epochs' : 200,
    'optimizer' : {"lr": 0.001, "weight_decay": 0},
    'scheduler' : {"patience": 10, "verbose": True},
    'early_stopping' : {"patience":15, "verbose": True, "delta" : 0},
    'tpro' : 0.2
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",required=True,help="Save name")
    # parser.add_argument("-m",required=True,
    #                     help="Model Type: 0 - Seq2SeqAttn, 1 - Seq2Seq, 2 - Vanilla")
    parser.add_argument("-d",required=True,help="cuda cpu num")
    
    args = parser.parse_args()
    s = str(args.s)
    # m = int(args.m)
    d = int(args.d)

    os.environ["CUDA_VISIBLE_DEVICES"]= f"{d}"

    trainer = Trainer(
        dataset_dir = DATASET_DIR,
        save_dir = SAVE_DIR,
        save_name = s
    )

    trainer.train(model_args,train_args)
