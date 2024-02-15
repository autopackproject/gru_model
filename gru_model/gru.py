from common import model
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe

import numpy as np
import math


glove = GloVe(name='840B', dim=300)

#RNN from Lab 5, changes made to the function for project purposes
class RNN(model.Model):
    def __init__(self, name, hidden_size=64, batch_size=64,
                 n_layers=1, dropout=0.0, bidir=False, pooling='max',
                 num_epochs=100, lr=0.01,
                 glove=glove):
        super(RNN, self).__init__(name, batch_size=batch_size, num_epochs=num_epochs, lr=lr)
        self._glove=glove
        self.name = name
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidir = bidir
        self.batch_size = batch_size
        self.ident = torch.eye(self._glove.dim) # type: ignore
        self.pool = pooling

        #layers
        self.rnn = nn.GRU(self._glove.dim,
                          hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=bidir)
        if bidir: x = 2
        else: x = 1
        # self.classifier = nn.Linear(x*hidden_size, 1) 
        # we don't need a classifier at this stage

    def forward(self, inp, hidden=None):
        output, hidden = self.rnn(inp)
        # use pooling
        if self.pool == 'max':
            # print("1", output.shape)
            output = torch.max(output, dim=1)[0]
            # print("2", output.shape)
        elif self.pool == 'mean':
            output = torch.mean(output, dim=1)[0]
        elif self.pool == 'cat':
            output = torch.cat([torch.max(output, dim=1)[0],
                                torch.mean(output, dim=1)], dim=0)
        # output = self.classifier(output)
        return output


#Anime dataset class
class animeDataset(Dataset):
    def __init__(self, df):
        self.df, self.max_len = self.tokenize(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        input_data = self.df['inputs']
        output_data = self.df['outputs']
        return input_data, output_data
        # label = torch.tensor(row['popularity']).float()
        # data = row['synopsis']
        # return data, label

    def tokenize(self, df):
        # cols = ['popularity', 'synopsis']
        # df = df[cols]
        # df['popularity'] = (df['popularity'] - df['popularity'].min()) / (df['popularity'].max() - df['popularity'].min())

        # assume a maximum file length of 10000 for simplicity sake for now.
        # the maximum length can be explored in the future, but we want to standardize the
        # such that we can train in batches.
        max_len = 10000 
        tokenizer = get_tokenizer("basic_english")
        for input in df['inputs']:
            syn_len = len(tokenizer(input))
            if syn_len > max_len:
                max_len = syn_len

        for output in df['outputs']:
            syn_len = len(tokenizer(output))
            if syn_len > max_len:
                max_len = syn_len

        # for i in range(len(df['synopsis'])):
        def process_example(ex):
            ex = tokenizer(ex)
            ex += ['<pad>'] * (max_len - len(ex))
            ex = glove.get_vecs_by_tokens(ex)
            # ex = torch.transpose(ex, 0, 1)
            return ex

        df['inputs'] = df['inputs'].map(process_example)
        return df, max_len

#Data loader function
def get_data_loaders(path_to_input, path_to_output, batch_size=32):
    df = {'inputs': torch.tensor([]), 'outputs': torch.tensor([])}

    with open(path_to_input, 'r') as file:
        input = file.read()
    
    with open(path_to_output, 'r') as file:
        output = file.read()

    df['inputs'].append(input)
    df['outputs'].append(output)
    print(df['inputs'])
    print(df['outputs'])

    ds = animeDataset(df)
    max_len = ds.max_len

    train_size = int(0.8*len(ds))
    val_size = len(ds) - train_size
    train, val = random_split(ds, [train_size, val_size])
    return DataLoader(train, batch_size), DataLoader(val, batch_size), max_len
