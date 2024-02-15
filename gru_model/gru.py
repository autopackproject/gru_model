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
        print(output)
        return output

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
    def dot_score(self, hidden_state, encoder_states):
        return torch.sum(hidden_state * encoder_states, dim=2)
      
    def forward(self, hidden, encoder_outputs, mask):
        attn_scores = self.dot_score(hidden, encoder_outputs)# Transpose max_length and batch_size dimensions
        attn_scores = attn_scores.t()# Apply mask so network does not attend <pad> tokens        
        attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        # Return softmax over attention scores      
        return F.softmax(attn_scores, dim=1).unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, dropout=0.1):
        
        super(Decoder, self).__init__()
        
        # Basic network params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
                
        self.gru = nn.GRU(input_size,
                          hidden_size,
                          num_layers=n_layers,
                          batch_first=True,
                          dropout=dropout)
        
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)
        
    def forward(self, current_token, hidden_state, encoder_outputs, mask):        
        # Pass through GRU
        rnn_output, hidden_state = self.gru(current_token, hidden_state)
        
        # Calculate attention weights
        attention_weights = self.attn(rnn_output, encoder_outputs, mask)
        
        # Calculate context vector
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))
        
        # Concatenate  context vector and GRU output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        
        # Pass concat_output to final output layer
        output = self.out(concat_output)
        
        # Return output and final hidden state
        return output, hidden_state

#Anime dataset class
class packageDataset(Dataset):
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
        print(df)
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
            return torch.tensor(ex)
        
        for i in range(len(df['inputs'])):
            new_df = {}
            new_df['inputs'] = torch.empty((len(df['inputs']), max_len, 300))
            new_df['outputs'] = torch.empty((len(df['outputs']), max_len, 300))
            new_df['inputs'][i,:,:] = process_example(df['inputs'][i])
            new_df['outputs'][i,:,:] = process_example(df['outputs'][i])
            print(new_df['inputs'].size())
            new_df['inputs'] = new_df['inputs'].squeeze(0).squeeze(0)
            new_df['outputs'] = new_df['outputs'].squeeze(0).squeeze(0)

        return new_df, max_len

#Data loader function
def get_data_loaders(path_to_input, path_to_output, batch_size=32):
    df = {'inputs': [], 'outputs': []}

    with open(path_to_input, 'r') as file:
        input = file.read()
    
    with open(path_to_output, 'r') as file:
        output = file.read()

    df['inputs'].append(input)
    df['outputs'].append(output)

    ds = packageDataset(df)
    max_len = ds.max_len

    train_size = int(0.8*len(ds))
    val_size = len(ds) - train_size
    train, val = random_split(ds, [train_size, val_size])
    return DataLoader(train, batch_size), DataLoader(val, batch_size), max_len
