#!/usr/bin/env python
# coding: utf-8

# Install and import the required packages. 

# In[1]:

import os
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from scipy.stats import mode
import pickle
import pandas as pd
from util.data_handling.data_loader import get_dataloaders
from edit_distance.train import load_edit_distance_dataset,train,test
from edit_distance.models.pair_encoder import PairEmbeddingDistance
from edit_distance.models.linear_encoder import LinearEncoder

print(torch.__version__)
# In this notebook, we only show the code to run a simple linear layer on the sequence which, in the hyperbolic space, already gives particularly good results. Later we will also report results for more complex models whose implementation can be found in the [NeuroSEED repository](https://github.com/gcorso/NeuroSEED).

# General training and evaluation routines used to train the models:

# The linear model is trained on 7000 sequences (+700 of validation) and tested on 1500 different sequences: 

# In[2]:


def run_model(dataset_name, embedding_size, dist_type, string_size, n_epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2021)
    if device == 'cuda':
        torch.cuda.manual_seed(2021)

    # load data
    datasets = load_edit_distance_dataset(dataset_name)
    loaders = get_dataloaders(datasets, batch_size=128, workers=5)

    # model, optimizer and loss

    encoder = LinearEncoder(string_size, embedding_size)

    model = PairEmbeddingDistance(embedding_model=encoder, distance=dist_type,scaling=True)
    loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad() 


    # training
    for epoch in range(0, n_epoch):
        t = time.time()
        loss_train = train(model, loaders['train'], optimizer, loss, device)
        loss_val = test(model, loaders['val'], loss, device)

        # print progress
        if epoch % 5 == 0:
            print('Epoch: {:02d}'.format(epoch),
                'loss_train: {:.6f}'.format(loss_train),
                'loss_val: {:.6f}'.format(loss_val),
                'time: {:.4f}s'.format(time.time() - t))
        
    # testing
    for dset in loaders.keys():
        avg_loss = test(model, loaders[dset], loss, device)
        print('Final results {}: loss = {:.6f}'.format(dset, avg_loss))

    return model, avg_loss


# %%

string_size=153
n_epoch=20
e_size=np.logspace(1,9,num=9-1, base=2,endpoint=False, dtype=int)

dist_types=['hyperbolic', 'euclidean']

model, avg_loss=np.zeros((len(e_size),len(dist_types)),dtype=object),np.zeros((len(e_size),len(dist_types)))

names=['largest_group_strings', 'string_for_test', 'string_subset']
dataset_name='./datasets/'+name+'.pkl'

for name in names:
    for i in range(len(e_size)):
        for j in range(len(dist_types)):
            model[i][j],avg_loss[i][j]=run_model(dataset_name,e_size[i],dist_types[j],string_size,n_epoch)
    pickle.dump((model,avg_loss,e_size,dist_types), open(name+'.pkl', "wb"))



# %%
