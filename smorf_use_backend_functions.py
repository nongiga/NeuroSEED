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


def compare_models(dataset_name, embedding_size, dist_types, string_size, n_epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(2021)
    if device == 'cuda':
        torch.cuda.manual_seed(2021)

    # load data
    datasets = load_edit_distance_dataset(dataset_name)
    loaders = get_dataloaders(datasets, batch_size=128, workers=5)

    # model, optimizer and loss
    model,optimizer,loss,loss_train,loss_val,avg_loss={},{},{},{},{},{}

    encoder = LinearEncoder(string_size, embedding_size)

    for dt in dist_types:
        print(dt)

        model[dt] = PairEmbeddingDistance(embedding_model=encoder, distance=dt,scaling=True)
        loss = nn.MSELoss()

        optimizer[dt] = optim.Adam(model[dt].parameters(), lr=1e-3)
        optimizer[dt].zero_grad() 


        # training
        for epoch in range(0, n_epoch):
            t = time.time()
            loss_train[dt] = train(model[dt], loaders['train'], optimizer[dt], loss, device)
            loss_val[dt] = test(model[dt], loaders['val'], loss, device)

            # print progress
            if epoch % 5 == 0:
                print('Epoch: {:02d}'.format(epoch),
                    'loss_train: {:.6f}'.format(loss_train[dt]),
                    'loss_val: {:.6f}'.format(loss_val[dt]),
                    'time: {:.4f}s'.format(time.time() - t))
            
        # testing
        for dset in loaders.keys():
            avg_loss[dt] = test(model[dt], loaders[dset], loss, device)
            print('Final results {}: loss = {:.6f}'.format(dset, avg_loss[dt]))

    return model, avg_loss

name='largest_group_strings'

dataset_name='./datasets/'+name+'.pkl'
dist_types=['hyperbolic', 'euclidean']
string_size=153
n_epoch=20
e_size=[2, 8, 32, 128]#np.logspace(1,9,num=9-1, base=2,endpoint=False, dtype=int)
model, avg_loss={},{}
# %%
for i in range(len(e_size)):
    model[e_size[i]],avg_loss[e_size[i]]=compare_models(dataset_name,e_size[i],dist_types,string_size,n_epoch)

pickle.dump((model,avg_loss), open(name+'.pkl', "wb"))



# %%
