#!/usr/bin/env python
# coding: utf-8

# Install and import the required packages. 

# In[1]:

import os
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import torch
import torch.nn as nn
import torch.optim as optim
import time

from util.data_handling.data_loader import get_dataloaders
from edit_distance.train import load_edit_distance_dataset,train,test
from edit_distance.models.pair_encoder import PairEmbeddingDistance
from edit_distance.models.linear_encoder import LinearEncoder


# In this notebook, we only show the code to run a simple linear layer on the sequence which, in the hyperbolic space, already gives particularly good results. Later we will also report results for more complex models whose implementation can be found in the [NeuroSEED repository](https://github.com/gcorso/NeuroSEED).

# General training and evaluation routines used to train the models:

# The linear model is trained on 7000 sequences (+700 of validation) and tested on 1500 different sequences: 

# In[2]:



EMBEDDING_SIZE = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(2021)
if device == 'cuda':
    torch.cuda.manual_seed(2021)

# load data
datasets = load_edit_distance_dataset('./datasets/string_for_test.pkl')
loaders = get_dataloaders(datasets, batch_size=128, workers=5)

# model, optimizer and loss
model,optimizer,loss,loss_train,loss_val,avg_loss={},{},{},{},{},{}

encoder = LinearEncoder(153, EMBEDDING_SIZE)

dist_types=['hyperbolic','euclidean']

from util.ml_and_math.loss_functions import AverageMeter

    
for dt in dist_types:
    print(dt)

    model[dt] = PairEmbeddingDistance(embedding_model=encoder, distance=dt)
    loss[dt] = nn.MSELoss()

    optimizer[dt] = optim.Adam(model[dt].parameters(), lr=0.001)


    # training
    for epoch in range(0, 1):
        t = time.time()
        loss_train[dt] = train(model[dt], loaders['train'], optimizer[dt], loss[dt], device)
        loss_val[dt] = test(model[dt], loaders['val'], loss[dt], device)[0]

        # print progress
        if epoch % 5 == 0:
            print('Epoch: {:02d}'.format(epoch),
                'loss_train: {:.6f}'.format(loss_train[dt]),
                'loss_val: {:.6f}'.format(loss_val[dt]),
                'time: {:.4f}s'.format(time.time() - t))
        
    # testing
    for dset in loaders.keys():
        avg_loss = test(model[dt], loaders[dset], loss[dt], device)[0]
        print('Final results {}: loss = {:.6f}'.format(dset, avg_loss))

