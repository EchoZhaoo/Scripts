# -*- coding: utf-8 -*-
"""
WiDS Processing Script

@Thao Nguyen
"""

#%% Import
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

#%% Load in data
train = pd.read_csv('E:/WiDS18/train.csv')
test = pd.read_csv('E:/WiDS18/test.csv')
#print(train.shape)
#train.head()
train.isnull().sum()

#%% Subset df where there are NAs in column
na_list = train.columns[train.isnull().any()].tolist()
train_na = train[na_list]
train_na.isnull().sum()
train[train[:,'DG4']]
train[train['MT5'].isnull()]
