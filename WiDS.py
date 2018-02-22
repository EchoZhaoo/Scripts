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
from sklearn.preprocessing import Imputer

#%% Load in data
train = pd.read_csv('//Files.umn.edu/cse/UmSaveDocs/nguy3409/Documents/train.csv')
test = pd.read_csv('//Files.umn.edu/cse/UmSaveDocs/nguy3409/Documents/test.csv')
data_dict = pd.read_csv('//Files.umn.edu/cse/UmSaveDocs/nguy3409/Documents/WiDS_dict.csv')

# Get list of variable names
train_list = list(train)
dict_list = data_dict.iloc[:,0].tolist()
dict_list = ' '.join(dict_list)
dict_list = dict_list.replace('"', '')
dict_list = dict_list.replace('\n', '')
dict_list = dict_list.split(' ')
#print(train.shape)
#train.head()
train.isnull().sum()
# Only select columns that are available in the data dictionary
L = list(set(train_list).intersection(dict_list))
train = train[L]

#%% Subset df where there are < 80% NAs in column
df_list = train.columns[train.isnull().mean() < 0.8].tolist()
train_df = train[df_list]
#train_df.isnull().sum()

#%% Check categorical columns
train_df.select_dtypes(exclude=["number"])

#%% Try imputing the missing values with the mean for each column
#train_imputed = train_df.fillna(train_df.mean(), inplace=True)
train_tf = train_df.apply(lambda x:x.fillna(x.value_counts().index[0]))
# Check for NA
train_tf.isnull().sum()
