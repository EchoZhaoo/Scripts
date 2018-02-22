# -*- coding: utf-8 -*-
"""
WiDS Processing Script
@Thao Nguyen
"""

#%% Import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

########################################################
############### Train DATA #############################
########################################################
#%% Load in data
train = pd.read_csv('//Files.umn.edu/cse/UmSaveDocs/nguy3409/Documents/train.csv')
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

#%% Try imputing the missing values with the most frequently appeared value for each column
train_tf = train_df.apply(lambda x:x.fillna(x.value_counts().index[0]))
# Check for NA
train_tf.isnull().sum()

########################################################
############### TEST DATA ##############################
########################################################
#%% Load in data
test = pd.read_csv('//Files.umn.edu/cse/UmSaveDocs/nguy3409/Documents/test.csv')
# Subset columns that match training data
tst_list = df_list.remove('is_female')
test = test[df_list]
# Impute missing data with the most frequently appeared value for each column
test_tf = test.apply(lambda x:x.fillna(x.value_counts().index[0]))

#######################################################
########## Write output into csv files ################
#######################################################
train_tf.to_csv('trainpp.csv', index=None)
test_tf.to_csv('testpp.csv', index=None)
