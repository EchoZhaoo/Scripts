# -*- coding: utf-8 -*-
"""
Classifier algorithms

@Thao Nguyen
"""
#%% Import
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

#%% Train/test split
# Load in data
train = pd.read_csv('C:/Users/nguy3409/trainpp.csv')
test = pd.read_csv('C:/Users/nguy3409/testpp.csv')

# Separate response variables from predictors
X = train.drop(['is_female'],axis=1)
y = list(train.is_female)

# Split the training data into training and test sets for cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#%%
#######################################################
############### LOGISTIC REGRESSION ###################
#######################################################
# Fit model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# Predict on test set
y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)
# AUC
metrics.roc_auc_score(y_test, y_pred_prob[:,1])

# Cross-validation (5-fold)
kfold = model_selection.KFold(n_splits = 5)
modelCV = LogisticRegression()
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X, y, cv=kfold, scoring=scoring)
print("5-fold cross validation average accuracy: %.3f" % (results.mean()))

#%%
#######################################################
################## RANDOM FOREST ######################
#######################################################
# Fit model
Ntree = 500
rfc = RandomForestClassifier(n_estimators=Ntree)
rfc.fit(X_train, y_train)
# Predict on test set
y_pred = rfc.predict(X_test)
y_pred_prob = rfc.predict_proba(X_test)
# AUC
metrics.roc_auc_score(y_test, y_pred_prob[:,1])

# Cross-validation (5-fold)
kfold = model_selection.KFold(n_splits = 5)
modelCV = RandomForestClassifier(n_estimators=Ntree)
scoring = 'roc_auc'
results = model_selection.cross_val_score(modelCV, X, y, cv=kfold, scoring=scoring)
print("5-fold cross validation average accuracy: %.3f" % (results.mean()))

#%%
#######################################################
####################### SVM ###########################
#######################################################
# Scale data for SVM training
scaler = StandardScaler()
X_tf = scaler.fit_transform(X)

######################################################
# Train classifiers
C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_tf, y)
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

######################################################
# Fit model
svm = SVC(probability=True)
svm.fit(X_train, y_train)
# Predict on the test set
y_pred = svm.predict(X_test)
y_pred_prob = svm.predict_proba(X_test)
# AUC
metrics.roc_auc_score(y_test, y_pred_prob[:,1])
