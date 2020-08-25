import pandas as pd
import re
import numpy as np
import cmudict
from sklearn import *
import matplotlib.pyplot as plt
from collections import Counter
import pickle

from reg_resampler import resampler
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import xgboost as xgb
from base_models import Regressor, map_to_class

print("Load data...")
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
print("Load pretrained regressor...")
with open ('regs.pkl', 'rb') as f:
    regs = pickle.load(f)
X_train, y_train_clf, y_train = data['train'].values()
X_test, y_test_clf, y_test = data['test'].values()
X_val, X_test, y_val, y_test = model_selection.train_test_split(X_test, y_test, test_size = 0.75, random_state=42)

y_train_clf = [map_to_class(y) for y in y_train]
y_test_clf = [map_to_class(y) for y in y_test]
y_val_clf = [map_to_class(y) for y in y_val]
print(sorted(Counter(y_train_clf).items()))

print("---------Averaging Blending Regressor---------")
param = {
	'n_estimators': 3000,
	'learning_rate': 0.1, 
	'objective': 'reg:squarederror', 
	'verbosity' : 0}
XGB = Regressor(xgb.XGBRegressor(**param))
XGB_pred = XGB.run(X_train, y_train, X_val, y_val)

param = {'loss': 'huber', 'learning_rate': 0.1, 'n_estimators': 3000}
GB = Regressor(GradientBoostingRegressor(**param))
GB_pred = GB.run(X_train, y_train, X_val, y_val)
regs = [XGB, GB]
y_pred_vals = [XGB_pred, GB_pred]
weights = [0.5, 0.5]
y_pred_val = sum(x*y for x, y in zip(weights, y_pred_vals))
clf_val = [map_to_class(y) for y in y_pred_val]
print(metrics.confusion_matrix(y_val_clf, clf_val))
print(metrics.classification_report(y_val_clf, clf_val))

y_pred_tests = [reg.predict(X_test) for reg in regs]
weights = [0.5, 0.5]
y_pred_test = sum(x*y for x, y in zip(weights, y_pred_tests))
clf_test = [map_to_class(y) for y in y_pred_test]
print(metrics.confusion_matrix(y_test_clf, clf_test))
print(metrics.classification_report(y_test_clf, clf_test))