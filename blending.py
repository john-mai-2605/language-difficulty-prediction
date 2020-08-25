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

y_pred_vals = [reg.predict(X_val) for reg in regs]
weights = [0.5, 0, 0, 0, 0.5, 0]
y_pred_val = sum(x*y for x, y in zip(weights, y_pred_vals))
clf_val = [map_to_class(y) for y in y_pred_val]
print(metrics.confusion_matrix(y_val_clf, clf_val))
print(metrics.classification_report(y_val_clf, clf_val))

y_pred_tests = [reg.predict(X_test) for reg in regs]
weights = [0.5, 0, 0, 0, 0.5, 0]
y_pred_test = sum(x*y for x, y in zip(weights, y_pred_tests))
clf_test = [map_to_class(y) for y in y_pred_test]
print(metrics.confusion_matrix(y_test_clf, clf_test))
print(metrics.classification_report(y_test_clf, clf_test))

print("---------Blending Regressor---------")
X_val = np.append(X_val, np.asarray(y_pred_vals).T, axis = 1)
X_test = np.append(X_test, np.asarray(y_pred_tests).T, axis = 1)
model = Regressor(linear_model.LinearRegression())
model.run(X_val, y_val, X_test, y_test)