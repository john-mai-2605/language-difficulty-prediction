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

from base_models import Regressor
print("Load data...")
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
print("Load pretrained regressor...")
with open ('regs.pkl', 'rb') as f:
    regs = pickle.load(f)
X_train, y_train_clf, y_train = data['train'].values()
X_test, y_test_clf, y_test = data['test'].values()
tree_model = regs[0]
base_model = regs[2]
X_lr_train = tree_model.reg.apply(X_train)
X_lr_test = tree_model.reg.apply(X_test)
X_train = np.append(X_train, X_lr_train, axis=1)
X_test = np.append(X_test, X_lr_test, axis=1)
X_train = X_lr_train
X_test = X_lr_test
print("Train tree-aware model")
base_model.run(X_lr_train, y_train, X_lr_test, y_test)