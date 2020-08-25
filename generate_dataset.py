import pandas as pd
import re
import numpy as np
import cmudict
from sklearn import *
import matplotlib.pyplot as plt
from collections import Counter
import pickle

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import GridSearchCV

from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb 
import lightgbm as lgb

# clustering
n_clusters = 35
tag_cluster_error_rate = 'tag_cluster_error_rate'
tag_clustering_features = [
    'action', 'adventure', 'american', 'animal',
    'animation', 'australian', 'british', 'comedy',
    'cooking', 'documentary', 'drama', 'education',
    'english', 'fantasy', 'food', 'foreign accent',
    'interview', 'monologue', 'movie', 'news',
    'review', 'romance', 'sciencefiction', 'sitcom',
    'song', 'speech', 'superhero', 'tvseries',
    'talkshow', 'technology', 'thriller', 'trailer',
]
tag_cluster = 'tag_cluster'
corr_cluster_error_rate = 'corr_cluster_error_rate'
corr_clustering_features = ['elapse_time', 'speed', 'wpm', 'aveAmbiguity']
corr_cluster = 'corr_cluster'

feature_columns = ['length',
            'aveLength', 'maxLength', 'minLength',
            'aveFreq', 'maxFreq', 'minFreq', 
            'aveDepth', 'maxDepth', 'minDepth', 
            'aveDensity', 'minDensity', 'maxDensity',
            'aveAmbiguity', 'minAmbiguity', 'maxAmbiguity', 
            'wpm', 'elapse_time', 'speed',
            'noun', 'verb', 'adj', 'adv',
            'det', 'prep', 'norm',
            'action', 'adventure', 'american', 'animal',
            'animation', 'australian', 'british', 'comedy',
            'cooking', 'documentary', 'drama', 'education',
            'english', 'fantasy', 'food', 'foreign accent',
            'interview', 'monologue', 'movie', 'news',
            'review', 'romance', 'sciencefiction', 'sitcom',
            'song', 'speech', 'superhero', 'tvseries',
            'talkshow', 'technology', 'thriller', 'trailer',
            tag_cluster_error_rate, corr_cluster_error_rate,
            ]

class Dataset:
    def __init__(self, path):
        self.data = pd.read_csv(path)
        # self.X = self.data[schema]
        self.y = self.data['error_rate']
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        self.enc_oh = OneHotEncoder()
        self.enc_ord = OrdinalEncoder()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    def split(self, test_ratio):
        return model_selection.train_test_split(self.data, self.y, test_size = test_ratio, random_state=42)
    
    def encode(self):
        pass
        # self.X['tag'] = self.enc_ord.fit_transform(self.X['tag'].to_numpy().reshape(-1, 1))
        # self.X['genreTag'] = self.enc_ord.fit_transform(self.X['genreTag'].to_numpy().reshape(-1, 1))

    def normalize(self, X_train, X_test):
        self.imputer.fit(X_train)
        X_train = self.imputer.transform(X_train)
        X_test = self.imputer.transform(X_test)
        self.scaler.fit(X_train) 
        return self.scaler.transform(X_train), self.scaler.transform(X_test)
    def select(self, X_train, y_train, X_test, k = 20):
        self.selector = SelectKBest(chi2, k)
        self.selector.fit(X_train, y_train)
        return self.selector.transform(X_train), self.selector.transform(X_test)
    def add_clustering_features(self, train, test, clustering_features, cluster_feature_name, normalize=False):
        clustering_X_train = train[clustering_features]
        clustering_X_test = test[clustering_features]
        if (normalize == True):
            scaler = MinMaxScaler()
            scaler.fit(clustering_X_train)
            train_for_clustering = scaler.transform(clustering_X_train)
            test_for_clustering = scaler.transform(clustering_X_test)
        # duplicate code to impute before clustering
        self.imputer.fit(clustering_X_train)
        clustering_X_train = self.imputer.transform(clustering_X_train)
        clustering_X_test = self.imputer.transform(clustering_X_test)
        #####
        self.kmeans.fit(clustering_X_train)
        train_labels = self.kmeans.predict(clustering_X_train)
        test_labels = self.kmeans.predict(clustering_X_test)
        train['cluster'] = train_labels
        test['cluster'] = test_labels
        cluster_to_median = {}
        for i in range(n_clusters):
            cluster_to_median[i] = train['error_rate'].loc[train['cluster'] == i].median()
        train[cluster_feature_name] = train['cluster'].apply(lambda c:cluster_to_median[c])
        test[cluster_feature_name] = test['cluster'].apply(lambda c:cluster_to_median[c])
        return [train, test]


def map_to_class(score, s1 = 2/3, s2 = 1/3):
    if score > s1:
        return 2
    if score > s2:
        return 1
    return 0

data = Dataset('../processed_data.csv')
# data.encode()
# data.selection()
train, test, y_train, y_test = data.split(0.2)
# add tag clustering feature
train, test = data.add_clustering_features(train, test, tag_clustering_features, tag_cluster_error_rate)
# add correlation based clustering feature
train, test = data.add_clustering_features(train, test, corr_clustering_features, corr_cluster_error_rate, normalize = True)

X_train = train[feature_columns]
X_test = test[feature_columns]
print(X_train[tag_cluster_error_rate].describe())
print(X_train[corr_cluster_error_rate].describe())
print(X_test[tag_cluster_error_rate].describe())
print(X_test[corr_cluster_error_rate].describe())
print('features', feature_columns)
print('train size', X_train.shape)
print('test size', X_test.shape)
X_train, X_test = data.normalize(X_train, X_test)
y_train_clf = [map_to_class(y) for y in y_train]
y_test_clf = [map_to_class(y) for y in y_test]

#X_train, X_test = data.select(X_train, y_train, X_test, 20)

X = np.insert(X_train, 0, y_train, axis=1)
smote = SMOTE(random_state=27)
y_train_clf = [map_to_class(y) for y in y_train]
X_new, _ = smote.fit_resample(X, y_train_clf)
X_train, y_train = X_new[:, 1:], X_new[:,[0]].flatten()

y_train_clf = [map_to_class(y) for y in y_train]
y_test_clf = [map_to_class(y) for y in y_test]
print(sorted(Counter(y_train_clf).items()))
data = {'train': {'X': X_train, 'y_clf': y_train_clf, 'y_reg': y_train},
		'test': {'X': X_test, 'y_clf': y_test_clf, 'y_reg': y_test}}
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)