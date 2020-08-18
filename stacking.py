import pandas as pd
import re
import numpy as np
import cmudict
from sklearn import *
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingRegressor, StackingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
unigrams_df = pd.read_csv('unigram_freq.csv', index_col = 'word')

class Dataset:
	def __init__(self, path):
		self.data = pd.read_csv(path)
		self.X = self.data[[
			'length',
			'aveLength', 'maxLength', 'minLength',
			'aveFreq', 'maxFreq', 'minFreq', 
			'aveDepth', 'maxDepth', 'minDepth', 
			'aveDensity', 'minDensity', 'maxDensity',
			'aveAmbiguity', 'minAmbiguity', 'maxAmbiguity', 
			'wpm', 'elapse_time', 'speed',
			'noun', 'verb', 'adj', 'adv',
			'action', 'adventure', 'american', 'animal',
			'animation', 'australian', 'british', 'comedy',
			'cooking', 'documentary', 'drama', 'education',
			'english', 'fantasy', 'food', 'foreign accent',
			'interview', 'monologue', 'movie', 'news',
			'review', 'romance', 'sciencefiction', 'sitcom',
			'song', 'speech', 'superhero', 'tvseries',
			'talkshow', 'technology', 'thriller', 'trailer']]
		self.y = self.data['error_rate']
		self.scaler = RobustScaler()
		self.imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
		self.enc_oh = OneHotEncoder()
		self.enc_ord = OrdinalEncoder()
	def split(self, test_ratio):
		return model_selection.train_test_split(self.X, self.y, test_size = test_ratio, random_state=42)
	
	def encode(self):
		self.X['tag'] = self.enc_ord.fit_transform(self.X['tag'].to_numpy().reshape(-1, 1))
		# self.X['genreTag'] = self.enc_ord.fit_transform(self.X['genreTag'].to_numpy().reshape(-1, 1))

	def normalize(self, X_train, X_test):
		self.imputer.fit(X_train)
		X_train = self.imputer.transform(X_train)
		X_test = self.imputer.transform(X_test)
		self.scaler.fit(X_train) 
		return self.scaler.transform(X_train), self.scaler.transform(X_test)

class Regressor:
	def __init__(self, reg):
		self.reg = reg
	def fit(self, X, y, **kwargs):
		return self.reg.fit(X, y, **kwargs)
	def predict(self, X_test):
		return self.reg.predict(X_test)
	def evaluate(self, y_test, y_pred):	
		mse = metrics.mean_squared_error(y_test, y_pred)
		r2 = metrics.r2_score(y_test, y_pred)
		print(np.sqrt(mse), r2)

		clf_test = [map_to_class(y) for y in y_test]
		clf_pred = [map_to_class(y) for y in y_pred]
		print(metrics.confusion_matrix(clf_test, clf_pred))
		print(metrics.classification_report(clf_test, clf_pred))
	def run(self, X, y, X_test, y_test, **kwargs):
		self.fit(X, y, **kwargs)
		y_pred = self.predict(X_test)
		self.evaluate(y_test, y_pred)
	def select(self, X_train, X_test):
		self.selector = SelectFromModel(self.reg, prefit = True, threshold = -np.inf, max_features = 15)
		return self.selector.transform(X_train), self.selector.transform(X_test)

class Classifier:
	def __init__(self, clf):
		self.clf = clf
	def fit(self, X, y, **kwargs):
		return self.clf.fit(X, y, **kwargs)
	def predict(self, X_test):
		return self.clf.predict(X_test)		
	def evaluate(self, y_test, y_pred):	
		print(metrics.confusion_matrix(y_test, y_pred))
		print(metrics.classification_report(y_test, y_pred))
	def run(self, X, y, X_test, y_test, **kwargs):
		self.fit(X, y, **kwargs)
		y_pred = self.predict(X_test)
		self.evaluate(y_test, y_pred)
	def select(self, X_train, X_test):
		self.selector = SelectFromModel(self.clf, prefit = True)
		return self.selector.transform(X_train), self.selector.transform(X_test)

def map_to_class(score, s1 = 2/3, s2 = 1/3):
	if score > s1:
		return 2
	if score > s2:
		return 1
	return 0

data = Dataset('../processed_data.csv')
# data.encode()
# data.selection()
X_train, X_test, y_train, y_test = data.split(0.2)
X_train, X_test = data.normalize(X_train, X_test)
y_train_clf = [map_to_class(y) for y in y_train]
y_test_clf = [map_to_class(y) for y in y_test]

import lightgbm as lgb
LGB = lgb.LGBMRegressor(objective='regression',
                              learning_rate=0.1, n_estimators=5000)
GBoost = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
DT = tree.DecisionTreeRegressor()
import xgboost as xgb 
param = {
	'n_estimators': 10000,
	'learning_rate': 0.1, 
	'objective': 'reg:squarederror', 
	'verbosity' : 0,
	'n_jobs' : -1}
fit_param = {
	'eval_set':[(X_train, y_train), (X_test, y_test)],
	'early_stopping_rounds': 200,
    'verbose' : False}
BT = xgb.XGBRegressor(**param)
SVM = svm.SVR()
RF = ensemble.RandomForestRegressor(random_state=42)
NN = neural_network.MLPRegressor(hidden_layer_sizes = (100,), random_state=1, max_iter=100, alpha = 0.001)
estimators = [('dt', DT), ('bt', BT), ('lgb', LGB), ('rf', RF), ('gb', GBoost)]

reg = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), n_jobs = -1)
stack = Regressor(reg)
y_pred = stack.run(X_train, y_train, X_test, y_test)

DT = tree.DecisionTreeClassifier()
import xgboost as xgb 
param = {
	'n_estimators': 10000,
	'learning_rate': 0.1, 
	'objective': 'reg:squarederror', 
	'verbosity' : 0,
	'n_jobs': -1}
fit_param = {
	'eval_set':[(X_train, y_train), (X_test, y_test)],
	'early_stopping_rounds': 200,
    'verbose' : False}
BT = xgb.XGBClassifier(**param)

RF = ensemble.RandomForestClassifier(random_state=42)
estimators = [('dt', DT), ('bt', BT), ('lgb', LGB), ('rf', RF), ('gb', GBoost)]
final_estimator = LogisticRegression(multi_class = "ovr", random_state = 13)
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator, n_jobs = -1)
stack = Classifier(clf)
y_pred = stack.run(X_train, y_train_clf, X_test, y_test_clf)
